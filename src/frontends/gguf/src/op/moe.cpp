// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Mixture-of-experts (MoE) routing op translators. These lower the gguf MoE primitives
// to standard OpenVINO opset ops:
//   GGML_OP_MUL_MAT_ID  - per-token expert-selected batched matmul
//   GGML_OP_ADD_ID      - per-token expert-selected bias add
//   GGML_OP_ARGSORT     - descending argsort (expert ranking)
//   GGML_OP_TOP_K       - top-k indices (expert selection)
//   GGML_OP_SUM_ROWS    - reduce-sum over the last (ggml ne0) axis
//
// Shapes here are in the OpenVINO logical order (reverse of gguf ne[]), which the decoder
// reports: a gguf tensor [ne0, ne1, ne2, ne3] is exposed as [ne3, ne2, ne1, ne0].

#include <cstdint>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <ov_ops/gather_matmul.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

using namespace ov::op;

// ggml: c = mul_mat_id(as, b, ids)
//   as  -> [cols, rows, n_expert]            (OV: [n_expert, rows, cols])
//   b   -> [cols, n_expert_used, n_tokens]   (OV: [n_tokens, n_expert_used, cols])
//   ids -> [n_expert_used, n_tokens] i32     (OV: [n_tokens, n_expert_used])
//   c   -> [rows, n_expert_used, n_tokens]   (OV: [n_tokens, n_expert_used, rows])
//
// Lowered to the internal ov::op::internal::GatherMatmul, which the CPU/GPU plugins execute
// as a single optimized batched expert-matmul (and, when the expert weights are a compressed
// Constant->Convert->[Subtract]->Multiply block, fold into GatherMatmulCompressed so the
// weights stay compressed -- no host f32 expansion). GatherMatmul's contract:
//   A       [n_activated, T, cols]   (n_activated == 1 broadcasts the same input to every
//                                      selected expert; == K gives a per-slot input)
//   B       [n_expert, rows, cols]   (transpose_b=true -> A . Bᵀ)
//   indices [T, K] i32               (the selected expert per (token, slot))
//   out     [K, T, rows]
// which we reshape back to the builder's [1, T, K, rows] convention.
OutputVector translate_mul_mat_id(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto as = context.get_input(0);   // expert weights, OV [n_expert, rows, cols]
    auto b = context.get_input(1);    // routed input
    auto ids = context.get_input(2);  // selected experts

    // Canonicalize ids to 2D [T, K] (the builder may carry leading 1-dims).
    const auto ids_rank = static_cast<int>(ids.get_partial_shape().size());
    auto ids_2d = std::make_shared<v1::Reshape>(
        ids,
        std::make_shared<v0::Concat>(OutputVector{v0::Constant::create(element::i64, Shape{1}, {-1}),
                                                  get_dimensions(ids.get_node_shared_ptr(), {ids_rank - 1})},
                                     0),
        false);  // [T, K]
    const int64_t K = ids.get_partial_shape()[ids_rank - 1].get_length();

    // Build the GatherMatmul activation A = [n_activated, T, cols].
    //   gate/up: b is [.., T, cols] (one shared input fanned out to all experts) -> A = [1, T, cols]
    //   down   : b is [.., T, K, cols] (already per-slot)                         -> A = [K, T, cols]
    const auto& bps = b.get_partial_shape();
    const int64_t cols = bps[bps.size() - 1].get_length();
    const bool has_k =
        K > 1 && bps.size() >= 2 && bps[bps.size() - 2].is_static() && bps[bps.size() - 2].get_length() == K;
    Output<Node> a;
    if (has_k) {
        // [.., T, K, cols] -> [T, K, cols] -> transpose to [K, T, cols]
        auto b_tkc = std::make_shared<v1::Reshape>(
            b,
            v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{-1, K, cols}),
            false);
        a = std::make_shared<v1::Transpose>(
            b_tkc,
            v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2}));
    } else {
        // [.., T, cols] -> [1, T, cols]; n_activated == 1 lets GatherMatmul broadcast it to
        // each selected expert.
        a = std::make_shared<v1::Reshape>(
            b,
            v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, -1, cols}),
            false);
    }

    // B must be a 3D [n_expert, rows, cols] tensor. `as` already has this layout; keep it in
    // its (possibly compressed) precision so ConvertGatherMatmulToGatherMatmulCompressed can
    // pick up the decompression subgraph and keep the weights compressed.
    auto gmm = std::make_shared<ov::op::internal::GatherMatmul>(a, as, ids_2d);  // [K, T, rows]

    // [K, T, rows] -> [1, T, K, rows] (builder convention).
    const int64_t rows = as.get_partial_shape()[as.get_partial_shape().size() - 2].get_length();
    auto kt2tk = std::make_shared<v1::Transpose>(
        gmm,
        v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 0, 2}));  // [T, K, rows]
    auto res = std::make_shared<v1::Reshape>(
        kt2tk,
        v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, -1, K, rows}),
        false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

// ggml: add_id(a, b, ids): a [n, n_expert_used, n_tokens] + b[:, ids] where b is
// [n, n_expert] per-expert bias. OV: a [n_tokens, n_expert_used, n], b [n_expert, n],
// ids [n_tokens, n_expert_used] -> gather b by ids -> [n_tokens, n_expert_used, n], add.
OutputVector translate_add_id(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto a = context.get_input(0);    // [1, T, K, n]
    auto b = context.get_input(1);    // per-expert bias [n_expert, n]
    auto ids = context.get_input(2);  // selected experts [.., T, K]

    // Canonicalize ids to 2D [T, K] so the gather adds exactly a [T, K, n] tensor (matching
    // a's [1, T, K, n] after a leading unsqueeze), not an extra ids-rank dim.
    const int64_t K = ids.get_partial_shape()[ids.get_partial_shape().size() - 1].get_length();
    auto ids_2d =
        std::make_shared<v1::Reshape>(ids,
                                      v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, K}),
                                      false);  // [T, K]
    ov::Output<ov::Node> b_gathered =
        std::make_shared<v8::Gather>(b, ids_2d, v0::Constant::create(element::i32, Shape{}, {0}));  // [T, K, n]
    if (b_gathered.get_element_type() != a.get_element_type()) {
        b_gathered = std::make_shared<v0::Convert>(b_gathered, a.get_element_type());
    }
    // [T,K,n] -> [1,T,K,n] to match a.
    auto b4 = std::make_shared<v0::Unsqueeze>(b_gathered, v0::Constant::create(element::i64, Shape{1}, {0}));
    auto res = std::make_shared<v1::Add>(a, b4);
    return rename_outputs_with_suffix({res}, context.get_name());
}

// ggml_argsort(a, DESC): indices that sort the last axis descending. Implemented with
// TopK over the full last-axis length (k == ne0).
OutputVector translate_argsort(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto a = context.get_input(0);
    int64_t k = context.get_output_shape().to_shape().back();
    auto k_node = v0::Constant::create(element::i64, Shape{}, {k});
    auto topk = std::make_shared<v11::TopK>(a,
                                            k_node,
                                            -1,
                                            v11::TopK::Mode::MAX,
                                            v11::TopK::SortType::SORT_VALUES,
                                            element::i32);
    auto idx = std::make_shared<v0::Convert>(topk->output(1), element::i32);
    return rename_outputs_with_suffix({idx}, context.get_name());
}

// ggml_top_k(a, k): the top-k indices along the last axis (descending). Wrap the index
// output in a Convert so downstream consumers see a single-output source node.
OutputVector translate_top_k(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto a = context.get_input(0);
    int64_t k = context.get_output_shape().to_shape().back();
    auto k_node = v0::Constant::create(element::i64, Shape{}, {k});
    auto topk = std::make_shared<v11::TopK>(a,
                                            k_node,
                                            -1,
                                            v11::TopK::Mode::MAX,
                                            v11::TopK::SortType::SORT_VALUES,
                                            element::i32);
    auto idx = std::make_shared<v0::Convert>(topk->output(1), element::i32);
    return rename_outputs_with_suffix({idx}, context.get_name());
}

// ggml_sum_rows: sum over the last (ggml ne0 / OV last) axis, keeping that dim as 1.
OutputVector translate_sum_rows(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto a = context.get_input(0);
    auto axis = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto res = std::make_shared<v1::ReduceSum>(a, axis, true);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
