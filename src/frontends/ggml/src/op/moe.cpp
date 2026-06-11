// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Mixture-of-experts (MoE) routing op translators. These lower the ggml MoE primitives
// to standard OpenVINO opset ops:
//   GGML_OP_MUL_MAT_ID  - per-token expert-selected batched matmul
//   GGML_OP_ADD_ID      - per-token expert-selected bias add
//   GGML_OP_ARGSORT     - descending argsort (expert ranking)
//   GGML_OP_TOP_K       - top-k indices (expert selection)
//   GGML_OP_SUM_ROWS    - reduce-sum over the last (ggml ne0) axis
//
// Shapes here are in the OpenVINO logical order (reverse of ggml ne[]), which the decoder
// reports: a ggml tensor [ne0, ne1, ne2, ne3] is exposed as [ne3, ne2, ne1, ne0].

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
#include <openvino/op/unsqueeze.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

using namespace ov::op;

// ggml: c = mul_mat_id(as, b, ids)
//   as  -> [cols, rows, n_expert]            (OV: [n_expert, rows, cols])
//   b   -> [cols, n_expert_used, n_tokens]   (OV: [n_tokens, n_expert_used, cols])
//   ids -> [n_expert_used, n_tokens] i32     (OV: [n_tokens, n_expert_used])
//   c   -> [rows, n_expert_used, n_tokens]   (OV: [n_tokens, n_expert_used, rows])
// Lowering: gather the per-(token,slot) expert matrix from `as` by id, then a batched
// matmul against the corresponding column of `b`.
OutputVector translate_mul_mat_id(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto as = context.get_input(0);   // expert weights, OV [n_expert, rows, cols]
    auto b = context.get_input(1);    // routed input
    auto ids = context.get_input(2);  // selected experts

    if (as.get_element_type() != b.get_element_type()) {
        as = std::make_shared<v0::Convert>(as, b.get_element_type());
    }

    // Canonicalize ids to 2D [T, K] (the builder may carry leading 1-dims).
    auto ids_2d = std::make_shared<v1::Reshape>(
        ids,
        std::make_shared<v0::Concat>(
            OutputVector{v0::Constant::create(element::i64, Shape{1}, {-1}),
                         get_dimensions(ids.get_node_shared_ptr(), {static_cast<int>(ids.get_partial_shape().size()) - 1})},
            0),
        false);  // [T, K]
    const int64_t K = ids.get_partial_shape()[ids.get_partial_shape().size() - 1].get_length();

    // Gather per-(token,slot) expert matrix: [T, K, rows, cols].
    auto experts = std::make_shared<v8::Gather>(as, ids_2d, v0::Constant::create(element::i32, Shape{}, {0}));

    // Canonicalize b to [T, K, cols]: it is either [.., T, cols] (broadcast across K, e.g.
    // gate/up from the shared norm) or [.., T, K, cols] (e.g. the down input which already
    // has a per-expert dim). Detect by the last-but-one dim against K.
    const auto& bps = b.get_partial_shape();
    const int64_t cols = bps[bps.size() - 1].get_length();
    Output<Node> b_ck;
    bool has_k = K > 1 && bps.size() >= 2 && bps[bps.size() - 2].is_static() &&
                 bps[bps.size() - 2].get_length() == K;
    if (has_k) {
        // [.., T, K, cols] -> [T, K, cols]
        b_ck = std::make_shared<v1::Reshape>(
            b, v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{-1, K, cols}), false);
    } else {
        // [.., T, cols] -> [T, 1, cols] -> broadcast to [T, K, cols]
        auto b_t1 = std::make_shared<v1::Reshape>(
            b, v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{-1, 1, cols}), false);
        auto tgt = std::make_shared<v0::Concat>(
            OutputVector{get_dimensions(b_t1, {0}), v0::Constant::create(element::i64, Shape{1}, {K}),
                         v0::Constant::create(element::i64, Shape{1}, {cols})},
            0);
        b_ck = std::make_shared<v3::Broadcast>(b_t1, tgt, ov::op::BroadcastType::BIDIRECTIONAL);
    }

    // [T,K,rows,cols] @ [T,K,cols,1] -> [T,K,rows,1] -> squeeze -> [T,K,rows] -> [1,T,K,rows].
    auto b_col = std::make_shared<v0::Unsqueeze>(b_ck, v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto mm = std::make_shared<v0::MatMul>(experts, b_col, false, false);
    auto sq = std::make_shared<v0::Squeeze>(mm, v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto res = std::make_shared<v0::Unsqueeze>(sq, v0::Constant::create(element::i64, Shape{1}, {0}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

// ggml: add_id(a, b, ids): a [n, n_expert_used, n_tokens] + b[:, ids] where b is
// [n, n_expert] per-expert bias. OV: a [n_tokens, n_expert_used, n], b [n_expert, n],
// ids [n_tokens, n_expert_used] -> gather b by ids -> [n_tokens, n_expert_used, n], add.
OutputVector translate_add_id(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto a = context.get_input(0);
    auto b = context.get_input(1);
    auto ids = context.get_input(2);
    auto gather_axis = v0::Constant::create(element::i32, Shape{}, {0});
    ov::Output<ov::Node> b_gathered =
        std::make_shared<v8::Gather>(b, ids, gather_axis);  // [n_tokens, n_expert_used, n]
    if (b_gathered.get_element_type() != a.get_element_type()) {
        b_gathered = std::make_shared<v0::Convert>(b_gathered, a.get_element_type());
    }
    auto res = std::make_shared<v1::Add>(a, b_gathered);
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
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
