// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

#include <cstdint>
#include <memory>
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include <stdexcept>
#include <vector>

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_reshape(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    if (context.get_input_shape(0) == context.get_output_shape()) {
        return {context.get_input(0)};
    }

    // Cases 1-8 are emitted by the llama.cpp cgraph decoder (see ggml-decoder.cpp::compute_op_case);
    // cases >= 100 are emitted only by the native .gguf builder decoder, which owns its own numbering
    // so the two decoders never collide.
    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(
        op_case == 1 || op_case == 2 || op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6 ||
            op_case == 7 || op_case == 8 || op_case == 107 || op_case == 108,
        "Unsupported RESHAPE case");

    if (op_case == 8) {
        // Identity reshape (ggml src ne == node ne): a no-op. Pass the input through so any dynamic
        // token axis it carries is preserved (a static reshape would bake in the compile-time count).
        return {context.get_input(0)};
    }

    auto output_shape = context.get_output_shape().to_shape();
    std::shared_ptr<ov::Node> new_shape_node;
    if (op_case == 1) {
        // [B, 1, T, n_head*head_size] -> [B, T, n_head, head_size]: split the last dim into heads and
        // flatten whatever leads it into dim 1. Same shape in both stateful and non-stateful paths;
        // the 3D form was causing RoPE broadcasting to T×T when the trailing dimensions are 1 (MQA,
        // n_head_kv=1).
        //
        // The leading dim is COPIED from the input via special_zero rather than written as
        // output_shape[0] (a literal 1). That is what makes the attention block layout-polymorphic:
        // ov::pass::SDPAToPagedAttention moves the token count into dim 0 by rewriting input_ids, and
        // a literal here would discard that and leave PA deriving [1, T*H*S] operands where the
        // plugin wants [T, H*S]. With the 0 the same constant serves both:
        //   SDPA inference: in [1, 1, T, H*S]  -> [1, T, H, S]
        //   PagedAttention: in [T, 1, 1, H*S]  -> [T, 1, H, S]  (identical buffer, tokens in dim 0)
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64,
            {4},
            std::vector<int64_t>{0, -1, (int64_t)output_shape[2], (int64_t)output_shape[3]});
        return rename_outputs_with_suffix(
            {std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, /*special_zero=*/true)},
            context.get_name());
    } else if (op_case == 2) {
        // Merge the heads back after attention. Like op_case 1, the leading dim is copied from the input
        // (special_zero) rather than pinned to output_shape[0], so the token axis stays wherever the
        // active attention backend put it.
        //
        // The output rank follows the mode's activation convention, because the very next op is the
        // residual Add against the layer input and OV broadcasts elementwise operands from the RIGHT.
        // Emitting rank 4 here while the residual is the rank-3 stateful activation only appears to work
        // when dim 0 is a literal batch 1 ([1,1,T,E] right-aligns onto [1,T,E]); once the token count
        // moves into dim 0 the same alignment silently forms a token x token outer product. So the
        // stateful path stays rank 3, matching get_rows / rms_norm / mulmat (cf. op_case 5).
        //   stateful,     SDPA inference: in [1, T, H, S] -> [1, T, H*S]
        //   stateful,     PagedAttention: in [T, 1, H, S] -> [T, 1, H*S]
        //   non-stateful (ggml is 4D throughout): in [1, T, H, S] -> [1, 1, T, H*S]
        // Either way the last dim is the static n_head*head_size and the -1 absorbs the remaining axis,
        // so the following MatMul against [n_embd, n_embd] is unaffected.
        if (context.is_stateful()) {
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64,
                                                          {3},
                                                          std::vector<int64_t>{0, -1, (int64_t)output_shape[3]});
        } else {
            new_shape_node = ov::op::v0::Constant::create(
                ov::element::i64,
                {4},
                std::vector<int64_t>{0, (int64_t)output_shape[1], -1, (int64_t)output_shape[3]});
        }
        return rename_outputs_with_suffix(
            {std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, /*special_zero=*/true)},
            context.get_name());

    } else if (op_case == 3) {
        // Flatten-for-SET_ROWS: [F, tok, 1, 1] -> [1, F*tok, -1, 1] (the KV-cache write path, e.g.
        // gpt-oss cache_v). Token count stays on the dynamic axis via -1.
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64,
            {4},
            std::vector<int64_t>{(int64_t)output_shape[0], (int64_t)output_shape[1], -1, 1});

    } else if (op_case == 4) {
        return {context.get_input(0).get_node_shared_ptr()->input_value(0)};

    } else if (op_case == 5) {
        if (context.is_stateful()) {
            std::vector<int64_t> shape_vec = {1, -1, (int64_t)context.get_output_shape().to_shape()[3]};
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {3}, shape_vec);
        } else {
            std::vector<int64_t> shape_vec = {1, 1, -1, (int64_t)context.get_output_shape().to_shape()[3]};
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, shape_vec);
        }

        // // Alternative
        // auto token_len = context.get_input("token_len");
        // auto emb_size =
        //     ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t)
        //     context.get_output_shape().to_shape()[3]});
        // auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        // new_shape_node = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, one, token_len, emb_size}, 0);

    } else if (op_case == 6) {
        // The output layout rearranges dims relative to the input (e.g. qwen3-next q/k_conv_predelta:
        // [128,2,8,T] -> [128,16,T,1]). The decoder supplies the OV-order target with -1 on the dynamic
        // token axis so the stateful model reuses across token counts; fall back to the static output
        // shape when no dynamic axis was inferred.
        auto tgt = context.get_attribute<std::vector<int64_t>>("reshape_target", {});
        if (tgt.empty()) {
            tgt.assign(output_shape.begin(), output_shape.end());
        }
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt);

    } else if (op_case == 7) {
        // General fully-static reshape (no dynamic token axis): reshape straight to the static
        // output shape. Used by qwen3-next's recurrent-state predelta reshape [262144]->[16,128,128].
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64, {output_shape.size()},
            std::vector<int64_t>(output_shape.begin(), output_shape.end()));

    } else if (op_case == 107) {
        // Builder: dynamic-safe collapse to [1, 1, -1, last_dim] (MoE aggregation output): the token
        // axis stays dynamic via -1; only the last dim (n_embd) is static.
        int64_t last = (int64_t)output_shape[3];
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, last});
    } else if (op_case == 108) {
        // Builder: per-layer embedding reshape+transpose.
        //   stateful:     [T, n_layer*pe_dim] -> reshape [T, n_layer, pe_dim] -> transpose [n_layer, T, pe_dim]
        //   non-stateful: [1, 1, T, n_layer*pe_dim] -> reshape [1, T, n_layer, pe_dim] -> transpose [1, n_layer, T, pe_dim]
        // output_shape is {1, n_layer, T, pe_dim}; dim[1] (n_layer) and dim[3] (pe_dim) are static.
        // A naive reshape directly to [n_layer, T, pe_dim] is WRONG for T>1: the data is
        // contiguous as [T, n_layer, pe_dim] (one row per token), so we must reshape then
        // transpose the first two non-batch axes.
        int64_t n_layer = (int64_t)output_shape[1];
        int64_t pe_dim = (int64_t)output_shape[3];
        if (context.is_stateful()) {
            // Step 1: reshape to [T, n_layer, pe_dim] (-1 on T axis for dynamic)
            new_shape_node =
                ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, n_layer, pe_dim});
            auto reshaped = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
            // Step 2: transpose [T, n_layer, pe_dim] -> [n_layer, T, pe_dim]
            auto perm = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 0, 2});
            auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
            return rename_outputs_with_suffix({transposed}, context.get_name());
        } else {
            // Step 1: reshape to [1, T, n_layer, pe_dim]
            new_shape_node =
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, -1, n_layer, pe_dim});
            auto reshaped = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
            // Step 2: transpose [1, T, n_layer, pe_dim] -> [1, n_layer, T, pe_dim]
            auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
            auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
            return rename_outputs_with_suffix({transposed}, context.get_name());
        }
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
