// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_rope(const NodeContext& context) {
    num_inputs_check(context, 2, 3);

    int op_case = context.get_attribute<int>("op_case", 0);

    ov::Output<Node> res;

    auto data_node = context.get_input(0).get_node_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    auto rope_config = context.get_attribute<RopeConfig>("rope_config");
    const int mode = (op_case & 0xFFFF0000) >> 16;
    op_case = (op_case & 0x0000FFFF);

    constexpr int TYPE_NORMAL = 0;
    constexpr int TYPE_NEOX = 1;
    constexpr int TYPE_IMROPE = 2;

    Output<Node> cos_theta_node;
    Output<Node> sin_theta_node;
    if (context.has_input("rope_cos")) {
        cos_theta_node = context.get_input("rope_cos");
        sin_theta_node = context.get_input("rope_sin");
    } else {
        auto inp_pos = context.get_input(1).get_node_shared_ptr();
        std::shared_ptr<ov::Node> rope_freqs_weight;
        if (context.get_input_size() == 3) {
            rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
        }
        auto sin_cos = make_sin_cos(rope_config, inp_pos, rope_freqs_weight, mode == TYPE_IMROPE);
        sin_theta_node = sin_cos.first;
        cos_theta_node = sin_cos.second;
    }

    // The canonical [1, -1, n_head, head_size] reshape target (token count on the dynamic axis),
    // used by the VIEW prologue and the TYPE_NORMAL stack below.
    auto make_bhsd_shape = [&]() {
        return ov::op::v0::Constant::create(
            ov::element::i64,
            {4},
            std::vector<int64_t>{1, -1, (int64_t)output_shape[2], (int64_t)output_shape[3]});
    };

    if (op_case == 2) {
        // The input comes from a VIEW
        int slice_len = output_shape[2] * output_shape[3];
        data_node = process_view_input(context, 0, slice_len).get_node_shared_ptr();
        data_node = std::make_shared<ov::op::v1::Reshape>(data_node, make_bhsd_shape(), false);
    }

    if (mode == TYPE_NORMAL) {
        // Emit the Flux-style interleaved RoPE pattern so ov::pass::RoPEFusionFlux
        // folds this subgraph into ov::op::internal::RoPE, enabling the GPU plugin's
        // ocl::rope::opt kernel.  RoPEFusionFlux requires rank-4 x with static
        // last two dims [num_heads, head_size].  After the VIEW prologue the data
        // is already [1, L, n_heads, head_size] satisfying this contract.
        //
        // Formula:  y = x * cos_full + x_rotated * sin_full
        // where x_rotated uses the Flux interleave:
        //   Reshape(x,[1,L,H,half,2]) → Split(-1,2) → neg(x1) → Concat([neg_x1,x0]) → Reshape

        const int64_t n_heads   = static_cast<int64_t>(output_shape[2]);
        const int64_t head_size = static_cast<int64_t>(output_shape[3]);
        const int64_t half      = head_size / 2;

        // Reshape data to canonical [1, L, n_heads, head_size] if needed.
        data_node = std::make_shared<ov::op::v1::Reshape>(data_node, make_bhsd_shape(), false);

        // Reshape to [1, L, n_heads, half, 2] to expose interleaved pairs.
        auto paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{1, -1, n_heads, half, 2});
        auto x_paired = std::make_shared<ov::op::v1::Reshape>(data_node, paired_shape, false);

        // Split last axis: x0 (even / real) and x1 (odd / imaginary).
        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1LL});
        auto data_split = std::make_shared<ov::op::v1::Split>(x_paired, split_axis, 2);
        auto x0 = data_split->output(0);  // [1, L, n_heads, half, 1]
        auto x1 = data_split->output(1);

        // Negate x1: the Flux fusion expects Concat([neg_x1, x0]).
        auto neg_one_f = ov::op::v0::Constant::create(data_node->get_element_type(), ov::Shape{}, {-1.0f});
        auto x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);

        // x_rotated: flatten the interleaved pairs back to [1, L, n_heads, head_size].
        auto x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_neg, x0}, -1);
        auto x_rotated = std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, make_bhsd_shape(), false);

        // Expand cos/sin from [B, L, 1, half] to [B, L, 1, head_size] by tiling each
        // entry twice (broadcast) and flattening, matching the full rotation.
        auto expand_cos_sin = [&](ov::Output<ov::Node> cs) -> ov::Output<ov::Node> {
            auto cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1LL}));
            auto bcast_target = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            auto bcast = std::make_shared<ov::op::v3::Broadcast>(
                cs_unsq, bcast_target, ov::op::BroadcastType::BIDIRECTIONAL);
            auto flat = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, head_size});
            return std::make_shared<ov::op::v1::Reshape>(bcast, flat, /*special_zero=*/true);
        };
        auto cos_full = expand_cos_sin(cos_theta_node);
        auto sin_full = expand_cos_sin(sin_theta_node);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(data_node, cos_full);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_full);
        res    = std::make_shared<ov::op::v1::Add>(y1, y2);
    } else if (mode == TYPE_NEOX) {
        // Partial rotary (ggml n_dims < head_dim): only the first n_dims elements of each head
        // are rotated; the remainder [n_dims:head_dim] passes through unchanged. cos/sin are built
        // with width n_dims/2, so the rotated block must be exactly n_dims wide -- splitting the
        // full head would give halves wider than cos/sin and fail the elementwise broadcast.
        const int64_t head_dim = static_cast<int64_t>(output_shape[3]);
        const int64_t n_rot = rope_config.n_dims > 0 ? rope_config.n_dims : head_dim;

        Output<Node> rotary_in = data_node;
        Output<Node> pass_through;
        if (n_rot < head_dim) {
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto n_rot_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_rot});
            auto head_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            rotary_in = std::make_shared<ov::op::v8::Slice>(data_node, zero, n_rot_c, one, neg_one);
            pass_through = std::make_shared<ov::op::v8::Slice>(data_node, n_rot_c, head_c, one, neg_one);
        }

        auto data_split =
            std::make_shared<ov::op::v1::Split>(rotary_in,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}),
                                                2);
        Output<Node> slice_data_node_0 = data_split->outputs()[0];
        Output<Node> slice_data_node_1 = data_split->outputs()[1];

        auto first_half_node = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, cos_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, sin_theta_node));

        auto second_half_node = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, sin_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));

        Output<Node> rotated =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, -1);
        if (n_rot < head_dim) {
            res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{rotated, pass_through}, -1);
        } else {
            res = rotated;
        }
    } else if (mode == TYPE_IMROPE) {
        // Use output_shape, not data_node->get_shape() which throws on a dynamic dim.
        int64_t n_dims = output_shape[3];
        auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                    ov::Shape{4},
                                                                    std::vector<int64_t>{1, -1, 1, (n_dims >> 1)});
        auto cos_reshaped = std::make_shared<ov::op::v1::Reshape>(cos_theta_node, cos_sin_shape, true);
        auto sin_reshaped = std::make_shared<ov::op::v1::Reshape>(sin_theta_node, cos_sin_shape, true);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_a = std::make_shared<ov::op::v1::Split>(data_node, split_axis, 2);
        auto x0 = split_a->output(0);
        auto x1 = split_a->output(1);
        auto mul_a = std::make_shared<ov::op::v1::Multiply>(x0, cos_reshaped);
        auto mul_b = std::make_shared<ov::op::v1::Multiply>(x1, sin_reshaped);
        auto sub = std::make_shared<ov::op::v1::Subtract>(mul_a, mul_b);

        auto mul_c = std::make_shared<ov::op::v1::Multiply>(x0, sin_reshaped);
        auto mul_d = std::make_shared<ov::op::v1::Multiply>(x1, cos_reshaped);
        auto add = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub, add}, 3);
    }

    // Guard against an unmapped rope mode: without this, res stays a null Output and
    // rename_outputs_with_suffix dereferences it (crash) instead of a clean unsupported-op error.
    FRONT_END_CHECK_IMPLEMENTED(res.get_node_shared_ptr() != nullptr, "Unsupported ROPE mode");

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
