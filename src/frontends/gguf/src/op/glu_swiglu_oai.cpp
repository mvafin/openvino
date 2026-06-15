// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGML_GLU_OP_SWIGLU_OAI: the gpt-oss MoE gated activation.
//   x   = min(gate, limit)
//   y   = clamp(up, -limit, limit)
//   glu = x * sigmoid(alpha * x)        (== x / (1 + exp(-alpha*x)))
//   out = glu * (y + 1)
// with alpha=1.702, limit=7.0 by default (passed as typed attributes "alpha"/"limit").
// Inputs: either two tensors (gate, up) or one combined tensor split in half on the last
// axis. A "swapped" attribute swaps gate/up.

#include <cstdint>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/slice.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_glu_swiglu_oai(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> gate;
    ov::Output<ov::Node> up;
    if (context.get_input_size() == 2) {
        gate = context.get_input(0);
        up = context.get_input(1);
    } else {
        auto combined = context.get_input(0);
        int64_t last = combined.get_partial_shape()[combined.get_partial_shape().rank().get_length() - 1].get_length();
        int64_t nc = last / 2;
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto z = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto m = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto e = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});
        gate = std::make_shared<ov::op::v8::Slice>(combined, z, m, step, axis);
        up = std::make_shared<ov::op::v8::Slice>(combined, m, e, step, axis);
    }

    if (context.get_attribute<bool>("swapped")) {
        std::swap(gate, up);
    }

    const float alpha = context.get_attribute<float>("alpha");
    const float limit = context.get_attribute<float>("limit");

    // x = min(gate, limit)
    auto limit_c = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {limit});
    auto x = std::make_shared<ov::op::v1::Minimum>(gate, limit_c);
    // y = clamp(up, -limit, limit)
    auto y = std::make_shared<ov::op::v0::Clamp>(up, -limit, limit);
    // glu = x * sigmoid(alpha * x)
    auto alpha_c = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {alpha});
    auto ax = std::make_shared<ov::op::v1::Multiply>(x, alpha_c);
    auto sig = std::make_shared<ov::op::v0::Sigmoid>(ax);
    auto glu = std::make_shared<ov::op::v1::Multiply>(x, sig);
    // out = glu * (y + 1)
    auto one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto y1 = std::make_shared<ov::op::v1::Add>(y, one);
    auto res = std::make_shared<ov::op::v1::Multiply>(glu, y1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
