// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Element-wise unary math ops: SQR, SQRT, LOG, SIN, COS, GELU_QUICK.

#include <cmath>
#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/log.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/tanh.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_sqr(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto two = ov::op::v0::Constant::create(ov::element::f32, {}, {2.0f});
    auto res = std::make_shared<ov::op::v1::Power>(context.get_input(0), two);
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sqrt(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = std::make_shared<ov::op::v0::Sqrt>(context.get_input(0));
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_log(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = std::make_shared<ov::op::v0::Log>(context.get_input(0));
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_sin(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = std::make_shared<ov::op::v0::Sin>(context.get_input(0));
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_cos(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = std::make_shared<ov::op::v0::Cos>(context.get_input(0));
    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_UNARY_OP_GELU_QUICK: fast GELU approximation using tanh.
// Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// Equivalent to PyTorch's F.gelu(x, approximate='tanh').
OutputVector translate_unary_gelu_quick(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto x = context.get_input(0);

    auto c1 = ov::op::v0::Constant::create(ov::element::f32, {}, {0.044715f});
    auto c2 = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(std::sqrt(2.0 / M_PI))});
    auto c_half = ov::op::v0::Constant::create(ov::element::f32, {}, {0.5f});
    auto c_one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    auto c_three = ov::op::v0::Constant::create(ov::element::f32, {}, {3.0f});

    // x^3
    auto x_cubed = std::make_shared<ov::op::v1::Power>(x, c_three);
    // 0.044715 * x^3
    auto x_cubed_scaled = std::make_shared<ov::op::v1::Multiply>(c1, x_cubed);
    // x + 0.044715 * x^3
    auto inner = std::make_shared<ov::op::v1::Add>(x, x_cubed_scaled);
    // sqrt(2/pi) * (x + 0.044715 * x^3)
    auto scaled = std::make_shared<ov::op::v1::Multiply>(c2, inner);
    // tanh(...)
    auto t = std::make_shared<ov::op::v0::Tanh>(scaled);
    // 1 + tanh(...)
    auto one_plus_t = std::make_shared<ov::op::v1::Add>(c_one, t);
    // 0.5 * x * (1 + tanh(...))
    auto half_x = std::make_shared<ov::op::v1::Multiply>(c_half, x);
    auto res = std::make_shared<ov::op::v1::Multiply>(half_x, one_plus_t);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
