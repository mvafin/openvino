// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Element-wise unary math ops: LOG, SIN, COS, GELU_QUICK.

#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/log.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/sin.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

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

// GGML_UNARY_OP_GELU_QUICK: sigmoid-based GELU approximation.
// Formula: x * sigmoid(1.702 * x), matching ggml_gelu_quick_f32 in ggml/src/ggml-cpu/vec.h:
//   x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x))) with GELU_QUICK_COEF = -1.702f.
// This is a different approximation from GGML_UNARY_OP_GELU (tanh/erf); the two are not
// interchangeable -- they diverge by ~2e-2 on [-6, 6], most sharply in the negative tail.
OutputVector translate_unary_gelu_quick(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto x = context.get_input(0);

    auto coef = ov::op::v0::Constant::create(ov::element::f32, {}, {1.702f});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(x, coef);
    auto s = std::make_shared<ov::op::v0::Sigmoid>(scaled);
    auto res = std::make_shared<ov::op::v1::Multiply>(x, s);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
