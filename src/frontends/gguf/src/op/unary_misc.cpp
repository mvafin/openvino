// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/elu.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/tanh.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_unary_relu(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = std::make_shared<ov::op::v0::Relu>(context.get_input(0));
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_tanh(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = std::make_shared<ov::op::v0::Tanh>(context.get_input(0));
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_sigmoid(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = std::make_shared<ov::op::v0::Sigmoid>(context.get_input(0));
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_unary_elu(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    double alpha = static_cast<double>(context.get_attribute<float>("alpha", 1.0f));
    auto res = std::make_shared<ov::op::v0::Elu>(context.get_input(0), alpha);
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_clamp(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    double min_val = static_cast<double>(context.get_attribute<float>("min", std::numeric_limits<float>::lowest()));
    double max_val = static_cast<double>(context.get_attribute<float>("max", std::numeric_limits<float>::max()));
    auto res = std::make_shared<ov::op::v0::Clamp>(context.get_input(0), min_val, max_val);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
