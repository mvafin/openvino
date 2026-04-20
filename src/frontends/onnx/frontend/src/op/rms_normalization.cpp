// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ::ONNX_NAMESPACE::TensorProto_DataType;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector rms_normalization(const ov::frontend::onnx::Node& node) {
    // ONNX RMSNormalization, opset 23:
    // https://onnx.ai/onnx/operators/onnx__RMSNormalization.html
    // Y = (X / sqrt(mean(X^2, axes=[axis..rank-1], keepdims=True) + epsilon)) * scale
    common::default_op_checks(node, 2);

    const auto inputs = node.get_ov_inputs();
    ov::Output<ov::Node> X = inputs.at(0);
    const auto scale = inputs.at(1);

    const auto default_stash_type_i = static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT);
    const int64_t stash_type_i = node.get_attribute_value<int64_t>("stash_type", default_stash_type_i);
    const element::Type stash_type = common::get_ov_element_type(stash_type_i);

    const float epsilon = node.get_attribute_value<float>("epsilon", 1e-5f);
    const int64_t axis = node.get_attribute_value<int64_t>("axis", -1);

    const element::Type original_type = X.get_element_type();
    const bool needs_type_casting = stash_type != original_type;
    if (needs_type_casting) {
        X = std::make_shared<v0::Convert>(X, stash_type);
    }

    // Build axes = [axis, axis+1, ..., rank-1] (handles negative axis via dynamic rank).
    auto rank_node = std::make_shared<v0::Squeeze>(
        std::make_shared<v3::ShapeOf>(std::make_shared<v3::ShapeOf>(X), element::i64));
    auto axes = std::make_shared<v4::Range>(
        v0::Constant::create(element::i64, {}, {axis}),
        (axis < 0 ? v0::Constant::create(element::i64, {}, {0})->output(0) : rank_node->output(0)),
        v0::Constant::create(element::i64, {}, {1}),
        element::i64);

    auto x_squared = std::make_shared<v1::Multiply>(X, X);
    auto x_squared_mean = std::make_shared<v1::ReduceMean>(x_squared, axes, true);
    auto rms = std::make_shared<v0::Sqrt>(
        std::make_shared<v1::Add>(x_squared_mean, v0::Constant::create(stash_type, {}, {epsilon})));
    ov::Output<ov::Node> normalized = std::make_shared<v1::Divide>(X, rms);

    if (needs_type_casting) {
        normalized = std::make_shared<v1::ConvertLike>(normalized, inputs.at(0));
    }

    return {std::make_shared<v1::Multiply>(normalized, scale)->output(0)};
}

ONNX_OP("RMSNormalization", OPSET_SINCE(1), ai_onnx::opset_1::rms_normalization);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
