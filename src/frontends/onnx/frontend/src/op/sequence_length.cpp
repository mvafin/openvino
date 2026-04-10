// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

ov::OutputVector sequence_length(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 1, 1);
    const auto& inputs = node.get_ov_inputs();

    if (const auto seq_mark = as_type_ptr<SequenceMark>(inputs[0].get_node_shared_ptr())) {
        // Compile-time: return constant length
        auto length = static_cast<int64_t>(seq_mark->get_sequence().size());
        return {v0::Constant::create(ov::element::i64, ov::Shape{}, {length})};
    }

    // Runtime: create a placeholder that the normalize pass will resolve.
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs.set_type_name("SequenceLength");
    auto placeholder = std::make_shared<ov::op::util::FrameworkNode>(ov::OutputVector{inputs[0]}, 1);
    placeholder->set_attrs(attrs);
    placeholder->set_friendly_name(node.get_name());
    return placeholder->outputs();
}

ONNX_OP("SequenceLength", OPSET_SINCE(1), ai_onnx::opset_11::sequence_length);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
