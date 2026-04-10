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

ov::OutputVector sequence_erase(const ov::frontend::onnx::Node& node) {
    const auto& inputs = node.get_ov_inputs();
    OPENVINO_ASSERT(inputs.size() >= 1 && inputs.size() <= 2,
                    "SequenceErase expects 1 or 2 inputs, got ",
                    inputs.size());

    // When the input is a compile-time SequenceMark, remove the element directly.
    if (const auto seq_mark = as_type_ptr<SequenceMark>(inputs[0].get_node_shared_ptr())) {
        auto elements = seq_mark->get_sequence();
        int64_t pos = -1;  // default: erase last
        if (inputs.size() > 1) {
            const auto pos_const = ov::as_type_ptr<v0::Constant>(inputs[1].get_node_shared_ptr());
            OPENVINO_ASSERT(pos_const, "SequenceErase: 'position' must be constant");
            pos = pos_const->cast_vector<int64_t>()[0];
        }
        auto len = static_cast<int64_t>(elements.size());
        if (pos < 0)
            pos += len;
        OPENVINO_ASSERT(pos >= 0 && pos < len, "SequenceErase: position out of bounds");
        elements.erase(elements.begin() + pos);
        return std::make_shared<SequenceMark>(elements)->outputs();
    }

    // Runtime: create a placeholder.
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs.set_type_name("SequenceErase");
    auto placeholder = std::make_shared<ov::op::util::FrameworkNode>(inputs, 1);
    placeholder->set_attrs(attrs);
    placeholder->set_friendly_name(node.get_name());
    return placeholder->outputs();
}

ONNX_OP("SequenceErase", OPSET_SINCE(1), ai_onnx::opset_11::sequence_erase);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
