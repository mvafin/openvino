#pragma once

#include <openvino/opsets/opset8.hpp>

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {

namespace op {
namespace util {
class FrameworkNode;
}
}  // namespace op

namespace frontend {
namespace pytorch {

Output<Node> make_optional_bias(const Output<Node>& base_op,
                                const NodeContext& context,
                                size_t bias_input_idx,
                                const std::vector<int>& unsqueeze_dims = {});

std::shared_ptr<ov::Node> get_rank_node(const Output<Node>& node);

Output<Node> reshape_kernel_for_group(const NodeContext& context,
                                      const Output<Node>& input,
                                      const Output<Node>& kernel,
                                      int64_t groups);

std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model,
                                                 const TensorMap& external_tensor_map = {});

OutputVector convert_node(NodeContext* context);

template <OutputVector (*T)(NodeContext&), size_t idx = 0>
OutputVector inplace_op(NodeContext& context) {
    auto translation_res = T(context);
    FRONT_END_OP_CONVERSION_CHECK(translation_res.size() == 1,
                                  "inplace_op function must be used on single output translators");
    context.mutate_input(idx, translation_res[0]);
    return translation_res;
}

template <typename T>
OutputVector translate_1to1_match_1_inputs(NodeContext& context) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 1, "Operation has no inputs.");
    for (int i = 1; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0), "Input should not be None.");
    return {context.mark_node(std::make_shared<T>(inputs[0]))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs(NodeContext& context) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 2, "Operation has no inputs.");
    for (int i = 2; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0) && !context.input_is_none(1), "Inputs should not be None.");
    return {context.mark_node(std::make_shared<T>(inputs[0], inputs[1]))};
}

std::shared_ptr<ov::op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node, const std::string& type);

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
