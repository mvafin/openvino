#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_soft_max(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    auto input_node = context.get_input(0).get_node_shared_ptr();
    ov::Output<Node> res;

    // scale defaults to 1.0 (MoE gating path); attention sets it explicitly.
    float scale = context.get_attribute<float>("scale", 1.0f);
    // Softmax axis: attention reduces axis 2 (the key axis); MoE gating reduces the last axis.
    int64_t softmax_axis = context.get_attribute<int64_t>("softmax_axis", 2);

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
    auto scaled_input = std::make_shared<ov::op::v1::Multiply>(input_node, scale_node);

    if (context.get_input_size() < 2) {
        res = std::make_shared<ov::op::v8::Softmax>(scaled_input, softmax_axis);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    ov::Output<ov::Node> mask_node_sliced;
    if (context.has_input("KQ_mask_sliced")) {
        mask_node_sliced = context.get_input("KQ_mask_sliced");
    } else {
        auto token_len = get_dimensions(input_node, {1});
        auto mask_node = context.get_input(1);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        mask_node_sliced = std::make_shared<ov::op::v8::Slice>(mask_node, zero, token_len, one, one);
    }

    if (mask_node_sliced.get_element_type() != context.get_output_type()) {
        mask_node_sliced = std::make_shared<ov::op::v0::Convert>(mask_node_sliced, context.get_output_type());
    }

    auto input_mask_node = std::make_shared<ov::op::v1::Add>(scaled_input, mask_node_sliced);
    res = std::make_shared<ov::op::v8::Softmax>(input_mask_node, 2);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
