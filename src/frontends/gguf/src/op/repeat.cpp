// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/tile.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_REPEAT: tile input so its shape matches the output shape.
// The repeat counts per axis = output_dim / input_dim.
// Both shapes are static (GGUF cgraph always produces static shapes).
OutputVector translate_repeat(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto in_shape = context.get_input_shape(0).to_shape();
    auto out_shape = context.get_output_shape().to_shape();

    FRONT_END_OP_CONVERSION_CHECK(in_shape.size() == out_shape.size(),
                                  "GGML_OP_REPEAT: input and output ranks must match");

    std::vector<int64_t> repeats(in_shape.size());
    for (size_t i = 0; i < in_shape.size(); ++i) {
        FRONT_END_OP_CONVERSION_CHECK(in_shape[i] > 0 && out_shape[i] % in_shape[i] == 0,
                                      "GGML_OP_REPEAT: output dim must be a multiple of input dim");
        repeats[i] = static_cast<int64_t>(out_shape[i] / in_shape[i]);
    }

    auto repeats_node = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    auto res = std::make_shared<ov::op::v0::Tile>(input, repeats_node);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
