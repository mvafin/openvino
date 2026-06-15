// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/concat.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_CONCAT: concatenate two tensors along a given axis.
// op_case encodes the OV axis (GGML ne[] dimension mapped to OV reversed layout:
// GGML dim 0 (ne0, innermost) = OV axis rank-1; GGML dim 2 (ne2) = OV axis 1 for 4D).
// Default (op_case == 0) concatenates along axis 1 which is the most common use-case
// (GGML dim 2, the sequence/batch axis for KV cache concat in non-stateful models).
OutputVector translate_concat(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();
    int64_t axis = (op_case == 0) ? 1 : static_cast<int64_t>(op_case);

    auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{context.get_input(0), context.get_input(1)}, axis);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
