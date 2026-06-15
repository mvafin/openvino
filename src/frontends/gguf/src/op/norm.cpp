// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/mvn.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_NORM: standard LayerNorm (mean subtraction + variance normalization).
// Maps to MVN v6 with normalize_variance=true, reducing over the last axis.
OutputVector translate_norm(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    float eps = context.get_attribute<float>("eps", 1e-5f);

    auto input = context.get_input(0);
    int64_t rank = static_cast<int64_t>(input.get_partial_shape().rank().get_length());
    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {rank - 1});

    auto res = std::make_shared<ov::op::v6::MVN>(input,
                                                 reduction_axes,
                                                 true,  // normalize_variance
                                                 eps,
                                                 ov::op::MVNEpsMode::INSIDE_SQRT);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
