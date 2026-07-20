// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/topk.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

using namespace ov::op;

// ggml_top_k(a, k): the top-k indices along the last axis (descending). Used by the native
// .gguf builder for MoE expert selection. Wrap the index output in a Convert so downstream
// consumers see a single-output source node.
OutputVector translate_top_k(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto a = context.get_input(0);
    int64_t k = context.get_output_shape().to_shape().back();
    auto k_node = v0::Constant::create(element::i64, Shape{}, {k});
    auto topk = std::make_shared<v11::TopK>(a,
                                            k_node,
                                            -1,
                                            v11::TopK::Mode::MAX,
                                            v11::TopK::SortType::SORT_VALUES,
                                            element::i32);
    auto idx = std::make_shared<v0::Convert>(topk->output(1), element::i32);
    return rename_outputs_with_suffix({idx}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
