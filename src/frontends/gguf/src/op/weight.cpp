// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_map>

#include "node_context.hpp"
#include "op_table.hpp"
#include "quant/weights.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// A GGUF weight surfaced as a node. A weight is a ggml leaf (op type "GGML_OP_NONE") that the
// decoder marks as a weight. Two payload shapes are supported, both dequantized here so the
// decoder never builds OV nodes itself:
//
//   1. Native .gguf builder path: the parser already extracted the weight into OpenVINO tensors
//      (weight [+ scales [+ zp]]); the node carries them as attributes "gguf.blob.<sub>" plus the
//      quant type id "gguf_qtype" (and marker "gguf_weight"). We rebuild the make_weight_node(
//      base, weights, qtypes) inputs and call that overload -- the exact dequant path the builder
//      used before, unchanged numerics, and it handles fused-QKV parts / MoE experts uniformly.
//
//   2. llama.cpp cgraph path: the node carries the raw ggml bytes in "data" plus the ggml type
//      name "quant_type"; make_weight_node(data, quant_type, shape) re-extracts and builds.
//
// (Model-input leaves are also GGML_OP_NONE, but they are resolved to Parameters before the graph
// walk and never reach this translator.)
OutputVector translate_weight(const NodeContext& context) {
    // Path 1: pre-extracted tensors from the native builder.
    if (context.get_attribute<bool>("gguf_weight", false)) {
        const std::string base = "weight";
        std::unordered_map<std::string, ov::Tensor> weights;
        for (const char* sub : {"weight", "scales", "zp"}) {
            // scales/zp are absent for plain/symmetric types -> defaulted get_attribute (empty
            // tensor) so the missing-key Any doesn't throw.
            auto blob = context.get_attribute<ov::Tensor>(std::string("gguf.blob.") + sub, ov::Tensor());
            if (blob) {
                weights[base + "." + sub] = blob;
            }
        }
        FRONT_END_OP_CONVERSION_CHECK(weights.count(base + ".weight"),
                                      "GGML_OP_NONE weight leaf has no 'gguf.blob.weight' attribute");
        auto qtype = static_cast<gguf_tensor_type>(context.get_attribute<int>("gguf_qtype"));
        std::unordered_map<std::string, gguf_tensor_type> qtypes{{base + ".qtype", qtype}};
        auto node = make_weight_node(base, weights, qtypes);
        return rename_outputs_with_suffix({node}, context.get_name());
    }

    // Path 2: raw ggml bytes from a live cgraph decoder.
    auto data = context.get_attribute<ov::Tensor>("data");
    FRONT_END_OP_CONVERSION_CHECK(data, "GGML_OP_NONE node has no weight payload; not a weight");
    auto quant_type = context.get_attribute<std::string>("quant_type");
    auto shape = context.get_output_shape().to_shape();

    auto node = make_weight_node(data, quant_type, shape, context.get_name());
    return rename_outputs_with_suffix({node}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
