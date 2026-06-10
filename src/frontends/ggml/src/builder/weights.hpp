// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "gguf.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace ggml {

// Build the OpenVINO node for a GGUF weight with base name `base` (the tensor name without
// the trailing ".weight", e.g. "blk.0.attn_q" or "token_embd"). Quantized weights become a
// low-bitness compressed subgraph (u4/u8 weights + zero-point + f16 scale, Convert ->
// Subtract -> Multiply -> Reshape), matching what the cgraph path produces; F16/F32 weights
// become a plain Constant. The returned node's output is f32 and feeds the translators.
//
// `weights` holds the parser output (tensors by ggml name; quantized tensors expanded to
// "<base>.weight" + "<base>.scales" + "<base>.biases"). `qtypes` maps "<base>.qtype" ->
// gguf_tensor_type.
std::shared_ptr<ov::Node> make_weight_node(const std::string& base,
                                           const std::unordered_map<std::string, ov::Tensor>& weights,
                                           const std::unordered_map<std::string, gguf_tensor_type>& qtypes);

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
