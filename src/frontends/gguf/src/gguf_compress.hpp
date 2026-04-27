// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Build an OpenVINO graph node that materializes a GGUF tensor as FP32, while
// preserving the on-disk compression for the formats that map cleanly onto
// native OV element types:
//
//     Q4_0 -> u4 + group-wise scale (with implicit zero-point of 8)
//     Q4_1 -> u4 + group-wise scale + group-wise min
//     Q8_0 -> i8 + group-wise scale
//
// The resulting subgraph is the same `Convert -> [Subtract] -> Multiply`
// decompression pattern that NNCF / optimum-intel emit for compressed-weights
// MatMuls, so CPU/GPU plugins keep the constant in compressed form at runtime.
//
// All other quantized variants (Q5_0/Q5_1, Q2_K..Q6_K) fall back to
// dequantizing to FP16 at load time (see gguf_dequant.hpp).

#pragma once

#include <string>

#include "gguf_reader.hpp"
#include "openvino/core/node_output.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Produces an Output<Node> with element_type::f32 representing this tensor.
//
// `row_reorder_rope`, `head_dim`: if true, permute the 2D tensor's rows (or a
// 1D tensor's elements) using llama's GGUF -> HF "split halves" RoPE layout,
// applied at the byte level so the permutation works for both FP and quantized
// data without touching the values.
ov::Output<ov::Node> build_weight_node(const TensorDescriptor& td,
                                       const uint8_t* raw,
                                       const std::string& name,
                                       bool row_reorder_rope = false,
                                       int head_dim = 0);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
