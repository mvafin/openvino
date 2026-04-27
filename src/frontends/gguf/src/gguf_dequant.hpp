// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GGML quantized tensor → FP16 ov::Tensor.
//
// This module ports the reference dequantization routines from ggml's
// public-domain `ggml-quants.c` (https://github.com/ggml-org/ggml). They are
// straightforward and well-known, so we keep them inline rather than depending
// on an external GGML library.
//
// Strategy chosen for the PoC: dequantize every quant variant to FP16 at load
// time. This yields a single, simple code path and full file-format coverage
// at the cost of giving up the on-disk compression in the produced IR.
// Preserving Q4/Q8 layouts as native u4/u8 + Convert + Subtract(zp) + Multiply
// subgraphs is an obvious follow-up optimization; see README.

#pragma once

#include "gguf_reader.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Returns an owning ov::Tensor:
//   - F32         -> element::f32  (verbatim copy)
//   - F16 / BF16  -> element::f16 / element::bf16  (verbatim copy)
//   - any quantized type below -> element::f16 (computed by dequantizing)
//
// Supported quantized types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
//                            Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
// Unsupported (IQ* codebook-based, TQ* ternary, Q8_K intermediate) throw.
ov::Tensor materialize_tensor_f16_or_native(const TensorDescriptor& td, const uint8_t* raw);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
