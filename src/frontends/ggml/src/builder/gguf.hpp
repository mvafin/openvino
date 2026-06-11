// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace frontend {
namespace ggml {

// GGUF tensor (quantization) type ids, matching the on-disk GGUF format numbering.
// Only the subset the frontend handles is enumerated explicitly; others are accepted
// numerically but rejected at dequant time.
enum gguf_tensor_type {
    GGUF_TYPE_F32 = 0,
    GGUF_TYPE_F16 = 1,
    GGUF_TYPE_Q4_0 = 2,
    GGUF_TYPE_Q4_1 = 3,
    GGUF_TYPE_Q5_0 = 6,
    GGUF_TYPE_Q5_1 = 7,
    GGUF_TYPE_Q8_0 = 8,
    GGUF_TYPE_Q8_1 = 9,
    GGUF_TYPE_Q2_K = 10,
    GGUF_TYPE_Q3_K = 11,
    GGUF_TYPE_Q4_K = 12,
    GGUF_TYPE_Q5_K = 13,
    GGUF_TYPE_Q6_K = 14,
    GGUF_TYPE_Q8_K = 15,
    GGUF_TYPE_I8 = 24,
    GGUF_TYPE_I16 = 25,
    GGUF_TYPE_I32 = 26,
    GGUF_TYPE_I64 = 27,
    GGUF_TYPE_F64 = 28,
    GGUF_TYPE_BF16 = 30,
    GGUF_TYPE_MXFP4 = 39,  // 4-bit microscaling (gpt-oss): 1-byte E8M0 scale + 32x E2M1
    GGUF_TYPE_COUNT,
};

// GGUF metadata value type ids (the kv-pair value encoding).
enum gguf_value_type {
    GGUF_VALUE_TYPE_UINT8 = 0,
    GGUF_VALUE_TYPE_INT8 = 1,
    GGUF_VALUE_TYPE_UINT16 = 2,
    GGUF_VALUE_TYPE_INT16 = 3,
    GGUF_VALUE_TYPE_UINT32 = 4,
    GGUF_VALUE_TYPE_INT32 = 5,
    GGUF_VALUE_TYPE_FLOAT32 = 6,
    GGUF_VALUE_TYPE_BOOL = 7,
    GGUF_VALUE_TYPE_STRING = 8,
    GGUF_VALUE_TYPE_ARRAY = 9,
    GGUF_VALUE_TYPE_UINT64 = 10,
    GGUF_VALUE_TYPE_INT64 = 11,
    GGUF_VALUE_TYPE_FLOAT64 = 12,
};

// A single tensor descriptor parsed from the GGUF file. `weights_data` points directly
// into the memory-mapped file (zero-copy); the owning reader keeps the mapping alive.
// Field names mirror the classic gguf-tools layout so the dequant code in
// gguf_quants.cpp reads them unchanged.
struct gguf_tensor {
    const char* name = nullptr;  // points into the mmap (not null-terminated)
    size_t namelen = 0;
    uint32_t type = 0;  // gguf_tensor_type
    uint32_t ndim = 0;
    uint64_t dim[4] = {1, 1, 1, 1};
    uint64_t offset = 0;       // offset from the start of the tensor-data section
    uint64_t bsize = 0;        // total size in bytes
    uint64_t num_weights = 0;  // total number of elements
    const uint8_t* weights_data = nullptr;
};

// A metadata value: scalars are stored as an ov::Tensor of shape {} (so that numeric
// metadata round-trips through ov::element types), strings as std::string, and arrays as
// an ov::Tensor of shape {n} or a vector<std::string>.
using GGUFMetaData =
    std::variant<std::monostate, float, int, ov::Tensor, std::string, std::vector<std::string>, std::vector<int32_t>>;

using GGUFLoad = std::tuple<std::unordered_map<std::string, GGUFMetaData>,
                            std::unordered_map<std::string, ov::Tensor>,
                            std::unordered_map<std::string, gguf_tensor_type>>;

// printf-style string formatter used for building tensor names.
template <typename... Args>
std::string format(std::string fmt, Args... args);

// Reverse of the GGML dimension order (GGUF stores dims fastest-first).
ov::Shape get_shape(const gguf_tensor& tensor);

// Dequantize a quantized tensor (Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K) into OpenVINO layout
// (u32-packed weights + f16 scales + f16 biases) and insert into `a`/`qtype_map`.
void gguf_load_quantized(std::unordered_map<std::string, ov::Tensor>& a,
                         std::unordered_map<std::string, gguf_tensor_type>& qtype_map,
                         const gguf_tensor& tensor);

// Fully dequantize an MXFP4 tensor to a plain f16 ov::Tensor. MXFP4 uses a non-uniform
// E2M1 value LUT per element with a shared per-32 E8M0 scale, so it does not fit the
// uniform (weight*scale + bias) compressed layout; we materialize f16 weights directly.
ov::Tensor gguf_dequantize_mxfp4(const gguf_tensor& tensor);

// Parse a GGUF file: returns (metadata, tensors-by-ggml-name, per-tensor qtype). Tensor
// names are the raw GGUF names (e.g. "blk.0.attn_q.weight", "token_embd.weight").
// Quantized tensors are dequantized to (.weight + .scales + .biases) entries.
GGUFLoad get_gguf_data(const std::string& file);

// Extract the architecture config (architecture, layer_num, head_num, head_size,
// head_num_kv, hidden_size, max_position_embeddings, rms_norm_eps, rope_freq_base,
// file_type) from parsed metadata.
std::map<std::string, GGUFMetaData> config_from_meta(const std::unordered_map<std::string, GGUFMetaData>& metadata);

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
