// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Weight node construction for the native GGUF path. Quantized weights become a
// low-bitness compressed decompression subgraph (u4 for 4-bit, u8 for 8-bit; OpenVINO
// supports every GGUF bitness used here except 3-bit). Adapted from the genai gguf_utils
// make_int4/int8_weights helpers, working from the parser's compressed tensors
// (.weight u32-packed + .scales f16 + .biases f16).

#include "weights.hpp"

#include <cmath>
#include <cstring>

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace frontend {
namespace ggml {

namespace {

const ov::Tensor& get(const std::unordered_map<std::string, ov::Tensor>& weights, const std::string& key) {
    auto it = weights.find(key);
    OPENVINO_ASSERT(it != weights.end(), "[ggml] missing weight tensor: ", key);
    return it->second;
}

// Copy a contiguous row range [r0, r1) of a 2D tensor. In the GGUF/parser layout, rows
// (output features) are independent: quant blocks tile along the last (input) dim, so a
// row slice is just a contiguous byte copy. Used to split a fused attn_qkv tensor (and its
// scales/biases) into separate q/k/v weights.
ov::Tensor slice_rows(const ov::Tensor& t, size_t r0, size_t r1) {
    const auto& s = t.get_shape();
    OPENVINO_ASSERT(s.size() == 2 && r1 <= s[0] && r0 <= r1, "[ggml] bad row slice");
    ov::Shape out_shape{r1 - r0, s[1]};
    ov::Tensor out(t.get_element_type(), out_shape);
    const size_t row_bytes = t.get_byte_size() / s[0];
    std::memcpy(out.data(), static_cast<const uint8_t*>(t.data()) + r0 * row_bytes, (r1 - r0) * row_bytes);
    return out;
}

// 8-bit (Q8_0): u8 weights + per-group f16 scale and f16 bias -> zero-point u8.
std::shared_ptr<ov::Node> make_int8(const std::string& name,
                                    const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // u32-packed
    ov::Tensor scales = get(weights, name + ".scales");
    ov::Tensor biases = get(weights, name + ".biases");

    ov::Shape orig_shape = weight.get_shape();
    orig_shape[1] *= sizeof(uint32_t) / sizeof(uint8_t);  // u32 packs 4 x u8
    // Derive the group layout from the scales tensor (number of groups per row) rather than
    // assuming a fixed group size: K-quants (Q6_K) and legacy quants (Q8_0) group
    // differently, and the parser already produced one scale/bias per group.
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape[1] / num_groups;

    ov::Shape scale_shape = scales.get_shape();
    scale_shape.push_back(1);
    scales.set_shape(scale_shape);
    biases.set_shape(scale_shape);

    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u8,
                                                              ov::Shape{orig_shape[0], num_groups, group_size},
                                                              static_cast<uint8_t*>(weight.data()),
                                                              nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    // zero-point = round(-bias / scale), stored u8.
    ov::Tensor zp(ov::element::u8, scale_shape);
    const auto* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const auto* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto* zp_data = zp.data<uint8_t>();
    for (size_t i = 0; i < zp.get_size(); ++i) {
        zp_data[i] = static_cast<uint8_t>(std::round(-1.f * static_cast<float>(bias_data[i]) / static_cast<float>(scale_data[i])));
    }
    auto zp_node = std::make_shared<ov::op::v0::Constant>(zp);

    auto w_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    auto zp_f16 = std::make_shared<ov::op::v0::Convert>(zp_node, ov::element::f16);
    auto w_zp = std::make_shared<ov::op::v1::Subtract>(w_f16, zp_f16, ov::op::AutoBroadcastType::NUMPY);
    auto w_zp_s = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);

    auto final_shape =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(w_zp_s, final_shape, false);
    return std::make_shared<ov::op::v0::Convert>(reshaped, ov::element::f32);
}

// 4-bit (Q4_0/Q4_1/Q4_K): u4 weights + per-group f16 scale and f16 bias -> u4 zero-point.
std::shared_ptr<ov::Node> make_int4(const std::string& name,
                                    const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // u32-packed
    ov::Tensor scales = get(weights, name + ".scales");
    ov::Tensor biases = get(weights, name + ".biases");

    ov::Shape orig_shape = weight.get_shape();
    orig_shape[1] *= sizeof(uint32_t) / sizeof(uint8_t) * 2;  // u32 packs 8 x u4
    // Group layout derived from the scales tensor (see make_int8).
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape[1] / num_groups;

    ov::Shape scale_shape = scales.get_shape();
    scale_shape.push_back(1);
    scales.set_shape(scale_shape);
    biases.set_shape(scale_shape);

    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u4,
                                                              ov::Shape{orig_shape[0], num_groups, group_size},
                                                              static_cast<uint8_t*>(weight.data()),
                                                              nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto w_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);

    // zero-point packed two u4 per byte.
    const auto* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const auto* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::Tensor zp(ov::element::u4, scale_shape);
    auto* zp_data = static_cast<uint8_t*>(zp.data());
    for (size_t i = 0; i < zp.get_byte_size(); ++i) {
        uint8_t b1 = static_cast<uint8_t>(
            std::round(-1.f * static_cast<float>(bias_data[i * 2]) / static_cast<float>(scale_data[i * 2])));
        uint8_t b2 = static_cast<uint8_t>(
            std::round(-1.f * static_cast<float>(bias_data[i * 2 + 1]) / static_cast<float>(scale_data[i * 2 + 1])));
        zp_data[i] = (b2 << 4) | (b1 & 0x0F);
    }
    auto zp_node = std::make_shared<ov::op::v0::Constant>(zp);
    auto zp_f16 = std::make_shared<ov::op::v0::Convert>(zp_node, ov::element::f16);

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    auto w_zp = std::make_shared<ov::op::v1::Subtract>(w_f16, zp_f16, ov::op::AutoBroadcastType::NUMPY);
    auto w_zp_s = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);

    auto final_shape =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(w_zp_s, final_shape, false);
    return std::make_shared<ov::op::v0::Convert>(reshaped, ov::element::f32);
}

}  // namespace

std::shared_ptr<ov::Node> make_weight_node(const std::string& base,
                                           const std::unordered_map<std::string, ov::Tensor>& weights,
                                           const std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    gguf_tensor_type qtype = GGUF_TYPE_F16;
    if (auto it = qtypes.find(base + ".qtype"); it != qtypes.end()) {
        qtype = it->second;
    }

    std::shared_ptr<ov::Node> node;
    switch (qtype) {
    case GGUF_TYPE_Q4_0:
    case GGUF_TYPE_Q4_1:
    case GGUF_TYPE_Q4_K:
        node = make_int4(base, weights);
        break;
    case GGUF_TYPE_Q5_0:  // 5-bit values stored in u8 by gguf_quants
    case GGUF_TYPE_Q8_0:
    case GGUF_TYPE_Q5_K:  // 5-bit values stored in u8 by gguf_quants
    case GGUF_TYPE_Q6_K:  // dequantized to (u8 weight + scale + bias) by gguf_quants
        node = make_int8(base, weights);
        break;
    case GGUF_TYPE_F16:
    case GGUF_TYPE_F32:
    case GGUF_TYPE_BF16:
    default: {
        // Non-quantized weight: a plain Constant (converted to f32 for the translators).
        ov::Tensor w = get(weights, base + ".weight");
        auto cnst = std::make_shared<ov::op::v0::Constant>(w);
        node = (w.get_element_type() == ov::element::f32)
                   ? std::static_pointer_cast<ov::Node>(cnst)
                   : std::make_shared<ov::op::v0::Convert>(cnst, ov::element::f32);
        break;
    }
    }
    node->set_friendly_name(base + ".weight");
    return node;
}

std::array<std::shared_ptr<ov::Node>, 3> make_fused_qkv_weights(
    const std::string& base,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    size_t n_q,
    size_t n_k,
    size_t n_v) {
    gguf_tensor_type qtype = GGUF_TYPE_F16;
    if (auto it = qtypes.find(base + ".qtype"); it != qtypes.end()) {
        qtype = it->second;
    }
    const bool quantized = qtype == GGUF_TYPE_Q4_0 || qtype == GGUF_TYPE_Q4_1 || qtype == GGUF_TYPE_Q4_K ||
                           qtype == GGUF_TYPE_Q8_0 || qtype == GGUF_TYPE_Q5_K || qtype == GGUF_TYPE_Q6_K;

    const ov::Tensor& w = get(weights, base + ".weight");
    const size_t total_rows = w.get_shape()[0];
    OPENVINO_ASSERT(n_q + n_k + n_v == total_rows, "[ggml] fused qkv row mismatch for ", base);

    // For each of q/k/v, assemble a temporary one-entry weights map holding the sliced
    // tensors under a fresh base name, then reuse make_weight_node.
    const std::array<std::pair<size_t, size_t>, 3> ranges = {
        std::make_pair(size_t(0), n_q), std::make_pair(n_q, n_q + n_k), std::make_pair(n_q + n_k, total_rows)};
    const std::array<std::string, 3> parts = {base + ".q", base + ".k", base + ".v"};

    std::array<std::shared_ptr<ov::Node>, 3> out;
    for (size_t i = 0; i < 3; ++i) {
        const auto [r0, r1] = ranges[i];
        std::unordered_map<std::string, ov::Tensor> sub;
        std::unordered_map<std::string, gguf_tensor_type> subq;
        sub[parts[i] + ".weight"] = slice_rows(w, r0, r1);
        if (quantized) {
            sub[parts[i] + ".scales"] = slice_rows(get(weights, base + ".scales"), r0, r1);
            sub[parts[i] + ".biases"] = slice_rows(get(weights, base + ".biases"), r0, r1);
            subq[parts[i] + ".qtype"] = qtype;
        }
        out[i] = make_weight_node(parts[i], sub, subq);
    }
    return out;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
