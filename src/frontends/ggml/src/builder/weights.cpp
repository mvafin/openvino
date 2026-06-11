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
    case GGUF_TYPE_Q8_0:
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

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
