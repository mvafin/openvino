// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Weight node construction for the native GGUF path. Quantized weights become a
// low-bitness compressed decompression subgraph (u4 for 4-bit, u8 for 8-bit; OpenVINO
// supports every GGUF bitness used here except 3-bit). Adapted from the genai gguf_utils
// make_int4/int8_weights helpers, working from the parser's compressed tensors
// (.weight u32-packed + .scales f16 + .biases f16).

#include "weights.hpp"

#include <cstring>

#include "openvino/core/except.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
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

// Shared shape helpers for grouped weight layouts. See make_int8 comment for why we keep
// all leading dims separate rather than flattening: the trailing Reshape must be
// (orig_rank+1)D -> orig_rank for the CompressedWeightsBlock matcher to fire.
ov::Shape grouped_weight_shape(const ov::Shape& orig, size_t num_groups, size_t group_size) {
    ov::Shape s(orig.begin(), orig.end() - 1);
    s.push_back(num_groups);
    s.push_back(group_size);
    return s;
}
ov::Shape per_group_shape(const ov::Shape& orig, size_t num_groups) {
    ov::Shape s(orig.begin(), orig.end() - 1);
    s.push_back(num_groups);
    s.push_back(1);
    return s;
}

// Q4_0 symmetric: i4 weights (XOR-converted from u4) + f16 scale, no zero-point.
// Emits: Multiply(Convert(i4_const, f16), scale) [-> Reshape] via low_precision_dequantize.
std::shared_ptr<ov::Node> make_q4_0(const std::string& name,
                                    const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // u32-packed i4 nibbles
    ov::Tensor scales = get(weights, name + ".scales");

    ov::Shape orig_shape = weight.get_shape();
    orig_shape.back() *= sizeof(uint32_t) / sizeof(uint8_t) * 2;  // u32 packs 8 i4
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);

    auto weights_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i4,
                                               grouped_shape,
                                               static_cast<const void*>(weight.data()),
                                               std::shared_ptr<void>(new ov::Tensor(weight), [](ov::Tensor* p) {
                                                   delete p;
                                               }));
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              {},
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// Symmetric 8-bit (Q8_0, Q5_0, Q6_K): i8 weights (pre-centered) + per-group f16 scale.
// No zero-point. Emits: Multiply(Convert(i8_const, f16), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_sym_int8(const std::string& name,
                                        const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // i8 byte per element
    ov::Tensor scales = get(weights, name + ".scales");

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);

    auto weights_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i8,
                                               grouped_shape,
                                               static_cast<const void*>(weight.data()),
                                               std::shared_ptr<void>(new ov::Tensor(weight), [](ov::Tensor* p) {
                                                   delete p;
                                               }));
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              {},
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// 4-bit asymmetric (Q4_1/Q4_K): u4 weights + per-group f16 scale + u4 integer zero-points.
// Emits: Multiply(Subtract(Convert(u4_const, f16), zp_u4), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_int4(const std::string& name,
                                    const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // u32-packed u4
    ov::Tensor scales = get(weights, name + ".scales");
    ov::Tensor zp_t = get(weights, name + ".zp");  // u4 packed integer zero-points

    ov::Shape orig_shape = weight.get_shape();
    orig_shape.back() *= sizeof(uint32_t) / sizeof(uint8_t) * 2;  // u32 packs 8 u4
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);
    zp_t.set_shape(scale_shape);

    auto weights_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::u4,
                                               grouped_shape,
                                               static_cast<const void*>(weight.data()),
                                               std::shared_ptr<void>(new ov::Tensor(weight), [](ov::Tensor* p) {
                                                   delete p;
                                               }));
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto zp_node = std::make_shared<ov::op::v0::Constant>(zp_t);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              zp_node->output(0),
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// Asymmetric 8-bit (Q5_K): i8 weights (raw 5-bit value, not centered) + f16 scales + u8 zp.
// Emits: Multiply(Subtract(Convert(i8_const, f16), zp_u8), scale) [-> Reshape].
std::shared_ptr<ov::Node> make_asym_int8(const std::string& name,
                                         const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, name + ".weight");  // i8 byte per element
    ov::Tensor scales = get(weights, name + ".scales");
    ov::Tensor zp_t = get(weights, name + ".zp");  // u8 integer zero-points

    const ov::Shape& orig_shape = weight.get_shape();
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto grouped_shape = grouped_weight_shape(orig_shape, num_groups, group_size);
    auto scale_shape = per_group_shape(orig_shape, num_groups);
    scales.set_shape(scale_shape);
    zp_t.set_shape(scale_shape);

    auto weights_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i8,
                                               grouped_shape,
                                               static_cast<const void*>(weight.data()),
                                               std::shared_ptr<void>(new ov::Tensor(weight), [](ov::Tensor* p) {
                                                   delete p;
                                               }));
    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto zp_node = std::make_shared<ov::op::v0::Constant>(zp_t);
    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);

    auto result = ov::decomposition::low_precision_dequantize(weights_node->output(0),
                                                              scales_node->output(0),
                                                              zp_node->output(0),
                                                              final_shape_node->output(0));
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// MXFP4 (gpt-oss): native compressed weights = f4e2m1 weight * f8e8m0 per-32 scale, both
// kept compressed so the CPU plugin decompresses on the fly (no host f16 expansion). The
// parser already deinterleaved into natural order; here we just build the subgraph.
std::shared_ptr<ov::Node> make_mxfp4(const std::string& base,
                                     const std::unordered_map<std::string, ov::Tensor>& weights) {
    ov::Tensor weight = get(weights, base + ".weight");  // f4e2m1 [.., cols]
    ov::Tensor scales = get(weights, base + ".scales");  // f8e8m0 [.., groups]

    ov::Shape orig_shape = weight.get_shape();
    size_t rows = 1;
    for (size_t i = 0; i + 1 < orig_shape.size(); ++i) {
        rows *= orig_shape[i];
    }
    const size_t num_groups = scales.get_shape().back();
    const size_t group_size = orig_shape.back() / num_groups;

    auto w_node = std::make_shared<ov::op::v0::Constant>(weight);
    auto w_grp = std::make_shared<ov::op::v1::Reshape>(
        w_node,
        ov::op::v0::Constant::create(ov::element::i64,
                                     {3},
                                     std::vector<int64_t>{(int64_t)rows, (int64_t)num_groups, (int64_t)group_size}),
        false);
    auto w_f16 = std::make_shared<ov::op::v0::Convert>(w_grp, ov::element::f16);

    scales.set_shape(ov::Shape{rows, num_groups, 1});
    auto s_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto s_f16 = std::make_shared<ov::op::v0::Convert>(s_node, ov::element::f16);

    auto scaled = std::make_shared<ov::op::v1::Multiply>(w_f16, s_f16, ov::op::AutoBroadcastType::NUMPY);
    auto final_shape =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(scaled, final_shape, false);
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
    case GGUF_TYPE_MXFP4:
        node = make_mxfp4(base, weights);
        break;
    case GGUF_TYPE_Q4_0:
        node = make_q4_0(base, weights);
        break;
    case GGUF_TYPE_Q4_1:
    case GGUF_TYPE_Q4_K:
        node = make_int4(base, weights);
        break;
    case GGUF_TYPE_Q5_K:
        node = make_asym_int8(base, weights);
        break;
    case GGUF_TYPE_Q5_0:
    case GGUF_TYPE_Q8_0:
    case GGUF_TYPE_Q6_K:
        node = make_sym_int8(base, weights);
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
    const bool has_scales = qtype == GGUF_TYPE_Q4_0 || qtype == GGUF_TYPE_Q4_1 || qtype == GGUF_TYPE_Q4_K ||
                            qtype == GGUF_TYPE_Q5_0 || qtype == GGUF_TYPE_Q8_0 || qtype == GGUF_TYPE_Q5_K ||
                            qtype == GGUF_TYPE_Q6_K;
    // Asymmetric types have integer zero-points stored in ".zp".
    const bool has_zp = qtype == GGUF_TYPE_Q4_1 || qtype == GGUF_TYPE_Q4_K || qtype == GGUF_TYPE_Q5_K;

    const ov::Tensor& w = get(weights, base + ".weight");
    const size_t total_rows = w.get_shape()[0];
    OPENVINO_ASSERT(n_q + n_k + n_v == total_rows, "[ggml] fused qkv row mismatch for ", base);

    const std::array<std::pair<size_t, size_t>, 3> ranges = {std::make_pair(size_t(0), n_q),
                                                             std::make_pair(n_q, n_q + n_k),
                                                             std::make_pair(n_q + n_k, total_rows)};
    const std::array<std::string, 3> parts = {base + ".q", base + ".k", base + ".v"};

    std::array<std::shared_ptr<ov::Node>, 3> out;
    for (size_t i = 0; i < 3; ++i) {
        const auto [r0, r1] = ranges[i];
        std::unordered_map<std::string, ov::Tensor> sub;
        std::unordered_map<std::string, gguf_tensor_type> subq;
        sub[parts[i] + ".weight"] = slice_rows(w, r0, r1);
        if (has_scales) {
            sub[parts[i] + ".scales"] = slice_rows(get(weights, base + ".scales"), r0, r1);
            subq[parts[i] + ".qtype"] = qtype;
        }
        if (has_zp) {
            sub[parts[i] + ".zp"] = slice_rows(get(weights, base + ".zp"), r0, r1);
        }
        out[i] = make_weight_node(parts[i], sub, subq);
    }
    return out;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
