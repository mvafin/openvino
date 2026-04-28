// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_linear(const NodeContext& context) {
    // schema: aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto weight = context.get_input(1);
    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, true));
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

OutputVector translate_linear_ext(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto initial_x = x;
    auto weight = context.get_input(1);
    bool convert_back = false;
    if (weight.get_element_type() != element::f32) {
        // In case of patched linear it can have mixed fp16/bf16 and fp32 input type.
        // In other cases these conversion is not required.
        weight = context.mark_node(std::make_shared<v0::Convert>(weight, element::f32));
        if (x.get_element_type() != element::f32) {
            // Convert to f32
            x = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
            convert_back = true;
        }
    }
    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, true));
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);

        if (bias.get_element_type() != element::f32) {
            // Same reason as for weight.
            bias = context.mark_node(std::make_shared<v0::Convert>(bias, element::f32));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    if (convert_back) {
        matmul = context.mark_node(std::make_shared<v1::ConvertLike>(matmul, initial_x));
    }
    return {matmul};
};

namespace {

// Write a u4 value at a given linear index in a packed u4 buffer.
inline void set_u4(uint8_t* data, size_t idx, uint8_t val) {
    size_t byte_idx = idx / 2;
    if (idx & 1)
        data[byte_idx] = (data[byte_idx] & 0x0F) | static_cast<uint8_t>((val & 0x0F) << 4);
    else
        data[byte_idx] = (data[byte_idx] & 0xF0) | (val & 0x0F);
}

Output<Node> low_precision_subgraph(const NodeContext& context,
                                    const Output<Node>& x,
                                    const Output<Node>& weights,
                                    const Output<Node>& zero_points,
                                    const Output<Node>& scales,
                                    const Output<Node>& out_shape) {
    auto new_qweight = context.mark_node(std::make_shared<v0::Convert>(weights, scales.get_element_type()));
    auto new_qzeros = context.mark_node(std::make_shared<v0::Convert>(zero_points, scales.get_element_type()));

    auto w_s = context.mark_node(std::make_shared<v1::Subtract>(new_qweight, new_qzeros));
    auto weight = context.mark_node(std::make_shared<v1::Multiply>(w_s, scales));
    auto weight_shape = weights.get_shape();
    if (out_shape.get_node() != nullptr) {
        weight = context.mark_node(std::make_shared<v1::Reshape>(weight, out_shape, false));
    }
    weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, x));
    return weight;
}

uint32_t rearrange_awq_bits(uint32_t num) {
    uint32_t result = 0;
    uint32_t mask = 0xF;

    // Rearrange each 4-bit part in accordance with the AWQ i32->u4 unpacking schema
    result |= (num & (mask << 0)) << 0;
    result |= (num & (mask << 16)) >> 12;
    result |= (num & (mask << 4)) << 4;
    result |= (num & (mask << 20)) >> 8;
    result |= (num & (mask << 8)) << 8;
    result |= (num & (mask << 24)) >> 4;
    result |= (num & (mask << 12)) << 12;
    result |= (num & (mask << 28)) >> 0;

    return result;
}

Output<Node> rearrange_constant(const Output<Node>& c, uint32_t groups) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "weight must be Constant.");
    FRONT_END_OP_CONVERSION_CHECK(constant->get_byte_size() == shape_size(constant->get_shape()) * sizeof(uint32_t),
                                  "AWQ constant storage size does not match expected int32 packing.");
    auto src = constant->get_data_ptr<uint32_t>();
    auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D constants are supported.");
    FRONT_END_OP_CONVERSION_CHECK(groups > 0, "AWQ group size must be greater than 0.");
    FRONT_END_OP_CONVERSION_CHECK(initial_shape[0] % groups == 0,
                                  "AWQ qweight first dimension must be divisible by group size.");
    auto new_shape = Shape{initial_shape[0] / groups, groups, initial_shape[1] * 8};
    auto new_qweight = std::make_shared<v0::Constant>(element::u4, new_shape);
    auto dst = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(new_qweight->get_data_ptr()));
    const size_t src_elements_count = shape_size(initial_shape);
    const size_t dst_elements_count = shape_size(new_shape) / 8;
    FRONT_END_OP_CONVERSION_CHECK(dst_elements_count == src_elements_count,
                                  "Unexpected AWQ constant size mismatch after rearrangement.");
    for (size_t i = 0; i < src_elements_count; i++) {
        dst[i] = rearrange_awq_bits(src[i]);
    }
    new_qweight->set_friendly_name(constant->get_friendly_name());
    return new_qweight;
}

// GPTQ packs 8 u4 values per int32 along the INPUT dimension.
// Each int32 at [i, j] holds u4 values for rows i*8..i*8+7 at column j.
// This is a transpose of the inner dims: [K, N, 8] u4 → [K, 8, N] u4.
Output<Node> unpack_gptq_qweight(const Output<Node>& c, int64_t group_size) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "qweight must be Constant.");
    FRONT_END_OP_CONVERSION_CHECK(constant->get_byte_size() == shape_size(constant->get_shape()) * sizeof(uint32_t),
                                  "GPTQ qweight storage size does not match expected int32 packing.");
    auto src = constant->get_data_ptr<uint32_t>();
    auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D qweight constants are supported.");
    const size_t K = initial_shape[0];  // in_features / 8
    const size_t N = initial_shape[1];  // out_features
    const size_t in_features = K * 8;
    FRONT_END_OP_CONVERSION_CHECK(group_size > 0, "GPTQ group_size must be greater than 0.");
    FRONT_END_OP_CONVERSION_CHECK(static_cast<size_t>(group_size) <= in_features,
                                  "GPTQ group_size must not exceed in_features.");
    const size_t group_size_u = static_cast<size_t>(group_size);
    FRONT_END_OP_CONVERSION_CHECK(in_features % group_size_u == 0, "GPTQ in_features must be divisible by group_size.");
    const size_t n_groups = in_features / group_size_u;
    auto new_shape = Shape{n_groups, group_size_u, N};
    auto new_const = std::make_shared<v0::Constant>(element::u4, new_shape, 0);
    auto dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            uint32_t val = src[i * N + j];
            for (size_t k = 0; k < 8; ++k) {
                set_u4(dst, (i * 8 + k) * N + j, (val >> (k * 4)) & 0xF);
            }
        }
    }
    new_const->set_friendly_name(constant->get_friendly_name());
    return new_const;
}

// GPTQ qzeros: packs 8 u4 values per int32 along the OUTPUT dimension.
// Byte layout matches u4 layout directly — single-pass copy with +1 offset.
Output<Node> unpack_gptq_qzeros(const Output<Node>& c) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "qzeros must be Constant.");
    FRONT_END_OP_CONVERSION_CHECK(constant->get_byte_size() == shape_size(constant->get_shape()) * sizeof(uint32_t),
                                  "GPTQ qzeros storage size does not match expected int32 packing.");
    auto src = reinterpret_cast<const uint8_t*>(constant->get_data_ptr());
    auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D qzeros constants are supported.");
    auto new_shape = Shape{initial_shape[0], 1, initial_shape[1] * 8};
    auto new_const = std::make_shared<v0::Constant>(element::u4, new_shape);
    auto dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));
    // Apply +1 offset per u4 value while copying (GPTQ stores zp-1)
    const size_t n_bytes = shape_size(initial_shape) * sizeof(uint32_t);
    for (size_t i = 0; i < n_bytes; ++i) {
        uint8_t lo = (src[i] & 0x0F) + 1;
        uint8_t hi = ((src[i] >> 4) & 0x0F) + 1;
        dst[i] = (lo & 0x0F) | static_cast<uint8_t>((hi & 0x0F) << 4);
    }
    new_const->set_friendly_name(constant->get_friendly_name());
    return new_const;
}

}  // namespace

OutputVector translate_linear_awq(const NodeContext& context) {
    num_inputs_check(context, 4, 7);
    auto x = context.get_input(0);
    auto qweight = context.get_input(1);
    auto qzeros = context.get_input(2);
    auto scales = context.get_input(3);
    auto groups = context.const_input<int64_t>(4);
    auto bits = context.const_input<int64_t>(5);

    FRONT_END_OP_CONVERSION_CHECK(bits == 4, "Only 4 bit AWQ is supported.");

    auto new_qweight = rearrange_constant(qweight, static_cast<uint32_t>(groups));
    auto new_qzeros = rearrange_constant(qzeros, 1);
    FRONT_END_OP_CONVERSION_CHECK(scales.get_partial_shape().is_static(), "Scales must be constant.");
    auto scales_shape = scales.get_shape();
    auto new_scales_shape =
        v0::Constant::create(element::i32, {3}, std::vector<uint64_t>{scales_shape[0], 1, scales_shape[1]});
    auto new_scales = context.mark_node(std::make_shared<v1::Reshape>(scales, new_scales_shape, false));
    auto out_shape =
        v0::Constant::create(element::i32, {2}, std::vector<int32_t>{static_cast<int32_t>(qweight.get_shape()[0]), -1});
    auto weight = low_precision_subgraph(context, x, new_qweight, new_qzeros, new_scales, out_shape);

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, false));
    if (!context.input_is_none(6)) {
        auto bias = context.get_input(6);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, x));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

OutputVector translate_linear_gptq(const NodeContext& context) {
    // ov_ext::gptq_gemm(input, qweight, qzeros, scales, group_size, bits, sym, bias?)
    num_inputs_check(context, 7, 8);
    auto x = context.get_input(0);
    auto qweight = context.get_input(1);
    auto qzeros = context.get_input(2);
    auto scales = context.get_input(3);
    auto group_size = context.const_input<int64_t>(4);
    auto bits = context.const_input<int64_t>(5);
    auto sym = context.const_input<bool>(6);

    FRONT_END_OP_CONVERSION_CHECK(bits == 4, "Only 4-bit GPTQ is supported.");

    // qweight: [in_features/8, out_features] int32 → [n_groups, group_size, out_features] u4
    auto new_qweight = unpack_gptq_qweight(qweight, group_size);

    Output<Node> new_qzeros;
    if (sym) {
        // Symmetric quantisation: zero point is always 8 (midpoint of u4 range)
        new_qzeros = context.mark_node(v0::Constant::create(element::u4, Shape{}, std::vector<uint8_t>{8}));
    } else {
        // qzeros: [n_groups, out_features/8] int32 → [n_groups, 1, out_features] u4 (with +1 offset)
        new_qzeros = unpack_gptq_qzeros(qzeros);
    }

    FRONT_END_OP_CONVERSION_CHECK(scales.get_partial_shape().is_static(), "Scales must be constant.");
    auto scales_shape = scales.get_shape();
    FRONT_END_OP_CONVERSION_CHECK(scales_shape.size() == 2,
                                  "GPTQ scales input is expected to be 2D, but got rank ",
                                  scales_shape.size(),
                                  ".");
    auto new_scales_shape =
        v0::Constant::create(element::i32, {3}, std::vector<uint64_t>{scales_shape[0], 1, scales_shape[1]});
    auto new_scales = context.mark_node(std::make_shared<v1::Reshape>(scales, new_scales_shape, false));
    // Reshape dequantised weight to [in_features, out_features] for matmul
    auto qweight_shape = qweight.get_shape();
    auto out_shape =
        v0::Constant::create(element::i32, {2}, std::vector<int32_t>{static_cast<int32_t>(qweight_shape[0] * 8), -1});
    auto weight = low_precision_subgraph(context, x, new_qweight, new_qzeros, new_scales, out_shape);

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, false));
    if (!context.input_is_none(7)) {
        auto bias = context.get_input(7);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, x));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

OutputVector translate_linear_bitnet(const NodeContext& context) {
    num_inputs_check(context, 3, 4);
    const auto x = context.get_input(0);
    const auto weight = context.get_input(1);
    const auto scales = context.get_input(2);

    const auto constant = ov::as_type_ptr<v0::Constant>(weight.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "weight must be Constant.");
    const auto src = reinterpret_cast<const uint8_t*>(constant->get_data_ptr());
    const auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D constants are supported.");
    const uint8_t values_per_pack = 4;  // Number of 2-bit values packed into a byte
    const size_t rows = initial_shape[0];
    const size_t cols = initial_shape[1];
    FRONT_END_OP_CONVERSION_CHECK(cols % values_per_pack == 0,
                                  "The second dimension of weight must be divisible by 4.");
    const auto new_shape = Shape{rows * values_per_pack, cols};
    auto new_weight = std::make_shared<v0::Constant>(element::u2, new_shape, 0);
    auto dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_weight->get_data_ptr()));
    const size_t row_len = cols / values_per_pack;
    // This lambda extracts 2 bits from each of 4 consecutive bytes at a given bit position,
    // then packs them into a single byte, placing each 2-bit value in its respective position (6, 4, 2, 0).
    const auto reorder_bitnet_2bit_values =
        [](const uint8_t* const src, const size_t src_idx, const size_t pos) -> uint8_t {
        const uint8_t values_per_byte = 4;
        const uint8_t value_mask = 0x3;
        const uint8_t value_size = 2;  // Size of each value is 2 bits
        uint8_t value{};               // Should be zeroed

        for (size_t value_idx = 0; value_idx != values_per_byte; ++value_idx) {
            value |= ((src[src_idx + value_idx] >> pos) & value_mask) << value_idx * value_size;
        }
        return value;
    };
    // In each 8bit value we have 4 2-bit values, first value contains first element, second value first element of a
    // next row. We need to repack them in contiguous way.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < row_len; ++j) {
            const size_t src_idx = values_per_pack * j + i * cols;
            dst[j + (i + 0 * rows) * row_len] = reorder_bitnet_2bit_values(src, src_idx, 0);
            dst[j + (i + 1 * rows) * row_len] = reorder_bitnet_2bit_values(src, src_idx, 2);
            dst[j + (i + 2 * rows) * row_len] = reorder_bitnet_2bit_values(src, src_idx, 4);
            dst[j + (i + 3 * rows) * row_len] = reorder_bitnet_2bit_values(src, src_idx, 6);
        }
    }
    new_weight->set_friendly_name(constant->get_friendly_name());
    const auto zero_point = context.mark_node(std::make_shared<v0::Constant>(element::u2, Shape{}, 1));
    auto mm_weight = low_precision_subgraph(context, x, new_weight, zero_point, scales, {});

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, mm_weight, false, true));
    if (!context.input_is_none(3)) {
        auto bias = context.get_input(3);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, x));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

namespace {

/// Build the standard MXFP4 weight decompression subgraph that CPU/GPU plugins
/// recognise and fuse into MatMul natively.
///
/// The graph emitted is:
///   Constant(f4e2m1)[E, O, G, 32]
///       → Convert(f32)
///       → Multiply( Constant(f8e8m0)[E, O, G] → Convert(f32) → Unsqueeze )
///       → Reshape[E, O, I]
///
/// The Multiply→Reshape→MatMul chain is detected by
/// Transformations::is_decompression_multiply() in the CPU plugin, which marks
/// it for native on-the-fly decompression inside oneDNN (no f32 materialisation).
///
/// blocks: [E, O, G, 16] uint8  (each byte = 2 E2M1 values)
/// scales: [E, O, G]    uint8  (each byte = one E8M0 exponent)
///
/// Returns: weight [1, E, O, I] in f32 (logically; never materialised when fused).
/// The leading 1 enables NumPy-style broadcast with [T, 1, 1, I] in 4D MatMul.
Output<Node> mxfp4_decompression_subgraph(const NodeContext& context,
                                          const Output<Node>& blocks,
                                          const Output<Node>& scales) {
    auto blocks_const = ov::as_type_ptr<v0::Constant>(blocks.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(blocks_const, "MXFP4 blocks must be a Constant.");
    auto blocks_shape = blocks_const->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(blocks_shape.size() == 4, "MXFP4 blocks must be 4D.");
    const auto num_experts = blocks_shape[0];
    const auto out_features = blocks_shape[1];
    const auto num_groups = blocks_shape[2];
    const auto bytes_per_group = blocks_shape[3];               // 16
    const auto in_features = num_groups * bytes_per_group * 2;  // 2 f4e2m1 values per byte

    // Reinterpret uint8 blocks as f4e2m1: [E, O, G, 16] u8 → [E, O, G, 32] f4e2m1
    auto f4e2m1_shape = Shape{num_experts, out_features, num_groups, bytes_per_group * 2};
    auto f4e2m1_weight = std::make_shared<v0::Constant>(element::f4e2m1, f4e2m1_shape, blocks_const->get_data_ptr());
    f4e2m1_weight->set_friendly_name(blocks_const->get_friendly_name() + "_f4e2m1");

    // Convert f4e2m1 → f32   [E, O, G, 32]
    auto weight_f32 = context.mark_node(std::make_shared<v0::Convert>(f4e2m1_weight, element::f32));

    // Reinterpret uint8 scales as f8e8m0: [E, O, G]
    auto scales_const = ov::as_type_ptr<v0::Constant>(scales.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(scales_const, "MXFP4 scales must be a Constant.");
    auto f8e8m0_scales =
        std::make_shared<v0::Constant>(element::f8e8m0, scales_const->get_shape(), scales_const->get_data_ptr());
    f8e8m0_scales->set_friendly_name(scales_const->get_friendly_name() + "_f8e8m0");

    // Convert f8e8m0 → f32, then Unsqueeze to [E, O, G, 1] for broadcast
    auto scales_f32 = context.mark_node(std::make_shared<v0::Convert>(f8e8m0_scales, element::f32));
    auto unsqueeze_axis = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{3});
    auto scales_4d = context.mark_node(std::make_shared<v0::Unsqueeze>(scales_f32, unsqueeze_axis));

    // Multiply: [E, O, G, 32] * [E, O, G, 1] → [E, O, G, 32]
    auto scaled_weight = context.mark_node(std::make_shared<v1::Multiply>(weight_f32, scales_4d));

    // Reshape to [1, E, O, I]  — merges the group dimension, adds leading 1 for broadcast
    auto target_shape = v0::Constant::create(element::i64,
                                             Shape{4},
                                             std::vector<int64_t>{1,
                                                                  static_cast<int64_t>(num_experts),
                                                                  static_cast<int64_t>(out_features),
                                                                  static_cast<int64_t>(in_features)});
    auto weight_4d = context.mark_node(std::make_shared<v1::Reshape>(scaled_weight, target_shape, false));

    return weight_4d;
}

}  // anonymous namespace

OutputVector translate_mxfp4_experts(const NodeContext& context) {
    // ov_ext::mxfp4_experts(input, gate_up_blocks, gate_up_scales, gate_up_bias,
    //                        down_blocks, down_scales, down_bias,
    //                        router_indices, routing_weights) -> Tensor
    num_inputs_check(context, 9, 9);
    auto input = context.get_input(0);            // [tokens, hidden]
    auto gate_up_blocks = context.get_input(1);   // [E, 2*inter, hidden//32, 16] uint8
    auto gate_up_scales = context.get_input(2);   // [E, 2*inter, hidden//32] uint8
    auto gate_up_bias = context.get_input(3);     // [E, 2*inter]
    auto down_blocks = context.get_input(4);      // [E, hidden, inter//32, 16] uint8
    auto down_scales = context.get_input(5);      // [E, hidden, inter//32] uint8
    auto down_bias = context.get_input(6);        // [E, hidden]
    auto router_indices = context.get_input(7);   // [tokens, top_k]
    auto routing_weights = context.get_input(8);  // [tokens, top_k]

    // Build decompression subgraphs — CPU plugin will recognise the
    // Const(f4e2m1)→Convert→Multiply(scales)→Reshape→MatMul pattern and
    // fuse the decompression into oneDNN, avoiding f32 weight materialisation.
    //
    // gate_up_w: [1, E, 2*inter, hidden]  (out=2*inter, in=hidden)
    auto gate_up_w = mxfp4_decompression_subgraph(context, gate_up_blocks, gate_up_scales);
    // down_w: [1, E, hidden, inter]  (out=hidden, in=inter)
    auto down_w = mxfp4_decompression_subgraph(context, down_blocks, down_scales);

    // Convert biases to f32
    gate_up_bias = context.mark_node(std::make_shared<v0::Convert>(gate_up_bias, element::f32));
    down_bias = context.mark_node(std::make_shared<v0::Convert>(down_bias, element::f32));

    // --- Gate-up projection (all experts at once via 4D broadcast MatMul) ---
    // weight: [1, E, 2*inter, hidden] with transpose_b=true → [1, E, hidden, 2*inter]
    // input:  [tokens, hidden] → [tokens, 1, 1, hidden]
    // MatMul: [tokens, 1, 1, hidden] @ [1, E, hidden, 2*inter] → [tokens, E, 1, 2*inter]
    auto unsqueeze_12 = v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 2});
    auto input_4d = context.mark_node(std::make_shared<v0::Unsqueeze>(input, unsqueeze_12));

    auto gate_up_4d = context.mark_node(std::make_shared<v0::MatMul>(input_4d, gate_up_w, false, true));
    // gate_up_4d: [tokens, E, 1, 2*inter]

    // Reshape to [tokens, E, 2*inter] — squeeze the matmul dim
    auto axis_0 = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto tokens_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i64));
    auto tokens_dim = context.mark_node(
        std::make_shared<v8::Gather>(tokens_shape,
                                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                     axis_0));
    auto hidden_dim = context.mark_node(
        std::make_shared<v8::Gather>(tokens_shape,
                                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}),
                                     axis_0));

    // Get E and 2*inter from the weight shape [1, E, 2*inter, hidden]
    auto w_shape = context.mark_node(std::make_shared<v3::ShapeOf>(gate_up_w, element::i64));
    auto e_dim = context.mark_node(
        std::make_shared<v8::Gather>(w_shape,
                                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}),
                                     axis_0));
    auto gate_up_out_dim = context.mark_node(
        std::make_shared<v8::Gather>(w_shape,
                                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2}),
                                     axis_0));

    auto gate_up_3d_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{tokens_dim, e_dim, gate_up_out_dim}, 0));
    auto gate_up_all = context.mark_node(std::make_shared<v1::Reshape>(gate_up_4d, gate_up_3d_shape, false));
    // gate_up_all: [tokens, E, 2*inter]

    // Add bias: [E, 2*inter] broadcasts to [tokens, E, 2*inter]
    gate_up_all = context.mark_node(std::make_shared<v1::Add>(gate_up_all, gate_up_bias));

    // --- Select per-token expert results using router_indices ---
    auto ri_shape = context.mark_node(std::make_shared<v3::ShapeOf>(router_indices, element::i64));
    auto top_k_val = context.mark_node(
        std::make_shared<v8::Gather>(ri_shape,
                                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}),
                                     axis_0));

    // Gather per-token expert gate_up results:
    // gate_up_all[tokens, E, 2*inter] → gather along axis=1 using router_indices
    // For each token t, gather gate_up_all[t, router_indices[t, k], :] for k in top_k
    // Use GatherND with batch_dims=1:
    //   data:    [tokens, E, 2*inter]
    //   indices: [tokens, top_k, 1]  (the "1" selects along the E dimension)
    //   result:  [tokens, top_k, 2*inter]
    auto indices_unsqueezed = context.mark_node(
        std::make_shared<v0::Unsqueeze>(router_indices,
                                        v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2})));
    // indices_unsqueezed: [tokens, top_k, 1]

    auto gate_up_selected = context.mark_node(std::make_shared<v8::GatherND>(gate_up_all, indices_unsqueezed, 1));
    // gate_up_selected: [tokens, top_k, 2*inter]

    // Flatten to [tokens*top_k, 2*inter] for activation
    auto n_total = context.mark_node(std::make_shared<v1::Multiply>(tokens_dim, top_k_val));
    auto one = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto flat_2d = context.mark_node(std::make_shared<v0::Concat>(
        OutputVector{n_total, v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1})},
        0));
    auto gate_up_result = context.mark_node(std::make_shared<v1::Reshape>(gate_up_selected, flat_2d, false));

    // --- SwiGLU activation (interleaved gate/up) ---
    // Reshape to [tokens*top_k, inter, 2], then split
    auto inter_shape = context.mark_node(std::make_shared<v0::Concat>(
        OutputVector{n_total,
                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}),
                     v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2})},
        0));
    auto gate_up_3d = context.mark_node(std::make_shared<v1::Reshape>(gate_up_result, inter_shape, false));

    auto axis_2 = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2});
    auto idx_0 = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto idx_1 = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto gate = context.mark_node(std::make_shared<v8::Gather>(gate_up_3d, idx_0, axis_2));
    auto up = context.mark_node(std::make_shared<v8::Gather>(gate_up_3d, idx_1, axis_2));

    // Squeeze trailing dim: [tokens*top_k, inter, 1] → [tokens*top_k, inter]
    gate = context.mark_node(std::make_shared<v1::Reshape>(gate, flat_2d, false));
    up = context.mark_node(std::make_shared<v1::Reshape>(up, flat_2d, false));

    // gate = clamp(gate, max=7.0)
    gate = context.mark_node(std::make_shared<v0::Clamp>(gate, -std::numeric_limits<double>::infinity(), 7.0));
    // up = clamp(up, min=-7.0, max=7.0)
    up = context.mark_node(std::make_shared<v0::Clamp>(up, -7.0, 7.0));

    // glu = gate * sigmoid(gate * 1.702)
    Output<Node> alpha = v0::Constant::create(element::f32, Shape{}, std::vector<float>{1.702f});
    auto gate_alpha = context.mark_node(std::make_shared<v1::Multiply>(gate, alpha));
    auto sigmoid_out = context.mark_node(std::make_shared<v0::Sigmoid>(gate_alpha));
    auto glu = context.mark_node(std::make_shared<v1::Multiply>(gate, sigmoid_out));

    // gated_output = (up + 1) * glu
    Output<Node> one_f = v0::Constant::create(element::f32, Shape{}, std::vector<float>{1.0f});
    auto up_plus_1 = context.mark_node(std::make_shared<v1::Add>(up, one_f));
    auto gated_output = context.mark_node(std::make_shared<v1::Multiply>(up_plus_1, glu));
    // gated_output: [tokens*top_k, inter]

    // --- Down projection (all experts at once via 4D broadcast MatMul) ---
    // gated_output: [tokens*top_k, inter] → [tokens*top_k, 1, 1, inter]
    // down_w: [1, E, hidden, inter] with transpose_b=true → [1, E, inter, hidden]
    // MatMul: [tokens*top_k, 1, 1, inter] @ [1, E, inter, hidden] → [tokens*top_k, E, 1, hidden]
    auto gated_4d = context.mark_node(std::make_shared<v0::Unsqueeze>(gated_output, unsqueeze_12));
    // gated_4d: [tokens*top_k, 1, 1, inter]

    auto down_4d = context.mark_node(std::make_shared<v0::MatMul>(gated_4d, down_w, false, true));
    // down_4d: [tokens*top_k, E, 1, hidden]

    // Reshape to [tokens*top_k, E, hidden] — squeeze the matmul dim
    auto down_3d_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{n_total, e_dim, hidden_dim}, 0));
    auto down_all = context.mark_node(std::make_shared<v1::Reshape>(down_4d, down_3d_shape, false));
    // down_all: [tokens*top_k, E, hidden]

    // Add bias: [E, hidden] broadcasts to [tokens*top_k, E, hidden]
    down_all = context.mark_node(std::make_shared<v1::Add>(down_all, down_bias));

    // Select the correct expert for each (token, k) pair.
    // flat_indices: [tokens*top_k] — expert index for each (token, k) pair
    auto flat_shape = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto flat_indices = context.mark_node(std::make_shared<v1::Reshape>(router_indices, flat_shape, false));
    // Unsqueeze to [tokens*top_k, 1] for GatherND batch_dims=1
    auto unsqueeze_1 = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto flat_indices_2d = context.mark_node(std::make_shared<v0::Unsqueeze>(flat_indices, unsqueeze_1));

    // Reshape down_all to [tokens*top_k, E, hidden]
    auto down_2d_select = context.mark_node(std::make_shared<v8::GatherND>(down_all, flat_indices_2d, 1));
    // down_2d_select: [tokens*top_k, hidden]

    // --- Weighted accumulation ---
    // routing_weights: [tokens, top_k] → [tokens*top_k, 1]
    auto rw_flat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{n_total, one}, 0));
    auto rw_flat = context.mark_node(std::make_shared<v1::Reshape>(routing_weights, rw_flat_shape, false));
    rw_flat = context.mark_node(std::make_shared<v0::Convert>(rw_flat, element::f32));

    // Weight the outputs: [tokens*top_k, hidden] * [tokens*top_k, 1]
    auto weighted = context.mark_node(std::make_shared<v1::Multiply>(down_2d_select, rw_flat));

    // Reshape to [tokens, top_k, hidden] and sum over top_k
    auto result_3d_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{tokens_dim, top_k_val, hidden_dim}, 0));
    auto result_3d = context.mark_node(std::make_shared<v1::Reshape>(weighted, result_3d_shape, false));
    auto reduce_axis = v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto result = context.mark_node(std::make_shared<v1::ReduceSum>(result_3d, reduce_axis, false));
    // result: [tokens, hidden]

    return {result};
}

OutputVector translate_bmm_ext(const NodeContext& context) {
    // ov_ext::bmm - batch matrix multiplication for 16-bit models
    // schema: ov_ext::bmm(Tensor batch1, Tensor batch2) -> Tensor
    num_inputs_check(context, 2, 2);
    auto batch1 = context.get_input(0);
    auto batch2 = context.get_input(1);
    const auto initial_batch1 = batch1;

    // Handle mixed precision - convert to f32 if inputs are fp16/bf16
    const bool convert_back = batch1.get_element_type() != element::f32;
    if (batch2.get_element_type() != element::f32) {
        batch2 = context.mark_node(std::make_shared<v0::Convert>(batch2, element::f32));
    }
    if (convert_back) {
        batch1 = context.mark_node(std::make_shared<v0::Convert>(batch1, element::f32));
    }

    // bmm: (b, n, m) @ (b, m, p) -> (b, n, p)
    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(std::move(batch1), std::move(batch2), false, false));

    if (convert_back) {
        matmul = context.mark_node(std::make_shared<v1::ConvertLike>(std::move(matmul), initial_batch1));
    }
    return {matmul};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
