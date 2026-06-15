// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the GGUF quantisation fill functions and make_weight_node.
//
// Each test constructs synthetic GGUF blocks in memory, runs the fill function, then either
// checks the output tensors directly or evaluates the full dequant subgraph (via constant-
// folding) and compares against a reference computed from the GGML formula.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// Internal GGUF frontend headers — included directly because these functions are not part
// of the installed public API.
#include "builder/gguf.hpp"
#include "builder/weights.hpp"

// For constant-folding the make_weight_node graph.
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov::frontend::gguf;

// ── Helpers ────────────────────────────────────────────────────────────────────

// Run constant-folding on `node`, assert the result is a single Constant, and return
// its data as a flat float vector.
static std::vector<float> eval_as_f32(std::shared_ptr<ov::Node> node) {
    auto result = std::make_shared<ov::op::v0::Result>(node);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{});
    ov::pass::Manager pm;
    pm.register_pass<ov::pass::ConstantFolding>();
    pm.run_passes(model);

    auto folded =
        std::dynamic_pointer_cast<ov::op::v0::Constant>(model->get_results()[0]->get_input_node_shared_ptr(0));
    EXPECT_NE(folded, nullptr) << "make_weight_node graph did not constant-fold to a single Constant";
    if (!folded)
        return {};
    const float* data = folded->get_data_ptr<float>();
    return std::vector<float>(data, data + ov::shape_size(folded->get_shape()));
}

// Build a minimal gguf_tensor that points at `raw_data`.
static gguf_tensor make_tensor(const void* raw_data, uint64_t bsize, uint64_t rows, uint64_t cols, uint32_t type) {
    gguf_tensor t{};
    t.type = type;
    t.ndim = 2;
    // GGUF stores dims fastest-first: dim[0]=cols, dim[1]=rows.
    t.dim[0] = cols;
    t.dim[1] = rows;
    t.num_weights = rows * cols;
    t.bsize = bsize;
    t.weights_data = static_cast<const uint8_t*>(raw_data);
    return t;
}

// ── Q4_0 (symmetric i4, group=32) ──────────────────────────────────────────────

TEST(GGUFQuantDequant, Q4_0_single_block) {
    // Block layout: 2-byte f16 scale + 16 bytes (32x4-bit weights).
    // weights in [0..15]; dequant = scale * (w - 8)  (i4 centre).
    const float scale_val = 0.5f;
    const uint16_t scale_bits = ov::float16(scale_val).to_bits();

    // All weights = nibble value 12 → i4 = 12-8 = 4 → dequant = 0.5*4 = 2.0
    std::vector<uint8_t> block(18, 0);
    std::memcpy(block.data(), &scale_bits, 2);
    // Each byte packs two nibbles, both = 12 (0xCC).
    std::fill(block.begin() + 2, block.end(), static_cast<uint8_t>(0xCC));

    gguf_tensor tensor = make_tensor(block.data(), block.size(), 1, 32, GGUF_TYPE_Q4_0);

    ov::Shape w_shape{1, 4};  // 32 i4 nibbles → 32/(8/4)=4 u32s in packed layout ... but
    // gguf_fill_q4_0 takes u32 shape where last dim = cols/8:
    // cols=32 → 32/8 = 4 u32s
    ov::Tensor weights(ov::element::u32, w_shape);
    ov::Tensor scales(ov::element::f16, ov::Shape{1, 1});

    gguf_fill_q4_0(tensor, weights, scales);

    // Verify scale
    const auto* s = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    EXPECT_FLOAT_EQ(static_cast<float>(s[0]), scale_val);

    // Build the weight node and evaluate.
    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["blk.weight"] = weights;
    w_map["blk.scales"] = scales;
    q_map["blk.qtype"] = GGUF_TYPE_Q4_0;
    auto node = make_weight_node("blk", w_map, q_map);
    auto out = eval_as_f32(node);

    ASSERT_EQ(out.size(), 32u);
    for (float v : out)
        EXPECT_NEAR(v, 2.0f, 1e-3f);
}

TEST(GGUFQuantDequant, Q4_0_symmetric_no_zero_point) {
    // Verify that nibble=0 → dequant = scale*(0-8) = -scale*8, and nibble=8 → 0.
    const float scale_val = 1.0f;
    const uint16_t scale_bits = ov::float16(scale_val).to_bits();

    std::vector<uint8_t> block(18, 0);
    std::memcpy(block.data(), &scale_bits, 2);
    // Low nibble = 0, high nibble = 8 → first 16 elements = 0, last 16 = 8.
    for (int i = 0; i < 16; ++i)
        block[2 + i] = 0x80;  // low=0, high=8

    gguf_tensor tensor = make_tensor(block.data(), block.size(), 1, 32, GGUF_TYPE_Q4_0);
    ov::Tensor weights(ov::element::u32, ov::Shape{1, 4});
    ov::Tensor scales(ov::element::f16, ov::Shape{1, 1});
    gguf_fill_q4_0(tensor, weights, scales);

    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["x.weight"] = weights;
    w_map["x.scales"] = scales;
    q_map["x.qtype"] = GGUF_TYPE_Q4_0;
    auto out = eval_as_f32(make_weight_node("x", w_map, q_map));
    ASSERT_EQ(out.size(), 32u);
    // First 16: nibble 0 → i4 = -8 → -8 * 1.0 = -8
    for (int i = 0; i < 16; ++i)
        EXPECT_NEAR(out[i], -8.0f, 1e-3f);
    // Last 16: nibble 8 → i4 = 0 → 0
    for (int i = 16; i < 32; ++i)
        EXPECT_NEAR(out[i], 0.0f, 1e-3f);
}

// ── Q8_0 (symmetric i8, group=32) ──────────────────────────────────────────────

TEST(GGUFQuantDequant, Q8_0_symmetric) {
    // Block: 2-byte f16 scale + 32 i8 weights. No zero-point.
    const float scale_val = 0.25f;
    const uint16_t scale_bits = ov::float16(scale_val).to_bits();

    std::vector<uint8_t> block(34, 0);
    std::memcpy(block.data(), &scale_bits, 2);
    // Weights: alternating 10 and -10 (stored as int8_t).
    for (int i = 0; i < 32; ++i)
        block[2 + i] = static_cast<uint8_t>(i % 2 == 0 ? 10 : -10);  // -10 as two's complement

    gguf_tensor tensor = make_tensor(block.data(), block.size(), 1, 32, GGUF_TYPE_Q8_0);
    ov::Tensor weights(ov::element::i8, ov::Shape{1, 32});
    ov::Tensor scales(ov::element::f16, ov::Shape{1, 1});
    gguf_fill_sym(tensor, weights, scales);

    // Check raw weights and scale
    const int8_t* w = weights.data<int8_t>();
    EXPECT_EQ(w[0], 10);
    EXPECT_EQ(w[1], -10);
    const auto* s = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    EXPECT_FLOAT_EQ(static_cast<float>(s[0]), scale_val);

    // Evaluate full dequant
    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["y.weight"] = weights;
    w_map["y.scales"] = scales;
    q_map["y.qtype"] = GGUF_TYPE_Q8_0;
    auto out = eval_as_f32(make_weight_node("y", w_map, q_map));
    ASSERT_EQ(out.size(), 32u);
    for (int i = 0; i < 32; ++i) {
        float expected = (i % 2 == 0 ? 10.f : -10.f) * scale_val;
        EXPECT_NEAR(out[i], expected, 1e-3f);
    }
}

TEST(GGUFQuantDequant, Q8_0_two_blocks) {
    // Two blocks with different scales; verify per-block scaling.
    const float s0 = 1.0f, s1 = 2.0f;
    const uint16_t s0_bits = ov::float16(s0).to_bits();
    const uint16_t s1_bits = ov::float16(s1).to_bits();

    std::vector<uint8_t> data(68, 0);
    std::memcpy(data.data(), &s0_bits, 2);
    for (int i = 0; i < 32; ++i)
        data[2 + i] = static_cast<uint8_t>(3);  // weight=3
    std::memcpy(data.data() + 34, &s1_bits, 2);
    for (int i = 0; i < 32; ++i)
        data[36 + i] = static_cast<uint8_t>(-5);  // weight=-5

    gguf_tensor tensor = make_tensor(data.data(), data.size(), 2, 32, GGUF_TYPE_Q8_0);
    ov::Tensor weights(ov::element::i8, ov::Shape{2, 32});
    ov::Tensor scales(ov::element::f16, ov::Shape{2, 1});
    gguf_fill_sym(tensor, weights, scales);

    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["z.weight"] = weights;
    w_map["z.scales"] = scales;
    q_map["z.qtype"] = GGUF_TYPE_Q8_0;
    auto out = eval_as_f32(make_weight_node("z", w_map, q_map));
    ASSERT_EQ(out.size(), 64u);
    for (int i = 0; i < 32; ++i)
        EXPECT_NEAR(out[i], 3.f * s0, 1e-2f);
    for (int i = 32; i < 64; ++i)
        EXPECT_NEAR(out[i], -5.f * s1, 1e-2f);
}

// ── Q5_0 (symmetric i8 after subtract-16, group=32) ───────────────────────────

TEST(GGUFQuantDequant, Q5_0_symmetric) {
    // Block: 2-byte f16 d + 4-byte qh + 16 bytes ql (32 weights, 5-bit each [0..31]).
    // dequant = d * (w - 16).
    const float d = 0.5f;
    const uint16_t d_bits = ov::float16(d).to_bits();

    // All weights = 20 → dequant = 0.5 * (20-16) = 2.0
    // w=20: low 4 bits = 4 (0x4), high bit = 1 (bit (20>>4)&1 = 1).
    // Block encodes 32 weights: ql has lo nibbles, qh has high bits packed 1/bit.
    std::vector<uint8_t> block(22, 0);
    std::memcpy(block.data(), &d_bits, 2);
    // qh at bytes 2..5: bit j set for each weight j where (w>>4)&1 == 1.
    // For w=20=0x14: high bit = 1 for all 32 weights.
    uint32_t qh = 0xFFFFFFFFu;
    std::memcpy(block.data() + 2, &qh, 4);
    // ql: low 4 bits = 4 for even j (j<16), high nibble = 4 for j>=16.
    // In extract: lo = (j<16) ? (ql[j]&0xF) : (ql[j-16]>>4); so ql[j] carries both.
    for (int j = 0; j < 16; ++j)
        block[6 + j] = 0x44;  // low nibble=4 (j<16), high nibble=4 (j+16)

    gguf_tensor tensor = make_tensor(block.data(), block.size(), 1, 32, GGUF_TYPE_Q5_0);
    ov::Tensor weights(ov::element::i8, ov::Shape{1, 32});
    ov::Tensor scales(ov::element::f16, ov::Shape{1, 1});
    gguf_fill_sym(tensor, weights, scales);

    // Raw weights should be 20-16 = 4 (as i8).
    const int8_t* w = weights.data<int8_t>();
    for (int i = 0; i < 32; ++i)
        EXPECT_EQ(w[i], 4) << "at i=" << i;

    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["q5.weight"] = weights;
    w_map["q5.scales"] = scales;
    q_map["q5.qtype"] = GGUF_TYPE_Q5_0;
    auto out = eval_as_f32(make_weight_node("q5", w_map, q_map));
    ASSERT_EQ(out.size(), 32u);
    for (float v : out)
        EXPECT_NEAR(v, 2.0f, 1e-3f);
}

// ── Q4_1 (asymmetric u4 + u4 zp, group=32) ─────────────────────────────────────

TEST(GGUFQuantDequant, Q4_1_asymmetric) {
    // Block: 2-byte f16 scale + 2-byte f16 min + 16 bytes (32x4-bit weights).
    // dequant = scale * w + min  →  scale * (w - zp)  where zp = round(-min/scale).
    const float scale_val = 1.0f;
    const float min_val = -4.0f;  // zp = round(4/1) = 4
    const uint16_t scale_bits = ov::float16(scale_val).to_bits();
    const uint16_t min_bits = ov::float16(min_val).to_bits();

    // All weights = nibble 6 → dequant = 1*(6-4) = 2
    std::vector<uint8_t> block(20, 0);
    std::memcpy(block.data(), &scale_bits, 2);
    std::memcpy(block.data() + 2, &min_bits, 2);
    std::fill(block.begin() + 4, block.end(), static_cast<uint8_t>(0x66));

    gguf_tensor tensor = make_tensor(block.data(), block.size(), 1, 32, GGUF_TYPE_Q4_1);

    // Shapes for Q4_1: 1 block → weights u32[1,4], scales f16[1,1], zp u4[1,1]
    ov::Tensor weights(ov::element::u32, ov::Shape{1, 4});
    ov::Tensor scales(ov::element::f16, ov::Shape{1, 1});
    ov::Tensor zp(ov::element::u4, ov::Shape{1, 1});
    gguf_fill_asym(tensor, weights, scales, zp);

    // Verify zp = 4
    const uint8_t* zp_data = static_cast<const uint8_t*>(zp.data());
    EXPECT_EQ(zp_data[0] & 0x0F, 4u);

    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["q41.weight"] = weights;
    w_map["q41.scales"] = scales;
    w_map["q41.zp"] = zp;
    q_map["q41.qtype"] = GGUF_TYPE_Q4_1;
    auto out = eval_as_f32(make_weight_node("q41", w_map, q_map));
    ASSERT_EQ(out.size(), 32u);
    for (float v : out)
        EXPECT_NEAR(v, 2.0f, 1e-2f);
}

TEST(GGUFQuantDequant, Q4_1_min_zero_is_symmetric) {
    // When min=0, zp=0, dequant = scale * w.
    const float scale_val = 0.5f;
    const float min_val = 0.0f;
    const uint16_t scale_bits = ov::float16(scale_val).to_bits();
    const uint16_t min_bits = ov::float16(min_val).to_bits();

    std::vector<uint8_t> block(20, 0);
    std::memcpy(block.data(), &scale_bits, 2);
    std::memcpy(block.data() + 2, &min_bits, 2);
    // All nibbles = 10 → dequant = 0.5 * 10 = 5.0
    std::fill(block.begin() + 4, block.end(), static_cast<uint8_t>(0xAA));

    gguf_tensor tensor = make_tensor(block.data(), block.size(), 1, 32, GGUF_TYPE_Q4_1);
    ov::Tensor weights(ov::element::u32, ov::Shape{1, 4});
    ov::Tensor scales(ov::element::f16, ov::Shape{1, 1});
    ov::Tensor zp(ov::element::u4, ov::Shape{1, 1});
    gguf_fill_asym(tensor, weights, scales, zp);

    const uint8_t* zp_data = static_cast<const uint8_t*>(zp.data());
    EXPECT_EQ(zp_data[0] & 0x0F, 0u);

    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["q41b.weight"] = weights;
    w_map["q41b.scales"] = scales;
    w_map["q41b.zp"] = zp;
    q_map["q41b.qtype"] = GGUF_TYPE_Q4_1;
    auto out = eval_as_f32(make_weight_node("q41b", w_map, q_map));
    ASSERT_EQ(out.size(), 32u);
    for (float v : out)
        EXPECT_NEAR(v, 5.0f, 1e-2f);
}

// ── make_weight_node F16 passthrough ──────────────────────────────────────────

TEST(GGUFQuantDequant, F16_passthrough) {
    // Non-quantized f16 tensor passes through as a Convert(f16->f32) Constant.
    constexpr size_t N = 4;
    const float src_vals[N] = {1.f, -2.f, 0.5f, 3.14f};
    ov::Tensor t(ov::element::f16, ov::Shape{1, N});
    auto* dst = t.data<ov::element_type_traits<ov::element::f16>::value_type>();
    for (size_t i = 0; i < N; ++i)
        dst[i] = ov::float16(src_vals[i]);

    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["emb.weight"] = t;
    q_map["emb.qtype"] = GGUF_TYPE_F16;
    auto out = eval_as_f32(make_weight_node("emb", w_map, q_map));
    ASSERT_EQ(out.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out[i], src_vals[i], 1e-2f);
}

// ── Q4_0 two rows (basic multi-row check) ─────────────────────────────────────

TEST(GGUFQuantDequant, Q4_0_two_rows) {
    // Two blocks: different scales, all weights = nibble 4 → i4 = -4 each.
    const float s0 = 1.0f, s1 = 2.0f;
    const uint16_t s0_bits = ov::float16(s0).to_bits();
    const uint16_t s1_bits = ov::float16(s1).to_bits();

    std::vector<uint8_t> data(36, 0);
    std::memcpy(data.data(), &s0_bits, 2);
    std::fill(data.begin() + 2, data.begin() + 18, static_cast<uint8_t>(0x44));
    std::memcpy(data.data() + 18, &s1_bits, 2);
    std::fill(data.begin() + 20, data.end(), static_cast<uint8_t>(0x44));

    gguf_tensor tensor = make_tensor(data.data(), data.size(), 2, 32, GGUF_TYPE_Q4_0);
    ov::Tensor weights(ov::element::u32, ov::Shape{2, 4});
    ov::Tensor scales(ov::element::f16, ov::Shape{2, 1});
    gguf_fill_q4_0(tensor, weights, scales);

    std::unordered_map<std::string, ov::Tensor> w_map;
    std::unordered_map<std::string, gguf_tensor_type> q_map;
    w_map["r.weight"] = weights;
    w_map["r.scales"] = scales;
    q_map["r.qtype"] = GGUF_TYPE_Q4_0;
    auto out = eval_as_f32(make_weight_node("r", w_map, q_map));
    ASSERT_EQ(out.size(), 64u);
    // nibble 4 → i4 = 4-8 = -4
    for (int i = 0; i < 32; ++i)
        EXPECT_NEAR(out[i], -4.f * s0, 1e-2f);
    for (int i = 32; i < 64; ++i)
        EXPECT_NEAR(out[i], -4.f * s1, 1e-2f);
}
