// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Per-op conformance tests for the GGUF frontend.
//
// Test flow for each op:
//   1. Load pre-calculated input/expected .npy files from tests/test_data/.
//      (Files are generated once by generate_test_data.py and committed.)
//   2. Build a minimal single-op GgufGraph via OpTestGraph helpers.
//   3. Call TranslateSession to convert the graph to an ov::Model.
//   4. Run the model on the CPU plugin.
//   5. Compare outputs against the pre-calculated expected values.
//
// Reference data sources (see generate_test_data.py):
//   - Most ops: numpy formulas equivalent to what translate_* produces.
//   - The numpy formulas match llama.cpp test-backend-ops.cpp's CPU-baseline
//     semantics for the same ops (rms_norm, rope, softmax, glu variants, etc.).
//
// To add a new op test: see the "Adding a new op test" section in
// docs/developer_guide.md.

#include <gtest/gtest.h>

#include "builder/weights.hpp"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/op/constant.hpp"

using namespace ov_gguf_test;

// ── Tolerance constants ───────────────────────────────────────────────────────

static constexpr float kAtolF32 = 1e-5f;   // plain f32 elementwise ops
static constexpr float kAtolAct = 1e-5f;   // activations (gelu / silu)
static constexpr float kAtolRope = 2e-5f;  // RoPE (trigonometric)
static constexpr float kAtolSdpa = 1e-3f;  // flash-attn (f16 internal)

// ── Helpers ───────────────────────────────────────────────────────────────────

// Build a two-input elementwise op model (ADD, SUB, MUL, DIV, ADD1).
static std::shared_ptr<ov::Model> build_binary(const std::string& op_type, const ov::PartialShape& shape) {
    OpTestGraph g;
    g.add_input("a", ov::element::f32, shape);
    g.add_input("b", ov::element::f32, shape);
    g.add_op(op_type,
             {"a", "b"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"a", shape}, {"b", shape}},
             {{"a", ov::element::f32}, {"b", ov::element::f32}});
    return g.build();
}

// ── GGML_OP_RMS_NORM ─────────────────────────────────────────────────────────

TEST(GGUFOps, RmsNorm) {
    ov::PartialShape shape;
    auto input_data = load_f32("rms_norm_input", &shape);
    auto expected = load_f32("rms_norm_expected");
    float eps = 1e-5f;

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_RMS_NORM",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}},
             {{"eps", ov::Any(eps)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "rms_norm");
}

// ── GGML_OP_ADD ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Add) {
    ov::PartialShape shape;
    auto a = load_f32("add_input_a", &shape);
    auto b = load_f32("add_input_b");
    auto expected = load_f32("add_expected");

    ov::Shape s = shape.to_shape();
    auto result = run_on_cpu(build_binary("GGML_OP_ADD", shape), {make_f32_tensor(s, a), make_f32_tensor(s, b)});
    expect_near(result[0], expected, kAtolF32, "add");
}

// ── GGML_OP_ADD1 (same translator as ADD) ────────────────────────────────────

TEST(GGUFOps, Add1) {
    ov::PartialShape shape;
    auto a = load_f32("add_input_a", &shape);
    auto b = load_f32("add_input_b");
    auto expected = load_f32("add_expected");

    ov::Shape s = shape.to_shape();
    auto result = run_on_cpu(build_binary("GGML_OP_ADD1", shape), {make_f32_tensor(s, a), make_f32_tensor(s, b)});
    expect_near(result[0], expected, kAtolF32, "add1");
}

// ── GGML_OP_SUB ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Sub) {
    ov::PartialShape shape;
    auto a = load_f32("sub_input_a", &shape);
    auto b = load_f32("sub_input_b");
    auto expected = load_f32("sub_expected");

    ov::Shape s = shape.to_shape();
    auto result = run_on_cpu(build_binary("GGML_OP_SUB", shape), {make_f32_tensor(s, a), make_f32_tensor(s, b)});
    expect_near(result[0], expected, kAtolF32, "sub");
}

// ── GGML_OP_MUL ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Mul) {
    ov::PartialShape shape;
    auto a = load_f32("mul_input_a", &shape);
    auto b = load_f32("mul_input_b");
    auto expected = load_f32("mul_expected");

    ov::Shape s = shape.to_shape();
    auto result = run_on_cpu(build_binary("GGML_OP_MUL", shape), {make_f32_tensor(s, a), make_f32_tensor(s, b)});
    expect_near(result[0], expected, kAtolF32, "mul");
}

// ── GGML_OP_DIV ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Div) {
    ov::PartialShape shape;
    auto a = load_f32("div_input_a", &shape);
    auto b = load_f32("div_input_b");
    auto expected = load_f32("div_expected");

    ov::Shape s = shape.to_shape();
    auto result = run_on_cpu(build_binary("GGML_OP_DIV", shape), {make_f32_tensor(s, a), make_f32_tensor(s, b)});
    expect_near(result[0], expected, kAtolF32, "div");
}

// ── GGML_OP_SCALE ────────────────────────────────────────────────────────────

TEST(GGUFOps, Scale) {
    ov::PartialShape shape;
    auto input_data = load_f32("scale_input", &shape);
    auto expected = load_f32("scale_expected");
    float scale = load_scalar_f32("scale_param_scale");
    float bias = load_scalar_f32("scale_param_bias");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_SCALE",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}},
             {{"scale", ov::Any(scale)}, {"bias", ov::Any(bias)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "scale");
}

TEST(GGUFOps, ScaleNoBias) {
    ov::PartialShape shape;
    auto input_data = load_f32("scale_input", &shape);
    auto expected = load_f32("scale_nobias_expected");
    float scale = load_scalar_f32("scale_param_scale");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_SCALE",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}},
             {{"scale", ov::Any(scale)}, {"bias", ov::Any(0.0f)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "scale_nobias");
}

// ── GGML_OP_SOFT_MAX ─────────────────────────────────────────────────────────

TEST(GGUFOps, SoftMax) {
    ov::PartialShape shape;
    auto input_data = load_f32("softmax_input", &shape);
    auto expected = load_f32("softmax_expected");
    float scale = load_scalar_f32("softmax_param_scale");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_SOFT_MAX",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}},
             {{"scale", ov::Any(scale)}, {"softmax_axis", ov::Any(int64_t{3})}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "softmax");
}

// ── GGML_UNARY_OP_SILU ───────────────────────────────────────────────────────

TEST(GGUFOps, SiLU) {
    ov::PartialShape shape;
    auto input_data = load_f32("silu_input", &shape);
    auto expected = load_f32("silu_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_UNARY_OP_SILU", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolAct, "silu");
}

// ── GGML_UNARY_OP_GELU ───────────────────────────────────────────────────────

TEST(GGUFOps, GeLU) {
    ov::PartialShape shape;
    auto input_data = load_f32("gelu_input", &shape);
    auto expected = load_f32("gelu_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_UNARY_OP_GELU", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolAct, "gelu");
}

// ── GGML_OP_MUL_MAT ──────────────────────────────────────────────────────────

TEST(GGUFOps, MulMat) {
    // A[1,1,m,k]  B[1,1,n,k]  → out[1,1,m,n]  (translate_mulmat: MatMul(A,B,false,true))
    ov::PartialShape shape_a, shape_b;
    auto a_data = load_f32("mul_mat_input_a", &shape_a);
    auto b_data = load_f32("mul_mat_input_b", &shape_b);
    auto expected = load_f32("mul_mat_expected");

    auto sa = shape_a.to_shape();
    auto sb = shape_b.to_shape();
    ov::Shape out_s = {sa[0], sa[1], sa[2], sb[2]};
    ov::PartialShape out_ps(std::vector<ov::Dimension>(out_s.begin(), out_s.end()));

    OpTestGraph g;
    // translate_mulmat: input(0)=B (weight), input(1)=A (activation)
    g.add_input("B", ov::element::f32, shape_b);
    g.add_input("A", ov::element::f32, shape_a);
    g.add_op("GGML_OP_MUL_MAT",
             {"B", "A"},
             "y",
             out_ps,
             ov::element::f32,
             0,
             {{"B", shape_b}, {"A", shape_a}},
             {{"B", ov::element::f32}, {"A", ov::element::f32}});
    auto model = g.build();

    // Parameters in add_input() order: B first, A second.
    auto result = run_on_cpu(model, {make_f32_tensor(sb, b_data), make_f32_tensor(sa, a_data)});
    expect_near(result[0], expected, kAtolF32, "mul_mat");
}

// ── GGML_OP_GET_ROWS ─────────────────────────────────────────────────────────

TEST(GGUFOps, GetRows) {
    ov::PartialShape weight_shape;
    auto weight_data = load_f32("get_rows_weight", &weight_shape);
    std::vector<size_t> idx_sz;
    auto idx_data = load_npy<int32_t>("get_rows_indices", &idx_sz);
    auto expected = load_f32("get_rows_expected");

    ov::PartialShape idx_shape(std::vector<ov::Dimension>(idx_sz.begin(), idx_sz.end()));
    auto ws = weight_shape.to_shape();
    size_t seq = idx_sz[3];
    ov::PartialShape out_ps({1, 1, (int64_t)seq, (int64_t)ws[1]});

    OpTestGraph g;
    g.add_input("data", ov::element::f32, weight_shape);
    g.add_input("indices", ov::element::i32, idx_shape);
    g.add_op("GGML_OP_GET_ROWS",
             {"data", "indices"},
             "y",
             out_ps,
             ov::element::f32,
             0,
             {{"data", weight_shape}, {"indices", idx_shape}},
             {{"data", ov::element::f32}, {"indices", ov::element::i32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(ws, weight_data), make_i32_tensor(to_ov_shape(idx_sz), idx_data)});
    expect_near(result[0], expected, kAtolF32, "get_rows");
}

// ── GGML_OP_ROPE (NORMAL mode) ───────────────────────────────────────────────

TEST(GGUFOps, RopeNormal) {
    ov::PartialShape x_shape;
    auto x_data = load_f32("rope_normal_input", &x_shape);
    std::vector<size_t> pos_sz;
    auto pos_data = load_npy<int32_t>("rope_normal_positions", &pos_sz);
    auto expected = load_f32("rope_normal_expected");
    float freq_base = load_scalar_f32("rope_normal_param_freq_base");
    int32_t head_dim = load_scalar_i32("rope_normal_param_head_dim");

    ov::PartialShape pos_shape(std::vector<ov::Dimension>(pos_sz.begin(), pos_sz.end()));

    ov::frontend::gguf::RopeConfig rope_cfg;
    rope_cfg.n_dims = head_dim;
    rope_cfg.freq_base = freq_base;
    rope_cfg.freq_scale = 1.0f;
    rope_cfg.attn_factor = 1.0f;  // must be 1; default 0 zeros out all sin/cos

    OpTestGraph g;
    g.add_input("x", ov::element::f32, x_shape);
    g.add_input("pos", ov::element::i32, pos_shape);
    // op_case high 16 bits = mode (0=NORMAL), low 16 bits = variant (0=non-view)
    g.add_op("GGML_OP_ROPE",
             {"x", "pos"},
             "y",
             x_shape,
             ov::element::f32,
             0,
             {{"x", x_shape}, {"pos", pos_shape}},
             {{"x", ov::element::f32}, {"pos", ov::element::i32}},
             {{"rope_config", ov::Any(rope_cfg)}});
    auto model = g.build();

    ov::Shape xs = x_shape.to_shape();
    // Parameters in add_input() order: x (0), pos (1).
    auto result = run_on_cpu(model, {make_f32_tensor(xs, x_data), make_i32_tensor(to_ov_shape(pos_sz), pos_data)});
    expect_near(result[0], expected, kAtolRope, "rope_normal");
}

// ── GGML_OP_ROPE (NEOX mode) ─────────────────────────────────────────────────

TEST(GGUFOps, RopeNeox) {
    ov::PartialShape x_shape;
    auto x_data = load_f32("rope_neox_input", &x_shape);
    std::vector<size_t> pos_sz;
    auto pos_data = load_npy<int32_t>("rope_neox_positions", &pos_sz);
    auto expected = load_f32("rope_neox_expected");
    float freq_base = load_scalar_f32("rope_neox_param_freq_base");
    int32_t head_dim = load_scalar_i32("rope_neox_param_head_dim");

    ov::PartialShape pos_shape(std::vector<ov::Dimension>(pos_sz.begin(), pos_sz.end()));

    ov::frontend::gguf::RopeConfig rope_cfg;
    rope_cfg.n_dims = head_dim;
    rope_cfg.freq_base = freq_base;
    rope_cfg.freq_scale = 1.0f;
    rope_cfg.attn_factor = 1.0f;

    // NEOX mode: op_case = (TYPE_NEOX << 16) | 0  = 0x00010000
    constexpr int NEOX_OP_CASE = (1 << 16);

    OpTestGraph g;
    g.add_input("x", ov::element::f32, x_shape);
    g.add_input("pos", ov::element::i32, pos_shape);
    g.add_op("GGML_OP_ROPE",
             {"x", "pos"},
             "y",
             x_shape,
             ov::element::f32,
             NEOX_OP_CASE,
             {{"x", x_shape}, {"pos", pos_shape}},
             {{"x", ov::element::f32}, {"pos", ov::element::i32}},
             {{"rope_config", ov::Any(rope_cfg)}});
    auto model = g.build();

    ov::Shape xs = x_shape.to_shape();
    auto result = run_on_cpu(model, {make_f32_tensor(xs, x_data), make_i32_tensor(to_ov_shape(pos_sz), pos_data)});
    expect_near(result[0], expected, kAtolRope, "rope_neox");
}

// ── GGML_OP_TRANSPOSE ────────────────────────────────────────────────────────

TEST(GGUFOps, Transpose) {
    ov::PartialShape shape;
    auto input_data = load_f32("transpose_input", &shape);
    auto expected = load_f32("transpose_expected");

    // translate_transpose applies perm {0,1,3,2}; output shape has last two dims swapped.
    auto s = shape.to_shape();
    ov::PartialShape out_ps({(int64_t)s[0], (int64_t)s[1], (int64_t)s[3], (int64_t)s[2]});

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_TRANSPOSE", {"x"}, "y", out_ps, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(s, input_data)});
    expect_near(result[0], expected, kAtolF32, "transpose");
}

// ── GGML_OP_PERMUTE (op_case=1 → perm {0,2,1,3}) ────────────────────────────

TEST(GGUFOps, Permute) {
    ov::PartialShape shape;
    auto input_data = load_f32("permute_input", &shape);
    auto expected = load_f32("permute_expected");

    // op_case=1 → perm {0,2,1,3}; output shape has dims 1 and 2 swapped.
    auto s = shape.to_shape();
    ov::PartialShape out_ps({(int64_t)s[0], (int64_t)s[2], (int64_t)s[1], (int64_t)s[3]});

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_PERMUTE",
             {"x"},
             "y",
             out_ps,
             ov::element::f32,
             /*op_case=*/1,
             {{"x", shape}},
             {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(s, input_data)});
    expect_near(result[0], expected, kAtolF32, "permute");
}

// ── GGML_OP_CPY ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Cpy) {
    ov::PartialShape shape;
    auto input_data = load_f32("cpy_input", &shape);
    auto expected = load_f32("cpy_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_CPY", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "cpy");
}

// ── GGML_OP_CONT (op_case=2 → pass-through from TRANSPOSE) ──────────────────

TEST(GGUFOps, Cont) {
    ov::PartialShape shape;
    auto input_data = load_f32("cont_input", &shape);
    auto expected = load_f32("cont_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    // op_case=2 = "from TRANSPOSE" — translator returns input unchanged.
    g.add_op("GGML_OP_CONT",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             /*op_case=*/2,
             {{"x", shape}},
             {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "cont");
}

// ── GGML_OP_RESHAPE (op_case=6 → full output-shape reshape) ──────────────────

TEST(GGUFOps, Reshape) {
    ov::PartialShape in_shape;
    auto input_data = load_f32("reshape_input", &in_shape);
    auto expected = load_f32("reshape_expected");
    // In [2,3,4,5] → out [1,6,4,5]
    ov::PartialShape out_shape({1, 6, 4, 5});

    OpTestGraph g;
    g.add_input("x", ov::element::f32, in_shape);
    g.add_op("GGML_OP_RESHAPE",
             {"x"},
             "y",
             out_shape,
             ov::element::f32,
             /*op_case=*/6,
             {{"x", in_shape}},
             {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(in_shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "reshape");
}

// ── GGML_OP_VIEW (default op_case → pass-through) ────────────────────────────

TEST(GGUFOps, View) {
    ov::PartialShape shape;
    auto input_data = load_f32("view_input", &shape);
    auto expected = load_f32("view_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    // op_case=0 (default) → pass-through in translate_view
    g.add_op("GGML_OP_VIEW", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "view");
}

// ── GGML_GLU_OP_SWIGLU (1-input: split last axis, silu(gate)*up) ─────────────

TEST(GGUFOps, SwiGLU) {
    ov::PartialShape in_shape;
    auto input_data = load_f32("swiglu_input", &in_shape);
    auto expected = load_f32("swiglu_expected");

    // Output last dim = input last dim / 2
    auto is = in_shape.to_shape();
    ov::PartialShape out_shape({(int64_t)is[0], (int64_t)is[1], (int64_t)(is[2] / 2)});

    OpTestGraph g;
    g.add_input("x", ov::element::f32, in_shape);
    g.add_op("GGML_GLU_OP_SWIGLU",
             {"x"},
             "y",
             out_shape,
             ov::element::f32,
             0,
             {{"x", in_shape}},
             {{"x", ov::element::f32}},
             {{"swapped", ov::Any(false)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(is, input_data)});
    expect_near(result[0], expected, kAtolAct, "swiglu");
}

// ── GGML_GLU_OP_GEGLU (1-input: split last axis, gelu(gate)*up) ──────────────

TEST(GGUFOps, GeGLU) {
    ov::PartialShape in_shape;
    auto input_data = load_f32("geglu_input", &in_shape);
    auto expected = load_f32("geglu_expected");

    auto is = in_shape.to_shape();
    ov::PartialShape out_shape({(int64_t)is[0], (int64_t)is[1], (int64_t)(is[2] / 2)});

    OpTestGraph g;
    g.add_input("x", ov::element::f32, in_shape);
    g.add_op("GGML_GLU_OP_GEGLU",
             {"x"},
             "y",
             out_shape,
             ov::element::f32,
             0,
             {{"x", in_shape}},
             {{"x", ov::element::f32}},
             {{"swapped", ov::Any(false)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(is, input_data)});
    expect_near(result[0], expected, kAtolAct, "geglu");
}

// ── GGML_GLU_OP_SWIGLU_OAI ───────────────────────────────────────────────────

TEST(GGUFOps, SwiGLU_OAI) {
    ov::PartialShape in_shape;
    auto input_data = load_f32("swiglu_oai_input", &in_shape);
    auto expected = load_f32("swiglu_oai_expected");
    float alpha = load_scalar_f32("swiglu_oai_param_alpha");
    float limit = load_scalar_f32("swiglu_oai_param_limit");

    auto is = in_shape.to_shape();
    ov::PartialShape out_shape({(int64_t)is[0], (int64_t)is[1], (int64_t)(is[2] / 2)});

    OpTestGraph g;
    g.add_input("x", ov::element::f32, in_shape);
    g.add_op("GGML_GLU_OP_SWIGLU_OAI",
             {"x"},
             "y",
             out_shape,
             ov::element::f32,
             0,
             {{"x", in_shape}},
             {{"x", ov::element::f32}},
             {{"swapped", ov::Any(false)}, {"alpha", ov::Any(alpha)}, {"limit", ov::Any(limit)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(is, input_data)});
    expect_near(result[0], expected, kAtolAct, "swiglu_oai");
}

// ── GGML_OP_ARGSORT ──────────────────────────────────────────────────────────

TEST(GGUFOps, Argsort) {
    ov::PartialShape shape;
    auto input_data = load_f32("argsort_input", &shape);
    std::vector<size_t> exp_sz;
    auto expected = load_npy<int32_t>("argsort_expected", &exp_sz);

    // Output shape = input shape, dtype i32
    ov::PartialShape out_ps(std::vector<ov::Dimension>(exp_sz.begin(), exp_sz.end()));

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_ARGSORT", {"x"}, "y", out_ps, ov::element::i32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});

    ASSERT_EQ(ov::shape_size(result[0].get_shape()), expected.size());
    const int32_t* ptr = result[0].data<int32_t>();
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(ptr[i], expected[i]) << "argsort at element " << i;
}

// ── GGML_OP_TOP_K ─────────────────────────────────────────────────────────────

TEST(GGUFOps, TopK) {
    ov::PartialShape shape;
    auto input_data = load_f32("top_k_input", &shape);
    std::vector<size_t> exp_sz;
    auto expected = load_npy<int32_t>("top_k_expected", &exp_sz);
    int32_t k = load_scalar_i32("top_k_param_k");

    auto is = shape.to_shape();
    ov::PartialShape out_ps({(int64_t)is[0], (int64_t)is[1], (int64_t)k});

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_TOP_K", {"x"}, "y", out_ps, ov::element::i32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(is, input_data)});

    ASSERT_EQ(ov::shape_size(result[0].get_shape()), expected.size());
    // Top-k: indices are in descending value order — same indices must appear, same order.
    const int32_t* ptr = result[0].data<int32_t>();
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(ptr[i], expected[i]) << "top_k at element " << i;
}

// ── GGML_OP_SUM_ROWS ─────────────────────────────────────────────────────────

TEST(GGUFOps, SumRows) {
    ov::PartialShape shape;
    auto input_data = load_f32("sum_rows_input", &shape);
    auto expected = load_f32("sum_rows_expected");

    // Output: same shape with last dim = 1
    auto s = shape.to_shape();
    ov::PartialShape out_ps({(int64_t)s[0], (int64_t)s[1], (int64_t)s[2], 1});

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_SUM_ROWS", {"x"}, "y", out_ps, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(s, input_data)});
    expect_near(result[0], expected, kAtolF32, "sum_rows");
}

// ── GGML_OP_SET_ROWS (non-stateful: ScatterUpdate) ───────────────────────────

TEST(GGUFOps, SetRows) {
    ov::PartialShape dst_shape, data_shape;
    auto dst_data = load_f32("set_rows_dst", &dst_shape);
    auto data_data = load_f32("set_rows_data", &data_shape);
    std::vector<size_t> idx_sz;
    auto idx_data = load_npy<int32_t>("set_rows_indices", &idx_sz);
    auto expected = load_f32("set_rows_expected");

    ov::PartialShape idx_shape(std::vector<ov::Dimension>(idx_sz.begin(), idx_sz.end()));

    OpTestGraph g;
    g.add_input("data", ov::element::f32, data_shape);
    g.add_input("indices", ov::element::i32, idx_shape);
    g.add_input("dst", ov::element::f32, dst_shape);
    g.add_op("GGML_OP_SET_ROWS",
             {"data", "indices", "dst"},
             "y",
             dst_shape,
             ov::element::f32,
             0,
             {{"data", data_shape}, {"indices", idx_shape}, {"dst", dst_shape}},
             {{"data", ov::element::f32}, {"indices", ov::element::i32}, {"dst", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model,
                             {make_f32_tensor(data_shape.to_shape(), data_data),
                              make_i32_tensor(to_ov_shape(idx_sz), idx_data),
                              make_f32_tensor(dst_shape.to_shape(), dst_data)});
    expect_near(result[0], expected, kAtolF32, "set_rows");
}

// ── GGML_OP_FLASH_ATTN_EXT ───────────────────────────────────────────────────

TEST(GGUFOps, FlashAttnExt) {
    // In the real model K/V come from the f16 KV cache (after SET_ROWS).
    // The translator converts Q→f16 internally, so K and V must also be f16
    // to avoid mixed-type inputs to SDPA.
    // Layout: [B, seq, heads, head_dim] (GGUF-natural).  mask: [B, 1, seq_q, seq_k].
    ov::PartialShape q_shape, k_shape, v_shape, mask_shape;
    auto q_data = load_f32("flash_attn_input_q", &q_shape);
    auto k_data = load_f32("flash_attn_input_k", &k_shape);
    auto v_data = load_f32("flash_attn_input_v", &v_shape);
    auto mask_data = load_f32("flash_attn_input_mask", &mask_shape);
    auto expected = load_f32("flash_attn_expected");
    float scale = load_scalar_f32("flash_attn_param_scale");

    // Name the mask "KQ_mask_sliced" so the translator detects it in the tensor_map
    // and uses it directly, bypassing the fallback Slice that uses q.shape[2]
    // (which equals n_heads in [B,L,H,S] layout, not seq_q).
    OpTestGraph g;
    g.add_input("q", ov::element::f32, q_shape);
    g.add_input("k", ov::element::f16, k_shape);
    g.add_input("KQ_mask_sliced", ov::element::f32, mask_shape);
    g.add_input("v", ov::element::f16, v_shape);
    g.add_op("GGML_OP_FLASH_ATTN_EXT",
             {"q", "k", "v", "KQ_mask_sliced"},
             "y",
             q_shape,
             ov::element::f32,
             0,
             {{"q", q_shape}, {"k", k_shape}, {"v", v_shape}, {"KQ_mask_sliced", mask_shape}},
             {{"q", ov::element::f32},
              {"k", ov::element::f16},
              {"v", ov::element::f16},
              {"KQ_mask_sliced", ov::element::f32}},
             {{"scale", ov::Any(scale)}});
    auto model = g.build();

    // Parameters in add_input() order: q(f32), k(f16), KQ_mask_sliced(f32), v(f16).
    auto qs = q_shape.to_shape();
    auto ks = k_shape.to_shape();
    auto vs = v_shape.to_shape();
    auto ms = mask_shape.to_shape();
    auto result = run_on_cpu(model,
                             {make_f32_tensor(qs, q_data),
                              make_f16_tensor(ks, k_data),
                              make_f32_tensor(ms, mask_data),
                              make_f16_tensor(vs, v_data)});
    expect_near(result[0], expected, kAtolSdpa, "flash_attn_ext");
}

// ── GGML_OP_ADD_ID ────────────────────────────────────────────────────────────

TEST(GGUFOps, AddId) {
    // a [1,T,K,n]  b [n_expert,n]  ids [1,T,K]  → out [1,T,K,n]
    ov::PartialShape a_shape, b_shape;
    auto a_data = load_f32("add_id_input_a", &a_shape);
    auto b_data = load_f32("add_id_input_b", &b_shape);
    std::vector<size_t> ids_sz;
    auto ids_data = load_npy<int32_t>("add_id_input_ids", &ids_sz);
    auto expected = load_f32("add_id_expected");

    ov::PartialShape ids_shape(std::vector<ov::Dimension>(ids_sz.begin(), ids_sz.end()));

    OpTestGraph g;
    g.add_input("a", ov::element::f32, a_shape);
    g.add_input("b", ov::element::f32, b_shape);
    g.add_input("ids", ov::element::i32, ids_shape);
    g.add_op("GGML_OP_ADD_ID",
             {"a", "b", "ids"},
             "y",
             a_shape,
             ov::element::f32,
             0,
             {{"a", a_shape}, {"b", b_shape}, {"ids", ids_shape}},
             {{"a", ov::element::f32}, {"b", ov::element::f32}, {"ids", ov::element::i32}});
    auto model = g.build();

    auto result = run_on_cpu(model,
                             {make_f32_tensor(a_shape.to_shape(), a_data),
                              make_f32_tensor(b_shape.to_shape(), b_data),
                              make_i32_tensor(to_ov_shape(ids_sz), ids_data)});
    expect_near(result[0], expected, kAtolF32, "add_id");
}

// ── GGML_OP_MUL_MAT with MXFP4 weight ────────────────────────────────────────
//
// Tests the make_mxfp4() dequantization subgraph (f4e2m1 * f8e8m0) that is built by
// make_weight_node when qtype == GGUF_TYPE_MXFP4.
//
// Weight layout [2 rows × 32 cols, group_size=32]:
//   Row 0: all f4e2m1 nibbles = 2 (OV LUT value = 1.0), scale e=128 → 2.0  → dequant = 2.0
//   Row 1: nibbles alternating 1 (0.5) / 3 (1.5), scale e=127 → 1.0        → dequant as-is
// Activation: all-ones vector [1, 1, 1, 32].
//   result[0] = 32 × 2.0 = 64.0
//   result[1] = 16 × 0.5 + 16 × 1.5 = 32.0
TEST(GGUFOps, MulMatMxfp4) {
    using namespace ov::frontend::gguf;

    // f4e2m1 nibble LUT (OV):  idx→value: 0→0, 1→0.5, 2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
    // (negative counterparts for nibbles 8-15).  group_size = 32 elements per group.
    // Packed storage: two nibbles per byte, lower nibble = element[2*j], upper = element[2*j+1].

    const int rows = 2, cols = 32, groups = cols / 32;

    // f4e2m1 weight tensor: [rows, cols] packed as [rows, cols/2] bytes.
    ov::Tensor w_tensor(ov::element::f4e2m1, ov::Shape{(size_t)rows, (size_t)cols});
    auto* wb = static_cast<uint8_t*>(w_tensor.data());
    // Row 0: all nibbles = 2.  Two nibbles per byte → 0x22.
    std::fill(wb, wb + cols / 2, uint8_t(0x22));
    // Row 1: alternating nibbles 1 (lo) and 3 (hi) → 0x31 per byte.
    std::fill(wb + cols / 2, wb + rows * cols / 2, uint8_t(0x31));

    // f8e8m0 scale tensor: [rows, groups=1], exponent-only bytes.
    // Row 0: e=128 → scale = 2^(128-127) = 2.0.  Row 1: e=127 → 1.0.
    ov::Tensor s_tensor(ov::element::f8e8m0, ov::Shape{(size_t)rows, (size_t)groups});
    auto* sb = static_cast<uint8_t*>(s_tensor.data());
    sb[0] = 128;  // row 0 scale = 2.0
    sb[1] = 127;  // row 1 scale = 1.0

    // Build the MXFP4 weight node via make_weight_node (exercises make_mxfp4 path).
    std::unordered_map<std::string, ov::Tensor> fake_weights;
    fake_weights["W.weight"] = w_tensor;
    fake_weights["W.scales"] = s_tensor;
    std::unordered_map<std::string, gguf_tensor_type> fake_qtypes;
    fake_qtypes["W.qtype"] = GGUF_TYPE_MXFP4;
    auto w_node = make_weight_node("W", fake_weights, fake_qtypes);

    // Activation: [1,1,1,32] all-ones.
    ov::PartialShape act_shape({1, 1, 1, cols});
    ov::PartialShape w_shape({1, 1, rows, cols});
    ov::PartialShape out_shape({1, 1, 1, rows});

    OpTestGraph g;
    g.add_input("A", ov::element::f32, act_shape);
    g.add_weight_node("W.weight", w_node);
    g.add_op("GGML_OP_MUL_MAT",
             {"W.weight", "A"},
             "y",
             out_shape,
             ov::element::f32,
             0,
             {{"W.weight", w_shape}, {"A", act_shape}},
             {{"W.weight", ov::element::f32}, {"A", ov::element::f32}});
    auto model = g.build();

    ov::Shape act_s = act_shape.to_shape();
    std::vector<float> act_data(ov::shape_size(act_s), 1.0f);
    auto result = run_on_cpu(model, {make_f32_tensor(act_s, act_data)});

    // Expected: [64.0, 32.0] (see comment above).
    std::vector<float> expected = {64.0f, 32.0f};
    expect_near(result[0], expected, kAtolF32, "mul_mat_mxfp4");
}

// ── GGML_UNARY_OP_RELU ───────────────────────────────────────────────────────

TEST(GGUFOps, ReLU) {
    ov::PartialShape shape;
    auto input_data = load_f32("relu_input", &shape);
    auto expected = load_f32("relu_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_UNARY_OP_RELU", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "relu");
}

// ── GGML_UNARY_OP_TANH ───────────────────────────────────────────────────────

TEST(GGUFOps, Tanh) {
    ov::PartialShape shape;
    auto input_data = load_f32("tanh_input", &shape);
    auto expected = load_f32("tanh_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_UNARY_OP_TANH", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolAct, "tanh");
}

// ── GGML_UNARY_OP_SIGMOID ────────────────────────────────────────────────────

TEST(GGUFOps, Sigmoid) {
    ov::PartialShape shape;
    auto input_data = load_f32("sigmoid_input", &shape);
    auto expected = load_f32("sigmoid_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_UNARY_OP_SIGMOID",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolAct, "sigmoid");
}

// ── GGML_UNARY_OP_ELU ────────────────────────────────────────────────────────

TEST(GGUFOps, ELU) {
    ov::PartialShape shape;
    auto input_data = load_f32("elu_input", &shape);
    auto expected = load_f32("elu_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_UNARY_OP_ELU",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}},
             {{"alpha", ov::Any(1.0f)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolAct, "elu");
}

// ── GGML_OP_CLAMP ────────────────────────────────────────────────────────────

TEST(GGUFOps, Clamp) {
    ov::PartialShape shape;
    auto input_data = load_f32("clamp_input", &shape);
    auto expected = load_f32("clamp_expected");
    float lo = load_f32("clamp_param_min")[0];
    float hi = load_f32("clamp_param_max")[0];

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_CLAMP",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}},
             {{"min", ov::Any(lo)}, {"max", ov::Any(hi)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "clamp");
}

// ── GGML_OP_CONCAT ───────────────────────────────────────────────────────────

TEST(GGUFOps, Concat) {
    // Concatenate two [2,3,4,8] and [2,5,4,8] tensors along axis 1.
    ov::PartialShape shape_a, shape_b;
    auto a_data = load_f32("concat_input_a", &shape_a);
    auto b_data = load_f32("concat_input_b", &shape_b);
    auto expected = load_f32("concat_expected");

    // output shape: [2, 8, 4, 8]
    ov::PartialShape out_shape({2, 8, 4, 8});

    OpTestGraph g;
    g.add_input("a", ov::element::f32, shape_a);
    g.add_input("b", ov::element::f32, shape_b);
    g.add_op("GGML_OP_CONCAT",
             {"a", "b"},
             "y",
             out_shape,
             ov::element::f32,
             0,  // op_case 0 → axis 1
             {{"a", shape_a}, {"b", shape_b}},
             {{"a", ov::element::f32}, {"b", ov::element::f32}});
    auto model = g.build();

    auto result =
        run_on_cpu(model, {make_f32_tensor(shape_a.to_shape(), a_data), make_f32_tensor(shape_b.to_shape(), b_data)});
    expect_near(result[0], expected, kAtolF32, "concat");
}

// ── GGML_OP_NORM ─────────────────────────────────────────────────────────────

TEST(GGUFOps, Norm) {
    ov::PartialShape shape;
    auto input_data = load_f32("norm_input", &shape);
    auto expected = load_f32("norm_expected");
    float eps = 1e-5f;

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_NORM",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}},
             {{"eps", ov::Any(eps)}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "norm");
}

// ── GGML_OP_SQR ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Sqr) {
    ov::PartialShape shape;
    auto input_data = load_f32("sqr_input", &shape);
    auto expected = load_f32("sqr_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_SQR", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "sqr");
}

// ── GGML_OP_SQRT ─────────────────────────────────────────────────────────────

TEST(GGUFOps, Sqrt) {
    ov::PartialShape shape;
    auto input_data = load_f32("sqrt_input", &shape);
    auto expected = load_f32("sqrt_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_SQRT", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "sqrt");
}

// ── GGML_OP_LOG ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Log) {
    ov::PartialShape shape;
    auto input_data = load_f32("log_input", &shape);
    auto expected = load_f32("log_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_LOG", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "log");
}

// ── GGML_OP_SIN ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Sin) {
    ov::PartialShape shape;
    auto input_data = load_f32("sin_input", &shape);
    auto expected = load_f32("sin_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_SIN", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "sin");
}

// ── GGML_OP_COS ──────────────────────────────────────────────────────────────

TEST(GGUFOps, Cos) {
    ov::PartialShape shape;
    auto input_data = load_f32("cos_input", &shape);
    auto expected = load_f32("cos_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_OP_COS", {"x"}, "y", shape, ov::element::f32, 0, {{"x", shape}}, {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "cos");
}

// ── GGML_UNARY_OP_GELU_QUICK ─────────────────────────────────────────────────

TEST(GGUFOps, GeLUQuick) {
    ov::PartialShape shape;
    auto input_data = load_f32("gelu_quick_input", &shape);
    auto expected = load_f32("gelu_quick_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, shape);
    g.add_op("GGML_UNARY_OP_GELU_QUICK",
             {"x"},
             "y",
             shape,
             ov::element::f32,
             0,
             {{"x", shape}},
             {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolAct, "gelu_quick");
}

// ── GGML_OP_REPEAT ───────────────────────────────────────────────────────────

TEST(GGUFOps, Repeat) {
    // Tile [2,1,4,8] → [2,3,4,8] (repeat dim 1 ×3).
    ov::PartialShape in_shape({2, 1, 4, 8});
    ov::PartialShape out_shape({2, 3, 4, 8});
    auto input_data = load_f32("repeat_input");
    auto expected = load_f32("repeat_expected");

    OpTestGraph g;
    g.add_input("x", ov::element::f32, in_shape);
    g.add_op("GGML_OP_REPEAT",
             {"x"},
             "y",
             out_shape,
             ov::element::f32,
             0,
             {{"x", in_shape}},
             {{"x", ov::element::f32}});
    auto model = g.build();

    auto result = run_on_cpu(model, {make_f32_tensor(in_shape.to_shape(), input_data)});
    expect_near(result[0], expected, kAtolF32, "repeat");
}
