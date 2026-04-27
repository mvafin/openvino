// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end tests for the weight-conversion paths in the GGUF frontend.
//
// For each ggml quant type the frontend understands we:
//   1. Synthesize a deterministic raw block buffer (well-formed bytes; fp16
//      scale fields clamped to a finite, sane range).
//   2. Run the reference fp16 dequantizer (`materialize_tensor_f16_or_native`).
//   3. Build the native decompression subgraph (`build_weight_node`), compile
//      it on CPU, and infer it as a no-input model.
//   4. Compare the inferred f32 output element-wise against the f16 reference
//      promoted to f32. The two paths algebraically compute the same value;
//      they only differ in fp16 rounding order, so we expect agreement to a
//      few ULPs of f16.
//
// We additionally assert that the dispatcher actually picked the native path
// by checking the element type of the lowest-precision Constant in the graph.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gguf_compress.hpp"
#include "gguf_dequant.hpp"
#include "gguf_reader.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

using ov::frontend::gguf::TensorDescriptor;
using ov::frontend::gguf::ggml_type;
using ov::frontend::gguf::build_weight_node;
using ov::frontend::gguf::materialize_tensor_f16_or_native;

// ---- block geometry (mirrors gguf_dequant.cpp / gguf_compress.cpp) ----- //
struct BlockInfo {
    int    qk;     // elements per block
    size_t bytes;  // bytes per block
};

BlockInfo block_info(ggml_type t) {
    switch (t) {
        case ov::frontend::gguf::GGML_TYPE_Q4_0: return {32, 18};
        case ov::frontend::gguf::GGML_TYPE_Q4_1: return {32, 20};
        case ov::frontend::gguf::GGML_TYPE_Q5_0: return {32, 22};
        case ov::frontend::gguf::GGML_TYPE_Q5_1: return {32, 24};
        case ov::frontend::gguf::GGML_TYPE_Q8_0: return {32, 34};
        case ov::frontend::gguf::GGML_TYPE_Q2_K: return {256, 84};
        case ov::frontend::gguf::GGML_TYPE_Q3_K: return {256, 110};
        case ov::frontend::gguf::GGML_TYPE_Q4_K: return {256, 144};
        case ov::frontend::gguf::GGML_TYPE_Q5_K: return {256, 176};
        case ov::frontend::gguf::GGML_TYPE_Q6_K: return {256, 210};
        default: break;
    }
    ADD_FAILURE() << "unknown ggml_type " << static_cast<uint32_t>(t);
    return {1, 1};
}

// ---- synthetic block generator ----------------------------------------- //
//
// We need raw bytes that are *valid* (no NaN/Inf in fp16 scale fields) but
// otherwise exercise random codes. Strategy:
//   * fp16 fields: random fp32 in [-1, 1], then ov::float16 (canonicalizes).
//   * everything else (codes, packed scales, etc.): random uint8 — any byte
//     pattern is a valid encoding for these formats.
//
// The fp16 fields' offsets within each block format:
struct ScaleField { size_t offset; bool present; };
struct BlockScales { ScaleField d, dmin; };

BlockScales scale_fields(ggml_type t) {
    using namespace ov::frontend::gguf;
    switch (t) {
        case GGML_TYPE_Q4_0: return { {0,  true}, {0, false} };
        case GGML_TYPE_Q4_1: return { {0,  true}, {2, true } };
        case GGML_TYPE_Q5_0: return { {0,  true}, {0, false} };
        case GGML_TYPE_Q5_1: return { {0,  true}, {2, true } };
        case GGML_TYPE_Q8_0: return { {0,  true}, {0, false} };
        // K-quants:
        case GGML_TYPE_Q2_K: return { {80, true}, {82, true} };  // d, dmin at end
        case GGML_TYPE_Q3_K: return { {108, true}, {0, false} }; // d only
        case GGML_TYPE_Q4_K: return { {0,  true}, {2, true } };
        case GGML_TYPE_Q5_K: return { {0,  true}, {2, true } };
        case GGML_TYPE_Q6_K: return { {208, true}, {0, false} };
        default: break;
    }
    return { {0, false}, {0, false} };
}

void synthesize_blocks(ggml_type t, size_t num_blocks,
                       std::vector<uint8_t>& out, std::mt19937& rng) {
    const auto bi = block_info(t);
    out.resize(num_blocks * bi.bytes);

    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(-1.0f, 1.0f);

    for (auto& b : out) b = static_cast<uint8_t>(byte_dist(rng));

    const auto sf = scale_fields(t);
    auto write_fp16 = [&](uint8_t* p, float v) {
        ov::float16 h(v);
        uint16_t bits = h.to_bits();
        std::memcpy(p, &bits, sizeof(bits));
    };
    for (size_t i = 0; i < num_blocks; ++i) {
        uint8_t* blk = out.data() + i * bi.bytes;
        if (sf.d.present)    write_fp16(blk + sf.d.offset,    scale_dist(rng));
        if (sf.dmin.present) write_fp16(blk + sf.dmin.offset, scale_dist(rng));
    }
}

// ---- comparison helpers ------------------------------------------------ //

void compare_f32_to_f16(const float* candidate, const ov::float16* reference,
                        size_t n, float abs_tol, float rel_tol,
                        const char* label) {
    size_t mismatches = 0;
    float worst_abs = 0.0f, worst_rel = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float r = static_cast<float>(reference[i]);
        const float c = candidate[i];
        const float a = std::abs(c - r);
        const float rl = a / std::max(std::abs(r), 1e-6f);
        if (a > worst_abs) worst_abs = a;
        if (rl > worst_rel) worst_rel = rl;
        if (a > abs_tol && rl > rel_tol) {
            if (++mismatches <= 5) {
                ADD_FAILURE() << label << ": element " << i << " ref=" << r
                              << " got=" << c << " abs=" << a << " rel=" << rl;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << label << ": " << mismatches << "/" << n
                              << " elements outside tolerance "
                              << "(worst abs=" << worst_abs
                              << ", worst rel=" << worst_rel << ")";
}

// Find a Constant whose element type is a sub-byte / int8 / uint8 type.
// Returns an empty string if none is found.
std::string lowest_precision_const_type(const std::shared_ptr<ov::Model>& m) {
    static const std::vector<std::string> compressed = {
        "u1", "u2", "u3", "u4", "u6", "u8", "i4", "i8",
    };
    for (const auto& op : m->get_ops()) {
        if (op->get_type_name() != std::string("Constant")) continue;
        const auto et = op->get_element_type().get_type_name();
        for (const auto& c : compressed) {
            if (et == c) return et;
        }
    }
    return {};
}

// Run the candidate path: build the weight subgraph, wrap in a model, compile
// on CPU, infer with no inputs, return f32 contents and the lowest-precision
// Constant element type observed in the graph.
struct CandidateResult {
    std::vector<float> values;
    std::string        lowest_const_et;
};

CandidateResult run_candidate(const TensorDescriptor& td, const uint8_t* raw) {
    auto out = build_weight_node(td, raw, "w");
    auto result = std::make_shared<ov::op::v0::Result>(out);
    auto model  = std::make_shared<ov::Model>(
        ov::ResultVector{result}, ov::ParameterVector{}, "weight_only");

    CandidateResult r;
    r.lowest_const_et = lowest_precision_const_type(model);

    ov::Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    req.infer();
    auto t = req.get_output_tensor(0);
    EXPECT_EQ(t.get_element_type(), ov::element::f32);
    const size_t n = t.get_size();
    r.values.assign(t.data<const float>(), t.data<const float>() + n);
    return r;
}

// ---- parameterized test fixture --------------------------------------- //

struct WeightConvCase {
    const char*           label;
    ggml_type             type;
    size_t                out_dim;
    size_t                in_dim;
    const char*           expected_const_et;  // "" = no native path expected
    float                 abs_tol;
    float                 rel_tol;
};

class WeightConversionTests : public ::testing::TestWithParam<WeightConvCase> {};

TEST_P(WeightConversionTests, MatchesReferenceDequantizer) {
    const auto p = GetParam();
    const auto bi = block_info(p.type);
    ASSERT_EQ(p.in_dim % bi.qk, 0u) << "in_dim must be multiple of block size";

    const size_t n_elems  = p.out_dim * p.in_dim;
    const size_t n_blocks = n_elems / bi.qk;

    std::mt19937 rng(0xC0FFEEu + static_cast<uint32_t>(p.type));
    std::vector<uint8_t> raw;
    synthesize_blocks(p.type, n_blocks, raw, rng);

    TensorDescriptor td;
    td.name   = "w";
    td.type   = p.type;
    td.dims   = {static_cast<uint64_t>(p.out_dim), static_cast<uint64_t>(p.in_dim)};
    td.offset = 0;

    // Reference: fp16 dequantizer.
    ov::Tensor ref_tensor = materialize_tensor_f16_or_native(td, raw.data());
    ASSERT_EQ(ref_tensor.get_element_type(), ov::element::f16);
    ASSERT_EQ(ref_tensor.get_size(), n_elems);

    // Candidate: native decompression chain + CPU inference.
    auto cand = run_candidate(td, raw.data());
    ASSERT_EQ(cand.values.size(), n_elems);

    // Sanity: dispatcher selected the native path -> a sub-byte/i8 Constant
    // is present.
    if (p.expected_const_et[0] != '\0') {
        EXPECT_EQ(cand.lowest_const_et, std::string(p.expected_const_et))
            << "Expected native path with " << p.expected_const_et
            << " Constant, got: '" << cand.lowest_const_et << "'";
    }

    compare_f32_to_f16(cand.values.data(),
                       ref_tensor.data<const ov::float16>(),
                       n_elems, p.abs_tol, p.rel_tol, p.label);
}

// Tolerances:
//   abs_tol covers values close to zero (where relative is meaningless).
//   rel_tol allows ~3 ULPs of fp16 (~ 3 * 2^-10 ~= 3e-3) for the difference
//   between (d * sc) -> fp16 -> * q vs d -> * sc -> * q in fp32 -> fp16.
//
// Empty `expected_const_et` means we don't check the dispatched path (used as
// a future hook; today every entry exercises a native path).
INSTANTIATE_TEST_SUITE_P(
    NativeQuants,
    WeightConversionTests,
    ::testing::Values(
        WeightConvCase{"Q4_0",  ov::frontend::gguf::GGML_TYPE_Q4_0, 4, 64,  "u4", 1e-4f, 5e-3f},
        WeightConvCase{"Q4_1",  ov::frontend::gguf::GGML_TYPE_Q4_1, 4, 64,  "u4", 1e-4f, 5e-3f},
        WeightConvCase{"Q5_0",  ov::frontend::gguf::GGML_TYPE_Q5_0, 4, 64,  "i8", 1e-4f, 5e-3f},
        WeightConvCase{"Q5_1",  ov::frontend::gguf::GGML_TYPE_Q5_1, 4, 64,  "i8", 1e-4f, 5e-3f},
        WeightConvCase{"Q8_0",  ov::frontend::gguf::GGML_TYPE_Q8_0, 4, 64,  "i8", 1e-4f, 5e-3f},
        WeightConvCase{"Q2_K",  ov::frontend::gguf::GGML_TYPE_Q2_K, 2, 256, "u2", 1e-4f, 5e-3f},
        WeightConvCase{"Q3_K",  ov::frontend::gguf::GGML_TYPE_Q3_K, 2, 256, "u3", 1e-4f, 5e-3f},
        WeightConvCase{"Q4_K",  ov::frontend::gguf::GGML_TYPE_Q4_K, 2, 256, "u4", 1e-4f, 5e-3f},
        WeightConvCase{"Q5_K",  ov::frontend::gguf::GGML_TYPE_Q5_K, 2, 256, "i8", 1e-4f, 5e-3f},
        WeightConvCase{"Q6_K",  ov::frontend::gguf::GGML_TYPE_Q6_K, 2, 256, "u6", 1e-4f, 5e-3f}
    ),
    [](const ::testing::TestParamInfo<WeightConvCase>& info) {
        return std::string(info.param.label);
    });

}  // namespace
