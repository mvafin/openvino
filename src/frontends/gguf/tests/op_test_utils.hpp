// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Utilities shared by the GGUF frontend per-op test suite (test_ops.cpp).
//
// Provides:
//   - OpTestGraph  — a thin builder that wraps GgufGraph/GgufBuilderDecoder/
//                   TranslateSession so a test can describe one op in a few
//                   lines and get an ov::Model back without touching file I/O.
//   - load_npy<T>  — load a flat std::vector<T> plus shape from a .npy file
//                   using the bundled cnpy library.
//   - run_on_cpu   — compile an ov::Model on CPU and run one inference.
//   - expect_near  — element-wise |actual-expected| <= atol check via GTest.
//   - TEST_DATA_DIR — compile-time path injected by CMake.

#pragma once

#include <cnpy.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "builder/gguf_builder_decoder.hpp"
#include "builder/gguf_graph.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"
#include "translate_session.hpp"

// TEST_DATA_DIR is injected by CMakeLists.txt as an absolute path string.
#ifndef TEST_DATA_DIR
#    error "TEST_DATA_DIR must be defined by CMake (add_compile_definitions)"
#endif

namespace ov_gguf_test {

using namespace ov::frontend::gguf;

// ── npy helpers ───────────────────────────────────────────────────────────────

inline std::string test_data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename + ".npy";
}

// Load a .npy array and return its data as a flat vector.
// The shape is written into *shape_out if the pointer is non-null.
template <typename T>
std::vector<T> load_npy(const std::string& stem, std::vector<size_t>* shape_out = nullptr) {
    cnpy::NpyArray arr = cnpy::npy_load(test_data_path(stem));
    if (shape_out)
        *shape_out = arr.shape;
    const T* begin = arr.data<T>();
    return std::vector<T>(begin, begin + arr.num_vals);
}

// Convenience: load an f32 npy, also return the OV PartialShape.
inline std::vector<float> load_f32(const std::string& stem, ov::PartialShape* ps_out = nullptr) {
    std::vector<size_t> sh;
    auto data = load_npy<float>(stem, &sh);
    if (ps_out) {
        std::vector<ov::Dimension> dims(sh.begin(), sh.end());
        *ps_out = ov::PartialShape(dims);
    }
    return data;
}

// Load a scalar f32 npy (0-d or 1-element).
inline float load_scalar_f32(const std::string& stem) {
    return load_npy<float>(stem)[0];
}

inline int32_t load_scalar_i32(const std::string& stem) {
    return load_npy<int32_t>(stem)[0];
}

// ── OV tensor helpers ─────────────────────────────────────────────────────────

inline ov::Tensor make_f32_tensor(const ov::Shape& shape, const std::vector<float>& data) {
    ov::Tensor t(ov::element::f32, shape);
    std::copy(data.begin(), data.end(), t.data<float>());
    return t;
}

inline ov::Tensor make_i32_tensor(const ov::Shape& shape, const std::vector<int32_t>& data) {
    ov::Tensor t(ov::element::i32, shape);
    std::copy(data.begin(), data.end(), t.data<int32_t>());
    return t;
}

// Convert float32 data to an f16 tensor (translator expects KV-cache inputs in f16).
inline ov::Tensor make_f16_tensor(const ov::Shape& shape, const std::vector<float>& data) {
    ov::Tensor t(ov::element::f16, shape);
    ov::float16* dst = t.data<ov::float16>();
    for (size_t i = 0; i < data.size(); ++i)
        dst[i] = ov::float16(data[i]);
    return t;
}

// Convert a std::vector<size_t> shape to ov::Shape.
inline ov::Shape to_ov_shape(const std::vector<size_t>& s) {
    return ov::Shape(s.begin(), s.end());
}

// ── Graph builder ─────────────────────────────────────────────────────────────

// Builds a single-op GgufGraph for use in per-op tests.
//
// Usage:
//   OpTestGraph g;
//   g.add_input("x", ov::element::f32, {2, 8});
//   g.add_input("y", ov::element::f32, {2, 8});
//   g.add_op("GGML_OP_ADD", {"x", "y"}, "out", {2, 8}, ov::element::f32);
//   auto model = g.build();
//
// run_on_cpu() expects tensors in the same order as add_input() calls.
// (model_inputs uses std::map internally, so OpTestGraph maintains an explicit
// insertion-order list to decouple the test from alphabetical parameter ordering.)
class OpTestGraph {
public:
    // Add a runtime input (Parameter) in call order.
    std::shared_ptr<ov::op::v0::Parameter> add_input(const std::string& name,
                                                     ov::element::Type et,
                                                     const ov::PartialShape& shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(et, shape);
        param->set_friendly_name(name);
        m_graph.model_inputs[name] = param;
        m_input_order.push_back(name);
        return param;
    }

    // Add a constant weight tensor.
    void add_weight(const std::string& name, const ov::Tensor& tensor) {
        auto cnst = std::make_shared<ov::op::v0::Constant>(tensor);
        cnst->set_friendly_name(name);
        m_graph.model_weights[name] = cnst;
    }

    // Add a pre-built weight node (e.g. from make_weight_node for quantized weights).
    void add_weight_node(const std::string& name, std::shared_ptr<ov::Node> node) {
        node->set_friendly_name(name);
        m_graph.model_weights[name] = std::move(node);
    }

    // Register a single op node.
    void add_op(const std::string& op_type,
                const std::vector<std::string>& input_names,
                const std::string& output_name,
                const ov::PartialShape& output_shape,
                ov::element::Type output_type,
                int op_case = 0,
                const std::map<std::string, ov::PartialShape>& input_shapes = {},
                const std::map<std::string, ov::element::Type>& input_types = {},
                const std::map<std::string, ov::Any>& attributes = {}) {
        GgufOp op;
        op.op_type = op_type;
        op.name = output_name + "_node";
        op.input_names = input_names;
        op.output_name = output_name;
        op.output_shape = output_shape;
        op.output_type = output_type;
        op.op_case = op_case;
        op.input_shapes = input_shapes;
        op.input_types = input_types;
        op.attributes = attributes;
        m_graph.nodes.push_back(std::move(op));
        m_graph.model_output_names.push_back(output_name);
    }

    // Build and return the converted ov::Model.
    // The model's parameter order matches the add_input() call order.
    std::shared_ptr<ov::Model> build() {
        auto graph = std::make_shared<GgufGraph>(m_graph);
        auto decoder = std::make_shared<GgufBuilderDecoder>(graph);
        auto input_model = std::make_shared<InputModel>(decoder, /*naive=*/true);
        auto translator_map = get_supported_ops();
        TranslateSession session(input_model, translator_map, /*naive=*/true);
        auto model = session.get_converted_model();

        // Re-order model parameters to match add_input() insertion order so that
        // callers can pass tensors in the same order without worrying about the
        // alphabetical ordering of std::map.
        const auto& params = model->get_parameters();
        ov::ParameterVector ordered;
        ordered.reserve(m_input_order.size());
        for (const auto& name : m_input_order) {
            for (const auto& p : params) {
                if (p->get_friendly_name() == name) {
                    ordered.push_back(p);
                    break;
                }
            }
        }
        // Replace the model's parameter vector with the insertion-ordered one.
        // ov::Model does not expose a setter; rebuild from results with ordered params.
        model =
            std::make_shared<ov::Model>(model->get_results(), model->get_sinks(), ordered, model->get_friendly_name());
        return model;
    }

    // Return the list of input names in add_input() order (for documentation/debug).
    const std::vector<std::string>& input_order() const {
        return m_input_order;
    }

private:
    GgufGraph m_graph;
    std::vector<std::string> m_input_order;
};

// ── CPU inference ─────────────────────────────────────────────────────────────

// Compile model on CPU and run one forward pass.
// inputs: one ov::Tensor per model parameter (in parameter order).
// Returns one ov::Tensor per model output.
inline std::vector<ov::Tensor> run_on_cpu(const std::shared_ptr<ov::Model>& model,
                                          const std::vector<ov::Tensor>& inputs) {
    static ov::Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto req = compiled.create_infer_request();
    const auto& params = model->get_parameters();
    EXPECT_EQ(params.size(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        req.set_input_tensor(i, inputs[i]);
    req.infer();
    std::vector<ov::Tensor> results;
    for (size_t i = 0; i < model->get_output_size(); ++i)
        results.push_back(req.get_output_tensor(i));
    return results;
}

// ── Comparison ────────────────────────────────────────────────────────────────

// Assert |actual[i] - expected[i]| <= atol for every element.
inline void expect_near(const ov::Tensor& actual,
                        const std::vector<float>& expected,
                        float atol,
                        const std::string& label = "") {
    ASSERT_EQ(ov::shape_size(actual.get_shape()), expected.size()) << label << ": size mismatch";
    const float* ptr = actual.data<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(ptr[i], expected[i], atol) << label << " at element " << i;
    }
}

}  // namespace ov_gguf_test
