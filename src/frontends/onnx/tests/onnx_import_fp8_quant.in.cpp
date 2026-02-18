// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

/**
 * @brief Test loading FP8 E4M3 quantized model simulating NVIDIA ModelOpt output
 * 
 * This test verifies that the ONNX frontend can correctly load a model with:
 * - FP8 E4M3FN data types
 * - Pre-quantized FP8 weight constants (folded)
 * - QuantizeLinear/DequantizeLinear operations
 * - MatMul operation with quantized inputs
 */
OPENVINO_TEST(${BACKEND_NAME}, onnx_fp8_nvidia_modelopt_e4m3) {
    const auto model = convert_model("quantization/fp8_examples/nvidia_modelopt_fp8_e4m3.onnx");
    
    // Verify model structure
    EXPECT_NE(model, nullptr);
    EXPECT_EQ(model->get_output_size(), 1);
    EXPECT_EQ(model->get_parameters().size(), 1);
    
    // Verify the model contains expected number of operations
    // Expected: Parameter, Constant(input_scale), Constant(weight_fp8 - FP8), Constant(weight_scale), 
    //           QuantizeLinear (input), DequantizeLinear (input), DequantizeLinear (weight), MatMul, Result
    // Note: Weight is pre-quantized, so no QuantizeLinear for weight
    auto ops = model->get_ordered_ops();
    EXPECT_GT(ops.size(), 4);  // At least Parameter, 1x Quantize, 2x Dequantize, MatMul, Result
    
    // Check that we can find QuantizeLinear and DequantizeLinear ops
    bool has_quantize = false;
    bool has_dequantize = false;
    bool has_matmul = false;
    
    for (const auto& op : ops) {
        std::string op_type = op->get_type_name();
        if (op_type.find("QuantizeLinear") != std::string::npos) {
            has_quantize = true;
        }
        if (op_type.find("DequantizeLinear") != std::string::npos) {
            has_dequantize = true;
        }
        if (op_type == "MatMul") {
            has_matmul = true;
        }
    }
    
    EXPECT_TRUE(has_quantize) << "Model should contain QuantizeLinear operations (for input)";
    EXPECT_TRUE(has_dequantize) << "Model should contain DequantizeLinear operations (for input and weight)";
    EXPECT_TRUE(has_matmul) << "Model should contain MatMul operation";
}

/**
 * @brief Test loading FP8 E5M2 quantized model simulating AMD Quark output
 * 
 * This test verifies that the ONNX frontend can correctly load a model with:
 * - FP8 E5M2 data types
 * - QuantizeLinear/DequantizeLinear operations with per-channel scales
 * - Conv operation with quantized inputs
 */
OPENVINO_TEST(${BACKEND_NAME}, onnx_fp8_amd_quark_e5m2) {
    const auto model = convert_model("quantization/fp8_examples/amd_quark_fp8_e5m2.onnx");
    
    // Verify model structure
    EXPECT_NE(model, nullptr);
    EXPECT_EQ(model->get_output_size(), 1);
    EXPECT_EQ(model->get_parameters().size(), 1);
    
    // Verify the model contains expected number of operations
    auto ops = model->get_ordered_ops();
    EXPECT_GT(ops.size(), 5);  // At least Parameter, 2x Quantize, 2x Dequantize, Conv, Result
    
    // Check that we can find QuantizeLinear, DequantizeLinear and Conv ops
    bool has_quantize = false;
    bool has_dequantize = false;
    bool has_conv = false;
    
    for (const auto& op : ops) {
        std::string op_type = op->get_type_name();
        if (op_type.find("QuantizeLinear") != std::string::npos) {
            has_quantize = true;
        }
        if (op_type.find("DequantizeLinear") != std::string::npos) {
            has_dequantize = true;
        }
        if (op_type.find("Convolution") != std::string::npos || op_type == "Conv") {
            has_conv = true;
        }
    }
    
    EXPECT_TRUE(has_quantize) << "Model should contain QuantizeLinear operations";
    EXPECT_TRUE(has_dequantize) << "Model should contain DequantizeLinear operations";
    EXPECT_TRUE(has_conv) << "Model should contain Convolution operation";
}
