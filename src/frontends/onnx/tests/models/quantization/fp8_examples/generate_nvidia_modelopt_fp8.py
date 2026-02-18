#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Generate an example ONNX model with FP8 E4M3 Quantize/Dequantize operations
simulating the output from NVIDIA ModelOpt.

NVIDIA ModelOpt typically uses:
- FLOAT8E4M3FN for activations (wider range)
- Per-tensor quantization with dynamic scales
- QuantizeLinear -> Compute -> DequantizeLinear pattern
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

def create_nvidia_modelopt_fp8_model():
    """
    Creates an ONNX model with FP8 E4M3 quantization pattern similar to NVIDIA ModelOpt output.
    
    Model structure:
    Input (FP32) -> QuantizeLinear (to FP8 E4M3) -> DequantizeLinear (to FP32) 
    -> MatMul with FP8 quantized weights (pre-quantized and folded) -> Output
    """
    
    # Create input tensor
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    
    # Create output tensor
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    
    # Quantization scale for input (scalar, FP32)
    input_scale = helper.make_tensor('input_scale', TensorProto.FLOAT, [], [0.05])
    
    # Create QuantizeLinear node for input
    quantize_input = helper.make_node(
        'QuantizeLinear',
        inputs=['input', 'input_scale'],
        outputs=['input_quantized'],
        name='quantize_input'
    )
    
    # Create DequantizeLinear node for input
    dequantize_input = helper.make_node(
        'DequantizeLinear',
        inputs=['input_quantized', 'input_scale'],
        outputs=['input_dequantized'],
        name='dequantize_input'
    )
    
    # Create FP8 weight constant directly (pre-quantized)
    # Original FP32 weights: [[0.5, -0.3, 0.2, 0.8], [0.1, 0.4, -0.6, 0.3], [0.7, -0.2, 0.5, -0.4], [0.3, 0.6, -0.1, 0.9]]
    # Quantized to FP8 E4M3FN (using ml_dtypes for proper conversion)
    try:
        import ml_dtypes
        weight_data_fp32 = np.array([[0.5, -0.3, 0.2, 0.8],
                                      [0.1, 0.4, -0.6, 0.3],
                                      [0.7, -0.2, 0.5, -0.4],
                                      [0.3, 0.6, -0.1, 0.9]], dtype=np.float32)
        # Convert to FP8 E4M3FN
        weight_data_fp8 = weight_data_fp32.astype(ml_dtypes.float8_e4m3fn)
    except ImportError:
        # Fallback: Use raw bytes representation
        # This is an approximation - in reality, proper FP8 conversion would be needed
        weight_data_fp8 = np.array([[0.5, -0.3125, 0.1875, 0.75],
                                     [0.09375, 0.375, -0.625, 0.3125],
                                     [0.6875, -0.1875, 0.5, -0.375],
                                     [0.3125, 0.625, -0.09375, 0.875]], dtype=np.float32)
        # Cast the array to use FP8 type (this is a workaround for demonstration)
        weight_data_fp8 = weight_data_fp8.view(dtype=np.uint8)
    
    # Create the weight tensor in FP8 format
    weight_tensor_fp8 = helper.make_tensor(
        name='weight_fp8',
        data_type=TensorProto.FLOAT8E4M3FN,
        dims=[4, 4],
        vals=weight_data_fp8.tobytes(),
        raw=True
    )
    
    # Weight quantization scale
    weight_scale = helper.make_tensor('weight_scale', TensorProto.FLOAT, [], [1.0])  # Scale is 1.0 for pre-quantized weights
    
    # Dequantize weight (no QuantizeLinear needed - weight is already in FP8)
    dequantize_weight = helper.make_node(
        'DequantizeLinear',
        inputs=['weight_fp8', 'weight_scale'],
        outputs=['weight_dequantized'],
        name='dequantize_weight'
    )
    
    # MatMul operation
    matmul = helper.make_node(
        'MatMul',
        inputs=['input_dequantized', 'weight_dequantized'],
        outputs=['output'],
        name='matmul'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[quantize_input, dequantize_input, dequantize_weight, matmul],
        name='nvidia_modelopt_fp8_example',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[input_scale, weight_tensor_fp8, weight_scale],
    )
    
    # Add type information for quantized tensors
    # FP8 E4M3FN is type 17 in ONNX
    input_quantized_info = helper.make_tensor_value_info('input_quantized', TensorProto.FLOAT8E4M3FN, [1, 4])
    
    graph.value_info.extend([input_quantized_info])
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='NVIDIA ModelOpt Example Generator',
        opset_imports=[helper.make_opsetid('', 19)]  # ONNX opset 19 supports FP8
    )
    
    # Check the model
    onnx.checker.check_model(model)
    
    return model

if __name__ == '__main__':
    model = create_nvidia_modelopt_fp8_model()
    output_path = 'nvidia_modelopt_fp8_e4m3.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")
    print("\nModel summary:")
    print(f"  - Input quantization: FP32 -> FP8 E4M3FN -> FP32")
    print(f"  - Weight storage: Pre-quantized FP8 E4M3FN constant (folded)")
    print(f"  - Weight dequantization: FP8 E4M3FN -> FP32")
    print(f"  - Operation: MatMul with dequantized inputs")
    print(f"  - Simulates NVIDIA ModelOpt FP8 quantization pattern with folded weights")
