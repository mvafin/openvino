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
    -> MatMul with FP8 quantized weights -> Output
    """
    
    # Create input tensor
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    
    # Create output tensor
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    
    # Quantization scale for input (scalar, FP32)
    input_scale = helper.make_tensor('input_scale', TensorProto.FLOAT, [], [0.05])
    
    # Quantization zero point for input (scalar, FP8 E4M3 - but represented as 0 for FP8)
    # Note: FP8 types don't use zero point in the same way as integer quantization
    
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
    
    # Create weight constant (FP32, will be quantized)
    weight_data = np.array([[0.5, -0.3, 0.2, 0.8],
                            [0.1, 0.4, -0.6, 0.3],
                            [0.7, -0.2, 0.5, -0.4],
                            [0.3, 0.6, -0.1, 0.9]], dtype=np.float32)
    weight_tensor = numpy_helper.from_array(weight_data, name='weight')
    
    # Weight quantization scale
    weight_scale = helper.make_tensor('weight_scale', TensorProto.FLOAT, [], [0.01])
    
    # Quantize weight
    quantize_weight = helper.make_node(
        'QuantizeLinear',
        inputs=['weight', 'weight_scale'],
        outputs=['weight_quantized'],
        name='quantize_weight'
    )
    
    # Dequantize weight
    dequantize_weight = helper.make_node(
        'DequantizeLinear',
        inputs=['weight_quantized', 'weight_scale'],
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
        nodes=[quantize_input, dequantize_input, quantize_weight, dequantize_weight, matmul],
        name='nvidia_modelopt_fp8_example',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[input_scale, weight_tensor, weight_scale],
    )
    
    # Add type information for quantized tensors
    # FP8 E4M3FN is type 17 in ONNX
    input_quantized_info = helper.make_tensor_value_info('input_quantized', TensorProto.FLOAT8E4M3FN, [1, 4])
    weight_quantized_info = helper.make_tensor_value_info('weight_quantized', TensorProto.FLOAT8E4M3FN, [4, 4])
    
    graph.value_info.extend([input_quantized_info, weight_quantized_info])
    
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
    print(f"  - Weight quantization: FP32 -> FP8 E4M3FN -> FP32")
    print(f"  - Operation: MatMul with dequantized inputs")
    print(f"  - Simulates NVIDIA ModelOpt FP8 quantization pattern")
