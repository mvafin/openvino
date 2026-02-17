#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Generate an example ONNX model with FP8 E5M2 Quantize/Dequantize operations
simulating the output from AMD Quark.

AMD Quark typically supports:
- Multiple FP8 formats including E5M2 (better precision, smaller range)
- Per-tensor and per-channel quantization
- Support for OCP MX formats
- QuantizeLinear -> Compute -> DequantizeLinear pattern
"""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

def create_amd_quark_fp8_model():
    """
    Creates an ONNX model with FP8 E5M2 quantization pattern similar to AMD Quark output.
    
    Model structure:
    Input (FP32) -> QuantizeLinear (to FP8 E5M2) -> DequantizeLinear (to FP32) 
    -> Conv with FP8 quantized weights -> Output
    """
    
    # Create input tensor (batch, channel, height, width)
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])
    
    # Create output tensor
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 6, 6])
    
    # Quantization scale for input (scalar, FP32)
    # E5M2 has higher precision than E4M3, suitable for activations needing more precision
    input_scale = helper.make_tensor('input_scale', TensorProto.FLOAT, [], [0.1])
    
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
    
    # Create weight constant for Conv (output_channels=2, input_channels=3, kernel_h=3, kernel_w=3)
    np.random.seed(42)
    weight_data = np.random.randn(2, 3, 3, 3).astype(np.float32) * 0.5
    weight_tensor = numpy_helper.from_array(weight_data, name='conv_weight')
    
    # Per-channel weight quantization scales (one scale per output channel)
    # AMD Quark supports per-channel quantization
    weight_scales = np.array([0.02, 0.03], dtype=np.float32)
    weight_scale_tensor = numpy_helper.from_array(weight_scales, name='weight_scale')
    
    # Quantize weight
    quantize_weight = helper.make_node(
        'QuantizeLinear',
        inputs=['conv_weight', 'weight_scale'],
        outputs=['weight_quantized'],
        name='quantize_weight',
        axis=0  # Per-channel quantization along output channel axis
    )
    
    # Dequantize weight
    dequantize_weight = helper.make_node(
        'DequantizeLinear',
        inputs=['weight_quantized', 'weight_scale'],
        outputs=['weight_dequantized'],
        name='dequantize_weight',
        axis=0  # Per-channel dequantization along output channel axis
    )
    
    # Conv operation
    conv = helper.make_node(
        'Conv',
        inputs=['input_dequantized', 'weight_dequantized'],
        outputs=['output'],
        name='conv',
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1]
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[quantize_input, dequantize_input, quantize_weight, dequantize_weight, conv],
        name='amd_quark_fp8_example',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[input_scale, weight_tensor, weight_scale_tensor],
    )
    
    # Add type information for quantized tensors
    # FP8 E5M2 is type 19 in ONNX
    input_quantized_info = helper.make_tensor_value_info('input_quantized', TensorProto.FLOAT8E5M2, [1, 3, 8, 8])
    weight_quantized_info = helper.make_tensor_value_info('weight_quantized', TensorProto.FLOAT8E5M2, [2, 3, 3, 3])
    
    graph.value_info.extend([input_quantized_info, weight_quantized_info])
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='AMD Quark Example Generator',
        opset_imports=[helper.make_opsetid('', 19)]  # ONNX opset 19 supports FP8
    )
    
    # Check the model
    onnx.checker.check_model(model)
    
    return model

if __name__ == '__main__':
    model = create_amd_quark_fp8_model()
    output_path = 'amd_quark_fp8_e5m2.onnx'
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")
    print("\nModel summary:")
    print(f"  - Input quantization: FP32 -> FP8 E5M2 -> FP32")
    print(f"  - Weight quantization: FP32 -> FP8 E5M2 -> FP32 (per-channel)")
    print(f"  - Operation: Conv2D with dequantized inputs")
    print(f"  - Simulates AMD Quark FP8 quantization pattern")
