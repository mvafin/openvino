# FP8 Quantize/Dequantize ONNX Model Examples - Implementation Summary

## Overview

This implementation provides example ONNX models with FP8 (8-bit floating-point) Quantize/Dequantize operations, demonstrating patterns commonly produced by NVIDIA ModelOpt and AMD Quark quantization tools.

## What Was Implemented

### 1. Model Generation Scripts

Created two Python scripts that generate synthetic ONNX models with FP8 quantization patterns:

#### `generate_nvidia_modelopt_fp8.py`
- Generates a model with FP8 E4M3FN quantization
- Uses per-tensor quantization (typical of NVIDIA ModelOpt)
- Pattern: Input -> QuantizeLinear(E4M3) -> DequantizeLinear -> MatMul -> Output
- Demonstrates weight quantization for MatMul operation
- Output: `nvidia_modelopt_fp8_e4m3.onnx`

#### `generate_amd_quark_fp8.py`
- Generates a model with FP8 E5M2 quantization
- Uses per-channel quantization for weights (typical of AMD Quark)
- Pattern: Input -> QuantizeLinear(E5M2) -> DequantizeLinear -> Conv -> Output
- Demonstrates per-channel weight quantization for Conv operation
- Output: `amd_quark_fp8_e5m2.onnx`

### 2. Generated Models

Both binary ONNX models (.onnx) and human-readable prototxt versions (.prototxt) are included:

- `nvidia_modelopt_fp8_e4m3.onnx` / `.prototxt` - NVIDIA ModelOpt style model
- `amd_quark_fp8_e5m2.onnx` / `.prototxt` - AMD Quark style model

All models have been validated using:
- ONNX model checker (`onnx.checker.check_model()`)
- ONNX shape inference (`onnx.shape_inference.infer_shapes()`)

### 3. Documentation

Created comprehensive `README.md` in the `fp8_examples/` directory covering:
- Background on FP8 quantization
- Detailed description of each model
- Model structures and characteristics
- Usage instructions
- Technical details about FP8 data types
- Quantization strategies comparison
- References to NVIDIA ModelOpt and AMD Quark

### 4. Test Cases

Created `onnx_import_fp8_quant.in.cpp` with test cases that:
- Verify NVIDIA ModelOpt FP8 E4M3 model can be loaded
- Verify AMD Quark FP8 E5M2 model can be loaded
- Check for expected operations (QuantizeLinear, DequantizeLinear, MatMul/Conv)
- Validate model structure (inputs, outputs, parameters)

Updated `CMakeLists.txt` to include the new test file in the build.

## File Structure

```
src/frontends/onnx/tests/models/quantization/fp8_examples/
├── README.md                              # Comprehensive documentation
├── generate_nvidia_modelopt_fp8.py        # Generator script for NVIDIA model
├── generate_amd_quark_fp8.py              # Generator script for AMD model
├── nvidia_modelopt_fp8_e4m3.onnx          # Binary ONNX model (NVIDIA)
├── nvidia_modelopt_fp8_e4m3.prototxt      # Text format (NVIDIA)
├── amd_quark_fp8_e5m2.onnx                # Binary ONNX model (AMD)
└── amd_quark_fp8_e5m2.prototxt            # Text format (AMD)

src/frontends/onnx/tests/
├── onnx_import_fp8_quant.in.cpp           # Test cases for FP8 models
└── CMakeLists.txt                         # Updated to include new tests
```

## Key Technical Details

### FP8 Data Types Used

1. **FLOAT8E4M3FN (Type ID: 17)**
   - 1 sign bit, 4 exponent bits, 3 mantissa bits
   - Wider dynamic range, suitable for activations
   - Used in NVIDIA ModelOpt example

2. **FLOAT8E5M2 (Type ID: 19)**
   - 1 sign bit, 5 exponent bits, 2 mantissa bits
   - Higher precision, smaller range
   - Used in AMD Quark example

### ONNX Operations

Both models use:
- **QuantizeLinear**: Converts FP32 to FP8
- **DequantizeLinear**: Converts FP8 back to FP32
- Standard compute operations (MatMul, Conv)

### Quantization Patterns

**NVIDIA ModelOpt pattern:**
```
FP32 Input -> QuantizeLinear -> FP8 -> DequantizeLinear -> FP32 -> Compute
```

**AMD Quark pattern:**
```
FP32 Input -> QuantizeLinear (per-channel) -> FP8 -> DequantizeLinear -> FP32 -> Compute
```

## Validation Results

Both models have been validated and confirmed to:
- ✓ Pass ONNX model checker
- ✓ Support shape inference
- ✓ Use correct FP8 data types (E4M3FN and E5M2)
- ✓ Include proper QuantizeLinear/DequantizeLinear operations
- ✓ Have valid tensor shapes and connections

## Usage

### Regenerating Models

```bash
cd src/frontends/onnx/tests/models/quantization/fp8_examples/
python3 generate_nvidia_modelopt_fp8.py
python3 generate_amd_quark_fp8.py
```

### Running Tests

The test cases can be built and run as part of the OpenVINO test suite:

```bash
# From OpenVINO build directory
cmake --build . --target ov_onnx_frontend_tests
ctest -R onnx_fp8
```

## References

1. [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
2. [AMD Quark](https://github.com/amd/quark)
3. [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
4. [ONNX Operators - QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
5. [ONNX Operators - DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html)

## Notes

- These are synthetic models designed to demonstrate FP8 quantization patterns
- They are not actual models quantized by NVIDIA ModelOpt or AMD Quark
- The patterns and structures are based on typical outputs from these tools
- The models use ONNX opset 19, which includes FP8 support
- Both models are small and suitable for testing and validation purposes

## Future Work

Potential enhancements:
- Add more complex multi-layer models
- Include examples of FLOAT8E4M3FNUZ and FLOAT8E5M2FNUZ variants
- Add models with mixed precision (FP8 + FP16)
- Include KV-cache quantization examples
- Add more compute operations (Attention, LayerNorm, etc.)
