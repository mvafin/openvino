# Running PyTorch Layer Tests

This directory contains scripts for running PyTorch layer tests for OpenVINO.

## Prerequisites

1. **Install OpenVINO**:
   ```bash
   pip install openvino
   ```

2. **Install PyTorch test dependencies**:
   ```bash
   pip install -r tests/requirements_pytorch
   ```

## Running Tests

### Quick Start

The simplest way to run PyTorch layer tests is using the provided Python script:

```bash
# Run precommit tests on CPU with FP32 precision
python3 run_pytorch_tests.py
```

### Advanced Usage

The Python script provides many options for customizing test execution:

```bash
# Show all available options
python3 run_pytorch_tests.py --help

# Run all tests (not just precommit) on CPU and GPU with both precisions
python3 run_pytorch_tests.py --device CPU GPU --precision FP32 FP16 --marker ""

# Run a specific test file
python3 run_pytorch_tests.py --test-file test_add.py

# Run tests in parallel for faster execution
python3 run_pytorch_tests.py --parallel

# Verbose output
python3 run_pytorch_tests.py -v

# Dry run to see what would be executed
python3 run_pytorch_tests.py --dry-run
```

### Using the Bash Script

Alternatively, you can use the bash script:

```bash
# Run with default settings (CPU, FP32, precommit tests)
./run_pytorch_layer_tests.sh

# Run with custom device and precision
TEST_DEVICE="CPU;GPU" TEST_PRECISION="FP32;FP16" ./run_pytorch_layer_tests.sh
```

### Manual Execution

You can also run tests directly with pytest:

```bash
cd tests/layer_tests

# Set environment variables
export TEST_DEVICE="CPU"
export TEST_PRECISION="FP32"

# Run precommit tests
python3 -m pytest pytorch_tests -m precommit -v

# Run a specific test file
python3 -m pytest pytorch_tests/test_add.py -v

# Run all tests
python3 -m pytest pytorch_tests -v

# Run with parallel execution
python3 -m pytest pytorch_tests -m precommit -n logical -v
```

## Test Markers

The tests use pytest markers to categorize tests:

- `precommit` - Tests that should run before each commit (faster subset)
- `precommit_torch_export` - Tests for torch.export mode
- `precommit_fx_backend` - Tests for FX backend
- `nightly` - Comprehensive tests that run nightly

## Environment Variables

- `TEST_DEVICE` - Device(s) to run tests on (default: `CPU;GPU`)
  - Supported values: `CPU`, `GPU`, or both separated by `;`
- `TEST_PRECISION` - Inference precision(s) to use (default: `FP32;FP16`)
  - Supported values: `FP32`, `FP16`, or both separated by `;`

## Test Output

Test results are saved to:
- `TEST-pytorch.xml` - JUnit XML format for CI/CD integration
- `TEST-pytorch.html` - HTML report for human review

## Examples

### Run precommit tests on CPU
```bash
python3 run_pytorch_tests.py
```

### Run all tests with parallel execution
```bash
python3 run_pytorch_tests.py --marker "" --parallel
```

### Run convolution tests only
```bash
python3 run_pytorch_tests.py --test-file test_convnd.py -v
```

### Run tests on both CPU and GPU with both precisions
```bash
python3 run_pytorch_tests.py --device CPU GPU --precision FP32 FP16
```

## Troubleshooting

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r tests/requirements_pytorch
```

### OpenVINO not found
Install OpenVINO:
```bash
pip install openvino
```

### Test failures
Check the HTML report for detailed error messages:
```bash
# Open in browser
xdg-open tests/layer_tests/TEST-pytorch.html  # Linux
open tests/layer_tests/TEST-pytorch.html       # macOS
start tests/layer_tests/TEST-pytorch.html      # Windows
```

## Additional Information

For more information about layer tests, see:
- [Layer Tests README](tests/layer_tests/README.md)
- [OpenVINO Documentation](https://docs.openvino.ai/)
