#!/bin/bash
# Script to run PyTorch layer tests for OpenVINO
# This script runs the PyTorch layer tests with the precommit marker

set -e

# Set default values for environment variables if not set
export TEST_DEVICE="${TEST_DEVICE:-CPU}"
export TEST_PRECISION="${TEST_PRECISION:-FP32}"

# Change to the layer_tests directory
cd "$(dirname "$0")/tests/layer_tests"

echo "=================================="
echo "PyTorch Layer Tests Configuration"
echo "=================================="
echo "TEST_DEVICE: $TEST_DEVICE"
echo "TEST_PRECISION: $TEST_PRECISION"
echo "=================================="
echo ""

# Run PyTorch layer tests with precommit marker
# Using -n logical for parallel execution (like in CI)
echo "Running PyTorch layer tests with precommit marker..."
python3 -m pytest pytorch_tests \
    -m precommit \
    -v \
    --tb=short \
    --junitxml=TEST-pytorch.xml \
    --html=TEST-pytorch.html \
    --self-contained-html

echo ""
echo "=================================="
echo "Test execution completed!"
echo "Results saved to:"
echo "  - TEST-pytorch.xml"
echo "  - TEST-pytorch.html"
echo "=================================="
