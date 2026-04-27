# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gguf_frontend_tests)

# We test the frontend's weight-conversion path end-to-end (build a Constant +
# decompression subgraph, compile it on CPU, infer, compare to the reference
# fp16 dequantizer in `gguf_dequant.cpp`). The two relevant translation units
# (`gguf_compress.cpp` and `gguf_dequant.cpp`) are not part of the public ABI
# of openvino_gguf_frontend (symbols are hidden by default), so we recompile
# them directly into the test executable. We exclude the rest of the frontend
# sources to keep the test self-contained and free of plugin-init code paths.
set(GGUF_FE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADDITIONAL_SOURCE_DIRS
        ${GGUF_FE_SRC_DIR}
    EXCLUDED_SOURCE_PATHS
        ${GGUF_FE_SRC_DIR}/frontend.cpp
        ${GGUF_FE_SRC_DIR}/gguf.cpp
        ${GGUF_FE_SRC_DIR}/graph_builder.cpp
    INCLUDES
        ${GGUF_FE_SRC_DIR}
    DEPENDENCIES
        openvino_gguf_frontend
        openvino_intel_cpu_plugin
    LINK_LIBRARIES
        gtest
        gtest_main
        openvino::runtime
        openvino::runtime::dev
        common_test_utils
    LABELS
        OV UNIT GGUF_FE
)

ov_build_target_faster(${TARGET_NAME} PCH)
