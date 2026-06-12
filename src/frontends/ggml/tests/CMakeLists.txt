# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_ggml_frontend_tests)

# Builder sources are compiled directly into the test binary because the fill functions
# and make_weight_node are internal (not exported from the shared library).
set(BUILDER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/builder")
set(BUILDER_SRCS
    "${BUILDER_DIR}/gguf.cpp"
    "${BUILDER_DIR}/gguf_quants.cpp"
    "${BUILDER_DIR}/weights.cpp"
)

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADDITIONAL_SOURCE_DIRS
    LINK_LIBRARIES
        gtest
        gtest_main
        openvino::runtime::dev
        openvino_transformations
    INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/../src"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/builder"
    ADD_CLANG_FORMAT
    LABELS
        OV UNIT GGML_FE
)

# Add the builder sources explicitly (ov_add_test_target only scans ROOT for .cpp).
target_sources(${TARGET_NAME} PRIVATE ${BUILDER_SRCS})

ov_build_target_faster(${TARGET_NAME} PCH)
