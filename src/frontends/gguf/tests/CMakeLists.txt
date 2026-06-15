# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gguf_frontend_tests)

# Builder sources are compiled directly into the test binary because the fill functions
# and make_weight_node are internal (not exported from the shared library).
set(BUILDER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/builder")
set(BUILDER_SRCS
    "${BUILDER_DIR}/gguf.cpp"
    "${BUILDER_DIR}/gguf_quants.cpp"
    "${BUILDER_DIR}/weights.cpp"
    "${BUILDER_DIR}/gguf_builder_decoder.cpp"
)

# Per-op tests (test_ops.cpp) also need the translate pipeline sources that are
# part of the frontend shared library.  We compile them directly here so the
# test binary is self-contained without linking the full gguf_frontend .so.
set(FRONTEND_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/translate_session.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/input_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op_table.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/cont.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/cpy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/flash_attn_ext.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/get_rows.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/glu_geglu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/glu_swiglu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/glu_swiglu_oai.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/moe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/mulmat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/permute.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/reshape.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/rms_norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/rope.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/scale.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/set_rows.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/softmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/transpose.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/unary_gelu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/unary_silu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/unary_misc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/unary_math.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/concat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/repeat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/op/view.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/pass/squeeze_matmul.cpp"
)

# test_data/ lives next to this CMakeLists.txt; expose the absolute path to the
# C++ test binary so load_npy() can find files without hard-coding a path.
set(GGUF_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_data")

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADDITIONAL_SOURCE_DIRS
    LINK_LIBRARIES
        gtest
        gtest_main
        openvino::runtime::dev
        openvino::runtime
        openvino_transformations
        openvino::cnpy
    INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/../src"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src/builder"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    ADD_CLANG_FORMAT
    LABELS
        OV UNIT GGUF_FE
)

# Add all extra sources explicitly (ov_add_test_target only scans ROOT for .cpp).
target_sources(${TARGET_NAME} PRIVATE ${BUILDER_SRCS} ${FRONTEND_SRCS})

# Inject the test data directory path as a compile-time string constant.
target_compile_definitions(${TARGET_NAME} PRIVATE
    TEST_DATA_DIR="${GGUF_TEST_DATA_DIR}"
)

ov_build_target_faster(${TARGET_NAME} PCH)
