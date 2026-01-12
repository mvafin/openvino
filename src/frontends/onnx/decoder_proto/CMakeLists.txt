# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_onnx_decoder")

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

add_library(${TARGET_NAME} STATIC ${SOURCES} ${PUBLIC_HEADERS})

add_library(openvino::onnx_proto_decoder ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        openvino::runtime
        openvino::util
        onnx
)

target_compile_definitions(${TARGET_NAME} PRIVATE
    ONNX_BUILD_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>
)

# Installation
install(TARGETS ${TARGET_NAME}
    EXPORT OpenVINOTargets
    ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR}
    COMPONENT ${OV_CPACK_COMP_CORE}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${OV_CPACK_INCLUDEDIR}
    COMPONENT ${OV_CPACK_COMP_CORE_DEV}
)

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/")
