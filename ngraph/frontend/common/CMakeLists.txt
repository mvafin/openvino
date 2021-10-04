# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_common")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(${TARGET_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME} STATIC  ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
add_library(openvino::frontend::common ALIAS ${TARGET_NAME})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(${TARGET_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${${TARGET_NAME}_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${Protobuf_INCLUDE_DIRS}
                                                         ${CMAKE_CURRENT_BINARY_DIR})

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::common)
install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT ngraph
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT ngraph
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT ngraph)

export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
