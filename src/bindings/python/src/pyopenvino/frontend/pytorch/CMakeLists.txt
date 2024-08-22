# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(${pyopenvino_SOURCE_DIR}/frontend/frontend_module.cmake)
frontend_module(py_pytorch_frontend pytorch ${OV_CPACK_COMP_PYTHON_OPENVINO}_${pyversion})

include(FetchContent)

FetchContent_Declare(safetensors.h
    URL https://github.com/hsnyder/safetensors.h/archive/974a85d7dfd6e010558353226638bb26d6b9d756.tar.gz
    URL_HASH SHA256=9aaf5961609601cf9aaa96582a207bce7c6e5fbf57ed2cc669bb7bde6a937d4b)
FetchContent_MakeAvailable(safetensors.h)

target_include_directories(py_pytorch_frontend SYSTEM PRIVATE "${safetensors.h_SOURCE_DIR}")
