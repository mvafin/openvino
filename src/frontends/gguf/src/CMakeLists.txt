# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME gguf
                FILEDESCRIPTION "FrontEnd to load GGUF (GGML Unified Format) model files (PoC)"
                LINK_LIBRARIES openvino::core::dev)
