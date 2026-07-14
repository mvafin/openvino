# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The GGUF frontend converts GGUF models to OpenVINO, built only when ENABLE_OV_GGUF_FRONTEND is
# ON. It installs the library + headers alongside the other frontends (LINKABLE_FRONTEND, no
# SKIP_INSTALL) so direct linkers -- the llama.cpp ggml-openvino backend and OpenVINO GenAI --
# can link openvino::frontend::gguf, and it is discoverable via FrontEndManager so
# core.read_model("model.gguf") selects it by the .gguf extension + GGUF magic (see
# supported_impl / get_front_end_data in frontend.cpp). Two ingest paths converge on the same op
# translators: a live GgufDecoder (cgraph path) and a native .gguf-file builder.
ov_add_frontend(NAME gguf
                LINKABLE_FRONTEND
                FILEDESCRIPTION "FrontEnd to convert GGUF models"
                LINK_LIBRARIES openvino::util openvino::core::dev openvino::offline_transformations)

ov_build_target_faster(openvino_gguf_frontend PCH)

# The quant dequant code uses ov::parallel_for (openvino/core/parallel.hpp), which
# needs the threading backend (TBB) include directories and libraries.
ov_set_threading_interface_for(openvino_gguf_frontend)

# The op translators are ported verbatim from llama.cpp and contain narrowing
# conversions (size_t/uint64_t -> int/uint32_t/float).  Silence the corresponding
# warnings so the frontend builds with warnings-as-error (notably MSVC C4244/C4267).
if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    target_compile_options(openvino_gguf_frontend PRIVATE -Wno-conversion)
elseif(MSVC)
    # MSVC and icx on Windows accept MSVC-style flags
    target_compile_options(openvino_gguf_frontend PRIVATE /wd4244 /wd4267)
endif()

# The unit tests do not link this library: they compile the frontend sources directly
# (see tests/CMakeLists.txt), keeping the test binary self-contained.
