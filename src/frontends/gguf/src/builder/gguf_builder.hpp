// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ggml_graph.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Build a GgufGraph natively from a .gguf file (no llama.cpp / gguf dependency).
// Parses the container, then dispatches to a per-architecture builder that emits nodes in
// the GGML op vocabulary reproducing llama.cpp's cgraph topology for that architecture.
// Throws if the architecture is not supported natively.
std::shared_ptr<GgufGraph> build_ggml_graph_from_gguf(const std::string& file);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
