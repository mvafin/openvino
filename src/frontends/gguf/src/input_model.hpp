// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "gguf_reader.hpp"
#include "openvino/frontend/input_model.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Minimal InputModel: just owns the parsed GGUF file. The PoC does not expose
// Place objects (a GGUF file has no graph nodes to expose); the dynamic shapes
// of input_ids / attention_mask / position_ids / beam_idx are fixed by the
// converter and not user-tunable.
class InputModel : public ov::frontend::InputModel {
public:
    explicit InputModel(std::shared_ptr<GGUFFile> file) : m_file(std::move(file)) {}
    const std::shared_ptr<GGUFFile>& file() const {
        return m_file;
    }

private:
    std::shared_ptr<GGUFFile> m_file;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
