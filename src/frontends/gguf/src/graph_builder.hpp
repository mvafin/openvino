// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "gguf_reader.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Build a stateful llama / qwen2 / qwen3 / qwen3vl decoder Model from a parsed
// GGUF file. Throws if the GGUF architecture is not supported by this PoC.
std::shared_ptr<ov::Model> build_model(const GGUFFile& file);

// Build a Qwen3-VL multi-modal projector (mmproj) Model: 24-layer ViT +
// projector MLP + DeepStack mergers. Inputs `pixel_values [B,3,H,W]` (where H
// and W must equal the trained `clip.vision.image_size`); outputs
// `vision_features [B,N/m^2,proj_dim]` plus one `deepstack_<k>` tensor of the
// same shape per layer index `k` flagged in `clip.vision.is_deepstack_layers`.
// Throws if the GGUF is not a Qwen3-VL `qwen3vl_merger` mmproj.
std::shared_ptr<ov::Model> build_mmproj_model(const GGUFFile& file);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
