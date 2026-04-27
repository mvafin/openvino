// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GGUF frontend: load_impl / convert / supported_impl.
//
// `convert()` synthesizes a stateful llama / qwen2 decoder graph from the
// hyperparameters and weight tensors of a GGUF file. The produced ov::Model
// matches the I/O contract used by openvino.genai's LLMPipeline:
//
//   inputs : input_ids[i64, ?, ?], attention_mask[i64, ?, ?],
//            position_ids[i64, ?, ?], beam_idx[i32, ?]
//   outputs: logits[f32, ?, ?, vocab]
//   state  : past_key_values.{i}.keypresent.{i}.key  (one per layer)
//            past_key_values.{i}.valuepresent.{i}.key

#include "openvino/frontend/gguf/frontend.hpp"

#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gguf_reader.hpp"
#include "graph_builder.hpp"
#include "input_model.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/common/path_util.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

std::string get_path(const std::vector<ov::Any>& variants) {
    if (variants.empty())
        return {};
    if (auto p = get_path_from_any(variants[0])) {
        return p->string();
    }
    return {};
}

}  // namespace

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    const auto path = get_path(variants);
    if (path.empty())
        return false;
    if (path.size() < 5 || path.substr(path.size() - 5) != ".gguf") {
        // Fall back to magic sniff (cheap, no full parse).
        return is_gguf(path);
    }
    return is_gguf(path);
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    const auto path = get_path(variants);
    OPENVINO_ASSERT(!path.empty(), "GGUF frontend: only file-path inputs are supported by this PoC.");
    auto file = std::make_shared<GGUFFile>(path);
    return std::make_shared<InputModel>(std::move(file));
}

std::shared_ptr<ov::Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    auto im = std::dynamic_pointer_cast<InputModel>(model);
    OPENVINO_ASSERT(im, "GGUF frontend: invalid InputModel.");
    // Dispatch by GGUF `general.architecture`:
    //   - "clip" + "qwen3vl_merger" projector  -> standalone vision encoder
    //     (mmproj). Produces an OV Model with `pixel_values` input and
    //     `vision_features` + `deepstack_<k>` outputs.
    //   - everything else                      -> stateful LLM decoder.
    const auto& meta = im->file()->metadata();
    auto arch = meta.find("general.architecture");
    if (arch != meta.end() && arch->second.get<std::string>() == "clip") {
        return build_mmproj_model(*im->file());
    }
    return build_model(*im->file());
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
