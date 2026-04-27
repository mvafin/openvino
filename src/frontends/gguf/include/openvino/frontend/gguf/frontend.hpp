// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

/// \brief Proof-of-concept FrontEnd that loads GGUF (GGML Unified Format) files
///        and synthesizes an OpenVINO Model.
///
/// Unlike file-format frontends such as ONNX or TensorFlow, GGUF does not store
/// a computation graph: it stores hyperparameters + named weights. This frontend
/// therefore *synthesizes* the graph from the architecture string + metadata.
///
/// PoC scope: llama / qwen2 architectures, F32 / F16 / BF16 weights only,
/// no tokenizer / generation config (those are out of scope for ov::Model).
class GGUF_API FrontEnd : public ov::frontend::FrontEnd {
public:
    FrontEnd() = default;

    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override;
    std::string get_name() const override {
        return "gguf";
    }

protected:
    bool supported_impl(const std::vector<ov::Any>& variants) const override;
    InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
