// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

/// \brief Rewrite a GGUF-frontend model's llama.cpp-style IO into the OpenVINO GenAI
///        LLMPipeline IO contract, so the model can be driven by genai's stateful pipeline.
///
/// The GGUF frontend emits a stateful decoder with the gguf IO contract:
///   inputs : inp_tokens [1,1,1,D] i32, inp_pos [1,1,1,D] i32, inp_out_ids [1,1,1,D] i32,
///            self_kq_mask [1,1,D,D] f32 (+ self_kq_mask_swa for gpt-oss SWA),
///            token_len_per_seq [1] i64, beam_idx [D] i32
///   output : logits [1,1,seq,vocab]
///
/// genai's StatefulLLMPipeline instead feeds:
///   inputs : input_ids [b,seq] i64, attention_mask [b,kv_len] i64,
///            position_ids [b,seq] i64, beam_idx [b] i32
///   output : logits [b,seq,vocab]
///
/// This pass prepends a small subgraph that derives the gguf inputs from the genai inputs
/// (the graph-level equivalent of the python prototype tests/genai_io_adapter.py), rewires
/// the gguf Parameters to it, and reshapes the [1,1,seq,vocab] logits to [b,seq,vocab].
/// The stateful KV cache (sinks) is preserved. beam_idx is kept as a live input (genai sets
/// it) but is unused by the batch-1 stateful cache.
///
/// If the required gguf inputs are absent (e.g. the model is already in genai form), the
/// pass is a no-op and returns false.
class GGUF_FRONTEND_API AdaptToGenAI : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::AdaptToGenAI");

    /// \brief Which genai input contract to expose.
    /// IdsToLogits  : input_ids -> logits (text LLMPipeline). The only mode implemented today.
    /// EmbedsToLogits: inputs_embeds -> logits (reserved for the VLM language model, where
    ///                 image+text embeddings are merged outside the graph). Not yet implemented.
    enum class InputMode { IdsToLogits, EmbedsToLogits };

    explicit AdaptToGenAI(InputMode mode = InputMode::IdsToLogits) : m_mode(mode) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    InputMode m_mode;
};

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
