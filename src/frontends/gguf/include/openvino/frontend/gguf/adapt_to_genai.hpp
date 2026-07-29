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
///
/// LAYOUT POLYMORPHISM (why this pass derives the leading dims instead of pinning them).
/// The result must be valid under BOTH attention backends, which disagree about where the token
/// count lives. Plain SDPA inference feeds input_ids as [1, tokens]. ov::pass::SDPAToPagedAttention
/// instead rewrites input_ids to rank-1 [tokens] and splices an Unsqueeze(axis=1) in front of its
/// consumers, so the body sees [tokens, 1] -- its hardcoded flattens (Reshape({0,-1}) on the PA
/// operands) then read the token count out of dim 0.
///
/// Both are the same buffer: ggml's activation layout is [batch, tokens, heads, head_size] with
/// batch == 1, and [1, tokens, H, D] and [tokens, 1, H, D] are element-for-element identical. So a
/// single graph serves both, PROVIDED no node pins the leading two dims to constants. This pass
/// therefore derives them from the live input_ids (and the op translators reshape with
/// special_zero, copying dim 0 through rather than writing a literal 1). Under SDPA that is exactly
/// the old batch-major graph; under PagedAttention the tokens reach dim 0 on their own, which is
/// what lets SDPAToPagedAttention's rewrite bind. No backend flag is needed, and no attention
/// re-layout is performed here.
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
