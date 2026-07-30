// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <string>

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf::pass {

/// \brief Turn the frontend's stateless GGUF model into an OpenVINO stateful one.
///
/// The GGUF frontend is universal: it always converts to a STATELESS graph, in which every KV cache
/// is an explicit model Parameter written by an ov::frontend::gguf::SetRows placeholder and read
/// back as a Result. That mirrors how optimum-intel exports (a stateless model plus a caller-side
/// `apply_make_stateful_transformation`), and it is what keeps statefulness out of the decoder
/// interface: a decoder describes ggml operations, not a deployment mode.
///
/// Statefulness is therefore a CALLER concern. Register this pass as a
/// ov::frontend::DecoderTransformationExtension before conversion:
///
///     ov::frontend::gguf::FrontEnd fe;
///     fe.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
///         ov::frontend::gguf::pass::MakeStateful()));
///     auto model = fe.convert(fe.load(decoder_or_gguf_path));
///
/// or, when going through ov::Core (which forwards its extensions to the frontend before load):
///
///     core.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
///         ov::frontend::gguf::pass::MakeStateful()));
///     auto model = core.read_model("model.gguf");
///
/// Extensions run in the frontend's normalization stage AHEAD of the built-in
/// LowerSetRowsStateless, so this pass consumes the KV-cache SetRows ops and the default stateless
/// lowering only ever sees the ones left over (e.g. MoE routing writes, which stay stateless).
///
/// Per KV cache it replaces the Parameter/Result pair with a Variable + ReadValue(empty init) +
/// Concat(past, this step's rows) + Assign, optionally reordering the past by beam_idx first. Only a
/// SetRows whose destination is a model Parameter is converted; everything else is untouched.
///
/// The empty ReadValue init is deliberate and required, not cosmetic: CPU's stateful_sdpa_fusion
/// folds the cache into ScaledDotProductAttentionWithKVCache, whose MemoryInputSDPA aborts on a
/// MemoryInput with zero parent edges. genai / optimum build the cache the same way.
///
/// SCOPE. This pass does the one part that is common to every stateless GGUF graph: growing the
/// cache. It deliberately does NOT touch the attention mask. A graph whose mask Parameter is
/// dynamically sized (what the native .gguf builder emits) needs no mask change at all -- the
/// caller simply feeds a mask as wide as the grown cache. A graph that preallocates a fixed mask
/// window instead reads it through a slice sized for that window, and the caller must re-slice it
/// to (query_len, past + query_len); the llama.cpp backend does exactly that in its own extension.
class GGUF_FRONTEND_API MakeStateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("gguf::MakeStateful");

    /// \param skip_caches Friendly names of cache Parameters to leave stateless. A sliding-window
    ///        cache is evicted from the front rather than only appended to, so an append-grown
    ///        Variable would not reproduce it; such caches keep the stateless form.
    /// \param append_axis Cache axis the new rows are appended along (the token axis). -1 infers it
    ///        as the cache Parameter's single dynamic axis, which is how a graph that does not
    ///        preallocate the cache states its token axis. Pass an explicit axis for a fully static
    ///        (preallocated) cache, where there is nothing to infer from.
    /// \param beam_idx_name Name of the beam-reorder input. When the model has such a Parameter, the
    ///        past cache is gathered by it along the batch axis before the append Concat. With
    ///        batch 1 / beam_idx [0] that Gather is an identity, but emitting it is what lets CPU's
    ///        stateful_sdpa_fusion match, and it is what makes beam search work. Absent from the
    ///        model means no Gather is emitted.
    explicit MakeStateful(std::set<std::string> skip_caches = {},
                          int64_t append_axis = -1,
                          std::string beam_idx_name = "beam_idx")
        : m_skip_caches(std::move(skip_caches)),
          m_append_axis(append_axis),
          m_beam_idx_name(std::move(beam_idx_name)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::set<std::string> m_skip_caches;
    int64_t m_append_axis;
    std::string m_beam_idx_name;
};

}  // namespace ov::frontend::gguf::pass
