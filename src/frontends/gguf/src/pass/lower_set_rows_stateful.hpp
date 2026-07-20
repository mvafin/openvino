// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

// Stateful lowering of the internal SetRows op for the KV-cache write path:
//   SetRows(data, indices, dst) -> Concat(Gather(dst, beam_idx, axis0), data, axis1)
// i.e. the new rows are appended to the destination cache tensor (which is a graph Parameter at
// this stage) along the sequence axis, after reordering the past cache by beam_idx. This produces
// the exact ReadValue->Gather(beam_idx)->Concat shape that the CPU plugin's stateful_sdpa_fusion
// matches, so the downstream SDPA fuses into ScaledDotProductAttentionWithKVCache. It is a
// ModelPass (not a MatcherPass) because it needs the model-level beam_idx Parameter, and it must
// run BEFORE ov::pass::MakeStateful (which rewrites the dst Parameter into a ReadValue and the
// corresponding Result into an Assign). Mirrors the llama.cpp cgraph / genai create_cache shape.
//
// Ported from the gguf-frontend-tokenizer branch's op/set_rows.cpp stateful branch, re-expressed
// as a lowering over the port branch's SetRows placeholder op.
class LowerSetRowsStateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::gguf::pass::LowerSetRowsStateful");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
