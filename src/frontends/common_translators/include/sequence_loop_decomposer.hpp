// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pass {

/// \brief Decomposes ONNX-style sequence state variables in Loop nodes.
///
/// When an ONNX model carries a sequence (list of tensors) as Loop state — e.g.
/// for KV-cache in autoregressive models — the sequence is represented as:
///   - Initial value:  SequenceMark with 0 elements (SequenceEmpty)
///   - Body output:    SequenceMark with N elements (SequenceConstruct)
///   - Body consumers: NotSupportedONNXNode(SequenceAt), ONNXFrameworkNode(SequenceLength), etc.
///
/// This pass replaces the single sequence state variable with N individual tensor
/// state variables, eliminating all sequence helper nodes from the graph.
///
/// The pass recursively processes Loop body and nested If/Loop subgraph models.
class SequenceLoopDecomposer : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pass::SequenceLoopDecomposer");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace frontend
}  // namespace ov
