// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "gguf_graph.hpp"
#include "openvino/frontend/gguf/decoder.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// GgufDecoder implementation over a GgufGraph built natively from a .gguf file (no
// llama.cpp / gguf dependency). It is the OpenVINO-side counterpart of llama.cpp's
// cgraph-backed GgmlOvDecoder: both feed the same op translators / TranslateSession.
//
// The whole-graph constructor wraps a complete GgufGraph. Following the node-scoped
// GgufDecoder contract, visit_subgraph hands the translator a per-node view by cloning
// this decoder with a fixed node index (m_node_idx); every per-node accessor then reads
// the node it is bound to, so no node index is threaded through the interface.
class GgufBuilderDecoder : public GgufDecoder {
public:
    explicit GgufBuilderDecoder(std::shared_ptr<GgufGraph> graph);

    // Per-node accessors (bound to the node this decoder instance was cloned for).
    ov::Any get_attribute(const std::string& name) const override;
    PartialShape get_input_shape(const std::string& name) const override;
    std::vector<size_t> get_input_stride(const std::string& name) const override;
    int64_t get_input_view_offset(const std::string& name) const override;
    element::Type get_input_type(const std::string& name) const override;
    size_t get_input_size() const override;
    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override;
    std::vector<std::string> get_input_names() const override;
    PartialShape get_output_shape() const override;
    element::Type get_output_type() const override;
    std::vector<std::string> get_output_names() const override;
    const std::string& get_op_type() const override;
    const std::string& get_op_name() const override;
    int get_op_case() const override;

    void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> node_visitor) const override;

    // Model-level I/O.
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const override;
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const override;
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const override;
    std::vector<std::string> get_model_output_names() const override;

    std::map<std::string, std::string> get_kv_param_res_names() const override;

    bool is_static() const override;
    bool is_stateful() const override;
    bool is_swa_layer(int layer) const override;
    const ov::AnyMap& get_tokenizer_config() const override;

private:
    std::shared_ptr<GgufGraph> m_graph;
    // Index of the node this decoder instance is bound to. -1 for the whole-graph decoder
    // (only model-scope queries are valid on it).
    int m_node_idx = -1;

    const GgufOp& node() const;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
