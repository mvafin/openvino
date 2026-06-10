// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ggml_graph.hpp"
#include "openvino/frontend/ggml/decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

// GgmlDecoder implementation over a GgmlGraph built natively from a .gguf file (no
// llama.cpp / ggml dependency). It is the OpenVINO-side counterpart of llama.cpp's
// cgraph-backed GgmlOvDecoder: both feed the same op translators / TranslateSession.
//
// The whole-graph constructor wraps a complete GgmlGraph. visit_subgraph hands the
// translator a per-node view by cloning this decoder with a fixed node index.
class GgufBuilderDecoder : public GgmlDecoder {
public:
    explicit GgufBuilderDecoder(std::shared_ptr<GgmlGraph> graph);

    // DecoderBase overrides (whole-op queries use the node this decoder is bound to).
    ov::Any get_attribute(const std::string& name) const override;
    size_t get_input_size() const override;
    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override;
    const std::string& get_op_type() const override;
    const std::string& get_op_name() const override;

    // Per-node typed attribute access.
    ov::Any get_attribute(int node_idx, const std::string& name) const override;

    // Per-input metadata.
    PartialShape get_input_shape(int node_idx, const std::string& name) const override;
    std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const override;
    int64_t get_input_view_offset(int node_idx, const std::string& name) const override;
    element::Type get_input_type(int node_idx, const std::string& name) const override;
    size_t get_input_size(int node_idx) const override;
    std::vector<std::string> get_input_names(int node_idx) const override;

    // Per-node output metadata.
    PartialShape get_output_shape(int node_idx) const override;
    element::Type get_output_type(int node_idx) const override;
    std::vector<std::string> get_output_names(int node_idx) const override;

    // Op type / name.
    const std::string& get_op_type(int node_idx) const override;
    const std::string& get_op_name(int node_idx) const override;

    void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const override;

    int get_op_case(int node_idx) const override;

    // Model-level I/O.
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const override;
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const override;
    const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const override;
    std::vector<std::string> get_model_output_names() const override;

    bool has_rope() const override;
    RopeConfig get_rope_config() const override;

    std::map<std::string, std::string> get_kv_param_res_names() const override;

    bool is_static() const override;
    bool is_stateful() const override;
    int is_swa_layer(int layer) const override;

private:
    std::shared_ptr<GgmlGraph> m_graph;
    // Index of the node this decoder instance is bound to (for the no-arg op queries used
    // through DecoderBase / NodeContext). -1 for the whole-graph decoder.
    int m_node_idx = -1;

    const GgmlOp& node(int node_idx) const;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
