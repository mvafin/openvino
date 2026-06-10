// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gguf_builder_decoder.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov {
namespace frontend {
namespace ggml {

GgufBuilderDecoder::GgufBuilderDecoder(std::shared_ptr<GgmlGraph> graph) : m_graph(std::move(graph)) {}

const GgmlOp& GgufBuilderDecoder::node(int node_idx) const {
    OPENVINO_ASSERT(node_idx >= 0 && static_cast<size_t>(node_idx) < m_graph->nodes.size(),
                    "[ggml] node index out of range: ",
                    node_idx);
    return m_graph->nodes[node_idx];
}

// ---- DecoderBase whole-op queries (bound to m_node_idx) ----

ov::Any GgufBuilderDecoder::get_attribute(const std::string& name) const {
    return get_attribute(m_node_idx, name);
}

size_t GgufBuilderDecoder::get_input_size() const {
    return get_input_size(m_node_idx);
}

void GgufBuilderDecoder::get_input_node(size_t,
                                        std::string&,
                                        std::string&,
                                        size_t&) const {
    // Not used by the ggml translators (they pull inputs by name through NodeContext).
}

const std::string& GgufBuilderDecoder::get_op_type() const {
    return get_op_type(m_node_idx);
}

const std::string& GgufBuilderDecoder::get_op_name() const {
    return get_op_name(m_node_idx);
}

// ---- Per-node typed attribute ----

ov::Any GgufBuilderDecoder::get_attribute(int node_idx, const std::string& name) const {
    const auto& attrs = node(node_idx).attributes;
    auto it = attrs.find(name);
    if (it == attrs.end()) {
        return {};
    }
    return it->second;
}

// ---- Per-input metadata ----

PartialShape GgufBuilderDecoder::get_input_shape(int node_idx, const std::string& name) const {
    const auto& m = node(node_idx).input_shapes;
    auto it = m.find(name);
    OPENVINO_ASSERT(it != m.end(), "[ggml] no input shape for '", name, "'");
    return it->second;
}

std::vector<size_t> GgufBuilderDecoder::get_input_stride(int node_idx, const std::string& name) const {
    const auto& m = node(node_idx).input_strides;
    auto it = m.find(name);
    OPENVINO_ASSERT(it != m.end(), "[ggml] no input stride for '", name, "'");
    return it->second;
}

int64_t GgufBuilderDecoder::get_input_view_offset(int node_idx, const std::string& name) const {
    const auto& m = node(node_idx).input_view_offsets;
    auto it = m.find(name);
    return it == m.end() ? 0 : it->second;
}

element::Type GgufBuilderDecoder::get_input_type(int node_idx, const std::string& name) const {
    const auto& m = node(node_idx).input_types;
    auto it = m.find(name);
    OPENVINO_ASSERT(it != m.end(), "[ggml] no input type for '", name, "'");
    return it->second;
}

size_t GgufBuilderDecoder::get_input_size(int node_idx) const {
    return node(node_idx).input_names.size();
}

std::vector<std::string> GgufBuilderDecoder::get_input_names(int node_idx) const {
    return node(node_idx).input_names;
}

// ---- Per-node output metadata ----

PartialShape GgufBuilderDecoder::get_output_shape(int node_idx) const {
    return node(node_idx).output_shape;
}

element::Type GgufBuilderDecoder::get_output_type(int node_idx) const {
    return node(node_idx).output_type;
}

std::vector<std::string> GgufBuilderDecoder::get_output_names(int node_idx) const {
    return {node(node_idx).output_name};
}

// ---- Op type / name ----

const std::string& GgufBuilderDecoder::get_op_type(int node_idx) const {
    return node(node_idx).op_type;
}

const std::string& GgufBuilderDecoder::get_op_name(int node_idx) const {
    return node(node_idx).name;
}

void GgufBuilderDecoder::visit_subgraph(
    std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const {
    for (size_t i = 0; i < m_graph->nodes.size(); i++) {
        auto per_node = std::make_shared<GgufBuilderDecoder>(*this);
        per_node->m_node_idx = static_cast<int>(i);
        node_visitor(per_node, static_cast<int>(i));
    }
}

int GgufBuilderDecoder::get_op_case(int node_idx) const {
    return node(node_idx).op_case;
}

// ---- Model-level I/O ----

const std::map<std::string, std::shared_ptr<ov::Node>>& GgufBuilderDecoder::get_model_inputs() const {
    return m_graph->model_inputs;
}

const std::map<std::string, std::shared_ptr<ov::Node>>& GgufBuilderDecoder::get_model_extra_inputs() const {
    return m_graph->model_extra_inputs;
}

const std::map<std::string, std::shared_ptr<ov::Node>>& GgufBuilderDecoder::get_model_weights() const {
    return m_graph->model_weights;
}

std::vector<std::string> GgufBuilderDecoder::get_model_output_names() const {
    return m_graph->model_output_names;
}

bool GgufBuilderDecoder::has_rope() const {
    return m_graph->has_rope;
}

RopeConfig GgufBuilderDecoder::get_rope_config() const {
    return m_graph->rope_config;
}

std::map<std::string, std::string> GgufBuilderDecoder::get_kv_param_res_names() const {
    return m_graph->kv_param_res_names;
}

bool GgufBuilderDecoder::is_static() const {
    return m_graph->is_static;
}

bool GgufBuilderDecoder::is_stateful() const {
    return m_graph->is_stateful;
}

int GgufBuilderDecoder::is_swa_layer(int) const {
    return 0;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
