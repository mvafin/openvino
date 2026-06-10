// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF -> GgmlGraph builders. Each architecture builder emits nodes in the GGML op
// vocabulary that reproduce llama.cpp's cgraph topology, so the resulting GgmlGraph drives
// the same op translators as the llama.cpp cgraph path. No llama.cpp / ggml dependency.
//
// Ground truth for the qwen3 topology: llama.cpp src/models/qwen3.cpp and the
// build_norm / build_qkv / build_attn / build_attn_mha / build_ffn expansions in
// src/llama-graph.cpp, plus the KV-cache cpy_k/get_k (SET_ROWS + VIEW) in
// src/llama-kv-cache.cpp. Op-case values follow ggml-decoder.cpp::compute_op_case.

#include "gguf_builder.hpp"

#include <cmath>
#include <memory>

#include "gguf.hpp"
#include "ggml_graph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace frontend {
namespace ggml {

namespace {

// ggml ROPE op-case bit for NEOX mode (see ggml-decoder.cpp::compute_op_case). qwen3 uses
// NEOX rope. The low bits encode whether the rope input is a VIEW (not the case here).
constexpr int ROPE_OP_CASE_NEOX = 0x00010000;

// Shapes are kept in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0] that the
// decoder's get_shape() produces (reverse of GGUF on-disk order). The translators consume
// these shapes verbatim.

// Helper accumulating GgmlOp nodes plus the model-level I/O of a GgmlGraph.
class GraphBuilder {
public:
    GraphBuilder(const std::map<std::string, GGUFMetaData>& config,
                 std::unordered_map<std::string, ov::Tensor>& weights,
                 std::unordered_map<std::string, gguf_tensor_type>& qtypes) :
        m_config(config),
        m_weights(weights),
        m_qtypes(qtypes) {
        m_graph = std::make_shared<GgmlGraph>();
        m_graph->is_static = false;
        m_graph->is_stateful = true;
    }

    std::shared_ptr<GgmlGraph> graph() {
        return m_graph;
    }

    int cfg_int(const std::string& k) const {
        return std::get<int>(m_config.at(k));
    }
    float cfg_float(const std::string& k) const {
        return std::get<float>(m_config.at(k));
    }

    // Register a weight Constant (built from the dequantized/raw gguf tensor) under its raw
    // ggml name. For quantized weights the (.weight,.scales,.biases) triple is combined
    // into a decompressed f16 constant here (kept simple for the first bring-up; a later
    // step can emit the compressed u4/u8 + scale subgraph to match the cgraph exactly).
    void add_weight(const std::string& ggml_name) {
        if (m_graph->model_weights.count(ggml_name)) {
            return;
        }
        auto it = m_weights.find(ggml_name);
        OPENVINO_ASSERT(it != m_weights.end(), "[ggml] weight not found in gguf: ", ggml_name);
        auto cnst = std::make_shared<ov::op::v0::Constant>(it->second);
        cnst->set_friendly_name(ggml_name);
        cnst->output(0).set_names({ggml_name});
        m_graph->model_weights[ggml_name] = cnst;
    }

    // Register a model input Parameter.
    void add_input(const std::string& name, ov::element::Type type, const ov::PartialShape& shape) {
        auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
        p->set_friendly_name(name);
        p->output(0).set_names({name});
        m_graph->model_inputs[name] = p;
    }

    std::shared_ptr<GgmlGraph> m_graph;

private:
    const std::map<std::string, GGUFMetaData>& m_config;
    std::unordered_map<std::string, ov::Tensor>& m_weights;
    std::unordered_map<std::string, gguf_tensor_type>& m_qtypes;
};

}  // namespace

std::shared_ptr<GgmlGraph> build_ggml_graph_from_gguf(const std::string& file) {
    auto [metadata, weights, qtypes] = get_gguf_data(file);
    auto config = config_from_meta(metadata);

    const std::string arch = std::get<std::string>(config.at("architecture"));
    OPENVINO_ASSERT(arch == "qwen3" || arch == "qwen2" || arch == "llama",
                    "[ggml] native GGUF builder does not yet support architecture '",
                    arch,
                    "'");

    GraphBuilder b(config, weights, qtypes);

    // NOTE: This is the scaffolding entry point. The full qwen3 node emission
    // (embed -> per-layer attn/ffn -> norm -> lm_head, matching llama.cpp's cgraph)
    // is built up incrementally in the steps that follow.
    b.m_graph->has_rope = true;
    b.m_graph->rope_config.n_dims = config.count("head_size") ? std::get<int>(config.at("head_size")) : 0;
    b.m_graph->rope_config.n_ctx_orig = std::get<int>(config.at("max_position_embeddings"));
    b.m_graph->rope_config.freq_base = std::get<float>(config.at("rope_freq_base"));
    b.m_graph->rope_config.freq_scale = 1.0f;
    b.m_graph->rope_config.attn_factor = 1.0f;
    b.m_graph->rope_config.beta_fast = 32.0f;
    b.m_graph->rope_config.beta_slow = 1.0f;

    return b.graph();
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
