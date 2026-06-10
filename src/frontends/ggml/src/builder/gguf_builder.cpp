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
//
// This first bring-up targets the dynamic (non-NPU) stateful graph for a non-SWA model
// (qwen3-0.6B/8B): is_static=false, is_stateful=true.

#include "gguf_builder.hpp"

#include <cmath>
#include <memory>

#include "gguf.hpp"
#include "ggml_graph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "weights.hpp"

namespace ov {
namespace frontend {
namespace ggml {

namespace {

// ggml ROPE op-case bit for NEOX mode (see ggml-decoder.cpp::compute_op_case). qwen3 uses
// NEOX rope; the input is not a VIEW here so the low bits stay 0.
constexpr int ROPE_OP_CASE_NEOX = 0x00010000;

// Shapes are kept in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0] (reverse of GGUF
// on-disk order), matching the decoder's get_shape(). The translators consume them as-is.
// Token length is dynamic (-1) in dim ne1.

class Qwen3Builder {
public:
    Qwen3Builder(const std::map<std::string, GGUFMetaData>& config,
                 std::unordered_map<std::string, ov::Tensor>& weights,
                 std::unordered_map<std::string, gguf_tensor_type>& qtypes) :
        m_config(config),
        m_weights(weights),
        m_qtypes(qtypes) {
        m_graph = std::make_shared<GgmlGraph>();
        m_graph->is_static = false;
        m_graph->is_stateful = true;

        m_n_layer = cfg_int("layer_num");
        m_n_head = cfg_int("head_num");
        m_n_head_kv = cfg_int("head_num_kv");
        m_head_size = cfg_int("head_size");
        m_n_embd = cfg_int("hidden_size");
        m_rms_eps = cfg_float("rms_norm_eps");

        m_graph->has_rope = true;
        m_graph->rope_config.n_dims = m_head_size;
        m_graph->rope_config.n_ctx_orig = cfg_int("max_position_embeddings");
        m_graph->rope_config.freq_base = cfg_float("rope_freq_base");
        m_graph->rope_config.freq_scale = 1.0f;
        m_graph->rope_config.ext_factor = 0.0f;
        m_graph->rope_config.attn_factor = 1.0f;
        m_graph->rope_config.beta_fast = 32.0f;
        m_graph->rope_config.beta_slow = 1.0f;
    }

    std::shared_ptr<GgmlGraph> build();

private:
    int cfg_int(const std::string& k) const {
        return std::get<int>(m_config.at(k));
    }
    float cfg_float(const std::string& k) const {
        return std::get<float>(m_config.at(k));
    }

    static ov::PartialShape ps(std::vector<int64_t> dims) {
        return ov::PartialShape(dims);
    }

    // `ggml_name` is the full tensor name ending in ".weight" (the name translators
    // reference). make_weight_node takes the base (without ".weight").
    void add_weight(const std::string& ggml_name) {
        if (m_graph->model_weights.count(ggml_name)) {
            return;
        }
        const std::string suffix = ".weight";
        const std::string base = (ggml_name.size() > suffix.size() &&
                                  ggml_name.compare(ggml_name.size() - suffix.size(), suffix.size(), suffix) == 0)
                                     ? ggml_name.substr(0, ggml_name.size() - suffix.size())
                                     : ggml_name;
        m_graph->model_weights[ggml_name] = make_weight_node(base, m_weights, m_qtypes);

        // Record the weight's shape padded to the decoder's 4D reversed layout
        // [1, 1, rows, cols] so add_op can fill per-input shape/type for translators
        // (MUL_MAT) that index dims [1] and [3]. The exact `cols` value (packed) is not
        // used in the batch-1 path; only rank and dim[1]==1 matter here.
        auto it = m_weights.find(ggml_name);
        if (it != m_weights.end()) {
            const auto& s = it->second.get_shape();  // [rows, cols(packed)]
            int64_t rows = s.size() >= 1 ? static_cast<int64_t>(s[0]) : 1;
            int64_t cols = s.size() >= 2 ? static_cast<int64_t>(s[1]) : 1;
            m_tensor_shapes[ggml_name] = ov::PartialShape({1, 1, rows, cols});
            m_tensor_types[ggml_name] = ov::element::f32;
        }
    }

    std::shared_ptr<ov::op::v0::Parameter> add_input(const std::string& name,
                                                     ov::element::Type type,
                                                     const ov::PartialShape& shape) {
        auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
        p->set_friendly_name(name);
        p->output(0).set_names({name});
        m_graph->model_inputs[name] = p;
        return p;
    }

    void add_extra_input(const std::string& name, int64_t value) {
        auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {value});
        c->set_friendly_name(name);
        m_graph->model_extra_inputs[name] = c;
    }

    // Append one op node. `inputs` are producer tensor names (weights / model inputs /
    // earlier node outputs). Returns the output tensor name (== node name).
    std::string add_op(const std::string& op_type,
                       const std::string& name,
                       const std::vector<std::string>& inputs,
                       const ov::PartialShape& out_shape,
                       ov::element::Type out_type,
                       int op_case = 0,
                       std::map<std::string, ov::Any> attrs = {}) {
        GgmlOp op;
        op.op_type = op_type;
        op.name = name;
        op.input_names = inputs;
        op.output_name = name;
        op.output_shape = out_shape;
        op.output_type = out_type;
        op.op_case = op_case;
        op.attributes = std::move(attrs);
        // Fill per-input shape/type from known producers so translators that query them
        // (MUL_MAT, RESHAPE) get sane values.
        for (const auto& in : inputs) {
            if (auto it = m_tensor_shapes.find(in); it != m_tensor_shapes.end()) {
                op.input_shapes[in] = it->second;
            }
            if (auto it = m_tensor_types.find(in); it != m_tensor_types.end()) {
                op.input_types[in] = it->second;
            }
        }
        m_tensor_shapes[name] = out_shape;
        m_tensor_types[name] = out_type;
        m_graph->nodes.push_back(std::move(op));
        return name;
    }

    // RMS_NORM followed by elementwise MUL with the norm weight (build_norm, LLM_NORM_RMS).
    std::string rms_norm(const std::string& in, const std::string& weight, const std::string& out_prefix) {
        add_weight(weight);
        auto norm = add_op("GGML_OP_RMS_NORM",
                           out_prefix + ".rms",
                           {in},
                           m_tensor_shapes.at(in),
                           ov::element::f32,
                           0,
                           {{"eps", m_rms_eps}});
        m_tensor_shapes[weight] = m_tensor_shapes.count(weight) ? m_tensor_shapes[weight] : ov::PartialShape{};
        return add_op("GGML_OP_MUL", out_prefix, {norm, weight}, m_tensor_shapes.at(in), ov::element::f32);
    }

    const std::map<std::string, GGUFMetaData>& m_config;
    std::unordered_map<std::string, ov::Tensor>& m_weights;
    std::unordered_map<std::string, gguf_tensor_type>& m_qtypes;
    std::shared_ptr<GgmlGraph> m_graph;

    int m_n_layer = 0, m_n_head = 0, m_n_head_kv = 0, m_head_size = 0, m_n_embd = 0;
    float m_rms_eps = 0.0f;

    std::map<std::string, ov::PartialShape> m_tensor_shapes;
    std::map<std::string, ov::element::Type> m_tensor_types;
};

std::shared_ptr<GgmlGraph> Qwen3Builder::build() {
    using ov::element::f32;
    using ov::element::i64;
    const int64_t D = -1;  // dynamic token length, used for model-input Parameters only

    // Per-node output shapes are STATIC, like the cgraph decoder (which builds the graph
    // for a concrete token length). We use a representative token length T; the translators
    // emit dynamic reshapes (-1 / 0) where needed, and MakeStateful + the dynamic input
    // Parameters carry the real dynamic-ness. T affects only the per-node shape metadata.
    const int64_t T = 1;

    // ---- Model inputs (names match the cgraph decoder's get_graph_input_ov_name) ----
    add_input("inp_tokens", i64, ps({1, 1, 1, D}));
    add_input("inp_pos", i64, ps({1, 1, 1, D}));
    add_input("inp_out_ids", i64, ps({1, 1, 1, D}));
    add_input("self_kq_mask", ov::element::f32, ps({1, 1, D, D}));

    // ---- Embedding: GET_ROWS(token_embd.weight, inp_tokens) -> "embd" ----
    add_weight("token_embd.weight");
    m_tensor_shapes["inp_tokens"] = ps({1, 1, 1, T});
    std::string cur = add_op("GGML_OP_GET_ROWS",
                             "embd",
                             {"token_embd.weight", "inp_tokens"},
                             ps({1, 1, T, m_n_embd}),
                             f32);

    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(m_head_size));

    for (int il = 0; il < m_n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const std::string inpSA = cur;

        // attn_norm
        std::string attn_norm = rms_norm(cur, p + "attn_norm.weight", p + "attn_norm");

        // Q/K/V projections: MUL_MAT(w, attn_norm), then conceptual reshape to heads.
        add_weight(p + "attn_q.weight");
        add_weight(p + "attn_k.weight");
        add_weight(p + "attn_v.weight");
        auto q = add_op("GGML_OP_MUL_MAT", p + "Qcur", {p + "attn_q.weight", attn_norm},
                        ps({1, 1, T, m_head_size * m_n_head}), f32);
        auto k = add_op("GGML_OP_MUL_MAT", p + "Kcur", {p + "attn_k.weight", attn_norm},
                        ps({1, 1, T, m_head_size * m_n_head_kv}), f32);
        auto v = add_op("GGML_OP_MUL_MAT", p + "Vcur", {p + "attn_v.weight", attn_norm},
                        ps({1, 1, T, m_head_size * m_n_head_kv}), f32);

        // reshape Q/K/V to [1, n_tokens, n_head(_kv), head_size]
        q = add_op("GGML_OP_RESHAPE", p + "Qcur_r", {q}, ps({1, T, m_n_head, m_head_size}), f32, 1);
        k = add_op("GGML_OP_RESHAPE", p + "Kcur_r", {k}, ps({1, T, m_n_head_kv, m_head_size}), f32, 1);
        v = add_op("GGML_OP_RESHAPE", p + "Vcur_r", {v}, ps({1, T, m_n_head_kv, m_head_size}), f32, 1);

        // per-head q_norm / k_norm (qwen3)
        q = rms_norm(q, p + "attn_q_norm.weight", p + "Qcur_normed");
        k = rms_norm(k, p + "attn_k_norm.weight", p + "Kcur_normed");

        // RoPE (NEOX)
        q = add_op("GGML_OP_ROPE", p + "Qcur_rope", {q, "inp_pos"}, ps({1, T, m_n_head, m_head_size}), f32,
                   ROPE_OP_CASE_NEOX, {{"rope_config", m_graph->rope_config}});
        k = add_op("GGML_OP_ROPE", p + "Kcur_rope", {k, "inp_pos"}, ps({1, T, m_n_head_kv, m_head_size}), f32,
                   ROPE_OP_CASE_NEOX, {{"rope_config", m_graph->rope_config}});

        // FLASH_ATTN_EXT(q, k, v, mask, scale) -> [1, n_tokens, n_head, head_size]
        auto attn = add_op("GGML_OP_FLASH_ATTN_EXT", p + "kqv", {q, k, v, "self_kq_mask"},
                           ps({1, T, m_n_head, m_head_size}), f32, 0, {{"scale", kq_scale}});

        // reshape back to [1, 1, n_tokens, n_head*head_size]
        auto attn_2d = add_op("GGML_OP_RESHAPE", p + "kqv_merged", {attn},
                              ps({1, 1, T, m_n_head * m_head_size}), f32, 2);

        // output projection
        add_weight(p + "attn_output.weight");
        auto attn_out = add_op("GGML_OP_MUL_MAT", p + "attn_out", {p + "attn_output.weight", attn_2d},
                               ps({1, 1, T, m_n_embd}), f32);

        std::string sa = inpSA;
        std::string ao = attn_out;
        if (il == m_n_layer - 1) {
            ao = add_op("GGML_OP_GET_ROWS", p + "attn_out_g", {attn_out, "inp_out_ids"},
                        ps({1, 1, T, m_n_embd}), f32);
            sa = add_op("GGML_OP_GET_ROWS", p + "inpSA_g", {inpSA, "inp_out_ids"},
                        ps({1, 1, T, m_n_embd}), f32);
        }

        auto ffn_inp = add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa}, ps({1, 1, T, m_n_embd}), f32);

        // ffn_norm
        auto ffn_norm = rms_norm(ffn_inp, p + "ffn_norm.weight", p + "ffn_norm");

        // SwiGLU FFN: swiglu(gate, up) then down. ggml uses ggml_swiglu_split(gate, up).
        add_weight(p + "ffn_gate.weight");
        add_weight(p + "ffn_up.weight");
        add_weight(p + "ffn_down.weight");
        const int n_ff = static_cast<int>(m_weights.at(p + "ffn_gate.weight").get_shape()[0]);
        auto gate = add_op("GGML_OP_MUL_MAT", p + "ffn_gate", {p + "ffn_gate.weight", ffn_norm},
                           ps({1, 1, T, n_ff}), f32);
        auto up = add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm},
                         ps({1, 1, T, n_ff}), f32);
        auto glu = add_op("GGML_GLU_OP_SWIGLU", p + "ffn_swiglu", {gate, up}, ps({1, 1, T, n_ff}), f32, 0,
                          {{"swapped", false}});
        auto down = add_op("GGML_OP_MUL_MAT", p + "ffn_out", {p + "ffn_down.weight", glu},
                           ps({1, 1, T, m_n_embd}), f32);

        cur = add_op("GGML_OP_ADD", p + "l_out", {down, ffn_inp}, ps({1, 1, T, m_n_embd}), f32);
    }

    // final norm + lm_head
    cur = rms_norm(cur, "output_norm.weight", "result_norm");

    const std::string lm_head_w = m_weights.count("output.weight") ? "output.weight" : "token_embd.weight";
    add_weight(lm_head_w);
    const int n_vocab = static_cast<int>(m_weights.at(lm_head_w).get_shape()[0]) *
                        (lm_head_w == "output.weight" ? 1 : 1);  // rows = vocab
    auto logits = add_op("GGML_OP_MUL_MAT", "result_output", {lm_head_w, cur},
                         ps({1, 1, T, n_vocab}), f32);

    m_graph->model_output_names = {logits};
    return m_graph;
}

}  // namespace

std::shared_ptr<GgmlGraph> build_ggml_graph_from_gguf(const std::string& file) {
    auto [metadata, weights, qtypes] = get_gguf_data(file);
    auto config = config_from_meta(metadata);

    const std::string arch = std::get<std::string>(config.at("architecture"));
    OPENVINO_ASSERT(arch == "qwen3",
                    "[ggml] native GGUF builder currently supports only 'qwen3', got '",
                    arch,
                    "'");

    Qwen3Builder builder(config, weights, qtypes);
    return builder.build();
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
