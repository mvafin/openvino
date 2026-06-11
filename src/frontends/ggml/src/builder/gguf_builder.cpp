// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF -> GgmlGraph builder. Emits nodes in the GGML op vocabulary that reproduce
// llama.cpp's cgraph topology, so the resulting GgmlGraph drives the same op translators as
// the llama.cpp cgraph path. No llama.cpp / ggml dependency.
//
// One generic dense-transformer builder covers the whole "llama-family" of architectures
// (llama-3, qwen2/2.5, qwen3, phi-3, minicpm, hunyuan-dense, ...). Per-architecture
// differences are derived almost entirely from the GGUF itself:
//   - presence of per-head q/k norm weights  (blk.N.attn_{q,k}_norm.weight)  -> qwen3, hunyuan
//   - presence of q/k/v projection biases     (blk.N.attn_{q,k,v}.bias)       -> qwen2/2.5
//   - presence of an output-projection bias   (blk.N.attn_output.bias)
//   - presence of rope frequency factors      (rope_freqs.weight)            -> llama-3, phi-3
//   - scalar scales from metadata             (embedding/residual/logit/attention scale)
//                                                                            -> minicpm
// so adding a new architecture of this family is just adding its name to kSupportedArchs.
// Structurally novel families (MoE routing, SSM, fused-QKV-only, etc.) still need new code.
//
// Ground truth for the topology: llama.cpp src/models/{llama,qwen2,qwen3,phi3,minicpm,
// hunyuan-*}.cpp and the build_norm / build_qkv / build_attn / build_attn_mha / build_ffn
// expansions in src/llama-graph.cpp, plus the KV-cache cpy_k/get_k (SET_ROWS + VIEW) in
// src/llama-kv-cache.cpp. Op-case values follow ggml-decoder.cpp::compute_op_case.
//
// Targets the dynamic (non-NPU) stateful graph for non-SWA models: is_static=false,
// is_stateful=true.

#include "gguf_builder.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <set>

#include "gguf.hpp"
#include "ggml_graph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "weights.hpp"

namespace ov {
namespace frontend {
namespace ggml {

namespace {

// ggml ROPE op-case (see ggml-decoder.cpp::compute_op_case). The high 16 bits encode the
// mode: NORMAL=0 (llama/minicpm: rotate consecutive pairs), NEOX=1 (qwen/phi/hunyuan:
// rotate halves). The input is not a VIEW here so the low bits stay 0.
constexpr int ROPE_OP_CASE_NORMAL = 0x00000000;
constexpr int ROPE_OP_CASE_NEOX = 0x00010000;

// Architectures whose rope is NEOX (rotate-halves); everything else in the supported set
// uses NORMAL (rotate consecutive pairs). Mirrors llama_model_rope_type.
bool arch_uses_neox_rope(const std::string& arch) {
    return arch == "qwen2" || arch == "qwen3" || arch == "phi3" || arch == "hunyuan-dense";
}

// Shapes are kept in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0] (reverse of GGUF
// on-disk order), matching the decoder's get_shape(). The translators consume them as-is.
// Token length is dynamic (-1) in dim ne1.

class TransformerBuilder {
public:
    TransformerBuilder(const std::map<std::string, GGUFMetaData>& config,
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

        // Per-architecture structure, auto-detected from the GGUF tensor table (layer 0).
        m_has_qk_norm = m_weights.count("blk.0.attn_q_norm.weight") > 0;
        m_has_fused_qkv = m_weights.count("blk.0.attn_qkv.weight") > 0;  // phi-3, minicpm
        m_has_qkv_bias = m_weights.count("blk.0.attn_q.bias") > 0 || m_weights.count("blk.0.attn_qkv.bias") > 0;
        m_has_attn_out_bias = m_weights.count("blk.0.attn_output.bias") > 0;
        m_has_rope_freqs = m_weights.count("rope_freqs.weight") > 0;

        // Per-architecture scalars from metadata (1.0 / 0.0 when absent -> no-op).
        m_embedding_scale = cfg_float("embedding_scale");
        m_residual_scale = cfg_float("residual_scale");
        m_logit_scale = cfg_float("logit_scale");
        m_attention_scale = cfg_float("attention_scale");  // 0 -> 1/sqrt(head_size)

        const std::string arch = std::get<std::string>(config.at("architecture"));
        m_rope_op_case = arch_uses_neox_rope(arch) ? ROPE_OP_CASE_NEOX : ROPE_OP_CASE_NORMAL;

        m_graph->has_rope = true;
        m_graph->rope_config.n_dims = cfg_int("rope_dimension_count");
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

    // Split a fused blk.<il>.attn_qkv weight into separate q/k/v weight nodes registered
    // under blk.<il>.attn_{q,k,v}.weight, so the rest of the builder is identical to the
    // separate-projection case. Rows split as [n_head*head_size | n_head_kv*head_size x2].
    void register_fused_qkv(int il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const size_t n_q = static_cast<size_t>(m_n_head) * m_head_size;
        const size_t n_kv = static_cast<size_t>(m_n_head_kv) * m_head_size;
        auto qkv = make_fused_qkv_weights(p + "attn_qkv", m_weights, m_qtypes, n_q, n_kv, n_kv);
        const std::array<std::string, 3> names = {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"};
        const std::array<int64_t, 3> rows = {(int64_t)n_q, (int64_t)n_kv, (int64_t)n_kv};
        for (size_t i = 0; i < 3; ++i) {
            qkv[i]->set_friendly_name(names[i]);
            m_graph->model_weights[names[i]] = qkv[i];
            m_tensor_shapes[names[i]] = ps({1, 1, rows[i], m_n_embd});
            m_tensor_types[names[i]] = ov::element::f32;
        }
    }

    // Register a plain (non-quantized, f32) weight stored under its full GGUF name, e.g. a
    // bias tensor "blk.N.attn_q.bias" (no ".weight" suffix). Records its 4D shape.
    void add_named_weight(const std::string& ggml_name) {
        if (m_graph->model_weights.count(ggml_name)) {
            return;
        }
        auto it = m_weights.find(ggml_name);
        OPENVINO_ASSERT(it != m_weights.end(), "[ggml] weight not found in gguf: ", ggml_name);
        const ov::Tensor& w = it->second;
        std::shared_ptr<ov::Node> node = std::make_shared<ov::op::v0::Constant>(w);
        if (w.get_element_type() != ov::element::f32) {
            node = std::make_shared<ov::op::v0::Convert>(node, ov::element::f32);
        }
        node->set_friendly_name(ggml_name);
        m_graph->model_weights[ggml_name] = node;
        const auto& s = w.get_shape();
        int64_t n = s.empty() ? 1 : static_cast<int64_t>(s[0]);
        m_tensor_shapes[ggml_name] = ps({1, 1, 1, n});
        m_tensor_types[ggml_name] = ov::element::f32;
    }

    // Elementwise add of a (broadcast) bias weight: GGML_OP_ADD(x, bias_weight).
    std::string add_bias(const std::string& x, const std::string& bias_weight, const std::string& name) {
        add_named_weight(bias_weight);
        return add_op("GGML_OP_ADD", name, {x, bias_weight}, m_tensor_shapes.at(x), ov::element::f32);
    }

    // Scale a tensor by a constant: GGML_OP_SCALE with attr "scale" (and bias 0).
    std::string scale(const std::string& x, float factor, const std::string& name) {
        return add_op("GGML_OP_SCALE", name, {x}, m_tensor_shapes.at(x), ov::element::f32, 0,
                      {{"scale", factor}, {"bias", 0.0f}});
    }

    const std::map<std::string, GGUFMetaData>& m_config;
    std::unordered_map<std::string, ov::Tensor>& m_weights;
    std::unordered_map<std::string, gguf_tensor_type>& m_qtypes;
    std::shared_ptr<GgmlGraph> m_graph;

    int m_n_layer = 0, m_n_head = 0, m_n_head_kv = 0, m_head_size = 0, m_n_embd = 0;
    float m_rms_eps = 0.0f;

    // auto-detected per-architecture structure
    bool m_has_qk_norm = false, m_has_qkv_bias = false, m_has_attn_out_bias = false, m_has_rope_freqs = false;
    bool m_has_fused_qkv = false;
    float m_embedding_scale = 1.0f, m_residual_scale = 1.0f, m_logit_scale = 1.0f, m_attention_scale = 0.0f;
    int m_rope_op_case = ROPE_OP_CASE_NEOX;

    std::map<std::string, ov::PartialShape> m_tensor_shapes;
    std::map<std::string, ov::element::Type> m_tensor_types;
};

std::shared_ptr<GgmlGraph> TransformerBuilder::build() {
    using ov::element::f32;
    using ov::element::i64;
    const int64_t D = -1;  // dynamic token length, used for model-input Parameters only

    // Per-node output shapes are STATIC, like the cgraph decoder (which builds the graph
    // for a concrete token length). We use a representative token length T; the translators
    // emit dynamic reshapes (-1 / 0) where needed, and MakeStateful + the dynamic input
    // Parameters carry the real dynamic-ness. T affects only the per-node shape metadata.
    const int64_t T = 1;

    using ov::element::i32;
    // ---- Model inputs (names match the cgraph decoder's get_graph_input_ov_name) ----
    // ggml uses i32 for token/position/index inputs.
    add_input("inp_tokens", i32, ps({1, 1, 1, D}));
    add_input("inp_pos", i32, ps({1, 1, 1, D}));
    add_input("inp_out_ids", i32, ps({1, 1, 1, D}));
    add_input("self_kq_mask", ov::element::f32, ps({1, 1, D, D}));
    // KV-cache update index (consumed by SET_ROWS; unused in the stateful Concat branch).
    add_input("inp_kv_idx", i32, ps({1, 1, 1, D}));
    m_tensor_shapes["inp_kv_idx"] = ps({1, 1, 1, T});
    m_tensor_types["inp_kv_idx"] = i32;

    // token_len_per_seq: number of new tokens per sequence; used by TranslateSession's mask
    // slicing (add_sliced_mask) to build KQ_mask_sliced. An extra (Parameter) input.
    {
        auto p = std::make_shared<ov::op::v0::Parameter>(i64, ps({1}));
        p->set_friendly_name("token_len_per_seq");
        p->output(0).set_names({"token_len_per_seq"});
        m_graph->model_extra_inputs["token_len_per_seq"] = p;
    }

    // ---- Embedding: GET_ROWS(token_embd.weight, inp_tokens) -> "embd" ----
    add_weight("token_embd.weight");
    m_tensor_shapes["inp_tokens"] = ps({1, 1, 1, T});
    std::string cur = add_op("GGML_OP_GET_ROWS",
                             "embd",
                             {"token_embd.weight", "inp_tokens"},
                             ps({1, 1, T, m_n_embd}),
                             f32);
    // MiniCPM scales the embeddings by a constant.
    if (m_embedding_scale != 1.0f) {
        cur = scale(cur, m_embedding_scale, "embd_scaled");
    }

    // rope freq factors (llama-3 long context, phi-3): an optional 3rd ROPE input.
    if (m_has_rope_freqs) {
        add_weight("rope_freqs.weight");
    }

    const float kq_scale =
        m_attention_scale != 0.0f ? m_attention_scale : 1.0f / std::sqrt(static_cast<float>(m_head_size));

    for (int il = 0; il < m_n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const std::string inpSA = cur;

        // attn_norm
        std::string attn_norm = rms_norm(cur, p + "attn_norm.weight", p + "attn_norm");

        // Q/K/V projections: MUL_MAT(w, attn_norm), then conceptual reshape to heads.
        // Fused-QKV archs (phi-3, minicpm) carry a single attn_qkv weight; split it into
        // separate q/k/v weights so the rest of the layer is architecture-agnostic.
        if (m_has_fused_qkv) {
            register_fused_qkv(il);
        } else {
            add_weight(p + "attn_q.weight");
            add_weight(p + "attn_k.weight");
            add_weight(p + "attn_v.weight");
        }
        auto q = add_op("GGML_OP_MUL_MAT", p + "Qcur", {p + "attn_q.weight", attn_norm},
                        ps({1, 1, T, m_head_size * m_n_head}), f32);
        auto k = add_op("GGML_OP_MUL_MAT", p + "Kcur", {p + "attn_k.weight", attn_norm},
                        ps({1, 1, T, m_head_size * m_n_head_kv}), f32);
        auto v = add_op("GGML_OP_MUL_MAT", p + "Vcur", {p + "attn_v.weight", attn_norm},
                        ps({1, 1, T, m_head_size * m_n_head_kv}), f32);

        // Q/K/V projection biases (qwen2 / qwen2.5: separate attn_{q,k,v}.bias).
        if (m_has_qkv_bias && !m_has_fused_qkv) {
            q = add_bias(q, p + "attn_q.bias", p + "Qcur_b");
            k = add_bias(k, p + "attn_k.bias", p + "Kcur_b");
            v = add_bias(v, p + "attn_v.bias", p + "Vcur_b");
        }

        // reshape Q/K/V to [1, n_tokens, n_head(_kv), head_size]
        q = add_op("GGML_OP_RESHAPE", p + "Qcur_r", {q}, ps({1, T, m_n_head, m_head_size}), f32, 1);
        k = add_op("GGML_OP_RESHAPE", p + "Kcur_r", {k}, ps({1, T, m_n_head_kv, m_head_size}), f32, 1);
        v = add_op("GGML_OP_RESHAPE", p + "Vcur_r", {v}, ps({1, T, m_n_head_kv, m_head_size}), f32, 1);

        // per-head q_norm / k_norm (qwen3, hunyuan)
        if (m_has_qk_norm) {
            q = rms_norm(q, p + "attn_q_norm.weight", p + "Qcur_normed");
            k = rms_norm(k, p + "attn_k_norm.weight", p + "Kcur_normed");
        }

        // RoPE (NEOX). rope_freqs.weight is passed as the optional 3rd input when present.
        const std::vector<std::string> q_rope_in =
            m_has_rope_freqs ? std::vector<std::string>{q, "inp_pos", "rope_freqs.weight"}
                             : std::vector<std::string>{q, "inp_pos"};
        const std::vector<std::string> k_rope_in =
            m_has_rope_freqs ? std::vector<std::string>{k, "inp_pos", "rope_freqs.weight"}
                             : std::vector<std::string>{k, "inp_pos"};
        q = add_op("GGML_OP_ROPE", p + "Qcur_rope", q_rope_in, ps({1, T, m_n_head, m_head_size}), f32,
                   m_rope_op_case, {{"rope_config", m_graph->rope_config}});
        k = add_op("GGML_OP_ROPE", p + "Kcur_rope", k_rope_in, ps({1, T, m_n_head_kv, m_head_size}), f32,
                   m_rope_op_case, {{"rope_config", m_graph->rope_config}});

        // ---- KV cache store (stateful) ----
        // Per-layer f16 KV caches, converted to ReadValue/Assign by MakeStateful. The
        // SET_ROWS translator's stateful branch concatenates the new K/V onto the cache, so
        // the FLASH_ATTN inputs are f16 (matching Q after its f16 convert in the translator).
        const std::string kc = "cache_k_l" + std::to_string(il);
        const std::string vc = "cache_v_l" + std::to_string(il);
        const ov::PartialShape cache_shape = ps({1, D, m_n_head_kv, m_head_size});
        add_input(kc, ov::element::f16, cache_shape);
        add_input(vc, ov::element::f16, cache_shape);
        m_tensor_shapes[kc] = ps({1, T, m_n_head_kv, m_head_size});
        m_tensor_shapes[vc] = ps({1, T, m_n_head_kv, m_head_size});
        m_tensor_types[kc] = ov::element::f16;
        m_tensor_types[vc] = ov::element::f16;
        m_graph->kv_param_res_names[kc] = kc;
        m_graph->kv_param_res_names[vc] = vc;

        // SET_ROWS(cur, idx, cache) -> combined f16 K/V. The translator (stateful branch)
        // concatenates the new K/V onto the cache. The node's OUTPUT is named after the
        // cache tensor (cache_k_l<il>) -- matching the cgraph, where SET_ROWS updates the
        // view_src in place -- so MakeStateful can pair the cache Parameter with this Result.
        k = add_op("GGML_OP_SET_ROWS", kc, {k, "inp_kv_idx", kc},
                   ps({1, T, m_n_head_kv, m_head_size}), ov::element::f16);
        v = add_op("GGML_OP_SET_ROWS", vc, {v, "inp_kv_idx", vc},
                   ps({1, T, m_n_head_kv, m_head_size}), ov::element::f16);
        // These combined caches are model outputs so they become Results that MakeStateful
        // converts into Assign sinks paired with the cache ReadValues.
        m_graph->model_output_names.push_back(kc);
        m_graph->model_output_names.push_back(vc);

        // PERMUTE q/k/v 0,2,1,3 -> [1, n_head(_kv), n_tokens, head_size] (Transpose in
        // stateful mode).
        q = add_op("GGML_OP_PERMUTE", p + "q_perm", {q}, ps({1, m_n_head, T, m_head_size}), f32, 1);
        k = add_op("GGML_OP_PERMUTE", p + "k_perm", {k}, ps({1, m_n_head_kv, T, m_head_size}), ov::element::f16, 1);
        v = add_op("GGML_OP_PERMUTE", p + "v_perm", {v}, ps({1, m_n_head_kv, T, m_head_size}), ov::element::f16, 1);

        // FLASH_ATTN_EXT(q, k, v, mask, scale) -> [1, n_tokens, n_head, head_size]
        auto attn = add_op("GGML_OP_FLASH_ATTN_EXT", p + "kqv", {q, k, v, "self_kq_mask"},
                           ps({1, T, m_n_head, m_head_size}), f32, 0, {{"scale", kq_scale}});

        // reshape back to [1, 1, n_tokens, n_head*head_size]
        auto attn_2d = add_op("GGML_OP_RESHAPE", p + "kqv_merged", {attn},
                              ps({1, 1, T, m_n_head * m_head_size}), f32, 2);

        // output projection (+ optional bias)
        add_weight(p + "attn_output.weight");
        auto attn_out = add_op("GGML_OP_MUL_MAT", p + "attn_out", {p + "attn_output.weight", attn_2d},
                               ps({1, 1, T, m_n_embd}), f32);
        if (m_has_attn_out_bias) {
            attn_out = add_bias(attn_out, p + "attn_output.bias", p + "attn_out_b");
        }
        // MiniCPM scales the attention sublayer output before the residual add.
        if (m_residual_scale != 1.0f) {
            attn_out = scale(attn_out, m_residual_scale, p + "attn_out_scaled");
        }

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

        // SwiGLU FFN. Two layouts:
        //  - separate gate+up (llama/qwen/minicpm): SWIGLU(gate, up) [2-input split].
        //  - fused ffn_up of width 2*n_ff (phi-3): SWIGLU(up) [1-input, splits internally].
        add_weight(p + "ffn_up.weight");
        add_weight(p + "ffn_down.weight");
        const bool fused_ffn = m_weights.count(p + "ffn_gate.weight") == 0;
        std::string glu;
        if (fused_ffn) {
            const int up2 = static_cast<int>(m_weights.at(p + "ffn_up.weight").get_shape()[0]);  // 2*n_ff
            auto up = add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm},
                             ps({1, 1, T, up2}), f32);
            glu = add_op("GGML_GLU_OP_SWIGLU", p + "ffn_swiglu", {up}, ps({1, 1, T, up2 / 2}), f32, 0,
                         {{"swapped", false}});
        } else {
            add_weight(p + "ffn_gate.weight");
            const int n_ff = static_cast<int>(m_weights.at(p + "ffn_gate.weight").get_shape()[0]);
            auto gate = add_op("GGML_OP_MUL_MAT", p + "ffn_gate", {p + "ffn_gate.weight", ffn_norm},
                               ps({1, 1, T, n_ff}), f32);
            auto up = add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm},
                             ps({1, 1, T, n_ff}), f32);
            glu = add_op("GGML_GLU_OP_SWIGLU", p + "ffn_swiglu", {gate, up}, ps({1, 1, T, n_ff}), f32, 0,
                         {{"swapped", false}});
        }
        auto down = add_op("GGML_OP_MUL_MAT", p + "ffn_out", {p + "ffn_down.weight", glu},
                           ps({1, 1, T, m_n_embd}), f32);
        // MiniCPM scales the FFN sublayer output before the residual add.
        if (m_residual_scale != 1.0f) {
            down = scale(down, m_residual_scale, p + "ffn_out_scaled");
        }

        cur = add_op("GGML_OP_ADD", p + "l_out", {down, ffn_inp}, ps({1, 1, T, m_n_embd}), f32);
    }

    // final norm + lm_head
    cur = rms_norm(cur, "output_norm.weight", "result_norm");

    const std::string lm_head_w = m_weights.count("output.weight") ? "output.weight" : "token_embd.weight";
    add_weight(lm_head_w);
    const int n_vocab = static_cast<int>(m_weights.at(lm_head_w).get_shape()[0]);  // rows = vocab
    auto logits = add_op("GGML_OP_MUL_MAT", "result_output", {lm_head_w, cur},
                         ps({1, 1, T, n_vocab}), f32);
    // MiniCPM scales the logits (1/(n_embd/dim_model_base)).
    if (m_logit_scale != 1.0f) {
        logits = scale(logits, m_logit_scale, "result_output_scaled");
    }

    // logits is the primary output; the cache_k/v_l* outputs (appended per layer above)
    // become Assign sinks via MakeStateful.
    m_graph->model_output_names.insert(m_graph->model_output_names.begin(), logits);
    return m_graph;
}

}  // namespace

// Dense-transformer architectures of the llama family handled by the generic builder.
// Adding a same-family architecture is just adding its GGUF architecture name here.
const std::set<std::string>& supported_archs() {
    static const std::set<std::string> archs = {
        "llama",    // llama-2 / llama-3
        "qwen2",    // qwen2 / qwen2.5
        "qwen3",
        "phi3",     // phi-3
        "minicpm",
        "hunyuan-dense",
    };
    return archs;
}

std::shared_ptr<GgmlGraph> build_ggml_graph_from_gguf(const std::string& file) {
    auto [metadata, weights, qtypes] = get_gguf_data(file);
    auto config = config_from_meta(metadata);

    const std::string arch = std::get<std::string>(config.at("architecture"));
    OPENVINO_ASSERT(supported_archs().count(arch),
                    "[ggml] native GGUF builder does not support architecture '",
                    arch,
                    "'. Supported: llama, qwen2, qwen3, phi3, minicpm, hunyuan-dense.");

    TransformerBuilder builder(config, weights, qtypes);
    return builder.build();
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
