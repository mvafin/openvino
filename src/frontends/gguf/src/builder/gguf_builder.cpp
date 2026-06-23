// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF -> GgufGraph builder. Emits nodes in the GGML op vocabulary that reproduce
// llama.cpp's cgraph topology, so the resulting GgufGraph drives the same op translators as
// the llama.cpp cgraph path. No llama.cpp / gguf dependency.
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
#include "gguf_graph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "weights.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

// gguf ROPE op-case (see ggml-decoder.cpp::compute_op_case). The high 16 bits encode the
// mode: NORMAL=0 (llama/minicpm: rotate consecutive pairs), NEOX=1 (qwen/phi/hunyuan:
// rotate halves). The input is not a VIEW here so the low bits stay 0.
constexpr int ROPE_OP_CASE_NORMAL = 0x00000000;
constexpr int ROPE_OP_CASE_NEOX = 0x00010000;

// Architectures whose rope is NEOX (rotate-halves); everything else in the supported set
// uses NORMAL (rotate consecutive pairs). Mirrors llama_model_rope_type.
bool arch_uses_neox_rope(const std::string& arch) {
    return arch == "qwen2" || arch == "qwen3" || arch == "phi3" || arch == "hunyuan-dense" || arch == "gpt-oss" ||
           arch == "gemma" || arch == "gemma2" || arch == "gemma4" || arch == "olmoe";
}

// Shapes are kept in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0] (reverse of GGUF
// on-disk order), matching the decoder's get_shape(). The translators consume them as-is.
// Token length is dynamic (-1) in dim ne1.

class TransformerBuilder {
public:
    TransformerBuilder(const std::map<std::string, GGUFMetaData>& config,
                       std::unordered_map<std::string, ov::Tensor>& weights,
                       std::unordered_map<std::string, gguf_tensor_type>& qtypes)
        : m_config(config),
          m_weights(weights),
          m_qtypes(qtypes) {
        m_graph = std::make_shared<GgufGraph>();
        m_graph->is_static = false;
        m_graph->is_stateful = true;

        m_n_layer = cfg_int("layer_num");
        m_n_head = cfg_int("head_num");
        m_n_head_kv = cfg_int("head_num_kv");
        m_head_size = cfg_int("head_size");
        m_n_embd = cfg_int("hidden_size");
        m_rms_eps = cfg_float("rms_norm_eps");

        const std::string arch_str = std::get<std::string>(config.at("architecture"));

        // Per-architecture structure, auto-detected from the GGUF tensor table (layer 0).
        m_has_qk_norm = m_weights.count("blk.0.attn_q_norm.weight") > 0;
        // q/k norm width: per-head (head_size, applied after reshape -> qwen3/hunyuan/gemma4)
        // vs full projection width (n_head*head_size, applied before reshape -> OLMoE).
        // For gemma4, norm width equals per-layer head_size (may differ by SWA/global); always
        // per-head. For OLMoE, norm width equals n_head*head_size. Detect by checking against
        // the global head_size; but for gemma4 override to per-head since it has mixed head sizes.
        if (m_has_qk_norm) {
            const size_t qn = m_weights.at("blk.0.attn_q_norm.weight").get_shape()[0];
            m_qk_norm_full = (arch_str != "gemma4") && (qn != static_cast<size_t>(m_head_size));
        }
        m_is_moe = m_weights.count("blk.0.ffn_gate_exps.weight") > 0;
        // Gemma/Gemma2 use GeGLU (GELU-gated FFN). Detected by arch name since other archs
        // in the supported set (llama, qwen2, qwen3, phi3) all use SwiGLU.
        m_is_geglu = arch_str == "gemma" || arch_str == "gemma2" || arch_str == "gemma4";
        // gemma4: V projection is also RMSNorm'd (no weight, just normalize).
        m_has_v_norm = (arch_str == "gemma4");
        m_n_expert = cfg_int("expert_count");
        m_n_expert_used = cfg_int("expert_used_count");
        m_has_moe_gate_bias = m_weights.count("blk.0.ffn_gate_inp.bias") > 0;     // gpt-oss
        m_has_moe_expert_bias = m_weights.count("blk.0.ffn_gate_exps.bias") > 0;  // gpt-oss
        m_has_sinks = m_weights.count("blk.0.attn_sinks.weight") > 0;             // gpt-oss
        // Gemma2: per-layer post-attention and post-FFN RMSNorm applied after the sublayer
        // output and before the residual add. Detected from the tensor table at layer 0.
        m_has_attn_post_norm = m_weights.count("blk.0.post_attention_norm.weight") > 0;
        m_has_ffn_post_norm = m_weights.count("blk.0.post_ffw_norm.weight") > 0;
        // gpt-oss uses "softmax-after-topk" gating + the OAI gated activation; OLMoE uses
        // softmax-before-topk + plain SwiGLU. Detect by weight-tensor presence so the logic
        // extends to future architectures without touching this file.
        m_moe_swiglu_oai = m_has_moe_gate_bias;      // gate_inp bias is OAI-gating-specific
        m_moe_softmax_weight = m_has_moe_gate_bias;  // same tensor signals softmax-after-topk
        // SWA is present if either sinks (gpt-oss) or per-layer flags (gemma4) indicate it.
        // For gemma4, m_swa_layer_flags will be populated after this block; detect by head_size_swa.
        m_has_swa = m_has_sinks || (arch_str == "gemma4");
        m_has_fused_qkv = m_weights.count("blk.0.attn_qkv.weight") > 0;  // phi-3, minicpm
        m_has_qkv_bias = m_weights.count("blk.0.attn_q.bias") > 0 || m_weights.count("blk.0.attn_qkv.bias") > 0;
        m_has_attn_out_bias = m_weights.count("blk.0.attn_output.bias") > 0;
        m_has_rope_freqs = m_weights.count("rope_freqs.weight") > 0;

        // Per-architecture scalars from metadata (1.0 / 0.0 when absent -> no-op).
        m_embedding_scale = cfg_float("embedding_scale");
        m_residual_scale = cfg_float("residual_scale");
        m_logit_scale = cfg_float("logit_scale");
        m_attention_scale = cfg_float("attention_scale");            // 0 -> 1/sqrt(head_size)
        m_expert_weights_scale = cfg_float("expert_weights_scale");  // 0 -> 1.0 no-op
        m_rope_freq_base_swa = cfg_float("rope_freq_base_swa");
        m_swa_layer_pattern = cfg_int("swa_layer_pattern");
        // Gemma4: per-layer SWA boolean flags (non-empty when swa_layer_pattern==0).
        if (config.count("swa_layer_flags"))
            m_swa_layer_flags = std::get<std::vector<int32_t>>(config.at("swa_layer_flags"));
        m_attn_soft_cap = cfg_float("attn_logit_softcapping");          // 0 -> no soft-cap
        m_final_logit_soft_cap = cfg_float("final_logit_softcapping");  // 0 -> no soft-cap
        // Gemma4: per-layer input embeddings and shared KV layers.
        m_n_embd_per_layer = cfg_int("n_embd_per_layer");
        m_shared_kv_layers = cfg_int("shared_kv_layers");
        // Gemma4: SWA layers use a smaller head size than global attention layers.
        m_head_size_swa = cfg_int("head_size_swa");
        m_rope_dim_swa = cfg_int("rope_dimension_count_swa");

        const std::string arch = std::get<std::string>(config.at("architecture"));
        m_rope_op_case = arch_uses_neox_rope(arch) ? ROPE_OP_CASE_NEOX : ROPE_OP_CASE_NORMAL;

        m_graph->has_rope = true;
        m_graph->rope_config.n_dims = cfg_int("rope_dimension_count");
        m_graph->rope_config.n_ctx_orig = cfg_int("rope_n_ctx_orig");
        m_graph->rope_config.freq_base = cfg_float("rope_freq_base");
        m_graph->rope_config.freq_scale = cfg_float("rope_freq_scale");
        m_graph->rope_config.ext_factor = cfg_float("rope_ext_factor");
        m_graph->rope_config.attn_factor = 1.0f;
        m_graph->rope_config.beta_fast = 32.0f;
        m_graph->rope_config.beta_slow = 1.0f;
        // Gemma4: separate rope config for SWA layers (different freq_base and n_dims).
        m_rope_config_swa = m_graph->rope_config;
        m_rope_config_swa.freq_base = cfg_float("rope_freq_base_swa");
        m_rope_config_swa.n_dims = cfg_int("rope_dimension_count_swa");
        // Gemma4 needs per-op sin/cos because SWA and global layers have different n_dims.
        if (m_rope_dim_swa > 0 && m_rope_dim_swa != m_graph->rope_config.n_dims) {
            m_graph->use_per_op_rope = true;
        }
    }

    std::shared_ptr<GgufGraph> build();

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
        GgufOp op;
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

    // Dense GeGLU FFN (Gemma/Gemma2). Same layout as SwiGLU but uses GELU activation.
    // `ffn_norm` is the post-norm hidden. Returns the down-projection output tensor name.
    std::string build_geglu_ffn(const std::string& p, const std::string& ffn_norm, int64_t T) {
        using ov::element::f32;
        add_weight(p + "ffn_gate.weight");
        add_weight(p + "ffn_up.weight");
        add_weight(p + "ffn_down.weight");
        const int n_ff = static_cast<int>(m_weights.at(p + "ffn_gate.weight").get_shape()[0]);
        auto gate =
            add_op("GGML_OP_MUL_MAT", p + "ffn_gate", {p + "ffn_gate.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
        auto up = add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
        auto glu =
            add_op("GGML_GLU_OP_GEGLU", p + "ffn_geglu", {gate, up}, ps({1, 1, T, n_ff}), f32, 0, {{"swapped", false}});
        return add_op("GGML_OP_MUL_MAT", p + "ffn_out", {p + "ffn_down.weight", glu}, ps({1, 1, T, m_n_embd}), f32);
    }

    // Dense SwiGLU FFN (llama/qwen/phi3/minicpm). `ffn_norm` is the post-norm hidden.
    // Returns the down-projection output tensor name. T is the static token length.
    std::string build_dense_ffn(const std::string& p, const std::string& ffn_norm, int64_t T) {
        using ov::element::f32;
        add_weight(p + "ffn_up.weight");
        add_weight(p + "ffn_down.weight");
        const bool fused_ffn = m_weights.count(p + "ffn_gate.weight") == 0;  // phi-3: fused gate+up
        std::string glu;
        if (fused_ffn) {
            const int up2 = static_cast<int>(m_weights.at(p + "ffn_up.weight").get_shape()[0]);  // 2*n_ff
            auto up = add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm}, ps({1, 1, T, up2}), f32);
            glu = add_op("GGML_GLU_OP_SWIGLU",
                         p + "ffn_swiglu",
                         {up},
                         ps({1, 1, T, up2 / 2}),
                         f32,
                         0,
                         {{"swapped", false}});
        } else {
            add_weight(p + "ffn_gate.weight");
            const int n_ff = static_cast<int>(m_weights.at(p + "ffn_gate.weight").get_shape()[0]);
            auto gate =
                add_op("GGML_OP_MUL_MAT", p + "ffn_gate", {p + "ffn_gate.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
            auto up =
                add_op("GGML_OP_MUL_MAT", p + "ffn_up", {p + "ffn_up.weight", ffn_norm}, ps({1, 1, T, n_ff}), f32);
            glu = add_op("GGML_GLU_OP_SWIGLU",
                         p + "ffn_swiglu",
                         {gate, up},
                         ps({1, 1, T, n_ff}),
                         f32,
                         0,
                         {{"swapped", false}});
        }
        return add_op("GGML_OP_MUL_MAT", p + "ffn_out", {p + "ffn_down.weight", glu}, ps({1, 1, T, m_n_embd}), f32);
    }

    // Mixture-of-experts FFN (OLMoE / gpt-oss), mirroring llm_graph_context::build_moe_ffn.
    // Routing: logits = gate_inp·x; probs = softmax/identity; pick top-k experts; per-token
    // expert matmuls via MUL_MAT_ID; gated activation; weighted sum over the used experts.
    std::string build_moe_ffn(const std::string& p, const std::string& ffn_norm, int64_t T) {
        using ov::element::f32;
        const int E = m_n_expert, K = m_n_expert_used;
        const int n_ff = static_cast<int>(m_weights.at(p + "ffn_gate_exps.weight").get_shape()[1]);

        // --- router: logits [1,1,T,E] = gate_inp · x ---
        add_weight(p + "ffn_gate_inp.weight");
        auto logits =
            add_op("GGML_OP_MUL_MAT", p + "moe_logits", {p + "ffn_gate_inp.weight", ffn_norm}, ps({1, 1, T, E}), f32);
        if (m_has_moe_gate_bias) {
            logits = add_bias(logits, p + "ffn_gate_inp.bias", p + "moe_logits_b");
        }

        // gating: softmax (OLMoE) or softmax-after-topk (gpt-oss "softmax_weight").
        std::string probs = logits;
        if (!m_moe_softmax_weight) {
            probs = add_op("GGML_OP_SOFT_MAX",
                           p + "moe_probs",
                           {logits},
                           ps({1, 1, T, E}),
                           f32,
                           0,
                           {{"softmax_axis", int64_t(-1)}});
        }

        // top-k expert selection -> indices [1,1,T,K] (i32).
        auto selected = add_op("GGML_OP_TOP_K", p + "moe_topk", {probs}, ps({1, 1, T, K}), ov::element::i32);

        // weights = gather probs by selected; op_case 10 returns a per-expert column
        // [1,T,K,1] (robust to dynamic T). gpt-oss softmaxes over the K (expert) axis.
        auto weights = add_op("GGML_OP_GET_ROWS", p + "moe_w", {probs, selected}, ps({1, T, K, 1}), f32, 10);
        if (m_moe_softmax_weight) {
            weights = add_op("GGML_OP_SOFT_MAX",
                             p + "moe_w_sm",
                             {weights},
                             ps({1, T, K, 1}),
                             f32,
                             0,
                             {{"softmax_axis", int64_t(2)}});
        }
        // gpt-oss expert_weights_scale: optional constant multiplier applied after softmax
        // (mirrors llama.cpp build_moe_ffn w_scale != 0 && w_scale != 1.0 path).
        if (m_expert_weights_scale != 0.0f && m_expert_weights_scale != 1.0f) {
            weights = scale(weights, m_expert_weights_scale, p + "moe_w_scaled");
        }

        // expert FFN via MUL_MAT_ID. The routed input x is broadcast to K slots; the
        // translator gathers each token's selected expert matrices. gpt-oss adds per-expert
        // biases (ADD_ID gathers the selected experts' bias rows).
        add_weight(p + "ffn_gate_exps.weight");
        add_weight(p + "ffn_up_exps.weight");
        add_weight(p + "ffn_down_exps.weight");
        const bool eb = m_has_moe_expert_bias;
        auto up = add_op("GGML_OP_MUL_MAT_ID",
                         p + "moe_up",
                         {p + "ffn_up_exps.weight", ffn_norm, selected},
                         ps({1, T, K, n_ff}),
                         f32);
        if (eb) {
            add_named_weight(p + "ffn_up_exps.bias");
            up = add_op("GGML_OP_ADD_ID",
                        p + "moe_up_b",
                        {up, p + "ffn_up_exps.bias", selected},
                        ps({1, T, K, n_ff}),
                        f32);
        }
        auto gate = add_op("GGML_OP_MUL_MAT_ID",
                           p + "moe_gate",
                           {p + "ffn_gate_exps.weight", ffn_norm, selected},
                           ps({1, T, K, n_ff}),
                           f32);
        if (eb) {
            add_named_weight(p + "ffn_gate_exps.bias");
            gate = add_op("GGML_OP_ADD_ID",
                          p + "moe_gate_b",
                          {gate, p + "ffn_gate_exps.bias", selected},
                          ps({1, T, K, n_ff}),
                          f32);
        }
        std::string act;
        if (m_moe_swiglu_oai) {
            act = add_op("GGML_GLU_OP_SWIGLU_OAI",
                         p + "moe_act",
                         {gate, up},
                         ps({1, T, K, n_ff}),
                         f32,
                         0,
                         {{"swapped", false}, {"alpha", 1.702f}, {"limit", 7.0f}});
        } else {
            act = add_op("GGML_GLU_OP_SWIGLU",
                         p + "moe_act",
                         {gate, up},
                         ps({1, T, K, n_ff}),
                         f32,
                         0,
                         {{"swapped", false}});
        }
        auto experts = add_op("GGML_OP_MUL_MAT_ID",
                              p + "moe_down",
                              {p + "ffn_down_exps.weight", act, selected},
                              ps({1, T, K, m_n_embd}),
                              f32);
        if (eb) {
            add_named_weight(p + "ffn_down_exps.bias");
            experts = add_op("GGML_OP_ADD_ID",
                             p + "moe_down_b",
                             {experts, p + "ffn_down_exps.bias", selected},
                             ps({1, T, K, m_n_embd}),
                             f32);
        }

        // Weighted sum over the K selected experts. weights is [1,T,K,1] (per-expert col).
        //   experts [1,T,K,n_embd] * weights -> [1,T,K,n_embd]
        //   TRANSPOSE last two axes -> [1,T,n_embd,K]; SUM_ROWS over K -> [1,T,n_embd,1]
        //   RESHAPE (dynamic) -> [1,1,T,n_embd].
        auto weighted = add_op("GGML_OP_MUL", p + "moe_weighted", {experts, weights}, ps({1, T, K, m_n_embd}), f32);
        auto tr = add_op("GGML_OP_TRANSPOSE", p + "moe_tr", {weighted}, ps({1, T, m_n_embd, K}), f32);
        auto summed = add_op("GGML_OP_SUM_ROWS", p + "moe_sum", {tr}, ps({1, T, m_n_embd, 1}), f32);
        return add_op("GGML_OP_RESHAPE", p + "moe_out", {summed}, ps({1, 1, T, m_n_embd}), f32, 7);
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
        return add_op("GGML_OP_SCALE",
                      name,
                      {x},
                      m_tensor_shapes.at(x),
                      ov::element::f32,
                      0,
                      {{"scale", factor}, {"bias", 0.0f}});
    }

    const std::map<std::string, GGUFMetaData>& m_config;
    std::unordered_map<std::string, ov::Tensor>& m_weights;
    std::unordered_map<std::string, gguf_tensor_type>& m_qtypes;
    std::shared_ptr<GgufGraph> m_graph;

    int m_n_layer = 0, m_n_head = 0, m_n_head_kv = 0, m_head_size = 0, m_n_embd = 0;
    float m_rms_eps = 0.0f;

    // auto-detected per-architecture structure
    bool m_has_qk_norm = false, m_has_qkv_bias = false, m_has_attn_out_bias = false, m_has_rope_freqs = false;
    bool m_has_fused_qkv = false, m_qk_norm_full = false, m_is_moe = false;
    bool m_has_moe_gate_bias = false, m_moe_softmax_weight = false, m_moe_swiglu_oai = false;
    bool m_has_moe_expert_bias = false, m_has_sinks = false, m_has_swa = false;
    bool m_has_attn_post_norm = false, m_has_ffn_post_norm = false, m_is_geglu = false;
    bool m_has_v_norm = false;  // gemma4: V is also RMSNorm'd like K
    int m_n_expert = 0, m_n_expert_used = 0;
    float m_embedding_scale = 1.0f, m_residual_scale = 1.0f, m_logit_scale = 1.0f, m_attention_scale = 0.0f;
    float m_expert_weights_scale = 0.0f;
    float m_attn_soft_cap = 0.0f, m_final_logit_soft_cap = 0.0f;
    float m_rope_freq_base_swa = 0.0f;
    int m_swa_layer_pattern = 2;
    std::vector<int32_t> m_swa_layer_flags;  // gemma4: per-layer SWA flags (1=SWA, 0=global)
    int m_n_embd_per_layer = 0;   // gemma4: per-layer embedding projection dimension
    int m_shared_kv_layers = 0;   // gemma4: N trailing layers that share KV from earlier layers
    int m_head_size_swa = 0;      // gemma4: head size for SWA layers (differs from global)
    int m_rope_dim_swa = 0;       // gemma4: rope dims for SWA layers
    RopeConfig m_rope_config_swa{};
    int m_rope_op_case = ROPE_OP_CASE_NEOX;

    std::map<std::string, ov::PartialShape> m_tensor_shapes;
    std::map<std::string, ov::element::Type> m_tensor_types;
};

std::shared_ptr<GgufGraph> TransformerBuilder::build() {
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
    // gguf uses i32 for token/position/index inputs.
    add_input("inp_tokens", i32, ps({1, 1, 1, D}));
    add_input("inp_pos", i32, ps({1, 1, 1, D}));
    add_input("inp_out_ids", i32, ps({1, 1, 1, D}));
    add_input("self_kq_mask", ov::element::f32, ps({1, 1, D, D}));
    // gpt-oss alternates sliding-window / full attention; the windowed mask is a separate
    // input. Only added when the model uses SWA.
    if (m_has_swa) {
        add_input("self_kq_mask_swa", ov::element::f32, ps({1, 1, D, D}));
    }
    // beam_idx: per-batch beam reorder index for the stateful KV cache. Used to gather the
    // cache along the batch axis before the SET_ROWS concat, which is the exact
    // ReadValue->Gather(beam_idx)->Concat->SDPA shape the CPU plugin's stateful_sdpa_fusion
    // matches (so SDPA becomes the fused ScaledDotProductAttentionWithKVCache). With batch=1
    // and beam_idx=[0] this is an identity reorder. Matches genai's create_cache.
    add_input("beam_idx", i32, ps({D}));

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
    std::string cur =
        add_op("GGML_OP_GET_ROWS", "embd", {"token_embd.weight", "inp_tokens"}, ps({1, 1, T, m_n_embd}), f32);
    // MiniCPM scales the embeddings by a constant.
    if (m_embedding_scale != 1.0f) {
        cur = scale(cur, m_embedding_scale, "embd_scaled");
    }

    // rope freq factors (llama-3 long context, phi-3): an optional 3rd ROPE input.
    if (m_has_rope_freqs) {
        add_weight("rope_freqs.weight");
    }

    // kq_scale is computed per-layer below to handle variable head sizes (e.g. gemma4 SWA
    // layers have head_size_swa != m_head_size). The global constant is only used when
    // m_attention_scale is set explicitly (custom scale, e.g. llama3.1 RoPE-scaling).
    const float kq_scale_global =
        m_attention_scale != 0.0f ? m_attention_scale : 1.0f / std::sqrt(static_cast<float>(m_head_size));

    // Gemma4: per-layer token embeddings are built before the layer loop and stored as a
    // 4D tensor "per_layer_embd" of shape [n_layer, T, n_embd_per_layer] (ggml logical order).
    // Inside each layer the VIEW op extracts the il-th slice [1, T, n_embd_per_layer].
    // The projection norm is applied per-slice (over the n_embd_per_layer dim).
    //
    // Topology (mirrors llama.cpp build_inp_per_layer + project_per_layer_inputs):
    //   pe_tok  = GET_ROWS(per_layer_token_embd, inp_tokens)  -> [1, n_layer, T, n_embd_per_layer]
    //             scaled by sqrt(n_embd_per_layer)
    //   pe_proj = MUL_MAT(per_layer_model_proj, cur_embd)     -> [1, n_layer, T, n_embd_per_layer]
    //             scaled by 1/sqrt(n_embd) + RMS_NORM(per_layer_proj_norm)
    //   per_layer_embd = (pe_proj + pe_tok) * 1/sqrt(2)       -> [1, n_layer, T, n_embd_per_layer]
    if (m_n_embd_per_layer > 0) {
        add_weight("per_layer_token_embd.weight");
        add_weight("per_layer_model_proj.weight");
        add_weight("per_layer_proj_norm.weight");
        const int pe_total = m_n_embd_per_layer * m_n_layer;

        // Token embedding lookup: [1,1,T, pe_total] -> reshape to [1, n_layer, T, n_embd_per_layer]
        auto pe_flat = add_op("GGML_OP_GET_ROWS",
                              "pe_tok_flat",
                              {"per_layer_token_embd.weight", "inp_tokens"},
                              ps({1, 1, T, pe_total}),
                              f32);
        const float pe_scale = std::sqrt(static_cast<float>(m_n_embd_per_layer));
        pe_flat = scale(pe_flat, pe_scale, "pe_tok_flat_scaled");
        auto pe_tok = add_op("GGML_OP_RESHAPE",
                             "pe_tok",
                             {pe_flat},
                             ps({1, m_n_layer, T, m_n_embd_per_layer}),
                             f32,
                             8);  // op_case 8: [T,n_layer*pe] -> [n_layer,-1,pe] (stateful) or [1,n_layer,-1,pe]

        // Model projection: MUL_MAT(per_layer_model_proj, embd) -> [1,1,T, pe_total]
        // per_layer_model_proj is [n_embd, pe_total] -> output [pe_total] per token
        auto proj_flat = add_op("GGML_OP_MUL_MAT",
                                "pe_proj_flat",
                                {"per_layer_model_proj.weight", cur},
                                ps({1, 1, T, pe_total}),
                                f32);
        const float proj_scale = 1.0f / std::sqrt(static_cast<float>(m_n_embd));
        proj_flat = scale(proj_flat, proj_scale, "pe_proj_flat_scaled");

        // Reshape to [1, n_layer, T, n_embd_per_layer] for per-slice RMS_NORM
        auto proj_4d = add_op("GGML_OP_RESHAPE",
                              "pe_proj_4d",
                              {proj_flat},
                              ps({1, m_n_layer, T, m_n_embd_per_layer}),
                              f32,
                              8);  // op_case 8: [T,n_layer*pe] -> [n_layer,-1,pe] (stateful)

        // RMS_NORM + per_layer_proj_norm weight (applied over last dim = n_embd_per_layer)
        auto proj_norm = add_op("GGML_OP_RMS_NORM",
                                "pe_proj_rms",
                                {proj_4d},
                                ps({1, m_n_layer, T, m_n_embd_per_layer}),
                                f32,
                                0,
                                {{"eps", m_rms_eps}});
        m_tensor_shapes["per_layer_proj_norm.weight"] = ps({1, 1, 1, m_n_embd_per_layer});
        auto proj_normed = add_op("GGML_OP_MUL",
                                  "pe_proj_normed",
                                  {proj_norm, "per_layer_proj_norm.weight"},
                                  ps({1, m_n_layer, T, m_n_embd_per_layer}),
                                  f32);

        // Sum token embd + projection, scale by 1/sqrt(2)
        auto pe_sum = add_op("GGML_OP_ADD",
                             "pe_sum",
                             {proj_normed, pe_tok},
                             ps({1, m_n_layer, T, m_n_embd_per_layer}),
                             f32);
        const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
        add_op("GGML_OP_SCALE",
               "per_layer_embd",
               {pe_sum},
               ps({1, m_n_layer, T, m_n_embd_per_layer}),
               f32,
               0,
               {{"scale", inv_sqrt2}, {"bias", 0.0f}});
    }

    // Gemma4 shared-KV: SWA shared layers reuse a different anchor than global shared layers.
    // Precompute the last own-KV layer for each SWA type (mirrors llama's layer_reuse_cb).
    const int n_own_kv = m_n_layer - m_shared_kv_layers;  // first n_own_kv layers have own KV
    int anchor_swa = -1, anchor_global = -1;
    if (m_shared_kv_layers > 0) {
        for (int i = n_own_kv - 1; i >= 0; --i) {
            bool i_is_swa = (!m_swa_layer_flags.empty() ? m_swa_layer_flags[i] != 0
                             : (m_has_swa && m_swa_layer_pattern > 0 &&
                                (i % m_swa_layer_pattern < (m_swa_layer_pattern - 1))));
            if (anchor_swa < 0 && i_is_swa)   anchor_swa    = i;
            if (anchor_global < 0 && !i_is_swa) anchor_global = i;
            if (anchor_swa >= 0 && anchor_global >= 0) break;
        }
    }

    for (int il = 0; il < m_n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const std::string inpSA = cur;

        // attn_norm
        std::string attn_norm = rms_norm(cur, p + "attn_norm.weight", p + "attn_norm");

        // SWA detection: gemma4 uses a per-layer boolean flag array; gpt-oss uses a period.
        bool is_swa_layer = false;
        if (!m_swa_layer_flags.empty()) {
            is_swa_layer = il < (int)m_swa_layer_flags.size() && m_swa_layer_flags[il] != 0;
        } else {
            is_swa_layer = m_has_swa && m_swa_layer_pattern > 0 &&
                           (il % m_swa_layer_pattern < (m_swa_layer_pattern - 1));
        }
        // Per-layer head/KV sizes: gemma4 SWA layers have smaller heads than global layers.
        const int head_size_l = (is_swa_layer && m_head_size_swa > 0) ? m_head_size_swa : m_head_size;
        const int n_head_kv_l = m_n_head_kv;
        // Per-layer attention scale: if m_attention_scale is set explicitly, keep it; otherwise
        // use 1/sqrt(head_size_l) so SWA and global layers each get the right scale.
        const float kq_scale = m_attention_scale != 0.0f
                                   ? kq_scale_global
                                   : 1.0f / std::sqrt(static_cast<float>(head_size_l));

        RopeConfig rope_config_l = is_swa_layer ? m_rope_config_swa : m_graph->rope_config;

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
        auto q = add_op("GGML_OP_MUL_MAT",
                        p + "Qcur",
                        {p + "attn_q.weight", attn_norm},
                        ps({1, 1, T, head_size_l * m_n_head}),
                        f32);
        auto k = add_op("GGML_OP_MUL_MAT",
                        p + "Kcur",
                        {p + "attn_k.weight", attn_norm},
                        ps({1, 1, T, head_size_l * n_head_kv_l}),
                        f32);
        auto v = add_op("GGML_OP_MUL_MAT",
                        p + "Vcur",
                        {p + "attn_v.weight", attn_norm},
                        ps({1, 1, T, head_size_l * n_head_kv_l}),
                        f32);

        // Q/K/V projection biases (qwen2 / qwen2.5: separate attn_{q,k,v}.bias).
        if (m_has_qkv_bias && !m_has_fused_qkv) {
            q = add_bias(q, p + "attn_q.bias", p + "Qcur_b");
            k = add_bias(k, p + "attn_k.bias", p + "Kcur_b");
            v = add_bias(v, p + "attn_v.bias", p + "Vcur_b");
        }

        // Full-width q/k norm (OLMoE): normalize the whole projection before splitting heads.
        if (m_has_qk_norm && m_qk_norm_full) {
            q = rms_norm(q, p + "attn_q_norm.weight", p + "Qcur_normed");
            k = rms_norm(k, p + "attn_k_norm.weight", p + "Kcur_normed");
        }

        // reshape Q/K/V to [1, n_tokens, n_head(_kv), head_size]
        q = add_op("GGML_OP_RESHAPE", p + "Qcur_r", {q}, ps({1, T, m_n_head, head_size_l}), f32, 1);
        k = add_op("GGML_OP_RESHAPE", p + "Kcur_r", {k}, ps({1, T, n_head_kv_l, head_size_l}), f32, 1);
        v = add_op("GGML_OP_RESHAPE", p + "Vcur_r", {v}, ps({1, T, n_head_kv_l, head_size_l}), f32, 1);

        // per-head q_norm / k_norm (qwen3, hunyuan, gemma4)
        if (m_has_qk_norm && !m_qk_norm_full) {
            q = rms_norm(q, p + "attn_q_norm.weight", p + "Qcur_normed");
            k = rms_norm(k, p + "attn_k_norm.weight", p + "Kcur_normed");
        }
        // gemma4: V gets a plain RMSNorm (no multiplicative weight, just normalize).
        if (m_has_v_norm) {
            v = add_op("GGML_OP_RMS_NORM",
                       p + "Vcur_normed",
                       {v},
                       ps({1, T, n_head_kv_l, head_size_l}),
                       f32,
                       0,
                       {{"eps", m_rms_eps}});
        }

        // RoPE (NEOX). rope_freqs.weight (per-dim frequency factor) is an optional 3rd input.
        // For gemma4: global layers use rope_freqs (proportional/NTK scaling), SWA layers don't.
        const bool use_rope_freqs = m_has_rope_freqs && !is_swa_layer;
        const std::vector<std::string> q_rope_in = use_rope_freqs
                                                       ? std::vector<std::string>{q, "inp_pos", "rope_freqs.weight"}
                                                       : std::vector<std::string>{q, "inp_pos"};
        const std::vector<std::string> k_rope_in = use_rope_freqs
                                                       ? std::vector<std::string>{k, "inp_pos", "rope_freqs.weight"}
                                                       : std::vector<std::string>{k, "inp_pos"};
        q = add_op("GGML_OP_ROPE",
                   p + "Qcur_rope",
                   q_rope_in,
                   ps({1, T, m_n_head, head_size_l}),
                   f32,
                   m_rope_op_case,
                   {{"rope_config", rope_config_l}});
        k = add_op("GGML_OP_ROPE",
                   p + "Kcur_rope",
                   k_rope_in,
                   ps({1, T, n_head_kv_l, head_size_l}),
                   f32,
                   m_rope_op_case,
                   {{"rope_config", rope_config_l}});

        // ---- KV cache store (stateful) ----
        // Gemma4: layers with shared_kv_layers have no K/V of their own; they reuse the KV
        // from the last layer of the same SWA type that has its own KV cache. SWA layers
        // reuse the last own-KV SWA layer; global layers reuse the last own-KV global layer.
        const bool has_own_kv = (m_shared_kv_layers == 0) || (il < n_own_kv);
        int anchor_il = il;
        if (!has_own_kv) {
            anchor_il = is_swa_layer ? anchor_swa : anchor_global;
            if (anchor_il < 0) anchor_il = n_own_kv - 1;  // fallback
        }
        const std::string kc = "cache_k_l" + std::to_string(anchor_il);
        const std::string vc = "cache_v_l" + std::to_string(anchor_il);

        if (has_own_kv) {
            // Per-layer f16 KV caches, converted to ReadValue/Assign by MakeStateful. The
            // SET_ROWS translator's stateful branch concatenates the new K/V onto the cache, so
            // the FLASH_ATTN inputs are f16 (matching Q after its f16 convert in the translator).
            const ov::PartialShape cache_shape = ps({1, D, n_head_kv_l, head_size_l});
            if (!m_graph->model_inputs.count(kc)) {
                add_input(kc, ov::element::f16, cache_shape);
                add_input(vc, ov::element::f16, cache_shape);
                m_graph->kv_param_res_names[kc] = kc;
                m_graph->kv_param_res_names[vc] = vc;
            }
            m_tensor_shapes[kc] = ps({1, T, n_head_kv_l, head_size_l});
            m_tensor_shapes[vc] = ps({1, T, n_head_kv_l, head_size_l});
            m_tensor_types[kc] = ov::element::f16;
            m_tensor_types[vc] = ov::element::f16;

            // SET_ROWS(cur, idx, cache) -> combined f16 K/V. The translator (stateful branch)
            // concatenates the new K/V onto the cache.
            k = add_op("GGML_OP_SET_ROWS",
                       kc,
                       {k, "inp_kv_idx", kc},
                       ps({1, T, n_head_kv_l, head_size_l}),
                       ov::element::f16);
            v = add_op("GGML_OP_SET_ROWS",
                       vc,
                       {v, "inp_kv_idx", vc},
                       ps({1, T, n_head_kv_l, head_size_l}),
                       ov::element::f16);
            // These combined caches are model outputs so they become Results that MakeStateful
            // converts into Assign sinks paired with the cache ReadValues.
            m_graph->model_output_names.push_back(kc);
            m_graph->model_output_names.push_back(vc);
        } else {
            // Shared-KV layer: K/V have already been set in the anchor layer's SET_ROWS.
            // Use the anchor's combined cache. If the current layer has a smaller head size
            // (SWA shared layer vs a global anchor), slice K/V to the layer's head_size along
            // the last dim, mirroring llama.cpp's ggml_view_4d with n_embd_head_k(il).
            const ov::PartialShape& anchor_kc_shape = m_tensor_shapes.at(kc);  // [1, T, n_kv, anchor_head]
            const int64_t anchor_hs = anchor_kc_shape[3].is_static() ? anchor_kc_shape[3].get_length() : m_head_size;
            if (head_size_l < static_cast<int>(anchor_hs)) {
                // Slice the head-size dimension to head_size_l (op_case=5).
                const ov::PartialShape k_slice_shape = ps({1, T, n_head_kv_l, head_size_l});
                const ov::PartialShape v_slice_shape = ps({1, T, n_head_kv_l, head_size_l});
                k = add_op("GGML_OP_VIEW", p + "k_hslice", {kc}, k_slice_shape, ov::element::f16, 5,
                           {{"head_size", int64_t(head_size_l)}});
                v = add_op("GGML_OP_VIEW", p + "v_hslice", {vc}, v_slice_shape, ov::element::f16, 5,
                           {{"head_size", int64_t(head_size_l)}});
            } else {
                k = kc;
                v = vc;
            }
        }

        // NOTE: q/k/v stay in the ggml-natural [1, n_tokens, n_head(_kv), head_size] layout
        // here. The PERMUTE to [1, n_head, n_tokens, head_size] is done INSIDE the FLASH_ATTN
        // translator, AFTER the GQA broadcast of K/V. That ordering -- Concat -> GQA tile ->
        // single Transpose -> SDPA -- is the exact shape the CPU plugin's stateful_sdpa_fusion
        // matches (its multi-query-bcst sits on the concat output, before one transpose), so
        // the attention fuses into ScaledDotProductAttentionWithKVCache. Permuting before the
        // tile (the old order) put the transpose between concat and tile and blocked the fuse.

        // FLASH_ATTN_EXT(q, k, v, mask[, sinks]) -> [1, n_tokens, n_head, head_size].
        // gpt-oss: SWA layers use the sliding-window mask; plus a per-head sink logit.
        const std::string mask_name = is_swa_layer ? "self_kq_mask_swa" : "self_kq_mask";
        std::vector<std::string> attn_in = {q, k, v, mask_name};
        if (m_has_sinks) {
            add_named_weight(p + "attn_sinks.weight");
            attn_in.push_back(p + "attn_sinks.weight");
        }
        std::map<std::string, ov::Any> attn_attrs = {{"scale", kq_scale}};
        if (m_attn_soft_cap != 0.0f) {
            attn_attrs["kq_soft_cap"] = m_attn_soft_cap;
        }
        auto attn = add_op("GGML_OP_FLASH_ATTN_EXT",
                           p + "kqv",
                           attn_in,
                           ps({1, T, m_n_head, head_size_l}),
                           f32,
                           0,
                           std::move(attn_attrs));

        // reshape back to [1, 1, n_tokens, n_head*head_size]
        auto attn_2d =
            add_op("GGML_OP_RESHAPE", p + "kqv_merged", {attn}, ps({1, 1, T, m_n_head * head_size_l}), f32, 2);

        // output projection (+ optional bias)
        add_weight(p + "attn_output.weight");
        auto attn_out = add_op("GGML_OP_MUL_MAT",
                               p + "attn_out",
                               {p + "attn_output.weight", attn_2d},
                               ps({1, 1, T, m_n_embd}),
                               f32);
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
            ao = add_op("GGML_OP_GET_ROWS", p + "attn_out_g", {attn_out, "inp_out_ids"}, ps({1, 1, T, m_n_embd}), f32);
            sa = add_op("GGML_OP_GET_ROWS", p + "inpSA_g", {inpSA, "inp_out_ids"}, ps({1, 1, T, m_n_embd}), f32);
        }
        // Gemma2: post-attention RMSNorm applied to the sublayer output before residual add.
        // Applied after GET_ROWS so the selected-token path matches gemma2.cpp's order.
        if (m_has_attn_post_norm) {
            ao = rms_norm(ao, p + "post_attention_norm.weight", p + "attn_post_norm");
        }

        auto ffn_inp = add_op("GGML_OP_ADD", p + "ffn_inp", {ao, sa}, ps({1, 1, T, m_n_embd}), f32);

        // Pre-FFN/MoE norm. gpt-oss names it post_attention_norm (applied before FFN, not after
        // attention); Gemma2 uses ffn_norm.weight here (the Gemma2 post_attention_norm is handled
        // above). Other archs always use ffn_norm.weight.
        const std::string ffn_norm_w = (!m_has_attn_post_norm && m_weights.count(p + "post_attention_norm.weight"))
                                           ? p + "post_attention_norm.weight"
                                           : p + "ffn_norm.weight";
        auto ffn_norm = rms_norm(ffn_inp, ffn_norm_w, p + "ffn_norm");

        std::string down = m_is_moe     ? build_moe_ffn(p, ffn_norm, T)
                           : m_is_geglu ? build_geglu_ffn(p, ffn_norm, T)
                                        : build_dense_ffn(p, ffn_norm, T);
        // MiniCPM scales the FFN sublayer output before the residual add.
        if (m_residual_scale != 1.0f) {
            down = scale(down, m_residual_scale, p + "ffn_out_scaled");
        }
        // Gemma2: post-FFN RMSNorm applied to the FFN output before residual add.
        if (m_has_ffn_post_norm) {
            down = rms_norm(down, p + "post_ffw_norm.weight", p + "ffn_post_norm");
        }

        cur = add_op("GGML_OP_ADD", p + "l_out", {down, ffn_inp}, ps({1, 1, T, m_n_embd}), f32);

        // Gemma4: per-layer embedding injection.
        // Each layer takes a slice of the pre-projected per-layer embedding (shape
        // [1,1,T,n_embd_per_layer]), gates it through inp_gate.weight (GELU), multiplies
        // by the per-layer slice, projects back to n_embd, post-norms, then adds residual.
        // per_layer_token_embd (global) is pre-projected before the loop in build().
        if (m_n_embd_per_layer > 0) {
            // Global per-layer projected embedding was added as "per_layer_embd" before the loop.
            // Slice out this layer's [1,1,T,n_embd_per_layer] chunk (dim 1 at index il).
            const std::string pl_slice = p + "per_layer_slice";
            add_op("GGML_OP_VIEW",
                   pl_slice,
                   {"per_layer_embd"},
                   ps({1, 1, T, m_n_embd_per_layer}),
                   f32,
                   4,  // op_case 4: layer-index slice using "layer_idx" attribute
                   {{"layer_idx", int64_t(il)}});

            // At the last layer, cur is already filtered to 1 token via inp_out_ids.
            // Mirror llama.cpp gemma4.cpp:347-349: also filter pl_slice so the MUL
            // doesn't broadcast it back to the full sequence length.
            std::string pl_slice_used = pl_slice;
            if (il == m_n_layer - 1) {
                pl_slice_used = p + "per_layer_slice_sel";
                add_op("GGML_OP_GET_ROWS",
                       pl_slice_used,
                       {pl_slice, "inp_out_ids"},
                       ps({1, 1, T, m_n_embd_per_layer}),
                       f32);
            }

            // gate: cur -> inp_gate.weight -> GELU -> [1,1,T,n_embd_per_layer]
            add_weight(p + "inp_gate.weight");
            auto gated = add_op("GGML_OP_MUL_MAT",
                                p + "inp_gate_mm",
                                {p + "inp_gate.weight", cur},
                                ps({1, 1, T, m_n_embd_per_layer}),
                                f32);
            gated = add_op("GGML_UNARY_OP_GELU", p + "inp_gate_gelu", {gated}, ps({1, 1, T, m_n_embd_per_layer}), f32);

            // elementwise multiply by per-layer slice
            auto mul_pe = add_op("GGML_OP_MUL", p + "pe_mul", {gated, pl_slice_used}, ps({1, 1, T, m_n_embd_per_layer}), f32);

            // project back to n_embd
            add_weight(p + "proj.weight");
            auto pe_proj = add_op("GGML_OP_MUL_MAT",
                                  p + "pe_proj",
                                  {p + "proj.weight", mul_pe},
                                  ps({1, 1, T, m_n_embd}),
                                  f32);

            // post-norm + residual add
            pe_proj = rms_norm(pe_proj, p + "post_norm.weight", p + "pe_post_norm");
            cur = add_op("GGML_OP_ADD", p + "pe_out", {cur, pe_proj}, ps({1, 1, T, m_n_embd}), f32);
        }

        // Gemma4: per-layer output scale (layer_output_scale.weight is a scalar [1]).
        if (m_weights.count(p + "layer_output_scale.weight")) {
            add_named_weight(p + "layer_output_scale.weight");
            cur = add_op("GGML_OP_MUL", p + "scaled_out", {cur, p + "layer_output_scale.weight"}, m_tensor_shapes.at(cur), f32);
        }
    }

    // final norm + lm_head
    cur = rms_norm(cur, "output_norm.weight", "result_norm");

    const std::string lm_head_w = m_weights.count("output.weight") ? "output.weight" : "token_embd.weight";
    add_weight(lm_head_w);
    const int n_vocab = static_cast<int>(m_weights.at(lm_head_w).get_shape()[0]);  // rows = vocab
    auto logits = add_op("GGML_OP_MUL_MAT", "result_output", {lm_head_w, cur}, ps({1, 1, T, n_vocab}), f32);
    // MiniCPM scales the logits (1/(n_embd/dim_model_base)).
    if (m_logit_scale != 1.0f) {
        logits = scale(logits, m_logit_scale, "result_output_scaled");
    }
    // Gemma2/Gemma3 final logit soft-cap: tanh(logits / cap) * cap.
    if (m_final_logit_soft_cap != 0.0f) {
        logits = scale(logits, 1.0f / m_final_logit_soft_cap, "logits_softcap_scaled");
        logits = add_op("GGML_UNARY_OP_TANH", "logits_softcap_tanh", {logits}, m_tensor_shapes.at(logits), f32);
        logits = scale(logits, m_final_logit_soft_cap, "result_output_softcapped");
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
        "llama",  // llama-2 / llama-3
        "qwen2",  // qwen2 / qwen2.5
        "qwen3",
        "phi3",  // phi-3
        "minicpm",
        "hunyuan-dense",
        "olmoe",    // MoE
        "gpt-oss",  // MoE + sinks + SWA
        "gemma",    // Gemma 2B / 7B
        "gemma2",   // Gemma 2 (2B / 9B / 27B) with post-norms and attention soft-cap
        "gemma4",   // Gemma 4 (E2B / E4B / 12B) with SWA, per-layer embeddings, shared KV
    };
    return archs;
}

// Pull the `tokenizer.*` ggml metadata into an ov::AnyMap keyed by the sub-key after the last
// dot (e.g. "tokenizer.ggml.tokens" -> "tokens", "tokenizer.chat_template" -> "chat_template").
// Each GGUF metadata variant is mapped to the ov::Any types a downstream tokenizer builder
// consumes: std::string / std::vector<std::string> / ov::Tensor (arrays and shape-{} scalars).
static ov::AnyMap extract_tokenizer_config(const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    const std::string prefix = "tokenizer.";
    ov::AnyMap cfg;
    for (const auto& [key, value] : metadata) {
        if (key.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const auto sub_key = key.substr(key.find_last_of('.') + 1);
        std::visit(
            [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    // skip empty
                } else {
                    cfg[sub_key] = v;
                }
            },
            value);
    }
    return cfg;
}

std::shared_ptr<GgufGraph> build_ggml_graph_from_gguf(const std::string& file) {
    auto [metadata, weights, qtypes, mmap, quant_buf] = get_gguf_data(file);
    auto config = config_from_meta(metadata);

    const std::string arch = std::get<std::string>(config.at("architecture"));
    OPENVINO_ASSERT(supported_archs().count(arch),
                    "[ggml] native GGUF builder does not support architecture '",
                    arch,
                    "'. Supported: llama, qwen2, qwen3, phi3, minicpm, hunyuan-dense, gemma, gemma2, gemma4.");

    TransformerBuilder builder(config, weights, qtypes);
    auto graph = builder.build();
    graph->tokenizer_config = extract_tokenizer_config(metadata);
    return graph;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
