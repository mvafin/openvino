// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Graph synthesis for llama / qwen2 / qwen3 / qwen3vl GGUF models. Port of the Python PoC.

#include "graph_builder.hpp"

#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "gguf_compress.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

using ov::Output;
using ov::op::util::Variable;
using ov::op::util::VariableInfo;
using ov::op::v0::Concat;
using ov::op::v0::Constant;
using ov::op::v0::Convert;
using ov::op::v0::Cos;
using ov::op::v0::MatMul;
using ov::op::v0::Parameter;
using ov::op::v0::Result;
using ov::op::v0::Sin;
using ov::op::v0::Sqrt;
using ov::op::v0::Squeeze;
using ov::op::v0::Unsqueeze;
using ov::op::v1::Add;
using ov::op::v1::Convolution;
using ov::op::v1::Divide;
using ov::op::v1::Multiply;
using ov::op::v1::OneHot;
using ov::op::v1::Power;
using ov::op::v1::ReduceMean;
using ov::op::v1::ReduceProd;
using ov::op::v1::ReduceSum;
using ov::op::v1::Reshape;
using ov::op::v1::Split;
using ov::op::v1::Transpose;
using ov::op::v11::TopK;
using ov::op::v12::ScatterElementsUpdate;
using ov::op::v13::ScaledDotProductAttention;
using ov::op::v3::Broadcast;
using ov::op::v3::NonZero;
using ov::op::v3::ShapeOf;
using ov::op::v4::Swish;
using ov::op::v6::Assign;
using ov::op::v6::MVN;
using ov::op::v6::ReadValue;
using ov::op::v7::Gelu;
using ov::op::v8::Gather;
using ov::op::v8::Slice;
using ov::op::v8::Softmax;

// ---------- small helpers ------------------------------------------------- //

std::shared_ptr<Constant> i64c(std::initializer_list<int64_t> v) {
    return std::make_shared<Constant>(ov::element::i64, ov::Shape{v.size()}, std::vector<int64_t>(v));
}
std::shared_ptr<Constant> i64s(int64_t v) {
    return std::make_shared<Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{v});
}
std::shared_ptr<Constant> i32c(std::initializer_list<int32_t> v) {
    return std::make_shared<Constant>(ov::element::i32, ov::Shape{v.size()}, std::vector<int32_t>(v));
}
std::shared_ptr<Constant> i32s(int32_t v) {
    return std::make_shared<Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{v});
}
std::shared_ptr<Constant> f32s(float v) {
    return std::make_shared<Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{v});
}

// Pull config integer from GGUF metadata. arch is e.g. "llama", "qwen2" or "qwen3".
template <typename T>
T meta_get(const GGUFFile& f, const std::string& key) {
    auto it = f.metadata().find(key);
    OPENVINO_ASSERT(it != f.metadata().end(), "GGUF: missing metadata key '", key, "'");
    return it->second.get<T>();
}
template <typename T>
T meta_get_or(const GGUFFile& f, const std::string& key, T def) {
    auto it = f.metadata().find(key);
    return it == f.metadata().end() ? def : it->second.get<T>();
}

struct Config {
    std::string arch;
    int n_layer{}, n_head{}, n_head_kv{}, n_embd{}, head_dim{}, n_ff{};
    int max_pos{};
    int vocab_size{};
    float rms_eps{};
    float rope_base{};
    // MoE-only (zero for dense architectures).
    int n_experts{};     // total number of experts per layer
    int experts_used{};  // top-k routed experts
    int n_ff_expert{};   // per-expert intermediate size (qwen3moe.expert_feed_forward_length)
};

Config read_config(const GGUFFile& f) {
    Config c;
    c.arch = meta_get<std::string>(f, "general.architecture");
    // For text-only inference, qwen3vl is structurally identical to qwen3:
    // same tensor layout, same q_norm/k_norm, same RMSNorm. The M-RoPE used
    // by qwen3vl collapses to standard 1D RoPE when all three position
    // channels (T, H, W) are equal, which is exactly the case for plain text
    // tokens. Vision tokens require multi-modal handling outside this LLM
    // graph (see the separate mmproj builder for the ViT + projector).
    //
    // mistral / mistral3 / mistral4 share the llama tensor schema verbatim
    // (separate q/k/v projections, gate+up SwiGLU MLP, no q_norm/k_norm).
    // The only difference from llama is the metadata-key prefix, which is
    // handled transparently by `meta_get<>(f, arch + ".<key>")` below.
    OPENVINO_ASSERT(c.arch == "llama" || c.arch == "qwen2" || c.arch == "qwen3" || c.arch == "qwen3vl" ||
                        c.arch == "qwen3moe" || c.arch == "mistral" || c.arch == "mistral3" || c.arch == "mistral4",
                    "GGUF frontend PoC: unsupported architecture '",
                    c.arch,
                    "'. Supported: llama, qwen2, qwen3, qwen3vl, qwen3moe, "
                    "mistral, mistral3, mistral4.");
    const auto& a = c.arch;
    c.n_layer = meta_get<int>(f, a + ".block_count");
    c.n_head = meta_get<int>(f, a + ".attention.head_count");
    c.n_head_kv = meta_get<int>(f, a + ".attention.head_count_kv");
    c.n_embd = meta_get<int>(f, a + ".embedding_length");
    c.n_ff = meta_get<int>(f, a + ".feed_forward_length");
    c.rms_eps = meta_get<float>(f, a + ".attention.layer_norm_rms_epsilon");
    c.rope_base = meta_get<float>(f, a + ".rope.freq_base");
    c.max_pos = meta_get<int>(f, a + ".context_length");
    // head_dim resolution order:
    //   1. <arch>.rope.dimension_count    (llama / qwen2)
    //   2. <arch>.attention.key_length    (qwen3 — head_dim may differ from
    //                                      n_embd / n_head)
    //   3. n_embd / n_head                (last-resort fallback)
    auto rope_dim = f.metadata().find(a + ".rope.dimension_count");
    auto key_len = f.metadata().find(a + ".attention.key_length");
    if (rope_dim != f.metadata().end()) {
        c.head_dim = rope_dim->second.get<int>();
    } else if (key_len != f.metadata().end()) {
        c.head_dim = key_len->second.get<int>();
    } else {
        c.head_dim = c.n_embd / c.n_head;
    }
    // vocab size from token list length.
    auto tit = f.metadata().find("tokenizer.ggml.tokens");
    if (tit != f.metadata().end() && std::holds_alternative<MetaArray>(tit->second.v)) {
        c.vocab_size = static_cast<int>(std::get<MetaArray>(tit->second.v).size());
    }
    // MoE: qwen3moe carries its expert count, top-k and per-expert intermediate
    // size in dedicated metadata keys. For dense architectures these stay 0,
    // which selects the dense MLP path below.
    if (c.arch == "qwen3moe") {
        c.n_experts = meta_get<int>(f, a + ".expert_count");
        c.experts_used = meta_get<int>(f, a + ".expert_used_count");
        c.n_ff_expert = meta_get<int>(f, a + ".expert_feed_forward_length");
        OPENVINO_ASSERT(c.n_experts > 0 && c.experts_used > 0 && c.n_ff_expert > 0,
                        "GGUF qwen3moe: invalid MoE metadata (n_experts=",
                        c.n_experts,
                        ", experts_used=",
                        c.experts_used,
                        ", n_ff_expert=",
                        c.n_ff_expert,
                        ")");
        OPENVINO_ASSERT(c.experts_used <= c.n_experts,
                        "GGUF qwen3moe: experts_used (",
                        c.experts_used,
                        ") exceeds expert_count (",
                        c.n_experts,
                        ")");
    }
    return c;
}

// ---------- GGUF -> HF tensor name mapping -------------------------------- //

std::string gguf_to_hf(const std::string& g) {
    if (g == "token_embd.weight")
        return "model.embed_tokens.weight";
    if (g == "output_norm.weight")
        return "model.norm.weight";
    if (g == "output.weight")
        return "lm_head.weight";
    if (g.rfind("blk.", 0) != 0)
        return {};
    auto dot = g.find('.', 4);
    if (dot == std::string::npos)
        return {};
    int i = std::stoi(g.substr(4, dot - 4));
    std::string rest = g.substr(dot + 1);
    std::string p = "model.layers." + std::to_string(i) + ".";
    static const std::pair<std::string, std::string> table[] = {
        {"attn_norm.weight", "input_layernorm.weight"},
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"attn_q.bias", "self_attn.q_proj.bias"},
        {"attn_k.bias", "self_attn.k_proj.bias"},
        {"attn_v.bias", "self_attn.v_proj.bias"},
        // Qwen3: per-head RMSNorm on Q and K before RoPE.
        {"attn_q_norm.weight", "self_attn.q_norm.weight"},
        {"attn_k_norm.weight", "self_attn.k_norm.weight"},
        {"ffn_norm.weight", "post_attention_layernorm.weight"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"},
        {"ffn_down.weight", "mlp.down_proj.weight"},
        // qwen3moe: router (logits = hidden @ W^T, W shape [n_experts, hidden]).
        // The 3D per-expert tensors (ffn_gate_exps / ffn_up_exps / ffn_down_exps)
        // are not in this 1:1 map; they are split into per-expert 2D constants
        // by the MoE loader in build_model().
        {"ffn_gate_inp.weight", "mlp.gate.weight"},
    };
    for (const auto& kv : table) {
        if (rest == kv.first)
            return p + kv.second;
    }
    return {};
}

// ---------- graph building blocks ---------------------------------------- //

// Weights map holds an FP32 Output<Node> per HF-named tensor. The node may be
// either a plain Constant + optional Convert, or a compressed-weights
// decompression chain (see gguf_compress.cpp).
struct Weights {
    std::unordered_map<std::string, Output<ov::Node>> by_hf_name;
    bool has(const std::string& n) const {
        return by_hf_name.count(n) > 0;
    }
    Output<ov::Node> at(const std::string& n) const {
        auto it = by_hf_name.find(n);
        OPENVINO_ASSERT(it != by_hf_name.end(), "GGUF: missing weight '", n, "'");
        return it->second;
    }
};

Output<ov::Node> rms_norm(const Output<ov::Node>& x, const Output<ov::Node>& w, float eps, const std::string& name) {
    auto sq = std::make_shared<Power>(x, f32s(2.0f));
    auto var = std::make_shared<ReduceMean>(sq, i32c({-1}), true);
    auto var_eps = std::make_shared<Add>(var, f32s(eps));
    auto inv = std::make_shared<Divide>(f32s(1.0f), std::make_shared<Sqrt>(var_eps));
    Output<ov::Node> normed = std::make_shared<Multiply>(x, inv);
    auto out = std::make_shared<Multiply>(normed, w);
    out->set_friendly_name(name);
    return out->output(0);
}

std::pair<Output<ov::Node>, Output<ov::Node>> rope_cos_sin(const Output<ov::Node>& position_ids,
                                                           int head_dim,
                                                           float base) {
    std::vector<float> inv_freq(head_dim / 2);
    for (int k = 0; k < head_dim / 2; ++k) {
        inv_freq[k] = 1.0f / std::pow(base, static_cast<float>(2 * k) / head_dim);
    }
    auto inv_freq_const =
        std::make_shared<Constant>(ov::element::f32, ov::Shape{1, static_cast<size_t>(head_dim / 2), 1}, inv_freq);

    auto pos_f32 = std::make_shared<Convert>(position_ids, ov::element::f32);
    auto pos_unsq = std::make_shared<Unsqueeze>(pos_f32, i64s(1));                  // [B, 1, L]
    auto freqs = std::make_shared<MatMul>(inv_freq_const, pos_unsq, false, false);  // [B, d/2, L]
    auto freqs_t = std::make_shared<Transpose>(freqs, i32c({0, 2, 1}));             // [B, L, d/2]
    auto emb = std::make_shared<Concat>(ov::OutputVector{freqs_t, freqs_t}, -1);    // [B, L, d]
    return {std::make_shared<Cos>(emb)->output(0), std::make_shared<Sin>(emb)->output(0)};
}

Output<ov::Node> rotate_half(const Output<ov::Node>& x, int head_dim) {
    int half = head_dim / 2;
    auto x1 = std::make_shared<Slice>(x, i64c({0}), i64c({half}), i64c({1}), i64c({-1}));
    auto x2 = std::make_shared<Slice>(x, i64c({half}), i64c({head_dim}), i64c({1}), i64c({-1}));
    auto neg_x2 = std::make_shared<Multiply>(x2, f32s(-1.0f));
    return std::make_shared<Concat>(ov::OutputVector{neg_x2, x1}, -1)->output(0);
}

std::pair<Output<ov::Node>, Output<ov::Node>> apply_rope(const Output<ov::Node>& q,
                                                         const Output<ov::Node>& k,
                                                         const Output<ov::Node>& cos,
                                                         const Output<ov::Node>& sin,
                                                         int head_dim) {
    auto cos_b = std::make_shared<Unsqueeze>(cos, i64s(1));
    auto sin_b = std::make_shared<Unsqueeze>(sin, i64s(1));
    auto q_rot = std::make_shared<Add>(std::make_shared<Multiply>(q, cos_b),
                                       std::make_shared<Multiply>(rotate_half(q, head_dim), sin_b));
    auto k_rot = std::make_shared<Add>(std::make_shared<Multiply>(k, cos_b),
                                       std::make_shared<Multiply>(rotate_half(k, head_dim), sin_b));
    return {q_rot->output(0), k_rot->output(0)};
}

Output<ov::Node> linear(const Output<ov::Node>& x,
                        const Output<ov::Node>& w,
                        const Output<ov::Node>* bias,
                        const std::string& name) {
    Output<ov::Node> out = std::make_shared<MatMul>(x, w, false, true);
    if (bias)
        out = std::make_shared<Add>(out, *bias);
    out.get_node_shared_ptr()->set_friendly_name(name);
    return out;
}

Output<ov::Node> split_heads(const Output<ov::Node>& x, int n_head, int head_dim) {
    auto rs = std::make_shared<Reshape>(x, i64c({0, 0, n_head, head_dim}), true);
    return std::make_shared<Transpose>(rs, i32c({0, 2, 1, 3}))->output(0);
}

Output<ov::Node> repeat_kv(const Output<ov::Node>& x, int n_rep) {
    if (n_rep == 1)
        return x;
    auto x_u = std::make_shared<Unsqueeze>(x, i64s(2));          // [B, n_kv, 1, L, d]
    auto sh = std::make_shared<ShapeOf>(x_u, ov::element::i64);  // [5]
    auto g0 = std::make_shared<Gather>(sh, i64c({0}), i64s(0));
    auto g1 = std::make_shared<Gather>(sh, i64c({1}), i64s(0));
    auto g3 = std::make_shared<Gather>(sh, i64c({3}), i64s(0));
    auto g4 = std::make_shared<Gather>(sh, i64c({4}), i64s(0));
    auto target = std::make_shared<Concat>(ov::OutputVector{g0, g1, i64c({n_rep}), g3, g4}, 0);
    auto bcast = std::make_shared<Broadcast>(x_u, target, ov::op::BroadcastType::BIDIRECTIONAL);
    auto final_shape = std::make_shared<Concat>(ov::OutputVector{g0, i64c({-1}), g3, g4}, 0);
    return std::make_shared<Reshape>(bcast, final_shape, false)->output(0);
}

struct AttnOut {
    Output<ov::Node> hidden;
    ov::SinkVector sinks;
};

AttnOut attention_block(const Output<ov::Node>& hidden,
                        const Config& cfg,
                        const Weights& w,
                        int layer_idx,
                        const Output<ov::Node>& cos,
                        const Output<ov::Node>& sin,
                        const Output<ov::Node>& batch_dim,
                        const Output<ov::Node>& beam_idx) {
    const int h = cfg.head_dim;
    const int n_q = cfg.n_head;
    const int n_kv = cfg.n_head_kv;
    const std::string pfx = "model.layers." + std::to_string(layer_idx) + ".self_attn";

    auto get_w = [&](const std::string& suf) -> Output<ov::Node> {
        return w.at(pfx + "." + suf + ".weight");
    };
    auto get_b = [&](const std::string& suf) -> std::optional<Output<ov::Node>> {
        auto k = pfx + "." + suf + ".bias";
        if (w.has(k))
            return w.at(k);
        return std::nullopt;
    };

    auto qb = get_b("q_proj");
    auto kb = get_b("k_proj");
    auto vb = get_b("v_proj");
    auto q = linear(hidden, get_w("q_proj"), qb ? &*qb : nullptr, pfx + ".q_proj");
    auto k = linear(hidden, get_w("k_proj"), kb ? &*kb : nullptr, pfx + ".k_proj");
    auto v = linear(hidden, get_w("v_proj"), vb ? &*vb : nullptr, pfx + ".v_proj");

    q = split_heads(q, n_q, h);
    k = split_heads(k, n_kv, h);
    v = split_heads(v, n_kv, h);

    // Qwen3: per-head RMSNorm on Q and K (along the head_dim axis) before
    // RoPE. The norm weight has shape [head_dim], broadcast across batch x
    // heads x seq. q_norm/k_norm are absent for llama / qwen2 -> no-op.
    if (w.has(pfx + ".q_norm.weight")) {
        q = rms_norm(q, w.at(pfx + ".q_norm.weight"), cfg.rms_eps, pfx + ".q_norm");
    }
    if (w.has(pfx + ".k_norm.weight")) {
        k = rms_norm(k, w.at(pfx + ".k_norm.weight"), cfg.rms_eps, pfx + ".k_norm");
    }

    auto [q_rot, k_rot] = apply_rope(q, k, cos, sin, h);

    auto make_cache = [&](const std::string& kind) -> std::pair<std::shared_ptr<Variable>, Output<ov::Node>> {
        std::string vid = "past_key_values." + std::to_string(layer_idx) + "." + kind + "present." +
                          std::to_string(layer_idx) + ".key";
        VariableInfo vi{ov::PartialShape{-1, n_kv, -1, h}, ov::element::f32, vid};
        auto var = std::make_shared<Variable>(vi);
        auto zero = f32s(0.0f);
        auto init_shape = std::make_shared<Concat>(ov::OutputVector{batch_dim, i64c({n_kv}), i64c({0}), i64c({h})}, 0);
        auto init = std::make_shared<Broadcast>(zero, init_shape);
        auto rv = std::make_shared<ReadValue>(init, var);
        auto gathered = std::make_shared<Gather>(rv, beam_idx, i64s(0));
        return {var, gathered->output(0)};
    };

    auto [k_var, k_past] = make_cache("key");
    auto [v_var, v_past] = make_cache("value");
    auto k_full = std::make_shared<Concat>(ov::OutputVector{k_past, k_rot}, 2);
    auto v_full = std::make_shared<Concat>(ov::OutputVector{v_past, v}, 2);
    auto k_assign = std::make_shared<Assign>(k_full, k_var);
    auto v_assign = std::make_shared<Assign>(v_full, v_var);

    Output<ov::Node> k_rep = repeat_kv(k_full, n_q / n_kv);
    Output<ov::Node> v_rep = repeat_kv(v_full, n_q / n_kv);

    auto attn = std::make_shared<ScaledDotProductAttention>(q_rot, k_rep, v_rep, /*causal=*/true);
    auto attn_t = std::make_shared<Transpose>(attn, i32c({0, 2, 1, 3}));
    auto attn_r = std::make_shared<Reshape>(attn_t, i64c({0, 0, n_q * h}), true);

    auto ob = get_b("o_proj");
    auto out = linear(attn_r, get_w("o_proj"), ob ? &*ob : nullptr, pfx + ".o_proj");
    return {out, ov::SinkVector{k_assign, v_assign}};
}

Output<ov::Node> mlp_block(const Output<ov::Node>& hidden, const Config& /*cfg*/, const Weights& w, int layer_idx) {
    const std::string pfx = "model.layers." + std::to_string(layer_idx) + ".mlp";
    auto gate = linear(hidden, w.at(pfx + ".gate_proj.weight"), nullptr, pfx + ".gate_proj");
    auto up = linear(hidden, w.at(pfx + ".up_proj.weight"), nullptr, pfx + ".up_proj");
    auto silu = std::make_shared<Swish>(gate);
    auto inter = std::make_shared<Multiply>(silu, up);
    return linear(inter, w.at(pfx + ".down_proj.weight"), nullptr, pfx + ".down_proj");
}

// qwen3moe MoE block. Emits the EXACT decomposed subgraph that
// `ov::pass::FuseMOEExperts` matches (see
// src/common/transformations/tests/common_optimizations/fuse_moe_test.cpp::BuildMOE),
// so the MOC pipeline folds it into `ov::op::internal::MOE` and the
// CPU/GPU plugins pick the optimized GatherMatMul / 3-GEMM kernels.
//
// Inputs:
//   residual_3d : pre-norm hidden, shape [B, T, H] (added to MoE output).
//   h_norm_3d   : post-RMSNorm hidden, shape [B, T, H].
// Output:
//   3D tensor [B, T, H] = residual_3d + scatter_sum_over_experts(...).
//
// The pattern requires the residual / accumulator / hidden to be 2D
// ([BT, H]) inside, so we reshape on the way in and on the way out.
Output<ov::Node> moe_block(const Output<ov::Node>& residual_3d,
                           const Output<ov::Node>& h_norm_3d,
                           const Config& cfg,
                           const Weights& w,
                           int layer_idx) {
    const std::string pfx = "model.layers." + std::to_string(layer_idx) + ".mlp";
    const int H = cfg.n_embd;
    const int E = cfg.n_experts;
    const int K = cfg.experts_used;
    OPENVINO_ASSERT(E > 0 && K > 0, "GGUF moe_block: cfg has zero experts/topk for layer ", layer_idx);

    auto reshape_to_2d = [&](const Output<ov::Node>& x) {
        return std::make_shared<Reshape>(x, i64c({-1, H}), /*special_zero=*/false)->output(0);
    };

    auto residual_2d = reshape_to_2d(residual_3d);
    auto h_norm_2d = reshape_to_2d(h_norm_3d);
    // Original 3D shape — used to reshape final 2D result back to [B, T, H].
    auto orig_shape_3d = std::make_shared<ShapeOf>(residual_3d, ov::element::i64);

    // -------- router: MatMul -> Softmax -> TopK --------
    auto router_logits = std::make_shared<MatMul>(h_norm_2d, w.at(pfx + ".gate.weight"), false, true);  // [BT, E]
    router_logits->set_friendly_name(pfx + ".gate");
    auto softmax = std::make_shared<Softmax>(router_logits, /*axis=*/1);

    auto k_const = i64s(K);
    auto topk = std::make_shared<TopK>(softmax,
                                       k_const,
                                       /*axis=*/-1,
                                       TopK::Mode::MAX,
                                       TopK::SortType::SORT_VALUES,
                                       /*idx_type=*/ov::element::i64,
                                       /*stable=*/false);
    auto topk_v = topk->output(0);
    auto topk_i = topk->output(1);

    auto reduce_neg1 = i64c({-1});
    auto sum_topk = std::make_shared<ReduceSum>(topk_v, reduce_neg1, /*keep_dims=*/true);
    auto norm_w = std::make_shared<Divide>(topk_v,
                                           sum_topk,
                                           /*pythondiv=*/true,
                                           ov::op::AutoBroadcastType::NUMPY);

    auto e_const = i64s(E);
    auto on_val = i64s(1);
    auto off_val = i64s(0);
    auto one_hot = std::make_shared<OneHot>(topk_i, e_const, on_val, off_val, /*axis=*/2);
    auto perm = i64c({2, 1, 0});
    auto permute = std::make_shared<Transpose>(one_hot, perm);  // [E, K, BT]

    // -------- routing weights chain (-> 2D [BT*K, 1]) --------
    auto unsq2_axis = i64s(2);
    auto norm_w_unsq = std::make_shared<Unsqueeze>(norm_w, unsq2_axis);  // [BT, K, 1]
    auto shape_unsq_i32 = std::make_shared<ShapeOf>(norm_w_unsq, ov::element::i32);
    auto slice_first2 = std::make_shared<Slice>(shape_unsq_i32, i64c({0}), i64c({2}), i64c({1}), i64c({0}));
    auto reduce_prod_axis = i64s(0);
    auto prod_first2 = std::make_shared<ReduceProd>(slice_first2,
                                                    reduce_prod_axis,
                                                    /*keep_dims=*/true);  // i32 vec [1]
    auto neg1_vec = i32c({-1});
    auto rw_shape = std::make_shared<Concat>(ov::OutputVector{prod_first2, neg1_vec}, 0);
    auto split3_axis = i64s(0);
    auto split3 = std::make_shared<Split>(shape_unsq_i32, split3_axis, /*num_splits=*/3);
    auto routing_weights_2d = std::make_shared<Reshape>(norm_w_unsq,
                                                        rw_shape,
                                                        /*special_zero=*/true);  // [BT*K, 1]

    // -------- accumulator init: zeros_like([BT, H]) --------
    auto target_shape_i32 = std::make_shared<ShapeOf>(residual_2d, ov::element::i32);
    auto zero_scalar = f32s(0.0f);
    Output<ov::Node> accumulator =
        std::make_shared<Broadcast>(zero_scalar, target_shape_i32, ov::op::BroadcastType::NUMPY);

    // -------- shared hidden_states for per-expert gather --------
    // Pattern requires: any_input -> Unsqueeze(axis=0) -> Reshape -> Gather.
    auto unsq0_axis = i64s(0);
    auto h_unsq0 = std::make_shared<Unsqueeze>(h_norm_2d, unsq0_axis);  // [1, BT, H]
    auto h_for_gather = std::make_shared<Reshape>(h_unsq0,
                                                  i64c({-1, H}),
                                                  /*special_zero=*/false);  // [BT, H]

    // -------- per-expert loop --------
    constexpr int64_t INT64_INF = std::numeric_limits<int64_t>::max();
    auto gather_axis_scalar = i64s(0);
    auto squeeze_axis_scalar = i64s(0);
    auto split2_axis = i64s(0);

    for (int e = 0; e < E; ++e) {
        // expert_id_const: rank-1 of size 1 so v8::Gather keeps the K dim.
        auto expert_id_const = i64c({static_cast<int64_t>(e)});
        auto expert_mask_e = std::make_shared<Gather>(permute, expert_id_const,
                                                      gather_axis_scalar);              // [1, K, BT]
        auto squeezed = std::make_shared<Squeeze>(expert_mask_e, squeeze_axis_scalar);  // [K, BT]
        auto nonzero = std::make_shared<NonZero>(squeezed, ov::element::i64);           // [2, M]
        auto split_nz = std::make_shared<Split>(nonzero, split2_axis, /*num_splits=*/2);
        // output(1) = token indices, output(0) = k-position indices (per BuildMOE).
        auto tok_i64 = std::make_shared<Squeeze>(split_nz->output(1), squeeze_axis_scalar);
        auto kpos_i64 = std::make_shared<Squeeze>(split_nz->output(0), squeeze_axis_scalar);
        auto tok_i32 = std::make_shared<Convert>(tok_i64, ov::element::i32);
        auto kpos_i32 = std::make_shared<Convert>(kpos_i64, ov::element::i32);

        // Indices for ScatterElementsUpdate broadcast, reshape to [-1, 1].
        auto idx_2d = std::make_shared<Reshape>(tok_i32,
                                                i64c({-1, 1}),
                                                /*special_zero=*/false);  // [M, 1]
        // Slice accumulator to [1, H] so its ShapeOf gives the per-row broadcast shape.
        auto acc_slice =
            std::make_shared<Slice>(accumulator, i64c({0, 0}), i64c({1, INT64_INF}), i64c({1, 1}), i64c({0, 1}));
        auto acc_slice_shape_i32 = std::make_shared<ShapeOf>(acc_slice, ov::element::i32);
        auto idx_bcast = std::make_shared<Broadcast>(idx_2d, acc_slice_shape_i32, ov::op::BroadcastType::BIDIRECTIONAL);

        // Per-expert hidden gather + 4 reshapes ending with a constant-shape Reshape.
        auto x_gathered = std::make_shared<Gather>(h_for_gather, tok_i32,
                                                   gather_axis_scalar);  // [M, H]
        Output<ov::Node> x_chain = x_gathered;
        for (int rs = 0; rs < 4; ++rs) {
            x_chain = std::make_shared<Reshape>(x_chain,
                                                i64c({-1, H}),
                                                /*special_zero=*/true);
        }

        // Per-expert weights (loaded into Weights map by build_model).
        const std::string ex = pfx + ".experts." + std::to_string(e);
        auto gate_w = w.at(ex + ".gate_proj.weight");
        auto up_w = w.at(ex + ".up_proj.weight");
        auto down_w = w.at(ex + ".down_proj.weight");

        auto gate_mm = std::make_shared<MatMul>(x_chain, gate_w, false, true);
        auto silu = std::make_shared<Swish>(gate_mm);
        auto up_mm = std::make_shared<MatMul>(x_chain, up_w, false, true);
        auto mul_gu = std::make_shared<Multiply>(silu, up_mm, ov::op::AutoBroadcastType::NUMPY);
        auto down_mm = std::make_shared<MatMul>(mul_gu, down_w, false, true);  // [M, H]

        // Per-token routing weight: routing_weights_2d[k_pos + tok_idx * K].
        auto idx_mul = std::make_shared<Multiply>(tok_i32, split3->output(1), ov::op::AutoBroadcastType::NUMPY);
        auto idx_add = std::make_shared<Add>(kpos_i32, idx_mul, ov::op::AutoBroadcastType::NUMPY);
        auto rw_gather = std::make_shared<Gather>(routing_weights_2d, idx_add,
                                                  gather_axis_scalar);  // [M, 1]
        auto rw_reshape = std::make_shared<Reshape>(rw_gather,
                                                    i64c({0, 1}),
                                                    /*special_zero=*/true);  // [M, 1]
        auto weighted = std::make_shared<Multiply>(down_mm, rw_reshape, ov::op::AutoBroadcastType::NUMPY);
        auto val_bcast =
            std::make_shared<Broadcast>(weighted, acc_slice_shape_i32, ov::op::BroadcastType::BIDIRECTIONAL);

        accumulator = std::make_shared<ScatterElementsUpdate>(accumulator,
                                                              idx_bcast,
                                                              val_bcast,
                                                              gather_axis_scalar,
                                                              ScatterElementsUpdate::Reduction::SUM,
                                                              /*use_init_val=*/true);
    }

    // last_reshape: 2D no-op reshape using ShapeOf(h_norm_2d), then residual add (2D).
    auto target_shape_i64 = std::make_shared<ShapeOf>(h_norm_2d, ov::element::i64);
    auto last_reshape = std::make_shared<Reshape>(accumulator,
                                                  target_shape_i64,
                                                  /*special_zero=*/false);
    auto final_2d = std::make_shared<Add>(residual_2d, last_reshape, ov::op::AutoBroadcastType::NUMPY);
    final_2d->set_friendly_name(pfx + ".moe");

    // Reshape back to [B, T, H] so the surrounding decoder code stays unchanged.
    return std::make_shared<Reshape>(final_2d,
                                     orig_shape_3d,
                                     /*special_zero=*/false)
        ->output(0);
}

}  // namespace

std::shared_ptr<ov::Model> build_model(const GGUFFile& file) {
    auto cfg = read_config(file);
    const bool do_rope_reorder = (cfg.arch == "llama");

    // Build a graph node per relevant tensor. Q4_0 / Q4_1 / Q8_0 are kept as
    // native u4 / i8 Constants with a Convert + [Subtract] + Multiply
    // decompression chain (preserves on-disk compression in the produced IR);
    // other quantized variants dequantize to FP16 at load time.
    Weights weights;
    for (const auto& td : file.tensors()) {
        // qwen3moe per-expert weights are 3D (`[E, out, in]` after the GGUF
        // reader's row-major reversal). They have no entry in `gguf_to_hf`;
        // we slice them into N per-expert 2D constants below.
        const bool is_moe_3d = (td.dims.size() == 3) && (td.name.find("ffn_gate_exps.weight") != std::string::npos ||
                                                         td.name.find("ffn_up_exps.weight") != std::string::npos ||
                                                         td.name.find("ffn_down_exps.weight") != std::string::npos);
        if (is_moe_3d)
            continue;

        std::string hf = gguf_to_hf(td.name);
        if (hf.empty())
            continue;
        const bool is_qk = (hf.size() > 14 && (hf.compare(hf.size() - 14, 14, ".q_proj.weight") == 0 ||
                                               hf.compare(hf.size() - 14, 14, ".k_proj.weight") == 0)) ||
                           (hf.size() > 12 && (hf.compare(hf.size() - 12, 12, ".q_proj.bias") == 0 ||
                                               hf.compare(hf.size() - 12, 12, ".k_proj.bias") == 0));
        const bool reorder = do_rope_reorder && is_qk;
        weights.by_hf_name.emplace(hf, build_weight_node(td, file.tensor_raw(td), hf, reorder, cfg.head_dim));
    }

    // ---- qwen3moe: split 3D per-expert tensors into N 2D constants. -------
    // Layout (post-reverse): td.dims = [E, out, in]. Each expert slab is a
    // contiguous run of bytes since experts is the slowest axis.
    if (cfg.arch == "qwen3moe") {
        struct MoEMap {
            const char* gguf_suffix;
            const char* hf_proj;
        };
        constexpr MoEMap kMoEMap[] = {
            {"ffn_gate_exps.weight", "gate_proj"},
            {"ffn_up_exps.weight", "up_proj"},
            {"ffn_down_exps.weight", "down_proj"},
        };
        for (const auto& td : file.tensors()) {
            if (td.dims.size() != 3)
                continue;
            // Resolve which projection this is and which layer index it belongs to.
            const char* hf_proj = nullptr;
            for (const auto& m : kMoEMap) {
                if (td.name.size() > std::strlen(m.gguf_suffix) &&
                    td.name.compare(td.name.size() - std::strlen(m.gguf_suffix),
                                    std::strlen(m.gguf_suffix),
                                    m.gguf_suffix) == 0) {
                    hf_proj = m.hf_proj;
                    break;
                }
            }
            if (!hf_proj)
                continue;
            OPENVINO_ASSERT(td.name.rfind("blk.", 0) == 0, "GGUF qwen3moe: unexpected MoE tensor name '", td.name, "'");
            const auto dot = td.name.find('.', 4);
            OPENVINO_ASSERT(dot != std::string::npos, "GGUF qwen3moe: cannot parse layer idx from '", td.name, "'");
            const int layer = std::stoi(td.name.substr(4, dot - 4));

            const int64_t n_experts = static_cast<int64_t>(td.dims[0]);
            OPENVINO_ASSERT(n_experts == cfg.n_experts,
                            "GGUF qwen3moe: tensor '",
                            td.name,
                            "' has ",
                            n_experts,
                            " experts but config has ",
                            cfg.n_experts);
            // Build a synthetic 2D descriptor per expert sharing the file's
            // raw bytes — `build_weight_node` only inspects `dims`, `type` and
            // `name`, plus the `raw` pointer we pass directly.
            TensorDescriptor td2d{};
            td2d.type = td.type;
            td2d.offset = td.offset;  // unused: we pass raw pointer ourselves
            td2d.dims = {td.dims[1], td.dims[2]};

            // Per-expert byte stride: total bytes / n_experts. The reader
            // guarantees block alignment (n_elements % qk == 0); per-expert
            // element count is n_total / n_experts which is also block-aligned
            // for qwen3moe (ffn_inter * hidden divisible by 256 for K-quants).
            const uint8_t* base = file.tensor_raw(td);
            // total bytes of the 3D tensor; divide evenly across experts.
            // We compute it via gguf_compress's internal logic, which is not
            // exposed — instead recompute here from dims and ggml block
            // geometry (see gguf_compress.cpp::geom).
            auto block_geom = [&](ggml_type t) -> std::pair<int, size_t> {
                switch (t) {
                case GGML_TYPE_F32:
                    return {1, 4};
                case GGML_TYPE_F16:
                    return {1, 2};
                case GGML_TYPE_BF16:
                    return {1, 2};
                case GGML_TYPE_Q4_0:
                    return {32, 18};
                case GGML_TYPE_Q4_1:
                    return {32, 20};
                case GGML_TYPE_Q5_0:
                    return {32, 22};
                case GGML_TYPE_Q5_1:
                    return {32, 24};
                case GGML_TYPE_Q8_0:
                    return {32, 34};
                case GGML_TYPE_Q2_K:
                    return {256, 84};
                case GGML_TYPE_Q3_K:
                    return {256, 110};
                case GGML_TYPE_Q4_K:
                    return {256, 144};
                case GGML_TYPE_Q5_K:
                    return {256, 176};
                case GGML_TYPE_Q6_K:
                    return {256, 210};
                }
                OPENVINO_THROW("GGUF qwen3moe: unsupported MoE tensor type ",
                               static_cast<uint32_t>(t),
                               " for '",
                               td.name,
                               "'");
            };
            const auto [qk, block_bytes] = block_geom(td.type);
            const size_t per_expert_elems = static_cast<size_t>(td.dims[1]) * static_cast<size_t>(td.dims[2]);
            OPENVINO_ASSERT(per_expert_elems % static_cast<size_t>(qk) == 0,
                            "GGUF qwen3moe: per-expert elements (",
                            per_expert_elems,
                            ") not divisible by block size ",
                            qk,
                            " for tensor '",
                            td.name,
                            "'");
            const size_t per_expert_bytes = (per_expert_elems / static_cast<size_t>(qk)) * block_bytes;

            for (int64_t e = 0; e < n_experts; ++e) {
                const std::string hf = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(e) +
                                       "." + hf_proj + ".weight";
                td2d.name = hf;  // for friendly_name inside build_weight_node
                const uint8_t* expert_raw = base + static_cast<size_t>(e) * per_expert_bytes;
                weights.by_hf_name.emplace(hf,
                                           build_weight_node(td2d,
                                                             expert_raw,
                                                             hf,
                                                             /*row_reorder_rope=*/false,
                                                             /*head_dim=*/0));
            }
        }
    }

    // Inputs.
    auto input_ids = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});

    auto attention_mask = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    attention_mask->set_friendly_name("attention_mask");
    attention_mask->output(0).set_names({"attention_mask"});

    auto position_ids = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).set_names({"position_ids"});

    auto beam_idx = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    beam_idx->output(0).set_names({"beam_idx"});

    // batch_dim as i64 vector of length 1: [B].
    auto in_shape = std::make_shared<ShapeOf>(input_ids, ov::element::i64);
    auto batch_dim = std::make_shared<Gather>(in_shape, i64c({0}), i64s(0));

    // Embedding.
    auto embed_const = weights.at("model.embed_tokens.weight");
    auto ids_i32 = std::make_shared<Convert>(input_ids, ov::element::i32);
    Output<ov::Node> hidden = std::make_shared<Gather>(embed_const, ids_i32, i32s(0));

    // Shared RoPE cos/sin.
    auto [cos, sin] = rope_cos_sin(position_ids, cfg.head_dim, cfg.rope_base);

    ov::SinkVector sinks;
    for (int i = 0; i < cfg.n_layer; ++i) {
        Output<ov::Node> residual = hidden;
        auto h_norm = rms_norm(hidden,
                               weights.at("model.layers." + std::to_string(i) + ".input_layernorm.weight"),
                               cfg.rms_eps,
                               "model.layers." + std::to_string(i) + ".input_layernorm");
        auto attn = attention_block(h_norm, cfg, weights, i, cos, sin, batch_dim, beam_idx);
        sinks.insert(sinks.end(), attn.sinks.begin(), attn.sinks.end());
        hidden = std::make_shared<Add>(residual, attn.hidden)->output(0);

        residual = hidden;
        h_norm = rms_norm(hidden,
                          weights.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"),
                          cfg.rms_eps,
                          "model.layers." + std::to_string(i) + ".post_attention_layernorm");
        if (cfg.arch == "qwen3moe") {
            // moe_block emits the residual Add internally so it matches the
            // FuseMOEExperts pattern. It returns the post-residual 3D tensor.
            hidden = moe_block(residual, h_norm, cfg, weights, i);
        } else {
            auto mlp = mlp_block(h_norm, cfg, weights, i);
            hidden = std::make_shared<Add>(residual, mlp)->output(0);
        }
    }

    hidden = rms_norm(hidden, weights.at("model.norm.weight"), cfg.rms_eps, "model.norm");

    Output<ov::Node> logits;
    if (weights.has("lm_head.weight")) {
        logits = linear(hidden, weights.at("lm_head.weight"), nullptr, "lm_head");
    } else {
        logits = std::make_shared<MatMul>(hidden, embed_const, false, true);
    }
    logits.get_node_shared_ptr()->set_friendly_name("logits");
    logits.set_names({"logits"});

    auto result = std::make_shared<Result>(logits);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             sinks,
                                             ov::ParameterVector{input_ids, attention_mask, position_ids, beam_idx},
                                             "gguf_" + cfg.arch);

    // Hints used by openvino.genai.
    model->set_rt_info(ov::element::f16, "runtime_options", "KV_CACHE_PRECISION");
    model->set_rt_info(8.0f, "runtime_options", "ACTIVATIONS_SCALE_FACTOR");
    return model;
}

// =========================================================================
// ============ Qwen3-VL multi-modal projector (mmproj) ====================
// =========================================================================
//
// A separate OV Model is built from the standalone *mmproj* GGUF (CLIP arch
// with `projector_type = qwen3vl_merger`). It encodes a preprocessed image
// tensor through a 24-layer pre-LN ViT, then projects each spatial-merged
// token to the LLM hidden size, while also emitting three "DeepStack" feature
// streams produced by additional MLPs reading from layers 5, 11 and 17.
//
// The model contract is:
//   inputs : pixel_values  [B, 3, image_size, image_size]   (f32)
//   outputs: vision_features [B, N/m^2, proj_dim]            (f32)
//            deepstack_<k>   [B, N/m^2, proj_dim]   (one per tap k, f32)
// where N = (image_size/patch_size)^2 and m = spatial_merge_size.
//
// Notes / simplifications for this PoC:
//   * `image_size` is fixed (the trained value, e.g. 768). The Python
//     pre-processor must resize images to that exact resolution. Variable
//     resolution support would require interpolating the absolute position
//     embedding; intentionally left out for now.
//   * If the GGUF carries `v.patch_embd.weight.1` (a second temporal patch
//     kernel used by Qwen3-VL when treating frames as 2-frame clips), we
//     apply it to the same image and sum, which is what the reference impl
//     does for single images.
//   * No RoPE in the ViT: position information comes entirely from the
//     learned absolute `v.position_embd.weight`.

namespace {

// Shared helpers for the mmproj builder. These mirror the fp32 pieces of the
// LLM builder above but live separately because they don't need the RoPE /
// stateful-cache machinery.

// Read a metadata key with a fallback for "either int or float depending on
// when the GGUF was written". Returns an int64_t that must be cast.
int64_t meta_int(const GGUFFile& f, const std::string& key) {
    auto it = f.metadata().find(key);
    OPENVINO_ASSERT(it != f.metadata().end(), "GGUF mmproj: missing metadata key '", key, "'");
    return it->second.get<int64_t>();
}
float meta_float(const GGUFFile& f, const std::string& key) {
    auto it = f.metadata().find(key);
    OPENVINO_ASSERT(it != f.metadata().end(), "GGUF mmproj: missing metadata key '", key, "'");
    return it->second.get<float>();
}
std::vector<int> meta_bool_array_indices(const GGUFFile& f, const std::string& key) {
    auto it = f.metadata().find(key);
    OPENVINO_ASSERT(it != f.metadata().end(), "GGUF mmproj: missing metadata key '", key, "'");
    OPENVINO_ASSERT(std::holds_alternative<MetaArray>(it->second.v), "GGUF mmproj: expected array for '", key, "'");
    const auto& arr = std::get<MetaArray>(it->second.v);
    std::vector<int> idx;
    for (size_t i = 0; i < arr.size(); ++i) {
        if (std::holds_alternative<bool>(arr[i].v) && std::get<bool>(arr[i].v))
            idx.push_back(static_cast<int>(i));
    }
    return idx;
}

// LayerNorm over the last dim. Matches torch.nn.LayerNorm semantics.
Output<ov::Node> layer_norm(const Output<ov::Node>& x,
                            const Output<ov::Node>& w,
                            const Output<ov::Node>& b,
                            float eps,
                            const std::string& name) {
    auto mvn = std::make_shared<MVN>(x, i32c({-1}), /*normalize_variance=*/true, eps, op::MVNEpsMode::INSIDE_SQRT);
    auto scaled = std::make_shared<Multiply>(mvn, w);
    auto out = std::make_shared<Add>(scaled, b);
    out->set_friendly_name(name);
    return out;
}

// Linear with bias: y = x @ W^T + b (W is stored as [out, in]).
Output<ov::Node> linear_b(const Output<ov::Node>& x,
                          const Output<ov::Node>& w,
                          const Output<ov::Node>& b,
                          const std::string& name) {
    auto mm = std::make_shared<MatMul>(x, w, false, true);
    auto out = std::make_shared<Add>(mm, b);
    out->set_friendly_name(name);
    return out;
}

// Spatial 2x2 merge along the token dim:
//   [B, gh*gw, h]  ->  [B, (gh/m)*(gw/m), m*m*h]
// Each output token concatenates a row-major mxm block of input tokens.
Output<ov::Node> spatial_merge(const Output<ov::Node>& x, int gh, int gw, int hidden, int m) {
    OPENVINO_ASSERT(gh % m == 0 && gw % m == 0, "GGUF mmproj: grid not divisible by merge size");
    // [B, gh, gw, h]
    auto r1 = std::make_shared<Reshape>(x, i64c({0, gh, gw, hidden}), true);
    // [B, gh/m, m, gw/m, m, h]
    auto r2 = std::make_shared<Reshape>(r1, i64c({0, gh / m, m, gw / m, m, hidden}), true);
    // [B, gh/m, gw/m, m, m, h]   (group rows-of-blocks together)
    auto t = std::make_shared<Transpose>(r2, i32c({0, 1, 3, 2, 4, 5}));
    // [B, (gh/m)*(gw/m), m*m*h]
    auto r3 = std::make_shared<Reshape>(t, i64c({0, (gh / m) * (gw / m), m * m * hidden}), true);
    return r3;
}

struct VitCfg {
    int n_layer{}, n_head{}, hidden{}, head_dim{}, ff_dim{};
    int patch_size{}, image_size{}, num_pos{};
    int merge_size{};
    int proj_dim{};
    float ln_eps{};
    std::vector<int> deepstack_layers;
};

VitCfg read_vit_cfg(const GGUFFile& f) {
    auto arch = f.metadata().find("general.architecture");
    OPENVINO_ASSERT(arch != f.metadata().end() && arch->second.get<std::string>() == "clip",
                    "GGUF mmproj builder: expected general.architecture == 'clip'");
    auto proj = f.metadata().find("clip.projector_type");
    OPENVINO_ASSERT(proj != f.metadata().end() && proj->second.get<std::string>() == "qwen3vl_merger",
                    "GGUF mmproj builder: only 'qwen3vl_merger' projector is supported");

    VitCfg c;
    c.n_layer = static_cast<int>(meta_int(f, "clip.vision.block_count"));
    c.n_head = static_cast<int>(meta_int(f, "clip.vision.attention.head_count"));
    c.hidden = static_cast<int>(meta_int(f, "clip.vision.embedding_length"));
    OPENVINO_ASSERT(c.hidden % c.n_head == 0, "GGUF mmproj: hidden ", c.hidden, " not divisible by n_head ", c.n_head);
    c.head_dim = c.hidden / c.n_head;
    c.ff_dim = static_cast<int>(meta_int(f, "clip.vision.feed_forward_length"));
    c.patch_size = static_cast<int>(meta_int(f, "clip.vision.patch_size"));
    c.image_size = static_cast<int>(meta_int(f, "clip.vision.image_size"));
    OPENVINO_ASSERT(c.image_size % c.patch_size == 0,
                    "GGUF mmproj: image_size ",
                    c.image_size,
                    " not divisible by patch_size ",
                    c.patch_size);
    const int g = c.image_size / c.patch_size;
    c.num_pos = g * g;
    c.merge_size = static_cast<int>(meta_int(f, "clip.vision.spatial_merge_size"));
    c.proj_dim = static_cast<int>(meta_int(f, "clip.vision.projection_dim"));
    c.ln_eps = meta_float(f, "clip.vision.attention.layer_norm_epsilon");
    c.deepstack_layers = meta_bool_array_indices(f, "clip.vision.is_deepstack_layers");
    return c;
}

}  // namespace

std::shared_ptr<ov::Model> build_mmproj_model(const GGUFFile& file) {
    const VitCfg cfg = read_vit_cfg(file);
    const int g = cfg.image_size / cfg.patch_size;  // grid side: e.g. 48
    const int N = g * g;                            // tokens per image: 2304

    // Materialize all weight tensors as f32 Output<Node>s. The mmproj is
    // typically F16 / F32 with no quantization, so build_weight_node here
    // just produces a Constant + Convert(f32) chain (no decompression needed).
    Weights weights;
    for (const auto& td : file.tensors()) {
        weights.by_hf_name.emplace(td.name, build_weight_node(td, file.tensor_raw(td), td.name));
    }

    // ----- input ------------------------------------------------------- //
    // Fixed spatial size (must match `clip.vision.image_size`).  Caller is
    // responsible for resizing + normalizing the image with image_mean/std
    // before feeding to this model.
    auto pixel_values =
        std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1, 3, cfg.image_size, cfg.image_size});
    pixel_values->set_friendly_name("pixel_values");
    pixel_values->output(0).set_names({"pixel_values"});

    // ----- patch embedding -------------------------------------------- //
    // Conv2d(3 -> hidden, kernel=patch_size, stride=patch_size).
    auto conv2d = [&](const Output<ov::Node>& w_) {
        return std::make_shared<Convolution>(
            pixel_values,
            w_,
            ov::Strides{static_cast<size_t>(cfg.patch_size), static_cast<size_t>(cfg.patch_size)},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});
    };

    Output<ov::Node> conv = conv2d(weights.at("v.patch_embd.weight"));
    if (weights.has("v.patch_embd.weight.1")) {
        // Qwen3-VL temporal patching: a second 16x16 kernel for the "next"
        // frame. For a single still image we apply it to the same pixels and
        // sum (matches treating the image as a 2-frame clip with both frames
        // identical).
        auto conv1 = conv2d(weights.at("v.patch_embd.weight.1"));
        conv = std::make_shared<Add>(conv, conv1);
    }
    // Add bias [hidden] broadcast over the [B, hidden, gh, gw] feature map.
    auto patch_bias = weights.at("v.patch_embd.bias");
    auto patch_bias_r = std::make_shared<Reshape>(patch_bias, i64c({1, cfg.hidden, 1, 1}), false);
    Output<ov::Node> patches = std::make_shared<Add>(conv, patch_bias_r);

    // [B, hidden, gh, gw] -> [B, gh, gw, hidden] -> [B, N, hidden]
    auto perm = std::make_shared<Transpose>(patches, i32c({0, 2, 3, 1}));
    Output<ov::Node> x = std::make_shared<Reshape>(perm, i64c({0, N, cfg.hidden}), true);

    // ----- learned absolute position embedding ------------------------ //
    // v.position_embd.weight is stored as [num_pos, hidden] (after our row-
    // major reversal); broadcast over the batch axis with an Unsqueeze.
    auto pos = weights.at("v.position_embd.weight");
    auto pos_unsq = std::make_shared<Unsqueeze>(pos, i64s(0));  // [1, num_pos, hidden]
    x = std::make_shared<Add>(x, pos_unsq);

    // ----- 24-layer ViT (pre-LN, GELU FFN) ---------------------------- //
    std::map<int, Output<ov::Node>> deepstack_taps;
    const std::set<int> tap_set(cfg.deepstack_layers.begin(), cfg.deepstack_layers.end());

    for (int i = 0; i < cfg.n_layer; ++i) {
        const std::string p = "v.blk." + std::to_string(i) + ".";

        // ---- attention ---- //
        auto h = layer_norm(x, weights.at(p + "ln1.weight"), weights.at(p + "ln1.bias"), cfg.ln_eps, p + "ln1");

        auto qkv = linear_b(h, weights.at(p + "attn_qkv.weight"), weights.at(p + "attn_qkv.bias"), p + "attn_qkv");

        // Split last dim 3*hidden -> 3 x hidden.
        auto split = std::make_shared<Split>(qkv, i64s(-1), 3);
        auto reshape_heads = [&](const Output<ov::Node>& t) -> Output<ov::Node> {
            auto rs = std::make_shared<Reshape>(t, i64c({0, 0, cfg.n_head, cfg.head_dim}), true);
            return std::make_shared<Transpose>(rs, i32c({0, 2, 1, 3}))->output(0);
        };
        Output<ov::Node> q = reshape_heads(split->output(0));
        Output<ov::Node> k = reshape_heads(split->output(1));
        Output<ov::Node> v_ = reshape_heads(split->output(2));

        auto attn = std::make_shared<ScaledDotProductAttention>(q, k, v_, /*causal=*/false);

        // [B, n_head, N, head_dim] -> [B, N, n_head, head_dim] -> [B, N, hidden]
        auto attn_t = std::make_shared<Transpose>(attn, i32c({0, 2, 1, 3}));
        auto attn_r = std::make_shared<Reshape>(attn_t, i64c({0, 0, cfg.hidden}), true);

        auto attn_out =
            linear_b(attn_r, weights.at(p + "attn_out.weight"), weights.at(p + "attn_out.bias"), p + "attn_out");
        x = std::make_shared<Add>(x, attn_out);

        // ---- feed-forward ---- //
        auto h2 = layer_norm(x, weights.at(p + "ln2.weight"), weights.at(p + "ln2.bias"), cfg.ln_eps, p + "ln2");
        auto up = linear_b(h2, weights.at(p + "ffn_up.weight"), weights.at(p + "ffn_up.bias"), p + "ffn_up");
        auto act = std::make_shared<Gelu>(up, op::GeluApproximationMode::ERF);
        auto down = linear_b(act, weights.at(p + "ffn_down.weight"), weights.at(p + "ffn_down.bias"), p + "ffn_down");
        x = std::make_shared<Add>(x, down);

        if (tap_set.count(i))
            deepstack_taps[i] = x;
    }

    // ----- post-LN ----- //
    auto post_x = layer_norm(x, weights.at("v.post_ln.weight"), weights.at("v.post_ln.bias"), cfg.ln_eps, "v.post_ln");

    // ----- spatial merge + main projector ----- //
    auto merged = spatial_merge(post_x, g, g, cfg.hidden, cfg.merge_size);
    auto mm0 = linear_b(merged, weights.at("mm.0.weight"), weights.at("mm.0.bias"), "mm.0");
    auto mm0_act = std::make_shared<Gelu>(mm0, op::GeluApproximationMode::ERF);
    auto main = linear_b(mm0_act, weights.at("mm.2.weight"), weights.at("mm.2.bias"), "vision_features");
    main.set_names({"vision_features"});

    // ----- DeepStack mergers (one per tapped layer) ----- //
    ResultVector results;
    results.push_back(std::make_shared<Result>(main));
    for (int k : cfg.deepstack_layers) {
        const std::string p = "v.deepstack." + std::to_string(k) + ".";
        const std::string out_name = "deepstack_" + std::to_string(k);

        auto tap_merged = spatial_merge(deepstack_taps.at(k), g, g, cfg.hidden, cfg.merge_size);
        auto h_k = layer_norm(tap_merged,
                              weights.at(p + "norm.weight"),
                              weights.at(p + "norm.bias"),
                              cfg.ln_eps,
                              out_name + "/norm");
        auto fc1 = linear_b(h_k, weights.at(p + "fc1.weight"), weights.at(p + "fc1.bias"), out_name + "/fc1");
        auto fc1_act = std::make_shared<Gelu>(fc1, op::GeluApproximationMode::ERF);
        auto fc2 = linear_b(fc1_act, weights.at(p + "fc2.weight"), weights.at(p + "fc2.bias"), out_name);
        fc2.set_names({out_name});
        results.push_back(std::make_shared<Result>(fc2));
    }

    auto model = std::make_shared<ov::Model>(results, ov::ParameterVector{pixel_values}, "gguf_qwen3vl_mmproj");
    return model;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
