// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <openvino/core/any.hpp>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace gguf {

// Typed RoPE configuration, replacing the raw gguf `op_params`/`rope_params` int32 array
// the translators used to dereference by byte offset. A decoder is responsible for
// producing this (the llama.cpp cgraph decoder by parsing ggml's layout, a future
// gguf-file decoder by construction), so the op translators never need to know ggml's
// memory layout.
struct RopeConfig {
    int n_dims = 0;  // 0 means the model uses no RoPE (replaces a separate has_rope() query)
    int n_ctx_orig = 0;
    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float ext_factor = 0.0f;
    float attn_factor = 0.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
    // When true, each ROPE op builds its own sin/cos from its per-op config (e.g. gemma4, where
    // SWA and global layers use different n_dims), so the shared rope_cos/rope_sin table that
    // TranslateSession::preprocess would otherwise pre-build is skipped.
    bool per_op = false;
};

// Decoder interface consumed by the gguf frontend translators.
//
// Following the established OpenVINO frontend pattern (cf. the PyTorch frontend), a GgufDecoder
// is node-scoped: visit_subgraph hands the visitor a fresh decoder bound to a single node, and
// every per-node accessor (get_attribute, get_input_*, get_output_*, get_op_*) refers to that
// node -- no node index is threaded through. The same object type, queried at model scope (the
// instance returned by InputModel::get_decoder and iterated by visit_subgraph), answers the
// model-level questions (get_model_inputs, get_model_output_names, get_rope_config, ...).
//
// This is a typed, ggml-free interface: operation parameters are exposed through
// get_attribute(name) / get_input_view_offset / RopeConfig rather than as raw ggml `op_params`
// int32 arrays. A concrete decoder (e.g. the llama.cpp cgraph decoder) only has to translate
// ggml's layout into these typed accessors -- the op translators here never touch ggml memory.
class GgufDecoder : public DecoderBase {
public:
    // Per-node typed attribute access (the bound node). The op translators use this to read
    // scalar operation parameters (e.g. "eps", "scale", "bias", "swapped", "rope_config")
    // without dereferencing ggml's raw op_params layout.
    ov::Any get_attribute(const std::string& name) const override = 0;

    virtual PartialShape get_input_shape(const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(const std::string& name) const = 0;

    // Byte offset of an input that is a gguf VIEW into a larger tensor (0 when the input
    // is not a view). Replaces the raw `get_input_op_params(...)[0]` read: a view's start
    // offset is a single semantic scalar the decoder knows (by parsing ggml, or by
    // construction), so translators never touch ggml's op_params layout. Translators
    // convert bytes -> elements using get_input_stride as needed.
    virtual int64_t get_input_view_offset(const std::string& name) const = 0;

    virtual element::Type get_input_type(const std::string& name) const = 0;

    size_t get_input_size() const override = 0;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override = 0;

    virtual std::vector<std::string> get_input_names() const = 0;

    virtual PartialShape get_output_shape() const = 0;

    virtual element::Type get_output_type() const = 0;

    virtual std::vector<std::string> get_output_names() const = 0;

    const std::string& get_op_type() const override = 0;

    const std::string& get_op_name() const override = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> node_visitor) const = 0;

    virtual int get_op_case() const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    // Pre-built OpenVINO weight nodes, keyed by ggml tensor name (e.g. "blk.0.attn_q.weight").
    // The native .gguf builder path dequantizes weights up front and returns them here; the
    // TranslateSession seeds them into the tensor map before the graph walk. A decoder that
    // instead surfaces each weight as a GGML_OP_NONE leaf carrying a "data" attribute (the
    // llama.cpp cgraph path) returns an empty map -- translate_weight then builds the node.
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;

    // KV-cache parameter/result friendly-name pairs consumed by ov::pass::MakeStateful when the
    // model is stateful (empty otherwise): each entry maps a cache Parameter name to its Result
    // name so MakeStateful turns them into a ReadValue/Assign pair.
    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    // Execution-mode flags. A static model uses fixed token length (NPU-friendly, SqueezeMatmul);
    // a stateful model carries an OpenVINO KV cache (ReadValue/Assign, beam_idx) and is what
    // OpenVINO GenAI consumes. The cgraph/decoder-replay path reports both false (stateless).
    virtual bool is_static() const = 0;
    virtual bool is_stateful() const = 0;

    // Whether attention layer `layer` uses a sliding window (gpt-oss/gemma SWA pattern). Used to
    // route the per-layer sliced attention mask. Returns false for models without SWA.
    virtual bool is_swa_layer(int layer) const = 0;

    // GGUF tokenizer metadata (the file's `tokenizer.*` keys), attached to the converted model's
    // rt_info so a downstream consumer (OpenVINO GenAI) can build the tokenizer without reopening
    // the .gguf. Empty when the decoder carries no tokenizer metadata.
    virtual const ov::AnyMap& get_tokenizer_config() const = 0;

    // RoPE configuration, exposed through get_attribute<RopeConfig>("rope_config"):
    //   - at model scope, used by TranslateSession::preprocess to pre-build the shared rope
    //     sin/cos table (skipped when RopeConfig::n_dims == 0, i.e. no RoPE, or per_op == true);
    //   - at node scope, the ROPE translator reads the same key for the op's own config.
    //
    // NOTE: weights may also be surfaced as GGML_OP_NONE leaves (the cgraph path): the decoder
    // marks such a leaf via get_attribute<ov::Tensor>("data") + get_attribute<std::string>(
    // "quant_type") + get_output_shape(), and translate_weight builds the node. The native .gguf
    // builder path uses get_model_weights() instead and returns those leaves' data pre-dequantized.
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
