// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Typed RoPE configuration, replacing the raw gguf `op_params`/`rope_params` int32 array
// the translators used to dereference by byte offset. A decoder is responsible for
// producing this (the cgraph decoder by parsing ggml's layout, the gguf-builder decoder
// by construction), so the op translators never need to know ggml's memory layout.
struct RopeConfig {
    int n_dims = 0;
    int n_ctx_orig = 0;
    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float ext_factor = 0.0f;
    float attn_factor = 0.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
};

class GgufDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const override = 0;

    // Per-node typed attribute access. This is the mechanism the op translators use to
    // read scalar operation parameters (e.g. "eps", "scale", "bias", "max_bias",
    // "swapped", "rope_config") without dereferencing ggml's raw op_params layout.
    virtual ov::Any get_attribute(int node_idx, const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    // Byte offset of an input that is a gguf VIEW into a larger tensor (0 when the input
    // is not a view). Replaces the raw `get_input_op_params(...)[0]` read: a view's start
    // offset is a single semantic scalar the decoder knows (by parsing ggml, or by
    // construction in the gguf-builder path), so translators never touch ggml's op_params
    // layout. Translators convert bytes -> elements using get_input_stride as needed.
    virtual int64_t get_input_view_offset(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const override = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const override = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const override = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const override = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    // Model-level RoPE configuration, used by TranslateSession::preprocess to pre-build
    // the shared rope sin/cos. has_rope() == false means the model uses no RoPE.
    virtual bool has_rope() const = 0;

    virtual RopeConfig get_rope_config() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual bool is_swa_layer(int layer) const = 0;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
