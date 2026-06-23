// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/adapt_to_genai.hpp"

#include <memory>
#include <unordered_map>

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

namespace {

using std::make_shared;

// f16 lowest, matches genai's causal-mask "-inf" fill (see tests/genai_io_adapter.py).
constexpr float NEG_INF = -65504.0f;

std::shared_ptr<ov::op::v0::Constant> const_i64(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

// Find a Parameter whose output tensor names (or friendly name) match `name`.
std::shared_ptr<ov::op::v0::Parameter> find_param(const std::shared_ptr<ov::Model>& model,
                                                  const std::string& name) {
    for (const auto& p : model->get_parameters()) {
        const auto& names = p->output(0).get_names();
        if (names.count(name) || p->get_friendly_name() == name) {
            return p;
        }
    }
    return nullptr;
}

void name_output(const ov::Output<ov::Node>& out, const std::string& name) {
    out.get_node_shared_ptr()->set_friendly_name(name);
    out.get_node_shared_ptr()->output(0).set_names({name});
}

// Largest attention head size across the stateful KV caches (the ReadValue last dim). The
// frontend emits f16 KV caches mirroring llama.cpp, but the CPU plugin defaults
// KV_CACHE_PRECISION to u8 (dynamic-quantized) -- faster and accurate enough for the common
// head sizes used by llama/qwen/phi3/gpt-oss (64-128). For large head sizes the u8
// quantization injects enough per-step error to compound across autoregressive decode into
// divergence and eventually NaN (observed on gemma4, global-attention head_size=512).
int64_t max_kv_cache_head_size(const std::shared_ptr<ov::Model>& model) {
    int64_t max_hs = 0;
    for (const auto& op : model->get_ops()) {
        if (!ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
            continue;
        }
        const auto& ps = op->get_output_partial_shape(0);
        if (ps.rank().is_static() && ps[ps.rank().get_length() - 1].is_static()) {
            max_hs = std::max(max_hs, ps[ps.rank().get_length() - 1].get_length());
        }
    }
    return max_hs;
}

}  // namespace

bool AdaptToGenAI::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(m_mode == InputMode::IdsToLogits,
                    "[gguf] AdaptToGenAI: only InputMode::IdsToLogits is implemented; "
                    "EmbedsToLogits (VLM language model) is reserved for future work.");

    // The gguf inputs we rewire. inp_tokens/inp_pos/self_kq_mask/token_len_per_seq are
    // required; if they are absent the model is not a gguf-IO model (e.g. already adapted),
    // so this pass is a no-op.
    auto inp_tokens = find_param(model, "inp_tokens");
    auto inp_pos = find_param(model, "inp_pos");
    auto self_kq_mask = find_param(model, "self_kq_mask");
    auto token_len_per_seq = find_param(model, "token_len_per_seq");
    if (!inp_tokens || !inp_pos || !self_kq_mask || !token_len_per_seq) {
        return false;
    }

    // ---- new genai inputs: input_ids / attention_mask / position_ids [b, seq] i64 ----
    auto input_ids = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    name_output(input_ids, "input_ids");
    auto attention_mask = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    name_output(attention_mask, "attention_mask");
    auto position_ids = make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    name_output(position_ids, "position_ids");

    // beam_idx already exists on the frontend model (i32 [D]); genai sets it via
    // set_tensor("beam_idx"). Keep the existing Parameter so its wiring (if any) is preserved.
    auto beam_idx = find_param(model, "beam_idx");
    OPENVINO_ASSERT(beam_idx, "[gguf] AdaptToGenAI: frontend model is missing the 'beam_idx' input.");

    const auto shape_1_1_1_m1 = const_i64({1, 1, 1, -1});

    // ---- inp_tokens = Reshape(Convert(input_ids, i32), [1,1,1,-1]) ----
    auto tokens_i32 = make_shared<ov::op::v0::Convert>(input_ids, ov::element::i32);
    auto tokens_4d = make_shared<ov::op::v1::Reshape>(tokens_i32, shape_1_1_1_m1, false);
    inp_tokens->output(0).replace(tokens_4d->output(0));

    // ---- inp_pos = Reshape(Convert(position_ids, i32), [1,1,1,-1]) ----
    auto pos_i32 = make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
    auto pos_4d = make_shared<ov::op::v1::Reshape>(pos_i32, shape_1_1_1_m1, false);
    inp_pos->output(0).replace(pos_4d->output(0));

    // ---- token_len_per_seq = ShapeOf(input_ids, i64)[1] -> [1] ----
    auto ids_shape = make_shared<ov::op::v3::ShapeOf>(input_ids, ov::element::i64);
    auto seq_len = make_shared<ov::op::v8::Gather>(ids_shape, const_i64({1}), const_i64({0}));  // [1]
    token_len_per_seq->output(0).replace(seq_len->output(0));

    // ---- self_kq_mask [1,1,seq,kv_len] f32: 0 where attended, -inf above causal ----
    // kv_len = attention_mask length (= past + seq). query absolute positions = position_ids[0].
    auto am_shape = make_shared<ov::op::v3::ShapeOf>(attention_mask, ov::element::i64);
    auto kv_len = make_shared<ov::op::v8::Gather>(am_shape, const_i64({1}), const_i64({0}));  // [1]

    auto q_pos = make_shared<ov::op::v0::Convert>(make_shared<ov::op::v0::Squeeze>(position_ids, const_i64({0})),
                                                  ov::element::i32);  // [seq]
    auto q_pos_col = make_shared<ov::op::v1::Reshape>(
        q_pos,
        make_shared<ov::op::v0::Concat>(ov::OutputVector{seq_len, const_i64({1})}, 0),
        false);  // [seq, 1]

    auto zero_i32 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto one_i32 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto kv_len_i32 = make_shared<ov::op::v0::Squeeze>(make_shared<ov::op::v0::Convert>(kv_len, ov::element::i32),
                                                       const_i64({0}));  // scalar
    auto k_range = make_shared<ov::op::v4::Range>(zero_i32, kv_len_i32, one_i32, ov::element::i32);  // [kv_len]
    auto k_row = make_shared<ov::op::v1::Reshape>(
        k_range,
        make_shared<ov::op::v0::Concat>(ov::OutputVector{const_i64({1}), kv_len}, 0),
        false);  // [1, kv_len]

    auto allowed = make_shared<ov::op::v1::LessEqual>(k_row, q_pos_col);  // [seq, kv_len] bool
    auto zero_f = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto neg_f = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {NEG_INF});
    auto mask2d = make_shared<ov::op::v1::Select>(allowed, zero_f, neg_f);  // [seq, kv_len] f32
    auto mask_4d = make_shared<ov::op::v1::Reshape>(
        mask2d,
        make_shared<ov::op::v0::Concat>(ov::OutputVector{const_i64({1, 1}), seq_len, kv_len}, 0),
        false);  // [1, 1, seq, kv_len]
    self_kq_mask->output(0).replace(mask_4d->output(0));

    // gpt-oss sliding-window mask: for prompts within the window it equals the full causal
    // mask, so the same value is correct here.
    if (auto self_kq_mask_swa = find_param(model, "self_kq_mask_swa")) {
        self_kq_mask_swa->output(0).replace(mask_4d->output(0));
    }

    // inp_out_ids selects which sequence rows to emit logits for; make it the whole sequence
    // (genai slices the last-token logits itself). Reshape(Range(0, seq, 1), [1,1,1,-1]).
    if (auto inp_out_ids = find_param(model, "inp_out_ids")) {
        auto seq_i32 = make_shared<ov::op::v0::Squeeze>(make_shared<ov::op::v0::Convert>(seq_len, ov::element::i32),
                                                        const_i64({0}));  // scalar
        auto out_range = make_shared<ov::op::v4::Range>(zero_i32, seq_i32, one_i32, ov::element::i32);
        auto out_ids = make_shared<ov::op::v1::Reshape>(out_range, shape_1_1_1_m1, false);
        inp_out_ids->output(0).replace(out_ids->output(0));
    }

    // ---- logits: [1,1,seq,vocab] -> [b, seq, vocab] (b = 1) ----
    auto old_result = model->get_results()[0];
    auto logits_src = old_result->input_value(0);
    auto vocab = make_shared<ov::op::v8::Gather>(make_shared<ov::op::v3::ShapeOf>(logits_src, ov::element::i64),
                                                 const_i64({3}),
                                                 const_i64({0}));  // [1]
    auto logits_3d = make_shared<ov::op::v1::Reshape>(
        logits_src,
        make_shared<ov::op::v0::Concat>(ov::OutputVector{const_i64({1, -1}), vocab}, 0),
        false);  // [1, seq, vocab]
    name_output(logits_3d, "logits");
    auto new_result = make_shared<ov::op::v0::Result>(logits_3d);
    new_result->set_friendly_name("logits");

    model->add_results({new_result});
    model->remove_result(old_result);

    // Swap the input list to the genai contract. beam_idx is kept as-is; every other old
    // gguf Parameter has had its output rewired (consumers now read the derived subgraph),
    // so removing it is safe.
    model->add_parameters({input_ids, attention_mask, position_ids});
    const auto params_snapshot = model->get_parameters();  // copy: remove_parameter mutates the list
    for (const auto& p : params_snapshot) {
        if (p == input_ids || p == attention_mask || p == position_ids || p == beam_idx) {
            continue;
        }
        model->remove_parameter(p);
    }

    // Pin the runtime KV-cache precision to f16 for large-head models so decode matches both
    // prefill and llama.cpp; mainstream small-head models keep the faster u8 default. This is a
    // consumer-side optimization policy (genai), not a property the frontend bakes into the model.
    constexpr int64_t kU8SafeHeadSize = 128;
    if (max_kv_cache_head_size(model) > kU8SafeHeadSize) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }

    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
