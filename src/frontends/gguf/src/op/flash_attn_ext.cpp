#include <cstdint>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/tanh.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <string>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_flash_attn_ext(const NodeContext& context) {
    num_inputs_check(context, 4, 5);
    auto q_f32 = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    auto mask = context.get_input(3);
    // gpt-oss: optional 5th input is the per-head attention sink logit [n_head].
    const bool has_sinks = context.get_input_size() == 5;

    float scale = context.get_attribute<float>("scale");
    float kq_soft_cap = context.get_attribute<float>("kq_soft_cap", 0.0f);

    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<float>{scale});

    ov::Output<ov::Node> mask_sliced, res;
    std::string mask_name = "KQ_mask_sliced";
    if (context.get_input_names()[3].find("swa") != std::string::npos) {
        mask_name = "KQ_mask_swa_sliced";
    }
    if (context.has_input(mask_name)) {
        mask_sliced = context.get_input(mask_name);
    } else {
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto token_len = get_dimensions(q, {2});
        mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero, token_len, one, two);
    }

    if (mask_sliced.get_element_type() != ov::element::f16) {
        mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
    }

    // q/k/v arrive in the ggml-natural [B, n_tokens, n_head(_kv), head_size] layout (the
    // builder no longer PERMUTEs them). For grouped-query attention we first broadcast K/V up
    // to n_head along the head axis (axis 2 here), THEN transpose all three to the canonical
    // [B, n_head, n_tokens, head_size] SDPA layout. Doing the tile before the single transpose
    // is what lets the CPU plugin's stateful_sdpa_fusion match (its multi-query-broadcast
    // pattern sits on the KV-cache concat output, ahead of exactly one transpose).
    auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
        int64_t factor = num_heads / num_heads_kv;
        if (factor > 1 && num_heads_kv > 1) {
            // kv: [B, L, n_head_kv, S] -> unsqueeze head axis -> [B, L, n_head_kv, 1, S]
            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {3});
            auto kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);
            auto kv_broadcast_shape =
                ov::op::v0::Constant::create(ov::element::i64,
                                             {5},
                                             {(int64_t)1, (int64_t)1, (int64_t)1, factor, (int64_t)1});
            auto new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t)0, (int64_t)0, num_heads, head_size});
            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed,
                                                         kv_broadcast_shape,
                                                         ov::op::BroadcastType::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, true);  // [B, L, n_head, S]
        }
        return kv;
    };

    auto q_shape = context.get_input_shape(0).to_shape();
    auto k_shape = context.get_input_shape(1).to_shape();
    k = tile_kv(q_shape[2], k_shape[2], q_shape[3], k);
    v = tile_kv(q_shape[2], k_shape[2], q_shape[3], v);

    // [B, L, H, S] -> [B, H, L, S] (canonical SDPA layout).
    auto to_bhls = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    ov::Output<ov::Node> q_t = std::make_shared<ov::op::v1::Transpose>(q, to_bhls);
    ov::Output<ov::Node> k_t = std::make_shared<ov::op::v1::Transpose>(k, to_bhls);
    ov::Output<ov::Node> v_t = std::make_shared<ov::op::v1::Transpose>(v, to_bhls);

    ov::Output<ov::Node> sdpa;
    if (kq_soft_cap != 0.0f) {
        // Gemma2 attention soft-cap: tanh(QK^T * scale * (1/cap)) * cap + mask -> softmax -> *V.
        // OV SDPA v13 has no native softcap parameter, so we decompose the attention manually.
        // Operates in f32 (q already converted to f16 for normal path; here stay f32).
        // q_t / k_t / v_t are already [B, H, L, S] from the transpose above but in f16;
        // convert to f32 for the manual decomposition.
        using namespace ov::op;
        auto q_f32_t = std::make_shared<v0::Convert>(q_t, element::f32);
        auto k_f32_t = std::make_shared<v0::Convert>(k_t, element::f32);
        auto v_f32_t = std::make_shared<v0::Convert>(v_t, element::f32);
        auto mask_f32 = mask_sliced.get_element_type() != element::f32
                            ? std::make_shared<v0::Convert>(mask_sliced, element::f32)->output(0)
                            : mask_sliced;

        // QK^T: [B, H, L, S] x [B, H, S, Lk] -> [B, H, L, Lk]
        auto kT =
            std::make_shared<v1::Transpose>(k_f32_t,
                                            v0::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 1, 3, 2}));
        auto qk = std::make_shared<v0::MatMul>(q_f32_t, kT, false, false);

        // Apply scale * (1/softcap), then tanh, then *softcap
        auto pre_cap_scale = v0::Constant::create(element::f32, Shape{}, std::vector<float>{scale / kq_soft_cap});
        auto qk_scaled = std::make_shared<v1::Multiply>(qk, pre_cap_scale);
        auto qk_tanh = std::make_shared<v0::Tanh>(qk_scaled);
        auto post_cap_scale = v0::Constant::create(element::f32, Shape{}, std::vector<float>{kq_soft_cap});
        auto qk_capped = std::make_shared<v1::Multiply>(qk_tanh, post_cap_scale);

        // Add mask (already sliced to [B, 1, L, Lk] or [B, 1, 1, Lk])
        auto qk_masked = std::make_shared<v1::Add>(qk_capped, mask_f32);

        // Softmax over last axis (key dimension)
        auto attn_weights = std::make_shared<v8::Softmax>(qk_masked, -1);

        // Weighted sum over values: [B, H, L, Lk] x [B, H, Lk, S] -> [B, H, L, S]
        auto attn_out_caps = std::make_shared<v0::MatMul>(attn_weights, v_f32_t, false, false);

        sdpa = attn_out_caps;
    } else if (!has_sinks) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_t, k_t, v_t, mask_sliced, scale_node, false);
    } else {
        // gpt-oss attention sinks: a learned per-head logit participates in the softmax
        // denominator (so the attention weights do not sum to 1) but contributes no value.
        // OpenVINO SDPA has a native 6-input form (q, k, v, mask, scale, sink) that the CPU
        // plugin folds the sink straight into its online-softmax, so we no longer decompose
        // attention by hand. The sink logit is per head: [n_head] -> [1, n_head, 1, 1] to
        // broadcast over [B, n_head, q, 1] (rank must equal the query rank, last dim 1).
        using namespace ov::op;
        auto sink = context.get_input(4);
        auto sink_f16 = sink.get_element_type() != element::f16
                            ? std::make_shared<v0::Convert>(sink, element::f16)->output(0)
                            : sink;
        auto sink_shape = v0::Constant::create(element::i64, {4}, std::vector<int64_t>{1, (int64_t)q_shape[2], 1, 1});
        auto sink_r = std::make_shared<v1::Reshape>(sink_f16, sink_shape, false);
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_t,
                                                                        k_t,
                                                                        v_t,
                                                                        mask_sliced,
                                                                        scale_node,
                                                                        sink_r,
                                                                        false);
    }
    // [B, H, L, S] -> [B, L, H, S] (ggml-natural layout expected by caller).
    res = std::make_shared<ov::op::v1::Transpose>(sdpa,
                                                  ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    // SDPA paths produce f16; the soft-cap path produces f32 directly.
    if (kq_soft_cap == 0.0f) {
        res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
