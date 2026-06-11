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
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <string>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
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
    // float max_bias      = context.get_attribute<float>("max_bias");
    // float logit_softcap = context.get_attribute<float>("logit_softcap");

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

    auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
        int64_t factor = num_heads / num_heads_kv;
        if (factor > 1 && num_heads_kv > 1) {
            ov::Output<ov::Node> kv_broadcast_shape, kv_unsqueezed, new_kv_shape;
            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_broadcast_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                              {5},
                                                              {(int64_t)1, (int64_t)1, factor, (int64_t)1, (int64_t)1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t)0, num_heads, (int64_t)-1, head_size});

            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed,
                                                         kv_broadcast_shape,
                                                         ov::op::BroadcastType::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, true);
        }
        return kv;
    };

    auto q_shape = context.get_input_shape(0).to_shape();
    auto k_shape = context.get_input_shape(1).to_shape();
    k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);

    ov::Output<ov::Node> sdpa;
    if (!has_sinks) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_sliced, scale_node, false);
    } else {
        // Attention with sinks (gpt-oss): a learned per-head logit participates in the
        // softmax denominator (so the attention weights do not sum to 1), but contributes
        // no value. Decompose SDPA explicitly:
        //   scores = (Q·Kᵀ)·scale + mask                          [.., q, kv]
        //   m      = max(rowmax(scores), sink)                    (numerical stability)
        //   denom  = Σ exp(scores - m) + exp(sink - m)
        //   out    = (exp(scores - m) / denom) · V
        using namespace ov::op;
        auto kt = std::make_shared<v1::Transpose>(k, v0::Constant::create(element::i64, {4}, {0, 1, 3, 2}));
        ov::Output<ov::Node> scores = std::make_shared<v0::MatMul>(q, kt, false, false);  // [.., q, kv]
        scores = std::make_shared<v1::Multiply>(scores, scale_node);
        scores = std::make_shared<v1::Add>(scores, mask_sliced);

        // sink logit per head: [n_head] -> [1, n_head, 1, 1] to broadcast over [.., q, kv].
        auto sink = context.get_input(4);
        auto sink_f16 = sink.get_element_type() != element::f16 ? std::make_shared<v0::Convert>(sink, element::f16)->output(0) : sink;
        auto sink_shape =
            v0::Constant::create(element::i64, {4}, std::vector<int64_t>{1, (int64_t)q_shape[1], 1, 1});
        auto sink_r = std::make_shared<v1::Reshape>(sink_f16, sink_shape, false);

        auto kv_axis = v0::Constant::create(element::i64, {1}, {-1});
        auto row_max = std::make_shared<v1::ReduceMax>(scores, kv_axis, true);   // [.., q, 1]
        auto m = std::make_shared<v1::Maximum>(row_max, sink_r);                 // [.., q, 1]
        auto exp_scores = std::make_shared<v0::Exp>(std::make_shared<v1::Subtract>(scores, m));
        auto sum_exp = std::make_shared<v1::ReduceSum>(exp_scores, kv_axis, true);  // [.., q, 1]
        auto exp_sink = std::make_shared<v0::Exp>(std::make_shared<v1::Subtract>(sink_r, m));
        auto denom = std::make_shared<v1::Add>(sum_exp, exp_sink);
        auto weights = std::make_shared<v1::Divide>(exp_scores, denom);
        sdpa = std::make_shared<v0::MatMul>(weights, v, false, false);  // [.., q, head_size]
    }
    res = std::make_shared<ov::op::v1::Transpose>(sdpa,
                                                  ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
