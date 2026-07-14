#include "openvino/decompositions/rope.hpp"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"
#include "openvino/pass/node_registry.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_rope(const NodeContext& context) {
    num_inputs_check(context, 2, 3);

    int op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_node = context.get_input(0).get_node_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    auto rope_config = context.get_attribute<RopeConfig>("rope_config");
    const int mode = (op_case & 0xFFFF0000) >> 16;
    op_case = (op_case & 0x0000FFFF);

    constexpr int TYPE_NORMAL = 0;
    constexpr int TYPE_NEOX = 1;
    constexpr int TYPE_IMROPE = 2;

    Output<Node> cos_theta_node;
    Output<Node> sin_theta_node;
    if (context.has_input("rope_cos")) {
        cos_theta_node = context.get_input("rope_cos");
        sin_theta_node = context.get_input("rope_sin");
    } else {
        auto inp_pos = context.get_input(1).get_node_shared_ptr();
        std::shared_ptr<ov::Node> rope_freqs_weight;
        if (context.get_input_size() == 3) {
            rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
        }
        auto sin_cos = make_sin_cos(rope_config, inp_pos, rope_freqs_weight, mode == TYPE_IMROPE);
        sin_theta_node = sin_cos.first;
        cos_theta_node = sin_cos.second;
    }

    if (op_case == 2) {
        // The input comes from a VIEW
        int slice_len = output_shape[2] * output_shape[3];
        data_node = process_view_input(context, 0, slice_len).get_node_shared_ptr();
        if (context.is_stateful()) {
            auto data_shape = ov::op::v0::Constant::create(
                ov::element::i64,
                {3},
                std::vector<int64_t>{-1, (int64_t)output_shape[2], (int64_t)output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
        } else {
            auto data_shape = ov::op::v0::Constant::create(
                ov::element::i64,
                {4},
                std::vector<int64_t>{1, -1, (int64_t)output_shape[2], (int64_t)output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
        }
    }

    if (mode == TYPE_NORMAL) {
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        Output<Node> even_slice;
        Output<Node> odd_slice;
        int32_t unsqueeze_dim = context.is_stateful() ? 3 : 4;
        even_slice = std::make_shared<ov::op::v8::Slice>(data_node, zero, end, two, neg_one);
        odd_slice = std::make_shared<ov::op::v8::Slice>(data_node, one, end, two, neg_one);

        Output<Node> first_half =
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta_node),
                                                   std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta_node));
        Output<Node> second_half =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta_node),
                                              std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta_node));

        first_half = std::make_shared<ov::op::v0::Unsqueeze>(
            first_half,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {unsqueeze_dim}));
        second_half = std::make_shared<ov::op::v0::Unsqueeze>(
            second_half,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {unsqueeze_dim}));
        auto stack = std::make_shared<ov::op::v0::Concat>(OutputVector{first_half, second_half}, unsqueeze_dim);

        auto data_shape = ov::op::v0::Constant::create(
            ov::element::i64,
            {4},
            std::vector<int64_t>{1, -1, (int64_t)output_shape[2], (int64_t)output_shape[3]});
        res = std::make_shared<ov::op::v1::Reshape>(stack, data_shape, false);
    } else if (mode == TYPE_NEOX) {
        // Build the canonical NEOX RoPE via the shared decomposition helper, which emits the
        // exact split-halves + Multiply(-1)+Add + Concat pattern that ov::pass::RoPEFusion
        // (specifically the RoPEFusionGPTOSS matcher) folds into the fused
        // ov::op::internal::RoPE primitive on CPU/GPU.
        //
        // That matcher only fires when the rotated tensor is laid out as [B, H, L, S] and the
        // cos/sin caches are [?, 1, ?, head/2]. Our tensors are ggml-natural: data is
        // [B, L, H, S] (or [L, H, S] when stateful) and cos/sin are [B, L, 1, head/2] (or
        // [L, 1, head/2]). So we transpose every operand into the canonical [B, H, L, S]
        // layout (heads on axis 1), run the decomposition there, and transpose the result
        // back to the gguf layout. The math is unchanged; the wrapping Transposes are sunk /
        // cancelled against the adjacent PERMUTE during TransposeSinking.
        //
        // The ggml-natural shapes here are: data [B,L,H,S] (or [B,L,H*S] when the
        // Kcur_r reshape was eliminated for n_head_kv=1) and cos/sin [B,L,1,head/2].
        // For rank-4 data [B,L,H,S]: Transpose {0,2,1,3} → [B,H,L,S].
        // For rank-3 data [B,L,H*S]: first reshape to [B,L,H,S] using output_shape,
        // then Transpose {0,2,1,3} → [B,H,L,S]. The cos/sin are always rank-4 [B,L,1,S/2].
        const int64_t n_head_rope = static_cast<int64_t>(output_shape[2]);
        const int64_t head_size_rope = static_cast<int64_t>(output_shape[3]);
        const auto perm_bhls = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});

        // The DATA reaches this op in inconsistent shapes depending on the layer's upstream rank:
        // rank-3 [B, L, H*S] (e.g. n_head_kv=1 layers fed by a rank-3 producer), or rank-4 that
        // may be [B, L, H, S] OR [B, 1, L, S] (when an upstream rank-4 l_out makes OV keep a
        // leading 1 and the head axis lands at position 1). A single fixed Transpose cannot
        // normalize all of these. Instead, always Reshape the data to the canonical ggml-natural
        // [B, L, H, S] using the op's output_shape (element order is preserved, so this correctly
        // reinterprets every incoming layout), then Transpose {0,2,1,3} -> [B, H, L, S].
        auto data_to_bhls = [&](ov::Output<ov::Node> x) -> ov::Output<ov::Node> {
            auto shape4d = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, n_head_rope, head_size_rope});
            x = std::make_shared<ov::op::v1::Reshape>(x, shape4d, false);  // [B, L, H, S]
            return std::make_shared<ov::op::v1::Transpose>(x, perm_bhls);  // [B, H, L, S]
        };
        // cos/sin always arrive rank-4 [B, L, 1, head/2]; just transpose to [B, 1, L, head/2].
        auto cossin_to_bhls = [&](ov::Output<ov::Node> x) -> ov::Output<ov::Node> {
            return std::make_shared<ov::op::v1::Transpose>(x, perm_bhls);
        };

        auto x_bhls = data_to_bhls(data_node);       // [B, H, L, S]
        auto cos_bhls = cossin_to_bhls(cos_theta_node);  // [B, 1, L, head/2]
        auto sin_bhls = cossin_to_bhls(sin_theta_node);  // [B, 1, L, head/2]

        ov::pass::NodeRegistry reg;
        const int64_t half = static_cast<int64_t>(output_shape[3]) / 2;
        auto roped = ov::decomposition::rope(reg, x_bhls, cos_bhls, sin_bhls, half);  // [B, H, L, S]

        // Back to the gguf layout the rest of the graph expects. The original NEOX branch
        // always produced a rank-4 [B, L, H, S] tensor (the rank-3 stateful data was lifted
        // to rank-4 by the NUMPY broadcast against the rank-4 cos/sin), and the downstream
        // PERMUTE consumes rank-4, so we emit rank-4 here too.
        res = std::make_shared<ov::op::v1::Transpose>(
            roped,
            ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));  // [B, L, H, S]
    } else if (mode == TYPE_IMROPE) {
        int64_t n_dims = data_node->get_shape()[3];
        auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                    ov::Shape{4},
                                                                    std::vector<int64_t>{1, -1, 1, (n_dims >> 1)});
        auto cos_reshaped = std::make_shared<ov::op::v1::Reshape>(cos_theta_node, cos_sin_shape, true);
        auto sin_reshaped = std::make_shared<ov::op::v1::Reshape>(sin_theta_node, cos_sin_shape, true);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_a = std::make_shared<ov::op::v1::Split>(data_node, split_axis, 2);
        auto x0 = split_a->output(0);
        auto x1 = split_a->output(1);
        auto mul_a = std::make_shared<ov::op::v1::Multiply>(x0, cos_reshaped);
        auto mul_b = std::make_shared<ov::op::v1::Multiply>(x1, sin_reshaped);
        auto sub = std::make_shared<ov::op::v1::Subtract>(mul_a, mul_b);

        auto mul_c = std::make_shared<ov::op::v1::Multiply>(x0, sin_reshaped);
        auto mul_d = std::make_shared<ov::op::v1::Multiply>(x1, cos_reshaped);
        auto add = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub, add}, 3);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
