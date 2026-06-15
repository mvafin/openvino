#include <cassert>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <vector>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_set_rows(const NodeContext& context) {
    num_inputs_check(context, 3, 3);

    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    auto dst = context.get_input(2);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    auto dst_shape = context.get_output_shape().to_shape();

    auto ind_squeezed =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
    auto data_reshaped = std::make_shared<ov::op::v1::Reshape>(
        data,
        ov::op::v0::Constant::create(ov::element::i64,
                                     {4},
                                     {(int64_t)1, (int64_t)1, (int64_t)-1, (int64_t)dst_shape[3]}),
        false);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    if (context.is_stateful()) {
        int concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_length();
        int64_t dim3 = dst.get_partial_shape()[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data,
            ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t)1, (int64_t)-1, dim2, dim3}),
            false);
        // Reorder the past cache along the batch axis by beam_idx before concatenating the
        // new tokens. With batch=1 / beam_idx=[0] this is an identity gather, but emitting it
        // produces the exact ReadValue->Gather(beam_idx)->Concat shape that the CPU plugin's
        // stateful_sdpa_fusion matches, so the downstream SDPA fuses into
        // ScaledDotProductAttentionWithKVCache (in-place KV update) instead of running as an
        // unfused MatMul/Softmax/MatMul. Mirrors genai's create_cache.
        Output<Node> past = dst;
        if (context.has_input("beam_idx")) {
            auto beam_idx = context.get_input("beam_idx");
            auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            past = std::make_shared<ov::op::v8::Gather>(dst, beam_idx, axis0);
        }
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{past, data}, concat_axis);
    } else {
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_reshaped, axes);
    }

    if (auto dst_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_input_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(),
                                          dst_shape_partial[1].get_length(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                          dst_shape_partial[3].get_length()};
        res = std::make_shared<ov::op::v1::Reshape>(res,
                                                    ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    false);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
