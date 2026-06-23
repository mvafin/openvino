#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>

#include "../op_table.hpp"
#include "../utils.hpp"
namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_view(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    if (context.get_op_case() == 2) {
        auto dst_shape = context.get_output_shape().to_shape();
        return rename_outputs_with_suffix({process_view_input(context, 0, dst_shape[2] * dst_shape[3])},
                                          context.get_name());
    }
    // op_case 3
    if (context.get_op_case() == 3) {
        auto input = context.get_input(0);
        auto input_ov_shape = input.get_partial_shape();

        auto input_llama_shape = context.get_input_shape(0).to_shape();

        // if the input ov shape size is different from the input llama shape size, it means the input is already
        // reshaped and we need to reshape it back to the original shape before slicing
        if (input_ov_shape.size() != input_llama_shape.size()) {
            input = std::make_shared<ov::op::v1::Reshape>(
                input,
                ov::op::v0::Constant::create(ov::element::i64, {input_llama_shape.size()}, input_llama_shape),
                false);
        }

        auto dst_shape = context.get_output_shape().to_shape();

        // find the index of dst_shape that is different from input shape, and use that index to slice the input
        int slice_dim = -1;
        for (size_t i = 0; i < dst_shape.size(); ++i) {
            if (dst_shape[i] != input_llama_shape[i]) {
                slice_dim = i;
                break;
            }
        }

        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {dst_shape[slice_dim]});
        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim});
        auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
        return {sliced};
    }
    // op_case 4: layer-index slice for per-layer embedding.
    // Non-stateful: input [1, n_layer, T, D] -> slice dim 1 -> [1, 1, T, D].
    // Stateful: input [n_layer, T, D] -> slice dim 0 -> [1, T, D].
    if (context.get_op_case() == 4) {
        const int64_t layer_idx = context.get_attribute<int64_t>("layer_idx");
        auto input = context.get_input(0);
        auto start = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_idx});
        auto stop = ov::op::v0::Constant::create(ov::element::i64, {1}, {layer_idx + 1});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        // In stateful mode the tensor is 3D [n_layer, T, D] — slice dim 0;
        // in non-stateful it is 4D [1, n_layer, T, D] — slice dim 1.
        const int64_t slice_axis = context.is_stateful() ? 0 : 1;
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_axis});
        auto sliced = std::make_shared<ov::op::v8::Slice>(input, start, stop, step, axes);
        return rename_outputs_with_suffix({sliced}, context.get_name());
    }
    // op_case 5: head-size slice. Slices the last dimension to "head_size" (stored in attributes).
    // Used for gemma4 shared SWA layers that reuse the global anchor's K/V but need only the
    // first head_size elements: [1, T, n_kv, anchor_hs] -> [1, T, n_kv, head_size].
    if (context.get_op_case() == 5) {
        const int64_t head_size = context.get_attribute<int64_t>("head_size");
        auto input = context.get_input(0);
        auto start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_size});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});  // last axis
        auto sliced = std::make_shared<ov::op::v8::Slice>(input, start, stop, step, axes);
        return rename_outputs_with_suffix({sliced}, context.get_name());
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
