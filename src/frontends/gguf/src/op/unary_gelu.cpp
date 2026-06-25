#include <openvino/core/node_output.hpp>
#include <openvino/op/gelu.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_unary_gelu(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    // GGML_UNARY_OP_GELU is the tanh approximation (ggml_gelu); the erf form is a separate
    // ggml op (GGML_UNARY_OP_GELU_ERF). OV's Gelu defaults to ERF, so request TANH explicitly
    // to match ggml -- the erf/tanh mismatch compounds across layers into wrong outputs.
    auto res = std::make_shared<ov::op::v7::Gelu>(input, ov::op::GeluApproximationMode::TANH);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
