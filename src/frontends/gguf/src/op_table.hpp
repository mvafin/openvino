#pragma once

#include "node_context.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace op {

#define GGUF_OP_CONVERTER(op) OutputVector op(const NodeContext& context)

GGUF_OP_CONVERTER(translate_add);
GGUF_OP_CONVERTER(translate_cont);
GGUF_OP_CONVERTER(translate_get_rows);
GGUF_OP_CONVERTER(translate_mul);
GGUF_OP_CONVERTER(translate_mulmat);
GGUF_OP_CONVERTER(translate_permute);
GGUF_OP_CONVERTER(translate_reshape);
GGUF_OP_CONVERTER(translate_rms_norm);
GGUF_OP_CONVERTER(translate_rope);
GGUF_OP_CONVERTER(translate_scale);
GGUF_OP_CONVERTER(translate_unary_silu);
GGUF_OP_CONVERTER(translate_unary_gelu);
GGUF_OP_CONVERTER(translate_soft_max);
GGUF_OP_CONVERTER(translate_transpose);
GGUF_OP_CONVERTER(translate_view);
GGUF_OP_CONVERTER(translate_glu_swiglu);
GGUF_OP_CONVERTER(translate_glu_geglu);
GGUF_OP_CONVERTER(translate_set_rows);
GGUF_OP_CONVERTER(translate_cpy);
GGUF_OP_CONVERTER(translate_flash_attn_ext);
GGUF_OP_CONVERTER(translate_glu_swiglu_oai);
// MoE (mixture-of-experts) routing ops.
GGUF_OP_CONVERTER(translate_mul_mat_id);
GGUF_OP_CONVERTER(translate_add_id);
GGUF_OP_CONVERTER(translate_argsort);
GGUF_OP_CONVERTER(translate_top_k);
GGUF_OP_CONVERTER(translate_sum_rows);
// Structural ops.
GGUF_OP_CONVERTER(translate_concat);
// Elementwise clamp.
GGUF_OP_CONVERTER(translate_clamp);
// Unary activations.
GGUF_OP_CONVERTER(translate_unary_relu);
GGUF_OP_CONVERTER(translate_unary_tanh);
GGUF_OP_CONVERTER(translate_unary_sigmoid);
GGUF_OP_CONVERTER(translate_unary_elu);
GGUF_OP_CONVERTER(translate_unary_gelu_quick);
// Unary element-wise math.
GGUF_OP_CONVERTER(translate_sqr);
GGUF_OP_CONVERTER(translate_sqrt);
GGUF_OP_CONVERTER(translate_log);
GGUF_OP_CONVERTER(translate_sin);
GGUF_OP_CONVERTER(translate_cos);
// LayerNorm (mean + variance normalization).
GGUF_OP_CONVERTER(translate_norm);
// Tile / broadcast repeat.
GGUF_OP_CONVERTER(translate_repeat);
// A GGML_OP_NONE leaf carrying a "data" attribute -> dequantized weight node (cgraph path).
GGUF_OP_CONVERTER(translate_weight);

}  // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops();

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
