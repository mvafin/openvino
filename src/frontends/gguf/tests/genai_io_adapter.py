#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# M3 IO adapter: wrap the GGUF-frontend model (gguf IO: inp_tokens / inp_pos /
# self_kq_mask / token_len_per_seq) so it exposes the OpenVINO GenAI LLMPipeline
# contract instead:
#
#   inputs : input_ids [b, seq] i64, attention_mask [b, kv_len] i64,
#            position_ids [b, seq] i64, beam_idx [b] i32
#   output : logits [b, seq, vocab]
#
# The adapter prepends a small subgraph that derives the gguf inputs from the genai
# inputs (a graph-level version of compare_with_llama.py::build_inputs), and reshapes
# the [1,1,seq,vocab] logits to [b, seq, vocab]. beam_idx is accepted (greedy passes
# all-zeros) but not used to gather the batch-1 stateful KV cache.
#
# This is a Python prototype of the C++ frontend/genai pass; the op sequence here maps
# 1:1 to ov::op builders.

import openvino as ov
from openvino import opset13 as op
from openvino import Type, PartialShape


def _const_i64(values):
    import numpy as np
    a = np.array(values, dtype=np.int64)
    return op.constant(a, Type.i64)


def adapt_to_genai(model):
    """Return a new ov.Model with the genai IO contract wrapping `model`."""
    params = {p.get_friendly_name(): p for p in model.get_parameters()}
    for required in ("inp_tokens", "inp_pos", "self_kq_mask", "token_len_per_seq"):
        assert required in params, f"frontend model missing gguf input {required!r}"

    # ---- new genai inputs ----
    input_ids = op.parameter(PartialShape([-1, -1]), Type.i64)
    input_ids.set_friendly_name("input_ids")
    input_ids.output(0).set_names({"input_ids"})

    attention_mask = op.parameter(PartialShape([-1, -1]), Type.i64)
    attention_mask.set_friendly_name("attention_mask")
    attention_mask.output(0).set_names({"attention_mask"})

    position_ids = op.parameter(PartialShape([-1, -1]), Type.i64)
    position_ids.set_friendly_name("position_ids")
    position_ids.output(0).set_names({"position_ids"})

    beam_idx = op.parameter(PartialShape([-1]), Type.i32)
    beam_idx.set_friendly_name("beam_idx")
    beam_idx.output(0).set_names({"beam_idx"})

    # ---- inp_tokens = reshape(convert(input_ids, i32), [1,1,1,-1]) ----
    inp_tokens = op.reshape(op.convert(input_ids, Type.i32), _const_i64([1, 1, 1, -1]), False)
    inp_pos = op.reshape(op.convert(position_ids, Type.i32), _const_i64([1, 1, 1, -1]), False)

    # ---- token_len_per_seq = seq = shape_of(input_ids)[1] -> [1] i64 ----
    ids_shape = op.shape_of(input_ids, Type.i64)
    seq_len = op.gather(ids_shape, _const_i64([1]), _const_i64(0))  # [1]

    # ---- self_kq_mask [1,1,seq,kv_len] f32: 0 where attended, -inf above causal ----
    # kv_len = attention_mask length (= past + seq). seq = ids length.
    am_shape = op.shape_of(attention_mask, Type.i64)
    kv_len = op.gather(am_shape, _const_i64([1]), _const_i64(0))  # [1]
    # query absolute positions = position_ids[0]  (shape [seq]); key positions = 0..kv_len-1
    q_pos = op.convert(op.squeeze(position_ids, _const_i64([0])), Type.i32)  # [seq]
    q_pos_col = op.reshape(q_pos, op.concat([seq_len, _const_i64([1])], 0), False)  # [seq,1]
    k_range = op.range(op.constant(0, Type.i32),
                       op.squeeze(op.convert(kv_len, Type.i32), _const_i64([0])),
                       op.constant(1, Type.i32), Type.i32)  # [kv_len]
    k_row = op.reshape(k_range, op.concat([_const_i64([1]), kv_len], 0), False)  # [1,kv_len]
    # allowed where key_pos <= query_pos
    allowed = op.less_equal(k_row, q_pos_col)  # [seq,kv_len] bool
    zero = op.constant(0.0, Type.f32)
    neg = op.constant(-65504.0, Type.f32)  # f16 lowest, matches genai
    mask2d = op.select(allowed, zero, neg)  # [seq,kv_len] f32
    self_kq_mask = op.reshape(mask2d, op.concat([_const_i64([1, 1]), seq_len, kv_len], 0), False)

    # ---- rewire: replace the gguf Parameters' uses with the computed tensors ----
    replacements = {
        "inp_tokens": inp_tokens,
        "inp_pos": inp_pos,
        "self_kq_mask": self_kq_mask,
        "token_len_per_seq": seq_len,
    }
    for name, new_out in replacements.items():
        old = params[name]
        old.output(0).replace(new_out.output(0))

    # inp_out_ids (last-token gather) is optional; if present, make it select the whole seq
    # (genai slices logits itself). Reshape range(0,seq) to [1,1,1,seq].
    if "inp_out_ids" in params:
        out_ids = op.reshape(
            op.range(op.constant(0, Type.i32), op.squeeze(op.convert(seq_len, Type.i32), _const_i64([0])),
                     op.constant(1, Type.i32), Type.i32),
            _const_i64([1, 1, 1, -1]), False)
        params["inp_out_ids"].output(0).replace(out_ids.output(0))

    # self_kq_mask_swa: SWA sliding-window mask (gemma4 / gpt-oss). Use the same causal mask
    # as self_kq_mask — correct for prompts; GenAI doesn't distinguish SWA vs global masks.
    if "self_kq_mask_swa" in params:
        params["self_kq_mask_swa"].output(0).replace(self_kq_mask.output(0))

    # beam_idx: wire up the new beam_idx Parameter to replace the original one
    # so the stateful KV-cache gather has a valid input.
    if "beam_idx" in params:
        params["beam_idx"].output(0).replace(beam_idx.output(0))

    # ---- logits: [1,1,seq,vocab] -> [b, seq, vocab] (b=1) ----
    result = model.get_results()[0]
    logits_src = result.input_value(0)
    logits_3d = op.reshape(logits_src, op.concat([_const_i64([1, -1]),
                                                  op.gather(op.shape_of(logits_src, Type.i64),
                                                            _const_i64([3]), _const_i64(0))], 0), False)
    logits_3d.set_friendly_name("logits")
    logits_3d.output(0).set_names({"logits"})
    new_result = op.result(logits_3d)
    new_result.set_friendly_name("logits")

    # keep beam_idx alive: it must remain a graph input for genai's set_tensor("beam_idx").
    # The frontend's KV is batch-1 stateful, so for greedy beam_idx is unused; we keep the
    # Parameter in the model's parameter list so the port exists.
    new_params = [input_ids, attention_mask, position_ids, beam_idx]

    adapted = ov.Model([new_result], model.get_sinks(), new_params, "qwen3_genai")

    # Pin KV-cache precision to f16 for large-head models (e.g. gemma4 global-attention
    # head_size=512): the CPU plugin's default u8 cache quantization compounds across decode
    # into divergence/NaN there, while small-head models (llama/qwen/phi3/gpt-oss, 64-128)
    # keep the faster u8 default. Mirrors the C++ AdaptToGenAI pass.
    max_head = 0
    for o in model.get_ops():
        if o.get_type_name() == "ReadValue":
            ps = o.get_output_partial_shape(0)
            last = ps[len(ps) - 1]
            if last.is_static:
                max_head = max(max_head, last.get_length())
    if max_head > 128:
        adapted.set_rt_info("f16", ["runtime_options", "KV_CACHE_PRECISION"])
    return adapted


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", default="/tmp/qwen3_genai/openvino_model.xml")
    args = ap.parse_args()
    core = ov.Core()
    m = core.read_model(args.gguf)
    adapted = adapt_to_genai(m)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ov.save_model(adapted, args.out, compress_to_fp16=False)
    print("saved adapted model to", args.out)
    print("inputs:", [p.get_friendly_name() for p in adapted.get_parameters()])
    print("outputs:", [r.get_friendly_name() for r in adapted.get_results()])
