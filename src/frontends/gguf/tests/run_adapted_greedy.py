#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Drive the genai-IO-adapted GGUF model (genai_io_adapter.adapt_to_genai) with the exact
# inputs OpenVINO GenAI's stateful LLMPipeline feeds (input_ids / attention_mask /
# position_ids / beam_idx), greedy-decoding a prompt. Output is token-comparable to
# llama-simple. This validates the M3 adapter independent of a genai rebuild.

import argparse
import sys

import numpy as np
import openvino as ov

from genai_io_adapter import adapt_to_genai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok(args.prompt, return_tensors="np")["input_ids"][0].astype(np.int64).tolist()

    core = ov.Core()
    adapted = adapt_to_genai(core.read_model(args.gguf))
    req = core.compile_model(adapted, args.device).create_infer_request()

    all_ids = list(ids)         # full sequence so far (genai feeds growing attention_mask)
    generated = []

    def step(new_ids):
        # genai contract: input_ids [1, n_new], attention_mask [1, total], position_ids
        # [1, n_new], beam_idx [1].
        n_new = len(new_ids)
        total = len(all_ids)
        input_ids = np.array(new_ids, dtype=np.int64).reshape(1, n_new)
        attention_mask = np.ones((1, total), dtype=np.int64)
        start = total - n_new
        position_ids = np.arange(start, total, dtype=np.int64).reshape(1, n_new)
        beam_idx = np.zeros((1,), dtype=np.int32)
        out = req.infer({
            "input_ids": ov.Tensor(input_ids),
            "attention_mask": ov.Tensor(attention_mask),
            "position_ids": ov.Tensor(position_ids),
            "beam_idx": ov.Tensor(beam_idx),
        })
        logits = list(out.values())[0]  # [1, seq, vocab]
        return int(logits[0, -1].argmax())

    # prefill
    nxt = step(ids)
    generated.append(nxt)
    all_ids.append(nxt)
    # decode
    for _ in range(args.n - 1):
        nxt = step([nxt])
        generated.append(nxt)
        all_ids.append(nxt)

    print(f"prompt: {args.prompt!r}")
    print(f"generated token ids: {generated}")
    print(f"full: {args.prompt}{tok.decode(generated)}")


if __name__ == "__main__":
    sys.exit(main())
