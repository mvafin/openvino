#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Tokenizer helper: reads tokenizer from a GGUF file via openvino_genai.Tokenizer.
# Falls back to a user-supplied HF tokenizer dir when genai can't parse the GGUF.
#
# Prints JSON {ids: [...], eos_id: int} or {text: str, eos_id: int} to stdout.
#
# Usage:
#   python3 gguf_tokenize.py encode <gguf> <text> [--hf-tokenizer <dir>]
#   python3 gguf_tokenize.py decode <gguf> <id1> <id2> ... [--hf-tokenizer <dir>]

import json
import sys


def _genai_tokenizer(gguf_path):
    import openvino_genai as genai
    return genai.Tokenizer(gguf_path)


def main():
    args = sys.argv[1:]

    # Parse --hf-tokenizer <dir> from the tail
    hf_tokenizer = None
    clean = []
    i = 0
    while i < len(args):
        if args[i] == "--hf-tokenizer" and i + 1 < len(args):
            hf_tokenizer = args[i + 1]
            i += 2
        else:
            clean.append(args[i])
            i += 1
    args = clean

    if len(args) < 2:
        print("Usage: gguf_tokenize.py encode <gguf> <text> [--hf-tokenizer <dir>]", file=sys.stderr)
        print("       gguf_tokenize.py decode <gguf> <id1> ... [--hf-tokenizer <dir>]", file=sys.stderr)
        sys.exit(1)

    # Parse --chat flag
    apply_chat = "--chat" in args
    args = [a for a in args if a != "--chat"]

    mode = args[0]
    gguf_path = args[1]

    if mode == "encode":
        text = args[2] if len(args) > 2 else ""
        try:
            tok = _genai_tokenizer(gguf_path)
            eos_id = tok.get_eos_token_id()
            if apply_chat:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                )
            ids = tok.encode(text).input_ids.data.flatten().tolist()
            print(json.dumps({"ids": ids, "eos_id": eos_id}))
            return
        except Exception as e:
            if not hf_tokenizer:
                raise RuntimeError(f"genai tokenizer failed and no --hf-tokenizer given: {e}") from e
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
        eos_id = hf.eos_token_id if hf.eos_token_id is not None else -1
        if apply_chat and hasattr(hf, "apply_chat_template") and hf.chat_template:
            ids = hf.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
            )
        else:
            ids = hf(text, return_tensors="np")["input_ids"][0].tolist()
        print(json.dumps({"ids": ids, "eos_id": eos_id}))

    elif mode == "decode":
        ids = [int(x) for x in args[2:]]
        try:
            tok = _genai_tokenizer(gguf_path)
            eos_id = tok.get_eos_token_id()
            text = tok.decode(ids)
            print(json.dumps({"text": text, "eos_id": eos_id}))
            return
        except Exception as e:
            if not hf_tokenizer:
                raise RuntimeError(f"genai tokenizer failed and no --hf-tokenizer given: {e}") from e
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
        eos_id = hf.eos_token_id if hf.eos_token_id is not None else -1
        text = hf.decode(ids)
        print(json.dumps({"text": text, "eos_id": eos_id}))

    else:
        print(f"Unknown mode: {mode!r}. Use 'encode' or 'decode'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
