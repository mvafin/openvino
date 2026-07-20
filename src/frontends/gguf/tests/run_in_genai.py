#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end: take a .gguf, convert it with the OpenVINO GGUF frontend, IO-adapt it to the
# GenAI LLMPipeline contract (genai_io_adapter), assemble a genai model directory
# (openvino_model.xml + tokenizer + configs), and run it through openvino_genai.LLMPipeline.
#
# This demonstrates the frontend's model running under GenAI. It requires an
# openvino_genai build/wheel that is ABI-compatible with the OpenVINO used to convert the
# gguf. When converting with a dev build of OpenVINO (this repo) and running with a release
# genai, save the adapted IR with the dev build (this script's --prepare step) and run with
# the release genai (the --run step) -- IR is version-portable.
#
# Steps:
#   # 1. prepare (dev OpenVINO that has the GGUF frontend):
#   PYTHONPATH=<ov>/bin/intel64/Release/python LD_LIBRARY_PATH=<ov>/bin/intel64/Release \
#     python3 run_in_genai.py prepare --gguf model.gguf --tokenizer <hf_dir> --out /tmp/m
#   # 2. run (any ABI-compatible openvino_genai):
#   python3 run_in_genai.py run --model-dir /tmp/m --prompt "The capital of France is"

import argparse
import os
import shutil
import sys


def cmd_prepare(args):
    import openvino as ov
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from genai_io_adapter import adapt_to_genai

    os.makedirs(args.out, exist_ok=True)
    core = ov.Core()
    adapted = adapt_to_genai(core.read_model(args.gguf))
    ov.save_model(adapted, os.path.join(args.out, "openvino_model.xml"), compress_to_fp16=False)

    # tokenizer + configs (needs openvino_tokenizers + transformers)
    from transformers import AutoTokenizer
    from openvino_tokenizers import convert_tokenizer

    hf = AutoTokenizer.from_pretrained(args.tokenizer)
    ov_tok, ov_detok = convert_tokenizer(hf, with_detokenizer=True)
    ov.save_model(ov_tok, os.path.join(args.out, "openvino_tokenizer.xml"))
    ov.save_model(ov_detok, os.path.join(args.out, "openvino_detokenizer.xml"))
    for fn in ("config.json", "generation_config.json", "tokenizer_config.json"):
        src = os.path.join(args.tokenizer, fn)
        if os.path.exists(src):
            shutil.copy(src, args.out)
    print("prepared genai model dir at", args.out)
    print("contents:", sorted(os.listdir(args.out)))


def cmd_run(args):
    import openvino_genai as genai

    pipe = genai.LLMPipeline(args.model_dir, args.device)
    cfg = genai.GenerationConfig()
    cfg.max_new_tokens = args.n
    cfg.do_sample = False
    cfg.apply_chat_template = args.chat_template
    out = pipe.generate(args.prompt, cfg)
    print(f"prompt: {args.prompt!r}")
    print(f"genai output: {str(out)!r}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="convert gguf via GGUF frontend + adapt + assemble genai dir")
    p.add_argument("--gguf", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default="/tmp/qwen3_genai")
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser("run", help="run the prepared dir through openvino_genai.LLMPipeline")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--prompt", default="The capital of France is")
    r.add_argument("--n", type=int, default=16)
    r.add_argument("--device", default="CPU")
    r.add_argument("--chat-template", action="store_true")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
