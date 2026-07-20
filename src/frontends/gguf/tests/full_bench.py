#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Full GGUF frontend benchmark:
#   - Conversion time (read_model) + peak RAM
#   - compile_model time + peak RAM
#   - Inference: prefill tok/s, decode tok/s, generate until EOS
#   - Correctness: compare generated text vs llama.cpp greedy decode
#
# The tokenizer is read directly from the GGUF file via openvino_genai.Tokenizer,
# invoked in a subprocess so it doesn't conflict with the OV 2026 Python bindings.
#
# Usage:
#   PYTHONPATH=<ov>/bin/intel64/Release/python \
#   LD_LIBRARY_PATH=<ov>/bin/intel64/Release \
#   python3 full_bench.py --gguf model1.gguf model2.gguf ... \
#       [--prompt "..."] [--gen-tokens 64]

import argparse
import json
import os
import re
import subprocess
import sys
import time
import resource

import numpy as np

# ---------------------------------------------------------------------------
# Paths — adjust if needed
# ---------------------------------------------------------------------------

OV_BIN = "/home/vmaxim/openvino/bin/intel64/Release"
# genai built against OV 2026.3 — same ABI as this process.
GENAI_BUILD = "/home/vmaxim/openvino.genai/build/openvino_genai"
TOKENIZE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_tokenize.py")
LLAMA_SIMPLE_DEFAULT = "/home/vmaxim/llama.cpp/build-ref/bin/llama-simple"

# The tokenize subprocess needs openvino_genai importable and OV 2026 libs on the path.
GENAI_ENV = {
    **{k: v for k, v in os.environ.items()
       if k not in ("LD_LIBRARY_PATH", "PYTHONPATH")},
    "LD_LIBRARY_PATH": f"{GENAI_BUILD}:{OV_BIN}",
    "PYTHONPATH": f"{os.path.dirname(GENAI_BUILD)}:{OV_BIN}/python",
}
GENAI_PYTHON = "/home/vmaxim/openvino/.venv/bin/python3"

# ---------------------------------------------------------------------------
# Tokenizer via subprocess (avoids OV 2025/2026 pybind11 clash)
# ---------------------------------------------------------------------------

def _tok_cmd(gguf_path, subcommand, extra_args, hf_tokenizer=None, chat=False):
    cmd = [GENAI_PYTHON, TOKENIZE_SCRIPT, subcommand, gguf_path] + extra_args
    if hf_tokenizer:
        cmd += ["--hf-tokenizer", hf_tokenizer]
    if chat:
        cmd += ["--chat"]
    return cmd


def tokenize(gguf_path, text, hf_tokenizer=None, chat=False):
    """Returns (ids: list[int], eos_id: int)."""
    result = subprocess.run(
        _tok_cmd(gguf_path, "encode", [text], hf_tokenizer, chat=chat),
        capture_output=True, text=True, env=GENAI_ENV, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tokenize failed: {result.stderr}")
    data = json.loads(result.stdout)
    return data["ids"], data["eos_id"]


def detokenize(gguf_path, ids, hf_tokenizer=None, chat=False):
    """Returns decoded string."""
    result = subprocess.run(
        _tok_cmd(gguf_path, "decode", [str(i) for i in ids], hf_tokenizer, chat=chat),
        capture_output=True, text=True, env=GENAI_ENV, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"detokenize failed: {result.stderr}")
    data = json.loads(result.stdout)
    return data["text"]

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def reset_hwm():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5\n")
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# OV helpers
# ---------------------------------------------------------------------------

def build_inputs(tokens, past_len, model_inputs):
    import openvino as ov
    n = len(tokens)
    total = past_len + n
    inp_tokens = np.array(tokens, dtype=np.int32).reshape(1, 1, 1, n)
    inp_pos = np.arange(past_len, past_len + n, dtype=np.int32).reshape(1, 1, 1, n)
    inp_out_ids = np.array([n - 1], dtype=np.int32).reshape(1, 1, 1, 1)
    mask = np.zeros((1, 1, n, total), dtype=np.float32)
    for i in range(n):
        mask[0, 0, i, past_len + i + 1:] = -np.inf
    token_len = np.array([n], dtype=np.int64)
    beam_idx = np.zeros((1,), dtype=np.int32)
    raw = {
        "inp_tokens": inp_tokens,
        "inp_pos": inp_pos,
        "inp_out_ids": inp_out_ids,
        "self_kq_mask": mask,
        "self_kq_mask_swa": mask.copy(),
        "token_len_per_seq": token_len,
        "beam_idx": beam_idx,
    }
    return {k: ov.Tensor(v) for k, v in raw.items() if k in model_inputs}


def greedy_decode_ov(req, model_inputs, prompt_ids, max_tokens, eos_id):
    """Greedy decode until EOS or max_tokens. Returns (token_ids, t_prefill, t_decode)."""
    t0 = time.perf_counter()
    out = req.infer(build_inputs(prompt_ids, 0, model_inputs))
    logits = list(out.values())[0].reshape(-1)
    next_id = int(logits.argmax())
    t_prefill = time.perf_counter() - t0

    generated = [next_id]
    past = len(prompt_ids)

    t0 = time.perf_counter()
    while len(generated) < max_tokens and next_id != eos_id:
        out = req.infer(build_inputs([next_id], past, model_inputs))
        logits = list(out.values())[0].reshape(-1)
        next_id = int(logits.argmax())
        generated.append(next_id)
        past += 1
    t_decode = time.perf_counter() - t0
    return generated, t_prefill, t_decode

# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------

def run_llamacpp(llama_simple, gguf_path, prompt_text, max_tokens):
    env = dict(os.environ)
    lib_dir = str(__import__("pathlib").Path(llama_simple).parent)
    old = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{old}" if old else lib_dir
    cmd = [llama_simple, "-m", gguf_path, "-n", str(max_tokens), prompt_text]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    return result.stdout, result.stderr, result.returncode


def parse_llamacpp_perf(combined):
    def _tps(label):
        m = re.search(rf"{re.escape(label)}.*?([\d.]+)\s+tokens per second", combined)
        return float(m.group(1)) if m else None
    def _ms(label):
        m = re.search(rf"{re.escape(label)}\s*=\s*([\d.]+)\s*ms", combined)
        return float(m.group(1)) / 1000.0 if m else None
    return {
        "load_s": _ms("load time"),
        "pp_tps": _tps("prompt eval time"),
        "tg_tps": _tps("        eval time"),
    }


def extract_llamacpp_output(stdout, prompt):
    """Extract generated text (continuation after the prompt) from llama-simple stdout."""
    lines = []
    for line in stdout.split("\n"):
        if any(line.startswith(p) for p in ("llama_", "ggml_", "[", "main:", "Log ")):
            continue
        lines.append(line)
    full = "\n".join(lines).strip()
    # llama-simple echoes the prompt first; strip it off.
    if full.startswith(prompt):
        full = full[len(prompt):]
    return full.strip()

# ---------------------------------------------------------------------------
# Main benchmark for one model
# ---------------------------------------------------------------------------

def bench_one(gguf_path, prompt, max_tokens, llama_simple, device="CPU", hf_tokenizer=None, chat=False):
    import openvino as ov

    gguf_mb = os.stat(gguf_path).st_size / (1024 * 1024)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Model : {os.path.basename(gguf_path)}")
    print(f"GGUF  : {gguf_mb:.0f} MB   max_tokens={max_tokens}")
    print(sep)

    # ---- Tokenizer from GGUF ----
    prompt_ids, eos_id = tokenize(gguf_path, prompt, hf_tokenizer, chat=chat)
    print(f"EOS token id : {eos_id}")
    print(f"Chat template: {'yes' if chat else 'no'}")
    print(f"Prompt ({len(prompt_ids)} tokens): {prompt!r}")

    # ---- OV read_model ----
    core = ov.Core()
    can_reset = reset_hwm()

    t0 = time.perf_counter()
    model = core.read_model(gguf_path)
    t_read = time.perf_counter() - t0
    rss_read = rss_mb()

    num_ops = len(model.get_ops())
    const_mb = sum(
        op.get_byte_size() / (1024 * 1024)
        for op in model.get_ops()
        if op.get_type_name() == "Constant"
    )
    print(f"\n[OV read_model]")
    print(f"  time      : {t_read:.2f}s")
    print(f"  peak RAM  : {rss_read:.0f} MB  {'(reset before)' if can_reset else '(cumulative VmHWM)'}")
    print(f"  ops       : {num_ops}")
    print(f"  const_mb  : {const_mb:.0f} MB")

    # ---- OV compile_model ----
    reset_hwm()
    rss_before_compile = rss_mb()
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    t_compile = time.perf_counter() - t0
    rss_compile = rss_mb()

    print(f"\n[OV compile_model ({device})]")
    print(f"  time      : {t_compile:.2f}s")
    print(f"  peak RAM  : {rss_compile:.0f} MB  (before: {rss_before_compile:.0f} MB)")

    req = compiled.create_infer_request()
    model_inputs = {n for p in compiled.inputs for n in p.get_names()}

    # ---- Warm-up (uncounted) ----
    req.infer(build_inputs(prompt_ids[:1], 0, model_inputs))
    req.infer(build_inputs([0], 1, model_inputs))
    for s in req.query_state():
        s.reset()

    # ---- OV greedy decode ----
    ov_tokens, t_prefill, t_decode = greedy_decode_ov(
        req, model_inputs, prompt_ids, max_tokens, eos_id
    )
    ov_text = detokenize(gguf_path, ov_tokens, hf_tokenizer, chat=chat)

    n_prefill = len(prompt_ids)
    n_decode = len(ov_tokens) - 1
    pp_tps = n_prefill / t_prefill if t_prefill > 0 else float("nan")
    tg_tps = n_decode / t_decode if t_decode > 0 and n_decode > 0 else float("nan")
    hit_eos = ov_tokens and ov_tokens[-1] == eos_id

    print(f"\n[OV inference]")
    print(f"  prefill   : {pp_tps:.1f} tok/s  ({n_prefill} tokens, {t_prefill*1000:.0f}ms)")
    print(f"  decode    : {tg_tps:.1f} tok/s  ({n_decode} tokens, {t_decode*1000:.0f}ms)")
    print(f"  EOS hit   : {hit_eos}  (generated {len(ov_tokens)} tokens total)")
    print(f"  output    : {ov_text!r}")

    # ---- llama.cpp ----
    lc_result = None
    lc_prompt = prompt
    if chat:
        # Build a minimal chat-formatted string for llama-simple (which takes a raw string).
        # llama-simple doesn't apply a chat template itself, so we pass the formatted text.
        try:
            hf_tok_path = hf_tokenizer
            _formatted_ids, _ = tokenize(gguf_path, prompt, hf_tok_path, chat=True)
            lc_prompt = detokenize(gguf_path, _formatted_ids, hf_tok_path)
        except Exception:
            lc_prompt = prompt

    if llama_simple and os.path.isfile(llama_simple):
        print(f"\n[llama.cpp]")
        try:
            stdout, stderr, rc = run_llamacpp(llama_simple, gguf_path, lc_prompt, max_tokens)
            combined = stdout + stderr
            lc_perf = parse_llamacpp_perf(combined)
            lc_text = extract_llamacpp_output(stdout, lc_prompt)
            lc_result = {"perf": lc_perf, "text": lc_text, "returncode": rc}

            if lc_perf["load_s"]:
                print(f"  load      : {lc_perf['load_s']:.2f}s")
            if lc_perf["pp_tps"]:
                print(f"  prefill   : {lc_perf['pp_tps']:.1f} tok/s")
            if lc_perf["tg_tps"]:
                print(f"  decode    : {lc_perf['tg_tps']:.1f} tok/s")
            if rc != 0:
                print(f"  [WARN] llama-simple exit code {rc}")
            print(f"  output    : {lc_text!r}")

            # Correctness: word-level prefix comparison
            ov_words = ov_text.strip().split()
            lc_words = lc_text.strip().split()
            n = min(len(ov_words), len(lc_words), 20)
            if n > 0:
                match = sum(a == b for a, b in zip(ov_words[:n], lc_words[:n]))
                print(f"  correctness: {match}/{n} words match in first {n}")
            else:
                print(f"  correctness: insufficient output to compare")
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] llama-simple timed out")
        except Exception as e:
            print(f"  [ERROR] {e}")
    else:
        print(f"\n[llama.cpp] skipped (binary not found: {llama_simple})")

    # ---- Summary ----
    print(f"\n[Summary for {os.path.basename(gguf_path)}]")
    print(f"  read_model : {t_read:.2f}s  |  peak RAM {rss_read:.0f} MB")
    print(f"  compile    : {t_compile:.2f}s  |  peak RAM {rss_compile:.0f} MB")
    print(f"  OV prefill : {pp_tps:.1f} tok/s  |  decode : {tg_tps:.1f} tok/s")
    if lc_result and lc_result["perf"]["tg_tps"]:
        ratio = tg_tps / lc_result["perf"]["tg_tps"]
        print(f"  llama.cpp  : {lc_result['perf']['tg_tps']:.1f} tok/s decode")
        print(f"  OV / llama : {ratio:.2f}x")

    return {
        "gguf": gguf_path,
        "gguf_mb": gguf_mb,
        "read_s": t_read,
        "read_rss_mb": rss_read,
        "compile_s": t_compile,
        "compile_rss_mb": rss_compile,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "ov_text": ov_text,
        "eos_hit": hit_eos,
        "n_generated": len(ov_tokens),
        "llama": lc_result,
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GGUF frontend full benchmark")
    ap.add_argument("--gguf", nargs="+", required=True, help="One or more .gguf files")
    ap.add_argument("--prompt", default="The capital of France is Paris. Tell me more about this city.",
                    help="Prompt text (tokenizer read from GGUF)")
    ap.add_argument("--gen-tokens", type=int, default=64, help="Max tokens to generate")
    ap.add_argument("--llama-simple", default=LLAMA_SIMPLE_DEFAULT)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--hf-tokenizer", default=None,
                    help="HF tokenizer dir fallback for all models (overridden by per-model map)")
    ap.add_argument("--chat", action="store_true",
                    help="Wrap prompt in chat template via openvino_genai.Tokenizer.apply_chat_template")
    args = ap.parse_args()

    # Per-model HF tokenizer fallback map: used when genai can't read the GGUF tokenizer.
    HF_TOKENIZER_MAP = {
        "tinyllama": "/home/vmaxim/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "minicpm":   "/home/vmaxim/.cache/huggingface/hub/models--openbmb--MiniCPM-2B-dpo-bf16/snapshots/9d6de77274c9f364037bf8506b4a831d3520ae79",
        "minicpm2b": "/home/vmaxim/.cache/huggingface/hub/models--openbmb--MiniCPM-2B-dpo-bf16/snapshots/9d6de77274c9f364037bf8506b4a831d3520ae79",
        "phi-3":     "/home/vmaxim/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e",
        "phi3":      "/home/vmaxim/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e",
        "olmoe":     "/home/vmaxim/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924-Instruct/snapshots/7f1c97f440f06ce36705e4f2b843edb5925f4498",
    }

    def _hf_tok_for(gguf_path):
        if args.hf_tokenizer:
            return args.hf_tokenizer
        name = os.path.basename(gguf_path).lower()
        for key, path in HF_TOKENIZER_MAP.items():
            if key in name:
                return path
        return None

    llama_bin = None if args.skip_llama else args.llama_simple
    results = []

    for gguf in args.gguf:
        try:
            r = bench_one(gguf, args.prompt, args.gen_tokens, llama_bin, args.device,
                          hf_tokenizer=_hf_tok_for(gguf), chat=args.chat)
            results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {gguf}: {e}")
            import traceback; traceback.print_exc()

    # Final table
    print(f"\n\n{'='*80}")
    print("FINAL TABLE")
    print(f"{'='*80}")
    hdr = (f"{'Model':<36} {'GGUF':>5} {'Rd(s)':>6} {'RdMB':>5} {'Cp(s)':>6} {'CpMB':>6}"
           f" {'PP':>7} {'TG':>7} {'LC_TG':>7} {'Ratio':>6} {'EOS':>4} {'Gen':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        name = os.path.basename(r["gguf"])[:35]
        lc_tg = (r["llama"]["perf"]["tg_tps"]
                 if r["llama"] and r["llama"]["perf"]["tg_tps"] else 0)
        ratio = r["tg_tps"] / lc_tg if lc_tg else 0
        eos_str = "yes" if r["eos_hit"] else "no"
        print(f"{name:<36} {r['gguf_mb']:>5.0f} {r['read_s']:>6.1f} {r['read_rss_mb']:>5.0f}"
              f" {r['compile_s']:>6.1f} {r['compile_rss_mb']:>6.0f}"
              f" {r['pp_tps']:>7.1f} {r['tg_tps']:>7.1f} {lc_tg:>7.1f} {ratio:>6.2f}"
              f" {eos_str:>4} {r['n_generated']:>4}")


if __name__ == "__main__":
    sys.exit(main())
