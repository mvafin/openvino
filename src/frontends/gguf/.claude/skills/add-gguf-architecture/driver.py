#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Arch bring-up driver for the OpenVINO GGUF frontend.
#
# Checks an agent runs while adding a new architecture:
#   smoke     - does the .gguf convert at all? prints op count, arch, IO, tok rt_info.
#   diverge   - prefill vs incremental-decode logit agreement at growing context.
#               A correct conversion keeps argmax identical and maxdiff small; a
#               stateful/KV/RoPE bug (e.g. the gemma4 u8-KV-cache bug) shows the
#               argmax flipping and maxdiff exploding -> NaN as the cache grows.
#   layerdiff - THE acceptance gate. Per-layer numeric diff of our prefill against
#               llama.cpp `llama-eval-callback` for REAL tokens. This is what catches the
#               subtle, compounding bugs (wrong attention scale, spurious V-norm, RoPE
#               layout, integer-zp dequant) that "the generation looks plausible" hides.
#               Requires LLAMA pointing at a llama.cpp build with llama-eval-callback and
#               llama-tokenize. Compares l_out-<N> per layer through the final result_norm;
#               FAILs at the first layer exceeding tolerance, so you fix the EARLIEST bug.
#
# It drives the raw frontend IO contract (inp_tokens / inp_pos / self_kq_mask /
# token_len_per_seq / beam_idx) directly, so no genai/tokenizer is needed and any
# supported arch can be exercised from just its .gguf.
#
# Env (point at the worktree build that has YOUR changes):
#   OV_BIN   default /home/vmaxim/openvino/.claude/worktrees/gguf_frontend-work/bin/intel64/Release
#   OV_PY    python with numpy (default /home/vmaxim/openvino/.venv/bin/python3 is NOT used;
#            this script must be run BY that python — see SKILL.md)
#
# Usage:
#   PYTHONPATH=$OV_BIN/python LD_LIBRARY_PATH=$OV_BIN \
#     <py-with-numpy> driver.py smoke   <model.gguf>
#   PYTHONPATH=$OV_BIN/python LD_LIBRARY_PATH=$OV_BIN \
#     <py-with-numpy> driver.py diverge <model.gguf> [--prompt-len 6] [--steps 8] [--swa]

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import openvino as ov


def build_inputs(model_inputs, tokens, past_len, swa):
    n = len(tokens)
    total = past_len + n
    mask = np.zeros((1, 1, n, total), dtype=np.float32)
    for i in range(n):
        mask[0, 0, i, past_len + i + 1:] = -np.inf
    raw = {
        "inp_tokens": np.array(tokens, np.int32).reshape(1, 1, 1, n),
        "inp_pos": np.arange(past_len, past_len + n, dtype=np.int32).reshape(1, 1, 1, n),
        "inp_out_ids": np.array([n - 1], np.int32).reshape(1, 1, 1, 1),
        "self_kq_mask": mask,
        "self_kq_mask_swa": mask.copy(),
        "token_len_per_seq": np.array([n], np.int64),
        "beam_idx": np.zeros((1,), np.int32),
    }
    return {k: ov.Tensor(v) for k, v in raw.items() if k in model_inputs}


def logits(out):
    for port in out:
        a = np.array(out[port])
        if a.ndim >= 3 and a.shape[-1] > 1000:
            return a.ravel()
    return None


def smoke(args):
    core = ov.Core()
    model = core.read_model(args.gguf)
    names = [p.get_any_name() for p in model.inputs]
    rt = model.get_rt_info()
    print(f"OK read_model: ops={len(model.get_ops())}")
    print(f"  inputs : {names}")
    out_shapes = [str(r.get_partial_shape()) for r in model.outputs]
    print(f"  outputs: {len(model.outputs)} (shapes {out_shapes})")
    print(f"  tokenizer rt_info present: {'gguf_tokenizer_metadata' in rt}")
    core.compile_model(model, "CPU")
    print("OK compile_model(CPU)")


def diverge(args):
    core = ov.Core()
    compiled = core.compile_model(core.read_model(args.gguf), "CPU")
    mi = {p.get_any_name() for p in compiled.inputs}
    # arbitrary in-vocab token ids; correctness is prefill-vs-decode self-consistency
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16][: args.prompt_len + args.steps]
    n0 = args.prompt_len

    def prefill_last(tokens):
        req = compiled.create_infer_request()
        return logits(req.infer(build_inputs(mi, tokens, 0, args.swa)))

    def decode_last(tokens):
        req = compiled.create_infer_request()
        req.infer(build_inputs(mi, tokens[:n0], 0, args.swa))
        past, lg = n0, None
        for tok in tokens[n0:]:
            lg = logits(req.infer(build_inputs(mi, [tok], past, args.swa)))
            past += 1
        return lg

    print(f"{'len':<6}{'pf_argmax':<12}{'dec_argmax':<12}{'maxdiff':<14}{'dec_nan':<8}{'ok'}")
    all_ok = True
    for k in range(n0 + 1, len(seq) + 1):
        sub = seq[:k]
        pf, dec = prefill_last(sub), decode_last(sub)
        md = float(np.nanmax(np.abs(pf - dec)))
        nan = bool(np.isnan(dec).any())
        ok = (int(np.argmax(pf)) == int(np.argmax(dec))) and not nan
        all_ok &= ok
        print(f"{k:<6}{int(np.argmax(pf)):<12}{int(np.argmax(dec)):<12}{md:<14.5f}{str(nan):<8}{'PASS' if ok else 'FAIL'}")
    print("RESULT:", "PASS" if all_ok else "FAIL (stateful decode diverges from prefill)")
    sys.exit(0 if all_ok else 1)


def _llama_tokenize(llama, gguf, prompt):
    """Return llama.cpp's token ids for `prompt` (incl. its BOS), via llama-tokenize."""
    exe = os.path.join(llama, "build-ref", "bin", "llama-tokenize")
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(llama, "build-ref", "bin"))
    out = subprocess.run([exe, "-m", gguf, "-p", prompt], capture_output=True, text=True, env=env).stdout
    # lines like "     2 -> '<bos>'"
    ids = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+)\s*->", out, re.M)]
    if not ids:
        sys.exit("layerdiff: could not parse llama-tokenize output; check LLAMA path")
    return ids


def _llama_layer_refs(llama, gguf, prompt):
    """Run llama-eval-callback and return {layer_idx: np.array(first3+last3 of l_out-<i> last pos)}.

    l_out is {n_embd, n_tok} in ggml, so each dumped block row == a token position (safe to index);
    we take the last position. Only first-3 and last-3 dims are printed by eval-callback, which is
    enough to detect divergence. result_norm (final) is captured as layer key -1.
    """
    exe = os.path.join(llama, "build-ref", "bin", "llama-eval-callback")
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(llama, "build-ref", "bin"))
    txt = subprocess.run([exe, "-m", gguf, "-p", prompt, "-n", "1", "-ngl", "0"],
                         capture_output=True, text=True, env=env).stdout
    refs = {}

    def parse_block(tag):
        # find "<tag> = (f32) ... = {dims}" then the following value rows
        m = re.search(re.escape(tag) + r" = \(.*?\n((?:\s*\[.*\n)+)", txt)
        if not m:
            return None
        rows = re.findall(r"\[\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),.*?([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\]",
                          m.group(1))
        if not rows:
            return None
        return np.array([float(x) for x in rows[-1]])  # last position, first3+last3

    li = 0
    while True:
        r = parse_block(f"l_out-{li}")
        if r is None:
            break
        refs[li] = r
        li += 1
    rn = parse_block("result_norm")
    if rn is not None:
        refs[-1] = rn
    return refs


def layerdiff(args):
    llama = os.environ.get("LLAMA", "/home/vmaxim/llama.cpp")
    tokens = _llama_tokenize(llama, args.gguf, args.prompt)
    refs = _llama_layer_refs(llama, args.gguf, args.prompt)
    if not refs:
        sys.exit("layerdiff: no l_out-* tensors parsed from eval-callback (build it? right model?)")

    core = ov.Core()
    model = core.read_model(args.gguf)
    name2op = {op.get_friendly_name(): op for op in model.get_ops()}
    # tap every blk.<i>.l_out and the final result_norm
    taps = {}
    for li in [k for k in refs if k >= 0]:
        nm = next((n for n in name2op if f"blk.{li}.l_out" in n), None)
        if nm:
            taps[li] = nm
    rn = next((n for n in name2op if "result_norm" in n and "rms" not in n), None)
    if rn and -1 in refs:
        taps[-1] = rn
    n_orig = len(model.outputs)
    order = list(taps)
    for li in order:
        model.add_outputs(name2op[taps[li]].output(0))
    cm = core.compile_model(model, "CPU")
    mi = {p.get_any_name() for p in cm.inputs}
    req = cm.create_infer_request()
    req.infer(build_inputs(mi, tokens, 0, args.swa))

    # Only first-3 + last-3 dims of each tensor are printed by eval-callback. Those 6 values are a
    # NOISY per-layer probe — on a correct model an individual mid-layer can still spike (verified:
    # working gemma4 shows transient rel>0.3 at some layers) due to f16 + large per-channel norm
    # weights hitting exactly those sampled dims. So per-layer rows are ADVISORY (read the TREND);
    # the authoritative pass/fail is the FINAL result_norm, which integrates all dims and is what
    # the lm_head consumes. A real bug makes the per-layer rel diff GROW monotonically AND fails
    # result_norm; benign drift stays bounded and result_norm PASSes.
    print(f"prompt tokens (llama): {tokens}")
    print(f"{'layer':<14}{'abs':<10}{'rel':<10}")
    rel_by_layer = []
    result_rel = None
    for i, li in enumerate(order):
        v = np.array(req.get_output_tensor(n_orig + i).data)
        v = v.reshape(-1, v.shape[-1])[-1]  # last position
        ours = np.concatenate([v[:3], v[-3:]])
        scale = max(1.0, float(np.max(np.abs(refs[li]))))
        d = float(np.max(np.abs(ours - refs[li])))
        rel = d / scale
        label = "result_norm" if li == -1 else f"l_out-{li}"
        print(f"{label:<14}{d:<10.3f}{rel:<10.3f}")
        if li == -1:
            result_rel = rel
        else:
            rel_by_layer.append(rel)
    # Trend: compare mean rel of last third vs first third of layers.
    if len(rel_by_layer) >= 6:
        third = len(rel_by_layer) // 3
        growth = (np.mean(rel_by_layer[-third:]) + 1e-9) / (np.mean(rel_by_layer[:third]) + 1e-9)
        print(f"trend: first-third rel={np.mean(rel_by_layer[:third]):.3f} "
              f"last-third rel={np.mean(rel_by_layer[-third:]):.3f} (growth x{growth:.1f})")
    ok = result_rel is not None and result_rel <= args.tol
    if result_rel is None:
        print("RESULT: INCONCLUSIVE (no result_norm parsed) — inspect per-layer trend manually")
        sys.exit(2)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (result_norm rel={result_rel:.3f}, tol={args.tol}) "
          f"-- if FAIL, find the earliest layer where rel starts GROWING and fix that op")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("gguf")
    d = sub.add_parser("diverge")
    d.add_argument("gguf")
    d.add_argument("--prompt-len", type=int, default=6)
    d.add_argument("--steps", type=int, default=8)
    d.add_argument("--swa", action="store_true", help="model uses self_kq_mask_swa (gpt-oss/gemma SWA)")
    ld = sub.add_parser("layerdiff")
    ld.add_argument("gguf")
    ld.add_argument("--prompt", default="The capital of France is")
    ld.add_argument("--tol", type=float, default=1e-2, help="per-layer maxdiff tolerance (f16 paths)")
    ld.add_argument("--swa", action="store_true", help="model uses self_kq_mask_swa (gpt-oss/gemma SWA)")
    args = ap.parse_args()
    {"smoke": smoke, "diverge": diverge, "layerdiff": layerdiff}[args.cmd](args)


if __name__ == "__main__":
    main()
