---
name: add-gguf-architecture
description: Add support for a new model architecture to the OpenVINO GGUF frontend. Use when asked to add/enable a new arch (e.g. gemma, qwen3moe, deepseek, phi4) in the gguf frontend, port an architecture from llama.cpp, debug why a converted gguf model diverges from llama.cpp, add a new ggml op translator, wire up a gguf tokenizer, or benchmark a converted model's accuracy/perf/memory vs llama.cpp.
---

How to add a new architecture to the OpenVINO **GGUF frontend** (`src/frontends/gguf`):
declare it in the graph builder, compare the converted graph against llama.cpp tensor-by-tensor,
add any missing ggml op translators, wire up its tokenizer (genai side), add tests, and benchmark
accuracy / perf / memory / conversion-time vs llama.cpp.

The verification harness is [.claude/skills/add-gguf-architecture/driver.py](.claude/skills/add-gguf-architecture/driver.py)
(`smoke` = does it convert; `diverge` = does stateful decode match prefill). llama.cpp's
`llama-eval-callback` is the per-tensor graph reference.

All paths below are relative to `src/frontends/gguf/` unless absolute. The frontend builds to
`libopenvino_ggml_frontend.so`; the build tree used here is the worktree at
`/home/vmaxim/openvino/.claude/worktrees/gguf_frontend-work` (env `OV_BIN` below).

## Prerequisites

A built OpenVINO with the GGUF frontend enabled (`-DENABLE_OV_GGUF_FRONTEND=ON`), a python with
numpy, and llama.cpp built as the reference oracle. In this container they already exist:

```bash
export OV_BIN=/home/vmaxim/openvino/.claude/worktrees/gguf_frontend-work/bin/intel64/Release
export OV_PY=/home/vmaxim/openvino/.venv/bin/python3          # python that has numpy + openvino deps
export LLAMA=/home/vmaxim/llama.cpp/build-ref/bin             # llama-eval-callback, llama-tokenize, llama-simple
ls $OV_BIN/libopenvino_ggml_frontend.so $LLAMA/llama-eval-callback
```

## Build (after each source edit)

```bash
cd /home/vmaxim/openvino/.claude/worktrees/gguf_frontend-work/build
cmake --build . --target openvino_ggml_frontend -j$(nproc)
```

If you add a **new top-level `src/*.cpp`** (e.g. a new op translator under `src/op/` is auto-globbed,
but a new file directly under `src/` is not picked up until you re-glob), re-run cmake first:

```bash
cmake .                      # re-glob the source list (cached otherwise -> undefined symbol at FE load)
cmake --build . --target openvino_ggml_frontend -j$(nproc)
```

## Where each piece lives

| Step | File | What |
|---|---|---|
| 1. Declare arch | `src/builder/gguf_builder.cpp` → `supported_archs()` | add the gguf `general.architecture` string |
| 1. Arch features | `src/builder/gguf_builder.cpp` ctor | flags: `m_is_moe`, `m_is_geglu`, `m_has_swa`, `m_has_v_norm`, `arch_uses_neox_rope()` |
| 1. Hparams | `src/builder/gguf.cpp` → `config_from_meta()` | per-arch scalars (rope dims, swa pattern, shared_kv_layers, attention/embedding scales) |
| 3. New op | `src/op/<op>.cpp` + register in `src/op_table.cpp` | a ggml op the arch needs that isn't translated yet |
| 4. Tokenizer | `openvino.genai` `src/cpp/src/gguf_utils/gguf_tokenizer.cpp` | dispatch on `tokenizer.ggml.model`; +`openvino_tokenizers` factory if a new normalization op |
| 5. Tests | `src/frontends/gguf/tests/` | unit op tests + WWB accuracy test |

## Bringing up a new arch correctly the FIRST time (the acceptance gate)

A "converts and the output looks plausible" arch is NOT done. Instruction-tuned models emit
plausible-looking `\n`/EOS even when the graph is wrong, and deep models hide ~1%/layer numeric
drift until it flips the final argmax. Declaring an arch done from a smoke test plus an eyeballed
generation is how a single arch accumulates many separate, individually-subtle bugs (scale, norm,
RoPE, quant-dequant, tokenizer) that each look "almost right". Follow this gate instead:

1. **Validate the tokenizer and the model graph SEPARATELY — they mask each other.** A wrong
   tokenizer makes a correct model look broken, and vice versa; chasing them together wastes hours.
   - Tokenizer: `genai.encode(text)` must equal `llama-tokenize -m <model> -p "<text>"` **including
     the leading BOS/special tokens**. Check the BOS explicitly — some families are BOS-sensitive
     and a missing/wrong BOS alone produces garbage.
   - Model graph: always drive it with **llama.cpp's exact token ids** (not our tokenizer's), so a
     tokenizer bug can't contaminate the model comparison.
2. **Per-layer numeric diff vs `llama-eval-callback` is the acceptance gate, not generation.** Use
   `driver.py layerdiff` (below) on a REAL model with REAL tokens; require the final `result_norm`
   to match within tolerance, and the per-layer relative diff to stay bounded (not grow) through
   the LAST layer — not just the first few.
3. **Re-run the full diff after EVERY fix.** Fixing one bug from "this prompt looks better" usually
   leaves the next one lurking. One clean end-to-end diff beats ten eyeballed generations.
4. **Read the EXACT arch's `llama.cpp/src/models/<arch>.cpp` for every flag — never inherit a
   sibling's.** Sibling archs in the same family routinely differ in Q/K/V-norm, attention scale,
   SWA period/freq_base, and rope dims; copying a sibling's config silently corrupts the output.
   See the per-arch-flags gotcha below.
5. **Reference values for op/dequant tests must come from real ggml, never a numpy reimplementation
   of the formula** (`gen_ggml_reference.c`). A hand-rolled reference encodes the same wrong
   assumption as the code, so the test passes while the op is wrong. Use the dequant-vs-ggml suite
   (`test_dequant_vs_ggml.cpp`) for every quant type the arch uses.
6. **Don't assume tensor rank/layout is invariant across layers.** OV shape inference can hand the
   same logical tensor different ranks/layouts at different layers (depending on each layer's
   upstream producer), so an op that branches on rank or applies one fixed transpose can be correct
   for one layer and wrong for another. When normalizing a layout, **Reshape to the canonical shape
   from `output_shape` first** (element order is preserved, so it reinterprets any incoming rank),
   then transpose — don't branch on rank alone. Verify a shape-sensitive op behaves identically
   across layers (diff blk.0 AND a later block, not just blk.0).

## Finding the bug when a model "converts but outputs garbage"

Diagnose in this order — each step localizes the fault to a layer of the stack, so you stop
guessing:

1. **Tokens.** `genai.encode(text)` vs `llama-tokenize -m <model.gguf> -p "<text>"`. Mismatch ⇒
   tokenizer bug (step 5), not the model. Most "model" garbage is actually this.
2. **Prefill graph (this is where most real model bugs are).** With the llama.cpp tokens, diff
   per-layer activations vs `llama-eval-callback` (step 2 below) and find the first divergence.
   Fix it, re-diff, repeat until a raw single-forward argmax is sane.
3. **Stateful decode.** If prefill is correct but generation still repeats/drifts, the bug is in
   KV-cache / SWA-mask / RoPE-during-decode — feed identical tokens to the raw refeed model vs
   `LLMPipeline`: same prefill token but divergent step-2 token ⇒ stateful bug (step 3 below).
4. **Precision.** Correct in prefill, NaN/drift only deep in decode ⇒ KV-cache precision (u8 vs
   f16, large head sizes) — see gotchas.

## Run (agent path) — the bring-up loop

**1. Smoke: does the arch convert at all?**

```bash
PYTHONPATH=$OV_BIN/python LD_LIBRARY_PATH=$OV_BIN $OV_PY \
  .claude/skills/add-gguf-architecture/driver.py smoke <model.gguf>
```
Prints op count, the IO contract, output shape, and whether tokenizer rt_info was attached, then
compiles for CPU. A missing op or unsupported arch throws here with the exact op/arch name.

**1b. Acceptance gate: `layerdiff` (automated per-layer diff vs llama.cpp).** This is the single
most important check — run it after smoke and after EVERY fix. It tokenizes a real prompt with
`llama-tokenize`, runs `llama-eval-callback`, and compares our prefill `l_out-<N>` per layer
(through the final `result_norm`) against it, reporting the first divergence:

```bash
LLAMA=/home/vmaxim/llama.cpp PYTHONPATH=$OV_BIN/python LD_LIBRARY_PATH=$OV_BIN $OV_PY \
  .claude/skills/add-gguf-architecture/driver.py layerdiff <model.gguf> [--swa] [--tol 0.15]
```
Verdict semantics (important): only first-3+last-3 dims per tensor are printed by eval-callback, so
**per-layer rows are ADVISORY** (an individual mid-layer can spike >0.3 rel even on a correct model
— verified on working gemma4). The **authoritative pass/fail is the final `result_norm`** (it
integrates all dims and feeds the lm_head): working gemma3 and gemma4 both PASS with result_norm
rel ≈0.05-0.08. Read the per-layer rows as a TREND — a real bug makes rel **grow monotonically**
from the offending layer onward AND fails result_norm; benign f16 drift stays bounded (the printed
`trend: ... growth x..` line summarizes first-third vs last-third). When it FAILs, find the
earliest layer where rel starts growing and fix that op, then re-run. Do not declare the arch done
until result_norm PASSes. (`--swa` for sliding-window archs; default prompt "The capital of France
is".) The hand recipe in step 2 drills into a specific layer once the trend points at it.

**2. Graph comparison vs llama.cpp — the decisive technique for "converts but outputs garbage".**

When a model converts cleanly but generates nonsense, **do not debug by reading llama.cpp's
`src/models/<arch>.cpp` and matching hyperparameters by eye** — that is guess-and-check and you
will fix the wrong things while the output stays wrong. Instead diff the **actual numeric
activations** layer-by-layer against llama.cpp. This is what reliably localizes the bug.

Dump llama.cpp's per-tensor graph (op names + shapes + the first values of each tensor) as the
reference. `llama-eval-callback` prints every `cb()`-tagged tensor with a values block:

```bash
LD_LIBRARY_PATH=$LLAMA $LLAMA/llama-eval-callback -m <model.gguf> -p "The capital of France is" -n 1 \
  2>&1 | grep -A6 "Qcur_normed-0 = "      # name format: <tag>-<layer> ; e.g. attn_norm-0, kqv_out-0, l_out-0
```

The ggml `cb()` tags map almost 1:1 onto our builder's `blk.N.*` friendly names: `inp_scaled`↔
`embd_scaled`, `attn_norm-0`, `Qcur_normed-0`, `Kcur_normed-0`, `kqv_out-0`↔`blk.0.kqv_merged`,
`sa_out-0`↔`blk.0.ffn_inp`, `ffn_out-0`, `l_out-0`. Pick a handful spanning each sublayer.

On the OV side, tap those nodes by adding them as extra outputs and printing the same first values:

```python
import openvino as ov, numpy as np
core = ov.Core(); model = core.read_model("<model.gguf>")
n2o = {op.get_friendly_name(): op for op in model.get_ops()}
# resolve the auto-numbered names by substring, e.g. 'blk.0.Qcur_normed', 'blk.0.kqv_merged', 'blk.0.l_out'
picks = {lbl: next(n for n in n2o if sub in n and "rms" not in n)
         for lbl, sub in {"Qcur_normed":"blk.0.Qcur_normed","kqv":"blk.0.kqv_merged","l_out":"blk.0.l_out"}.items()}
n0 = len(model.outputs)
for nm in picks.values(): model.add_outputs(n2o[nm].output(0))
cm = core.compile_model(model, "CPU")
T = len(tokens); NEG = -65504.0          # build a [1,1,T,T] causal mask, 0 on/below diag else NEG
m = np.triu(np.full((T,T), NEG, np.float32), 1)
req = cm.create_infer_request(); req.infer({
    "inp_tokens": np.array(tokens,np.int32).reshape(1,1,1,T),
    "inp_pos": np.arange(T,dtype=np.int32).reshape(1,1,1,T),
    "inp_out_ids": np.arange(T,dtype=np.int32).reshape(1,1,1,T),
    "self_kq_mask": m.reshape(1,1,T,T), "self_kq_mask_swa": m.reshape(1,1,T,T),
    "token_len_per_seq": np.array([T],np.int64), "beam_idx": np.array([0],np.int32)})
for i,lbl in enumerate(picks): print(lbl, req.get_output_tensor(n0+i).data.reshape(-1)[:4])
```

Walk the tags **in execution order** and find the FIRST one whose values diverge — its inputs
matched, so the bug is in that op's translation (or the hparam feeding it). Worked example from
gemma3: `embd_scaled`/`attn_norm`/`Qcur_normed`/`Kcur_normed` all matched but `kqv_merged` was
`1.2` vs ref `8.25` → the bug was an extra V-norm (`m_has_v_norm` wrongly true) corrupting attention.

> **CRITICAL — verify the input tokens match llama.cpp FIRST.** A single wrong token at position N
> makes every later position diverge and masquerades as a model bug. Compare `embd_scaled`/
> `inp_scaled` at every position before trusting any downstream comparison; if position N's
> embedding differs, your token IDs are wrong (tokenizer mismatch / missing BOS), not the model.
> Get the reference tokens from `llama-tokenize -m <model.gguf> -p "<text>"` and feed those exact
> ids (including the leading `<bos>`=2 that llama.cpp prepends).

**Probe ONE node per run** when checking the fused attention path — adding many `Result`s disables
CPU SDPA fusion and changes the numerics. (For the non-fused activations above, several taps are
fine; just be aware fusion-sensitive nodes can shift.)

**3. Divergence check (catches stateful/KV/RoPE bugs).** A correct conversion produces identical
prefill and incremental-decode logits; a KV-cache / SWA / RoPE bug shows argmax flipping and
maxdiff → NaN as the cache grows:

```bash
PYTHONPATH=$OV_BIN/python LD_LIBRARY_PATH=$OV_BIN $OV_PY \
  .claude/skills/add-gguf-architecture/driver.py diverge <model.gguf> --steps 8 [--swa]
```
Pass `--swa` for sliding-window models (gpt-oss, gemma). Exit 0 = PASS (decode matches prefill).

**4. Accuracy / perf / memory / conversion-time vs llama.cpp.** The benchmark prints read time,
read RAM, compile time, compile RAM, prefill t/s, decode t/s, OV/llama decode ratio, and word-level
correctness in one table:

```bash
cp src/frontends/gguf/tests/full_bench.py src/frontends/gguf/tests/gguf_tokenize.py /tmp/
sed -i "s#OV_BIN = .*#OV_BIN = \"$OV_BIN\"#" /tmp/full_bench.py     # point it at your build
PYTHONPATH=$OV_BIN/python LD_LIBRARY_PATH=$OV_BIN $OV_PY \
  /tmp/full_bench.py --gguf <model.gguf> --gen-tokens 24
```
(`full_bench.py` hardcodes paths to the main repo; the `sed` redirects it to the worktree build.
It reads the tokenizer from the gguf via genai in a subprocess — if the genai tokenizer can't read
this arch's tokenizer yet, pass `--hf-tokenizer <hf_dir>` or finish step 4 first.)

**5. Tokenizer end-to-end (genai).** Once the genai tokenizer dispatch supports the arch, build
genai and run a sample on the raw `.gguf` (see the genai gguf-frontend branch). Token IDs must
match `llama-tokenize -m <model.gguf> -p "<text>"` (modulo the BOS the pipeline adds).

## Test

Unit op tests live in `tests/test_ops.cpp` / `tests/test_quant_dequant.cpp` (target
`ov_gguf_frontend_tests`, built only when `-DENABLE_TESTS=ON`). The accuracy regression test is
`openvino.genai/tools/who_what_benchmark/tests/test_cli_text_gguf.py` (gated behind
`WWB_GGUF_TESTS=1`), comparing the genai gguf path vs llama.cpp via WWB similarity. Add the new
arch's model to its `GGUF_MODELS` list.

### Reference values MUST come from ggml, not a numpy reimplementation

`tests/generate_test_data.py` reimplements each op's formula in numpy — so it can only ever
confirm the formula its author *guessed*. That is how the GELU translator shipped with the wrong
approximation: the committed `gelu_expected.npy` was numpy-`erf`, the translator was OV-`erf`, the
test compared erf-vs-erf and passed, while real ggml uses the **tanh** approximation. A test is
worthless as an oracle when it derives the expected value the same way as the code under test.

Use `tests/gen_ggml_reference.py` (wraps `gen_ggml_reference.c`, which links real `libggml` from
llama.cpp) to dump authoritative ggml op outputs as `.npy`. When adding an op test, generate its
reference this way — never hand-port the formula.

**Tolerance caveat (don't set it to 1e-5).** ggml computes GELU / GELU_QUICK through an f16 lookup
table (`GGML_GELU_FP16`, `ggml-cpu/vec.h`): input rounded to f16, used as a 16-bit table index,
output f16. That quantization is ~2e-3 — *larger* than the erf-vs-tanh formula gap (~4e-4). So an
op-level test against ggml must tolerate ~3e-3, and at that tolerance it canNOT distinguish erf
from tanh for a single GELU. Op vectors catch GROSS errors (wrong op / axis / constant / layout);
subtle formula drift is only caught by the **per-layer end-to-end diff vs `llama-eval-callback` on
a deep model** (see "Finding the bug"), where a ~1e-3 per-call error compounds over dozens of
layers into a flipped argmax. Both layers of defense are needed.

## Gotchas

- **Probing perturbs fusion.** Adding multiple `Result`s to inspect activations un-fuses the CPU
  `ScaledDotProductAttentionWithKVCache`, changing the numbers — a layer can look "clean" when
  probed but actually diverge. The authoritative, fusion-preserving signal is `req.query_state()`
  comparing the KV-cache tensors after prefill vs after decode (no graph mutation).
- **`diverge` is token-sensitive for SWA/shared-KV archs.** Verified live: gemma4 PASSes on real
  prompt tokens but the driver's synthetic low ids (1=`<eos>`, 2=`<bos>`, 3=`<unk>`…) hit a
  data-dependent NaN and the run FAILs. An all-NaN row IS a real signal, but before concluding a
  regression, re-run with realistic tokens (tokenize a sentence with `llama-tokenize`). For dense
  archs (llama/qwen) the synthetic ids are fine.
- **KV-cache precision.** The CPU plugin defaults KV cache to u8; large head sizes (>128, e.g.
  gemma's 256/512) lose too much precision and compound to NaN over decode. The frontend pins f16
  for large-head models — if a new large-head arch NaNs in decode but is fine in prefill, this is
  the first suspect (`KV_CACHE_PRECISION` rt_info / the head-size gate in the AdaptToGenAI pass).
- **RoPE style.** `arch_uses_neox_rope()` lists NEOX (rotate-halves: qwen/phi/gemma/...) vs NORMAL
  (rotate-pairs: llama/minicpm). Getting this wrong gives a graph that converts cleanly but
  produces garbage from layer 0 — compare `Qcur`/`Kcur` after ROPE against `llama-eval-callback`.
- **RoPE input rank/layout is NOT the same for every layer.** OV shape inference can hand the ROPE
  op different shapes per layer for the *same* logical tensor: rank-3 `[B,L,H*S]` when the producer
  is rank-3 (e.g. the embedding feeds blk.0), rank-4 `[B,L,H,S]`, OR rank-4 `[B,1,L,S]` when the
  producer is a rank-4 `l_out` and OV keeps a leading 1 with the head axis at position 1. A
  `to_bhls` that only handles one layout (e.g. a fixed Transpose `{0,2,1,3}`) silently mis-rotates
  the others — symptom: K reaching SDPA equals the PRE-rope value for blk.1+ while blk.0 is fine, so
  attention diverges on mid sequence positions and compounds (the original gemma3 blocker). FIX
  pattern: always `Reshape` the data to the canonical `[1,-1,n_head,head_size]` from `output_shape`
  (element order is preserved → correctly reinterprets any incoming rank) BEFORE the Transpose; keep
  cos/sin (always rank-4 `[B,L,1,head/2]`) on a transpose-only path. ALWAYS verify a shape-sensitive
  op produces identical results on blk.0 AND a later block (and a global vs SWA layer) — per-layer
  rank drift is invisible if you only check blk.0.
- **Per-arch feature flags are family-specific — never copy a sibling arch's flags wholesale.**
  Sibling archs in one family differ in subtle, output-destroying ways; read the *exact* arch's
  `llama.cpp/src/models/<arch>.cpp` for each flag. The gemma family makes a good cautionary set of
  examples (each bullet is one real flag that differs between gemma2/3/4):
  - **Q/K/V norm is independent per arch.** gemma3 norms Q and K only (`build_norm` on Qcur/Kcur);
    gemma4 ALSO norms V (`ggml_rms_norm(Vcur)`). `m_has_v_norm` must be set from the real arch, not
    inherited — a spurious V-norm corrupts attention output entirely (kqv off by ~7×).
  - **Attention scale.** Most archs use the default `1/sqrt(head_size)`. Some (gemma4) set an
    explicit `1.0` (no pre-attn scaling). gemma3 applies `1/sqrt(n_embd_head_k)` as a Qcur pre-scale
    with `build_attn(scale=1.0)` — numerically the DEFAULT branch, so it must NOT be forced to 1.0.
    Check `f_attention_scale` and how `build_attn`'s scale arg combines with any `ggml_scale(Qcur)`.
  - **SWA defaults come from llama.cpp struct defaults, not the GGUF.** gemma3's GGUF often omits
    `rope.freq_base_swa` and `sliding_window_pattern`; llama.cpp's gemma3 loader keeps the struct
    defaults (`rope_freq_base_train_swa = 10000`, `swa_period = 6`) rather than falling back to the
    global value. Other archs (gemma2/gemma4/cohere2) explicitly reset SWA freq_base to the global
    first. Match the specific loader. A wrong SWA freq_base ropes the sliding-window layers wrong
    and shows up as position-dependent divergence in `Qcur`/`kqv` for SWA layers only.
  - **Per-op RoPE trigger.** The shared sin/cos table (`add_rope_sin_cos`) can only serve ONE rope
    config. If SWA and global layers differ in n_dims OR freq_base, set `use_per_op_rope` so each
    ROPE op builds its own table (check the trigger in the builder ctor covers freq_base, not just
    n_dims). When per-op rope is active, confirm `make_sin_cos` is called with the correct
    `stateful` arg in `op/rope.cpp` — the default is `false`.
- **Tokenizer `model` ≠ arch.** `tokenizer.ggml.model` is the tokenizer family (`gpt2`/`llama`/
  `gemma4`), independent of `general.architecture`. gemma4 is `BPE` (not SPM) with a metaspace
  normalizer and newline-only pre-split — adding a tokenizer often means adding a normalization op
  to the `openvino_tokenizers` `create_tokenizer_node` factory, not just a dispatch branch.
- **Wrong token IDs look exactly like a model bug — check the tokenizer FIRST when output is
  garbage.** Always diff `genai_tokenizer.encode(text)` against `llama-tokenize -m <model.gguf> -p
  "<text>"`. A mismatched ID at any position poisons all downstream activations. Real example:
  genai's SPM proto builder (`gguf_tokenizer.cpp build_spm_model_proto`) hardcoded
  `add_dummy_prefix=true`, ignoring gemma3's `tokenizer.ggml.add_space_prefix=false`, so it emitted
  `669 (▁The)` instead of `818 (The)`. SPM `normalizer_spec` flags (`add_dummy_prefix`) and the
  `add_bos_token`/`add_space_prefix` GGUF keys must be honored, not assumed. The `▁` (U+2581
  metaspace) prefix on a vocab token is the tell that a leading space was (in)correctly added.
- **BOS: honor `add_bos_token`, and SentencePiece resolves BOS by PIECE STRING, not id.** Some
  families are BOS-sensitive — a missing/wrong leading BOS alone produces garbage. Two traps in the
  genai SPM path (`gguf_tokenizer.cpp`): (1) `add_bos` was hardcoded `false` — wire it from
  `tokenizer.ggml.add_bos_token` (default true for SPM). (2) Even with `add_bos=true`, SentencePiece
  prepends `PieceToId(TrainerSpec.bos_piece)` where `bos_piece` is a STRING defaulting to `"<s>"`; if
  the arch's real BOS piece differs (gemma example: BOS is `"<bos>"`=2, but the default `"<s>"`
  resolves to 203) the wrong token is prepended. Fix: write `TrainerSpec.bos_piece`/`eos_piece`/
  `unk_piece`/`pad_piece` (proto fields 46/47/45/48, STRINGS) set to the actual vocab pieces at the
  GGUF `*_token_id`s — setting the integer id fields (40-43) is not enough because `bos_id()`
  ignores them. Verify `genai.encode(text)[0] == <bos id>` after wiring.
- **Missing-weight assert at read_model** (`weights.cpp:31`, e.g. `blk.5.attn_v.weight`) usually
  means the arch's KV/shared-KV tensor layout differs from what the builder expects for that layer.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `native GGUF builder does not support architecture '<x>'` | add `<x>` to `supported_archs()` and set its feature flags / hparams |
| `Unsupported operation type` at read_model | the arch needs a ggml op with no translator — add `src/op/<op>.cpp` + register in `op_table.cpp` |
| FE load fails / `undefined symbol …` after adding a `src/*.cpp` | re-run `cmake .` before building (source glob is cached) |
| `diverge` FAILs with NaN rows | check `--swa` flag; re-test with real prompt tokens; suspect KV-cache precision (large head) or RoPE style |
| genai tokenizer: `Unsupported tokenizer model '<x>'` | add the `<x>` dispatch in `gguf_tokenizer.cpp` (+ factory op in openvino_tokenizers if needed) |
| `layerdiff` relative diff grows from blk.1+ but blk.0 is fine | a shape-sensitive op (usually RoPE `to_bhls`) handles blk.0's input layout but not later layers' — see the RoPE-rank gotcha; reshape-to-canonical before transposing |
| `genai.encode` first token ≠ `<bos>` (e.g. gemma) | wire `add_bos_token` + set `TrainerSpec.bos_piece` string in `build_spm_model_proto` (not the id field) — see the BOS gotcha |
| `layerdiff` PASSes but generation is wrong / repeats | not a prefill bug — run `diverge` for the stateful KV/SWA/RoPE-decode path, and check the tokenizer adds BOS |
| `full_bench.py` tokenize step fails | the genai gguf tokenizer can't read this arch yet — pass `--hf-tokenizer <hf_dir>` |
