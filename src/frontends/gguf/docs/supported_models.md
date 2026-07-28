# GGUF Frontend — Supported Models

This document lists the model architectures the GGUF frontend can convert and run
end-to-end. An architecture is listed as **Supported** only when at least one *real*
(non-synthetic) model of that architecture has been verified to load, convert, and
produce coherent output through the frontend.

Verification is done by running a real `.gguf` through the OpenVINO backend swap in
`llama.cpp` (`llama-completion`, CPU device, stateful execution) and confirming the
generated text is coherent and matches the pure-ggml CPU reference for the same prompt.

## Supported architectures

Each row was verified with the named real model.

| Architecture | Verified model | Notes |
|---|---|---|
| `llama`   | TinyLlama-1.1B-Chat v1.0 (Q4_K_M) | Dense; standard RoPE + GQA. |
| `qwen2`   | Qwen2.5-0.5B-Instruct (Q8_0)      | Dense. |
| `qwen3`   | Qwen3-0.6B (Q8_0)                 | Dense; QK-norm. |
| `qwen3moe`| Qwen3-0.9B-A0.6B (Q4_K_M), Qwen3-4B (Q4_K_M) | Mixture-of-experts (`mul_mat_id`). |
| `olmoe`   | OLMoE-1B-7B-0924-Instruct (Q4_0)  | Mixture-of-experts. |
| `gemma3`  | gemma-3 family                    | Mixed sliding-window / global RoPE. |
| `gemma4`  | gemma-4-E4B-it (Q4_K_M)           | Per-op RoPE (SWA vs global); f16 KV cache. |
| `qwen35`  | Qwen3.5-4B (Q4_K_M)               | Hybrid GatedDeltaNet + full-attention layers; partial-rotary IMROPE; interleaved Q/gate joint projection; f16 KV cache. |

`llama`, `qwen2`, `qwen3`, `olmoe`, `gemma4`, and `qwen35` were verified with a fresh
end-to-end run of the model named above. `qwen3moe` and `gemma3` were verified in earlier
development on the models named above.

Quantization formats verified in the above runs: `Q2_K`, `Q4_0`, `Q4_K_M`, `Q6_K`,
`Q8_0`. The frontend weight path also handles `Q4_1`, `Q5_0`, `Q5_1` and the F16/F32
paths; these are exercised by the unit tests but have not each been tied to a specific
end-to-end real-model run.

## How verification was performed

```sh
GGML_OPENVINO_DEVICE=CPU GGML_OPENVINO_STATEFUL_EXECUTION=1 \
  llama-completion -m <model>.gguf -p "The capital of France is" -n 12 -no-cnv --no-warmup
```

A run counts as verification only when the output is coherent (e.g. completes
"...is Paris") and consistent with the pure-ggml CPU backend on the same prompt. A model
that loads but emits garbage (e.g. `hunyuan`) is **not** counted as supported.

## Architectures accepted by the native `.gguf` builder

Everything above is about the **llama.cpp cgraph** path. This section covers the *other*
decoder — the native `.gguf` builder (`TransformerBuilder` in
[`src/builder/gguf_builder.cpp`](../src/builder/gguf_builder.cpp)), which is what
`core.read_model("model.gguf")` and OpenVINO GenAI use. The two paths share all op
translators but have separate architecture lists.

The builder's accept-list is the union of two sets, both defined at the bottom of
`gguf_builder.cpp`:

- **`verified_archs()`** — convert + compile + generation checked against a reference on a
  real checkpoint.
- **`experimental_archs()`** — expected to work via the builder's GGUF-tensor-table
  auto-detection, but not end-to-end verified. These convert and emit a one-time
  `OPENVINO_WARN` so callers know they are best-effort.

Anything not in either set is rejected with an explicit `OPENVINO_ASSERT` at load time
rather than converting into a silently wrong graph.

### `verified_archs()` — 13 architectures

| Architecture | Notes |
|---|---|
| `llama` | llama-2 / llama-3 |
| `qwen2` | qwen2 / qwen2.5 |
| `qwen3` | QK-norm |
| `phi3` | fused QKV |
| `minicpm` | NORMAL rope + scalar embedding/residual/logit scales |
| `hunyuan-dense` | |
| `olmoe` | OLMoE 1B-7B (MoE) |
| `qwen3moe` | Qwen3 MoE; same topology as `olmoe` |
| `gpt-oss` | MoE + attention sinks + SWA + OAI gated activation |
| `gemma` | Gemma 2B / 7B |
| `gemma2` | post-norms + attention soft-cap |
| `gemma3` | post-norms + final logit soft-cap |
| `gemma4` | SWA, per-layer embeddings, shared KV |

### `experimental_archs()` — 15 architectures

| Architecture | Notes |
|---|---|
| `llama-embed` | Bidirectional LLaMA (embedding model, no causal mask) |
| `exaone4` | EXAONE 4.0: NEOX rope, post-norms (attn + ffn) |
| `plamo3` | PLaMo-3: NEOX rope, post-norms (attn + ffn) |
| `smollm3` | SmolLM3: NORMAL rope + SWA |
| `hunyuan-moe` | NEOX rope, MoE routing, QK-norm |
| `glm4moe` | GLM 4.5 MoE: 1 dense lead layer, MoE + attn post-norm |
| `exaone-moe` | EXAONE MoE: SWA + MoE, shared expert |
| `minimax-m2` | Minimax M2: pure MoE |
| `ernie4_5-moe` | Ernie 4.5 MoE: NORMAL rope, dense lead layers + MoE stride |
| `bailingmoe2` | BailingMoe V2: MoE + shared expert + QK-norm |
| `maincoder` | Maincoder-1B: NORMAL rope, QK-norm (auto-detected) |
| `mistral3` | Ministral-3B: NORMAL rope, dense |
| `mellum` | JetBrains Mellum: pure MoE |
| `deepseek2-ocr` | DeepSeekOCR: dense lead layers + MoE |
| `jais2` | JAIS-2: dense (biases auto-detected) |

RoPE flavor is **not** in these tables because it is a separate switch: archs listed in
`arch_uses_neox_rope()` use NEOX (rotate-halves), everything else uses NORMAL (rotate
consecutive pairs). Adding an arch to the accept-list without also classifying its RoPE is
the most common way to get a model that loads and produces garbage.

### Measured status through OpenVINO GenAI

Every architecture above was run through GenAI on CPU (`gguf_arch_check`, greedy, SDPA
backend) on the checkpoint named below. "Generates" means the model answered *"The capital
of France is"* correctly and coherently; **`llama.cpp` ref** is the same `.gguf` through
`llama-cli` on the default ggml CPU backend, which distinguishes a frontend bug from a
model/checkpoint that is simply weak on the prompt.

| Arch | Set | Model used | GenAI | llama.cpp ref |
|---|---|---|---|---|
| `llama` | verified | Llama-3.2-1B-Instruct Q4_K_M | generates | generates |
| `qwen2` | verified | Qwen2.5-0.5B-Instruct Q4_K_M | generates | generates |
| `qwen3` | verified | Qwen3-0.6B Q8_0 | generates (reasoning preamble) | same |
| `phi3` | verified | Phi-3-mini-4k-instruct Q4 | generates | generates |
| `minicpm` | verified | MiniCPM-2B-dpo Q4_K_M | generates | generates |
| `hunyuan-dense` | verified | Hunyuan-0.5B-Instruct Q4_K_M | **degenerate** | generates |
| `olmoe` | verified | OLMoE-1B-7B-Instruct Q4_K_M | generates | generates |
| `qwen3moe` | verified | Qwen3-0.9B-A0.6B Q4_K_M | **degenerate** | generates |
| `gpt-oss` | verified | gpt-oss-20b MXFP4 | generates (harmony format) | same |
| `gemma` | verified | gemma-2b Q4_K_M | **degenerate** | degenerate too |
| `gemma2` | verified | gemma-2-2b-it Q4_K_M | **degenerate** | generates |
| `gemma3` | verified | gemma-3-1b-it Q4_K_M | generates | generates |
| `gemma4` | verified | gemma-4-E4B-it Q4_K_M | generates | generates |
| `llama-embed` | experimental | llama-nemotron-embed-1b-v2 Q4_K_M | repeats (embedding model) | degenerate too |
| `exaone4` | experimental | EXAONE-4.0-1.2B Q4_K_M | **degenerate** | generates |
| `plamo3` | experimental | plamo-3-nict-2b-base Q4_K_M | **degenerate** | degenerate too |
| `smollm3` | experimental | SmolLM3-3B Q4_K_M | generates (reasoning preamble) | same |
| `maincoder` | experimental | Maincoder-1B Q4_K_M | generates | generates |
| `mistral3` | experimental | Ministral-3-3B-Instruct-2512 Q4_K_M | generates | generates |
| `deepseek2-ocr` | experimental | deepseek-ocr-2 Q4_K_M | **degenerate** | generates |
| `ernie4_5-moe` | experimental | ERNIE-4.5-21B-A3B Q4_K_M | **degenerate** (blank) | generates |
| `bailingmoe2` | experimental | Ling-mini-2.0 Q2_K | generates | generates |
| `mellum` | experimental | Mellum2-12B-A2.5B-Instruct Q4_K_M | **degenerate** | generates |
| `hunyuan-moe` | experimental | — | not tested (no checkpoint) | — |
| `glm4moe` | experimental | — | not tested (smallest GLM-4.5-Air ≈ 40 GiB) | — |
| `exaone-moe` | experimental | — | not tested (smallest ≈ 9 GiB, 32B) | — |
| `minimax-m2` | experimental | — | not tested (smallest ≈ 78 GiB) | — |
| `jais2` | experimental | — | not tested (no checkpoint) | — |

Two caveats on reading this table. `llama-embed` is an *embedding* model, so degenerate
greedy completion is expected of it, not a defect. `gemma` (v1 base) and `plamo3` (base, not
instruct) are degenerate on the reference too, so those rows are checkpoint/prompt artifacts
rather than frontend bugs.

That leaves **7 architectures that generate correctly under llama.cpp but not through the
builder** — `hunyuan-dense`, `qwen3moe`, `gemma2`, `exaone4`, `deepseek2-ocr`,
`ernie4_5-moe` (blank output) and `mellum` — i.e. real conversion defects. Three of them
(`hunyuan-dense`, `qwen3moe`, `gemma2`) are in `verified_archs()`, so that set is currently
**optimistic** and should be re-validated before it is relied on.

### Measured performance and memory (OpenVINO GenAI, CPU)

Same runs as the table above. i9-12900K (16C/24T), OV defaults. Prefill = prompt tokens /
TTFT on a ~90-340-token prompt; decode = 1/TPOT over 32 greedy tokens, steady-state
iteration. `peak RSS` and `peak anon` are the maxima of `Rss:`/`Anonymous:` from
`/proc/self/smaps_rollup`, sampled every 20 ms in-process; `anon` is the part that
genuinely requires RAM (see [`frontend_design.md`](frontend_design.md) on the memory model).
`load` is `.gguf` → OV graph → `compile_model`.

| Arch | Model MiB | load s | prefill t/s | decode t/s | peak RSS MiB | peak anon MiB |
|---|---|---|---|---|---|---|
| `qwen2` | 468 | 4.7 | 1196.3 | 78.68 | 1492 | 1418 |
| `qwen3` | 609 | 4.9 | 865.1 | 64.94 | 1612 | 1538 |
| `hunyuan-dense` | 338 | 4.2 | 630.3 | 62.93 | 1545 | 1473 |
| `gemma3` | 768 | 1.9 | 662.6 | 43.76 | 2610 | 2536 |
| `qwen3moe` | 531 | 5.7 | 158.0 | 44.96 | 1987 | 1912 |
| `maincoder` | 640 | 5.1 | 324.3 | 36.28 | 2321 | 2245 |
| `llama-embed` | 770 | 6.0 | 266.9 | 35.75 | 2601 | 2529 |
| `exaone4` | 774 | 4.3 | 236.5 | 35.54 | 2594 | 2520 |
| `olmoe` | 4018 | 12.9 | 86.6 | 35.17 | 11760 | 11684 |
| `llama` | 770 | 6.0 | 323.8 | 35.08 | 2628 | 2556 |
| `bailingmoe2` | 5573 | 44.9 | 108.1 | 26.97 | 36956 | 36237 |
| `deepseek2-ocr` | 1859 | 7.4 | 299.2 | 71.38 | 5392 | 5318 |
| `mellum` | 7697 | 27.7 | 70.8 | 21.48 | 21282 | 21194 |
| `gemma` | 1425 | 3.8 | 149.6 | 18.47 | 4570 | 4501 |
| `plamo3` | 1574 | 3.3 | 154.9 | 16.87 | 5661 | 5594 |
| `smollm3` | 1826 | 8.4 | 121.9 | 14.88 | 5972 | 5895 |
| `minicpm` | 1649 | 3.3 | 126.0 | 14.82 | 5041 | 4968 |
| `ernie4_5-moe` | 12873 | 46.6 | 45.8 | 14.67 | 36551 | 36460 |
| `gemma2` | 1629 | 3.4 | 132.8 | 14.35 | 5692 | 5617 |
| `mistral3` | 2047 | 8.8 | 103.6 | 12.96 | 6871 | 6796 |
| `phi3` | 2282 | 4.4 | 108.3 | 11.84 | 8197 | 8130 |
| `gemma4` | 4746 | 9.6 | 63.6 | 9.24 | 12309 | 11953 |
| `gpt-oss` | 11548 | 89.3 | 18.7 | 5.48 | 123720 | 123493 |

Numbers from architectures marked degenerate above still describe real compute cost (the
graph runs, it is just numerically wrong), so they are kept for completeness.

One outlier remains: `gpt-oss` peaks at **124 GiB from an 11.5 GiB file (11x)**, versus a
typical 3-4x elsewhere. On a smaller-RAM host it would OOM. The cause is the compressed-weights
type gate on the MoE expert matmul described below — for gpt-oss the expert type is MXFP4
(`f4e2m1`), which the frontend dequantizes on-graph in `MUL_MAT_ID` rather than routing through
`GatherMatmul` at all, so the plugin-side widening does not reach it.

### MoE expert weights and the compressed-weights type gate

Worth knowing when picking a quantization for a MoE model, though the handling is entirely
plugin-side. MoE expert weights do not go through `FullyConnected`: `MUL_MAT_ID` lowers to the
CPU plugin's `GatherMatmul` (equally, to `GroupedMatMul` on the public-op side — on CPU
`ConvertGroupedMatMulToGatherMatmul` rewrites it into the same node *before* the compression
pass, so the two are indistinguishable here). That node accepts a **narrower set of compressed
weight types than `FullyConnected` does**:

| | accepted compressed weight types |
|---|---|
| `FullyConnected` | `u8, i8, u4, i4, nf4, f4e2m1, u2` |
| `GatherMatmul` / GPU grouped-matmul | `u8, i8, u4, i4` |

If an expert weight's element type is outside the second set, `ConvertGatherMatmulToGather
MatmulCompressed` does not fire, the `Convert -> Subtract -> Multiply` dequantization block stays
in the graph, and constant folding materializes the experts **in f32** — a 16x expansion off a
2-bit type, i.e. far more than the quantization was saving.

Q2_K is the case this affects: its weights map to `u2`. The CPU plugin's
`WidenGatherMatmulWeights` pass handles it by re-emitting *expert* weight constants as `u4`
(lossless — raw Q2_K values are `[0..3]`, which fit a nibble) at 2x the weight bytes, which is
much cheaper than falling off the compressed path. Dense `u2` weights are left alone. This is a
plugin-side workaround for a missing `u2` expert-matmul executor and needs nothing from the
frontend, which emits plain `u2` either way. Measured on Q2_K models, peak anonymous memory:

| Model | file MiB | before | after |
|---|---|---|---|
| Qwen3-0.9B-A0.6B (`qwen3moe`) | 373 | 4251 | 2071 |
| Ling-mini-2.0 (`bailingmoe2`) | 5573 | 117245 | 36237 |

Decode also improves (bailingmoe2: 12.7 → 27.0 t/s) because the experts are no longer read from
f32.

## Adding a new architecture

Support for a new architecture is a combination of:
1. **Ops** — every ggml op in the model's compute graph must have a frontend translator
   (`src/op/<name>.cpp`) and backend admission.
2. **Weights** — every quantization format used by the model's tensors must be handled by
   the weight path (`src/quant/weights.cpp`).
3. **Real-model verification** — run a real `.gguf` end-to-end as above before adding the
   architecture to the Supported table.

For the native builder specifically, see
[`adding_an_architecture.md`](adding_an_architecture.md) — for a same-family arch the change
is usually just adding the name to `experimental_archs()` plus the `arch_uses_neox_rope()`
classification, and promotion to `verified_archs()` should require the GenAI-vs-llama.cpp
comparison above.
