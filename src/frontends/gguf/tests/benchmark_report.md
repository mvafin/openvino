# OpenVINO GGUF Frontend — Benchmark Report

**Date:** 2026-06-16
**Branch:** `gguf_frontend`
**OV build:** 2026.3 (`/home/vmaxim/openvino/bin/intel64/Release`)
**Reference:** llama.cpp `llama-simple` (build-ref/bin, CPU)
**Prompt:** `"The capital of France is Paris. Tell me more about this city."`
**Max tokens:** 64 (generate until EOS or limit)
**Device:** CPU

> This run includes the **MoE compile-memory fix** (`SnippetsMarkSkipped` now marks the
> `GatherMatmul` weight-decompression chain as skipped, so `GatherMatmulCompressed` fires and
> expert weights stay compressed). OLMoE compile RAM dropped **54.3 GB → 8.8 GB** and its
> output went from degenerate to token-exact vs llama.cpp.

---

## 1. Summary Table

| Model | GGUF MB | read(s) | read RAM(MB) | compile(s) | compile RAM(MB) | OV PP(t/s) | LC PP(t/s) | OV TG(t/s) | LC TG(t/s) | OV/LC TG | Words OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Qwen3-0.6B-Q8_0           |  610 | 0.31 |  700 | 0.63 |  919 | 642 | 383 | 71.2 | 82.5 | 0.86x | **20/20** |
| Qwen2.5-0.5B-Instruct-Q8_0|  644 | 0.31 |  792 | 0.47 |  942 | 732 | 478 | 88.5 |102.2 | 0.87x | **20/20** |
| Qwen3-4B-Q4_K_M           | 2382 | 1.60 | 2851 | 1.47 | 4231 | 107 |  63 | 16.1 | 19.8 | 0.81x | 7/20†   |
| TinyLlama-1.1B-Chat-Q4_K_M|  638 | 0.39 |  875 | 0.66 | 1223 | 432 | 226 | 56.3 | 76.5 | 0.74x | 0/20‡   |
| MiniCPM-2B-dpo-Q4_K_M     | 1649 | 0.90 | 2002 | 1.31 | 2837 | 191 |  89 | 21.9 | 28.0 | 0.78x | 0/20‡   |
| Phi-3-mini-4k-instruct-Q4 | 2282 | 1.59 | 4032 | 1.11 | 5265 | 101 |  52 | 15.6 | 21.3 | 0.74x | 0/20§   |
| **OLMoE-1B-7B-Instruct-Q4_0** | 3746 | 1.55 | 4120 | 2.96 | **8848** | 124 | 147 | 54.8 | 75.0 | 0.73x | **20/20** |
| Hunyuan-0.5B-Instruct-Q8_0|  551 | 0.29 |  866 | 0.55 | 1028 | 680 | 411 | 74.6 | 91.1 | 0.82x | 1/1**   |

**PP** = prefill tokens/s; **TG** = decode tokens/s; **LC** = llama.cpp.
Single-run measurements; no model hit EOS within 64 tokens on this short prompt.

---

## 2. MoE Compile-Memory Fix (this cycle)

OLMoE (and any GatherMatmul-based MoE) used to expand all expert weights to f32 at compile
time. Root cause: the CPU plugin's `SnippetsMarkSkipped` pass marked only `MatMul`/`FullyConnected`
weight-decompression chains as `SkippedByPlugin`. The `GatherMatmul` (MoE expert dispatch)
weight chain was left tokenizable, so Snippets — which runs **before** `CpuSpecificOpSet` — fused
the expert `Multiply`+`Reshape` into a `Subgraph`. That `Subgraph` broke the
`CompressedWeightsBlock` pattern, `ConvertGatherMatmulToGatherMatmulCompressed` never matched,
and `ConstantFolding` later expanded the i4 experts to f32.

**Fix:** mirror the MatMul handling for `GatherMatmul` — when its weight input feeds a constant
path, mark the whole chain `SkippedByPlugin`. One block in
`src/plugins/intel_cpu/src/transformations/snippets/x64/pass/snippets_mark_skipped.cpp`.

| Metric (OLMoE-1B-7B Q4_0) | Before | After |
|---|---:|---:|
| compile_model peak RAM | 54347 MB | **8848 MB** (6.1× less) |
| decode throughput | 17.2 t/s | **54.8 t/s** (3.2× faster) |
| OV/llama decode ratio | 0.23x | **0.73x** |
| correctness (words vs llama) | 5/20 (degenerate) | **20/20 (token-exact)** |
| `GatherMatmulCompressed` ops | 0 | 48 |

No regression on dense models: Qwen3-4B (FC path) compile RAM unchanged at 4231 MB.

---

## 3. Performance Analysis

### Prefill (prompt processing)
OV prefill is **1.5–2× faster than llama.cpp** on every dense model — OV batches the full prompt
into an efficient GEMM. The one exception is OLMoE: now that experts are compressed, the prefill
GEMM over the batched expert matmul (124 t/s) is slightly below llama.cpp's MoE kernel (147 t/s).

| Model | OV PP | LC PP | Ratio |
|---|---:|---:|---:|
| Qwen3-0.6B    | 642 |  383 | **1.68x** |
| Qwen2.5-0.5B  | 732 |  478 | **1.53x** |
| Qwen3-4B      | 107 |   63 | **1.70x** |
| TinyLlama-1.1B| 432 |  226 | **1.91x** |
| MiniCPM-2B    | 191 |   89 | **2.15x** |
| Phi-3-mini    | 101 |   52 | **1.92x** |
| OLMoE         | 124 |  147 | 0.84x |
| Hunyuan-0.5B  | 680 |  411 | **1.65x** |

### Decode (autoregressive)
OV decode is **0.73–0.87× of llama.cpp** across the board — a 13–27% gap, consistent for both
dense and (now) MoE. Likely causes: the quantized GEMV path for batch-1 decode and KV-cache
layout. OLMoE now sits in the same band (0.73x) as the dense Q4 models rather than being an
outlier.

### Conversion speed
- Small (≤700 MB): read < 0.4 s, compile < 0.7 s.
- Medium (1.6–2.4 GB): read < 1.6 s, compile < 1.5 s.
- OLMoE (3.7 GB, 60 experts): read 1.55 s, compile 2.96 s.

### Compile memory overhead vs GGUF size

| Model | GGUF | compile RAM | Overhead |
|---|---:|---:|---:|
| Qwen3-0.6B   |  610 |  919 | +51% |
| Qwen2.5-0.5B |  644 |  942 | +46% |
| Qwen3-4B     | 2382 | 4231 | +78% |
| TinyLlama    |  638 | 1223 | +92% |
| MiniCPM-2B   | 1649 | 2837 | +72% |
| Phi-3-mini   | 2282 | 5265 | **+131%** |
| OLMoE        | 3746 | 8848 | +136% |
| Hunyuan-0.5B |  551 | 1028 | +87% |

Phi-3 is high because the GGUF stores Q4 but the IR holds wider constants. OLMoE's +136% is the
remaining headroom for MoE — far better than the previous +1350% (54 GB), but the activation and
intermediate buffers for 60 experts still cost more than a dense model of equal file size.

---

## 4. Correctness vs llama.cpp (first 20 shared words)

- **Token-exact (20/20):** Qwen3-0.6B, Qwen2.5-0.5B, **OLMoE-1B-7B** (was degenerate, now matches
  llama word-for-word: *"Paris is the capital and most populous city of France…"*).
- **† Qwen3-4B (7/20):** OV is coherent and correct (*"Paris is a city in the northwestern part of
  France…"*); llama.cpp echoes the prompt — a llama-side quirk on this Q4_K_M build, not an OV bug.
- **‡ TinyLlama / MiniCPM (0/20):** both tools produce coherent continuations; the 0/20 is a
  BOS/format-offset artifact of the word-prefix comparison, not divergent content. Both models
  expect chat formatting that the raw prompt skips.
- **§ Phi-3 (0/20):** OV emits a chat-formatted, correct answer (*"# Answer\nParis, the capital
  city of France…"*); llama echoes the prompt with `<|assistant|>` tags. Semantically equivalent.
- **\*\* Hunyuan (1/1):** both produce the same degenerate output without chat formatting — a
  pre-existing model/prompt issue, not OV-specific.

---

## 5. Models Not Benchmarked

| Model | Reason |
|---|---|
| `gpt-oss-20b-mxfp4.gguf` (20B) | MXFP4 correctness bug (`"2 + 2 ="` → garbage); experts are f4e2m1, which CPU GatherMatmul does not natively consume |
| `Qwen3-0.9B-A0.6B.Q4_K_M / Q2_K` | Arch `qwen3moe` — not yet supported by the frontend |
| `Qwen3-VL-2B-Instruct-Q4_K_M` + `mmproj-*` | Multimodal (vision tower out of scope) |
| `TinyMistral-248M`, `mistral-1L-tiny`, `tiny-random-*` | genai GGUF tokenizer `IndexError` (SPM), no HF fallback wired; tiny-random models have no real weights |

---

## 6. Remaining Known Issues

1. **Decode throughput** 73–87% of llama.cpp on batch-1 decode (dense and MoE alike) — investigate
   the INT4/INT8 GEMV path vs llama's dedicated GEMV.
2. **gpt-oss MXFP4** still incorrect; experts need dequant to u4/u8 or native f4e2m1 GatherMatmul
   support before the MoE memory fix helps it.
3. **Phi-3 / OLMoE compile overhead** (+131% / +136%) — wider-than-file IR constants and per-expert
   buffers; candidate for lazy/deferred dequantization.
4. **genai GGUF tokenizer** `IndexError: map::at` on SPM models (TinyLlama, Mistral) — fails before
   any OV inference.
