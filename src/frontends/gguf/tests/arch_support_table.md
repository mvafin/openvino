# OpenVINO GGUF Frontend — Supported Architectures: Inference Results

**Date:** 2026-06-25
**Build:** worktree `gguf_frontend-work` (OV 2026.3), CPU device
**Reference:** llama.cpp `llama-simple` / `llama-cli` (build-ref, CPU)
**Prompt:** `"The capital of France is Paris. Tell me more about this city."` (perf/bench) and
`"The capital of France is"` (accuracy spot-check, greedy, no chat template)
**Generation:** greedy, 32 tokens (perf) / 24 tokens (accuracy)
**Memory:** `read_model` and `compile_model` peak process RSS (MB); GGUF size in MB.
**Perf:** PP = prefill tok/s, TG = decode tok/s, Ratio = OV TG / llama.cpp TG.

> Accuracy column reflects whether OV output matches llama.cpp / is coherent. Base (non-instruct)
> models legitimately ramble on a raw prompt — only **degenerate loops / wrong tokens** are flagged
> as issues. Issues are listed but NOT investigated here.

## Architectures WITH a local model (measured)

| # | Arch | Model (quant) | GGUF MB | read s / RAM MB | compile s / RAM MB | PP t/s | TG t/s | llama TG | Ratio | Accuracy | Issue |
|---|------|---------------|--------:|----------------:|-------------------:|-------:|-------:|---------:|------:|----------|-------|
| 1 | `llama` | TinyLlama-1.1B-Chat Q4_K_M | 638 | 0.4 / 827 | 0.6 / 1387 | 183 | 45.0 | 77.5 | 0.58 | ✅ matches llama | — |
| 2 | `qwen2` | Qwen2.5-0.5B-Instruct Q8_0 | 644 | 0.3 / 784 | 0.5 / 936 | 682 | 78.2 | 81.4 | 0.96 | ✅ 20/20 words | — |
| 3 | `qwen3` | Qwen3-0.6B Q8_0 | 610 | 0.3 / 762 | 1.1 / 944 | 170 | 18.2 | 51.4 | 0.35 | ✅ 20/20 words | ⚠ TG ratio 0.35 (slow decode) |
| 4 | `qwen3moe` | Qwen3-0.9B-A0.6B Q4_K_M | 531 | 4.8 / 724 | 2.3 / 1518 | 60 | 30.2 | 76.6 | 0.39 | ⚠ 4/20, repetition ("the the capital") | ❗ degenerate-ish output + slow |
| 5 | `phi3` | Phi-3-mini-4k-instruct Q4 | 2282 | 5.2 / 3997 | 1.3 / 5805 | 46 | 11.9 | 19.3 | 0.61 | ✅ coherent (matches after prompt prefix) | — |
| 6 | `minicpm` | MiniCPM-2B-dpo Q4_K_M | 1649 | 1.2 / 2028 | 1.6 / 3308 | 92 | 15.5 | 26.2 | 0.59 | ✅ coherent | — |
| 7 | `hunyuan-dense` | Hunyuan-0.5B-Instruct Q8_0 | 551 | 0.6 / 871 | 0.9 / 936 | 182 | 51.5 | 47.4 | 1.09 | ❗ degenerate (`""""…` / `> > >`) | ❗ wrong output |
| 8 | `olmoe` | OLMoE-1B-7B-Instruct Q4_0 | 3746 | 5.0 / 4081 | 4.3 / 8800 | 105 | 50.6 | 65.9 | 0.77 | ✅ 20/20 words | — |
| 9 | `gpt-oss` | gpt-oss-20B MXFP4 | 11549 | 9.1 / 12950 | 47.1 / **119533** | 0.9 | 2.5 | 18.9 | 0.13 | ✅ 20/20 words | ❗ compile RAM ~120 GB, PP 0.9 t/s, ratio 0.13 |
| 10 | `gemma` | Gemma-2B (base) Q4_K_M | 1425 | — | — | — | — | — | — | ❗ "Paris" then `fte fte…` loop | ❗ degenerate after first tokens |
| 11 | `gemma3` | Gemma-3-1B-it Q4_K_M | 768 | — | — | — | — | — | — | ✅ "Paris." (fixed this session) | — |
| 12 | `gemma4` | Gemma-4-E4B-it Q4_K_M | 4747 | 3.3 / 6238 | 2.6 / 9601 | 41 | 9.9 | 11.3 | 0.87 | ✅ coherent ("…**Paris**…") | — |
| 13 | `llama-embed` | llama-nemotron-embed-1B v2 Q4_K_M | 770 | 0.9 / 1149 | 0.7 / 1826 | 149 | 37.5 | 82.3 | 0.46 | ❗ degenerate (`capitals capitals…`); llama ref also `!!!!` | ⚠ embedding model — not a causal LM; compare with care |
| 14 | `exaone4` | EXAONE-4.0-1.2B Q4_K_M | 775 | 0.7 / 1163 | 0.9 / 1836 | 139 | 36.8 | n/a | — | ❗ degenerate (`then then…`, `toto…`) | ❗ wrong output; no llama ref captured |
| 15 | `plamo3` | PLaMo-3-NICT-2B-base Q4_K_M | 1574 | — | — | — | — | — | — | ❗ degenerate (`Result p smack…`) | ❗ wrong output |
| 16 | `smollm3` | SmolLM3-3B Q4_K_M | 1827 | 1.3 / 2425 | 1.2 / 3971 | 56 | 17.1 | 26.1 | 0.66 | ⚠ 2/20 — coherent but diverges from llama | ⚠ partial divergence (verify) |
| 17 | `ernie4_5-moe` | ERNIE-4.5-21B-A3B Q4_K_M | 12873 | — | — | — | — | — | — | ❗ blank/whitespace output | ❗ wrong output |
| 18 | `maincoder` | Maincoder-1B Q4_K_M | 641 | 0.5 / 1054 | 0.9 / 1625 | 115 | 46.0 | 72.5 | 0.64 | ✅ 20/20 words | — |
| 19 | `mistral3` | Ministral-3-3B-Instruct Q4_K_M | 2047 | 1.4 / 2730 | 1.0 / 4468 | 57 | (EOS@1) | n/a | — | ✅ "…**Paris**." | — |
| 20 | `deepseek2-ocr` | DeepSeek-OCR-2 Q4_K_M | 1860 | 0.9 / 2511 | 1.2 / 5232 | 126 | (EOS@1) | n/a | — | ❗ degenerate (`is is is…`) | ❗ wrong output |

Notes:
- Rows 10/11/15/17 (`gemma` base, `gemma3`, `plamo3`, `ernie4_5-moe`) have no perf numbers because
  `full_bench.py`'s tokenizer-subprocess helper (`gguf_tokenize.py`, uses `tok.get_vocab()`) fails on
  their tokenizers — a **harness limitation, not a conversion failure**; they `read_model` +
  `compile_model` fine and were accuracy-checked via `LLMPipeline` directly.
- Rows 19/20 hit EOS at the first generated token under the bench prompt (so TG is n/a); accuracy
  was re-checked with the plain prompt.
- "✅ matches" with `0/N words` in the raw bench is a comparison artifact: llama.cpp's logged output
  includes the prompt prefix while OV's does not — the continuations are identical.

## Architectures WITHOUT a local model (not measured)

These are in `supported_archs()` but no GGUF is available locally to benchmark:

| Arch | Notes |
|------|-------|
| `gemma2` | Gemma-2 (post-norms + attn soft-cap); only a 37-byte stub present |
| `hunyuan-moe` | Hunyuan MoE |
| `glm4moe` | GLM-4.5 MoE |
| `exaone-moe` | EXAONE MoE |
| `minimax-m2` | Minimax M2 MoE |
| `bailingmoe2` | BailingMoe V2 |
| `mellum` | JetBrains Mellum MoE |
| `jais2` | JAIS-2 |

## Issues summary (not investigated)

- **Degenerate / wrong output:** `hunyuan-dense`, `exaone4`, `plamo3`, `ernie4_5-moe`,
  `deepseek2-ocr`, `gemma` (base, loops after first tokens), `qwen3moe` (repetition).
- **Partial divergence from llama.cpp:** `smollm3` (coherent but different continuation).
- **Performance / memory:** `gpt-oss` compile peak ~120 GB RAM, prefill 0.9 t/s, decode ratio 0.13;
  `qwen3` decode ratio 0.35; several models decode ratio 0.4–0.6.
- **Embedding model:** `llama-embed` is bidirectional/embedding, not a causal LM — degenerate
  greedy text is expected; needs an embedding-appropriate check.
- **No local model to validate:** `gemma2`, `hunyuan-moe`, `glm4moe`, `exaone-moe`, `minimax-m2`,
  `bailingmoe2`, `mellum`, `jais2`.
