# GGUF Frontend — Performance and Memory Notes

Extracted from `supported_models.md` — internal measurement notes, not yet ready to publish as
user-facing guidance. Kept here for future refinement; merge back into `supported_models.md` (or a
successor doc) once the numbers and methodology are considered stable enough to advertise.

### Measured performance and memory (OpenVINO GenAI, CPU)

Same runs (checkpoints, quantizations) as `supported_models.md`'s architecture-verification
table. i9-12900K (16C/24T), OV defaults. Prefill = prompt tokens /
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
| `qwen35` (Qwen3.5-0.8B Q8_0) | 795 | 1.5 | 537.4 | 41.38 | 2292 | 2207 |
| `qwen35` (Bonsai-27B Q2_g64) | 7234 | 28.7 | 21.4 | 3.75 | 24009 | 23903 |
| `muse-glimmer` | 15512 | 28.3 | 28.9 | 2.66 | 37457 | 37357 |

Numbers from architectures marked degenerate above still describe real compute cost (the
graph runs, it is just numerically wrong), so they are kept for completeness.

Notable outliers: Bonsai (`qwen35`, Q2_g64) decodes **6.5x faster** than llama.cpp (3.75 vs
0.58 tok/s) — not an OpenVINO win, but an upstream gap: ggml ships no x86 SIMD kernel for its
2-bit format, so it falls back to a scalar reference, while OpenVINO's compressed-weight matmul
kernel is already optimized for it; the trade-off is memory (llama.cpp mmaps and peaks at
7.5 GiB, the frontend materializes decompression constants and peaks at 22.7 GiB). `gpt-oss` is
the opposite kind of outlier: it peaks at **124 GiB from an 11.5 GiB file (11x)**, versus a
typical 3-4x elsewhere, because its MoE expert quantization type isn't one the plugin's
compressed-matmul path recognizes yet, so it gets dequantized eagerly instead of staying
compressed — see "MoE expert weights and quantization choice" below for the general issue. On a
smaller-RAM host this would OOM.

### Measuring performance correctly

Three traps have each produced a wrong published number at least once. Read this before
benchmarking, especially when comparing against llama.cpp.

**1. Disable prefix caching when measuring prefill under PagedAttention.** `ATTENTION_BACKEND=PA`
routes through GenAI's ContinuousBatching adapter, and `get_latency_oriented_scheduler_config()`
(GenAI `src/cpp/src/utils.cpp`) sets `enable_prefix_caching = true` by default. Benchmarks
typically repeat one fixed prompt for N iterations to amortize the first-request dynamic-shape
compile — with prefix caching on, **every iteration after the first is a cache hit**, so the
reported TTFT is not prefill work at all. On Llama-3.2-1B this reads 125 ms cached vs 300 ms
uncached (SDPA measures 304 ms): the cache made PA look 2.4x faster at prefill than an identical
computation. Pass an explicit scheduler config with it off:

```cpp
ov::genai::SchedulerConfig sched;
sched.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();  // as the latency default
sched.enable_prefix_caching = false;
props[ov::genai::scheduler_config.name()] = sched;
```

Note the asymmetry that makes this specifically a *comparison* hazard: SDPA ignores this knob
entirely, and neither llama.cpp reference path caches across runs. `llama-bench` calls
`llama_memory_clear()` inside the rep loop before the timer starts (its state-reuse path is gated on
`-d/--n-depth > 0`, which defaults to 0); `llama-cli` is single-shot and its `--prompt-cache`
defaults to empty. So a cached PA number is being compared against two uncached ones. Sanity check:
run llama-bench with `-r 5` and confirm variance stays under ~1% — a cache hit shows as a large drop
after rep 0, not as noise. Prefix caching is a real PA capability worth reporting *separately*; it
just is not prefill throughput.

**2. Confirm PA is actually in use — the fallback is silent.** GenAI catches a PA initialization
failure and falls back to SDPA with only a `GENAI_WARN` (`src/cpp/src/llm/pipeline.cpp`), and the
default log level is `ERR` (`src/cpp/src/logger.cpp`), so **the warning is invisible unless you set
`OPENVINO_LOG_LEVEL=4`**. Correct output and plausible timings therefore prove nothing. Counting
`PagedAttention` in the `ov::Model` is also insufficient — that is the graph handed *to* the plugin.
Check the compiled **runtime** graph:

```cpp
auto rt = compiled_model.get_runtime_model();
for (const auto& op : rt->get_ops())
    hist[op->get_rt_info().at("layerType").as<std::string>()]++;
```

For Llama-3.2-1B (16 layers) the two backends must look like this — note `MemoryInput`/`MemoryOutput`
disappearing, since PA replaces the stateful KV cache with the plugin's block-table cache. A rename
alone would not do that:

| runtime `layerType` | SDPA | PA |
|---|---|---|
| `PagedAttention` | 0 | 16 |
| `ScaledDotProductAttention` | 16 | 0 |
| `MemoryInput` / `MemoryOutput` | 32 / 32 | 0 / 0 |

**3. Drop iteration 0 and pin the comparison.** Iteration 0 carries the first-request dynamic-shape
compile (several hundred ms to seconds); average iterations 1..N-1 for steady state. Compare on the
same `.gguf` file, the same prompt text, and the same `n_ctx` — llama.cpp preallocates the whole
`n_ctx` KV cache up front while OV's stateful cache grows on demand, so a mismatched context length
makes the memory figures incomparable. Also record the thread counts: llama.cpp auto-selects
P-cores only (8 on an i9-12900K) where OV uses all 24 by default, which is not a like-for-like
core budget unless equalized.

Putting it together — the three commands behind the table below. llama.cpp is measured twice because
`llama-bench` gives steady-state kernel throughput with no process/load overhead, while `llama-cli`
walks the same end-to-end path as the GenAI sample and so is the fair peak-RSS comparison:

```sh
# steady-state kernel throughput (cache cleared per rep; -r 5 to confirm low variance)
llama-bench -m "$MODEL" -p 128 -n 128 -r 5

# end-to-end, for max-RSS parity with the GenAI sample
/usr/bin/time -v llama-cli -m "$MODEL" -p "$PROMPT" -n 128 -c 1024 \
    -no-cnv -st --temp 0 --seed 1 --no-warmup --ignore-eos

# GenAI, once per backend; the sample turns prefix caching off for PA (trap 1) and
# reports per-iteration TTFT/TPOT so iteration 0 can be dropped (trap 3)
/usr/bin/time -v bench_gguf_perf "$MODEL" "$PROMPT" 128 4 {SDPA|PA}
```

`bench_gguf_perf` is the GenAI sample at `samples/cpp/text_generation/bench_gguf_perf.cpp`; keep the
same `-c/n_ctx` on both sides and the same prompt text everywhere.

#### PagedAttention vs SDPA vs llama.cpp (measured under the rules above)

i9-12900K, Q4_K_M, 128 generated tokens, 4 iterations with iteration 0 dropped, `n_ctx=1024`,
prefix caching **off**, PA presence confirmed in the runtime graph for every row. llama.cpp is its
default ggml CPU backend (`llama-bench pp128/tg128`), 8 threads by its own auto-selection.

| Model | prompt tok | prefill t/s (lcpp / SDPA / PA) | decode t/s (lcpp / SDPA / PA) | PA/SDPA | PA/lcpp | peak RSS GB (lcpp / SDPA / PA) |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 87 | 528 / 291 / 291 | 73.7 / 39.7 / 41.3 | 1.04 | 0.56 | 1.30 / 2.51 / 2.51 |
| Maincoder-1B | 68 | 591 / 405 / 409 | 84.1 / 43.9 / 45.7 | 1.04 | 0.54 | 1.10 / 2.24 / 2.24 |
| gemma-3-1b | 75 | 411 / 467 / 429 | 75.5 / 47.1 / 47.1 | 1.00 | 0.62 | 0.92 / 2.49 / 2.49 |
| Ministral-3-3B | 631 | 158 / 111 / 120 | 27.1 / 14.8 / 15.0 | 1.01 | 0.55 | 3.60 / 6.67 / 6.65 |
| SmolLM3-3B | 302 | 173 / 135 / 136 | 30.2 / 16.8 / 17.2 | 1.02 | 0.57 | 3.21 / 5.71 / 5.70 |
| mistral-7b-v0.1 | 55 | 68 / 52 / 56 | 13.7 / 6.94 / 6.99 | 1.01 | 0.51 | 7.37 / 11.81 / 11.82 |
| Ministral-8B | 53 | 69 / 40 / 39 | 12.8 / 7.01 / 7.05 | 1.01 | 0.55 | 7.92 / 13.10 / 13.11 |
| gemma-4-E4B | 58 | 111 / 55 / 56 | 18.2 / 11.1 / 11.6 | 1.04 | 0.64 | 6.96 / 12.36 / 11.92 |

**PA vs SDPA: parity.** Decode 1.00-1.04x (PA marginally ahead on all 8), prefill within +-8%, peak
RSS within 0.2% except gemma-4 where PA is 0.44 GB lower. Enabling PA costs nothing; the reason to
use it is that continuous batching, prefix caching and multi-sequence serving become available at
all, which the SDPA-only graph could not do.

**PA vs llama.cpp: decode 0.51-0.64x**, prefill 0.55-1.04x, peak RSS 1.7-2.4x. These ratios match
what the SDPA path already measured, so PA neither introduces nor closes that gap — see
[`frontend_design.md`](frontend_design.md) on the memory model for the RSS side.

### MoE expert weights and quantization choice (CPU plugin specific)

Worth knowing when picking a quantization for a MoE model — this is entirely plugin-side
behavior; the frontend itself emits the same compressed weight representation regardless of
target device. On the CPU plugin, not every compressed weight type is recognized by the
grouped/expert matmul kernel MoE routing lowers to. When an expert weight's type falls outside
that supported set, the plugin dequantizes it eagerly at compile time instead of keeping it
compressed — a large, avoidable memory expansion (up to 16x for a 2-bit type).

Q2_K is the case that hits this today. The CPU plugin works around it by re-packing the affected
expert weights into a supported 4-bit type (lossless — raw Q2_K values are `[0..3]`, which fit a
nibble) at 2x the weight bytes, which is still far cheaper than losing compression entirely; dense
`u2` weights are unaffected. Measured on Q2_K models on CPU, peak anonymous memory:

| Model | file MiB | before | after |
|---|---|---|---|
| Qwen3-0.9B-A0.6B (`qwen3moe`) | 373 | 4251 | 2071 |
| Ling-mini-2.0 (`bailingmoe2`) | 5573 | 117245 | 36237 |

Decode also improves (bailingmoe2: 12.7 → 27.0 t/s) because the experts are no longer read from
f32.

