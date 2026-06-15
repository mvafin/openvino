# Proposal: A GGUF Frontend for OpenVINO with Two Input Paths

**Status:** Draft for discussion
**Author:** (fill in)
**Date:** 2026-06-05

## 1. Summary

Add a first-class **GGUF frontend** to OpenVINO (`src/frontends/gguf/`) that converts
GGML/GGUF models into an `ov::Model`. The frontend accepts **two kinds of input**
behind a single, shared op-translation engine:

1. **A `ggml_cgraph`** produced at runtime by **llama.cpp** (its existing OpenVINO
   backend feeds the compute graph in). This is the *quick-enablement* path — it
   reaches every architecture llama.cpp supports **whose ops are already
   translated** by the frontend. It is **not guaranteed zero-day**: an architecture
   that introduces a new ggml op still needs a translator added first (a one-time
   cost, shared by both paths).
2. **A `.gguf` file**, parsed by an **OpenVINO-native graph builder** that
   reconstructs the model's op graph (per architecture, by hand) and hands it to
   the *same* translators. This is the *production* path — OpenVINO owns weight
   memory (mmap + `ov::Constant` ownership transfer), the resulting `ov::Model` is
   self-contained and serializable to IR, and it ships in the default wheel with no
   heavy third-party dependency.

The unifying idea: both inputs are exposed through one abstract interface
(`ov::frontend::ggml::GgmlDecoder`) and converted by one set of op translators.
"Two inputs, one conversion code path."

Consumers:

- **llama.cpp** keeps a *decoder implementation* over its `ggml_cgraph` and links
  the frontend back from OpenVINO (deletes its vendored copy of the translators).
- **OpenVINO** gains native `GGUF → ov::Model` and `GGUF → IR` conversion.
- **OpenVINO GenAI** gets **both** paths, selectable: the OpenVINO native builder
  for production-ready models with optimal memory, and the llama.cpp path (opt-in
  dependency) for quick enablement of models not yet hand-supported natively.

## 2. Goals and non-goals

### Goals

- One frontend, reused in three places (llama.cpp, OpenVINO, GenAI).
- A single, tested op-translation engine shared by both input paths.
- Native `GGUF → ov::Model`/IR in OpenVINO with **self-contained, mmap-owned
  weights** (the `ov::Constant`-owns-the-mmap optimization the IR/PDPD/TF
  frontends already use).
- No mandatory heavy dependency in core OpenVINO / the default wheel.
- llama.cpp can consume the translators back, eliminating its vendored copy.

### Non-goals

- Replacing llama.cpp's runtime/compute. We translate graphs; we do not reimplement
  inference.
- Bundling the full llama.cpp / libllama into the default OpenVINO wheel.
- Day-one parity of the native builder with llama.cpp's architecture count. The
  native builder covers a curated set; the llama.cpp path covers the long tail.

## 3. Background (current state)

The investigation behind this proposal established the following facts.

- **llama.cpp already contains an OpenVINO frontend**, written in the
  `ov::frontend::ggml` namespace and clearly authored to be upstreamed
  (`Copyright (C) Intel Corporation / SPDX Apache-2.0` headers). It lives in
  `llama.cpp/ggml/src/ggml-openvino/openvino/` and is already split along the
  exact seam this proposal needs:
  - `decoder.h` — `class GgmlDecoder : public ov::frontend::DecoderBase`, a **pure
    abstract interface**.
  - `frontend.{h,cpp}`, `input_model.{h,cpp}`, `node_context.h`,
    `translate_session.cpp`, `op_table.cpp`, `op/*.cpp`, `pass/*`, `rt_info/*`.
  - The `openvino/` subdir (~2,400 LoC) depends on `openvino::core` **only** — it
    has effectively zero `ggml` coupling (one inlinable rope-yarn helper).
  - All op translators emit **standard opset (v0–v13)** ops today — fully
    serializable.
- **The ggml-specific code lives on the other side of the interface**:
  `ggml-decoder.{h,cpp}` (`GgmlOvDecoder : public GgmlDecoder`, ~1,280 LoC) wraps a
  real `ggml_cgraph`, and `ggml-openvino.cpp` is the ggml-backend glue that feeds
  the cgraph in at `graph_compute(cgraph)`.
- **The "graph builder" (gguf → cgraph) is inseparable from libllama.** A model
  builder (e.g. `src/models/llama.cpp`) is a method on `llama_model`, uses
  `llama_model_loader`, `hparams`, and `llm_graph_context::build_attn`, which is
  wired to KV-cache/memory state. The cgraph is only ever produced through
  `llama_context`. There are **131 files in `llama.cpp/src/models/`** — that is the
  breadth the cgraph path buys, and the dependency that breadth costs.
- **OpenVINO GenAI already has a from-scratch GGUF graph builder** in
  `openvino.genai/src/cpp/src/gguf_utils/` (~2,100 LoC: `gguf.cpp` parsing,
  `gguf_quants.cpp` dequant, `building_blocks.cpp` transformer ops,
  `gguf_modeling.cpp` arch dispatch). It builds an `ov::Model` directly, with **no
  ggml dependency**, and is wired into `LLMPipeline` via `read_model`. It covers
  **3 architectures** (llama, qwen2, qwen3); the per-architecture delta is tiny
  (~3 branches) because they share one transformer template.
- **Weight-memory ownership differs between the paths.** OpenVINO's native flow
  (IR frontend) wraps the mmap as a `SharedBuffer<MappedMemory>` (an
  `AlignedBuffer`) and builds Constants with the **ownership-taking** ctor
  (`Constant(type, shape, data, shared_ptr<void>)` /
  `Constant(type, shape, shared_ptr<AlignedBuffer>)`), so the `ov::Model`
  co-owns the mmap and is self-contained. llama.cpp's `process_weight_tensor` uses
  the **non-owning** `ov::Tensor(type, shape, raw_ptr)` ctor — byte-level zero-copy,
  but the model is only a *view* and stays valid only while the libllama instance
  lives. Quantized weights are copied/repacked on **both** paths (ggml block layout
  ≠ OpenVINO `u4 + scale + zp` layout) — unavoidable on any GGUF path.

This proposal turns those facts into a concrete plan.

## 4. Architecture

### 4.1 The shared contract

The cornerstone already exists: the abstract `GgmlDecoder` interface
(`DecoderBase`). Everything above it (op table, translators, passes,
`TranslateSession`) is input-agnostic. Everything below it is input-specific.

```
                ┌──────────────────────────────────────────────┐
                │  src/frontends/gguf/  (OpenVINO, core-only)    │
                │                                                │
   input A      │   FrontEnd / InputModel / NodeContext          │
 ggml_cgraph ─┐ │   TranslateSession                             │
              │ │   op_table.cpp + op/*.cpp   (THE shared        │
              ├─┼─►  pass/* , rt_info/*        translation engine)│
              │ │                ▲                                │
   input B    │ │                │ programs against              │
  .gguf file ─┘ │      ┌─────────┴─────────┐                      │
                │      │  GgmlDecoder       │  (abstract)         │
                │      └─────────┬─────────┘                      │
                └────────────────┼──────────────────────────────┘
                                 │ implemented by
              ┌──────────────────┴───────────────────┐
              │                                       │
   ┌──────────▼────────────┐          ┌───────────────▼─────────────────┐
   │ CgraphDecoder         │          │ GgufBuilderDecoder              │
   │ (lives in llama.cpp)  │          │ (lives in OpenVINO)             │
   │ wraps ggml_cgraph     │          │ walks an OV-built op graph      │
   │ weights = views       │          │ weights = mmap-owned Constants  │
   └───────────────────────┘          └─────────────────┬───────────────┘
                                                         │ built by
                                          ┌──────────────▼───────────────┐
                                          │ OpenVINO native graph builder │
                                          │ gguf parse + per-arch op graph│
                                          │ + gguf_quants (from GenAI)    │
                                          └───────────────────────────────┘
```

Key consequence: **all the expensive, correctness-sensitive logic is written once**
— op translation (RoPE, RMSNorm, SDPA fusion, GLU variants, quant-aware matmul),
the stateful/KV transforms, weightless caching, and the pass pipeline. The two
input paths converge *before* translation.

### 4.2 Two decoder implementations

- **`CgraphDecoder`** (llama.cpp side, = today's `GgmlOvDecoder`): walks a real
  `ggml_cgraph`, reads `ggml_tensor->op`, `src[]`, `op_params`, shapes/strides.
  Builds weight Constants as **non-owning views** over ggml-owned mmap memory
  (correct for a runtime that keeps the model resident).

- **`GgufBuilderDecoder`** (OpenVINO side, new): walks an in-memory op graph that
  OpenVINO's native builder produced from a `.gguf` file. Builds weight Constants
  with **ownership transfer** over OpenVINO's own mmap of the `.gguf`
  (`AlignedBuffer`/`SharedBuffer<MappedMemory>` + owning `Constant` ctor) →
  self-contained `ov::Model`, serializable to IR, weights paged from disk on demand.

Because the weight-ownership policy is entirely inside the decoder, the difference
"production memory vs. runtime view" is naturally encapsulated — the translators
never need to know which path they are on.

### 4.3 The OpenVINO native graph builder and its intermediate representation

The native path needs to answer the same questions the `GgmlDecoder` interface
asks: "list of nodes; for each, its op type, inputs, params, output shape/type;
the model inputs/outputs; the weights." So OpenVINO's builder produces a
**lightweight, ggml-independent op-graph** (call it `GgmlGraph`): a vector of nodes
each carrying a GGML-op tag (mirroring `GGML_OP_*` as an OpenVINO-local enum/string,
**not** including any ggml header), input references, `op_params`, and output
metadata, plus a weight table. `GgufBuilderDecoder` is a thin adapter over
`GgmlGraph`.

Crucially, the builder emits the model **in the ggml-op vocabulary** (ADD, MUL_MAT,
ROPE, RMS_NORM, SOFT_MAX, FLASH_ATTN_EXT, …) and lets the **shared translators**
lower those to optimized OpenVINO subgraphs — rather than emitting `ov` ops
directly the way GenAI's `building_blocks.cpp` does today. This is what makes "same
conversion code" literally true: a `MUL_MAT` from llama.cpp's cgraph and a
`MUL_MAT` from the OpenVINO builder go through the identical translator.

Reused as-is from GenAI:

- `gguf.cpp` — GGUF container parsing, metadata, tensor table (≈580 LoC, shared).
- `gguf_quants.cpp` — Q4_0/Q4_1/Q4_K/Q6_K/Q8_0 dequant/repack to OpenVINO layout
  (≈270 LoC, shared).

Re-targeted from GenAI:

- `building_blocks.cpp` / `gguf_modeling.cpp` — today they assemble `ov` ops.
  Under this proposal the per-architecture builder assembles `GgmlGraph` nodes
  instead. The architecture knowledge (layer loop, attention/FFN/norm wiring, RoPE
  config, per-head norms) is preserved; only the emitted vocabulary changes.

> Design alternative considered: keep GenAI's builder emitting `ov` ops directly and
> bypass the translators for the native path. Rejected as the primary design because
> it duplicates op-emission logic and defeats the "one conversion code path" goal.
> However, see §11 — the direct-`ov` builder is a valid *interim* milestone.

### 4.4 Weight memory and the ownership gap

This is the central reason the native path exists. With `GgufBuilderDecoder`:

- OpenVINO mmaps the `.gguf` once (`PROT_READ`), wrapped as
  `SharedBuffer<MappedMemory>`.
- **F16/F32/BF16** weights → `Constant(type, shape, ptr, shared_ptr<...>)` (owning
  ctor). The Constant co-owns the mapping; the `ov::Model` is self-contained and
  the file pages in lazily. No copy.
- **Quantized** weights → dequantized/repacked into OpenVINO layout via
  `gguf_quants`. A copy here is unavoidable (layout mismatch) and is identical to
  what GenAI and llama.cpp already pay.
- The model serializes to IR cleanly; reload-from-IR round-trips.

The `CgraphDecoder` path deliberately keeps today's non-owning behavior — that is
correct for the llama.cpp runtime and for GenAI's "quick enable, model stays
resident" use. The trade is explicit and documented: **the cgraph path does not
yield a self-contained model and is not the recommended path for `GGUF → IR`
conversion**.

### 4.5 Internal / non-serializable ops policy

Default: translators emit **standard opset ops** (status quo) → serializable IR.
For ops without a standard-opset lowering, use a `FrameworkNode`-style fallback
(as the PyTorch frontend does) so unknown ops degrade gracefully instead of failing
the whole convert. Internal/non-serializable ops are permitted only as a documented
exception, decomposed before compile; emitting them breaks the `GGUF → IR`
round-trip, so the native (production) path should avoid them.

## 5. Repository layout

```
openvino/
  src/frontends/gguf/
    include/openvino/frontend/ggml/
      frontend.hpp, decoder.hpp, visibility.hpp
    src/
      frontend.cpp, input_model.cpp, node_context.{hpp,cpp}
      translate_session.{hpp,cpp}
      op_table.cpp
      op/*.cpp                      # shared translators (from llama.cpp openvino/op)
      pass/*                        # fuse_to_sdpa, squeeze_matmul, ...
      rt_info/*
      builder/                      # OpenVINO-native path (optional component)
        gguf.{hpp,cpp}              # from GenAI gguf_utils
        gguf_quants.{hpp,cpp}       # from GenAI gguf_utils
        ggml_graph.{hpp,cpp}        # the ggml-independent intermediate graph
        gguf_builder_decoder.{hpp,cpp}
        arch/                       # per-architecture builders (llama, qwen2, qwen3, ...)
    tests/
  CMakeLists.txt                    # ov_add_frontend(NAME ggml ...)
```

- The **translators + interface** depend only on `openvino::core::dev` and build in
  every configuration (default wheel).
- The **native builder** (`builder/`) is self-contained OpenVINO code (gguf parsing
  + per-arch builders); it has **no ggml/libllama dependency** and can also ship in
  the default wheel.
- The **cgraph decoder is NOT in OpenVINO** — it lives in llama.cpp, which links the
  OpenVINO frontend.

## 6. Distribution / packaging

| Capability | Code | Dependency | Default wheel? |
|---|---|---|---|
| Translators + interface (cgraph→ov ops) | `frontends/ggml` core | `openvino::core` | **Yes** |
| Native `GGUF → ov::Model`/IR builder | `frontends/ggml/builder` | none (own parser) | **Yes** (gated by `ENABLE_GGML_GGUF`, default ON) |
| cgraph input path | `CgraphDecoder` in **llama.cpp** | libllama + ggml | n/a (lives in llama.cpp) |

The native builder needs no libllama and no submodule — it is the production path
and lives entirely in OpenVINO. There is **no** mandatory FetchContent of llama.cpp.
GenAI's optional use of the cgraph path (§7) is the only place libllama enters, and
it is opt-in there, not in OpenVINO core.

## 7. OpenVINO GenAI integration (the two paths)

`read_model("model.gguf")` selects a path:

1. **Native builder (default, production).** Architecture is supported by
   `frontends/ggml/builder/arch/*` → build `GgmlGraph` → translate →
   self-contained, mmap-owned `ov::Model`. Optimal memory, IR-serializable, no extra
   dependency. Replaces today's `gguf_utils` path (which is promoted into OpenVINO).

2. **llama.cpp path (opt-in, quick enable).** Architecture not yet supported
   natively, *and* GenAI was built with the optional llama.cpp dependency: GenAI
   asks libllama to load the gguf and build a `ggml_cgraph`, wraps it in
   llama.cpp's `CgraphDecoder`, and converts via the same frontend. Covers the long
   tail of architectures immediately, at the cost of non-self-contained weights and
   a heavy dependency.

Selection policy (suggested): try native; if the architecture is unsupported and the
llama.cpp backend is available, fall back with a logged notice; otherwise raise a
clear "architecture X not natively supported; rebuild with llama.cpp path to enable"
error. GenAI deletes its `gguf_utils` copy and calls the OpenVINO frontend in both
cases.

## 8. llama.cpp integration (consuming the frontend back)

- llama.cpp **deletes** `ggml/src/ggml-openvino/openvino/` (the translators) and
  links `openvino::frontend::ggml` instead.
- It **keeps** `CgraphDecoder` (today's `GgmlOvDecoder`) and the backend glue
  (`ggml-openvino.cpp`), which build a `ggml_cgraph` at runtime and feed it through
  the now-shared frontend.
- The `GgmlDecoder` ABI becomes a cross-repo contract (see §12).

## 9. Effort estimate

Baselines are real, measured code, so most of this is relocation + re-targeting, not
green-field.

| Work item | Nature | Rough size |
|---|---|---|
| Move translators + interface into `src/frontends/gguf/` | relocation; CMake via `ov_add_frontend`; namespace/visibility | ~2,400 LoC moved |
| Move `gguf.cpp` + `gguf_quants.cpp` from GenAI | relocation | ~850 LoC moved |
| Define `GgmlGraph` intermediate + `GgufBuilderDecoder` | new glue | ~300–600 LoC |
| Re-target per-arch builders to emit `GgmlGraph` (llama/qwen2/qwen3) | re-target from GenAI `building_blocks` | ~1,000 LoC re-targeted |
| Owning-Constant weight path (mmap → `AlignedBuffer`) in builder decoder | new, follows IR frontend | small |
| GenAI: switch `read_model` to OpenVINO frontend; add path selection; delete `gguf_utils` | integration | small |
| llama.cpp: delete vendored translators, link frontend, keep decoder | integration | small |
| Tests: GGUF→ov::Model, GGUF→IR round-trip, cgraph parity, both GenAI paths | new | medium |

Net: the **3-architecture native baseline is largely existing code re-homed.** The
ongoing cost is **per-new-architecture hand support on the native path**: cheap
(tens–low-hundreds of LoC) for transformer variants that fit the existing template,
expensive (a new builder + possibly new ops/quant) for structurally novel families
(MoE routing, SSM/Mamba, RWKV, hybrids). The llama.cpp path absorbs the
**model-assembly** cost of those cases as an opt-in — but only the assembly, not the
op cost: if the novel family also brings **new ggml ops**, the cgraph path fails to
convert until a translator is added (a cost the native path would pay anyway, and
which both paths then share). So "Path A for the long tail" holds for new
*architectures over known ops*, not for new *ops*.

## 10. Risks and mitigations

- **Cross-repo `GgmlDecoder` ABI drift.** llama.cpp and OpenVINO now share an
  interface across repos. *Mitigation:* version the interface; pin llama.cpp↔OpenVINO
  compatibility; define ownership of changes (a new ggml op needs a translator in
  OpenVINO **and** decoder support); CI that builds llama.cpp against a pinned
  OpenVINO.
- **Two op-emission implementations diverge** (native builder's `GgmlGraph` emission
  vs. llama.cpp's cgraph shapes). *Mitigation:* the translators are the single
  lowering authority; add parity tests that convert the same model both ways and
  compare op graphs/outputs.
- **Native builder lags new architectures.** *Mitigation:* that is the explicit
  division of labor — llama.cpp path covers the tail; promote to native when a model
  matters for production.
- **Inference-shaped graphs.** Both paths produce KV-cache/stateful graphs, not
  neutral architectural graphs; the IR reflects that. *Mitigation:* document; this is
  acceptable (and intended) for the GenAI/LLM use case.
- **Wheel size / build matrix.** Native builder is pure OpenVINO C++ (no libllama),
  so the default wheel impact is modest. The libllama dependency stays out of
  OpenVINO entirely (only optional in GenAI builds).
- **Weight lifetime on the cgraph path.** `ov::Model` is a view into libllama memory.
  *Mitigation:* document; recommend the native path for `GGUF → IR`; for the cgraph
  path, keep the libllama model alive for the model's lifetime (runtime) or only
  until `serialize()` (conversion).

## 11. Milestones

1. **M1 — Frontend skeleton + cgraph path.** Move translators + interface into
   `src/frontends/gguf/`; llama.cpp links it back and deletes its copy. Validates the
   shared engine end-to-end with zero behavior change. *(Lowest risk, immediate
   "one frontend" win.)*
2. **M2 — Native builder, direct-`ov` interim (optional).** Promote GenAI's
   `gguf_utils` largely as-is (emitting `ov` ops) into OpenVINO + owning-Constant
   mmap weights. Delivers production memory + self-contained IR + default-wheel GGUF
   support fast, before the `GgmlGraph` unification lands.
3. **M3 — Unify on `GgmlGraph`.** Re-target the native builder to emit the ggml-op
   intermediate and route through the shared translators. Now both inputs truly share
   one conversion path; retire the direct-`ov` emission.
4. **M4 — GenAI two-path integration.** `read_model` selection, delete `gguf_utils`,
   opt-in llama.cpp fallback.
5. **M5 — Hardening.** Parity tests, IR round-trip tests, architecture coverage,
   docs (supported-ops, supported-architectures, packaging).

## 12. Open decisions

- **Enum vs. string** for the ggml-op tag in `GgmlGraph` (avoid leaking any ggml
  header into OpenVINO either way).
- **Where the architecture registry lives** and how third parties add a native
  architecture builder (extension mechanism?).
- **Whether M2 (direct-`ov` interim) is worth it** or we go straight to M3.
- **libllama packaging for GenAI's opt-in path** — FetchContent pinned commit,
  CPU-only ggml (all compute backends off), or a separate `openvino-gguf`-style
  package analogous to `openvino-tokenizers`.
- **Interface versioning policy** for the cross-repo `GgmlDecoder` contract.
```
