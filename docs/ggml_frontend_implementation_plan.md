# GGML Frontend — Implementation Plan (qwen3-8b bring-up)

**Status:** Plan for execution
**Date:** 2026-06-10
**Companion doc:** [ggml_frontend_proposal.md](ggml_frontend_proposal.md) (design rationale & architecture)

This plan turns the proposal into an ordered, buildable sequence of steps. It reflects
the four decisions taken at kickoff:

1. **Mode 2 (.gguf without llama.cpp) uses a *synthetic ggml-op decoder*** — it parses
   the `.gguf` and emits a node list in the **GGML_OP_\*** vocabulary, then runs through
   the **same** `op_table` / `TranslateSession` as the cgraph path. No second op-emission
   implementation.
2. **The gguf reader is vendored into the OpenVINO frontend** (`gguf.cpp`,
   `gguf_quants.cpp` + `gguflib` via FetchContent). OpenVINO becomes self-sufficient; no
   llama.cpp or GenAI dependency.
3. **Target model = the model llama.cpp's cgraph produces**, not GenAI's
   `building_blocks` model. The synthetic decoder must reproduce llama.cpp's qwen3
   cgraph node-for-node so both modes converge on an identical `ov::Model`. GenAI
   compatibility is handled by a thin post-processing/naming layer (see §6), not by
   building a different graph.
4. **The OV-side `FrontEnd` becomes a standard, registered `ov::frontend::FrontEnd`** so
   `core.read_model("model.gguf")` works, while the cgraph entry point is kept for
   llama.cpp.

**Amendment (kickoff +1): the `GgmlDecoder` interface is NOT frozen.** We own both
decoder implementations (the cgraph one in llama.cpp can be modified freely when it
switches to consuming the OpenVINO frontend). So the interface is **co-designed for
generality first**, and the cgraph decoder is adapted to it — rather than the synthetic
decoder being forced to mimic ggml's raw memory layout. This removes what was the single
biggest risk in M2 (faking `op_params` byte layout) and is captured as a new
**Phase M1.5** below.

---

## 0. Current state (verified)

- `src/frontends/ggml/` already contains the **translators + interface** copied from
  `llama.cpp/ggml/src/ggml-openvino/openvino/` (~2.4k LoC): `decoder.h` (abstract
  `GgmlDecoder : DecoderBase`), `frontend.{h,cpp}`, `input_model.{h,cpp}`,
  `node_context.h`, `translate_session.cpp`, `op_table.cpp`, `op/*.cpp`, `pass/*`,
  `rt_info/*`.
- It does **not** build yet: no `CMakeLists.txt`, no `include/openvino/frontend/ggml/`
  public layout, no registration, and `translate_session.cpp` still includes
  `ggml-openvino/openvino/...` paths.
- The `FrontEnd` is the minimal `static convert(InputModel, naive)` shape used by
  llama.cpp — **not** an `ov::frontend::FrontEnd` subclass.
- The cgraph decoder (`GgmlOvDecoder`) lives in llama.cpp and stays there.
- The gguf reader + quant dequant + per-arch builder live in
  `openvino.genai/src/cpp/src/gguf_utils/` (`gguf.cpp`, `gguf_quants.cpp`,
  `building_blocks.cpp`, `gguf_modeling.cpp`) and depend on `gguflib` (FetchContent).
- qwen3 specifics confirmed: NEOX rope, per-head **q_norm / k_norm** RMSNorm, GQA
  (n_heads ≠ n_heads_kv). The cgraph path already handles these via existing
  translators (`rms_norm`, `rope` NEOX op_case `0x00010000`, `mulmat`, `flash_attn_ext`).

---

## Phase M1 — Make the frontend build & register (cgraph path, zero behavior change)

Goal: `src/frontends/ggml` compiles inside OpenVINO as a real frontend, validated by the
cgraph path (llama.cpp links it back). No `.gguf` reading yet.

### M1.1 Public-header layout & namespace
- Create `src/frontends/ggml/include/openvino/frontend/ggml/`:
  - `frontend.hpp` — `class GGML_FRONTEND_API FrontEnd : public ov::frontend::FrontEnd`.
  - `decoder.hpp` — move the abstract `GgmlDecoder` here (public ABI, the cross-repo
    contract). Keep it `DecoderBase`-derived and header-only-pure-virtual.
  - `visibility.hpp` — `GGML_FRONTEND_API` export macro (copy pattern from
    `tensorflow_lite`/`paddle`).
- Move implementation files under `src/frontends/ggml/src/` (mirror the other FEs:
  `frontend.cpp`, `input_model.cpp`, `node_context.hpp`, `translate_session.{hpp,cpp}`,
  `op_table.cpp`, `op/*`, `pass/*`, `rt_info/*`).
- Fix the stale includes in `translate_session.cpp`
  (`ggml-openvino/openvino/node_context.h` → `node_context.hpp`, etc.).

### M1.2 CMake
- Add `src/frontends/ggml/CMakeLists.txt` using `ov_add_frontend(NAME ggml ...)`
  (model out: `LINKABLE_FRONTEND`, link `openvino::core::dev`,
  `openvino::frontend::common`). Pattern from `src/frontends/paddle/CMakeLists.txt`.
- Add `ENABLE_OV_GGML_FRONTEND` option (default ON) and wire it into
  `src/frontends/CMakeLists.txt`.
- No third-party deps in this phase (builder lands in M2).

### M1.3 Standard FrontEnd surface
- Implement on the `ov::frontend::FrontEnd` subclass:
  - `get_name()` → `"ggml"`.
  - `convert(InputModel::Ptr) const` → existing `TranslateSession` flow (move the body
    of today's static `convert`). Keep a `convert(model, naive)` overload internally.
  - `supported_impl(variants)` / `load_impl(variants)` — in M1 these handle the
    **decoder-passed-in** variant: llama.cpp constructs `InputModel(decoder)` and calls
    `convert` directly, so `load_impl` can initially just wrap a provided
    `GgmlDecoder`. (File-path loading is added in M2.)
- Add `ggml.cpp` registration TU: `GetFrontEndData()` / `GetAPIVersion`
  (`OV_FRONTEND_API_VERSION`), `extern "C"` exports — copy from
  `src/frontends/pytorch/src/pytorch.cpp`.

### M1.4 llama.cpp side
- Point llama.cpp's `CMakeLists.txt` (`ggml/src/ggml-openvino`) at the OpenVINO
  `openvino::frontend::ggml` target; **delete** its vendored `openvino/` translator copy.
- Keep `ggml-decoder.{h,cpp}` (`GgmlOvDecoder`) and `ggml-openvino.cpp` backend glue;
  have them include the public `openvino/frontend/ggml/decoder.hpp`.
- Verify: build llama.cpp against the in-tree OpenVINO, run an existing qwen3 cgraph
  conversion → byte-identical `ov::Model` to before the move (regression gate).

**M1 exit:** OpenVINO builds with the ggml frontend; llama.cpp converts qwen3-8b through
it with zero behavior change; its translator copy is gone.

---

## Phase M1.5 — Generalize the `GgmlDecoder` contract (typed attributes + semantic roles)

Goal: make the decoder interface express *meaning*, not ggml memory layout, so that
**any** decoder (cgraph or gguf-builder) can serve it without reproducing ggml's byte
layout. Done while llama.cpp's `GgmlOvDecoder` is the **only** implementation, so there's
exactly one decoder to migrate in lockstep with the translators. This is the cheapest
moment to make the change and the highest-leverage one for M2.

### Why (verified against the current translators)

The translators currently reach through the interface into ggml's raw memory in two ways
that would otherwise force the synthetic gguf decoder to fake ggml layout:

**(a) Raw `op_params` pointer reads, by byte offset.** Each becomes a typed attribute:

| op translator | today (brittle) | new typed attribute |
|---|---|---|
| `rms_norm` | `memcpy(&eps, get_output_op_params(), 4)` | `get_attribute("eps") -> float` |
| `scale` | `(float*)params[0]`, `[1]` | `"scale"`, `"bias" -> float` |
| `soft_max` | `(float*)params[0]`, `[1]` | `"scale"`, `"max_bias" -> float` |
| `flash_attn_ext` | `(float*)params[0..2]` | `"scale"`, `"max_bias"`, `"logit_softcap" -> float` |
| `glu_swiglu`/`glu_geglu` | `params[1]` (swapped) | `"swapped" -> bool` |
| `rope` | `make_sin_cos(rope_params[1,4,5..10])` | `"rope" -> RopeConfig` (typed struct: n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, mode) |
| `permute` / `process_view_input` | `offset=*(size_t*)params; offset/stride` | `get_input_view_offset(node_idx, name) -> int64_t` (byte offset of the input VIEW; the two call sites divide by *different* strides, so the byte offset is the single source of truth and the bytes→elements division stays in the translator using `get_input_stride`) |

**(b) The opaque `op_case` integer**, computed *only* from ggml node topology
(`compute_op_case` walks `src[]->op` chains: is this PERMUTE on a kv-cache / SWA / rope'd
query; is MUL_MAT's B a `CONT(TRANSPOSE)`; is GET_ROWS' index a VIEW; RESHAPE shape
relations). The cgraph decoder *reverse-engineers tensor roles from topology*; the gguf
builder **knows each tensor's role by construction**. Replace the magic int with a small
set of **semantic enums/attributes** the decoder declares directly, e.g.:
  - tensor role: `INPUT_EMBD | KV_CACHE_K | KV_CACHE_V | ROPED_Q | WEIGHT | ...`
  - per-op intent: `mulmat_weight_is_transposed: bool`, `reshape_kind: enum`,
    `rope_input_is_view: bool`, `permute_target: {generic, kv_cache, kv_cache_swa, roped_q}`.
  The exact enum set is derived by enumerating the `op_case` values each translator
  branches on (already inventoried: rope `0x0001xxxx/0x0002xxxx | 0x2`, mulmat `2/3`,
  permute `1/2/3/4`, reshape `1..6`, cont `1/2/3`, view `2/3`, get_rows `2`).

### What to change

- **`decoder.hpp`:** keep `get_attribute(name) -> ov::Any` (already declared, currently
  unused — `GgmlOvDecoder::get_attribute` just returns `nullptr`). Promote it to the
  primary mechanism for scalar op params. Add typed convenience accessors or a small
  `RopeConfig`/role enum in the public header. Deprecate (but keep, for transition)
  `get_input_op_params` / `get_output_op_params` / `get_rope_params` / the raw
  `get_op_case` int.
- **Translators + `NodeContext`:** switch the 7 ops above from raw-pointer reads to
  `context.get_attribute<T>(...)`, and switch the `op_case` branches to the semantic
  enums. `node_context.h` gains typed `get_attribute<T>` helpers (pattern: other FEs'
  `NodeContext`).
- **`GgmlOvDecoder` (llama.cpp):** implement `get_attribute` by reading what it already
  reads from `op_params`/topology and returning it **typed** (it has all the data — this
  is a pure relocation of the offset arithmetic from the translator into the decoder).
  Implement the semantic-role attributes from the same `compute_op_case` logic it already
  runs. Net: the brittle knowledge stays in exactly one place and is computed once.
- Keep the change **behavior-preserving**: re-run the M1 cgraph parity gate; the produced
  `ov::Model` must be byte-identical.

### Sequencing note
Do this in **two commits**: (1) add typed attributes alongside the raw accessors and
migrate translators; (2) once green, remove the raw accessors. This keeps llama.cpp
buildable at each step and makes the diff reviewable.

**M1.5 exit:** translators no longer dereference ggml-shaped raw memory; every op
parameter and tensor role is a typed attribute / enum on the interface. A new decoder
only needs to answer *semantic* questions — not reproduce ggml's `op_params` layout.

---

## Phase M2 — Native `.gguf` reader + synthetic ggml-op decoder (qwen3-8b)

Goal: `core.read_model("qwen3-8b.gguf")` → `ov::Model` with **no llama.cpp dependency**,
producing a graph that matches the cgraph path's model.

### M2.1 Vendor the gguf reader
- Copy into `src/frontends/ggml/src/builder/`:
  - `gguf.{hpp,cpp}` (container parse, metadata, tensor table) from GenAI.
  - `gguf_quants.{hpp,cpp}` (Q4_0/Q4_1/Q4_K/Q6_K/Q8_0 dequant→OV `u4/u8 + scale + zp`).
- Add `gguflib` via FetchContent in the frontend CMake (pinned commit, matching the
  hash GenAI already uses), gated by the frontend enable flag.
- Strip GenAI-only coupling: replace `ov::genai::utils::*` logging with frontend-local
  logging; drop `gguf_tokenizer.*` (not needed for graph conversion).

### M2.2 `GgmlGraph` — the ggml-independent intermediate
- New `builder/ggml_graph.{hpp,cpp}`: a lightweight node list mirroring what
  `GgmlDecoder` exposes (post-M1.5). Each node carries:
  - op tag as a **string** (`"GGML_OP_MUL_MAT"`, …) — decision: string, not enum, to
    avoid any ggml header leak and to match the existing `op_table` keys verbatim.
  - input references (by output-name), **typed op attributes** (the M1.5 attribute set:
    `eps`, `scale`, `RopeConfig`, `swapped`, `get_input_view_offset`, semantic role/intent
    enums), output name/shape/type. **No raw ggml `op_params` layout** — that's the whole
    point of M1.5; the builder sets attributes by construction since it knows each
    tensor's role.
  - a weight table: name → `ov::Constant` (built with the **ownership-taking** ctor over
    the OV mmap of the `.gguf`; quantized weights dequant-copied via `gguf_quants`).
- This is the data `GgufBuilderDecoder` will serve.

### M2.3 `GgufBuilderDecoder : public GgmlDecoder`
- New `builder/gguf_builder_decoder.{hpp,cpp}`: a thin adapter implementing every
  `GgmlDecoder` virtual against a `GgmlGraph` (the mirror of llama.cpp's
  `GgmlOvDecoder`, but reading `GgmlGraph` instead of `ggml_cgraph`).
- Must replicate the metadata `GgmlOvDecoder` computes: `get_model_inputs` (inp_tokens,
  inp_pos, inp_out_ids, self_kq_mask[_swa], embd), `get_model_weights`,
  `get_model_output_names`, `get_rope_params`, `get_kv_param_res_names`, `is_static`,
  `is_stateful`, `is_swa_layer`, `get_op_case`. Reuse the naming conventions from
  `get_graph_input_ov_name` so `TranslateSession::preprocess` (sliced mask, rope sin/cos)
  fires identically.

### M2.4 qwen3 architecture builder (emit GGML_OP_\* nodes)
- New `builder/arch/qwen3.cpp`: re-target GenAI's `building_blocks.cpp` `layer()` so that
  instead of emitting `ov` ops it appends **`GgmlGraph` nodes in ggml vocabulary** that
  reproduce **llama.cpp's qwen3 cgraph topology**:
  - embed (GET_ROWS) → per-layer { RMS_NORM (input norm) → q/k/v MUL_MAT →
    **q_norm/k_norm RMS_NORM per head** → ROPE (NEOX) → SET_ROWS into KV → attention
    (MUL_MAT / SOFT_MAX / MUL_MAT, or FLASH_ATTN_EXT to match cgraph) → o-proj MUL_MAT →
    residual ADD → post-attn RMS_NORM → SwiGLU FFN (MUL_MAT, GLU_OP_SWIGLU, MUL_MAT) →
    residual ADD } → final RMS_NORM → lm_head MUL_MAT.
  - Pull head_num/head_num_kv/head_size/rms_norm_eps/rope_freq_base/layer_num from gguf
    metadata via `gguf.cpp` (keys already read by GenAI: `head_num`, `head_num_kv`,
    `head_size`, `rms_norm_eps`, `layer_num`, `rope_freq_base`, `max_position_embeddings`,
    `file_type`, `architecture`).
- **Parity rule:** the emitted node sequence and **typed attributes** must match the
  cgraph for qwen3 so that both decoders feed the translators identical inputs. (After
  M1.5 this is a comparison of semantic attributes, not raw byte layout.) This is the
  verifier (§M5) for "same model llama.cpp produces."
- Architecture dispatch: `builder/arch/registry` keyed by `architecture` metadata; qwen3
  first (llama/qwen2 share the template and come nearly free, but are out of scope until
  qwen3 is green).

### M2.5 FrontEnd file-path loading
- Extend `load_impl`/`supported_impl`: if the variant is a path/stream ending `.gguf`
  (sniff the `GGUF` magic), build `GgmlGraph` via the arch builder → wrap in
  `GgufBuilderDecoder` → `InputModel`. `supported_impl` returns true on the magic.
- `convert` is unchanged — both decoders flow through the same `TranslateSession`.

**M2 exit:** `core.read_model("qwen3-8b.gguf")` yields a self-contained, IR-serializable
`ov::Model` with no llama.cpp dependency, matching the cgraph path's graph.

---

## Phase M3 — Stateful / KV-cache parity & GenAI compatibility shaping

Goal: the produced model is consumable by GenAI's `LLMPipeline` the way the current
`gguf_utils` model is.

- The cgraph path already produces a **stateful** model (`MakeStateful` over
  `kv_param_res_names`, ReadValue/Assign sinks). Confirm `GgufBuilderDecoder` reports the
  same `get_kv_param_res_names()` / `is_stateful()==true` so `TranslateSession`'s existing
  `MakeStateful` pass produces the same sinks.
- **Naming/IO compatibility layer (thin):** GenAI expects `input_ids`, `attention_mask`,
  `position_ids`, `beam_idx`, `logits` and `past_key_values.N...` variable names. The
  cgraph/ggml graph uses `inp_tokens`, `inp_pos`, `self_kq_mask`, etc. Add an opt-in
  post-process (PrePostProcessor + variable-id remap) selected when loading for GenAI, to
  rename/adapt IO to the GenAI contract **without changing the compute graph**. Decide
  placement: frontend option vs. GenAI-side adapter (recommend GenAI-side, keeps the FE
  contract clean).
- Verify `runtime_options` rt_info (kv_cache_precision f16, activations_scale_factor)
  expected by GenAI is set (cgraph path sets analogous; add if missing).

**M3 exit:** the GGUF→ov::Model from the native path runs end-to-end in a GenAI
`LLMPipeline` for qwen3-8b with correct output.

---

## Phase M4 — GenAI integration (two selectable paths)

- Switch GenAI `read_model("*.gguf")` to call `openvino::frontend::ggml` (native path)
  instead of `gguf_utils`.
- Add path selection: native builder by default; optional llama.cpp cgraph fallback
  (opt-in dependency) for architectures the native builder doesn't yet cover.
- Delete GenAI's `gguf_utils/{gguf,gguf_quants,building_blocks,gguf_modeling}.cpp`
  (keep `gguf_tokenizer` if still used elsewhere).

---

## Phase M5 — Hardening & verification

- **Parity test (the key gate):** convert qwen3-8b both ways (llama.cpp cgraph decoder
  vs. native `GgufBuilderDecoder`) and assert the resulting `ov::Model`s are
  structurally equal (op types, topology, op_params) and numerically equal on a fixed
  prompt within tolerance.
- **IR round-trip:** native model → `serialize()` → `read_model()` → same outputs;
  confirm weights are mmap-owned (model valid after the source file handle context).
- **Accuracy:** logits vs. llama.cpp reference for a few prompts (reuse the
  WordFluency-style verify harness approach).
- Docs: supported-ops table, supported-architectures (qwen3 first), packaging notes.
- Coding style: clang-format + copyright headers (`ov-ensure-coding-style`).

---

## Risks / open items specific to this plan

- **Cgraph topology drift:** the native qwen3 builder must track whatever attention form
  the cgraph emits (explicit SDPA vs. `FLASH_ATTN_EXT`). Pin to one and assert in the
  parity test; the `fuse_to_sdpa` pass should reconcile minor differences.
- **~~op_params byte-layout fidelity~~ (resolved by M1.5):** previously the highest risk —
  the synthetic decoder would have had to fill ggml's exact `op_params`/`rope_params` byte
  layout. M1.5 removes this by converting all raw reads to typed attributes, so the
  builder sets meaning directly. The residual risk moves into M1.5 itself: getting the
  `GgmlOvDecoder` attribute implementation to reproduce the old offset arithmetic exactly
  — guarded by the M1 byte-identical-model parity gate.
- **`is_static` / SWA:** qwen3-8b is non-SWA; keep `is_static=false` (stateful dynamic)
  for the first bring-up and defer NPU/static.
- **gguflib pin** shared with GenAI to avoid format-parse divergence.
- **GgmlDecoder ABI** is now a public cross-repo contract — version it and add a
  llama.cpp-against-pinned-OpenVINO CI build (proposal §10).
