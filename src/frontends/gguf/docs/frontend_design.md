# OpenVINO GGUF Frontend — design & developer guide

The GGUF frontend converts GGUF model files into an `ov::Model`. It is a standard
`ov::frontend::FrontEnd` (registered as `"gguf"`, library `openvino_gguf_frontend`) and is the
path `core.read_model("model.gguf")` and OpenVINO GenAI take to run llama.cpp-style models on
OpenVINO devices.

For "how do I add a new model architecture", see [adding_an_architecture.md](adding_an_architecture.md).
This document explains the *design* — the layering, the two decoder paths, why the frontend loads
GGUF itself instead of depending on llama.cpp, and the memory-consumption model.

## Layering

```
 .gguf file / live ggml_cgraph
          |
          v
   GgufDecoder  (abstract)  ── include/openvino/frontend/gguf/decoder.hpp
     |            \
     |             \__ two concrete implementations (see "Two decoder paths")
     v
  TranslateSession  ── src/translate_session.cpp
     |   walks the decoder's nodes, calls one op translator per node
     v
  op translators  ── src/op/*.cpp  (GGML_OP_MUL_MAT -> MatMul, GGML_OP_ROPE -> RoPE, ...)
     |   + normalization passes (SetRows lowering, MakeStateful, ...)
     v
   ov::Model  (stateful, SDPA-shaped)
```

The **`GgufDecoder` interface** is the single seam. It is node-scoped: `visit_subgraph` hands the
translator a decoder bound to one node, and per-node accessors (`get_op_type`, `get_input_shape`,
`get_attribute`, `get_op_case`, ...) refer to that node; model-scope accessors (`get_model_inputs`,
`get_model_weights`, `get_kv_param_res_names`, `is_stateful`, ...) answer whole-model questions.
Everything downstream of the decoder (translators, passes, the produced `ov::Model`) is shared by
both decoder paths — so a new op or a graph fix benefits both at once.

## Two decoder paths (one frontend)

The frontend is fed by **two** `GgufDecoder` implementations, deliberately:

1. **`GgufBuilderDecoder`** (`src/builder/`) — the *native* path. `core.read_model("model.gguf")`
   parses the container (`src/quant/gguf.cpp`), and `TransformerBuilder` (`src/builder/gguf_builder.cpp`)
   builds a `GgufGraph` per-architecture (a flat, topologically-ordered node list in the GGML op
   vocabulary), which `GgufBuilderDecoder` exposes. **No llama.cpp in the process.** This is the
   default and the path GenAI uses.

2. **`GgmlOvDecoder`** (lives in llama.cpp's `ggml/src/ggml-openvino` backend, not in this repo) —
   wraps a live `ggml_cgraph` that llama.cpp already built. Used when OpenVINO is a *backend inside
   llama.cpp*; it links `openvino::frontend::gguf` and calls `FrontEnd::convert()` directly.

Both produce the exact same op vocabulary, so they share the translators and passes verbatim.

### Why the native path does not use llama.cpp to load the model

The obvious alternative — have `read_model` call into llama.cpp to build the cgraph (reusing path
2's `GgmlOvDecoder`) — was considered and rejected for the default path. llama.cpp's graph builder
is not a standalone library: it is entangled with `llama_model` loading and the `libllama`/`libggml`
runtime, so using it means linking substantial llama.cpp + ggml into the OpenVINO/GenAI process
(whether via submodule, FetchContent, or a prebuilt lib — the mechanism is not the issue, the
dependency is). The drawbacks that made the native builder the default:

- **A second model-loading runtime + ~2x transient memory.** llama.cpp would allocate the full
  model into ggml tensors just to *build* the cgraph, and OpenVINO would then re-materialize the
  weights as `Constant`s — roughly double the host memory during load. The native builder mmaps the
  file and zero-copies weight bytes straight into OpenVINO `Constant`s (see "Memory model").
- **Version coupling.** llama.cpp has no stable ABI and GGUF/architecture/tensor-naming conventions
  change quickly; pinning a version means chasing upstream, not pinning means breakage.
- **Binary size, build matrix, supply-chain.** Shipping libllama+libggml (CPU kernels, quant
  kernels, tokenizer, sampling) inside OpenVINO enlarges the binary, adds a CMake/SIMD/backend
  matrix to build on every target, and adds a CVE/provenance surface.
- **Loss of self-containment.** `core.read_model(".gguf")` works in any OpenVINO deployment with
  nothing extra; a llama.cpp dependency would break that.

The upside of llama.cpp — instant coverage of ~130 architectures — is real. The intended way to get
it *without* burdening the default path is to keep `GgmlOvDecoder` as an **optional, build-gated
alternative decoder** feeding the same frontend, never as the default. The native builder stays the
self-contained default.

## Weights: the GGML_OP_NONE convention

Weights are surfaced uniformly as `GGML_OP_NONE` leaf nodes (there is no `get_model_weights`
special path in the graph walk). `translate_weight` (`src/op/weight.cpp`) turns each leaf into a
compressed **decompression subgraph** — `Constant(low-bit) -> Convert -> Subtract(zp) ->
Multiply(scale) -> Reshape` — never a fully-materialized f32 constant. Two payload shapes are
accepted by the same translator:

- Native builder: the parser already extracted `weight`/`scales`/`zp` tensors; the leaf carries
  them as attributes and `translate_weight` calls `make_weight_node(base, weights, qtypes)`.
- cgraph path: the leaf carries the raw ggml bytes; `translate_weight` extracts them itself.

Both build the *identical* compressed OpenVINO subgraph, so **inference speed and compile memory
are independent of which path/payload was used** (verified: OLMoE compile peak 8673 MB via
GGML_OP_NONE vs 8672 MB via the older eager path).

## Memory model

Understanding where memory goes matters for large models (MoE especially).

**Load / `read_model`:**
- The GGUF file is memory-mapped (`ov::load_mmap_object`); non-quantized tensors are zero-copy views
  into the mmap. Quantized tensors are *repacked* once into a single `AlignedBuffer` (u4/u8 weights
  + f16 scales + integer zero-points in OpenVINO's compressed layout) and wrapped as `Constant`s
  via a `SharedBuffer` — so there is no second full-model host allocation and weights are never
  expanded to f32 at load. Measured: OLMoE-1B-7B q4_0 (3.9 GB file) read peak ≈ 3.9 GB.
- Because the graph keeps weights *compressed*, `read_model` memory is roughly the file size, not
  the dequantized (f32) model size.

**Compile / `compile_model`:**
- Weights stay compressed through compilation. Dense models fold the decompression into
  `FullyConnected`; MoE experts fold into `GatherMatmulCompressed`. Measured: OLMoE compile peak
  ≈ 8.7 GB (vs a ~53 GB blow-up if the expert decompression is *not* kept compressed).
- **Critical dependency for MoE:** the CPU plugin must recognize the expert decompression chain so
  it is not expanded to f32. Two changes make this work and both are required:
  `SnippetsMarkSkipped` must skip the `GatherMatmul` weight chain, and `is_decompression_multiply`
  must accept `GatherMatmul` consumers. Without them, `ConstantFolding` expands u4/u8 experts to
  f32 (the 53 GB case). See the `[CPU]` commits.
- KV cache precision: the stateful KV cache is f16 (set in `translate_session`); leaving it at the
  CPU default of u8 causes NaNs in some decode paths.

**Inference:** the compressed weights are decompressed on the fly by the plugin (dense: fused into
the FC kernel; MoE: `GatherMatmulCompressed`). No persistent f32 weight copy.

**Rule of thumb:** peak host memory ≈ `max(file_size_for_read, compressed_graph + plugin_scratch)`,
NOT the dequantized model size — *provided* the CPU decompression-recognition changes above are in
place. If a future change makes a model's weights expand to f32 at compile, that is the regression
to look for (compile peak jumping toward the dequantized size).

## Statefulness & the SDPA backend

The native builder emits a **stateful** graph: per-layer f16 KV-cache `Parameter`s written via a
`SetRows` placeholder, lowered to an appending `Concat` (+ `beam_idx` `Gather`) and turned into
`ReadValue`/`Assign` by `MakeStateful`. The graph is shaped so the CPU plugin's
`stateful_sdpa_fusion` folds attention into `ScaledDotProductAttentionWithKVCache`. The graph runs
on the **SDPA** attention backend, not PagedAttention — GenAI's `LLMPipeline` defaults to SDPA for
`.gguf` inputs (see `pipeline.cpp`). `AdaptToGenAI` (`src/pass/adapt_to_genai.cpp`, run by GenAI,
not the frontend) rewrites the llama.cpp-style IO (`inp_tokens`/`inp_pos`/`self_kq_mask`/... ) into
GenAI's contract (`input_ids`/`attention_mask`/`position_ids`/`beam_idx` -> `logits`).

## Tokenizer metadata

A GGUF file embeds not just the weights but the full tokenizer (vocab, merges, scores, token
types, special-token ids, pre-tokenizer regex, chat template) under its `tokenizer.*` metadata
keys. The frontend carries that metadata out on the converted model so a consumer can build a
matching OpenVINO tokenizer/detokenizer **without re-opening the `.gguf` and without any GGUF
parser of its own** — the model object is self-describing.

Mechanism (native path):
1. The builder scrapes every `tokenizer.*` key into an `ov::AnyMap` keyed by the sub-key after the
   last dot (`model`, `tokens`, `merges`, `scores`, `token_type`, `pre`, `bos_token_id`,
   `chat_template`, ...), each value being a `std::string` / `std::vector<std::string>` /
   `ov::Tensor` — `extract_tokenizer_config` in `gguf_builder.cpp`, surfaced through the decoder's
   `get_tokenizer_config()`.
2. `TranslateSession` attaches it to the converted model's **runtime info** as a
   `GGUFTokenizerMetadata` attribute under `gguf_tokenizer_metadata_key()`
   (`include/openvino/frontend/gguf/tokenizer_metadata.hpp`).
3. The attribute is deliberately **non-serializable** (`is_copyable() == false`, empty
   `to_string()`): the vocab+merges are large and only meaningful in-memory between conversion and
   tokenizer construction, so it is dropped on clone and emitted as an empty placeholder if the IR
   is serialized (it never bloats the XML). It is an in-process handoff, not part of the saved model.
4. A consumer (OpenVINO GenAI) reads `model->get_rt_info()[gguf_tokenizer_metadata_key()]` and
   builds the OpenVINO tokenizer/detokenizer from it — see GenAI's `create_tokenizer_from_model`
   (`gguf_utils/gguf_tokenizer.cpp`), which turns the map into BPE/Unigram tokenizer models via
   `openvino_tokenizers`. So `read_model(".gguf")` + this rt_info is enough for GenAI to produce
   both the inference model and its tokenizer.

### With and without a llama.cpp dependency

The tokenizer path is exactly where the frontend's llama.cpp-independence pays off, and it behaves
correctly on both decoder paths:

- **Without llama.cpp (native `.gguf` path, the default).** The frontend parses the `tokenizer.*`
  keys itself and emits the `GGUFTokenizerMetadata` rt_info. The whole tokenizer round-trip
  (`.gguf` file -> OpenVINO tokenizer) happens with **no llama.cpp and no separate GGUF/tokenizer
  library in the consumer** — the OpenVINO model is the single source of truth. This is what lets
  `core.read_model("model.gguf")` + GenAI stand alone.

- **With llama.cpp (the cgraph / `GgmlOvDecoder` path).** When OpenVINO runs as a backend *inside*
  llama.cpp, llama.cpp already owns the tokenizer natively (it parsed the same `tokenizer.*` keys
  to build its own `llama_vocab`). There is nothing for the frontend to hand off, so `GgmlOvDecoder`
  leaves `get_tokenizer_config()` empty and **no rt_info is attached** — tokenization is done by
  llama.cpp, the frontend only produces the compute graph. `TranslateSession` attaches the
  metadata only when `get_tokenizer_config()` is non-empty, so the same code serves both paths
  with no branching in the consumer.

In other words: the tokenizer metadata is populated by whichever side *owns* GGUF parsing. On the
native path that is the frontend (so it exports the metadata for a llama.cpp-free consumer); on the
cgraph path that is llama.cpp (so the frontend stays out of the tokenizer's way). Either way the
`GgufDecoder::get_tokenizer_config()` seam is the single contract, and no consumer needs to link
llama.cpp to obtain a tokenizer.

### How the tokenizer is constructed (consumer side)

The frontend only *exports* the metadata; turning it into a runnable tokenizer is the consumer's
job. This is described here because it defines the contract the frontend must satisfy (which keys,
which value types). The reference consumer is OpenVINO GenAI
(`src/cpp/src/gguf_utils/gguf_tokenizer.cpp`); the tokenizer is itself built as a pair of
**`ov::Model`s** (tokenizer + detokenizer) out of nodes from the **`openvino_tokenizers`** runtime
library — there is no bespoke tokenizer engine and, again, no llama.cpp.

Flow (native path): `read_model(".gguf")` attaches the rt_info, then
`create_tokenizer_from_model(model)`:

1. **Fetch** `model->get_rt_info()[gguf_tokenizer_metadata_key()]`, cast to
   `GGUFTokenizerMetadata`, and read its `.config` `AnyMap`. If the key is absent (model not from
   the frontend, or metadata stripped by serialization) it asserts — see the serialization caveat
   below.
2. **Normalize** the `AnyMap` into the same `map<string, GGUFMetaData>` that the from-file path
   (`tokenizer_config_from_meta`) produces, so a single builder serves both. (`tokenizer_config_from_rt_info`.)
3. **Build the two models** (`build_tokenizer_models`), loading `openvino_tokenizers`' factory
   entry point `create_tokenizer_node` at runtime (`get_symbol(..., "create_tokenizer_node")`).

The tokenizer `ov::Model` is assembled from these `openvino_tokenizers` operations (a `string`
`Parameter` in, token-id tensor out):

- `StringTensorUnpack` — unpack the input string tensor into (begins, ends, chars).
- `RegexNormalization` — text normalization, e.g. gemma4's SPM whitespace→metaspace (`U+2581 ▁`).
- `SpecialTokensSplit` — split out special tokens; the special-token set is derived from the GGUF
  `tokens` + `token_type` keys (entries whose type is CONTROL/USER_DEFINED via `is_special_token`).
- `RegexSplit` — the pre-tokenizer regex (BPE families).
- The core tokenizer node, dispatched on the GGUF `tokenizer.ggml.model` key:
  - `model == "llama"` (SPM/`plamo2`): `parse_spm_config` → **`SentencepieceTokenizer`**, fed the
    `tokens` + `scores` + `token_type` arrays.
  - `model == "gpt2"` / `"gemma4"` (byte-level BPE): `parse_bbpe_config` → **`BPETokenizer`**, fed
    the `tokens` vocab + `merges`.
- `RaggedToDense` — produce the dense `input_ids` output.

The detokenizer `ov::Model` is the inverse: **`VocabDecoder`** + `RegexNormalization` +
`StringTensorPack` for BPE, or **`SentencepieceDetokenizer`** for the llama/SPM family.

The GGUF metadata keys the builder relies on (so the frontend must preserve them verbatim):
`model`, `tokens`, `merges`, `scores`, `token_type`, `pre` (pre-tokenizer id), the special-token
ids (`bos_token_id`/`eos_token_id`/`unknown_token_id`/`padding_token_id`), `add_bos_token` /
`add_space_prefix` flags, and `chat_template` (the chat template is carried through for GenAI to
apply; GenAI additionally patches a few known-malformed templates, e.g. qwen2.5, in
`patch_gguf_chat_template`).

The resulting tokenizer/detokenizer `ov::Model`s are what `ov::genai::Tokenizer` wraps and compiles
like any other model — so tokenization runs on an OpenVINO device, consistent with the inference
model.

**Serialization caveat.** Because `GGUFTokenizerMetadata` is non-serializable, it exists only on the
in-memory model straight out of the frontend. A model that was serialized to IR and reloaded has no
such rt_info; GenAI then falls back to re-reading the `.gguf`
(`create_tokenizer_from_config` → `tokenizer_config_from_meta`), which needs the file but still no
llama.cpp. The rt_info path is the fast in-process handoff; the file path is the durable fallback.

## Source map

| Path | Contents |
|---|---|
| `include/openvino/frontend/gguf/` | public headers: `decoder.hpp`, `frontend.hpp`, `adapt_to_genai.hpp`, `tokenizer_metadata.hpp`, `set_rows_op.hpp` |
| `src/frontend.cpp` | FrontEnd: `.gguf` magic sniff + native load path; live-decoder path; extensions |
| `src/translate_session.cpp` | graph walk, weight seeding, statefulness passes, tokenizer rt_info |
| `src/op/*.cpp` | one op translator per GGML op |
| `src/builder/` | native `.gguf` parser adapter + `TransformerBuilder` + `GgufBuilderDecoder` |
| `src/quant/` | GGUF container parser (`gguf.cpp`), dequant fill fns (`gguf_quants.cpp`), weight-node construction (`weights.cpp`) |
| `src/pass/` | `LowerSetRows{Stateless,Stateful}`, `SqueezeMatmul`, `AdaptToGenAI` |
| `src/helper_ops/` | internal `SetRows` placeholder op |
| `tests/` | C++ op/dequant tests (in CI); standalone python dev/bench scripts |

## Testing

- C++ unit tests (`tests/*.cpp`, target `ov_gguf_frontend_tests`) cover op translators and weight
  dequant against real-ggml reference `.npy` fixtures, and run in CI.
- A graph-fingerprint check (sha256 over sorted `(op_type, output_shape)` pairs of the converted
  model, per architecture) is the recommended cheap regression gate for any builder change — it
  proves the produced graph is unchanged. Accuracy is validated opt-in against llama.cpp (WWB-style)
  and by comparing greedy tokens on the same prompt.
</content>
