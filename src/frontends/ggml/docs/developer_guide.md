# GGML Frontend — Developer Guide

The GGML frontend loads GGUF model files and translates them into OpenVINO IR.
It has no dependency on llama.cpp or the ggml library at runtime; it replicates
the relevant parts of the GGML compute graph in pure C++.

## Architecture overview

Two entry paths feed a single translation pipeline:

```
.gguf file ──► get_gguf_data() ──► build_ggml_graph_from_gguf()
                                            │
llama.cpp cgraph ──► GgmlDecoder ───────────┤
                                            ▼
                                    GgmlGraph (node list)
                                            │
                                    TranslateSession
                                    (op_table dispatch)
                                            │
                                    ov::Model (raw IR)
                                            │
                                    FuseToSDPA / SqueezeMatmul / MakeStateful
                                            │
                                    ov::Model (final IR)
```

The native GGUF path (`build_ggml_graph_from_gguf`) does not call into llama.cpp.
The llama.cpp cgraph path (`GgmlDecoder`) accepts a pre-built graph from an external
caller (e.g. a Python binding that runs llama.cpp inference and captures the graph).

## Source layout

```
src/frontends/ggml/
├── include/                   Public frontend headers (FrontEnd, visibility)
└── src/
    ├── builder/               Native GGUF path
    │   ├── gguf.hpp/.cpp      GGUF file parser, quant buffer allocation
    │   ├── gguf_quants.cpp    Per-type weight unpacking (fill functions)
    │   ├── gguf_builder.cpp   TransformerBuilder: arch detection + GgmlGraph emission
    │   ├── gguf_builder_decoder.cpp  Adapter: GgmlGraph → GgmlDecoder interface
    │   ├── weights.hpp/.cpp   make_weight_node / make_fused_qkv_weights
    │   └── ggml_graph.hpp     GgmlGraph / GgmlNode structs
    ├── op/                    One translator per GGML op
    ├── pass/
    │   ├── fuse_to_sdpa       Q·Kᵀ → softmax → V  ⟹  ScaledDotProductAttention
    │   └── squeeze_matmul     Remove redundant unsqueeze/squeeze around matmul
    ├── op_table.cpp           Op name → translator function map
    ├── translate_session.cpp  Walks GgmlGraph, calls op_table, runs passes
    └── frontend.cpp           ov::frontend::FrontEnd entry point
```

## Native GGUF path

### 1 — Parsing (`gguf.hpp`)

`get_gguf_data(file)` returns a `GGUFLoad` tuple:

```cpp
using GGUFLoad = std::tuple<
    std::unordered_map<std::string, GGUFMetaData>,  // metadata key-value pairs
    std::unordered_map<std::string, ov::Tensor>,    // tensors by GGML name
    std::unordered_map<std::string, gguf_tensor_type>, // qtype per weight base name
    std::shared_ptr<ov::MappedMemory>,              // mmap (keep alive)
    std::shared_ptr<ov::AlignedBuffer>              // single allocation for all quant data
>;
```

Non-quantized tensors (F16, BF16, F32) are zero-copy views into the mmap.
All quantized tensors share one `AlignedBuffer`; each tensor is a `SharedBuffer`
slice so the buffer stays alive as long as any tensor slice is alive.

After each tensor is repacked, `hint_evict()` calls `MADV_DONTNEED` on the mmap
pages to keep peak RSS near the output size rather than input+output.

`config_from_meta(metadata)` converts raw metadata into a typed config map
(`architecture`, `layer_num`, `head_num`, `head_size`, etc.) used by the builder.

### 2 — Weight unpacking (`gguf_quants.cpp`)

All fill functions write into **pre-allocated** `ov::Tensor` outputs.
Shapes must be set by the caller (see `quant_sizes` in `gguf.cpp`).

| Function | Types | Weight element | ZP element |
|---|---|---|---|
| `gguf_fill_q4_0` | Q4_0 | i4 (u32-packed, XOR-0x88) | none |
| `gguf_fill_sym` | Q8_0, Q5_0, Q6_K | i8 (centered) | none |
| `gguf_fill_asym` | Q4_1, Q4_K: u4 zp; Q5_K: u8 zp | u4 or i8 | u4 or u8 |
| `gguf_fill_mxfp4` | MXFP4 (gpt-oss) | f4e2m1 | f8e8m0 scale |

Symmetric types subtract the center value during unpacking so the stored i8
is already centered at 0; no zero-point tensor is produced. Asymmetric types
compute `zp = round(-min / scale)` (or `round(min / scale)` for K-quants) as
an integer, so the dequant subgraph is always `(w − zp) × scale`.

### 3 — Weight nodes (`weights.cpp`)

`make_weight_node(base, weights, qtypes)` returns an f32 output node.

Quantized weights become a compressed subgraph via `low_precision_dequantize`:
```
Constant(i4/u4/i8) → Subtract(zp) → Convert(f16) → Multiply(scale) → Reshape → Convert(f32)
```
This pattern is recognized by the CPU plugin's `MarkDequantization` and
`CompressedWeightsBlock` matchers, keeping weights compressed in memory.

For fused QKV tensors (phi-3, minicpm), `make_fused_qkv_weights` slices the
row dimension into separate Q, K, V nodes. Rows are block-independent in all
GGUF quant layouts, so the slice is a contiguous byte copy.

### 4 — Graph builder (`gguf_builder.cpp`)

`TransformerBuilder` emits a `GgmlGraph` using GGML op names and shapes.
Architecture differences are detected from the GGUF tensor table (layer 0):

| Detected condition | Architecture |
|---|---|
| `blk.0.attn_q_norm.weight` present | qwen3, hunyuan-dense |
| `blk.0.attn_q.bias` present | qwen2/2.5 |
| `blk.0.attn_qkv.weight` present (fused QKV) | phi-3, minicpm |
| `blk.0.ffn_gate_exps.weight` present | OLMoE, gpt-oss (MoE) |
| `blk.0.attn_sinks.weight` present | gpt-oss |
| `rope_freqs.weight` present | llama-3, phi-3 |
| `embedding_scale` / `residual_scale` metadata | minicpm |

RoPE mode is derived from the architecture name: NEOX (rotate-halves) for
`qwen2`, `qwen3`, `phi3`, `hunyuan-dense`; NORMAL (rotate consecutive pairs)
for everything else.

Currently supported GGUF architecture strings:

```
llama          qwen2          qwen3          phi3
minicpm        hunyuan-dense  olmoe          gpt-oss
```

Adding a structurally identical architecture (same attention + FFN topology,
same tensor naming) is done by adding its name to `supported_archs()` in
`gguf_builder.cpp` and verifying the RoPE mode.

Architectures with structural differences (SSM layers, hybrid attention,
per-layer input embeddings, shared KV layers) require new builder code.

## Op translation

`op_table.cpp` maps GGML op names to translator functions in `src/op/`.
Each translator receives a `NodeContext` and returns `ov::OutputVector`.

Registered ops:

| GGML op | OV translation |
|---|---|
| `GGML_OP_MUL_MAT` | `MatMul` (transposed B) |
| `GGML_OP_ROPE` | Decomposed RoPE (NEOX or NORMAL) |
| `GGML_OP_FLASH_ATTN_EXT` | `ScaledDotProductAttention` v13 |
| `GGML_OP_RMS_NORM` | `MVN` (no mean, normalize_variance) × weight |
| `GGML_OP_SOFT_MAX` | `Softmax` |
| `GGML_OP_SET_ROWS` / `GET_ROWS` | `Assign` / `ReadValue` for KV cache |
| `GGML_OP_MUL_MAT_ID` | MoE per-expert batched matmul |
| `GGML_OP_ADD_ID` | MoE per-expert bias add |
| `GGML_OP_TOP_K` | TopK |
| `GGML_OP_ARGSORT` | Argsort (descending) |
| `GGML_OP_SUM_ROWS` | ReduceSum (last axis) |
| `GGML_GLU_OP_SWIGLU` | SiLU(gate) × up |
| `GGML_GLU_OP_SWIGLU_OAI` | OAI gated activation (gpt-oss MoE) |
| `GGML_GLU_OP_GEGLU` | GELU(gate) × up |
| `GGML_UNARY_OP_SILU` | `Swish` |
| `GGML_UNARY_OP_GELU` | `Gelu` |
| ADD, MUL, SUB, DIV, SCALE | elementwise ops |
| RESHAPE, VIEW, PERMUTE, TRANSPOSE, CONT, CPY | shape/layout ops |

To add a new op: create `src/op/<name>.cpp`, implement the translator, and
register it in `op_table.cpp`.

## Post-translation passes

`TranslateSession` runs three passes after the op-by-op translation:

- **`FuseToSDPA`** — pattern-matches the manual attention sequence
  (`MulMat(K,Q) → Scale → SoftMax → MulMat(V,attn)`) into a single
  `ScaledDotProductAttention` v13 op.
- **`SqueezeMatmul`** — removes redundant `Unsqueeze/Squeeze` pairs
  that wrap `MatMul` in the MoE routing subgraph.
- **`MakeStateful`** — pairs `ReadValue`/`Assign` ops for KV cache slots
  to produce a stateful model for autoregressive generation.

## Adding a new architecture

**Same-family (llama-style attention + SwiGLU FFN):**
1. Add the GGUF arch string to `supported_archs()` in `gguf_builder.cpp`.
2. Check that RoPE mode is correct (`arch_uses_neox_rope()`).
3. If the arch has per-head Q/K normalization, QKV biases, or fused QKV,
   the existing detection flags (`m_has_qk_norm`, `m_has_qkv_bias`,
   `m_has_fused_qkv`) will pick them up automatically from the tensor table.

**Structurally different (new FFN, different normalization, SSM layers, etc.):**
1. Add fill functions in `gguf_quants.cpp` if new quant types are needed.
2. Add new ops in `src/op/` and register in `op_table.cpp`.
3. Add new builder logic in `gguf_builder.cpp` (new detection flags +
   new `build_*` methods called from the per-layer loop).
4. Update `config_from_meta` in `gguf.cpp` if new metadata keys are needed.

## Tests

Unit tests live in `tests/test_quant_dequant.cpp`. They compile the builder
sources directly into the test binary (the fill functions are internal and not
exported from the shared library):

```bash
cmake --build build --target ov_ggml_frontend_tests
ctest -R GGML_FE
```

Tests construct synthetic GGUF blocks in memory, call fill functions, then
evaluate the resulting `make_weight_node` graph via `ov::pass::ConstantFolding`
and compare against reference values.
