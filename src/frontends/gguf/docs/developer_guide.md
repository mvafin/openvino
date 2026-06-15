# GGUF Frontend — Developer Guide

The GGUF frontend loads GGUF model files and translates them into OpenVINO IR.
It has no dependency on llama.cpp or the gguf library at runtime; it replicates
the relevant parts of the GGML compute graph in pure C++.

## Architecture overview

Two entry paths feed a single translation pipeline:

```
.gguf file ──► get_gguf_data() ──► build_gguf_graph_from_gguf()
                                            │
llama.cpp cgraph ──► GgufDecoder ───────────┤
                                            ▼
                                    GgufGraph (node list)
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

The native GGUF path (`build_gguf_graph_from_gguf`) does not call into llama.cpp.
The llama.cpp cgraph path (`GgufDecoder`) accepts a pre-built graph from an external
caller (e.g. a Python binding that runs llama.cpp inference and captures the graph).

## Source layout

```
src/frontends/gguf/
├── include/                   Public frontend headers (FrontEnd, NodeContext, visibility)
└── src/
    ├── builder/               Native GGUF path
    │   ├── gguf.hpp/.cpp      GGUF file parser, quant buffer allocation
    │   ├── gguf_quants.cpp    Per-type weight unpacking (fill functions)
    │   ├── gguf_builder.cpp   TransformerBuilder: arch detection + GgufGraph emission
    │   ├── gguf_builder_decoder.cpp  Adapter: GgufGraph → GgufDecoder interface
    │   ├── weights.hpp/.cpp   make_weight_node / make_fused_qkv_weights
    │   └── gguf_graph.hpp     GgufGraph / GgufNode structs
    ├── op/                    One translator per GGML op
    ├── pass/
    │   └── squeeze_matmul     4D→3D activation reshape before MatMul (NPUW DQMatMulGQ2i)
    ├── op_table.cpp           Op name → translator function map
    ├── translate_session.cpp  Walks GgufGraph, calls op_table, runs passes
    └── frontend.cpp           ov::frontend::FrontEnd entry point + extension handling
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

`TransformerBuilder` emits a `GgufGraph` using GGML op names and shapes.
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

## Extensions

The frontend implements `FrontEnd::add_extension()` so callers can override or
supplement op translators and receive conversion diagnostics without modifying
the source tree.

### ConversionExtension — custom op translator

`ov::frontend::ConversionExtension` maps a GGML op name to a user-supplied
converter function.  The converter receives the base `ov::frontend::NodeContext`
and must return an `ov::OutputVector`.  The extension op name follows the same
naming convention as `op_table.cpp` (`"GGML_OP_*"`, `"GGML_UNARY_OP_*"`,
`"GGML_GLU_OP_*"`).

Extension translators override built-in translators when the name matches, so
they can replace an existing translation without changing any shared code.

```cpp
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/op/relu.hpp"

// Replace the built-in GGML_UNARY_OP_SILU with a ReLU for quick experiments.
auto fe = std::make_shared<ov::frontend::gguf::FrontEnd>();
fe->add_extension(std::make_shared<ov::frontend::ConversionExtension>(
    "GGML_UNARY_OP_SILU",
    [](const ov::frontend::NodeContext& ctx) -> ov::OutputVector {
        return {std::make_shared<ov::op::v0::Relu>(ctx.get_input(0))};
    }));
auto model = fe->convert(fe->load("model.gguf"));
```

The full node interface is available through `ctx.get_attribute<T>(name)` and
the standard `NodeContext` methods — no extra headers required:

| `ctx` call | Returns |
|---|---|
| `ctx.get_input(int)` | `ov::Output<ov::Node>` for input *i* |
| `ctx.get_input(string)` | `ov::Output<ov::Node>` by tensor name |
| `ctx.get_input_size()` | number of inputs |
| `ctx.get_name()` | op name string |
| `ctx.get_attribute<T>(name)` | op attribute or GGML metadata (see below) |

**GGML metadata keys** for `get_attribute<T>`:

| Key | Type `T` | Description |
|---|---|---|
| `"input_shape[N]"` | `ov::PartialShape` | shape of input *N* |
| `"input_type[N]"` | `ov::element::Type` | element type of input *N* |
| `"input_stride[N]"` | `std::vector<size_t>` | strides of input *N* (bytes) |
| `"input_view_offset[N]"` | `int64_t` | byte offset for view inputs |
| `"output_shape"` | `ov::PartialShape` | shape of the node's output |
| `"output_type"` | `ov::element::Type` | element type of the node's output |
| `"is_static"` | `bool` | true on the static/NPU path |
| `"is_stateful"` | `bool` | true when KV-cache is stateful |

Example using GGML metadata:

```cpp
fe->add_extension(std::make_shared<ov::frontend::ConversionExtension>(
    "GGML_OP_MY_CUSTOM",
    [](const ov::frontend::NodeContext& ctx) -> ov::OutputVector {
        auto shape     = ctx.get_attribute<ov::PartialShape>("input_shape[0]");
        auto stride    = ctx.get_attribute<std::vector<size_t>>("input_stride[0]");
        auto out_type  = ctx.get_attribute<ov::element::Type>("output_type");
        bool is_static = ctx.get_attribute<bool>("is_static");
        // ... build and return ov::OutputVector
    }));
```

### TelemetryExtension — conversion diagnostics

`ov::frontend::TelemetryExtension` delivers error and event callbacks from the
conversion pipeline back to the caller.

```cpp
#include "openvino/frontend/extension/telemetry.hpp"

fe->add_extension(std::make_shared<ov::frontend::TelemetryExtension>(
    "gguf_conversion",
    /* send_event */
    [](const std::string& category, const std::string& action,
       const std::string& label, int value) {
        std::cout << "[event] " << category << "/" << action << " " << label << "\n";
    },
    /* send_error */
    [](const std::string& category, const std::string& msg) {
        std::cerr << "[error] " << category << ": " << msg << "\n";
    },
    /* send_stack_trace */
    [](const std::string& category, const std::string& trace) {
        std::cerr << "[trace] " << category << ": " << trace << "\n";
    }));
```

### SOExtension — shared-library extensions

Passing an `ov::detail::SOExtension` wrapping a `.so`/`.dll` causes the frontend
to recursively unpack and register every `ov::Extension` exported from that
library.  The library handle is kept alive for the lifetime of the `FrontEnd`.

```cpp
fe->add_extension("/path/to/my_gguf_ops.so");  // via the string overload
```

## Post-translation passes

`TranslateSession` runs two passes after the op-by-op translation:

- **`SqueezeMatmul`** (static/NPU path only) — squeezes a leading size-1
  batch dimension off a 4D activation before `MatMul` (and unsqueezes
  after), normalizing to `3D × 2D`. Required because NPUW's
  `DQMatMulGQ2i` pattern only matches 3D activations.
- **`MakeStateful`** — pairs `ReadValue`/`Assign` ops for KV cache slots
  to produce a stateful model for autoregressive generation.

## gpt-oss specifics

gpt-oss (`LLM_ARCH_OPENAI_MOE` in llama.cpp) is an MoE model with three structural
features beyond the base llama family:

### SWA (Sliding-Window Attention) layer alternation

SWA layers use a **windowed attention mask** and a **distinct RoPE frequency base**.
The pattern is read from `gpt-oss.attention.sliding_window_pattern` (default: 2) and
follows llama.cpp `set_swa_pattern(period, dense_first=false)`:

```
layer is SWA  ⟺  il % period < period − 1
```

For period=2 this means even-indexed layers (0, 2, 4, …) are SWA; odd layers use
full attention.

The SWA rope base comes from `gpt-oss.rope.freq_base_swa` (falls back to the global
`rope.freq_base` when absent).

### Expert routing weight scale

`gpt-oss.expert_weights_scale` is an optional constant multiplier applied to the
routing probabilities after the per-expert softmax.  Mirrors llama.cpp's `w_scale`
argument to `build_moe_ffn`.  Defaults to 0 (no-op).

### Attention sinks

A learned per-head logit (`blk.N.attn_sinks.weight`, shape `[n_head]`) participates
in the SDPA softmax denominator without contributing a value.  Passed as the optional
5th input to `GGML_OP_FLASH_ATTN_EXT` and handled by the OV `ScaledDotProductAttention`
v13 6-input form.

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

The test suite lives in `tests/` and has two parts:

```
tests/
├── test_quant_dequant.cpp   Weight fill functions / dequant subgraph (9 tests)
├── test_ops.cpp             Per-op conformance tests   (30 tests)
├── op_test_utils.hpp        Shared helpers (OpTestGraph, load_npy, run_on_cpu)
├── generate_test_data.py    Generates tests/test_data/*.npy reference files
└── test_data/               Pre-calculated numpy arrays (committed to repo)
```

Build and run:

```bash
cmake --build build --target ov_gguf_frontend_tests
ctest -R GGUF_FE
```

### Per-op test design

Each test in `test_ops.cpp` follows this flow:

1. **Load inputs and expected outputs** from `tests/test_data/*.npy` using
   `load_f32` / `load_npy<T>` from `op_test_utils.hpp`.
2. **Build a single-op model** with `OpTestGraph`:
   - call `add_input(name, type, shape)` for each runtime input in the order
     they will be passed to the CPU plugin;
   - call `add_op(op_type, input_names, output_name, output_shape, output_type,
     op_case, input_shapes, input_types, attributes)`.
3. **Call `g.build()`** — creates a `GgufGraph`, wraps it in a
   `GgufBuilderDecoder`, runs `TranslateSession`, and reorders the model
   parameters to match the `add_input()` insertion order.
4. **Run on CPU** with `run_on_cpu(model, {tensor0, tensor1, ...})`.
5. **Compare** with `expect_near(result[0], expected, atol, label)`.

### Reference data generation

`tests/generate_test_data.py` contains one generator function per op.  Each
function uses only numpy so it can be run on any machine:

```bash
python3 src/frontends/gguf/tests/generate_test_data.py
```

The script writes all `*.npy` files into `tests/test_data/` and prints the
shape and dtype of each.  The generated files are committed to the repository
so the C++ tests have no Python dependency at runtime.

Regenerate when the test vectors need to change (new shapes, new seed, new op
variant), then commit both the updated script and the new `.npy` files together.

### Adding a new op translator

1. Create `src/op/<name>.cpp`.  The translator signature is:
   ```cpp
   OutputVector translate_<name>(const NodeContext& context);
   ```
   Read inputs with `context.get_input(int)` and attributes with
   `context.get_attribute<T>("key")`.  Return the output(s) via
   `rename_outputs_with_suffix({res}, context.get_name())`.

2. Declare the translator in `src/op_table.hpp`:
   ```cpp
   GGUF_OP_CONVERTER(translate_<name>);
   ```

3. Register it in `src/op_table.cpp`:
   ```cpp
   {"GGML_OP_<NAME>", op::translate_<name>},
   ```

4. Add the new `.cpp` to the `FRONTEND_SRCS` list in `tests/CMakeLists.txt`
   so the test binary can compile without linking the full frontend shared lib.

### Adding a new op test

1. **Add a generator function** in `generate_test_data.py`:
   ```python
   def gen_my_op(out_dir, rng):
       x = rng.standard_normal((4, 8)).astype(np.float32)
       expected = my_op_numpy(x).astype(np.float32)
       save(out_dir, "my_op_input", x)
       save(out_dir, "my_op_expected", expected)
   ```
   Register it in `main()`:
   ```python
   generators = [
       ...
       ("my_op", gen_my_op),
   ]
   ```

2. **Regenerate test data**:
   ```bash
   python3 src/frontends/gguf/tests/generate_test_data.py
   ```

3. **Add a C++ test** in `test_ops.cpp`:
   ```cpp
   TEST(GGUFOps, MyOp) {
       ov::PartialShape shape;
       auto input_data = load_f32("my_op_input", &shape);
       auto expected   = load_f32("my_op_expected");

       OpTestGraph g;
       g.add_input("x", ov::element::f32, shape);
       g.add_op("GGML_OP_MY_OP",
                {"x"}, "y", shape, ov::element::f32,
                /*op_case=*/0,
                {{"x", shape}},
                {{"x", ov::element::f32}},
                {{"my_attr", ov::Any(1.0f)}});
       auto model = g.build();

       auto result = run_on_cpu(model, {make_f32_tensor(shape.to_shape(), input_data)});
       expect_near(result[0], expected, 1e-5f, "my_op");
   }
   ```

   Key rules:
   - **Input order** for `run_on_cpu` must match the `add_input()` call order
     (both in the test and in the generator); `OpTestGraph::build()` guarantees
     the model parameter order matches `add_input()`.
   - **`op_case`** encodes GGML's internal variant field; use 0 for the default
     path.  See individual translator sources for the meaning of non-zero cases.
   - **`attn_factor`** in `RopeConfig` must be set to `1.0f` for standard RoPE
     (defaults to 0, which would zero out all sin/cos outputs).

4. **Build and verify**:
   ```bash
   cmake --build build --target ov_gguf_frontend_tests
   ctest -R GGUF_FE
   ```

### FlashAttnExt: mask input naming

The translator's mask dispatch works as follows (see `flash_attn_ext.cpp`):

1. If the tensor map contains `"KQ_mask_sliced"` (or `"KQ_mask_swa_sliced"` for
   sliding-window attention), it is used directly.
2. Otherwise, the translator slices the raw mask using `q.shape[2]` as the token
   length — but in the GGUF-natural `[B, L, H, S]` layout `q.shape[2]` is `n_heads`,
   not `seq_q`, so the fallback Slice produces the wrong extent.

In a test, always name the mask input `"KQ_mask_sliced"` (and list it in the op's
`input_names`) so path 1 is taken:

```cpp
g.add_input("KQ_mask_sliced", ov::element::f32, mask_shape);
g.add_op("GGML_OP_FLASH_ATTN_EXT",
         {"q", "k", "v", "KQ_mask_sliced"}, ...);
```

### Relationship to llama.cpp test-backend-ops.cpp

`test-backend-ops.cpp` in llama.cpp tests ops by running them on a registered
`gguf_backend` and comparing against a CPU reference.  The OV frontend cannot
be registered as a `gguf_backend` without implementing the full backend API, so
the harness is not directly reusable.

What was extracted for the OV test suite:
- Shape conventions and parameter sweep sizes (e.g. `[1, seq, heads, head_dim]`
  for RoPE; matmul `[1,1,m,k] × [1,1,n,k]` with independent m/n/k).
- RoPE NORMAL vs NEOX split-halves convention.
- The `attn_factor` must be `1.0f` (llama.cpp uses the same default-zero issue
  when calling `gguf_rope_ext` without explicitly passing `attn_factor`).

What was intentionally not ported:
- `GGML_OP_MUL_MAT_ID` — the op is tested indirectly via the MoE flow; a
  standalone test requires non-trivial expert-selection scaffolding.
- Quantized-weight matmul — covered by `test_quant_dequant.cpp`.
- `GGML_OP_POOL_2D`, `GGML_OP_CONV_TRANSPOSE_1D` — not in `op_table.cpp`.
- Random-seed sweep (llama.cpp runs many random seeds; OV tests use a single
  committed seed 42 for reproducibility).
