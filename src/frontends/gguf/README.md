# GGUF Frontend (Proof of Concept)

A minimal OpenVINO Frontend that loads a [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
file and synthesizes an `ov::Model` matching the I/O contract used by
[`openvino.genai`](https://github.com/openvinotoolkit/openvino.genai)'s
`LLMPipeline`.

## Status: PoC

| Aspect | Coverage |
|---|---|
| Architectures      | `llama`, `qwen2` |
| Tensor types — verbatim                                    | `F32`, `F16`, `BF16` |
| Tensor types — preserved as native compressed Constants    | `Q4_0` / `Q4_1` / `Q4_K` (`u4` + per-group fp16 scale ± min), `Q5_0` / `Q6_K` / `Q8_0` (`i8` + per-group fp16 scale) |
| Tensor types — dequantized to FP16 at load                 | `Q5_1`, `Q2_K`, `Q3_K`, `Q5_K` |
| Tensor types — not implemented                             | `IQ*` (codebook), `TQ*` (ternary), `Q8_K` |
| GGUF version       | 3 |
| Tokenizer / chat template / generation config | **Not handled** — out of scope for `ov::Model`; consume from GGUF separately |

### Quantization handling

For the formats that map cleanly onto OpenVINO's low-precision element types,
the frontend preserves the on-disk compression by emitting the standard
NNCF / optimum-intel decompression pattern:

```
Constant(u4|i8, [out, in])
    -> Convert(f16)
    -> Reshape([out, in/G, G])             # group on the inner axis
    -> [Subtract(8 f16)]                   # Q4_0 only (implicit zero-point)
    -> Multiply(Constant(f16, [out, in/G, 1]))   # per-group scale
    -> [Add(Constant(f16, [out, in/G, 1]))]      # Q4_1 / Q4_K (per-group min)
    -> Reshape([out, in])
    -> Convert(f32)
```

Group size `G` is the natural sub-block size of each format: **32** for the
legacy block-32 quants (Q4_0 / Q4_1 / Q5_0 / Q8_0) and the K-quant Q4_K, and
**16** for Q6_K. Element type and zero-point handling per format:

| GGUF type | OV element | Group | ZP / Min |
|---|---|---:|---|
| Q4_0 | `u4` | 32 | implicit `Subtract(8)` |
| Q4_1 | `u4` | 32 | per-group fp16 `Add(min)` |
| Q5_0 | `i8` (range `[-16, 15]`) | 32 | none |
| Q8_0 | `i8` | 32 | none |
| Q4_K | `u4` | 32 | per-group fp16 `Add(min)` |
| Q6_K | `i8` (range `[-32, 31]`) | 16 | none |

For the K-quants (Q4_K, Q6_K), the GGUF super-block structure carries a
fp16 scale-of-scales `d` plus 6-bit (Q4_K) or int8 (Q6_K) per-sub-block
`sc`, and Q4_K additionally carries `dmin` + 6-bit `m`. We **flatten the
two-level scaling at load time**: each per-group fp16 scale becomes
`d * sc[g]`, and Q4_K's per-group min becomes `-(dmin * m[g])`. No
information is lost — these are algebraic identities of the ggml dequant
formulas.

Q5_0 has no native 5-bit OV element type, so we sign-extend its
`[-16, 15]` values into `i8`. This costs ~1.5 bits/elem vs the 5.5 bits/elem
on disk but keeps the chain identical to the other native paths and the
numerical result is exact (no rounding vs ggml).

CPU and GPU plugins recognize this pattern and keep the constant in
compressed form at runtime.

### What still dequantizes to FP16

- `Q5_1` — same i8 trick would work; not yet wired up.
- `Q2_K`, `Q3_K`, `Q5_K` — would require `u2` / `u3` / split-bit layouts;
  uncommon enough that the dequant→fp16 fallback is acceptable for now.

### IR sizes (SmolLM2-135M)

| GGUF                | Composition                                           | IR `.bin` | GGUF |
|---------------------|-------------------------------------------------------|---------:|-----:|
| F16                 | F16                                                   | 256.6 MB | 258.6 MB |
| Q8_0                | Q8_0                                                  | 136.4 MB | 138.1 MB |
| Q4_0                | Q4_0                                                  |  85.8 MB |  87.5 MB |
| Q4_K_M              | Q5_0 52% + Q8_0 31% + Q6_K 10% + Q4_K 8% (+ F32 norms) | **131.2 MB** | 100.5 MB |

Note the "Q4_K_M" file is mostly Q5_0/Q8_0, not K-quants — `llama.cpp`
reverts to legacy block-32 quants for tensors whose contiguous dim is not
a multiple of 256 (which is most of the tensors in a 576-hidden model).

Files:
- `src/gguf_compress.{hpp,cpp}` — native-compression builder (Q4_0 / Q4_1 / Q5_0 / Q8_0 / Q4_K / Q6_K).
- `src/gguf_dequant.{hpp,cpp}`  — reference dequantizers ported from
  [ggml-quants.c](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c),
  used for the formats that don't map to a native OV element type.

Files:
- `src/gguf_compress.{hpp,cpp}` — native-compression builder (Q4_0/Q4_1/Q8_0).
- `src/gguf_dequant.{hpp,cpp}`  — reference dequantizers ported from
  [ggml-quants.c](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c),
  used for the formats that don't map to a native OV element type.

> See `tools/gguf_frontend_poc/README.md` at the repo root for the broader
> "should this live in OpenVINO core or in `openvino.genai`?" discussion.

## Usage

Once `ENABLE_OV_GGUF_FRONTEND=ON` (default), the frontend is auto-discovered
by `ov::Core::read_model`:

```cpp
#include <openvino/openvino.hpp>

ov::Core core;
auto model = core.read_model("SmolLM2-135M-Instruct.f16.gguf");
ov::serialize(model, "openvino_model.xml");
```

```python
import openvino as ov
core = ov.Core()
model = core.read_model("SmolLM2-135M-Instruct.f16.gguf")
ov.save_model(model, "openvino_model.xml")
```

## Produced model contract

| Name             | Type | Shape          | Notes |
|------------------|------|----------------|-------|
| `input_ids`      | i64  | `[?, ?]`       | batch, seq |
| `attention_mask` | i64  | `[?, ?]`       | currently unused (PoC) |
| `position_ids`   | i64  | `[?, ?]`       | for RoPE |
| `beam_idx`       | i32  | `[?]`          | batch-reorder for beam search |
| `logits`         | f32  | `[?, ?, vocab]`| output |

KV-cache state variables per layer follow the openvino.genai naming convention:

```
past_key_values.{i}.keypresent.{i}.key
past_key_values.{i}.valuepresent.{i}.key
```

## Layout

```
src/frontends/gguf/
├── CMakeLists.txt
├── README.md
├── include/openvino/frontend/gguf/
│   ├── frontend.hpp          # public FrontEnd class
│   └── visibility.hpp
└── src/
    ├── CMakeLists.txt        # ov_add_frontend(...)
    ├── gguf.cpp              # C plugin entry (get_front_end_data / get_api_version)
    ├── frontend.cpp          # supported_impl / load_impl / convert
    ├── input_model.hpp       # holds parsed GGUFFile
    ├── gguf_reader.hpp       # header-only GGUF v3 binary parser (mmap-based)
    ├── graph_builder.hpp
    └── graph_builder.cpp     # llama / qwen2 graph synthesis
```

## Disabling

```
cmake -DENABLE_OV_GGUF_FRONTEND=OFF ...
```
