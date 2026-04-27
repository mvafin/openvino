# GGUF model-hub tests

End-to-end tests for the OpenVINO GGUF frontend on real Hugging Face GGUF
artefacts. The tests are kept narrow (a small number of representative
models) and use the `huggingface_hub` cache so they are reusable from a
warm cache without re-downloading.

## Layout

* `test_qwen3_vl.py` — Qwen3-VL-2B-Instruct (Q4_K_M text decoder + F16
  mmproj vision encoder). Validates:
  - text-decoder logits vs. Hugging Face fp32 reference (precommit),
  - mmproj loads and produces finite outputs of the expected shapes
    (precommit),
  - end-to-end image+text orchestration smoke test (nightly).

## Running

```bash
pip install -r tests/model_hub_tests/gguf/requirements.txt
pytest tests/model_hub_tests/gguf -m precommit -v
```

The precommit subset downloads ~1.8 GiB of GGUF weights from Hugging Face
plus the ~5 GiB fp32 reference checkpoint; expect a slow first run.
Subsequent runs reuse the HF cache.

## Memory notes

On CPUs without AVX512_BF16/AVX512_FP16 support (e.g. 12th-gen Core), the
CPU plugin will materialize the Q4_K/Q6_K weights as fp32 at compile
time, inflating peak RSS to ~13 GB during `compile_model`. This is a
property of the CPU plugin's kernel coverage on the host, not of the
GGUF frontend (the in-graph constants total ~1.15 GB, matching the
GGUF file size).
