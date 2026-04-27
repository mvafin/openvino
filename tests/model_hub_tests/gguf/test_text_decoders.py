# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Universal end-to-end correlation test for the GGUF frontend on text decoders.

This single parametrized test loads a GGUF model through the OpenVINO GGUF
frontend, runs a single prefill on a fixed prompt, and compares the
last-token logits against the matching Hugging Face reference loaded in
fp32. Every architecture supported by the frontend should be representable
as a row in :data:`MODELS` and exercised by the same code path.

Coverage philosophy:
    * **Precommit**: one tiny model per arch family that downloads quickly
      from a warm cache and runs in <30 s on CPU.
    * **Nightly**: heavier models, MoE, or quantizations that exercise
      additional decompression paths.

The test asserts:
    * The synthesized I/O contract is the canonical
      ``input_ids / attention_mask / position_ids / beam_idx -> logits``.
    * Compile on CPU succeeds.
    * Last-token logits match the HF fp32 reference (Pearson correlation
      above a per-model floor, plus argmax agreement when requested).

When a model is **expected to fail** (e.g. an arch whose graph builder is
not yet implemented), tag it with :data:`pytest.mark.xfail` via the
``xfail_reason`` field instead of removing it from the table — this keeps
the row in plain sight and the test will start passing automatically once
support lands.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pytest

try:
    import openvino as ov
except ImportError:  # pragma: no cover
    ov = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
except ImportError:
    torch = None
    AutoTokenizer = AutoModelForCausalLM = None
    hf_hub_download = None


pytestmark = pytest.mark.skipif(
    ov is None or torch is None or hf_hub_download is None,
    reason="GGUF text-decoder tests require openvino, torch, transformers, huggingface_hub",
)


# ---------------------------------------------------------------------------
# Model table
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GGUFTextModel:
    """One row in the universal text-decoder coverage matrix."""

    test_id: str                  # pytest id (also used in skip/xfail messages)
    gguf_repo: str                # HF repo id of the GGUF
    gguf_file: str                # filename within the GGUF repo
    hf_ref_repo: str              # HF repo id for the fp32 reference model
    pytest_mark: str = "precommit"  # one of: "precommit", "nightly"
    corr_floor: float = 0.985     # Pearson floor on last-token logits
    require_argmax_match: bool = True
    xfail_reason: Optional[str] = None
    prompt: str = "The capital of France is"


# Keep this list short and high-signal: one row per architecture family.
# Adding a new architecture to the GGUF frontend should require *only*
# adding a row here.
MODELS: list[GGUFTextModel] = [
    # --- llama-family -----------------------------------------------------
    # TinyLlama-1.1B-Chat Q4_K_M, ~669 MiB. Fast, stable, well-known logits.
    GGUFTextModel(
        test_id="tinyllama_1.1b_chat_q4_k_m",
        gguf_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        gguf_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        hf_ref_repo="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        pytest_mark="precommit",
    ),

    # --- qwen3moe ---------------------------------------------------------
    # Smallest publicly-hosted GGUF that genuinely declares
    # general.architecture = "qwen3moe" (28 layers, 2 experts, top-k=2).
    # The C++ graph builder emits the decomposed MoE subgraph that
    # `ov::pass::FuseMOEExperts` is expected to fold into the optimized
    # `ov::op::internal::MOE` op.
    #
    # Quant choice: Q4_K_M, not Q2_K. Q2_K has a cos-similarity ceiling
    # near 0.66 vs the fp32 reference for this 0.9B-A0.6B model — that is
    # the *quantization precision floor*, not a frontend defect (verified
    # by running the same dequantized weights through HF in fp32, which
    # produced corr=0.6599 — within 0.5% of the OV result).
    GGUFTextModel(
        test_id="qwen3moe_0.9b_a0.6b_q4_k_m",
        gguf_repo="mradermacher/Qwen3-0.9B-A0.6B-GGUF",
        gguf_file="Qwen3-0.9B-A0.6B.Q4_K_M.gguf",
        hf_ref_repo="beyoru/Qwen3-0.9B-A0.6B",
        pytest_mark="nightly",
        corr_floor=0.95,
        require_argmax_match=True,
    ),
]


def _ids_for(models):
    return [m.test_id for m in models]


def _params_for(models):
    """Build pytest.param rows so per-model markers are visible at collection time.

    Marks attached dynamically inside the test body (via ``request.applymarker``)
    are invisible to the ``-m`` filter, so the marker must come from
    ``pytest.param(..., marks=...)`` instead.
    """
    out = []
    for m in models:
        marks = [getattr(pytest.mark, m.pytest_mark)]
        if m.xfail_reason:
            marks.append(pytest.mark.xfail(reason=m.xfail_reason, strict=False))
        out.append(pytest.param(m, id=m.test_id, marks=marks))
    return out


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ov_core():
    return ov.Core()


def _run_ov_text_decoder(compiled, input_ids: np.ndarray) -> np.ndarray:
    """Run a single prefill of a stateful GGUF text decoder.

    Returns logits of shape ``[T, vocab]``.
    """
    T = input_ids.shape[1]
    feeds = {
        "input_ids":      input_ids.astype(np.int64),
        "attention_mask": np.ones((1, T), dtype=np.int64),
        "position_ids":   np.arange(T, dtype=np.int64)[None, :],
        "beam_idx":       np.zeros((1,), dtype=np.int32),
    }
    out = compiled.create_infer_request().infer(feeds)
    return list(out.values())[0][0]   # drop batch -> [T, V]


def _has_fused_moe(model) -> bool:
    """True if the compiled graph contains the fused MoE op.

    Exposed as a helper so individual tests can assert that the
    FuseMOEExperts transformation actually fired on MoE archs (a regression
    here would silently fall back to the slow per-expert path).
    """
    for op in model.get_ops():
        if op.get_type_name() == "MOE":
            return True
    return False


# ---------------------------------------------------------------------------
# The universal test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_spec", _params_for(MODELS))
def test_text_decoder_matches_hf_reference(model_spec: GGUFTextModel, ov_core):
    """OV(GGUF) last-token logits must match the HF fp32 reference."""

    gguf_path = hf_hub_download(repo_id=model_spec.gguf_repo,
                                filename=model_spec.gguf_file)

    ov_model = ov_core.read_model(gguf_path)

    # I/O contract is the same for all supported text decoders.
    in_names = sorted(p.any_name for p in ov_model.inputs)
    assert in_names == ["attention_mask", "beam_idx", "input_ids", "position_ids"], in_names
    assert len(ov_model.outputs) == 1
    assert ov_model.outputs[0].any_name == "logits"

    compiled = ov_core.compile_model(ov_model, "CPU")

    tok = AutoTokenizer.from_pretrained(model_spec.hf_ref_repo, trust_remote_code=True)
    ids = tok(model_spec.prompt, return_tensors="np").input_ids  # [1, T]
    ov_logits = _run_ov_text_decoder(compiled, ids)              # [T, V]
    ov_last = ov_logits[-1].astype(np.float64)
    assert np.isfinite(ov_last).all(), "OV logits contain non-finite values"

    hf = AutoModelForCausalLM.from_pretrained(
        model_spec.hf_ref_repo, torch_dtype=torch.float32, trust_remote_code=True
    ).eval()
    with torch.no_grad():
        hf_out = hf(input_ids=torch.from_numpy(ids),
                    attention_mask=torch.ones_like(torch.from_numpy(ids)))
    hf_last = hf_out.logits[0, -1].float().cpu().numpy().astype(np.float64)

    corr = float(np.corrcoef(ov_last, hf_last)[0, 1])
    assert corr >= model_spec.corr_floor, (
        f"[{model_spec.test_id}] last-token logits correlation {corr:.4f} "
        f"below floor {model_spec.corr_floor}"
    )
    if model_spec.require_argmax_match:
        ov_top1 = int(ov_last.argmax())
        hf_top1 = int(hf_last.argmax())
        assert ov_top1 == hf_top1, (
            f"[{model_spec.test_id}] top-1 mismatch: "
            f"OV={tok.decode([ov_top1])!r} HF={tok.decode([hf_top1])!r}"
        )


# ---------------------------------------------------------------------------
# Per-arch invariants that don't need the HF reference
# ---------------------------------------------------------------------------

@pytest.mark.nightly
@pytest.mark.xfail(
    reason="FuseMOEExperts pattern does not yet match the subgraph emitted "
           "by the GGUF qwen3moe builder. Numerical correctness is verified "
           "by test_text_decoder_matches_hf_reference[qwen3moe_*]; this is "
           "a separate optimization path (fusion to ov::op::internal::MOE).",
    strict=False,
)
def test_qwen3moe_uses_fused_moe_op():
    """The qwen3moe model's compiled graph should contain the fused ``MOE``
    op produced by ``ov::pass::FuseMOEExperts`` — a fallback to the
    per-expert decomposed path silently regresses performance.

    ``FuseMOE`` runs in ``MOCTransformations`` which is invoked during
    ``compile_model``, so the assertion is made on the compiled runtime
    model, not on the raw graph returned by ``read_model``.
    """
    spec = next((m for m in MODELS if m.test_id.startswith("qwen3moe_")), None)
    assert spec is not None, "qwen3moe row missing from MODELS"

    gguf_path = hf_hub_download(repo_id=spec.gguf_repo, filename=spec.gguf_file)
    core = ov.Core()
    model = core.read_model(gguf_path)
    compiled = core.compile_model(model, "CPU")
    runtime = compiled.get_runtime_model()
    assert _has_fused_moe(runtime), (
        "FuseMOEExperts did not fire on qwen3moe — the GGUF frontend likely "
        "emitted a pattern that does not match the canonical fixture."
    )
