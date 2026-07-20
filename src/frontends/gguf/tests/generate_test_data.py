#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Generates pre-calculated reference data for GGUF frontend per-op tests.
#
# Run once (or whenever test vectors need to change), then commit the resulting
# .npy files under tests/test_data/.  The C++ test suite loads these files so
# it never needs a formula dependency at run time.
#
# Usage:
#   python3 generate_test_data.py
#   python3 generate_test_data.py --out-dir path/to/test_data
#
# Requirements: numpy only (no llama.cpp / torch needed).

import argparse
import math
import os

import numpy as np


def save(out_dir: str, name: str, arr: np.ndarray) -> None:
    path = os.path.join(out_dir, name + ".npy")
    np.save(path, arr)
    print(f"  {name}.npy  {arr.shape}  {arr.dtype}")


# ── helpers ──────────────────────────────────────────────────────────────────

def rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.vectorize(
        lambda v: math.erf(float(v) / math.sqrt(2.0)))(x))


def rope_normal(x: np.ndarray, positions: np.ndarray,
                freq_base: float = 10000.0, freq_scale: float = 1.0,
                attn_factor: float = 1.0) -> np.ndarray:
    """Rotate consecutive pairs (NORMAL mode).
    x: [1, seq, heads, head_dim]
    positions: [1, 1, 1, seq] int32
    """
    seq, heads, d = x.shape[1], x.shape[2], x.shape[3]
    pos = positions.flatten().astype(np.float64)
    half = d // 2
    inv_freq = 1.0 / (freq_base ** (np.arange(0, half, dtype=np.float64) * 2.0 / d))
    theta = (pos[:, np.newaxis] * inv_freq[np.newaxis, :]) * freq_scale  # [seq, half]
    cos = np.cos(theta) * attn_factor
    sin = np.sin(theta) * attn_factor
    cos = cos[:, np.newaxis, :]  # [seq, 1, half]
    sin = sin[:, np.newaxis, :]
    x0 = x[0, :, :, 0::2]  # [seq, heads, half]
    x1 = x[0, :, :, 1::2]
    out_even = x0 * cos - x1 * sin
    out_odd  = x0 * sin + x1 * cos
    out = np.stack([out_even, out_odd], axis=-1).reshape(seq, heads, d)
    return out[np.newaxis]


def rope_neox(x: np.ndarray, positions: np.ndarray,
              freq_base: float = 10000.0, freq_scale: float = 1.0,
              attn_factor: float = 1.0) -> np.ndarray:
    """Rotate split-halves (NEOX mode).
    x: [1, seq, heads, head_dim]
    """
    seq, heads, d = x.shape[1], x.shape[2], x.shape[3]
    pos = positions.flatten().astype(np.float64)
    half = d // 2
    inv_freq = 1.0 / (freq_base ** (np.arange(0, half, dtype=np.float64) * 2.0 / d))
    theta = (pos[:, np.newaxis] * inv_freq[np.newaxis, :]) * freq_scale  # [seq, half]
    cos = np.cos(theta) * attn_factor  # [seq, half]
    sin = np.sin(theta) * attn_factor
    cos = cos[:, np.newaxis, :]  # [seq, 1, half]
    sin = sin[:, np.newaxis, :]
    x0 = x[0, :, :, :half]   # [seq, heads, half] — first half
    x1 = x[0, :, :, half:]   # second half
    out_first  = x0 * cos - x1 * sin
    out_second = x0 * sin + x1 * cos
    out = np.concatenate([out_first, out_second], axis=-1)  # [seq, heads, d]
    return out[np.newaxis]


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return silu(gate) * up


def geglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    return gelu(gate) * up


def swiglu_oai(gate: np.ndarray, up: np.ndarray,
               alpha: float = 1.702, limit: float = 7.0) -> np.ndarray:
    x = np.minimum(gate, limit)
    y = np.clip(up, -limit, limit)
    g = x / (1.0 + np.exp(-alpha * x))
    return g * (y + 1.0)


# ── generators ───────────────────────────────────────────────────────────────

def gen_rms_norm(out_dir, rng):
    x = rng.standard_normal((2, 16)).astype(np.float32)
    save(out_dir, "rms_norm_input", x)
    save(out_dir, "rms_norm_expected", rms_norm(x, 1e-5).astype(np.float32))


def gen_add(out_dir, rng):
    a = rng.standard_normal((2, 4, 8)).astype(np.float32)
    b = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "add_input_a", a)
    save(out_dir, "add_input_b", b)
    save(out_dir, "add_expected", (a + b).astype(np.float32))


def gen_sub(out_dir, rng):
    a = rng.standard_normal((2, 4, 8)).astype(np.float32)
    b = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "sub_input_a", a)
    save(out_dir, "sub_input_b", b)
    save(out_dir, "sub_expected", (a - b).astype(np.float32))


def gen_mul(out_dir, rng):
    a = rng.standard_normal((2, 4, 8)).astype(np.float32)
    b = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "mul_input_a", a)
    save(out_dir, "mul_input_b", b)
    save(out_dir, "mul_expected", (a * b).astype(np.float32))


def gen_div(out_dir, rng):
    a = rng.standard_normal((2, 4, 8)).astype(np.float32)
    b = (rng.standard_normal((2, 4, 8)) + 2.0).astype(np.float32)  # avoid div by ~0
    save(out_dir, "div_input_a", a)
    save(out_dir, "div_input_b", b)
    save(out_dir, "div_expected", (a / b).astype(np.float32))


def gen_scale(out_dir, rng):
    x = rng.standard_normal((4, 8)).astype(np.float32)
    s, bias = 0.5, 1.0
    save(out_dir, "scale_input", x)
    save(out_dir, "scale_expected", (x * s + bias).astype(np.float32))
    save(out_dir, "scale_param_scale", np.array(s, dtype=np.float32))
    save(out_dir, "scale_param_bias",  np.array(bias, dtype=np.float32))
    # bias=0 variant
    save(out_dir, "scale_nobias_expected", (x * s).astype(np.float32))


def gen_softmax(out_dir, rng):
    x = rng.standard_normal((1, 1, 4, 8)).astype(np.float32)
    scale = 0.125
    y = softmax(x * scale, axis=-1).astype(np.float32)
    save(out_dir, "softmax_input", x)
    save(out_dir, "softmax_expected", y)
    save(out_dir, "softmax_param_scale", np.array(scale, dtype=np.float32))


def gen_silu(out_dir, rng):
    x = rng.standard_normal((4, 8)).astype(np.float32)
    save(out_dir, "silu_input", x)
    save(out_dir, "silu_expected", silu(x).astype(np.float32))


def gen_gelu(out_dir, rng):
    x = rng.standard_normal((4, 8)).astype(np.float32)
    save(out_dir, "gelu_input", x)
    save(out_dir, "gelu_expected", gelu(x).astype(np.float32))


def gen_mul_mat(out_dir, rng):
    # A[1,1,m,k]  B[1,1,n,k]  → out[1,1,m,n]  (MatMul(A, B, false, true))
    m, n, k = 3, 5, 7
    A = rng.standard_normal((1, 1, m, k)).astype(np.float32)
    B = rng.standard_normal((1, 1, n, k)).astype(np.float32)
    expected = (A @ B.swapaxes(-1, -2)).astype(np.float32)
    save(out_dir, "mul_mat_input_a", A)
    save(out_dir, "mul_mat_input_b", B)
    save(out_dir, "mul_mat_expected", expected)


def gen_get_rows(out_dir, rng):
    vocab, emb = 16, 8
    weight = rng.standard_normal((vocab, emb)).astype(np.float32)
    indices = rng.integers(0, vocab, size=(1, 1, 1, 4)).astype(np.int32)
    expected = weight[indices.flatten()][np.newaxis, np.newaxis].astype(np.float32)
    save(out_dir, "get_rows_weight", weight)
    save(out_dir, "get_rows_indices", indices)
    save(out_dir, "get_rows_expected", expected)


def gen_rope_normal(out_dir, rng):
    seq, heads, head_dim = 4, 2, 8
    freq_base = 10000.0
    x = rng.standard_normal((1, seq, heads, head_dim)).astype(np.float32)
    positions = np.arange(seq, dtype=np.int32).reshape(1, 1, 1, seq)
    expected = rope_normal(x, positions, freq_base, attn_factor=1.0).astype(np.float32)
    save(out_dir, "rope_normal_input", x)
    save(out_dir, "rope_normal_positions", positions)
    save(out_dir, "rope_normal_expected", expected)
    save(out_dir, "rope_normal_param_freq_base", np.array(freq_base, dtype=np.float32))
    save(out_dir, "rope_normal_param_head_dim",  np.array(head_dim, dtype=np.int32))


def gen_rope_neox(out_dir, rng):
    # NEOX (split-halves) — common for Qwen2/Falcon
    seq, heads, head_dim = 4, 2, 8
    freq_base = 10000.0
    x = rng.standard_normal((1, seq, heads, head_dim)).astype(np.float32)
    positions = np.arange(seq, dtype=np.int32).reshape(1, 1, 1, seq)
    expected = rope_neox(x, positions, freq_base, attn_factor=1.0).astype(np.float32)
    save(out_dir, "rope_neox_input", x)
    save(out_dir, "rope_neox_positions", positions)
    save(out_dir, "rope_neox_expected", expected)
    save(out_dir, "rope_neox_param_freq_base", np.array(freq_base, dtype=np.float32))
    save(out_dir, "rope_neox_param_head_dim",  np.array(head_dim, dtype=np.int32))


def gen_transpose(out_dir, rng):
    # translate_transpose swaps last two dims: perm {0,1,3,2}
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    save(out_dir, "transpose_input", x)
    save(out_dir, "transpose_expected", x.transpose(0, 1, 3, 2).astype(np.float32))


def gen_permute(out_dir, rng):
    # translate_permute op_case=1 — perm {0,2,1,3} (swap heads and seq)
    x = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    save(out_dir, "permute_input", x)
    save(out_dir, "permute_expected", x.transpose(0, 2, 1, 3).astype(np.float32))


def gen_cpy(out_dir, rng):
    # translate_cpy is a Convert — same type is a no-op cast; test f32→f32
    x = rng.standard_normal((4, 8)).astype(np.float32)
    save(out_dir, "cpy_input", x)
    save(out_dir, "cpy_expected", x.copy().astype(np.float32))


def gen_cont(out_dir, rng):
    # translate_cont op_case=2 (from TRANSPOSE): returns input as-is (pass-through)
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    save(out_dir, "cont_input", x)
    save(out_dir, "cont_expected", x.copy().astype(np.float32))


def gen_reshape(out_dir, rng):
    # translate_reshape op_case=6: full output-shape reshape
    # Input [2,3,4,5] -> output [1,6,4,5]
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    expected = x.reshape(1, 6, 4, 5).astype(np.float32)
    save(out_dir, "reshape_input", x)
    save(out_dir, "reshape_expected", expected)


def gen_view(out_dir, rng):
    # translate_view default op_case=0: pass-through
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    save(out_dir, "view_input", x)
    save(out_dir, "view_expected", x.copy().astype(np.float32))


def gen_swiglu(out_dir, rng):
    # 1-input form: split last axis in half, silu(gate)*up
    combined = rng.standard_normal((2, 3, 16)).astype(np.float32)
    gate = combined[..., :8]
    up   = combined[..., 8:]
    expected = swiglu(gate, up).astype(np.float32)
    save(out_dir, "swiglu_input", combined)
    save(out_dir, "swiglu_expected", expected)


def gen_geglu(out_dir, rng):
    combined = rng.standard_normal((2, 3, 16)).astype(np.float32)
    gate = combined[..., :8]
    up   = combined[..., 8:]
    expected = geglu(gate, up).astype(np.float32)
    save(out_dir, "geglu_input", combined)
    save(out_dir, "geglu_expected", expected)


def gen_swiglu_oai(out_dir, rng):
    combined = rng.standard_normal((2, 3, 16)).astype(np.float32)
    gate = combined[..., :8]
    up   = combined[..., 8:]
    alpha, limit = 1.702, 7.0
    expected = swiglu_oai(gate, up, alpha, limit).astype(np.float32)
    save(out_dir, "swiglu_oai_input", combined)
    save(out_dir, "swiglu_oai_expected", expected)
    save(out_dir, "swiglu_oai_param_alpha", np.array(alpha, dtype=np.float32))
    save(out_dir, "swiglu_oai_param_limit", np.array(limit, dtype=np.float32))


def gen_argsort(out_dir, rng):
    # translate_argsort: descending argsort over last axis
    x = rng.standard_normal((3, 2, 8)).astype(np.float32)
    # numpy argsort ascending; reverse for descending
    expected = np.argsort(-x, axis=-1).astype(np.int32)
    save(out_dir, "argsort_input", x)
    save(out_dir, "argsort_expected", expected)


def gen_top_k(out_dir, rng):
    # translate_top_k: top-k indices (descending) over last axis
    x = rng.standard_normal((3, 2, 8)).astype(np.float32)
    k = 3
    expected = np.argsort(-x, axis=-1)[..., :k].astype(np.int32)
    save(out_dir, "top_k_input", x)
    save(out_dir, "top_k_expected", expected)
    save(out_dir, "top_k_param_k", np.array(k, dtype=np.int32))


def gen_sum_rows(out_dir, rng):
    x = rng.standard_normal((2, 3, 4, 8)).astype(np.float32)
    expected = x.sum(axis=-1, keepdims=True).astype(np.float32)
    save(out_dir, "sum_rows_input", x)
    save(out_dir, "sum_rows_expected", expected)


def gen_set_rows(out_dir, rng):
    # translate_set_rows non-stateful: ScatterUpdate(dst, indices, data) along axis 2
    # dst [1,1,8,4], data [1,1,3,4], indices flat [3]
    dst   = rng.standard_normal((1, 1, 8, 4)).astype(np.float32)
    data  = rng.standard_normal((1, 1, 3, 4)).astype(np.float32)
    indices = np.array([0, 3, 5], dtype=np.int32).reshape(1, 1, 1, 3)
    # ScatterUpdate: dst[..., indices[i], :] = data[..., i, :]
    expected = dst.copy()
    for i, idx in enumerate(indices.flatten()):
        expected[0, 0, idx, :] = data[0, 0, i, :]
    save(out_dir, "set_rows_dst",      dst)
    save(out_dir, "set_rows_data",     data)
    save(out_dir, "set_rows_indices",  indices)
    save(out_dir, "set_rows_expected", expected.astype(np.float32))


def gen_flash_attn_ext(out_dir, rng):
    # Basic single-head, no mask, no GQA, no sinks
    # Q/K/V: [1, seq_q, heads, head_dim]  (GGML-natural layout [B,L,H,S])
    batch, seq_q, seq_k, heads, hd = 1, 4, 6, 2, 8
    scale = 1.0 / math.sqrt(hd)
    Q = rng.standard_normal((batch, seq_q, heads, hd)).astype(np.float32)
    K = rng.standard_normal((batch, seq_k, heads, hd)).astype(np.float32)
    V = rng.standard_normal((batch, seq_k, heads, hd)).astype(np.float32)
    # Build causal-like mask [1,1,seq_q,seq_k] (f16 zeros = no masking)
    mask = np.zeros((batch, 1, seq_q, seq_k), dtype=np.float32)

    # Reference SDPA in BHLS layout
    # Q/K/V -> [B, H, L, S]
    q_t = Q.transpose(0, 2, 1, 3)  # [1, 2, 4, 8]
    k_t = K.transpose(0, 2, 1, 3)  # [1, 2, 6, 8]
    v_t = V.transpose(0, 2, 1, 3)
    scores = (q_t @ k_t.swapaxes(-1, -2)) * scale + mask  # [1, 2, 4, 6]
    weights = softmax(scores.astype(np.float64), axis=-1).astype(np.float32)
    out_t = weights @ v_t  # [1, 2, 4, 8]
    out = out_t.transpose(0, 2, 1, 3)  # back to [B, L, H, S]

    save(out_dir, "flash_attn_input_q", Q)
    save(out_dir, "flash_attn_input_k", K)
    save(out_dir, "flash_attn_input_v", V)
    save(out_dir, "flash_attn_input_mask", mask.astype(np.float32))
    save(out_dir, "flash_attn_expected", out)
    save(out_dir, "flash_attn_param_scale", np.array(scale, dtype=np.float32))


def gen_add_id(out_dir, rng):
    # translate_add_id: a [1,T,K,n] + gather(b [n_expert,n], ids [T,K]) -> [1,T,K,n]
    T, K, n, n_expert = 4, 2, 8, 6
    a = rng.standard_normal((1, T, K, n)).astype(np.float32)
    b = rng.standard_normal((n_expert, n)).astype(np.float32)
    ids = rng.integers(0, n_expert, size=(T, K)).astype(np.int32)
    # gather b[ids]: [T, K, n]
    b_gathered = b[ids]               # [T, K, n]
    expected = (a + b_gathered[np.newaxis]).astype(np.float32)
    save(out_dir, "add_id_input_a",    a)
    save(out_dir, "add_id_input_b",    b)
    save(out_dir, "add_id_input_ids",  ids.reshape(1, T, K))
    save(out_dir, "add_id_expected",   expected)


def gen_relu(out_dir, rng):
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "relu_input", x)
    save(out_dir, "relu_expected", np.maximum(x, 0.0).astype(np.float32))


def gen_tanh(out_dir, rng):
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "tanh_input", x)
    save(out_dir, "tanh_expected", np.tanh(x).astype(np.float32))


def gen_sigmoid(out_dir, rng):
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "sigmoid_input", x)
    save(out_dir, "sigmoid_expected", (1.0 / (1.0 + np.exp(-x))).astype(np.float32))


def gen_elu(out_dir, rng):
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    alpha = 1.0
    save(out_dir, "elu_input", x)
    save(out_dir, "elu_expected", np.where(x >= 0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32))


def gen_clamp(out_dir, rng):
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    lo, hi = -0.5, 0.5
    save(out_dir, "clamp_input", x)
    save(out_dir, "clamp_expected", np.clip(x, lo, hi).astype(np.float32))
    save(out_dir, "clamp_param_min", np.array(lo, dtype=np.float32))
    save(out_dir, "clamp_param_max", np.array(hi, dtype=np.float32))


def gen_concat(out_dir, rng):
    # concatenate along axis 1 (OV axis 1 = GGML dim 2 for 4D): [2,3,4,8] + [2,5,4,8]
    a = rng.standard_normal((2, 3, 4, 8)).astype(np.float32)
    b = rng.standard_normal((2, 5, 4, 8)).astype(np.float32)
    save(out_dir, "concat_input_a", a)
    save(out_dir, "concat_input_b", b)
    save(out_dir, "concat_expected", np.concatenate([a, b], axis=1).astype(np.float32))


def gen_norm(out_dir, rng):
    # LayerNorm: mean subtraction + variance normalization over last axis.
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    eps = 1e-5
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    save(out_dir, "norm_input", x)
    save(out_dir, "norm_expected", y.astype(np.float32))


def gen_sqr(out_dir, rng):
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "sqr_input", x)
    save(out_dir, "sqr_expected", (x ** 2).astype(np.float32))


def gen_sqrt(out_dir, rng):
    x = np.abs(rng.standard_normal((2, 4, 8))).astype(np.float32) + 0.01
    save(out_dir, "sqrt_input", x)
    save(out_dir, "sqrt_expected", np.sqrt(x).astype(np.float32))


def gen_log(out_dir, rng):
    x = np.abs(rng.standard_normal((2, 4, 8))).astype(np.float32) + 0.1
    save(out_dir, "log_input", x)
    save(out_dir, "log_expected", np.log(x).astype(np.float32))


def gen_sin(out_dir, rng):
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "sin_input", x)
    save(out_dir, "sin_expected", np.sin(x).astype(np.float32))


def gen_cos(out_dir, rng):
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save(out_dir, "cos_input", x)
    save(out_dir, "cos_expected", np.cos(x).astype(np.float32))


def gen_gelu_quick(out_dir, rng):
    x = rng.standard_normal((4, 8)).astype(np.float32)
    k = math.sqrt(2.0 / math.pi)
    y = 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x ** 3)))
    save(out_dir, "gelu_quick_input", x)
    save(out_dir, "gelu_quick_expected", y.astype(np.float32))


def gen_repeat(out_dir, rng):
    # Tile [2,1,4,8] → [2,3,4,8] (repeat dim 1 ×3).
    x = rng.standard_normal((2, 1, 4, 8)).astype(np.float32)
    save(out_dir, "repeat_input", x)
    save(out_dir, "repeat_expected", np.tile(x, (1, 3, 1, 1)).astype(np.float32))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GGUF frontend op test data")
    parser.add_argument("--out-dir",
                        default=os.path.join(os.path.dirname(__file__), "test_data"),
                        help="Directory to write .npy files into")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    generators = [
        ("rms_norm",      gen_rms_norm),
        ("add",           gen_add),
        ("sub",           gen_sub),
        ("mul",           gen_mul),
        ("div",           gen_div),
        ("scale",         gen_scale),
        ("softmax",       gen_softmax),
        ("silu",          gen_silu),
        ("gelu",          gen_gelu),
        ("mul_mat",       gen_mul_mat),
        ("get_rows",      gen_get_rows),
        ("rope_normal",   gen_rope_normal),
        ("rope_neox",     gen_rope_neox),
        ("transpose",     gen_transpose),
        ("permute",       gen_permute),
        ("cpy",           gen_cpy),
        ("cont",          gen_cont),
        ("reshape",       gen_reshape),
        ("view",          gen_view),
        ("swiglu",        gen_swiglu),
        ("geglu",         gen_geglu),
        ("swiglu_oai",    gen_swiglu_oai),
        ("argsort",       gen_argsort),
        ("top_k",         gen_top_k),
        ("sum_rows",      gen_sum_rows),
        ("set_rows",      gen_set_rows),
        ("flash_attn_ext",gen_flash_attn_ext),
        ("add_id",        gen_add_id),
        ("relu",          gen_relu),
        ("tanh",          gen_tanh),
        ("sigmoid",       gen_sigmoid),
        ("elu",           gen_elu),
        ("clamp",         gen_clamp),
        ("concat",        gen_concat),
        ("norm",          gen_norm),
        ("sqr",           gen_sqr),
        ("sqrt",          gen_sqrt),
        ("log",           gen_log),
        ("sin",           gen_sin),
        ("cos",           gen_cos),
        ("gelu_quick",    gen_gelu_quick),
        ("repeat",        gen_repeat),
    ]

    for name, fn in generators:
        print(f"[{name}]")
        fn(args.out_dir, rng)

    print(f"\nWrote test data to: {args.out_dir}")


if __name__ == "__main__":
    main()
