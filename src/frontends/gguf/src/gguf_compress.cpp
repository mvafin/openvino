// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_compress.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "gguf_dequant.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

using ov::op::v0::Constant;
using ov::op::v0::Convert;
using ov::op::v1::Add;
using ov::op::v1::Multiply;
using ov::op::v1::Reshape;
using ov::op::v1::Subtract;

// ---- ggml block geometry ------------------------------------------------ //

struct BlockGeom {
    int qk = 0;
    size_t bytes = 0;
};

BlockGeom geom(ggml_type t) {
    switch (t) {
    case GGML_TYPE_F32:
        return {1, 4};
    case GGML_TYPE_F16:
        return {1, 2};
    case GGML_TYPE_BF16:
        return {1, 2};
    case GGML_TYPE_Q4_0:
        return {32, 18};
    case GGML_TYPE_Q4_1:
        return {32, 20};
    case GGML_TYPE_Q5_0:
        return {32, 22};
    case GGML_TYPE_Q5_1:
        return {32, 24};
    case GGML_TYPE_Q8_0:
        return {32, 34};
    case GGML_TYPE_Q2_K:
        return {256, 84};
    case GGML_TYPE_Q3_K:
        return {256, 110};
    case GGML_TYPE_Q4_K:
        return {256, 144};
    case GGML_TYPE_Q5_K:
        return {256, 176};
    case GGML_TYPE_Q6_K:
        return {256, 210};
    }
    OPENVINO_THROW("GGUF: unsupported tensor element type id ", static_cast<uint32_t>(t));
}

inline size_t total_elems(const TensorDescriptor& td) {
    size_t n = 1;
    for (auto d : td.dims)
        n *= static_cast<size_t>(d);
    return n;
}

inline size_t tensor_bytes(const TensorDescriptor& td) {
    auto g = geom(td.type);
    size_t n = total_elems(td);
    OPENVINO_ASSERT(n % g.qk == 0,
                    "GGUF: tensor '",
                    td.name,
                    "' element count ",
                    n,
                    " not divisible by block size ",
                    g.qk);
    return (n / g.qk) * g.bytes;
}

// ---- byte-level row permutation for llama RoPE -------------------------- //
//
// Replicates rope_reorder_rows() but on raw bytes, so it works identically
// whether the buffer holds FP, Q4_0/Q4_1/Q8_0 blocks, or a 1-D bias.
void permute_rope_rows_bytes(const uint8_t* src, uint8_t* dst, size_t row_bytes, size_t total_rows, int head_dim) {
    OPENVINO_ASSERT(total_rows % static_cast<size_t>(head_dim) == 0,
                    "GGUF rope reorder: rows not divisible by head_dim");
    const int half = head_dim / 2;
    for (size_t i = 0; i < total_rows; ++i) {
        size_t head = i / head_dim;
        int rih = static_cast<int>(i % head_dim);
        int src_rih = (rih < half) ? 2 * rih : 2 * (rih - half) + 1;
        size_t src_row = head * head_dim + src_rih;
        std::memcpy(dst + i * row_bytes, src + src_row * row_bytes, row_bytes);
    }
}

// Returns either `raw` directly (no reorder) or a pointer into `owned` that
// holds a permuted copy. Caller keeps `owned` alive for the duration of use.
const uint8_t* maybe_permute(const TensorDescriptor& td,
                             const uint8_t* raw,
                             bool row_reorder_rope,
                             int head_dim,
                             std::vector<uint8_t>& owned) {
    if (!row_reorder_rope)
        return raw;
    OPENVINO_ASSERT(head_dim > 0, "GGUF rope reorder: head_dim must be positive");
    const size_t total_bytes = tensor_bytes(td);
    const size_t total_rows = static_cast<size_t>(td.dims.front());
    OPENVINO_ASSERT(total_bytes % total_rows == 0, "GGUF rope reorder: byte size not divisible by rows");
    const size_t row_bytes = total_bytes / total_rows;
    owned.resize(total_bytes);
    permute_rope_rows_bytes(raw, owned.data(), row_bytes, total_rows, head_dim);
    return owned.data();
}

// ---- helpers for native quant Constants --------------------------------- //

// Build a 1D i64 Reshape pattern.
std::shared_ptr<Constant> i64_vec(std::initializer_list<int64_t> v) {
    return std::make_shared<Constant>(ov::element::i64, ov::Shape{v.size()}, std::vector<int64_t>(v));
}

// Wrap a Convert / chain into f32 and tag the friendly name.
ov::Output<ov::Node> finish_f32(const std::shared_ptr<ov::Node>& last_f16, const std::string& name) {
    auto cvt = std::make_shared<Convert>(last_f16, ov::element::f32);
    cvt->set_friendly_name(name);
    return cvt->output(0);
}

// Apply optional decompression: Convert(weights, f16) [- zp] [* scale] -> f32.
//
// `weights_const` already has shape [out, in] in the requested element type.
// `scale_data` and optional `min_data` (length out * (in/qk)) are stored as
// fp16, shape [out, in/qk, 1] for broadcasting on the inner group.
// `zero_point` is subtracted from every element before the scale multiply
// (the standard pattern emitted by NNCF / optimum-intel for unsigned packed
// weight formats); 0 means no subtract.
ov::Output<ov::Node> emit_decompression(const std::shared_ptr<Constant>& weights_const,
                                        int64_t out_dim,
                                        int64_t in_dim,
                                        int qk,
                                        const std::vector<ov::float16>& scale_data,
                                        const std::vector<ov::float16>* min_data,
                                        int zero_point,
                                        const std::string& name) {
    OPENVINO_ASSERT(in_dim % qk == 0, "GGUF: in_dim ", in_dim, " not divisible by qk ", qk);
    const int64_t n_groups = in_dim / qk;

    auto w_f16 = std::make_shared<Convert>(weights_const, ov::element::f16);

    // [out, in] -> [out, n_groups, qk]
    auto grouped = std::make_shared<Reshape>(w_f16, i64_vec({out_dim, n_groups, qk}), false);

    std::shared_ptr<ov::Node> cur = grouped;

    if (zero_point != 0) {
        auto zp = std::make_shared<Constant>(ov::element::f16,
                                             ov::Shape{},
                                             std::vector<ov::float16>{ov::float16(static_cast<float>(zero_point))});
        cur = std::make_shared<Subtract>(cur, zp);
    }

    OPENVINO_ASSERT(static_cast<int64_t>(scale_data.size()) == out_dim * n_groups,
                    "GGUF: scale_data has wrong size for ",
                    name);
    auto scale_const =
        std::make_shared<Constant>(ov::element::f16,
                                   ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(n_groups), 1},
                                   scale_data);
    cur = std::make_shared<Multiply>(cur, scale_const);

    if (min_data) {
        OPENVINO_ASSERT(static_cast<int64_t>(min_data->size()) == out_dim * n_groups,
                        "GGUF: min_data has wrong size for ",
                        name);
        auto min_const =
            std::make_shared<Constant>(ov::element::f16,
                                       ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(n_groups), 1},
                                       *min_data);
        cur = std::make_shared<Add>(cur, min_const);
    }

    // Back to [out, in].
    auto flat = std::make_shared<Reshape>(cur, i64_vec({out_dim, in_dim}), false);

    return finish_f32(flat, name);
}

// Repack a single Q4_0 / Q4_1 block (16 bytes of nibble pairs) to OV u4 layout.
//
//   GGUF: byte j (j=0..15)  -> low nibble = e[j], high nibble = e[j+16]
//   OV  : byte j (j=0..15)  -> low nibble = e[2j], high nibble = e[2j+1]
inline void repack_q4_block_to_u4(const uint8_t* qs, uint8_t* out16) {
    // OV bytes 0..7 hold elements 0..15 = GGUF low nibbles, paired up.
    for (int j = 0; j < 8; ++j) {
        const uint8_t lo = qs[2 * j] & 0x0F;      // e[2j]
        const uint8_t hi = qs[2 * j + 1] & 0x0F;  // e[2j+1]
        out16[j] = static_cast<uint8_t>(lo | (hi << 4));
    }
    // OV bytes 8..15 hold elements 16..31 = GGUF high nibbles, paired up.
    for (int j = 0; j < 8; ++j) {
        const uint8_t lo = (qs[2 * j] >> 4) & 0x0F;      // e[2j+16]
        const uint8_t hi = (qs[2 * j + 1] >> 4) & 0x0F;  // e[2j+17]
        out16[j + 8] = static_cast<uint8_t>(lo | (hi << 4));
    }
}

// Pack a row-major buffer of small non-negative integer codes into an
// `ov::Tensor` of the requested sub-byte / byte element type, using
// `ov::element::iterator<ET>` so any layout (LSB-packed nibble, bit, or 24-bit
// split-bit unit for u3/u6) is handled correctly. Returns a `Constant` that
// owns a copy of the tensor.
template <ov::element::Type_t ET, typename SrcInt>
std::shared_ptr<Constant> make_packed_constant(const ov::Shape& shape, const SrcInt* values) {
    ov::Tensor t(ov::element::Type{ET}, shape);
    using F = ov::fundamental_type_for<ET>;
    auto it = ov::element::iterator<ET>(reinterpret_cast<F*>(t.data()));
    const size_t n = ov::shape_size(shape);
    for (size_t i = 0; i < n; ++i, ++it) {
        *it = static_cast<F>(values[i]);
    }
    return std::make_shared<Constant>(t);
}

// ---- per-format native builders ---------------------------------------- //

constexpr int QK_LEGACY = 32;

ov::Output<ov::Node> build_q4_0_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr size_t BS = 18;  // dh(2) + qs(16)
    const size_t n = total_elems(td);
    const int64_t nb = static_cast<int64_t>(n / QK_LEGACY);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());

    // Pack u4 weights row-major: 1 byte per 2 elements.
    std::vector<uint8_t> packed(n / 2);
    std::vector<ov::float16> scales(static_cast<size_t>(nb));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        uint16_t dh;
        std::memcpy(&dh, src, 2);
        scales[static_cast<size_t>(i)] = ov::float16::from_bits(dh);
        repack_q4_block_to_u4(src + 2, packed.data() + i * 16);
    }

    auto w_const = std::make_shared<Constant>(ov::element::u4,
                                              ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
                                              packed.data());
    w_const->set_friendly_name(name + "/q4_0/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              QK_LEGACY,
                              scales,
                              /*min_data=*/nullptr,
                              /*zero_point=*/8,
                              name);
}

ov::Output<ov::Node> build_q4_1_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr size_t BS = 20;  // dh(2) + mh(2) + qs(16)
    const size_t n = total_elems(td);
    const int64_t nb = static_cast<int64_t>(n / QK_LEGACY);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());

    std::vector<uint8_t> packed(n / 2);
    std::vector<ov::float16> scales(static_cast<size_t>(nb));
    std::vector<ov::float16> mins(static_cast<size_t>(nb));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        uint16_t dh, mh;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        scales[static_cast<size_t>(i)] = ov::float16::from_bits(dh);
        mins[static_cast<size_t>(i)] = ov::float16::from_bits(mh);
        repack_q4_block_to_u4(src + 4, packed.data() + i * 16);
    }

    auto w_const = std::make_shared<Constant>(ov::element::u4,
                                              ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
                                              packed.data());
    w_const->set_friendly_name(name + "/q4_1/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              QK_LEGACY,
                              scales,
                              &mins,
                              /*zero_point=*/0,
                              name);
}

ov::Output<ov::Node> build_q8_0_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr size_t BS = 34;  // dh(2) + qs(32 i8)
    const size_t n = total_elems(td);
    const int64_t nb = static_cast<int64_t>(n / QK_LEGACY);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());

    std::vector<int8_t> q(n);
    std::vector<ov::float16> scales(static_cast<size_t>(nb));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        uint16_t dh;
        std::memcpy(&dh, src, 2);
        scales[static_cast<size_t>(i)] = ov::float16::from_bits(dh);
        std::memcpy(q.data() + i * QK_LEGACY, src + 2, QK_LEGACY);
    }

    auto w_const = std::make_shared<Constant>(ov::element::i8,
                                              ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
                                              q.data());
    w_const->set_friendly_name(name + "/q8_0/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              QK_LEGACY,
                              scales,
                              /*min_data=*/nullptr,
                              /*zero_point=*/0,
                              name);
}

// ---- Q5_0 native (stored as i8 + per-block fp16 scale) ----------------- //
//
// GGUF Q5_0 block (32 elements, 22 bytes):
//   dh fp16     block scale
//   qh u32      32 high bits (one per element)
//   qs[16]      32 low nibbles (low/high in same byte share 16-element halves)
//
// ggml dequant: q5 = ((qs[j] & 0xF)        | ((qh >> j) << 4 & 0x10)) - 16   (j in 0..15)
//               q5 = ((qs[j] >> 4) & 0xF) | ((qh >> (j+12)) & 0x10) - 16    (j+16)
//
// Range is [-16, 15] -> fits an i8 with no zero-point. We don't pursue a
// fully-native u5 layout because OpenVINO has no 5-bit element type; using i8
// costs ~1.5 bits/elem vs disk (8 vs 5.5) but keeps the chain identical to
// Q8_0 / Q6_K (Convert -> Reshape -> Multiply(scale) -> Reshape -> Convert).

ov::Output<ov::Node> build_q5_0_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr size_t BS = 22;  // dh(2) + qh(4) + qs(16)
    const size_t n = total_elems(td);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());
    const int64_t nb = static_cast<int64_t>(n / QK_LEGACY);

    std::vector<int8_t> qbuf(n);
    std::vector<ov::float16> scales(static_cast<size_t>(nb));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        uint16_t dh;
        std::memcpy(&dh, src, 2);
        scales[static_cast<size_t>(i)] = ov::float16::from_bits(dh);

        uint32_t qh;
        std::memcpy(&qh, src + 2, 4);
        const uint8_t* qs = src + 6;
        int8_t* dst = qbuf.data() + i * QK_LEGACY;

        for (int j = 0; j < QK_LEGACY / 2; ++j) {
            const uint8_t xh_0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
            const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
            const int q0 = static_cast<int>((qs[j] & 0xF) | xh_0) - 16;
            const int q1 = static_cast<int>((qs[j] >> 4) | xh_1) - 16;
            dst[j] = static_cast<int8_t>(q0);
            dst[j + QK_LEGACY / 2] = static_cast<int8_t>(q1);
        }
    }

    auto w_const = std::make_shared<Constant>(ov::element::i8,
                                              ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
                                              qbuf.data());
    w_const->set_friendly_name(name + "/q5_0/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              QK_LEGACY,
                              scales,
                              /*min_data=*/nullptr,
                              /*zero_point=*/0,
                              name);
}

// ---- Q4_K native ------------------------------------------------------- //
//
// GGUF Q4_K super-block (256 elements, 144 bytes):
//   d    fp16            super-block scale-of-scales
//   dmin fp16            super-block min-of-mins
//   sc12 12 bytes        eight 6-bit (sc, m) pairs (one per 32-elem sub-block)
//   qs   128 bytes       u4 weights (low/high nibble pattern, see below)
//
// ggml dequant (per sub-block g, 32 elements):  w[j] = d*sc[g] * q[j] - dmin*m[g]
//
// We flatten the two-level scaling at load time:
//   scale[g] = d * sc[g]                      (fp16, single per-32-elem group)
//   min  [g] = -(dmin * m[g])                 (fp16, negated so the standard
//                                              `Add(min)` chain implements the
//                                              ggml `- dmin*m` semantics)
//
// The chain emitted is identical to Q4_1, just with an additional outer level
// of scaling absorbed into the per-group constants. No information is lost
// (Q4_K is *defined* by this formula).
//
// qs layout per super-block:
//   for j = 0..3 (each j covers 64 elements -> two 32-elem sub-blocks):
//     low  nibbles of qs[32j .. 32j+31]  -> sub-block (2j+0)
//     high nibbles of qs[32j .. 32j+31]  -> sub-block (2j+1)
//
// OV u4 packed layout (shape [out, in], in row-major):
//   byte k  ->  low nibble = elem 2k, high nibble = elem 2k+1
//
// So per super-block we emit 128 packed bytes; element ordering within the
// super-block is sub-block 0 (32 elems), sub-block 1 (32 elems), ...

// 6-bit (sc, m) decoder for the 12-byte packed scales/mins block.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ov::Output<ov::Node> build_q4_k_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr int QK = 256;     // super-block elements
    constexpr int GS = 32;      // OV chain group size
    constexpr size_t BS = 144;  // bytes per super-block
    const size_t n = total_elems(td);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());
    const int64_t nb = static_cast<int64_t>(n / QK);
    const int64_t n_groups = in_dim / GS;
    OPENVINO_ASSERT(in_dim % QK == 0, "GGUF Q4_K: in_dim ", in_dim, " not divisible by 256");

    std::vector<uint8_t> packed(n / 2);
    std::vector<ov::float16> scales(static_cast<size_t>(out_dim * n_groups));
    std::vector<ov::float16> mins(static_cast<size_t>(out_dim * n_groups));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        uint16_t dh, mh;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        const float d = static_cast<float>(ov::float16::from_bits(dh));
        const float dmin = static_cast<float>(ov::float16::from_bits(mh));
        const uint8_t* sc12 = src + 4;
        const uint8_t* qs = src + 16;

        // Per-super-block index in the flat scales/mins layout. Because GGUF
        // tensor data is row-major and each row holds an integer number of
        // super-blocks, the flat super-block index `i` decomposes as
        // out_idx*(in_dim/256) + s, and the OV per-group index is i*8+g.
        const int64_t base = i * 8;
        for (int g = 0; g < 8; ++g) {
            uint8_t sc, m;
            get_scale_min_k4(g, sc12, sc, m);
            scales[static_cast<size_t>(base + g)] = ov::float16(d * sc);
            mins[static_cast<size_t>(base + g)] = ov::float16(-dmin * m);
        }

        // Repack 128 qs bytes -> 128 OV u4 bytes (covering 256 elements).
        // Per 64-element pair of sub-blocks (using qs[32j .. 32j+31]):
        //   OV bytes 32j .. 32j+15 hold sub-block 2j+0 (low nibbles, paired)
        //   OV bytes 32j+16 .. 32j+31 hold sub-block 2j+1 (high nibbles, paired)
        uint8_t* dst = packed.data() + i * (QK / 2);
        for (int j = 0; j < 4; ++j) {
            const uint8_t* qb = qs + 32 * j;
            uint8_t* db = dst + 32 * j;
            for (int k = 0; k < 16; ++k) {
                const uint8_t lo0 = qb[2 * k] & 0x0F;
                const uint8_t lo1 = qb[2 * k + 1] & 0x0F;
                db[k] = static_cast<uint8_t>(lo0 | (lo1 << 4));
            }
            for (int k = 0; k < 16; ++k) {
                const uint8_t hi0 = (qb[2 * k] >> 4) & 0x0F;
                const uint8_t hi1 = (qb[2 * k + 1] >> 4) & 0x0F;
                db[16 + k] = static_cast<uint8_t>(hi0 | (hi1 << 4));
            }
        }
    }

    auto w_const = std::make_shared<Constant>(ov::element::u4,
                                              ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
                                              packed.data());
    w_const->set_friendly_name(name + "/q4_k/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              GS,
                              scales,
                              &mins,
                              /*zero_point=*/0,
                              name);
}

// ---- Q6_K native ------------------------------------------------------- //
//
// GGUF Q6_K super-block (256 elements, 210 bytes):
//   ql[128]   : low 4 bits of each weight (interleaved across half-super-blocks)
//   qh[64]    : high 2 bits of each weight (interleaved)
//   scales[16]: int8 per-16-elem sub-scales
//   d         : fp16 super-block scale
//
// ggml dequant: q6 = (low4 | (high2 << 4)) - 32   (signed, range [-32, 31])
//               w  = d * scales[s] * q6           (s = sub-block index, 16 per super)
//
// We encode q6+32 (range [0, 63]) into ov::element::u6 (24-bit storage units,
// 4 elements per 3 bytes), then subtract a constant 32 in the f16 chain. This
// is a tight fit: no information is lost (ggml itself defines q6 in [-32, 31])
// and the on-the-wire weight tensor in the IR is exactly 6 bits/element vs 8
// for the previous i8-based path (~25% smaller weights).
//
//   Constant(u6, [out, in])           # contiguous sub-block-major, q6+32 in [0, 63]
//     -> Convert(f16)
//     -> Reshape([out, in/16, 16])    # group = sub-block size = 16
//     -> Subtract(f16 const 32)       # back to signed [-32, 31]
//     -> Multiply(Constant(f16, [out, in/16, 1]))   # = d * scales[s]
//     -> Reshape([out, in])
//     -> Convert(f32)

ov::Output<ov::Node> build_q6_k_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr int QK = 256;
    constexpr int GS = 16;  // sub-block size = OV chain group size
    constexpr size_t BS = 210;
    const size_t n = total_elems(td);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());
    const int64_t nb = static_cast<int64_t>(n / QK);
    const int64_t n_groups = in_dim / GS;
    OPENVINO_ASSERT(in_dim % QK == 0, "GGUF Q6_K: in_dim ", in_dim, " not divisible by 256");

    std::vector<uint8_t> qbuf(n);  // values in [0, 63] (q6 + 32)
    std::vector<ov::float16> scales(static_cast<size_t>(out_dim * n_groups));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        const uint8_t* ql = src;
        const uint8_t* qh = src + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(src + 192);
        uint16_t dh;
        std::memcpy(&dh, src + 208, 2);
        const float d = static_cast<float>(ov::float16::from_bits(dh));

        // Per super-block, decode 256 q6 values into contiguous sub-block order.
        // Mirror ggml's loop exactly; the resulting `dst` is already grouped by
        // sub-block (16 elements per group, 16 groups per super-block). The
        // values stored are `q6 + 32` so they fit unsigned u6 [0, 63].
        uint8_t* dst = qbuf.data() + i * QK;
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql_h = ql + 64 * half;
            const uint8_t* qh_h = qh + 32 * half;
            uint8_t* dst_h = dst + 128 * half;
            for (int l = 0; l < 32; ++l) {
                int q1 = static_cast<int>((ql_h[l + 0] & 0xF) | (((qh_h[l] >> 0) & 3) << 4));
                int q2 = static_cast<int>((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4));
                int q3 = static_cast<int>((ql_h[l + 0] >> 4) | (((qh_h[l] >> 4) & 3) << 4));
                int q4 = static_cast<int>((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4));
                dst_h[l + 0] = static_cast<uint8_t>(q1);
                dst_h[l + 32] = static_cast<uint8_t>(q2);
                dst_h[l + 64] = static_cast<uint8_t>(q3);
                dst_h[l + 96] = static_cast<uint8_t>(q4);
            }
        }

        // Per-sub-block fp16 scales. Sub-block s in [0..15] uses sc[s] directly
        // (no permutation; see derivation in the comment block above).
        const int64_t base = i * (QK / GS);  // first OV-group index for this super-block
        for (int s = 0; s < 16; ++s) {
            scales[static_cast<size_t>(base + s)] = ov::float16(d * static_cast<float>(sc[s]));
        }
    }

    auto w_const = make_packed_constant<ov::element::Type_t::u6>(
        ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
        qbuf.data());
    w_const->set_friendly_name(name + "/q6_k/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              GS,
                              scales,
                              /*min_data=*/nullptr,
                              /*zero_point=*/32,
                              name);
}

// ---- Q5_1 native (stored as i8 + per-block fp16 scale + fp16 min) ------ //
//
// GGUF Q5_1 block (32 elements, 24 bytes):
//   dh fp16, mh fp16, qh u32, qs[16]
//
// ggml dequant: q5 = (qs[j] & 0xF) | ((qh >> j) << 4 & 0x10)         (j in 0..15)
//               q5 = (qs[j] >> 4) | ((qh >> (j+12)) & 0x10)          (j+16)
//               w  = d * q5 + m
// Range of q5 is [0, 31] -> fits an unsigned 5-bit value, but OpenVINO has no
// `u5`. Storing as i8 doubles the bits/elem on disk (8 vs 5.5) but keeps the
// chain symmetric with Q4_1 / Q5_K (`Multiply(scale) + Add(min)`).

ov::Output<ov::Node> build_q5_1_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr size_t BS = 24;
    const size_t n = total_elems(td);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());
    const int64_t nb = static_cast<int64_t>(n / QK_LEGACY);

    std::vector<int8_t> qbuf(n);
    std::vector<ov::float16> scales(static_cast<size_t>(nb));
    std::vector<ov::float16> mins(static_cast<size_t>(nb));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        uint16_t dh, mh;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        scales[static_cast<size_t>(i)] = ov::float16::from_bits(dh);
        mins[static_cast<size_t>(i)] = ov::float16::from_bits(mh);

        uint32_t qh;
        std::memcpy(&qh, src + 4, 4);
        const uint8_t* qs = src + 8;
        int8_t* dst = qbuf.data() + i * QK_LEGACY;

        for (int j = 0; j < QK_LEGACY / 2; ++j) {
            const uint8_t xh_0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
            const uint8_t xh_1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
            const int q0 = static_cast<int>((qs[j] & 0xF) | xh_0);  // [0, 31]
            const int q1 = static_cast<int>((qs[j] >> 4) | xh_1);   // [0, 31]
            dst[j] = static_cast<int8_t>(q0);
            dst[j + QK_LEGACY / 2] = static_cast<int8_t>(q1);
        }
    }

    auto w_const = std::make_shared<Constant>(ov::element::i8,
                                              ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
                                              qbuf.data());
    w_const->set_friendly_name(name + "/q5_1/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              QK_LEGACY,
                              scales,
                              &mins,
                              /*zero_point=*/0,
                              name);
}

// ---- Q2_K native (u2 + flattened scale & min, group=16) ---------------- //
//
// GGUF Q2_K super-block (256 elements, 84 bytes):
//   scales[16]   8 (sc, m) pairs of 4+4 bits each (one pair per 16-elem sub-block)
//   qs[64]       2-bit packed weights (4 elems per byte, ggml-specific layout)
//   d  fp16      super-block scale of scales
//   dmin fp16    super-block scale of mins
//
// ggml dequant per 16-elem sub-block g (within a super-block):
//   sc, m = scales[g] (4 bits each)
//   dl = d * sc, ml = dmin * m
//   w[l] = dl * q2[l] - ml,  with q2[l] in [0, 3]
//
// We flatten the two-level scaling into per-sub-block fp16 constants:
//   scale[g] =  d * sc[g],   min[g] = -(dmin * m[g])
// and store q2 in OV `u2` (4 elems/byte, LSB-packed). Group size is 16.
//
// Element ordering within the super-block follows ggml's nested loops; we
// decode it into a contiguous int buffer (sub-block-major) and let the packing
// helper handle bit layout.

ov::Output<ov::Node> build_q2_k_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr int QK = 256;
    constexpr int GS = 16;
    constexpr size_t BS = 84;
    const size_t n = total_elems(td);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());
    const int64_t nb = static_cast<int64_t>(n / QK);
    const int64_t n_groups = in_dim / GS;
    OPENVINO_ASSERT(in_dim % QK == 0, "GGUF Q2_K: in_dim ", in_dim, " not divisible by 256");

    std::vector<uint8_t> qbuf(n);  // values in [0, 3]
    std::vector<ov::float16> scales(static_cast<size_t>(out_dim * n_groups));
    std::vector<ov::float16> mins(static_cast<size_t>(out_dim * n_groups));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        const uint8_t* sc_raw = src;
        const uint8_t* qs = src + 16;
        uint16_t dh, mh;
        std::memcpy(&dh, src + 80, 2);
        std::memcpy(&mh, src + 82, 2);
        const float d = static_cast<float>(ov::float16::from_bits(dh));
        const float dmin = static_cast<float>(ov::float16::from_bits(mh));

        // Flatten 16 (sc, m) pairs into per-group fp16 scale/min.
        const int64_t base = i * (QK / GS);
        for (int g = 0; g < 16; ++g) {
            const uint8_t sc = sc_raw[g] & 0xF;
            const uint8_t m = sc_raw[g] >> 4;
            scales[static_cast<size_t>(base + g)] = ov::float16(d * sc);
            mins[static_cast<size_t>(base + g)] = ov::float16(-dmin * m);
        }

        // Decode q2 codes in the same order that ggml writes them to `dst`.
        // ggml loop:  for n in 0..256 step 128:    (covers 2 groups of 4 sub-blocks)
        //               shift = 0
        //               for j in 0..4:             (4 (lo, hi) sub-block pairs)
        //                 sub-block A: q[l]      >> shift, l in 0..16
        //                 sub-block B: q[l + 16] >> shift, l in 0..16
        //                 shift += 2
        //               q += 32
        uint8_t* dst = qbuf.data() + i * QK;
        const uint8_t* q = qs;
        for (int n_off = 0; n_off < QK; n_off += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                for (int l = 0; l < 16; ++l)
                    *dst++ = static_cast<uint8_t>((q[l] >> shift) & 3);
                for (int l = 0; l < 16; ++l)
                    *dst++ = static_cast<uint8_t>((q[l + 16] >> shift) & 3);
                shift += 2;
            }
            q += 32;
            (void)n_off;
        }
    }

    auto w_const = make_packed_constant<ov::element::Type_t::u2>(
        ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
        qbuf.data());
    w_const->set_friendly_name(name + "/q2_k/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              GS,
                              scales,
                              &mins,
                              /*zero_point=*/0,
                              name);
}

// ---- Q3_K native (u3 + zp=4 + per-sub-block scale, group=16) ----------- //
//
// GGUF Q3_K super-block (256 elements, 110 bytes):
//   hmask[32]    high bit of each 3-bit weight (256 bits total)
//   qs[64]       low 2 bits of each weight
//   scales[12]   16 6-bit scales packed (signed, biased by +32 in the wire format)
//   d  fp16      super-block scale
//
// ggml dequant per 16-elem sub-block g:
//   sc = scales[g] - 32      (-32..31 effectively, but typical range narrower)
//   q3 = ((qs[l] >> shift) & 3) | (hmask_bit ? 0 : 4)   then  q3 -= 4   (l in 0..16)
//   w  = d * sc * q3                                    (q3 in [-4, 3])
//
// To map onto OV `u3` we encode `q3 + 4` ∈ [0, 7] and add a `Subtract(4)` in the
// f16 chain. The 6-bit signed-biased sc is folded into the per-group fp16
// scale = d * (sc - 32). Group size 16.

// Decoder for the 12-byte packed scales of Q3_K (mirrors ggml).
inline void unpack_q3_k_scales(const uint8_t* sc12, int8_t out[16]) {
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    std::memcpy(aux, sc12, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp)&kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(out, aux, 16);
}

ov::Output<ov::Node> build_q3_k_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr int QK = 256;
    constexpr int GS = 16;
    constexpr size_t BS = 110;
    const size_t n = total_elems(td);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());
    const int64_t nb = static_cast<int64_t>(n / QK);
    const int64_t n_groups = in_dim / GS;
    OPENVINO_ASSERT(in_dim % QK == 0, "GGUF Q3_K: in_dim ", in_dim, " not divisible by 256");

    std::vector<uint8_t> qbuf(n);  // values in [0, 7] (q3 + 4)
    std::vector<ov::float16> scales(static_cast<size_t>(out_dim * n_groups));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        const uint8_t* hm = src;
        const uint8_t* qs = src + 32;
        const uint8_t* sc12 = src + 96;
        uint16_t dh;
        std::memcpy(&dh, src + 108, 2);
        const float d_all = static_cast<float>(ov::float16::from_bits(dh));

        int8_t sc[16];
        unpack_q3_k_scales(sc12, sc);

        const int64_t base = i * (QK / GS);
        for (int g = 0; g < 16; ++g) {
            scales[static_cast<size_t>(base + g)] = ov::float16(d_all * static_cast<float>(sc[g] - 32));
        }

        // Decode q3 codes in ggml's order. q3_signed in [-4, 3]; we store
        // q3_unsigned = q3_signed + 4 in [0, 7] for the u3 path.
        uint8_t* dst = qbuf.data() + i * QK;
        uint8_t m = 1;
        const uint8_t* q = qs;
        const uint8_t* h = hm;
        for (int n_off = 0; n_off < QK; n_off += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                for (int l = 0; l < 16; ++l) {
                    int q3 = static_cast<int>((q[l] >> shift) & 3) - ((h[l] & m) ? 0 : 4);
                    *dst++ = static_cast<uint8_t>(q3 + 4);
                }
                for (int l = 0; l < 16; ++l) {
                    int q3 = static_cast<int>((q[l + 16] >> shift) & 3) - ((h[l + 16] & m) ? 0 : 4);
                    *dst++ = static_cast<uint8_t>(q3 + 4);
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
            (void)n_off;
        }
        (void)h;  // hmask is consumed via index `l` above; pointer not advanced
    }

    auto w_const = make_packed_constant<ov::element::Type_t::u3>(
        ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
        qbuf.data());
    w_const->set_friendly_name(name + "/q3_k/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              GS,
                              scales,
                              /*min_data=*/nullptr,
                              /*zero_point=*/4,
                              name);
}

// ---- Q5_K native (stored as i8 + per-32-elem scale & min, group=32) ---- //
//
// GGUF Q5_K super-block (256 elements, 176 bytes):
//   d  fp16, dmin fp16
//   scales[12]  : 8 (sc, m) 6-bit pairs (one per 32-elem sub-block; same encoding as Q4_K)
//   qh[32]      : 5th bit of each weight
//   qs[128]     : low 4 bits of each weight (low/high nibble pattern)
//
// ggml dequant per sub-block g:
//   sc, m = get_scale_min_k4(g, scales)
//   d1 = d * sc, m1 = dmin * m
//   q  = (ql[l] & 0xF | (qh_bit << 4))   for low nibble half
//   q  = (ql[l] >>  4 | (qh_bit << 4))   for high nibble half
//   w  = d1 * q - m1     (q in [0, 31])
//
// We use i8 as for Q5_0/Q5_1 (no native u5), group=32, with flattened scale/min.

ov::Output<ov::Node> build_q5_k_native(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    constexpr int QK = 256;
    constexpr int GS = 32;
    constexpr size_t BS = 176;
    const size_t n = total_elems(td);
    const int64_t out_dim = static_cast<int64_t>(td.dims.front());
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());
    const int64_t nb = static_cast<int64_t>(n / QK);
    const int64_t n_groups = in_dim / GS;
    OPENVINO_ASSERT(in_dim % QK == 0, "GGUF Q5_K: in_dim ", in_dim, " not divisible by 256");

    std::vector<int8_t> qbuf(n);  // values in [0, 31]
    std::vector<ov::float16> scales(static_cast<size_t>(out_dim * n_groups));
    std::vector<ov::float16> mins(static_cast<size_t>(out_dim * n_groups));

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* src = raw + i * BS;
        uint16_t dh, mh;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        const float d = static_cast<float>(ov::float16::from_bits(dh));
        const float dmin = static_cast<float>(ov::float16::from_bits(mh));
        const uint8_t* sc12 = src + 4;
        const uint8_t* qh = src + 16;
        const uint8_t* qs = src + 48;

        const int64_t base = i * (QK / GS);  // 8 groups per super-block
        int8_t* dst = qbuf.data() + i * QK;

        // ggml iterates the super-block in 64-element chunks (= 2 sub-blocks).
        // Per chunk, the low nibble half uses qh bit u1 (1 << 2j) and the high
        // nibble half uses bit u2 (2 << 2j), where j is the chunk index.
        const uint8_t* ql = qs;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc, m;
            get_scale_min_k4(2 * j + 0, sc12, sc, m);
            scales[static_cast<size_t>(base + 2 * j + 0)] = ov::float16(d * sc);
            mins[static_cast<size_t>(base + 2 * j + 0)] = ov::float16(-dmin * m);
            get_scale_min_k4(2 * j + 1, sc12, sc, m);
            scales[static_cast<size_t>(base + 2 * j + 1)] = ov::float16(d * sc);
            mins[static_cast<size_t>(base + 2 * j + 1)] = ov::float16(-dmin * m);

            const uint8_t bit_lo = static_cast<uint8_t>(1 << (2 * j + 0));
            const uint8_t bit_hi = static_cast<uint8_t>(1 << (2 * j + 1));
            for (int l = 0; l < 32; ++l) {
                const int v = (ql[l] & 0xF) + ((qh[l] & bit_lo) ? 16 : 0);
                *dst++ = static_cast<int8_t>(v);
            }
            for (int l = 0; l < 32; ++l) {
                const int v = (ql[l] >> 4) + ((qh[l] & bit_hi) ? 16 : 0);
                *dst++ = static_cast<int8_t>(v);
            }
            ql += 32;
        }
    }

    auto w_const = std::make_shared<Constant>(ov::element::i8,
                                              ov::Shape{static_cast<size_t>(out_dim), static_cast<size_t>(in_dim)},
                                              qbuf.data());
    w_const->set_friendly_name(name + "/q5_k/weights");

    return emit_decompression(w_const,
                              out_dim,
                              in_dim,
                              GS,
                              scales,
                              &mins,
                              /*zero_point=*/0,
                              name);
}

// ---- fallback: dequantize to FP16 then wrap as Constant + Convert(f32) -- //

ov::Output<ov::Node> build_dequantized(const TensorDescriptor& td, const uint8_t* raw, const std::string& name) {
    ov::Tensor t = materialize_tensor_f16_or_native(td, raw);
    auto c = std::make_shared<Constant>(t);
    c->set_friendly_name(name);
    if (t.get_element_type() == ov::element::f32) {
        return c->output(0);
    }
    return finish_f32(c, name);
}

// Whether we can preserve native compression for this tensor. Requires a 2D
// shape with the contiguous (last) dim divisible by the block size. 1D tensors
// (biases, norm weights) are typically fp32 or fp16 and are handled by the
// fallback path.
bool eligible_for_native(const TensorDescriptor& td) {
    if (td.dims.size() != 2)
        return false;
    const int64_t in_dim = static_cast<int64_t>(td.dims.back());
    auto g = geom(td.type);
    if (g.qk == 1)
        return false;  // no compression
    return (in_dim % g.qk) == 0;
}

}  // namespace

ov::Output<ov::Node> build_weight_node(const TensorDescriptor& td,
                                       const uint8_t* raw,
                                       const std::string& name,
                                       bool row_reorder_rope,
                                       int head_dim) {
    std::vector<uint8_t> permuted;
    const uint8_t* eff = maybe_permute(td, raw, row_reorder_rope, head_dim, permuted);

    if (eligible_for_native(td)) {
        switch (td.type) {
        case GGML_TYPE_Q4_0:
            return build_q4_0_native(td, eff, name);
        case GGML_TYPE_Q4_1:
            return build_q4_1_native(td, eff, name);
        case GGML_TYPE_Q5_0:
            return build_q5_0_native(td, eff, name);
        case GGML_TYPE_Q5_1:
            return build_q5_1_native(td, eff, name);
        case GGML_TYPE_Q8_0:
            return build_q8_0_native(td, eff, name);
        case GGML_TYPE_Q2_K:
            return build_q2_k_native(td, eff, name);
        case GGML_TYPE_Q3_K:
            return build_q3_k_native(td, eff, name);
        case GGML_TYPE_Q4_K:
            return build_q4_k_native(td, eff, name);
        case GGML_TYPE_Q5_K:
            return build_q5_k_native(td, eff, name);
        case GGML_TYPE_Q6_K:
            return build_q6_k_native(td, eff, name);
        default:
            break;
        }
    }
    return build_dequantized(td, eff, name);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
