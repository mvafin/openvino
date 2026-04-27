// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GGML quantization dequantizers. Reference algorithms ported from
// https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c (MIT/public).

#include "gguf_dequant.hpp"

#include <cstdint>
#include <cstring>
#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

// Bit-cast a GGUF half (uint16_t little-endian) to float.
inline float h2f(uint16_t h) {
    return static_cast<float>(ov::float16::from_bits(h));
}

inline ov::float16 f2h(float f) {
    return ov::float16(f);
}

// Total number of elements in a tensor.
inline size_t total_elements(const TensorDescriptor& td) {
    size_t n = 1;
    for (auto d : td.dims)
        n *= static_cast<size_t>(d);
    return n;
}

// Allocate an FP16 ov::Tensor with the same shape as td.
inline ov::Tensor make_f16_tensor(const TensorDescriptor& td) {
    ov::Shape shape(td.dims.begin(), td.dims.end());
    return ov::Tensor(ov::element::f16, shape);
}

// Q4_K / Q5_K share this 6-bit packed (scale, min) decoder for 8 sub-blocks.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// ----- block sizes (bytes) and element counts ---------------------------- //
constexpr int QK4_0 = 32;
constexpr int BS_Q4_0 = 18;
constexpr int QK4_1 = 32;
constexpr int BS_Q4_1 = 20;
constexpr int QK5_0 = 32;
constexpr int BS_Q5_0 = 22;
constexpr int QK5_1 = 32;
constexpr int BS_Q5_1 = 24;
constexpr int QK8_0 = 32;
constexpr int BS_Q8_0 = 34;
constexpr int QK_K = 256;
constexpr int BS_Q2_K = 84;
constexpr int BS_Q3_K = 110;
constexpr int BS_Q4_K = 144;
constexpr int BS_Q5_K = 176;
constexpr int BS_Q6_K = 210;

// Validate that n is a multiple of expected block_size; return # blocks.
inline int64_t check_blocks(const TensorDescriptor& td, size_t n_elems, int qk) {
    OPENVINO_ASSERT(n_elems % qk == 0,
                    "GGUF tensor '",
                    td.name,
                    "': element count ",
                    n_elems,
                    " is not a multiple of block size ",
                    qk);
    return static_cast<int64_t>(n_elems / qk);
}

// ===== legacy block-32 quants =========================================== //

void dequant_q4_0(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q4_0, dst += QK4_0) {
        uint16_t dh;
        std::memcpy(&dh, src, 2);
        const float d = h2f(dh);
        const uint8_t* qs = src + 2;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            int x0 = (qs[j] & 0xF) - 8;
            int x1 = (qs[j] >> 4) - 8;
            dst[j] = f2h(x0 * d);
            dst[j + QK4_0 / 2] = f2h(x1 * d);
        }
    }
}

void dequant_q4_1(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q4_1, dst += QK4_1) {
        uint16_t dh, mh;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        const float d = h2f(dh);
        const float m = h2f(mh);
        const uint8_t* qs = src + 4;
        for (int j = 0; j < QK4_1 / 2; ++j) {
            int x0 = qs[j] & 0xF;
            int x1 = qs[j] >> 4;
            dst[j] = f2h(x0 * d + m);
            dst[j + QK4_1 / 2] = f2h(x1 * d + m);
        }
    }
}

void dequant_q5_0(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q5_0, dst += QK5_0) {
        uint16_t dh;
        std::memcpy(&dh, src, 2);
        const float d = h2f(dh);
        uint32_t qh;
        std::memcpy(&qh, src + 2, 4);
        const uint8_t* qs = src + 6;
        for (int j = 0; j < QK5_0 / 2; ++j) {
            uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh_1 = (qh >> (j + 12)) & 0x10;
            int x0 = static_cast<int>((qs[j] & 0xF) | xh_0) - 16;
            int x1 = static_cast<int>((qs[j] >> 4) | xh_1) - 16;
            dst[j] = f2h(x0 * d);
            dst[j + QK5_0 / 2] = f2h(x1 * d);
        }
    }
}

void dequant_q5_1(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q5_1, dst += QK5_1) {
        uint16_t dh, mh;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        const float d = h2f(dh);
        const float m = h2f(mh);
        uint32_t qh;
        std::memcpy(&qh, src + 4, 4);
        const uint8_t* qs = src + 8;
        for (int j = 0; j < QK5_1 / 2; ++j) {
            uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            uint8_t xh_1 = (qh >> (j + 12)) & 0x10;
            int x0 = static_cast<int>((qs[j] & 0xF) | xh_0);
            int x1 = static_cast<int>((qs[j] >> 4) | xh_1);
            dst[j] = f2h(x0 * d + m);
            dst[j + QK5_1 / 2] = f2h(x1 * d + m);
        }
    }
}

void dequant_q8_0(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q8_0, dst += QK8_0) {
        uint16_t dh;
        std::memcpy(&dh, src, 2);
        const float d = h2f(dh);
        const int8_t* qs = reinterpret_cast<const int8_t*>(src + 2);
        for (int j = 0; j < QK8_0; ++j) {
            dst[j] = f2h(qs[j] * d);
        }
    }
}

// ===== K-quants (super-block of 256) ===================================== //

// block_q2_K layout: scales[16], qs[64], d (fp16), dmin (fp16). Total 84 bytes.
void dequant_q2_k(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q2_K) {
        const uint8_t* scales = src;
        const uint8_t* qs = src + 16;
        uint16_t dh, mh;
        std::memcpy(&dh, src + 80, 2);
        std::memcpy(&mh, src + 82, 2);
        const float d = h2f(dh);
        const float dmin = h2f(mh);

        int is = 0;
        const uint8_t* q = qs;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF);
                float ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    int q2 = (q[l] >> shift) & 3;
                    *dst++ = f2h(dl * q2 - ml);
                }
                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    int q2 = (q[l + 16] >> shift) & 3;
                    *dst++ = f2h(dl * q2 - ml);
                }
                shift += 2;
            }
            q += 32;
        }
    }
}

// block_q3_K layout: hmask[32], qs[64], scales[12], d (fp16). Total 110 bytes.
void dequant_q3_k(const uint8_t* src, ov::float16* dst, int64_t nb) {
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    for (int64_t i = 0; i < nb; ++i, src += BS_Q3_K) {
        const uint8_t* hm = src;
        const uint8_t* q = src + 32;
        const uint8_t* sc12 = src + 96;
        uint16_t dh;
        std::memcpy(&dh, src + 108, 2);
        const float d_all = h2f(dh);

        std::memcpy(aux, sc12, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp)&kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        uint8_t m = 1;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int q3 = static_cast<int>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                    *dst++ = f2h(dl * q3);
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    int q3 = static_cast<int>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
                    *dst++ = f2h(dl * q3);
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
            (void)n;
        }
    }
}

// block_q4_K layout: d (fp16), dmin (fp16), scales[12], qs[128]. Total 144.
void dequant_q4_k(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q4_K) {
        uint16_t dh, mh;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        const float d = h2f(dh);
        const float dmin = h2f(mh);
        const uint8_t* scales = src + 4;
        const uint8_t* qs = src + 16;

        int is = 0;
        const uint8_t* q = qs;
        for (int n = 0; n < QK_K; n += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l)
                *dst++ = f2h(d1 * (q[l] & 0xF) - m1);
            for (int l = 0; l < 32; ++l)
                *dst++ = f2h(d2 * (q[l] >> 4) - m2);
            q += 32;
            is += 2;
        }
    }
}

// block_q5_K layout: d (fp16), dmin (fp16), scales[12], qh[32], qs[128]. Total 176.
void dequant_q5_k(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q5_K) {
        uint16_t dh, mh;
        std::memcpy(&dh, src, 2);
        std::memcpy(&mh, src + 2, 2);
        const float d = h2f(dh);
        const float dmin = h2f(mh);
        const uint8_t* scales = src + 4;
        const uint8_t* qh = src + 16;
        const uint8_t* qs = src + 48;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        const uint8_t* ql = qs;
        for (int n = 0; n < QK_K; n += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                int v = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
                *dst++ = f2h(d1 * v - m1);
            }
            for (int l = 0; l < 32; ++l) {
                int v = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                *dst++ = f2h(d2 * v - m2);
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// block_q6_K layout: ql[128], qh[64], scales[16] (int8), d (fp16). Total 210.
void dequant_q6_k(const uint8_t* src, ov::float16* dst, int64_t nb) {
    for (int64_t i = 0; i < nb; ++i, src += BS_Q6_K) {
        const uint8_t* ql = src;
        const uint8_t* qh = src + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(src + 192);
        uint16_t dh;
        std::memcpy(&dh, src + 208, 2);
        const float d = h2f(dh);

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[l + 0] = f2h(d * sc[is + 0] * q1);
                dst[l + 32] = f2h(d * sc[is + 2] * q2);
                dst[l + 64] = f2h(d * sc[is + 4] * q3);
                dst[l + 96] = f2h(d * sc[is + 6] * q4);
            }
            dst += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ----- verbatim copy paths (FP types) ------------------------------------ //
ov::Tensor copy_as(const TensorDescriptor& td, const uint8_t* raw, ov::element::Type et) {
    ov::Shape shape(td.dims.begin(), td.dims.end());
    ov::Tensor out(et, shape);
    std::memcpy(out.data(), raw, out.get_byte_size());
    return out;
}

}  // namespace

ov::Tensor materialize_tensor_f16_or_native(const TensorDescriptor& td, const uint8_t* raw) {
    switch (td.type) {
    case GGML_TYPE_F32:
        return copy_as(td, raw, ov::element::f32);
    case GGML_TYPE_F16:
        return copy_as(td, raw, ov::element::f16);
    case GGML_TYPE_BF16:
        return copy_as(td, raw, ov::element::bf16);
    default:
        break;
    }

    const size_t n = total_elements(td);
    auto out = make_f16_tensor(td);
    auto* dst = out.data<ov::float16>();

    switch (td.type) {
    case GGML_TYPE_Q4_0:
        dequant_q4_0(raw, dst, check_blocks(td, n, QK4_0));
        return out;
    case GGML_TYPE_Q4_1:
        dequant_q4_1(raw, dst, check_blocks(td, n, QK4_1));
        return out;
    case GGML_TYPE_Q8_0:
        dequant_q8_0(raw, dst, check_blocks(td, n, QK8_0));
        return out;
    case GGML_TYPE_Q4_K:
        dequant_q4_k(raw, dst, check_blocks(td, n, QK_K));
        return out;
    case GGML_TYPE_Q6_K:
        dequant_q6_k(raw, dst, check_blocks(td, n, QK_K));
        return out;
    // Types not in the public enum subset of the reader use raw numbers.
    default:
        break;
    }

    // Numeric ggml ids for types we recognize beyond the enum constants.
    switch (static_cast<uint32_t>(td.type)) {
    case 6: /* Q5_0 */
        dequant_q5_0(raw, dst, check_blocks(td, n, QK5_0));
        return out;
    case 7: /* Q5_1 */
        dequant_q5_1(raw, dst, check_blocks(td, n, QK5_1));
        return out;
    case 10: /* Q2_K */
        dequant_q2_k(raw, dst, check_blocks(td, n, QK_K));
        return out;
    case 11: /* Q3_K */
        dequant_q3_k(raw, dst, check_blocks(td, n, QK_K));
        return out;
    case 13: /* Q5_K */
        dequant_q5_k(raw, dst, check_blocks(td, n, QK_K));
        return out;
    default:
        break;
    }

    OPENVINO_THROW("GGUF tensor '",
                   td.name,
                   "': unsupported ggml_type ",
                   static_cast<uint32_t>(td.type),
                   ". Codebook (IQ*) and ternary (TQ*) quants are not implemented in this PoC.");
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
