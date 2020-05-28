// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
    template <typename T, typename U>
    void embeddingBagPackedSum(
            const T* emb_table,
            const U* indices,
            const T* weights,
            T* out,
            const Shape& indicesShape,
            const Shape& outShape) {
        const size_t inDimsSize = outShape.size();
        const size_t indices_per_bag = indicesShape[1];

        std::vector<U> multipliers(inDimsSize, 1);
        for (int i = inDimsSize - 1; i > 0; i--) {
            multipliers[i - 1] = outShape[i] * multipliers[i];
        }
        memset(out, 0, shape_size(outShape) * sizeof(T));

        std::function<void(size_t, size_t, size_t, size_t, bool)> emb_cycle =
        [&](size_t src_index, size_t dst_index, size_t emb_idx, size_t weights_idx, bool with_weights) {
            for (size_t i = 0lu; i < outShape[emb_idx]; i++) {
                size_t new_src_idx = src_index + i * multipliers[emb_idx];
                size_t new_dst_idx = dst_index + i * multipliers[emb_idx];
                if (emb_idx == inDimsSize - 1) {
                    if (with_weights)
                        out[new_dst_idx] += emb_table[new_src_idx] * weights[weights_idx];
                    else
                        out[new_dst_idx] += emb_table[new_src_idx];
                } else {
                    emb_cycle(new_src_idx, new_dst_idx, emb_idx + 1, weights_idx, with_weights);
                }
            }
        };

        bool with_weights = (weights != nullptr);

        for (size_t obi = 0lu; obi < outShape.at(0); obi++) {
            size_t dst_index = obi * multipliers[0];
            size_t idx_obi = obi * indices_per_bag;
            for (size_t in_idx = 0lu; in_idx < indices_per_bag; in_idx++) {
                size_t idx_idx = idx_obi + in_idx;
                size_t src_index = indices[idx_idx] * multipliers[0];
                emb_cycle(src_index, dst_index, 1, idx_idx, with_weights);
            }
        }

    } // embeddingBagPackedSum

} // reference
} // runtime
} // ngraph
