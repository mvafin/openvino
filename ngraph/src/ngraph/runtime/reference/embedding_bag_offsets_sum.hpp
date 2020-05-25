// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
    template <typename T, typename U>
    void embeddingBagOffsetsSum(
            const T* emb_table,
            const U* indices,
            const U* offsets,
            const U* default_index,
            const T* weights,
            T* out,
            const size_t indices_count,
            const Shape& outShape) {
        const size_t offsets_size = outShape[0];
        std::vector<U> default_indices;
        if (default_index)
            default_indices.push_back(default_index[0]);
        size_t inDimsSize = outShape.size();

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

        auto get_indices = [&](size_t emb_index, const U*& indices_ref, size_t& indices_num, size_t& weights_idx, bool& with_weights) {
            if (emb_index >= offsets_size)
                throw ngraph_error("Invalid embedding bag index.");
            if (offsets[emb_index] >= indices_count)
                throw ngraph_error(std::string("Offset value exceeds indices size in the model.\noffset: ")
                    + std::to_string(offsets[emb_index]) + "; indices size: "
                    + std::to_string(indices_count));

            indices_ref = nullptr;
            indices_num = 0lu;
            with_weights = (weights != nullptr);

            if (emb_index == offsets_size - 1lu)
                indices_num = indices_count - offsets[emb_index];
            else
                indices_num = offsets[emb_index + 1lu] - offsets[emb_index];

            if (indices_num != 0lu) {
                indices_ref = indices + offsets[emb_index];
            } else {
                // Empty or default bag
                with_weights = false;
                if (default_indices.size() == 1lu) {
                    indices_ref = default_indices.data();
                    indices_num = 1lu;
                }
                return;
            }

            if (with_weights)
                weights_idx = offsets[emb_index];
        };

        size_t indices_size = 0lu;
        const U* indices_emb = nullptr;
        size_t weights_idx = 0lu;
        bool with_weights_b = (weights != nullptr);
        bool with_weights = with_weights_b;

        for (size_t obi = 0lu; obi < outShape.at(0); obi++) {
            size_t dst_index = obi * multipliers[0];
            get_indices(obi, indices_emb, indices_size, weights_idx, with_weights);
            if (indices_emb != nullptr) {
                with_weights = with_weights_b & with_weights;
                for (size_t in_idx = 0lu; in_idx < indices_size; in_idx++) {
                    size_t src_index = indices_emb[in_idx] * multipliers[0];
                    emb_cycle(src_index, dst_index, 1, weights_idx, with_weights);
                    weights_idx++;
                }
            }
        }

    } // embeddingBagOffsetsSum

} // reference
} // runtime
} // ngraph
