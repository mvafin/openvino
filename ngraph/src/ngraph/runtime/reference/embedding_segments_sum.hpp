// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {

    template <typename T, typename U>
    void embeddingSegmentsSum(
            const T* embTable,
            const U* indices,
            const U* segmentIds,
            const U* defaultIndex,
            const T* weights,
            T* out,
            const Shape& embTableShape,
            const Shape& indicesShape,
            const Shape& outShape) {
        const size_t indices_len = indicesShape[0];
        const size_t segments_num = outShape[0];
        const size_t inDimsSize = outShape.size();
        const size_t embDimsNum = outShape.size() - 1;

        std::vector<U> multipliers(inDimsSize, 1);
        for (int i = inDimsSize - 1; i > 0; i--) {
            multipliers[i - 1] = outShape[i] * multipliers[i];
        }
        memset(out, 0, shape_size(outShape) * sizeof(T));

        std::function<void(size_t, size_t, size_t)> emb_cycle =
        [&](size_t src_index, size_t dst_index, size_t emb_idx) {
            for (size_t i = 0lu; i < outShape[emb_idx]; i++) {
                size_t new_src_idx = src_index + i * multipliers[emb_idx];
                size_t new_dst_idx = dst_index + i * multipliers[emb_idx];
                if (emb_idx == embDimsNum) {
                    out[new_dst_idx] += embTable[new_src_idx];
                } else {
                    emb_cycle(new_src_idx, new_dst_idx, emb_idx + 1);
                }
            }
        };

        std::function<void(size_t, size_t, size_t, size_t)> emb_cycle_weights =
        [&](size_t src_index, size_t dst_index, size_t emb_idx, size_t weights_idx) {
            for (size_t i = 0lu; i < outShape[emb_idx]; i++) {
                size_t new_src_idx = src_index + i * multipliers[emb_idx];
                size_t new_dst_idx = dst_index + i * multipliers[emb_idx];
                if (emb_idx == embDimsNum) {
                    out[new_dst_idx] += embTable[new_src_idx] * weights[weights_idx];
                } else {
                    emb_cycle_weights(new_src_idx, new_dst_idx, emb_idx + 1, weights_idx);
                }
            }
        };

        bool with_weights = (weights != nullptr);

        for (size_t index = 0; index < indices_len; index++) {
            size_t obi = segmentIds[index];
            if (obi >= segments_num)
                throw ngraph_error("Segment index could not be more than segments number");
            size_t dst_index = obi * multipliers[0];
            size_t src_index = indices[index] * multipliers[0];
            if (with_weights)
                emb_cycle_weights(src_index, dst_index, 1, index);
            else
                emb_cycle(src_index, dst_index, 1);
        }
        if (defaultIndex != nullptr) {
            U defIndex = defaultIndex[0];
            if (defIndex < U(0) && defIndex >= embTableShape[0])
                throw ngraph_error(std::string("Invalid default index") + std::to_string(defIndex)) ;
            for (size_t obi = 0; obi < segments_num; obi++) {
                bool found = false;
                for (size_t index = 0; index < indices_len; index++) {
                    if (segmentIds[index] == obi) {
                        found  = true;
                        break;
                    }
                }
                if (found)
                    continue;
                size_t src_index = defIndex * multipliers[0];
                size_t dst_index = obi * multipliers[0];
                emb_cycle(src_index, dst_index, 1);
            }
        }
    } // embeddingSegmentsSum

} // reference
} // runtime
} // ngraph
