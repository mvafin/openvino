// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "op/global_average_pool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector global_average_pool(const Node& node)
                {
                    auto data = node.get_ng_inputs()[0];
                    auto data_rank = data.get_partial_shape().rank();

                    NGRAPH_CHECK(data_rank.is_static(),
                                 "The input data tensor's rank has to be known (static)");

                    auto data_rank_value = data_rank.get_length();

                    NGRAPH_CHECK(data_rank_value > 2,
                                 "The input data tensor's rank has to be greater than 2."
                                 "Provided data rank is: ",
                                 data_rank_value);

                    // Generate axes for reduce operation which contain all spatial dims indexes.
                    // Examples:
                    // Input shape: [N, C, H, W]
                    // Input spatial dimensions are H and W
                    // Expected spatial dims indexes: [2, 3]
                    //
                    // Input shape: [N, C, H, W, D]
                    // Input spatial dimensions are H, W and D
                    // Expected spatial dims indexes: [2, 3, 4]
                    uint64_t data_spatial_rank = data_rank_value - 2;
                    auto reduce_axes_vector = std::vector<std::int64_t>(data_spatial_rank);
                    std::iota(reduce_axes_vector.begin(), reduce_axes_vector.end(), 2);
                    auto reduce_axes = default_opset::Constant::create(
                        element::i64, Shape{data_spatial_rank}, reduce_axes_vector);

                    return {std::make_shared<default_opset::ReduceMean>(data, reduce_axes, true)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
