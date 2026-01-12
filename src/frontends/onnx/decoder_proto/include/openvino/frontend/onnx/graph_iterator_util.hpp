// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace frontend {
namespace onnx {

/**
 * @brief Checks if graph iterator is enabled via ONNX_ITERATOR environment variable
 * @return true if enabled (default), false if explicitly disabled
 */
bool is_graph_iterator_enabled();

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
