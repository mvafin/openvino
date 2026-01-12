// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/onnx/graph_iterator_util.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "openvino/util/log.hpp"

namespace ov {
namespace frontend {
namespace onnx {

bool is_graph_iterator_enabled() {
    const char* env_value = std::getenv("ONNX_ITERATOR");
    if (env_value == nullptr) {
        return true;  // Enabled by default
    }

    std::string value(env_value);
    // Remove whitespace
    value.erase(std::remove_if(value.begin(),
                               value.end(),
                               [](unsigned char ch) {
                                   return std::isspace(ch);
                               }),
                value.end());
    // Convert to lowercase
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    static const std::unordered_map<std::string, bool> valid_values = {{"1", true},
                                                                       {"true", true},
                                                                       {"on", true},
                                                                       {"enable", true},
                                                                       {"0", false},
                                                                       {"false", false},
                                                                       {"off", false},
                                                                       {"disable", false}};

    auto it = valid_values.find(value);
    if (it != valid_values.end()) {
        if (!it->second) {
            OPENVINO_WARN(
                "DEPRECATED: Disabling ONNX graph iterator via ONNX_ITERATOR environment variable is deprecated and "
                "will be removed in a future release. The graph iterator will become mandatory.");
        }
        return it->second;
    }

    // Unknown value - print error and default to enabled
    OPENVINO_WARN("Unknown value for ONNX_ITERATOR environment variable: '",
                  env_value,
                  "'. "
                  "Expected 1 (enable) or 0 (disable). "
                  "Defaulting to enabled.");
    return true;
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
