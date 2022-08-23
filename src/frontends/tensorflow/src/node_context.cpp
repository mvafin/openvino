// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow/node_context.hpp"

using namespace ov::frontend::tensorflow;

 ov::Any NodeContext::apply_additional_conversion_rules(const ov::Any& data,
                                                       const std::type_info& type_info) const override {
    if (data.is<EmptyList>(){
        switch (type_info) {
        case typeid(std::vector<int64_t>):
            return std::vector<int64_t>();
        case typeid(std::vector<float>):
            return std::vector<float>();
        case typeid(std::vector<std::string>):
            return std::vector<std::string>();
        case typeid(std::vector<bool>):
            return std::vector<bool>();
        case typeid(std::vector<ov::PartialShape>):
            return std::vector<ov::PartialShape>();
        case typeid(std::vector<ov::element::Type>):
            return std::vector<ov::element::Type>();
        default:
            FRONT_END_GENERAL_CHECK(false,
                                    "Could not decode empty list attribute for ",
                                    get_name(),
                                    " node. Provided type is not known.");
        }
    }
    // no conversion rules found
    return data;
}
