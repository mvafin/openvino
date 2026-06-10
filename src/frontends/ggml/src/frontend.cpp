// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/ggml/frontend.hpp"

#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/frontend/ggml/decoder.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto ggml_model = std::dynamic_pointer_cast<ggml::InputModel>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    const auto& supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, ggml_model->is_naive());
        converted_model = translate_session.get_converted_model();
    }
    return converted_model;
}

std::string FrontEnd::get_name() const {
    return "ggml";
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // The GGML frontend currently accepts a GgmlDecoder produced either by the llama.cpp
    // cgraph path or by the OpenVINO-native .gguf builder. File-path loading is added in a
    // later milestone.
    if (variants.size() != 1) {
        return false;
    }
    return variants[0].is<std::shared_ptr<GgmlDecoder>>();
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    FRONT_END_GENERAL_CHECK(variants.size() == 1,
                            "GGML Frontend supports exactly one parameter in model representation, got ",
                            variants.size(),
                            " instead.");
    FRONT_END_GENERAL_CHECK(variants[0].is<std::shared_ptr<GgmlDecoder>>(),
                            "GGML Frontend doesn't support provided model type. Please provide a GgmlDecoder.");
    auto decoder = variants[0].as<std::shared_ptr<GgmlDecoder>>();
    FRONT_END_GENERAL_CHECK(decoder, "Couldn't cast ov::Any to std::shared_ptr<GgmlDecoder>");
    return std::make_shared<ggml::InputModel>(decoder);
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
