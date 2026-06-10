// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/ggml/frontend.hpp"

#include <fstream>

#include "builder/gguf_builder.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/frontend/common/path_util.hpp"
#include "openvino/frontend/ggml/decoder.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace ggml {

namespace {

// True if the file at `path` begins with the GGUF magic ("GGUF").
bool has_gguf_magic(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    char magic[4] = {};
    f.read(magic, sizeof(magic));
    return f.gcount() == 4 && magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F';
}

}  // namespace

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
    // Two accepted inputs:
    //  1. a GgmlDecoder (the llama.cpp cgraph path passes one in directly), or
    //  2. a path to a .gguf file (the OpenVINO-native path; sniff the GGUF magic).
    if (variants.empty()) {
        return false;
    }
    if (variants[0].is<std::shared_ptr<GgmlDecoder>>()) {
        return true;
    }
    if (auto path = ov::frontend::get_path_from_any(variants[0])) {
        std::filesystem::path model_path = std::move(*path);
        return model_path.extension() == ".gguf" && has_gguf_magic(model_path);
    }
    return false;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    FRONT_END_GENERAL_CHECK(!variants.empty(),
                            "GGML Frontend requires at least one parameter in model representation.");

    // Path 1: a GgmlDecoder passed in directly (e.g. llama.cpp cgraph decoder).
    if (variants[0].is<std::shared_ptr<GgmlDecoder>>()) {
        auto decoder = variants[0].as<std::shared_ptr<GgmlDecoder>>();
        FRONT_END_GENERAL_CHECK(decoder, "Couldn't cast ov::Any to std::shared_ptr<GgmlDecoder>");
        return std::make_shared<ggml::InputModel>(decoder);
    }

    // Path 2: a .gguf file path -> native builder -> GgufBuilderDecoder.
    if (auto path = ov::frontend::get_path_from_any(variants[0])) {
        std::filesystem::path model_path = std::move(*path);
        FRONT_END_GENERAL_CHECK(model_path.extension() == ".gguf",
                                "GGML Frontend file loading expects a .gguf file, got: ",
                                model_path.string());
        auto graph = build_ggml_graph_from_gguf(model_path.string());
        auto decoder = std::make_shared<GgufBuilderDecoder>(graph);
        return std::make_shared<ggml::InputModel>(decoder);
    }

    FRONT_END_GENERAL_CHECK(false,
                            "GGML Frontend doesn't support the provided model representation. Provide a GgmlDecoder "
                            "or a path to a .gguf file.");
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
