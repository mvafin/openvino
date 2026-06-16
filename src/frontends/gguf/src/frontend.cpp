// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/frontend.hpp"

#include <fstream>

#include "builder/gguf_builder.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "input_model.hpp"
#include "op_table.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/frontend/common/path_util.hpp"
#include "openvino/frontend/gguf/decoder.hpp"
#include "translate_session.hpp"

namespace ov {
namespace frontend {
namespace gguf {

struct FrontEnd::Impl {
    std::unordered_map<std::string, CreatorFunction> op_extension_translators;
    std::vector<ConversionExtensionBase::Ptr> conversion_extensions;
    TelemetryExtension::Ptr telemetry;
};

namespace {

std::unordered_map<std::string, CreatorFunction> merged_ops(
    const std::unordered_map<std::string, CreatorFunction>& ext_translators) {
    auto ops = get_supported_ops();
    for (const auto& ext : ext_translators) {
        ops[ext.first] = ext.second;
    }
    return ops;
}

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

FrontEnd::FrontEnd() : m_impl(std::make_unique<Impl>()) {}
FrontEnd::~FrontEnd() = default;

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto ggml_model = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    {
        auto ops = merged_ops(m_impl->op_extension_translators);
        TranslateSession translate_session(model, ops, ggml_model->is_naive());
        converted_model = translate_session.get_converted_model();
    }
    return converted_model;
}

std::string FrontEnd::get_name() const {
    return "gguf";
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto conv_ext = ov::as_type_ptr<ov::frontend::ConversionExtension>(extension)) {
        m_impl->conversion_extensions.push_back(conv_ext);
        // Wrap the base CreatorFunction (takes ov::frontend::NodeContext) into a
        // ggml::CreatorFunction (takes ggml::NodeContext). ggml::NodeContext IS a
        // ov::frontend::NodeContext so the call is safe.
        m_impl->op_extension_translators[conv_ext->get_op_type()] = [conv_ext](const NodeContext& ctx) {
            return conv_ext->get_converter()(ctx);
        };
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_extensions.push_back(so_ext);
    } else if (const auto& telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_impl->telemetry = telemetry;
    } else if (auto op_base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(extension)) {
        for (const auto& attached_ext : op_base_ext->get_attached_extensions()) {
            add_extension(attached_ext);
        }
    }
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // Two accepted inputs:
    //  1. a GgufDecoder (the llama.cpp cgraph path passes one in directly), or
    //  2. a path to a .gguf file (the OpenVINO-native path; sniff the GGUF magic).
    if (variants.empty()) {
        return false;
    }
    if (variants[0].is<std::shared_ptr<GgufDecoder>>()) {
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
                            "GGUF Frontend requires at least one parameter in model representation.");

    // Path 1: a GgufDecoder passed in directly (e.g. llama.cpp cgraph decoder).
    if (variants[0].is<std::shared_ptr<GgufDecoder>>()) {
        auto decoder = variants[0].as<std::shared_ptr<GgufDecoder>>();
        FRONT_END_GENERAL_CHECK(decoder, "Couldn't cast ov::Any to std::shared_ptr<GgufDecoder>");
        return std::make_shared<InputModel>(decoder);
    }

    // Path 2: a .gguf file path -> native builder -> GgufBuilderDecoder.
    if (auto path = ov::frontend::get_path_from_any(variants[0])) {
        std::filesystem::path model_path = std::move(*path);
        FRONT_END_GENERAL_CHECK(model_path.extension() == ".gguf",
                                "GGUF Frontend file loading expects a .gguf file, got: ",
                                model_path.string());
        auto graph = build_ggml_graph_from_gguf(model_path.string());
        auto decoder = std::make_shared<GgufBuilderDecoder>(graph);
        return std::make_shared<InputModel>(decoder);
    }

    FRONT_END_GENERAL_CHECK(false,
                            "GGUF Frontend doesn't support the provided model representation. Provide a GgufDecoder "
                            "or a path to a .gguf file.");
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
