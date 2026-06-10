// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/ggml/visibility.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class GGML_FRONTEND_API FrontEnd : public ov::frontend::FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd();

    /// \brief Completely convert the input model, producing a fully converted OV Model.
    /// \param model Input model
    /// \return fully converted OV Model
    std::shared_ptr<Model> convert(const InputModel::Ptr& model) const override;

    /// \brief Gets name of this FrontEnd. Can be used by clients
    /// if frontend is selected automatically by FrontEndManager::load_by_model
    /// \return GGML frontend name.
    std::string get_name() const override;

protected:
    /// \brief Check if FrontEnd can recognize model from given parts
    /// \param variants Either a GgmlDecoder (cgraph / gguf-builder path) or a path to a
    /// .gguf file.
    /// \return true if the frontend can load the model
    bool supported_impl(const std::vector<ov::Any>& variants) const override;

    /// \brief Load the input model from a GgmlDecoder or a .gguf file path.
    /// \param variants Either a GgmlDecoder or a path to a .gguf file.
    /// \return InputModel::Ptr
    InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
