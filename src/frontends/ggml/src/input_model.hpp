// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/input_model.hpp>

#include "openvino/frontend/ggml/decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::ggml::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgmlDecoder>& gdecoder, bool naive = false);

    const std::shared_ptr<GgmlDecoder>& get_model_decoder() const;

    bool is_naive() const;

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
