// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include "openvino/frontend/ggml/decoder.hpp"

namespace ov {
namespace frontend {
namespace ggml {

InputModel::InputModel(const std::shared_ptr<GgmlDecoder>& gdecoder, bool naive)
    : m_decoder(gdecoder),
      m_naive(naive) {}

const std::shared_ptr<GgmlDecoder>& InputModel::get_model_decoder() const {
    return m_decoder;
}

bool InputModel::is_naive() const {
    return m_naive;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
