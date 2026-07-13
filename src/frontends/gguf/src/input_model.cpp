// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include "openvino/frontend/gguf/decoder.hpp"

namespace ov {
namespace frontend {
namespace gguf {

InputModel::InputModel(const std::shared_ptr<GgufDecoder>& gdecoder, bool naive)
    : m_decoder(gdecoder),
      m_naive(naive) {}

const std::map<std::string, std::shared_ptr<ov::Node>>& InputModel::get_model_inputs() const {
    return m_decoder->get_model_inputs();
}

std::vector<std::string> InputModel::get_model_output_names() const {
    return m_decoder->get_model_output_names();
}

RopeConfig InputModel::get_rope_config() const {
    return m_decoder->get_attribute("rope_config").as<RopeConfig>();
}

void InputModel::visit_subgraph(const std::function<void(std::shared_ptr<GgufDecoder>)>& node_visitor) const {
    m_decoder->visit_subgraph(node_visitor);
}

bool InputModel::is_naive() const {
    return m_naive;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
