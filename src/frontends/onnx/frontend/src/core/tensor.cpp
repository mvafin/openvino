// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/tensor.hpp"

#include "input_model.hpp"

namespace ov {
namespace frontend {
namespace onnx {

namespace {
template <typename T>
std::vector<T> read_data_from_tensor_place(const std::shared_ptr<TensorONNXPlace>& tensor_place) {
    const auto& tensor = tensor_place->get_tensor();
    FRONT_END_GENERAL_CHECK(tensor,
                            "TensorPlace tensor data is null while accessing constant data");
    const T* begin = tensor.data<T>();
    return std::vector<T>(begin, begin + tensor.get_size());
}
}  // namespace

detail::MappedMemoryHandles TensorONNXPlace::get_mmap_cache() {
    const auto model_onnx = dynamic_cast<const unify::InputModel*>(&m_input_model);
    return model_onnx->get_mmap_cache();
}
detail::LocalStreamHandles TensorONNXPlace::get_stream_cache() {
    const auto model_onnx = dynamic_cast<const unify::InputModel*>(&m_input_model);
    return model_onnx->get_stream_cache();
}

std::filesystem::path TensorONNXPlace::get_model_dir() const {
    const auto model_onnx = dynamic_cast<const unify::InputModel*>(&m_input_model);
    if (!model_onnx) {
        return {};
    }
    return model_onnx->get_model_dir();
}

Tensor::Tensor(const std::shared_ptr<TensorONNXPlace>& tensor_place) {
    m_tensor_proto = nullptr;
    m_shape = tensor_place->get_partial_shape().get_shape();
    m_model_dir = tensor_place->get_model_dir();
    m_mmap_cache = tensor_place->get_mmap_cache();
    m_tensor_place = tensor_place;
}

detail::ExternalDataBlob Tensor::load_external_blob() const {
    const auto ext_data = detail::TensorExternalData(*m_tensor_proto);
    if (ext_data.data_location() == detail::ORT_MEM_ADDR) {
        return ext_data.load_external_mem_data();
    }
    if (m_mmap_cache) {
        return ext_data.load_external_mmap_data(m_model_dir, m_mmap_cache);
    }
    return ext_data.load_external_data(m_model_dir);
}

template <>
std::vector<double> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<double>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<double>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<double>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_DOUBLE) {
        return detail::__get_data<double>(m_tensor_proto->double_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "DOUBLE, raw data");
}

template <>
std::vector<float> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<float>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<float>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<float>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_FLOAT) {
        return detail::__get_data<float>(m_tensor_proto->float_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT, raw data");
}

template <>
std::vector<ov::float16> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<ov::float16>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<ov::float16>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<ov::float16>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_FLOAT16) {
        using std::begin;
        using std::end;

        const auto& int32_data = m_tensor_proto->int32_data();
        std::vector<ov::float16> float16_data;
        float16_data.reserve(int32_data.size());
        std::transform(begin(int32_data), end(int32_data), std::back_inserter(float16_data), [](int32_t elem) {
            return ov::float16::from_bits(static_cast<uint16_t>(elem));
        });

        return detail::__get_data<ov::float16>(float16_data);
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT16, raw data");
}

template <>
std::vector<ov::bfloat16> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<ov::bfloat16>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<ov::bfloat16>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<ov::bfloat16>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_BFLOAT16) {
        const auto& int32_data = m_tensor_proto->int32_data();
        std::vector<ov::bfloat16> bf16_data;
        bf16_data.reserve(int32_data.size());
        std::transform(int32_data.begin(),
                       int32_data.end(),
                       std::back_inserter(bf16_data),
                       [](int32_t elem) {
                           return ov::bfloat16::from_bits(static_cast<uint16_t>(elem));
                       });
        return detail::__get_data<ov::bfloat16>(bf16_data);
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT32, raw data");
}

template <>
std::vector<int8_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<int8_t>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<int8_t>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<int8_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT8 ||
        m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT4) {
        return detail::__get_data<int8_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT4, INT8, raw data");
}

template <>
std::vector<int16_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<int16_t>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<int16_t>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<int16_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT16) {
        return detail::__get_data<int16_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT16, raw data");
}

template <>
std::vector<int32_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<int32_t>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<int32_t>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<int32_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT32) {
        return detail::__get_data<int32_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT32, raw data");
}

template <>
std::vector<int64_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<int64_t>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<int64_t>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<int64_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_INT64) {
        return detail::__get_data<int64_t>(m_tensor_proto->int64_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "INT64, raw data");
}

template <>
std::vector<uint8_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<uint8_t>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<uint8_t>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<uint8_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT8 ||
        m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT4) {
        return detail::__get_data<uint8_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "UINT4, UINT8, raw data");
}

template <>
std::vector<uint16_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<uint16_t>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<uint16_t>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<uint16_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT16) {
        return detail::__get_data<uint16_t>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "UINT16, raw data");
}

template <>
std::vector<uint32_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<uint32_t>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<uint32_t>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<uint32_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT32) {
        return detail::__get_data<uint32_t>(m_tensor_proto->uint64_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "UINT32, raw data");
}

template <>
std::vector<uint64_t> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<uint64_t>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<uint64_t>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<uint64_t>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_UINT64) {
        return detail::__get_data<uint64_t>(m_tensor_proto->uint64_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "UINT64, raw data");
}

template <>
std::vector<ov::float8_e4m3> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<ov::float8_e4m3>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<ov::float8_e4m3>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<ov::float8_e4m3>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN) {
        using std::begin;
        using std::end;

        const auto& int32_data = m_tensor_proto->int32_data();
        std::vector<ov::float8_e4m3> float8_data;
        float8_data.reserve(int32_data.size());
        std::transform(begin(int32_data), end(int32_data), std::back_inserter(float8_data), [](int32_t elem) {
            return ov::float8_e4m3::from_bits(static_cast<uint8_t>(elem));
        });

        return detail::__get_data<ov::float8_e4m3>(float8_data);
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT8E4M3, raw data");
}

template <>
std::vector<ov::float8_e5m2> Tensor::get_data() const {
    if (has_external_data()) {
        return get_external_data<ov::float8_e5m2>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<ov::float8_e5m2>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<ov::float8_e5m2>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2) {
        using std::begin;
        using std::end;

        const auto& int32_data = m_tensor_proto->int32_data();
        std::vector<ov::float8_e5m2> float8_data;
        float8_data.reserve(int32_data.size());
        std::transform(begin(int32_data), end(int32_data), std::back_inserter(float8_data), [](int32_t elem) {
            return ov::float8_e5m2::from_bits(static_cast<uint8_t>(elem));
        });

        return detail::__get_data<ov::float8_e5m2>(float8_data);
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "FLOAT8E5M2, raw data");
}

template <>
std::vector<char> Tensor::get_data() const {
    // Boolean values are stored as char because std::vector<bool>
    // can behave differently from other vector containers.
    if (has_external_data()) {
        return get_external_data<char>();
    }
    if (m_tensor_place != nullptr) {
        return read_data_from_tensor_place<char>(m_tensor_place);
    }
    if (m_tensor_proto->has_raw_data()) {
        return detail::__get_raw_data<char>(m_tensor_proto->raw_data(), m_tensor_proto->data_type());
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_BOOL) {
        return detail::__get_data<char>(m_tensor_proto->int32_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "BOOL, raw data");
}

template <>
std::vector<std::string> Tensor::get_data() const {
    if (has_external_data()) {
        FRONT_END_THROW("External strings are not supported");
    }
    if (m_tensor_place != nullptr) {
        const auto& tensor = m_tensor_place->get_tensor();
        if (tensor) {
            const std::string* begin = tensor.data<std::string>();
            return std::vector<std::string>(begin, begin + tensor.get_size());
        }
        FRONT_END_NOT_IMPLEMENTED(get_data);
    }
    if (m_tensor_proto->has_raw_data()) {
        FRONT_END_THROW("Loading strings from raw data isn't supported");
    }
    if (m_tensor_proto->data_type() == TensorProto_DataType::TensorProto_DataType_STRING) {
        return detail::__get_data<std::string>(m_tensor_proto->string_data());
    }
    ONNX_INVALID_DATA_TYPE(m_tensor_proto->data_type(), "STRING");
}

std::shared_ptr<ov::op::v0::Constant> Tensor::get_ov_constant() const {
    std::shared_ptr<ov::op::v0::Constant> constant{nullptr};
    if (m_tensor_proto != nullptr && m_tensor_proto->has_segment()) {
        FRONT_END_THROW("Loading segments isn't supported");
    }
    ov::element::Type ov_type = get_ov_type();
    size_t element_count = 0;
    if (m_tensor_place != nullptr) {
        element_count = ov::shape_size(m_shape);
    } else {
        element_count = get_data_size();
        if (ov::element::is_nibble_type(ov_type)) {
            element_count *= 2;  // Each byte contains 2 data items
            if (shape_size(m_shape) % 2) {
                // Odd elements
                element_count--;
            }
        }
    }
    if (m_tensor_place != nullptr) {
        const auto& place_tensor = m_tensor_place->get_tensor();
        FRONT_END_GENERAL_CHECK(place_tensor,
                                "TensorPlace tensor is null for initializer '" + get_name() + "'");
        constant = std::make_shared<ov::op::v0::Constant>(place_tensor);
    } else if (has_external_data()) {
        const auto blob = load_external_blob();
        const void* data_ptr = blob.data;
        if (ov::shape_size(m_shape) != 0 && data_ptr == nullptr) {
            throw error::invalid_external_data("External data blob is empty for initializer '" + get_name() + "'");
        }
        constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, data_ptr, blob.owner);
        element_count = constant->get_byte_size() / ov_type.size();
        if (ov::element::is_nibble_type(ov_type)) {
            element_count *= 2;  // Each byte contains 2 data items, so byte size must be multiplied
            if (ov::shape_size(m_shape) % 2) {
                element_count--;
            }
        }
        if (element_count != ov::shape_size(m_shape) ||
            (blob.length != 0 && constant->get_byte_size() != blob.length)) {
            throw error::invalid_external_data(
                "The size of the external data file does not match the byte size of an initializer '" + get_name() +
                "' in the model");
        }
    } else if (element_count == shape_size(m_shape) && m_tensor_proto != nullptr) {
        switch (m_tensor_proto->data_type()) {
        case TensorProto_DataType::TensorProto_DataType_FLOAT:
        case TensorProto_DataType::TensorProto_DataType_DOUBLE:
        case TensorProto_DataType::TensorProto_DataType_INT32:
        case TensorProto_DataType::TensorProto_DataType_INT64:
        case TensorProto_DataType::TensorProto_DataType_UINT32:
        case TensorProto_DataType::TensorProto_DataType_UINT64:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data_ptr());
            break;
        case TensorProto_DataType::TensorProto_DataType_INT4:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<int8_t>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_INT8:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<int8_t>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_INT16:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<int16_t>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_UINT4:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<uint8_t>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_UINT8:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<uint8_t>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_UINT16:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<uint16_t>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_BOOL:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<char>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<ov::bfloat16>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_FLOAT16:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<ov::float16>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<ov::float8_e4m3>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<ov::float8_e5m2>().data());
            break;
        case TensorProto_DataType::TensorProto_DataType_STRING:
            constant = std::make_shared<ov::op::v0::Constant>(ov_type, m_shape, get_data<std::string>().data());
            break;
        default:
            ONNX_UNSUPPORTED_DATA_TYPE(
                m_tensor_proto->data_type(),
                "BOOL, BFLOAT16, FLOAT8E4M3FN, FLOAT8E5M2, FLOAT, FLOAT16, DOUBLE, INT4, INT8, INT16, INT32, INT64, "
                "UINT4, UINT8, UINT16, UINT32, UINT64, STRING");
        }
    } else if (element_count == 0 && m_shape.size() == 0) {
        constant = common::make_failsafe_constant(ov_type);
    } else {
        FRONT_END_THROW("Tensor shape doesn't match data size: " + std::to_string(element_count) + " vs " +
                        std::to_string(ov::shape_size(m_shape)));
    }

    if (m_tensor_proto != nullptr && m_tensor_proto->has_name()) {
        constant->set_friendly_name(get_name());
    }
    if (m_tensor_place != nullptr) {
        const auto& names = m_tensor_place->get_names();
        if (names.size() > 0) {
            constant->set_friendly_name(names[0]);
            constant->get_default_output().set_names({names.begin(), names.end()});
        }
    }
    return constant;
}

void ov::frontend::onnx::TensorONNXPlace::translate(ov::Output<ov::Node>& output) {
    if (get_names().size() > 0) {
        output.add_names({*get_names().begin()});
    }
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
