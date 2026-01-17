// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <sstream>
#include <vector>

#include "exceptions.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace detail {
using ::ONNX_NAMESPACE::TensorProto;
using MappedMemoryHandles = std::shared_ptr<std::map<std::string, std::shared_ptr<ov::MappedMemory>>>;
using LocalStreamHandles = std::shared_ptr<std::map<std::string, std::shared_ptr<std::ifstream>>>;

struct ExternalDataBlob {
    std::shared_ptr<void> owner;
    const char* data = nullptr;
    size_t length = 0;
};
/*
As
https://github.com/microsoft/onnxruntime/blob/4f6ae14e09729b3e3aba921de2e5bcc26d3e7768/onnxruntime/core/framework/tensorprotoutils.h#L206
describes, this is a special marker used to indicate the external weights is already in the shared memory from ORT. if
location field is set to this marker, the offset field contain the address of the memory.
*/
inline const std::string ORT_MEM_ADDR = "*/_ORT_MEM_ADDR_/*";
/// \brief  Helper class used to load tensor data from external files
class TensorExternalData {
public:
    TensorExternalData(const TensorProto& tensor);
    TensorExternalData(const std::string& location, size_t offset, size_t size);

    /// \brief      Load external data from tensor passed to constructor
    ///
    /// \note       If read data from external file fails,
    /// \note       If reading data from external files fails,
    ///             the invalid_external_data exception is thrown.
    ///
    /// \return     External binary data description
    ExternalDataBlob load_external_data(const std::string& model_dir) const;

    /// \brief      Map (mmap for lin, MapViewOfFile for win) external data from tensor passed to constructor
    ///
    /// \note       If read data from external file fails,
    /// \note       If reading data from external files fails,
    ///             the invalid_external_data exception is thrown.
    ///
    /// \return     External binary data description
    ExternalDataBlob load_external_mmap_data(const std::string& model_dir, MappedMemoryHandles cache) const;

    /// \brief      Load external data from existing shared memory when m_data_location is ORT_MEM_ADDR
    ///
    /// \note       If reading data from existing shared memory fails,
    ///             the invalid_external_data exception is thrown.
    ///
    /// \return     External binary data description
    ExternalDataBlob load_external_mem_data() const;

    /// \brief      Represets parameter of external data as string
    ///
    /// \return     State of TensorExternalData as string representation
    std::string to_string() const;

    /// \brief      Validates that referenced external data is accessible and satisfies
    ///             basic bounds requirements without reading the payload.
    ///
    /// \param[in]  model_dir  Directory that contains the ONNX model file. May be empty
    ///                         when the model path is unknown.
    void validate(const std::string& model_dir) const;

    /// \brief      Object contains a data length after construction. Method allows read-only access to this
    ///             information.
    ///
    /// \return     Returns a stored data size in bytes
    uint64_t size() const {
        return m_data_length;
    }

    /// \brief      Object contains a data location after construction. Method allows read-only access to this
    ///             information.
    ///
    /// \return     Returns a stored data location
    std::string data_location() const {
        return m_data_location;
    }

private:
    std::string m_data_location{};
    uint64_t m_offset = 0;
    uint64_t m_data_length = 0;
    std::string m_sha1_digest{};

    std::filesystem::path compose_full_path(const std::string& model_dir) const;
    [[noreturn]] void throw_invalid_external_data() const {
        throw error::invalid_external_data{"invalid external data: " + to_string()};
    }
};

inline TensorExternalData::TensorExternalData(const TensorProto& tensor) {
    for (const auto& entry : tensor.external_data()) {
        if (entry.key() == "location") {
            m_data_location = ov::util::sanitize_path(entry.value());
        } else if (entry.key() == "offset") {
            m_offset = std::stoull(entry.value());
        } else if (entry.key() == "length") {
            m_data_length = std::stoull(entry.value());
        } else if (entry.key() == "checksum") {
            m_sha1_digest = entry.value();
        }
    }
#ifdef ENABLE_OPENVINO_DEBUG
    if (!m_sha1_digest.empty()) {
        OPENVINO_WARN("SHA1 checksum is not supported");
    }
#endif
}

inline TensorExternalData::TensorExternalData(const std::string& location, size_t offset, size_t size)
    : m_data_location{location},
      m_offset{offset},
      m_data_length{size} {}

inline ExternalDataBlob TensorExternalData::load_external_mmap_data(const std::string& model_dir,
                                                                    MappedMemoryHandles cache) const {
    ExternalDataBlob blob{};
    const auto full_path = compose_full_path(model_dir);
    const auto full_path_str = ov::util::path_to_string(full_path);
    const int64_t file_size = ov::util::file_size(full_path);
    if (file_size <= 0 || m_offset + m_data_length > static_cast<uint64_t>(file_size)) {
        throw_invalid_external_data();
    }
    auto cached_mapped_memory = cache->find(full_path_str);
    std::shared_ptr<ov::MappedMemory> mapped_memory;
    if (cached_mapped_memory != cache->end()) {
        mapped_memory = cached_mapped_memory->second;
    } else {
        mapped_memory = ov::load_mmap_object(full_path_str);
        (*cache)[full_path_str] = mapped_memory;
    }
    if (m_data_length > mapped_memory->size() || mapped_memory->size() == 0) {
        throw_invalid_external_data();
    }
    blob.owner = mapped_memory;
    blob.data = mapped_memory->data() + m_offset;
    blob.length = m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset;
    return blob;
}

inline ExternalDataBlob TensorExternalData::load_external_data(const std::string& model_dir) const {
    ExternalDataBlob blob{};
    const auto full_path = compose_full_path(model_dir);
    std::ifstream external_data_stream(full_path, std::ios::binary | std::ios::in | std::ios::ate);

    if (external_data_stream.fail()) {
        throw_invalid_external_data();
    }
    const uint64_t file_size = static_cast<uint64_t>(external_data_stream.tellg());
    if (m_offset + m_data_length > file_size) {
        throw_invalid_external_data();
    }

    const uint64_t read_data_length = m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset;
    external_data_stream.seekg(m_offset, std::ios::beg);

    auto read_data = std::make_shared<std::vector<char>>(read_data_length);
    if (read_data_length > 0) {
        external_data_stream.read(read_data->data(), read_data_length);
    }
    external_data_stream.close();
    blob.owner = read_data;
    blob.data = read_data->data();
    blob.length = read_data_length;
    return blob;
}

inline ExternalDataBlob TensorExternalData::load_external_mem_data() const {
    ExternalDataBlob blob{};
    if (m_data_location != ORT_MEM_ADDR) {
        throw_invalid_external_data();
    }
    const bool is_valid_buffer = m_offset && m_data_length;
    const bool is_empty_buffer = (m_data_length == 0);
    if (!(is_valid_buffer || is_empty_buffer)) {
        throw_invalid_external_data();
    }
    auto addr_ptr = reinterpret_cast<char*>(m_offset);
    auto owned_memory = std::make_shared<std::vector<char>>(m_data_length);
    if (m_data_length > 0) {
        std::memcpy(owned_memory->data(), addr_ptr, m_data_length);
    }
    blob.owner = owned_memory;
    blob.data = owned_memory->data();
    blob.length = m_data_length;
    return blob;
}

inline std::string TensorExternalData::to_string() const {
    std::stringstream s;
    s << "ExternalDataInfo(";
    s << "data_full_path: " << m_data_location;
    s << ", offset: " << m_offset;
    s << ", data_length: " << m_data_length;
    if (!m_sha1_digest.empty()) {
        s << ", sha1_digest: " << m_sha1_digest << ")";
    } else {
        s << ")";
    }
    return s.str();
}

inline std::filesystem::path TensorExternalData::compose_full_path(const std::string& model_dir) const {
    if (model_dir.empty()) {
        return ov::util::make_path(m_data_location);
    }

    return std::filesystem::absolute(std::filesystem::weakly_canonical(
        ov::util::path_join({ov::util::make_path(model_dir), ov::util::make_path(m_data_location)})));
}

inline void TensorExternalData::validate(const std::string& model_dir) const {
    if (m_data_location.empty()) {
        throw_invalid_external_data();
    }

    if (m_data_location == ORT_MEM_ADDR) {
        const bool is_valid_buffer = m_offset && m_data_length;
        const bool is_empty_buffer = (m_data_length == 0);
        if (!(is_valid_buffer || is_empty_buffer)) {
            throw_invalid_external_data();
        }
        return;
    }

    const auto full_path = compose_full_path(model_dir);
    std::error_code ec;
    const auto status = std::filesystem::status(full_path, ec);
    if (ec || !std::filesystem::exists(status) || !std::filesystem::is_regular_file(status)) {
        throw_invalid_external_data();
    }

    const int64_t file_size = ov::util::file_size(full_path);
    if (file_size < 0) {
        throw_invalid_external_data();
    }
    const auto available_bytes = static_cast<uint64_t>(file_size);
    if (m_offset > available_bytes) {
        throw_invalid_external_data();
    }
    if (m_data_length > 0 && m_data_length > (available_bytes - m_offset)) {
        throw_invalid_external_data();
    }
}


}  // namespace detail
}  // namespace onnx
}  // namespace frontend
}  // namespace ov

