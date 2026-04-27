// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Header-only GGUF v3 binary reader, internal to the GGUF frontend.
// Uses only OpenVINO + standard C++.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// GGUF metadata value type tags (see ggml/docs/gguf.md).
enum gguf_type : uint32_t {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ggml tensor element types. Numeric ids match ggml/ggml-common.h.
// (Subset relevant to this PoC: covers FP and all mainstream block quants.)
enum ggml_type : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_BF16 = 30,
};

struct MetaValue;
using MetaArray = std::vector<MetaValue>;
struct MetaValue {
    std::variant<std::monostate, bool, int64_t, uint64_t, double, std::string, MetaArray> v;

    template <typename T>
    T get() const {
        if constexpr (std::is_same_v<T, std::string>) {
            return std::get<std::string>(v);
        } else if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(std::get<double>(v));
        } else if constexpr (std::is_signed_v<T>) {
            if (auto p = std::get_if<int64_t>(&v))
                return static_cast<T>(*p);
            if (auto p = std::get_if<uint64_t>(&v))
                return static_cast<T>(*p);
            OPENVINO_THROW("GGUF metadata: not an integer");
        } else {
            if (auto p = std::get_if<uint64_t>(&v))
                return static_cast<T>(*p);
            if (auto p = std::get_if<int64_t>(&v))
                return static_cast<T>(*p);
            OPENVINO_THROW("GGUF metadata: not an unsigned integer");
        }
    }
};

struct TensorDescriptor {
    std::string name;
    std::vector<uint64_t> dims;  // already reversed to row-major / numpy order
    ggml_type type;
    uint64_t offset;  // bytes into the tensor data block
};

class GGUFFile {
public:
    explicit GGUFFile(const std::string& path) {
        m_mmap = ov::load_mmap_object(path);
        OPENVINO_ASSERT(m_mmap, "GGUF: failed to mmap file: ", path);
        m_base = reinterpret_cast<const uint8_t*>(m_mmap->data());
        m_size = m_mmap->size();
        parse_();
    }

    const std::map<std::string, MetaValue>& metadata() const {
        return m_meta;
    }
    const std::vector<TensorDescriptor>& tensors() const {
        return m_tensors;
    }

    const uint8_t* tensor_raw(const TensorDescriptor& t) const {
        return m_base + m_tensor_data_offset + t.offset;
    }

private:
    std::shared_ptr<ov::MappedMemory> m_mmap;
    const uint8_t* m_base = nullptr;
    size_t m_size = 0;
    size_t m_cursor = 0;
    size_t m_tensor_data_offset = 0;
    std::map<std::string, MetaValue> m_meta;
    std::vector<TensorDescriptor> m_tensors;

    template <typename T>
    T read_() {
        OPENVINO_ASSERT(m_cursor + sizeof(T) <= m_size, "GGUF: unexpected EOF");
        T v;
        std::memcpy(&v, m_base + m_cursor, sizeof(T));
        m_cursor += sizeof(T);
        return v;
    }
    std::string read_string_() {
        uint64_t n = read_<uint64_t>();
        OPENVINO_ASSERT(m_cursor + n <= m_size, "GGUF: unexpected EOF in string");
        std::string s(reinterpret_cast<const char*>(m_base + m_cursor), n);
        m_cursor += n;
        return s;
    }
    MetaValue read_value_(uint32_t t) {
        MetaValue m;
        switch (t) {
        case GGUF_TYPE_UINT8:
            m.v = static_cast<uint64_t>(read_<uint8_t>());
            break;
        case GGUF_TYPE_INT8:
            m.v = static_cast<int64_t>(read_<int8_t>());
            break;
        case GGUF_TYPE_UINT16:
            m.v = static_cast<uint64_t>(read_<uint16_t>());
            break;
        case GGUF_TYPE_INT16:
            m.v = static_cast<int64_t>(read_<int16_t>());
            break;
        case GGUF_TYPE_UINT32:
            m.v = static_cast<uint64_t>(read_<uint32_t>());
            break;
        case GGUF_TYPE_INT32:
            m.v = static_cast<int64_t>(read_<int32_t>());
            break;
        case GGUF_TYPE_FLOAT32:
            m.v = static_cast<double>(read_<float>());
            break;
        case GGUF_TYPE_BOOL:
            m.v = static_cast<bool>(read_<uint8_t>());
            break;
        case GGUF_TYPE_STRING:
            m.v = read_string_();
            break;
        case GGUF_TYPE_UINT64:
            m.v = read_<uint64_t>();
            break;
        case GGUF_TYPE_INT64:
            m.v = read_<int64_t>();
            break;
        case GGUF_TYPE_FLOAT64:
            m.v = read_<double>();
            break;
        case GGUF_TYPE_ARRAY: {
            uint32_t et = read_<uint32_t>();
            uint64_t n = read_<uint64_t>();
            MetaArray arr;
            for (uint64_t i = 0; i < n; ++i)
                arr.push_back(read_value_(et));
            m.v = std::move(arr);
            break;
        }
        default:
            OPENVINO_THROW("GGUF: unknown value type ", t);
        }
        return m;
    }

    void parse_() {
        OPENVINO_ASSERT(m_size >= 24, "GGUF: file too small");
        OPENVINO_ASSERT(m_base[0] == 'G' && m_base[1] == 'G' && m_base[2] == 'U' && m_base[3] == 'F',
                        "GGUF: bad magic (not a GGUF file)");
        m_cursor = 4;
        uint32_t version = read_<uint32_t>();
        OPENVINO_ASSERT(version == 3, "GGUF: unsupported version ", version, " (PoC supports v3 only)");
        uint64_t n_tensors = read_<uint64_t>();
        uint64_t n_meta = read_<uint64_t>();

        for (uint64_t i = 0; i < n_meta; ++i) {
            std::string key = read_string_();
            uint32_t t = read_<uint32_t>();
            m_meta[std::move(key)] = read_value_(t);
        }

        m_tensors.reserve(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorDescriptor td;
            td.name = read_string_();
            uint32_t nd = read_<uint32_t>();
            td.dims.resize(nd);
            for (uint32_t k = 0; k < nd; ++k)
                td.dims[k] = read_<uint64_t>();
            std::reverse(td.dims.begin(), td.dims.end());
            td.type = static_cast<ggml_type>(read_<uint32_t>());
            td.offset = read_<uint64_t>();
            m_tensors.push_back(std::move(td));
        }

        uint64_t alignment = 32;
        if (auto it = m_meta.find("general.alignment"); it != m_meta.end()) {
            alignment = it->second.get<uint64_t>();
        }
        size_t pad = (alignment - (m_cursor % alignment)) % alignment;
        m_cursor += pad;
        m_tensor_data_offset = m_cursor;
    }
};

// Sniff the first 4 bytes for the GGUF magic without touching the rest of the file.
inline bool is_gguf(const std::string& path) {
    try {
        auto m = ov::load_mmap_object(path);
        if (!m || m->size() < 4)
            return false;
        const auto* b = reinterpret_cast<const uint8_t*>(m->data());
        return b[0] == 'G' && b[1] == 'G' && b[2] == 'U' && b[3] == 'F';
    } catch (...) {
        return false;
    }
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
