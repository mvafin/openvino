// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF container parser. Replaces the third-party gguflib dependency: the file is
// memory-mapped via ov::load_mmap_object and parsed directly per the GGUF v2/v3 format
// (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). No llama.cpp / ggml
// dependency.

#include "gguf.hpp"

#include <cstdint>
#include <cstring>
#include <optional>

#include "openvino/core/except.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace frontend {
namespace ggml {

template <typename... Args>
std::string format(std::string fmt, Args... args) {
    int n = std::snprintf(nullptr, 0, fmt.c_str(), args...);
    OPENVINO_ASSERT(n >= 0, "[load_gguf] formatting error");
    std::string out(static_cast<size_t>(n) + 1, '\0');
    std::snprintf(out.data(), out.size(), fmt.c_str(), args...);
    out.resize(static_cast<size_t>(n));
    return out;
}

// Explicit instantiation for the name patterns used by the reader and builders.
template std::string format(std::string, int);

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" little-endian
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;

// (items_per_block, bytes_per_block) per gguf_tensor_type, indexed by the type id.
struct TypeTraits {
    uint32_t items_per_block;
    uint32_t bytes_per_block;
};

TypeTraits type_traits(uint32_t type) {
    switch (type) {
    case GGUF_TYPE_F32:
        return {1, 4};
    case GGUF_TYPE_F16:
        return {1, 2};
    case GGUF_TYPE_Q4_0:
        return {32, 18};
    case GGUF_TYPE_Q4_1:
        return {32, 20};
    case GGUF_TYPE_Q5_0:
        return {32, 22};
    case GGUF_TYPE_Q5_1:
        return {32, 24};
    case GGUF_TYPE_Q8_0:
        return {32, 34};
    case GGUF_TYPE_Q8_1:
        return {32, 40};
    case GGUF_TYPE_Q2_K:
        return {256, 82};
    case GGUF_TYPE_Q3_K:
        return {256, 110};
    case GGUF_TYPE_Q4_K:
        return {256, 144};
    case GGUF_TYPE_Q5_K:
        return {256, 176};
    case GGUF_TYPE_Q6_K:
        return {256, 210};
    case GGUF_TYPE_Q8_K:
        return {256, 292};
    case GGUF_TYPE_I8:
        return {1, 1};
    case GGUF_TYPE_I16:
        return {1, 2};
    case GGUF_TYPE_I32:
        return {1, 4};
    case GGUF_TYPE_I64:
        return {1, 8};
    case GGUF_TYPE_F64:
        return {1, 8};
    case GGUF_TYPE_BF16:
        return {1, 2};
    default:
        return {0, 0};
    }
}

std::optional<ov::element::Type> gguf_type_to_dtype(uint32_t gguf_type) {
    switch (gguf_type) {
    case GGUF_TYPE_F64:
        return ov::element::f64;
    case GGUF_TYPE_F32:
        return ov::element::f32;
    case GGUF_TYPE_F16:
        return ov::element::f16;
    case GGUF_TYPE_BF16:
        return ov::element::bf16;
    case GGUF_TYPE_I8:
        return ov::element::i8;
    case GGUF_TYPE_I16:
        return ov::element::i16;
    case GGUF_TYPE_I32:
        return ov::element::i32;
    case GGUF_TYPE_I64:
        return ov::element::i64;
    default:
        return std::nullopt;
    }
}

// Sequential little-endian reader over the mmaped file. Bounds-checked on every read.
class Cursor {
public:
    Cursor(const uint8_t* data, uint64_t size) : m_data(data), m_size(size) {}

    uint64_t offset() const {
        return m_off;
    }
    const uint8_t* ptr() const {
        return m_data + m_off;
    }

    template <typename T>
    T read() {
        OPENVINO_ASSERT(m_off + sizeof(T) <= m_size, "[load_gguf] unexpected end of file");
        T v;
        std::memcpy(&v, m_data + m_off, sizeof(T));
        m_off += sizeof(T);
        return v;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        OPENVINO_ASSERT(m_off + len <= m_size, "[load_gguf] string runs past end of file");
        std::string s(reinterpret_cast<const char*>(m_data + m_off), static_cast<size_t>(len));
        m_off += len;
        return s;
    }

    void skip(uint64_t n) {
        OPENVINO_ASSERT(m_off + n <= m_size, "[load_gguf] skip past end of file");
        m_off += n;
    }

private:
    const uint8_t* m_data;
    uint64_t m_size;
    uint64_t m_off = 0;
};

size_t value_type_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
    case GGUF_VALUE_TYPE_INT8:
    case GGUF_VALUE_TYPE_BOOL:
        return 1;
    case GGUF_VALUE_TYPE_UINT16:
    case GGUF_VALUE_TYPE_INT16:
        return 2;
    case GGUF_VALUE_TYPE_UINT32:
    case GGUF_VALUE_TYPE_INT32:
    case GGUF_VALUE_TYPE_FLOAT32:
        return 4;
    case GGUF_VALUE_TYPE_UINT64:
    case GGUF_VALUE_TYPE_INT64:
    case GGUF_VALUE_TYPE_FLOAT64:
        return 8;
    default:
        return 0;  // string / array handled separately
    }
}

ov::element::Type value_type_to_dtype(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
        return ov::element::u8;
    case GGUF_VALUE_TYPE_INT8:
        return ov::element::i8;
    case GGUF_VALUE_TYPE_UINT16:
        return ov::element::u16;
    case GGUF_VALUE_TYPE_INT16:
        return ov::element::i16;
    case GGUF_VALUE_TYPE_UINT32:
        return ov::element::u32;
    case GGUF_VALUE_TYPE_INT32:
        return ov::element::i32;
    case GGUF_VALUE_TYPE_FLOAT32:
        return ov::element::f32;
    case GGUF_VALUE_TYPE_BOOL:
        return ov::element::boolean;
    case GGUF_VALUE_TYPE_UINT64:
        return ov::element::u64;
    case GGUF_VALUE_TYPE_INT64:
        return ov::element::i64;
    case GGUF_VALUE_TYPE_FLOAT64:
        return ov::element::f64;
    default:
        OPENVINO_THROW("[load_gguf] unexpected scalar metadata type ", type);
    }
}

// Read a single metadata value of `type` at the cursor into `out`.
void read_metadata_value(Cursor& cur, uint32_t type, GGUFMetaData& out) {
    if (type == GGUF_VALUE_TYPE_STRING) {
        out = cur.read_string();
        return;
    }
    if (type == GGUF_VALUE_TYPE_ARRAY) {
        uint32_t elem_type = cur.read<uint32_t>();
        uint64_t len = cur.read<uint64_t>();
        OPENVINO_ASSERT(elem_type != GGUF_VALUE_TYPE_ARRAY, "[load_gguf] nested arrays are not supported.");
        if (elem_type == GGUF_VALUE_TYPE_STRING) {
            std::vector<std::string> strs(len);
            for (auto& s : strs) {
                s = cur.read_string();
            }
            out = std::move(strs);
            return;
        }
        auto dtype = value_type_to_dtype(elem_type);
        ov::Tensor t(dtype, ov::Shape{static_cast<size_t>(len)});
        const size_t nbytes = static_cast<size_t>(len) * value_type_size(elem_type);
        std::memcpy(t.data(), cur.ptr(), nbytes);
        cur.skip(nbytes);
        out = std::move(t);
        return;
    }
    // Scalar: store as a shape-{} ov::Tensor of the right element type.
    auto dtype = value_type_to_dtype(type);
    ov::Tensor t(dtype, ov::Shape(0));
    const size_t nbytes = value_type_size(type);
    std::memcpy(t.data(), cur.ptr(), nbytes);
    cur.skip(nbytes);
    out = std::move(t);
}

// Copy a non-quantized tensor into an ov::Tensor. Quantized tensors go through
// gguf_load_quantized instead.
ov::Tensor extract_tensor_data(const gguf_tensor& tensor) {
    auto dtype = gguf_type_to_dtype(tensor.type);
    OPENVINO_ASSERT(dtype.has_value(),
                    "[load_gguf] tensor '",
                    std::string(tensor.name, tensor.namelen),
                    "' has unsupported non-quantized type ",
                    tensor.type);
    auto shape = get_shape(tensor);
    ov::Tensor weights(dtype.value(), shape);
    std::memcpy(weights.data(), tensor.weights_data, tensor.num_weights * dtype.value().size());
    return weights;
}

float metadata_to_float(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    auto tensor = std::get<ov::Tensor>(metadata.at(key));
    return *(tensor.data<ov::element_type_traits<ov::element::f32>::value_type>());
}

int metadata_to_int(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    auto tensor = std::get<ov::Tensor>(metadata.at(key));
    // GGUF stores these counts as u32; reinterpret as i32 (values fit comfortably).
    return static_cast<int>(*(tensor.data<ov::element_type_traits<ov::element::u32>::value_type>()));
}

}  // namespace

ov::Shape get_shape(const gguf_tensor& tensor) {
    ov::Shape shape;
    // GGUF stores dimensions fastest-varying first; the logical (GGML) order is reversed.
    for (int i = static_cast<int>(tensor.ndim) - 1; i >= 0; i--) {
        shape.push_back(tensor.dim[i]);
    }
    return shape;
}

GGUFLoad get_gguf_data(const std::string& file) {
    std::unordered_map<std::string, GGUFMetaData> metadata;
    std::unordered_map<std::string, ov::Tensor> arrays;
    std::unordered_map<std::string, gguf_tensor_type> qtype;

    auto mapped = ov::load_mmap_object(file);
    OPENVINO_ASSERT(mapped && mapped->data(), "[load_gguf] failed to mmap '", file, "'");
    const auto* base = reinterpret_cast<const uint8_t*>(mapped->data());
    const uint64_t fsize = mapped->size();

    Cursor cur(base, fsize);

    // ---- Header ----
    uint32_t magic = cur.read<uint32_t>();
    OPENVINO_ASSERT(magic == GGUF_MAGIC, "[load_gguf] '", file, "' is not a GGUF file (bad magic)");
    uint32_t version = cur.read<uint32_t>();
    OPENVINO_ASSERT(version == 2 || version == 3, "[load_gguf] unsupported GGUF version ", version);
    uint64_t tensor_count = cur.read<uint64_t>();
    uint64_t kv_count = cur.read<uint64_t>();

    // ---- Metadata kv pairs ----
    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key = cur.read_string();
        uint32_t vtype = cur.read<uint32_t>();
        auto& slot = metadata.insert({key, GGUFMetaData{}}).first->second;
        read_metadata_value(cur, vtype, slot);
    }

    uint64_t alignment = GGUF_DEFAULT_ALIGNMENT;
    if (auto it = metadata.find("general.alignment"); it != metadata.end()) {
        if (auto* t = std::get_if<ov::Tensor>(&it->second)) {
            alignment = *(t->data<ov::element_type_traits<ov::element::u32>::value_type>());
        }
    }

    // ---- Tensor info section ----
    struct TensorInfo {
        std::string name;
        uint32_t type = 0;
        uint32_t ndim = 0;
        uint64_t dim[4] = {1, 1, 1, 1};
        uint64_t offset = 0;
    };
    std::vector<TensorInfo> infos(tensor_count);
    for (uint64_t i = 0; i < tensor_count; i++) {
        TensorInfo& ti = infos[i];
        ti.name = cur.read_string();
        ti.ndim = cur.read<uint32_t>();
        OPENVINO_ASSERT(ti.ndim <= 4, "[load_gguf] tensor '", ti.name, "' has unsupported ndim ", ti.ndim);
        for (uint32_t d = 0; d < ti.ndim; d++) {
            ti.dim[d] = cur.read<uint64_t>();
        }
        ti.type = cur.read<uint32_t>();
        ti.offset = cur.read<uint64_t>();
    }

    // Tensor data starts at the next `alignment`-aligned offset after the info section.
    uint64_t data_off = cur.offset();
    if (uint64_t rem = data_off % alignment) {
        data_off += alignment - rem;
    }

    // ---- Materialize tensors ----
    for (const auto& ti : infos) {
        gguf_tensor tensor;
        tensor.name = ti.name.data();
        tensor.namelen = ti.name.size();
        tensor.type = ti.type;
        tensor.ndim = ti.ndim;
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < ti.ndim; d++) {
            tensor.dim[d] = ti.dim[d];
            nelem *= ti.dim[d];
        }
        tensor.num_weights = nelem;
        tensor.offset = ti.offset;

        auto tr = type_traits(ti.type);
        OPENVINO_ASSERT(tr.bytes_per_block != 0, "[load_gguf] tensor '", ti.name, "' has unsupported type ", ti.type);
        tensor.bsize = (nelem / tr.items_per_block) * tr.bytes_per_block;

        uint64_t abs_off = data_off + ti.offset;
        OPENVINO_ASSERT(abs_off + tensor.bsize <= fsize, "[load_gguf] tensor '", ti.name, "' data runs past EOF");
        tensor.weights_data = base + abs_off;

        const std::string& name = ti.name;
        if (ti.type == GGUF_TYPE_Q4_0 || ti.type == GGUF_TYPE_Q4_1 || ti.type == GGUF_TYPE_Q8_0 ||
            ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q6_K) {
            gguf_load_quantized(arrays, qtype, tensor);
        } else {
            ov::Tensor loaded = extract_tensor_data(tensor);
            OPENVINO_ASSERT(arrays.emplace(name, loaded).second, "[load_gguf] duplicate tensor name '", name, "'");
            constexpr std::string_view weight_suffix = ".weight";
            if (name.size() >= weight_suffix.size()) {
                const std::string name_prefix = name.substr(0, name.length() - weight_suffix.length());
                qtype.emplace(name_prefix + ".qtype", static_cast<gguf_tensor_type>(ti.type));
            }
        }
    }

    return {metadata, arrays, qtype};
}

std::map<std::string, GGUFMetaData> config_from_meta(const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    std::map<std::string, GGUFMetaData> config;
    auto arch = std::get<std::string>(metadata.at("general.architecture"));
    config["architecture"] = arch;
    config["layer_num"] = metadata_to_int(metadata, arch + ".block_count");
    config["head_num"] = metadata_to_int(metadata, arch + ".attention.head_count");
    config["head_size"] = metadata.count(arch + ".attention.key_length")
                              ? metadata_to_int(metadata, arch + ".attention.key_length")
                              : (metadata_to_int(metadata, arch + ".embedding_length") /
                                 metadata_to_int(metadata, arch + ".attention.head_count"));
    config["head_num_kv"] = metadata.count(arch + ".attention.head_count_kv")
                                ? metadata_to_int(metadata, arch + ".attention.head_count_kv")
                                : metadata_to_int(metadata, arch + ".attention.head_count");
    config["hidden_size"] = metadata_to_int(metadata, arch + ".embedding_length");
    config["max_position_embeddings"] =
        metadata.count(arch + ".context_length") ? metadata_to_int(metadata, arch + ".context_length") : 2048;
    config["rms_norm_eps"] = metadata_to_float(metadata, arch + ".attention.layer_norm_rms_epsilon");
    config["rope_freq_base"] =
        metadata.count(arch + ".rope.freq_base") ? metadata_to_float(metadata, arch + ".rope.freq_base") : 10000.0f;
    config["file_type"] = metadata_to_int(metadata, "general.file_type");
    return config;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
