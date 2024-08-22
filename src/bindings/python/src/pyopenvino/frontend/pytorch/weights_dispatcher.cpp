// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <fstream>

#include "openvino/op/constant.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "utils.hpp"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

namespace py = pybind11;

// using namespace ov::frontend::pytorch;

using Buffer = std::vector<std::uint8_t>;
using BufferPtr = std::shared_ptr<Buffer>;
using ConstantMap = std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>;

using namespace ov::op;

BufferPtr read_file_helper(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    OPENVINO_ASSERT(file.is_open(), "Cannot open file ", filename);

    size_t filesize = file.tellg();
    auto buffer = std::make_shared<std::vector<std::uint8_t>>();
    buffer->reserve(filesize);
    file.seekg(0, std::ios::beg);
    // FIXME: Use mmapped AlignedBuffer as ov::Core::read_model can do, necessary functionality is not available in
    // public OV API
    std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), std::back_inserter(*buffer));

    return buffer;
}

ov::element::Type safetensors_to_ov_element_type(int dtype) {
    switch (dtype) {
    case SAFETENSORS_F32:
        return ov::element::f32;
    case SAFETENSORS_F16:
        return ov::element::f16;
    case SAFETENSORS_BF16:
        return ov::element::bf16;
    default:
        OPENVINO_THROW("Not supported safetensors dtype: ", dtype);
    }
}

std::shared_ptr<ov::op::v0::Constant> make_constant_from_safetensors(const std::string& filename,
                                                                     const std::string& name) {
    auto mapped_memory = ov::load_mmap_object(filename);
    auto weights = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data(),
                                                                                         mapped_memory->size(),
                                                                                         mapped_memory);
    safetensors_File safe_tensors_file = {0};
    OPENVINO_ASSERT(!safetensors_file_init(mapped_memory->data(), mapped_memory->size(), &safe_tensors_file),
                    "Cannot parse ",
                    filename,
                    " using safetensors");

    auto idx = safetensors_lookup(&safe_tensors_file, name.c_str());
    OPENVINO_ASSERT(idx != -1);
    auto tensor_desc = safe_tensors_file.tensors[idx];
    ov::Shape shape(tensor_desc.shape, tensor_desc.shape + tensor_desc.n_dimensions);
    auto type = safetensors_to_ov_element_type(tensor_desc.dtype);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        (char*)tensor_desc.ptr,
        tensor_desc.end_offset_bytes - tensor_desc.begin_offset_bytes,
        mapped_memory);
    auto constant = std::make_shared<ov::op::v0::Constant>(type, shape, buffer);
    // Link MMAP to Constant to hold the memory
    constant->get_rt_info()["__safetensors_buffer_holder"] = mapped_memory;
    std::cout << "Created Constant with name: " << name << " from file: " << filename << std::endl;
    return constant;
}

ConstantMap read_safetensors(const std::string& filename) {
    ConstantMap tensors;
    auto buffer = read_file_helper(filename);
    safetensors_File safe_tensors_file = {0};
    OPENVINO_ASSERT(safetensors_file_init(&(*buffer)[0], buffer->size(), &safe_tensors_file) == nullptr,
                    "Cannot parse ",
                    filename,
                    " using safetensors");
    std::cout << "Opened " << filename << " as safetensors file format, it contains " << safe_tensors_file.num_tensors
              << " tensors" << std::endl;
    for (int i = 0; i < safe_tensors_file.num_tensors; i++) {
        safetensors_TensorDescriptor tensor = safe_tensors_file.tensors[i];
        std::string name(tensor.name.ptr, tensor.name.ptr + tensor.name.len);
        ov::Shape shape(tensor.shape, tensor.shape + tensor.n_dimensions);
        void* ptr = tensor.ptr;  // FIXME: needs a non-constant pointer because Tensor doesn't accept a constant pointer
        OPENVINO_ASSERT(ov::shape_size(shape) <= tensor.end_offset_bytes - tensor.begin_offset_bytes,
                        " ",
                        ov::shape_size(shape),
                        " ",
                        tensor.end_offset_bytes - tensor.begin_offset_bytes);
        auto type = safetensors_to_ov_element_type(tensor.dtype);
        // FIXME: Extend OV with a new Constant ctor that shares memory to avoid two stage Tensor->Constant
        // initialization
        ov::Tensor wrapper(type, shape, ptr);                     // wraps existing memory, no ownership
        auto constant = std::make_shared<v0::Constant>(wrapper);  // wraps existing memory, no ownership
        constant->get_rt_info()["__safetensors_buffer_holder"] =
            buffer;  // to automatically deallocate underlying memory buffer when last constant holding it is destroyed
        // DEBUG_PRINT("Tensor with name " << name << ", shape " << shape << " and type " << type << " was allocated.");
        tensors[name] = constant;
    }
    free(safe_tensors_file.tensors);
    free(safe_tensors_file.metadata);
    return std::move(tensors);
}

class WeightsDispatcher {
public:
    WeightsDispatcher(const std::map<std::string, std::map<std::string, std::string>>& weights_index)
        : m_weights_index(weights_index) {}

    ov::OutputVector get_constant_for_name(const std::string& name) {
        auto weight_it = m_weights_index.find(name);
        OPENVINO_ASSERT(weight_it != m_weights_index.end());
        auto filename_it = weight_it->second.find("safetensors_file");
        OPENVINO_ASSERT(filename_it != weight_it->second.end());

        auto constant = make_constant_from_safetensors(filename_it->second, name);
        return {constant};
    }

private:
    std::map<std::string, std::map<std::string, std::string>> m_weights_index;
};

void regclass_frontend_pytorch_WeightsDispatcher(py::module m) {
    py::class_<WeightsDispatcher, std::shared_ptr<WeightsDispatcher>> wd(m, "WeightsDispatcher");
    wd.def(py::init([](const std::map<std::string, std::map<std::string, std::string>>& weights_index) {
        return std::make_shared<WeightsDispatcher>(weights_index);
    }));
    wd.def("get_constant_for_name", &WeightsDispatcher::get_constant_for_name);
}