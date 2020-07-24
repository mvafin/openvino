// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_interpolate_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_quantize_node.h"
#include "mkldnn_depthwise_node.h"
#include "mkldnn_activation_node.h"
#include <ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_layers_internal.hpp>
#include "ie_parallel.hpp"
#include <algorithm>

#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"
#include "jit_uni_quantization.hpp"
#include "common/simple_copy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;


#define GET_OFF(field) offsetof(jit_interpolate_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_interpolate_kernel_f32 : public jit_uni_interpolate_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_interpolate_kernel_f32)

    explicit jit_uni_interpolate_kernel_f32(jit_interpolate_config_params jcp, const mkldnn_primitive_attr &attr)
    : jit_uni_interpolate_kernel(jcp, attr), jit_generator() {
        const auto &p = attr_.post_ops_;
        for (int i = 0; i < p.len_; i++) {
            auto &post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this,
                        post_op.eltwise.alg,
                        post_op.eltwise.alpha,
                        post_op.eltwise.beta));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this,
                        post_op.depthwise.alg));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        if (attr_.post_ops_.len_ != 0)
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

        if (isa == cpu::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        switch (jcp_.layout) {
            case LayoutType::planar: {
                nn_planar();
                break;
            }
            case LayoutType::block: {
                nn_blk();
                break;
            }
            case LayoutType::by_channel: {
                nn_by_channel();
                break;
            }
            default:
                assert(!"unsupported layout type for interpolate layer with nearest neighbor mode.");
        }

        this->postamble();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_work_amount = r13;
    Xbyak::Reg64 reg_index = r14;
    Xbyak::Reg64 reg_src_aux = r15;
    Xbyak::Reg64 reg_params = abi_param1;

    Reg8 reg_tmp_8 = r10b;
    Reg32 reg_tmp_32 = r10d;
    Reg64 reg_tmp_64 = r10;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rcx;
    Xbyak::Reg32 reg_index_oc = edx;

    Vmm vmm_val = Vmm(0);
    Xmm xmm_val = Xmm(0);
    Vmm vmm_index = Vmm(1);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_mask = Vmm(3);
    Vmm vmm_d_weights = Vmm(4);
    Vmm vmm_d_bias = Vmm(5);

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    void nn_planar() {
        int step = vlen / sizeof(float);

        Xbyak::Label nn_loop_label;
        Xbyak::Label nn_loop_end_label;
        Xbyak::Label nn_tail_loop_label;
        Xbyak::Label nn_tail_loop_end_label;
        L(nn_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(nn_loop_end_label, T_NEAR);

            uni_vmovdqu(vmm_index, ptr[reg_index]);
            uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
            vgatherdps(vmm_val, ptr[reg_src + vmm_index], vmm_mask);
            if (attr_.post_ops_.len_ != 0)
                apply_post_ops(jcp_.dst_dt, 1);
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_index, step * sizeof(int));
            sub(reg_work_amount, step);

            jmp(nn_loop_label, T_NEAR);
        }
        L(nn_loop_end_label);

        step = 1;
        L(nn_tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(nn_tail_loop_end_label, T_NEAR);

            mov(reg_src_aux, reg_src);
            mov(reg_index_oc, dword[reg_index]);
            add(reg_src_aux, reg_index_oc);

            load_scalar(xmm_val, ptr[reg_src_aux], jcp_.src_dt);
            if (attr_.post_ops_.len_ != 0)
                apply_post_ops(jcp_.dst_dt, 1);
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_index, step * sizeof(int));
            sub(reg_work_amount, step);

            jmp(nn_tail_loop_label, T_NEAR);
        }
        L(nn_tail_loop_end_label);
    }

    void nn_blk() {
        int step = vlen / sizeof(float);
        if (isa == cpu::sse42)
            step *= 2;

        Xbyak::Label nn_loop_label;
        Xbyak::Label nn_loop_end_label;
        L(nn_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(nn_loop_end_label, T_NEAR);

            mov(reg_src_aux, reg_src);
            mov(reg_index_oc, dword[reg_index]);
            add(reg_src_aux, reg_index_oc);

            load_vector(vmm_val, ptr[reg_src_aux], jcp_.src_dt);
            if (attr_.post_ops_.len_ != 0)
                apply_post_ops(jcp_.dst_dt, 0);
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            if (isa == cpu::sse42) {
                int sse42_offset = 4;
                add(reg_src_aux, sse42_offset * jcp_.src_data_size);
                load_vector(vmm_val, ptr[reg_src_aux], jcp_.src_dt);
                if (attr_.post_ops_.len_ != 0) {
                    add(reg_oc_off, sse42_offset * sizeof(float));
                    apply_post_ops(jcp_.dst_dt, 0);
                    sub(reg_oc_off, sse42_offset * sizeof(float));
                }
                store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
            }

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_index, 1 * jcp_.indices_size);
            sub(reg_work_amount, 1);

            jmp(nn_loop_label, T_NEAR);
        }
        L(nn_loop_end_label);
    }

    void nn_by_channel() {
        int step = vlen / sizeof(float);

        Xbyak::Label nn_loop_label;
        Xbyak::Label nn_loop_end_label;
        Xbyak::Label nn_tail_loop_label;
        Xbyak::Label nn_tail_loop_end_label;
        L(nn_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(nn_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            if (attr_.post_ops_.len_ != 0)
                apply_post_ops(jcp_.dst_dt, 0);
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_src, step * jcp_.src_data_size);
            add(reg_oc_off, step * sizeof(float));
            sub(reg_work_amount, step);

            jmp(nn_loop_label, T_NEAR);
        }
        L(nn_loop_end_label);

        step = 1;
        L(nn_tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(nn_tail_loop_end_label, T_NEAR);

            load_scalar(xmm_val, ptr[reg_src], jcp_.src_dt);
            if (attr_.post_ops_.len_ != 0)
                apply_post_ops(jcp_.dst_dt, 0);
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            add(reg_dst, step * jcp_.dst_data_size);
            add(reg_src, step * jcp_.src_data_size);
            add(reg_oc_off, step * sizeof(float));
            sub(reg_work_amount, step);

            jmp(nn_tail_loop_label, T_NEAR);
        }
        L(nn_tail_loop_end_label);
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(vmm_src, op);
                break;
            case memory::s8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case memory::u8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            case memory::bf16:
                if (isa != cpu::sse42) {
                    vpmovzxwd(vmm_src, op);
                } else {
                    pmovzxwd(vmm_src, op);
                }
                uni_vpslld(vmm_src, vmm_src, 16);
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != memory::f32 && src_dt != data_type::bf16)
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                movss(xmm_src, op);
                break;
            case memory::s8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::u8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::bf16:
                pinsrw(xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32 && src_dt != data_type::bf16) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());

        if (dst_dt == memory::f32) {
            uni_vmovups(op, vmm_dst);
        } else if (dst_dt == memory::u8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::avx512_common) {
                vpmaxsd(vmm_dst, vmm_dst, vmm_zero);
                vpmovusdb(op, vmm_dst);
            } else {
                uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::sse42)
                    vpermq(ymm_dst, ymm_dst, 0x08);
                uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
        } else if (dst_dt == memory::s8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::avx512_common) {
                vpmovsdb(op, vmm_dst);
            } else {
                uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::sse42)
                    vpermq(ymm_dst, ymm_dst, 0x08);
                uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::sse42)
                    vmovq(op, xmm_dst);
                else
                    movd(op, xmm_dst);
            }
        } else if (dst_dt == memory::bf16) {
            if (isa == cpu::avx512_common) {
                if (mayiuse(avx512_core_bf16)) {
                    vcvtneps2bf16(ymm_dst, vmm_dst);
                    uni_vmovups(op, ymm_dst);
                } else {
                    vpsrad(vmm_dst, vmm_dst, 16);
                    vpmovdw(op, vmm_dst);
                }
            } else if (isa == cpu::avx2) {
                pshuflw(vmm_dst, vmm_dst, 0x0d);  // ox ox ox ox ox ox ox ox --> ox ox xx oo ox ox xx oo  imm=0b00001101
                pshufhw(vmm_dst, vmm_dst, 0x0d);  // ox ox xx oo ox ox xx oo --> xx oo xx oo xx oo xx oo
                pshufd(vmm_dst, vmm_dst, 0x08);   // xx oo xx oo xx oo xx oo --> xx xx oo oo xx xx oo oo  imm=0b00001000
                vpermq(ymm_dst, ymm_dst, 0x08);   // xx xx oo oo xx xx oo oo --> xx xx xx xx oo oo oo oo
                uni_vmovups(op, xmm_dst);
            } else {
                pshuflw(vmm_dst, vmm_dst, 0x0d);  // ox ox ox ox --> ox ox xx oo  imm=0b00001101
                pshufhw(vmm_dst, vmm_dst, 0x0d);  // ox ox xx oo --> xx oo xx oo
                pshufd(vmm_dst, vmm_dst, 0x08);   // xx oo xx oo --> xx xx oo oo  imm=0b00001000
                vmovq(op, xmm_dst);
            }
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (dst_dt != data_type::f32 && dst_dt != data_type::bf16) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                movss(op, xmm_dst);
                break;
            case memory::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::bf16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                pextrw(op, xmm_dst, 0x0);
            default:
                assert(!"unknown dst_dt");
        }
    }

    // scalar: load scalar to xmm, process on xmm with padded param, store xmm to scalar.
    // is_broadcast for broadcasting param for depth_wise and quantize(channel-sensitive post-ops), for fusion with plain layout.
    void apply_post_ops(memory::data_type dst_dt, bool is_broadcast) {
        const auto &p = attr_.post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        for (int i = 0; i < p.len_; i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data));
                add(reg_d_weights, reg_oc_off);
                add(reg_d_bias, reg_oc_off);
                // weight and bias is padded. scalar as vector.
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1, reg_d_weights, reg_d_bias, is_broadcast);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_dt == memory::f32 || i != p.len_ - 1;

                int s_idx = vmm_val.getIdx();

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

                if (do_dequantization) {
                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx, s_idx + 1, 0, 0, is_broadcast);
                }

                quantization_inj_idx++;
            }
        }
    }
};

MKLDNNInterpolateNode::MKLDNNInterpolateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNInterpolateNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 3)
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' has incorrect number of input edges";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' has incorrect number of output edges";

    auto *layer = getCnnLayer().get();
    std::string modeString = layer->GetParamAsString("mode");
    if (modeString == "nearest") {
        mode = InterpolateMode::nearest;
    } else if (modeString == "linear") {
        mode = InterpolateMode::linear;
    } else if (modeString == "linear_onnx") {
        mode = InterpolateMode::linear_onnx;
    } else if (modeString == "cubic") {
        mode = InterpolateMode::cubic;
    } else {
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' do not support interpolate mode:" << modeString;
    }

    modeString = layer->GetParamAsString("coordinate_transformation_mode", "half_pixel");
    if (modeString == "half_pixel") {
        coordTransMode = CoordTransMode::half_pixel;
    } else if (modeString == "pytorch_half_pixel") {
        coordTransMode = CoordTransMode::pytorch_half_pixel;
    } else if (modeString == "asymmetric") {
        coordTransMode = CoordTransMode::asymmetric;
    } else if (modeString == "tf_half_pixel_for_nn") {
        coordTransMode = CoordTransMode::tf_half_pixel_for_nn;
    } else if (modeString == "align_corners") {
        coordTransMode = CoordTransMode::align_corners;
    } else {
        THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' do not support coordinate transformation mode:" << modeString;
    }

    if (mode == InterpolateMode::nearest) {
        modeString = layer->GetParamAsString("nearest_mode", "round_prefer_floor");
        if (modeString == "round_prefer_floor") {
            nearestMode = NearestMode::round_prefer_floor;
        } else if (modeString == "round_prefer_ceil") {
            nearestMode = NearestMode::round_prefer_ceil;
        } else if (modeString == "floor") {
            nearestMode = NearestMode::floor;
        } else if (modeString == "ceil") {
            nearestMode = NearestMode::ceil;
        } else if (modeString == "simple") {
            nearestMode = NearestMode::simple;
        } else {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' do not support nearest round mode:" << modeString;
        }
    } else if (mode == InterpolateMode::cubic) {
        cubeCoeff = layer->GetParamAsFloat("cube_coeff");
    }
    antialias = layer->GetParamAsBool("antialias", false);
}

void MKLDNNInterpolateNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    if (getParentEdgeAt(0)->getDims().ndims() < 4 || getParentEdgeAt(0)->getDims().ndims() > 5) {
        return;
    }

    setPostOps(attr, true);

    Precision inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();

    if (!fusedWith.empty()) {
        auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
        if (lastFusedLayer) {
            outputPrecision = lastFusedLayer->outData[0]->getPrecision();
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    inputPrec = inputPrecision;
    outputPrec = outputPrecision;
    srcDataSize = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dstDataSize = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.outConfs[0].inPlace = -1;

    auto pushDesc = [&](memory::format format) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, format);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, format);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, format});
    };

    if (mode == InterpolateMode::nearest) {
        // blk and by_channel kernel only on sse42 or above machine
        if (mayiuse(cpu::sse42)) {
            if (getParentEdgeAt(0)->getDims().ndims() == 4) {
                pushDesc(memory::nhwc);
                if (mayiuse(cpu::avx512_common)) {
                    pushDesc(memory::nChw16c);
                } else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42)) {
                    pushDesc(memory::nChw8c);
                }
            } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
                pushDesc(memory::ndhwc);
                if (mayiuse(cpu::avx512_common)) {
                    pushDesc(memory::nCdhw16c);
                } else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42)) {
                    pushDesc(memory::nCdhw8c);
                }
            }
        }
        // planar always support. planar kernel for f32, else planar ref.
        pushDesc(MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims()));
    }
    if (mode == InterpolateMode::linear) {
        if (getParentEdgeAt(0)->getDims().ndims() == 4) {
            pushDesc(memory::nchw);
        } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
            pushDesc(memory::ncdhw);
        }
    }
}

void MKLDNNInterpolateNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    auto selectedPD = getSelectedPrimitiveDescriptor();
    Layout selected_layout = selectedPD->getConfig().inConfs[0].desc.getLayout();
    auto jcp = jit_interpolate_config_params();
    jcp.src_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[0].desc.getPrecision());
    jcp.dst_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].desc.getPrecision());
    jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.src_dt);
    jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.dst_dt);
    jcp.indices_size = sizeof(int);
    if (MKLDNNMemory::GetPlainLayout(getChildEdgeAt(0)->getDims()) == selected_layout) {
        jcp.layout = LayoutType::planar;
    } else if ((selected_layout == NHWC) || (selected_layout == NDHWC)) {
        jcp.layout = LayoutType::by_channel;
    } else {
        jcp.layout = LayoutType::block;
    }

    if (mode == InterpolateMode::nearest) {
        if (mayiuse(cpu::avx512_common)) {
            if (jcp.layout == LayoutType::planar) {
                interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::avx2>(jcp, *attr.get()));
                blk_size = 8;
            } else {
                interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::avx512_common>(jcp, *attr.get()));
                blk_size = 16;
            }
        } else if (mayiuse(cpu::avx2)) {
            interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::avx2>(jcp, *attr.get()));
            blk_size = 8;
        } else if (mayiuse(cpu::sse42) && !jcp.layout == LayoutType::planar) {
            interpolateKernel.reset(new jit_uni_interpolate_kernel_f32<cpu::sse42>(jcp, *attr.get()));
            blk_size = 8;
        }
    }

    const auto &p = (*attr.get()).post_ops_;
    for (int i = 0; i < p.len_; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors_ref.push_back(std::make_shared<ref_eltwise_scalar_fwd_t>(
                post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta));
        } else if (post_op.is_depthwise()) {
            depthwise_injectors_ref.push_back(std::make_shared<ref_depthwise_scalar_fwd_t>(
                    post_op.depthwise.alg));
        }
    }
}

void MKLDNNInterpolateNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights) {
    int blob_idx = 0;
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            quantizeNode->appendPostOps(ops);
            continue;
        }

        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(node.get());
        if (depthwiseNode) {
            if (initWeights) {
                auto* depthwiseLayer = reinterpret_cast<WeightableLayer*>(depthwiseNode->getCnnLayer().get());
                MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(getChildEdgeAt(0)->getDims()[1], 16))});

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx]->Create(depthwiseDims, memory::data_type::f32, memory::format::x);

                PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x,
                                                        depthwiseLayer->_weights->buffer(),
                                                        depthwiseLayer->_weights->size() *
                                                        MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                if (depthwiseNode->isBroadcast()) {
                    float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[0];
                    for (int i = 1; i < PostOpsIntBlobMemory[blob_idx]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                        static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[i] = broadcastValue;
                    }
                }

                if (depthwiseNode->getAlgorithm() == depthwise_scale_shift) {
                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    PostOpsIntBlobMemory[blob_idx + 1]->Create(depthwiseDims, memory::data_type::f32,
                                                               memory::format::x);
                    PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x,
                                                                depthwiseLayer->_biases->buffer(),
                                                                depthwiseLayer->_biases->size() *
                                                                MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                    if (depthwiseNode->isBroadcast()) {
                        float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[0];
                        for (int i = 1; i < PostOpsIntBlobMemory[blob_idx + 1]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                            static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[i] = broadcastValue;
                        }
                    }

                    ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                    blob_idx += 2;
                }
            } else {
                ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                     nullptr,
                                     nullptr);
            }

            continue;
        }

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(), activationNode->getBeta());

            continue;
        }

        THROW_IE_EXCEPTION << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}


void MKLDNNInterpolateNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();

    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();

    const auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());

    SizeVector src_dim = getParentEdgeAt(0)->getDesc().getDims();
    SizeVector dst_dim = getChildEdgeAt(0)->getDesc().getDims();

    size_t dims_size = src_dim.size();
    size_t N = src_dim[0];
    size_t C = src_dim[1];
    size_t ID = (dims_size == 5) ? src_dim[dims_size - 3] : 1lu;
    size_t IH = src_dim[dims_size - 2];
    size_t IW = src_dim[dims_size - 1];

    size_t OD = (dims_size == 5) ? dst_dim[dims_size - 3] : 1lu;
    size_t OH = dst_dim[dims_size - 2];
    size_t OW = dst_dim[dims_size - 1];

    float fx = static_cast<float>(IW) / static_cast<float>(OW);
    float fy = static_cast<float>(IH) / static_cast<float>(OH);
    float fz = static_cast<float>(ID) / static_cast<float>(OD);

    if (mode == InterpolateMode::nearest) {
        if (layout == NCHW || layout == NCDHW) {
            NearestNeighbor_PLN(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
        } else {
            if (outputPrec == Precision::U8) {
                auto dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetData());
                if (inputPrec == Precision::U8) {
                    auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<uint8_t, uint8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (inputPrec == Precision::I8) {
                    auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<int8_t, uint8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (inputPrec == Precision::FP32) {
                    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<float, uint8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                }
            } else if (outputPrec == Precision::I8) {
                auto dst_data = reinterpret_cast<int8_t *>(dstMemPtr->GetData());
                if (inputPrec == Precision::U8) {
                    auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<uint8_t, int8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (inputPrec == Precision::I8) {
                    auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<int8_t, int8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (inputPrec == Precision::FP32) {
                    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<float, int8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                }
            } else if (outputPrec == Precision::FP32) {
                auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
                if (inputPrec == Precision::U8) {
                    auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<uint8_t, float>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (inputPrec == Precision::I8) {
                    auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<int8_t, float>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                } else if (inputPrec == Precision::FP32) {
                    auto src_data = reinterpret_cast<float *>(srcMemPtr->GetData());
                    NearestNeighbor_BLK<float, float>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW);
                }
            }
        }
    } else if (mode == InterpolateMode::linear) {
        // currently no fusion, the input and output precision is the same
        // TODO bf16 pass
        bool isDownsample = (fx > 1) || (fy > 1) || (fz > 1);
        int kernel_width = 2;
        if (inputPrec == Precision::U8) {
            auto src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData());
            auto dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetData());
            LinearInterpolation<uint8_t, uint8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
        } else if (inputPrec == Precision::I8) {
            auto src_data = reinterpret_cast<const int8_t *>(srcMemPtr->GetData());
            auto dst_data = reinterpret_cast<int8_t *>(dstMemPtr->GetData());
            LinearInterpolation<int8_t, int8_t>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
        } else if (inputPrec == Precision::FP32) {
            auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
            auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
            LinearInterpolation<float, float>(src_data, dst_data, N, C, ID, IH, IW, fx, fy, fz, OD, OH, OW, kernel_width, isDownsample && antialias);
        }
    }
}

// input is f32(gatherISA works with fp32), fuse->output varies
void MKLDNNInterpolateNode::NearestNeighbor_PLN(const float *in_ptr_, float *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW) {
    std::vector<int> index_buffer(OD * OH * OW);
    bool isDDownsample = (fz > 1) ? true : false;
    bool isHDownsample = (fy > 1) ? true : false;
    bool isWDownsample = (fx > 1) ? true : false;
    int nearestVoxel = 0;
    for (int oz = 0; oz < OD; oz++) {
        float iz = coordTransToInput(oz, fz, ID, OD);
        nearestVoxel = nearestRound(iz, isDDownsample);
        int iz_offset = nearestVoxel * IH * IW;
        int oz_offset = oz * OH * OW;
        for (int oy = 0; oy < OH; oy++) {
            float iy = coordTransToInput(oy, fy, IH, OH);
            nearestVoxel = nearestRound(iy, isHDownsample);
            int iy_offset = nearestVoxel * IW + iz_offset;
            int oy_offset = oy * OW + oz_offset;
            for (int ox = 0; ox < OW; ox++) {
                float ix = coordTransToInput(ox, fx, IW, OW);
                nearestVoxel = nearestRound(ix, isWDownsample);
                int ix_index = nearestVoxel + iy_offset;
                index_buffer[oy_offset + ox] = ix_index * srcDataSize;
            }
        }
    }
    if (interpolateKernel) {
        parallel_for2d(B, C, [&](size_t b, size_t c) {
            const float *in_ptr = in_ptr_ + IW * IH * ID * C * b + IW * IH * ID * c;
            float *out_ptr = out_ptr_ + OW * OH * OD * C * b + OW * OH * OD * c;
            // for OW*OH*OD
            auto arg = jit_interpolate_call_args();
            arg.src = in_ptr;
            arg.dst = out_ptr;
            arg.index = static_cast<int*>(&index_buffer[0]);
            arg.oc_off = static_cast<size_t>(c);
            arg.work_amount = OW * OH * OD;
            (*interpolateKernel)(&arg);
        });
    }
}

// for ndhwc and nCdhw8/16d
// input may be f32/bf16/int8, fused->output varies
template <typename in_data_t, typename out_data_t>
void MKLDNNInterpolateNode::NearestNeighbor_BLK(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW) {
    std::vector<int> index_d(OD);
    std::vector<int> index_h(OH);
    std::vector<int> index_w(OW);
    bool isDDownsample = (fz > 1) ? true : false;
    bool isHDownsample = (fy > 1) ? true : false;
    bool isWDownsample = (fx > 1) ? true : false;
    for (int oz = 0; oz < OD; oz++) {
        float iz = coordTransToInput(oz, fz, ID, OD);
        index_d[oz] = nearestRound(iz, isDDownsample);
    }
    for (int oy = 0; oy < OH; oy++) {
        float iy = coordTransToInput(oy, fy, IH, OH);
        index_h[oy] = nearestRound(iy, isHDownsample);
    }
    for (int ox = 0; ox < OW; ox++) {
        float ix = coordTransToInput(ox, fx, IW, OW);
        index_w[ox] = nearestRound(ix, isWDownsample);
    }

    Layout layout = getParentEdgeAt(0)->getDesc().getLayout();
    bool is_nhwc = (layout == NHWC || layout == NDHWC) ? true : false;

    for (int b = 0; b < B; b++) {
        if (is_nhwc) {
            const in_data_t *in_ptr = in_ptr_ + IW * IH * ID * C * b;
            out_data_t *out_ptr = out_ptr_ + OW * OH * OD * C * b;
            if (interpolateKernel) {
                int tail = (C / blk_size) * blk_size;
                parallel_for2d(OD, OH, [&](size_t d, size_t h) {
                    // better that same core process continuous memory
                    out_data_t *out_ptr_dh = out_ptr + C * OW * OH * d + C * OW * h;
                    const in_data_t *in_ptr_dh = in_ptr + C * IW * IH * index_d[d] + C * IW * index_h[h];
                    auto arg = jit_interpolate_call_args();
                    for (int ox = 0; ox < OW; ox++) {
                        // kernel for OC
                        arg.dst = out_ptr_dh + C * ox;
                        arg.src = in_ptr_dh + C * index_w[ox];
                        arg.work_amount = C;
                        arg.oc_off = 0;
                        (*interpolateKernel)(&arg);
                    }
                });
            }
        } else {  // for nC(d)hw8/16c
            int CB = div_up(C, blk_size);
            const in_data_t *in_ptr = in_ptr_ + IW * IH * ID * CB * blk_size * b;
            out_data_t *out_ptr = out_ptr_ + OW * OH * OD * CB * blk_size * b;
            if (interpolateKernel) {
                std::vector<int> index_w_kernel(OW);
                for (int ox = 0; ox < OW; ox++) {
                    index_w_kernel[ox] = index_w[ox] * blk_size * sizeof(in_data_t);
                }
                parallel_for2d(CB, OD, [&](size_t cb, size_t d) {
                    out_data_t *out_ptr_cbd = out_ptr + blk_size * OW * OH * OD * cb + blk_size * OW * OH * d;
                    const in_data_t *in_ptr_cbd = in_ptr +  blk_size * IW * IH * ID * cb + blk_size * IW * IH * index_d[d];
                    auto arg = jit_interpolate_call_args();
                    for (int h = 0; h < OH; h++) {  // kernel for blk_size * OW
                        arg.dst = out_ptr_cbd + blk_size * OW * h;
                        arg.src = in_ptr_cbd + blk_size * IW * index_h[h];
                        arg.index = static_cast<int*>(&(index_w_kernel[0]));
                        arg.work_amount = static_cast<size_t>(OW);
                        arg.oc_off = cb * blk_size;
                        (*interpolateKernel)(&arg);
                    }
                });
            }
        }
    }  // batch end
}

template <typename in_data_t, typename out_data_t>
void MKLDNNInterpolateNode::NearestNeighbor_ref(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW) {
    std::vector<int> index_buffer(OD * OH * OW);
    bool isDDownsample = (fz > 1) ? true : false;
    bool isHDownsample = (fy > 1) ? true : false;
    bool isWDownsample = (fx > 1) ? true : false;
    int nearestVoxel = 0;
    for (int oz = 0; oz < OD; oz++) {
        float iz = coordTransToInput(oz, fz, ID, OD);
        nearestVoxel = nearestRound(iz, isDDownsample);
        int iz_offset = nearestVoxel * IH * IW;
        int oz_offset = oz * OH * OW;
        for (int oy = 0; oy < OH; oy++) {
            float iy = coordTransToInput(oy, fy, IH, OH);
            nearestVoxel = nearestRound(iy, isHDownsample);
            int iy_offset = nearestVoxel * IW + iz_offset;
            int oy_offset = oy * OW + oz_offset;
            for (int ox = 0; ox < OW; ox++) {
                float ix = coordTransToInput(ox, fx, IW, OW);
                nearestVoxel = nearestRound(ix, isWDownsample);
                int ix_index = nearestVoxel + iy_offset;
                index_buffer[oy_offset + ox] = ix_index;
            }
        }
    }

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        const in_data_t *in_ptr = in_ptr_ + IW * IH * ID * C * b + IW * IH * ID * c;
        out_data_t *out_ptr = out_ptr_ + OW * OH * OD * C * b + OW * OH * OD * c;

        for (int i_dst = 0; i_dst < OW * OH * OD; i_dst++) {
            if (fusedWith.empty() && outputPrec == inputPrec) {
                out_ptr[i_dst] = in_ptr[index_buffer[i_dst]];
            } else {
                float dst_value = static_cast<float>(in_ptr[index_buffer[i_dst]]);
                apply_post_ops_scalar(dst_value, c);
                if (outputPrec == Precision::U8) {
                    out_ptr[i_dst] = (dst_value >= 0) ? dst_value : 0;
                } else {
                    out_ptr[i_dst] = dst_value;
                }
            }
        }
    });
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0f, 1 - std::abs(x));
}

template <typename in_data_t, typename out_data_t>
void MKLDNNInterpolateNode::LinearInterpolation(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias) {
    if (IW == OW && IH == OH && ID == OD) {
        size_t spatialDimSize = IW * IH * ID;
        if (fusedWith.empty() && inputPrec == outputPrec) {
            size_t size = B * C * spatialDimSize * srcDataSize;
            simple_copy(out_ptr_, size, in_ptr_, size);
        } else {
            parallel_for2d(B, C, [&](size_t b, size_t c) {
                const in_data_t *in_ptr_nc = in_ptr_ + spatialDimSize * C * b + spatialDimSize * c;
                out_data_t *out_ptr_nc = out_ptr_ + spatialDimSize * C * b + spatialDimSize * c;
                for (size_t i = 0; i < spatialDimSize; i++) {
                    float dst_value = static_cast<float>(in_ptr_nc[i]);
                    apply_post_ops_scalar(dst_value, c);
                    if (outputPrec == Precision::U8) {
                        out_ptr_nc[i] = (dst_value >= 0) ? dst_value : 0;
                    } else {
                        out_ptr_nc[i] = dst_value;
                    }
                }
            });
        }
        return;
    }

    float ax = 1.0f / (antialias ? fx : 1.0f);
    float ay = 1.0f / (antialias ? fy : 1.0f);
    float az = 1.0f / (antialias ? fz : 1.0f);

    int rx = (fx < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ax));
    int ry = (fy < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / ay));
    int rz = (fz < 1.0f) ? 2 : static_cast<int>(ceil(static_cast<float>(kernel_width) / az));

    parallel_for2d(B, C, [&](size_t b, size_t c) {
        const in_data_t *in_ptr_nc = in_ptr_ + IW * IH * ID * C * b + IW * IH * ID * c;
        out_data_t *out_ptr_nc = out_ptr_ + OW * OH * OD * C * b + OW * OH * OD * c;
        for (size_t oz = 0; oz < OD; oz++) {
            out_data_t *out_ptr_ncd = out_ptr_nc + OW * OH * oz;
            float iz = coordTransToInput(oz, fz, ID, OD);
            int iz_r = static_cast<int>(round(iz));
            for (size_t oy = 0; oy < OH; oy++) {
                out_data_t *out_ptr_ncdh = out_ptr_ncd + OW * oy;
                float iy = coordTransToInput(oy, fy, IH, OH);
                int iy_r = static_cast<int>(round(iy));
                for (size_t ox = 0; ox < OW; ox++) {
                    float ix = coordTransToInput(ox, fx, IW, OW);
                    int ix_r = static_cast<int>(round(ix));

                    float sum = 0;
                    float wsum = 0;

                    for (int z = iz_r - rz; z <= iz_r + rz; z++) {
                        for (int y = iy_r - ry; y <= iy_r + ry; y++) {
                            for (int x = ix_r - rx; x <= ix_r + rx; x++) {
                                bool is_continue =  z < 0                     ||
                                                    y < 0                     ||
                                                    x < 0                     ||
                                                    z >= static_cast<int>(ID) ||
                                                    y >= static_cast<int>(IH) ||
                                                    x >= static_cast<int>(IW);
                                if (is_continue)
                                    continue;

                                float dx = ix - x;
                                float dy = iy - y;
                                float dz = iz - z;

                                float w = ax * triangleCoeff(ax * dx) *
                                          ay * triangleCoeff(ay * dy) *
                                          az * triangleCoeff(az * dz);

                                sum += w * static_cast<float>(in_ptr_nc[z * IH * IW + y * IW + x]);
                                wsum += w;
                            }
                        }
                    }
                    if (!wsum) {
                        out_ptr_ncdh[ox] = 0;
                    } else {
                        float dst_value = sum / wsum;
                        if (fusedWith.empty() && inputPrec == outputPrec) {
                            out_ptr_ncdh[ox] = dst_value;
                        } else {
                            apply_post_ops_scalar(dst_value, c);
                            if (outputPrec == Precision::U8) {
                                out_ptr_ncdh[ox] = (dst_value >= 0) ? dst_value : 0;
                            } else {
                                out_ptr_ncdh[ox] = dst_value;
                            }
                        }
                    }
                }
            }
        }
    });
}

inline void MKLDNNInterpolateNode::apply_post_ops_scalar(float &dst_value, int index_c) {
    const auto &p = (*attr.get()).post_ops_;
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    for (int i = 0; i < p.len_; i++) {
        auto &post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            dst_value = eltwise_injectors_ref[eltwise_inj_idx]->compute_scalar(dst_value);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            auto depthwise_weights = post_op.depthwise.weights_data + index_c;
            auto depthwise_bias = post_op.depthwise.biases_data + index_c;
            dst_value = depthwise_injectors_ref[depthwise_inj_idx]->compute_scalar(dst_value, depthwise_weights, depthwise_bias);
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || outputPrec == Precision::FP32 || i != p.len_ - 1;

            auto quant = post_op.quantization;

            float crop_low = quant.crop_low_data->shifts_[quant.crop_low_data->count_ == 1 ? 0 : index_c];
            float crop_high = quant.crop_high_data->shifts_[quant.crop_high_data->count_ == 1 ? 0 : index_c];
            float input_scale = quant.input_scale_data->scales_[quant.input_scale_data->count_ == 1 ? 0 : index_c];
            float input_shift = quant.input_shift_data->shifts_[quant.input_shift_data->count_ == 1 ? 0 : index_c];

            dst_value = nstl::min(crop_high, nstl::max(crop_low, dst_value));
            dst_value = dst_value * input_scale + input_shift;

            if (do_rounding) {
                dst_value = roundf(dst_value);
            }

            if (do_dequantization) {
                float output_scale = quant.output_scale_data->scales_[quant.output_scale_data->count_ == 1 ? 0 : index_c];
                float output_shift = quant.output_shift_data->shifts_[quant.output_shift_data->count_ == 1 ? 0 : index_c];
                dst_value = dst_value * output_scale + output_shift;
            }
        }
    }
}

// scale is inShape/outShape
inline float MKLDNNInterpolateNode::coordTransToInput(int outCoord, float scale, int inShape, int outShape) {
    switch (coordTransMode) {
        case CoordTransMode::half_pixel: {
            return (outCoord + 0.5f) * scale - 0.5f;
            break;
        }
        case CoordTransMode::pytorch_half_pixel: {
            if (outShape > 1)
                return (outCoord + 0.5f) * scale - 0.5f;
            else
                return 0;
            break;
        }
        case CoordTransMode::asymmetric: {
            return outCoord * scale;
            break;
        }
        case CoordTransMode::tf_half_pixel_for_nn: {
            return (outCoord + 0.5f) * scale;
            break;
        }
        case CoordTransMode::align_corners: {
            if (outShape > 1)
                return (inShape - 1) / (outShape - 1) * outCoord;
            else
                return 0;
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' do not support specified coordinate transformation mode";
            break;
        }
    }
}

inline int MKLDNNInterpolateNode::nearestRound(float originCoord, bool isDownsample) {
    switch (nearestMode) {
        case NearestMode::round_prefer_floor: {
            if (originCoord == (static_cast<int>(originCoord) + 0.5))
                return static_cast<int>(originCoord);
            else
                return static_cast<int>(std::round(originCoord));
            break;
        }
        case NearestMode::round_prefer_ceil: {
            return static_cast<int>(std::round(originCoord));
            break;
        }
        case NearestMode::floor: {
            return static_cast<int>(std::floor(originCoord));
            break;
        }
        case NearestMode::ceil: {
            return static_cast<int>(std::ceil(originCoord));
            break;
        }
        case NearestMode::simple: {
            if (isDownsample)
                return static_cast<int>(std::ceil(originCoord));
            else
                return static_cast<int>(originCoord);
        }
        default: {
            THROW_IE_EXCEPTION << "Interpolate layer with name '" << getName() << "' do not support specified nearest round mode";
            break;
        }
    }
}

bool MKLDNNInterpolateNode::created() const {
    return getType() == Interpolate;
}

REG_MKLDNN_PRIM_FOR(MKLDNNInterpolateNode, Interpolate);