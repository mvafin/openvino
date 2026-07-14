// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <openvino/core/node.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/convert_like.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/make_stateful.hpp>

#include "input_model.hpp"
#include "node_context.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/frontend/gguf/tokenizer_metadata.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "pass/lower_set_rows_stateful.hpp"
#include "pass/lower_set_rows_stateless.hpp"
#include "pass/squeeze_matmul.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_convertlike.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {

using namespace ov::op;

namespace {

// Build the Parameter/Result pairs MakeStateful needs, matching the decoder's KV-cache
// parameter/result friendly names against the model's parameters and results.
ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    const std::shared_ptr<ov::Model>& model,
    const std::map<std::string, std::string>& kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    const auto& params = model->get_parameters();
    const auto& results = model->get_results();

    for (const auto& param_res : kv_param_res_names) {
        const auto& param_name = param_res.first;
        const auto& res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<v0::Parameter>& node) {
            return node->get_friendly_name() == param_name;
        });
        OPENVINO_ASSERT(param_it != params.end(),
                        "The tensor name ",
                        param_name,
                        " is not associated with any of Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<v0::Result>& node) {
            return node->get_friendly_name() == res_name;
        });
        OPENVINO_ASSERT(res_it != results.end(),
                        "The tensor name ",
                        res_name,
                        " is not associated with any of Results in the network.");

        pairs.emplace_back(*param_it, *res_it);
    }
    return pairs;
}

// CPU plugin's stateful_sdpa_fusion folds the cache into ScaledDotProductAttentionWithKVCache,
// whose MemoryInputSDPA requires the ReadValue to have an init input (matching how genai /
// optimum build the cache: ReadValue(zeros, variable)). Without it the CPU graph builder hits
// a MemoryInput with zero parent edges and aborts. The init is an empty tensor with the
// cache's element type and partial shape, with the (single) dynamic sequence dimension set to
// 0, so it contributes no past tokens on the first inference.
void add_kvcache_readvalue_init(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ops()) {
        auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(op);
        if (!rv || rv->get_input_size() != 0) {
            continue;
        }
        const auto variable = rv->get_variable();
        const auto& info = variable->get_info();
        const auto& pshape = info.data_shape;
        if (pshape.rank().is_dynamic()) {
            continue;
        }
        std::vector<int64_t> init_dims;
        init_dims.reserve(pshape.size());
        for (const auto& d : pshape) {
            init_dims.push_back(d.is_static() ? d.get_length() : 0);
        }
        ov::Shape init_shape(init_dims.begin(), init_dims.end());
        auto init = std::make_shared<ov::op::v0::Constant>(info.data_type, init_shape, std::vector<float>{});

        auto rv_with_init = std::make_shared<ov::op::v6::ReadValue>(init, variable);
        rv_with_init->set_friendly_name(rv->get_friendly_name());
        ov::copy_runtime_info(rv, rv_with_init);
        ov::replace_node(rv, rv_with_init);
    }
}

void add_sliced_mask(TensorMap& tensor_map, GgufDecoder& gguf_model_decoder) {
    // Slice the full attention mask down to the current token window for the attention ops.
    // Three cases: static (fixed shape -> pass through), stateful (KV cache grows -> slice by
    // token_len_per_seq and the incremented position), and plain stateless (slice by
    // token_len_per_seq along axis 2).
    auto create_sliced_mask = [&](const std::string& mask_name, const std::string& sliced_name, bool is_static) {
        if ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end())) {
            auto token_len_per_seq = tensor_map.at("token_len_per_seq").get_node_shared_ptr();
            auto mask = tensor_map.at(mask_name).get_node_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced;
            if (is_static) {
                mask_sliced = mask;
            } else if (gguf_model_decoder.is_stateful()) {
                auto zero_2d = ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 0});
                auto one_2d = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 1});
                auto three_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
                auto neg_one_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
                auto axes = ov::op::v0::Constant::create(ov::element::i64, {2}, {-2, -1});
                auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
                auto gather_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one_1d, three_1d);
                auto reshaped_inp_pos =
                    std::make_shared<ov::op::v1::Reshape>(gather_inp_pos,
                                                          ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                          false);
                auto inp_pos_incremented = std::make_shared<ov::op::v1::Add>(
                    reshaped_inp_pos,
                    ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}));
                auto stop = std::make_shared<ov::op::v0::Concat>(
                    ov::OutputVector{token_len_per_seq,
                                     std::make_shared<v1::ConvertLike>(inp_pos_incremented, token_len_per_seq)},
                    0);
                mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero_2d, stop, one_2d, axes);
                mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
                mask_sliced->set_friendly_name(sliced_name);
            } else {
                auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
                mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero, token_len_per_seq, one, two);
                mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
                mask_sliced->set_friendly_name(sliced_name);
            }
            tensor_map.insert({sliced_name, mask_sliced->output(0)});
        }
    };

    create_sliced_mask("self_kq_mask", "KQ_mask_sliced", gguf_model_decoder.is_static());
    create_sliced_mask("self_kq_mask_swa", "KQ_mask_swa_sliced", gguf_model_decoder.is_static());
}

void add_rope_sin_cos(TensorMap& tensor_map, GgufDecoder& gguf_model_decoder) {
    const auto rope_config = gguf_model_decoder.get_attribute("rope_config").as<RopeConfig>();
    // n_dims == 0 means the model uses no RoPE; per_op means each ROPE op builds its own sin/cos
    // (e.g. gemma4 where SWA and global layers differ), so skip the shared table entirely.
    if (tensor_map.find("inp_pos") == tensor_map.end() || rope_config.n_dims == 0 || rope_config.per_op) {
        return;
    }
    auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    if (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_node_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_config, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_node_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_node_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
void preprocess(TensorMap& tensor_map, GgufDecoder& gguf_model_decoder) {
    add_sliced_mask(tensor_map, gguf_model_decoder);
    add_rope_sin_cos(tensor_map, gguf_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(const frontend::InputModel::Ptr& input_model,
                                   const std::unordered_map<std::string, CreatorFunction>& translator_map,
                                   bool naive,
                                   const std::vector<DecoderTransformationExtension::Ptr>& transformation_extensions)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_ov_model(nullptr),
      m_naive(naive),
      m_transformation_extensions(transformation_extensions) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputModel::Ptr& input_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    const auto& gguf_model = std::dynamic_pointer_cast<InputModel>(input_model);
    std::shared_ptr<GgufDecoder> gguf_model_decoder = gguf_model->get_model_decoder();

    // Declared input Parameters (model_inputs + model_extra_inputs) whose consumer is created only
    // by a later normalization pass -- notably beam_idx, whose Gather is emitted by
    // LowerSetRowsStateful in apply_transformations, AFTER the unused-Parameter pruning below.
    // Track them so pruning never drops them for lack of a consumer at translate time. A pass that
    // ends up not consuming one leaves it as a dangling input, which later constant folding /
    // MakeStateful removes -- matching the stateless-path behaviour where translate_set_rows
    // consumed beam_idx during the walk.
    std::set<ov::Node*> deferred_use_params;

    for (const auto& it : gguf_model_decoder->get_model_inputs()) {
        auto p = std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second);
        params.push_back(p);
        if (p && it.first == "beam_idx") {
            deferred_use_params.insert(p.get());
        }
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto& it : gguf_model_decoder->get_model_extra_inputs()) {
        if (auto p = std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second)) {
            params.push_back(p);
            deferred_use_params.insert(p.get());
        }
        (*tensor_map)[it.first] = it.second;
    }

    // The native .gguf builder path dequantizes weights up front and exposes them via
    // get_model_weights(); seed them into the tensor map before the walk. The llama.cpp cgraph
    // path returns an empty map and instead surfaces each weight as a "GGML_OP_NONE" leaf carrying
    // a "data" attribute, translated by translate_weight during visit_subgraph (below). The two
    // paths are mutually exclusive, so seeding here is a no-op for the cgraph decoder.
    for (const auto& it : gguf_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto node_visitor = [&](std::shared_ptr<GgufDecoder> decoder) {
        auto operation_type = decoder->get_op_type();
        if (operation_type == "GGML_OP_NONE") {
            // A GGML_OP_NONE leaf is a weight if the decoder marks it as one: either the native
            // builder's pre-extracted payload (bool "gguf_weight") or the cgraph decoder's raw
            // bytes ("data"). Otherwise it is a model-input leaf (already seeded as a Parameter
            // above) and there is nothing to translate.
            const bool is_builder_weight = decoder->get_attribute("gguf_weight").is<bool>() &&
                                           decoder->get_attribute("gguf_weight").as<bool>();
            const bool is_cgraph_weight = decoder->get_attribute("data").is<ov::Tensor>();
            if (!is_builder_weight && !is_cgraph_weight) {
                return;
            }
        }

        ov::OutputVector converted_outputs;
        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(),
                                      "Translation for operation type ",
                                      operation_type,
                                      " is not implemented.");
        NodeContext node_context(decoder, tensor_map);
        converted_outputs = it->second(node_context);

        const auto& node_output_names = decoder->get_output_names();
        FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(),
                                      "Number of ",
                                      operation_type,
                                      " outputs greater than number of converted outputs, which are ",
                                      node_output_names.size(),
                                      " and ",
                                      converted_outputs.size(),
                                      " respectively.");

        // Record the decoder-declared output shape (node-scoped) so apply_transformations can
        // restore the leading dim on stateful-model outputs (see m_output_declared_shapes).
        m_output_declared_shapes[node_output_names.empty() ? std::string{} : node_output_names[0]] =
            decoder->get_output_shape();

        for (size_t i = 0; i < node_output_names.size(); ++i) {
            auto output_name = node_output_names[i];
            if (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
    };

    if (!m_naive) {
        preprocess(*tensor_map, *gguf_model_decoder);
    }
    gguf_model_decoder->visit_subgraph(node_visitor);

    for (const auto& name : gguf_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ",
                                name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    ov::ParameterVector used_params;
    for (const auto& param : params) {
        // Keep a Parameter if it currently feeds something, OR if its consumer is created by a
        // later normalization pass (e.g. beam_idx -> Gather in LowerSetRowsStateful).
        if (!param->output(0).get_target_inputs().empty() || deferred_use_params.count(param.get())) {
            used_params.push_back(param);
        }
    }
    resulting_model = std::make_shared<Model>(results, used_params);

    // apply_transformations may rebuild the model (PrePostProcessor for stateful output reshaping),
    // so use its return value.
    resulting_model = apply_transformations(resulting_model);

    // Attach GGUF tokenizer metadata (the file's tokenizer.* keys) to the model's rt_info as a
    // non-serializable attribute, so a downstream consumer (OpenVINO GenAI) can build the
    // tokenizer without reopening the .gguf. Empty when the decoder carries no tokenizer config.
    const auto& tok_cfg = gguf_model_decoder->get_tokenizer_config();
    if (!tok_cfg.empty()) {
        resulting_model->get_rt_info()[gguf_tokenizer_metadata_key()] =
            std::make_shared<GGUFTokenizerMetadata>(tok_cfg);
    }

    // Set WeightlessCacheAttribute on large constants to avoid unnecessary memory copies
    // in the NPUW plugin. Without this attribute, NPUW's LazyTensor constructor
    // (lazy_tensor.cpp, op::Const::Const) will memcpy every constant "in case export
    // occurs", doubling memory usage per compile_model call.
    //
    // The bin_offset field serves as a unique key (not a real file offset) — this is
    // the same convention the GPU plugin uses for non-IR models (see
    // Plugin::set_weightless_cache_attributes in intel_gpu/src/plugin/plugin.cpp).
    // Each constant must have a distinct bin_offset, otherwise GPU's weightless cache
    // import will map multiple constants to the same data.
    //
    // Small constants (< 16 elements) are excluded since they may be introduced by
    // optimization patterns and the overhead is negligible.
    size_t offset = 0;
    for (auto& node : resulting_model->get_ordered_ops()) {
        if (auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
            cnst && cnst->get_element_type().size() > 0 &&
            cnst->get_byte_size() / cnst->get_element_type().size() >= 16) {
            auto& rt_info = cnst->get_rt_info();
            if (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
            }
        }
    }
    return resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    auto gguf_model_decoder = std::dynamic_pointer_cast<InputModel>(m_input_model)->get_model_decoder();
    const bool is_stateful = gguf_model_decoder->is_stateful();
    const bool is_static = gguf_model_decoder->is_static();

    ov::pass::Manager manager;
    manager.set_per_pass_validation(true);
    manager.register_pass<ov::pass::MarkCompressedFloatConstants>();

    // Caller-registered transformation extensions run first. A SetRows-lowering extension (e.g. a
    // backend's own stateful lowering) may consume the KV-cache SetRows ops here; the built-in
    // lowering below then only fires on the ops left untouched.
    for (const auto& ext : m_transformation_extensions) {
        ext->register_pass(manager);
    }

    // Lower the SetRows placeholders. For a stateful model (the native .gguf path), the KV-cache
    // SetRows become an appending Concat (+ beam_idx Gather) that MakeStateful then turns into a
    // ReadValue/Assign pair -- this is the model OpenVINO GenAI consumes. Otherwise every SetRows
    // becomes a ScatterUpdate into the passed-in tensor (the llama.cpp-faithful stateless form).
    if (is_stateful) {
        manager.register_pass<pass::LowerSetRowsStateful>();
    } else {
        manager.register_pass<pass::LowerSetRowsStateless>();
    }

    // MakeStateful must run after the stateful SetRows lowering (so the cache Concat -> Result and
    // Parameter -> Concat edges exist) to pair the KV-cache Parameters/Results into ReadValue/Assign.
    if (is_stateful) {
        const auto kv_param_res_names = gguf_model_decoder->get_kv_param_res_names();
        manager.register_pass<ov::pass::MakeStateful>(get_kv_param_res_pairs(model, kv_param_res_names));
    }

    // Static (fixed token length, NPU) models: NPUW DynamicQuantization expects 3D activations.
    if (is_static) {
        manager.register_pass<pass::SqueezeMatmul>();
    }

    manager.register_pass<ov::pass::ConvertConvertLike>();
    manager.run_passes(model);

    if (is_stateful) {
        // MakeStateful produced init-less ReadValues; the CPU fused SDPA-with-KV-cache path needs
        // an init subgraph on each (see helper).
        add_kvcache_readvalue_init(model);

        // The decoder declares rank-4 outputs (e.g. logits [1,1,seq,vocab]) but the converted
        // graph collapses the leading 1; Unsqueeze such outputs back so the model matches the
        // decoder's declared IO (what AdaptToGenAI consumes).
        ov::preprocess::PrePostProcessor ppp(model);
        bool needs_ppp = false;
        for (size_t i = 0; i < model->get_output_size(); i++) {
            auto output_friendly_name = model->output(i).get_node_shared_ptr()->get_friendly_name();
            auto it = m_output_declared_shapes.find(output_friendly_name);
            if (it == m_output_declared_shapes.end()) {
                continue;
            }
            auto model_output_shape = model->output(i).get_partial_shape();
            const auto& decoder_output_shape = it->second;
            if (model_output_shape.rank().is_static() && decoder_output_shape.rank().is_static() &&
                model_output_shape.rank().get_length() + 1 == decoder_output_shape.rank().get_length() &&
                decoder_output_shape[0].is_static() && decoder_output_shape[0].get_length() == 1) {
                ppp.output(i).postprocess().custom([](const ov::Output<ov::Node>& node) {
                    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                    return std::make_shared<ov::op::v0::Unsqueeze>(node, axes);
                });
                needs_ppp = true;
            }
        }
        if (needs_ppp) {
            model = ppp.build();
        }
    }
    return model;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
