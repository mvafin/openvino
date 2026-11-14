// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"
#include "transformations/common_optimizations/fuse_ssm.hpp"
#include "transformations/common_optimizations/sdpa_fusion.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/paged_attention/eliminate_conv_padding_mask_gating.hpp"
#include "transformations/paged_attention/gemma4_mtp_state_management_pattern.hpp"
#include "transformations/paged_attention/paged_causal_conv1d_fusion.hpp"
#include "transformations/paged_attention/paged_gated_delta_net_fusion.hpp"
#include "transformations/paged_attention/paged_selective_ssm_fusion.hpp"
#include "transformations/paged_attention/position_ids_replacer.hpp"
#include "transformations/paged_attention/prev_sequence_length_pattern.hpp"
#include "transformations/paged_attention/state_management_pattern.hpp"
#include "transformations/paged_attention/total_sequence_length_pattern.hpp"
#include "transformations/utils/print_model.hpp"
#include "transformations/utils/utils.hpp"

#include <unordered_set>

using namespace ov::op;
using ov::pass::paged_attention::PaParams;
using ov::pass::paged_attention::PaResults;

namespace {

std::shared_ptr<v0::Parameter> get_parameter(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& param : model->inputs()) {
        const auto& names = param.get_names();
        if (names.count(name)) {
            if (auto casted_param = ov::as_type_ptr<v0::Parameter>(param.get_node_shared_ptr())) {
                return casted_param;
            } else {
                OPENVINO_THROW("The model is in the inconsistent state. Found input '",
                               name,
                               "', but couldn't cast it to v0::Parameter.");
            }
        }
    }

    return nullptr;
}

void collect_ancestors(const std::shared_ptr<ov::Node>& node, std::unordered_set<ov::Node*>& ancestors) {
    if (!ancestors.insert(node.get()).second) {
        return;
    }
    for (const auto& input : node->input_values()) {
        collect_ancestors(input.get_node_shared_ptr(), ancestors);
    }
}

std::unordered_set<std::string> get_read_value_ids(const ov::Output<ov::Node>& output) {
    std::unordered_set<ov::Node*> ancestors;
    collect_ancestors(output.get_node_shared_ptr(), ancestors);
    std::unordered_set<std::string> ids;
    for (auto* node : ancestors) {
        if (const auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(node->shared_from_this())) {
            ids.insert(read_value->get_variable_id());
        }
    }
    return ids;
}

std::vector<std::shared_ptr<v13::ScaledDotProductAttention>> get_shared_kv_layers(
    const std::shared_ptr<ov::Model>& model) {
    std::unordered_set<std::string> seen_cache_pairs;
    std::vector<std::shared_ptr<v13::ScaledDotProductAttention>> shared_layers;
    for (const auto& node : model->get_ordered_ops()) {
        const auto sdpa = ov::as_type_ptr<v13::ScaledDotProductAttention>(node);
        if (!sdpa) {
            continue;
        }
        const auto k_ids = get_read_value_ids(sdpa->input_value(1));
        const auto v_ids = get_read_value_ids(sdpa->input_value(2));
        for (const auto& k_id : k_ids) {
            for (const auto& v_id : v_ids) {
                const auto pair = k_id + "\n" + v_id;
                if (!seen_cache_pairs.insert(pair).second) {
                    shared_layers.push_back(sdpa);
                }
            }
        }
    }
    return shared_layers;
}

ov::Node* get_row_selection_source(const std::shared_ptr<ov::Node>& node) {
    auto current = node;
    while (ov::is_type<v1::Reshape>(current) || ov::is_type<v0::Squeeze>(current) ||
           ov::is_type<v0::Unsqueeze>(current) || ov::is_type<v0::Convert>(current)) {
        current = current->get_input_node_shared_ptr(0);
    }
    return current.get();
}

bool is_row_selection(const std::shared_ptr<ov::Node>& node, ov::Node*& source) {
    const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(node);
    const auto gather = unsqueeze ? ov::as_type_ptr<v8::Gather>(unsqueeze->get_input_node_shared_ptr(0)) : nullptr;
    const auto indices = gather ? ov::as_type_ptr<v1::Reshape>(gather->get_input_node_shared_ptr(1)) : nullptr;
    if (!indices) {
        return false;
    }
    source = get_row_selection_source(indices->get_input_node_shared_ptr(0));
    return true;
}

std::unordered_set<ov::Node*> find_shared_kv_row_selections(
    const std::vector<std::shared_ptr<v13::ScaledDotProductAttention>>& shared_layers,
    const std::shared_ptr<ov::Model>& model) {
    std::unordered_set<ov::Node*> selected_sources;
    for (const auto& sdpa : shared_layers) {
        auto q_source = std::unordered_set<ov::Node*>{};
        collect_ancestors(sdpa->input_value(0).get_node_shared_ptr(), q_source);
        std::unordered_set<ov::Node*> layer_sources;
        for (auto* ancestor : q_source) {
            const auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(ancestor->shared_from_this());
            const auto gather =
                unsqueeze ? ov::as_type_ptr<v8::Gather>(unsqueeze->get_input_node_shared_ptr(0)) : nullptr;
            if (!gather) {
                continue;
            }
            std::unordered_set<ov::Node*> gather_data_ancestors;
            collect_ancestors(gather->input_value(0).get_node_shared_ptr(), gather_data_ancestors);
            const bool depends_on_attention = std::any_of(gather_data_ancestors.begin(),
                                                          gather_data_ancestors.end(),
                                                          [](auto* data_node) {
                                                              return ov::is_type<v13::ScaledDotProductAttention>(
                                                                  data_node->shared_from_this());
                                                          });
            if (!depends_on_attention) {
                continue;
            }
            ov::Node* source = nullptr;
            if (is_row_selection(ancestor->shared_from_this(), source)) {
                layer_sources.insert(source);
            }
        }
        if (layer_sources.size() > 1) {
            return {};
        }
        selected_sources.insert(layer_sources.begin(), layer_sources.end());
    }

    std::unordered_set<ov::Node*> selected_nodes;
    if (selected_sources.size() != 1) {
        return selected_nodes;
    }
    for (const auto& node : model->get_ordered_ops()) {
        ov::Node* source = nullptr;
        if (is_row_selection(node, source) && selected_sources.count(source)) {
            selected_nodes.insert(node.get());
        }
    }
    return selected_nodes;
}

void apply_shared_kv_row_selection(const std::unordered_set<ov::Node*>& row_selections,
                                   const std::shared_ptr<v0::Parameter>& shared_kv_tokens_indices) {
    const auto ids_shape = v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, -1, 1});
    const auto selected_ids = std::make_shared<v1::Reshape>(shared_kv_tokens_indices, ids_shape, false);
    bool selected_nodes = false;
    for (auto* node : row_selections) {
        const auto gather = ov::as_type_ptr<v8::Gather>(node->get_input_node_shared_ptr(0));
        const auto indices_reshape =
            gather ? ov::as_type_ptr<v1::Reshape>(gather->get_input_node_shared_ptr(1)) : nullptr;
        OPENVINO_ASSERT(indices_reshape, "Shared-KV row-selection graph changed during conversion");
        if (!selected_nodes) {
            copy_runtime_info(indices_reshape, selected_ids);
        }
        indices_reshape->input(0).replace_source_output(selected_ids);
        selected_nodes = true;
    }
    OPENVINO_ASSERT(selected_nodes, "Shared-KV pruning has no detected row-selection nodes");
}

}  // namespace

ov::pass::SDPAToPagedAttention::SDPAToPagedAttention(bool use_per_layer_block_indices_inputs,
                                                     bool use_score_outputs,
                                                     bool allow_score_aggregation,
                                                     bool allow_cache_rotation,
                                                     bool allow_xattention,
                                                     bool allow_adaptive_rkv,
                                                     bool allow_qq_bias,
                                                     bool is_draft_model)
    : m_options{use_per_layer_block_indices_inputs,
                use_score_outputs,
                allow_score_aggregation,
                allow_cache_rotation,
                allow_xattention,
                allow_adaptive_rkv,
                allow_qq_bias,
                is_draft_model} {}

bool ov::pass::SDPAToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToPagedAttention);

    // Gemma4 MTP draft model reads the target model's KV cache through its inputs and has no state of its own.
    OPENVINO_ASSERT(m_options.is_draft_model || !model->get_variables().empty(),
                    "Model is supposed to be stateful, cannot perform "
                    "the SDPAToPagedAttention transformation. "
                    "For proper conversion run: optimum-cli export openvino --task text-generation-with-past instead "
                    "of --task text-generation");

    OPENVINO_ASSERT(ov::op::util::has_op_with_type<ov::op::v13::ScaledDotProductAttention>(model),
                    "No ScaledDotProductAttention operation observed in the graph, cannot perform "
                    "the SDPAToPagedAttention transformation.");

    m_params = PaParams{model->get_parameters()};
    m_results = PaResults{model->get_results()};
    const auto shared_kv_layers =
        m_params.exists("token_type_ids") ? std::vector<std::shared_ptr<v13::ScaledDotProductAttention>>{}
                                          : get_shared_kv_layers(model);
    const auto shared_kv_row_selections = find_shared_kv_row_selections(shared_kv_layers, model);
    if (!shared_kv_row_selections.empty()) {
        auto sampled_tokens_indices = m_params.add("sampled_tokens_indices", element::i64, PartialShape{-1});
        auto shared_kv_tokens_indices = m_params.add("shared_kv_tokens_indices", element::i64, PartialShape{-1});
        auto shared_kv_past_lens = m_params.add("shared_kv_past_lens", element::i32, PartialShape{-1});
        auto shared_kv_subsequence_begins =
            m_params.add("shared_kv_subsequence_begins", element::i32, PartialShape{-1});
        model->add_parameters(
            {sampled_tokens_indices, shared_kv_tokens_indices, shared_kv_past_lens, shared_kv_subsequence_begins});
        apply_shared_kv_row_selection(shared_kv_row_selections, shared_kv_tokens_indices);
    }
    auto max_context_len = m_params.add("max_context_len", element::i32, PartialShape{});
    m_params.add("past_lens", element::i32, PartialShape{-1});
    m_params.add("subsequence_begins", element::i32, PartialShape{-1});
    m_params.add("block_indices_begins", element::i32, PartialShape{-1});
    if (!m_options.use_per_layer_block_indices_inputs) {
        m_params.add("block_indices", element::i32, PartialShape{-1});
    }

    std::shared_ptr<v0::Parameter> input_ids_node;
    for (const auto& name : {"input_ids", "inputs_embeds"}) {
        if ((input_ids_node = get_parameter(model, name))) {
            break;
        }
    }

    OPENVINO_ASSERT(input_ids_node, "The model doesn't contain input_ids or input_embeds input. Aborting.");

    if (input_ids_node->get_friendly_name() == "input_ids") {
        input_ids_node->set_partial_shape(PartialShape{-1});
    } else if (input_ids_node->get_friendly_name() == "inputs_embeds") {
        input_ids_node->set_partial_shape(PartialShape{-1, -1});
    }

    auto input_ids_target_inputs = input_ids_node->get_output_target_inputs(0);
    auto processed_input_ids =
        std::make_shared<v0::Unsqueeze>(input_ids_node, v0::Constant::create(element::i32, Shape{}, {1}));
    for (const auto& target : input_ids_target_inputs) {
        target.replace_source_output(processed_input_ids);
    }

    std::unordered_set<std::string> var_ids_to_remove;
    std::unordered_set<std::string> params_to_remove{"beam_idx", "attention_mask"};

    // Get-or-create the flattened position_ids parameter and restore its rank at each existing consumer with a
    // single shared Unsqueeze(-1). The PositionIDsReplacer* passes below consume the raw parameter and add their
    // own rank-restoring nodes; EliminateDropBatch handles the batch-drop select branch separately.
    auto position_ids = m_params.get("position_ids");
    if (!position_ids) {
        position_ids = m_params.add("position_ids", element::i64, PartialShape{-1});
    } else {
        const auto& position_ids_shape = position_ids->get_partial_shape();
        if (position_ids_shape.rank().is_static() && position_ids_shape.rank().get_length() == 2) {
            position_ids->set_partial_shape(PartialShape{-1});
        } else if (position_ids_shape.rank().is_static() && position_ids_shape.rank().get_length() == 3) {
            // Qwen2.5 VL M-RoPE: [3, total_token_num] -> Unsqueeze(axis=-1) -> [3, total_token_num, 1]
            position_ids->set_partial_shape(PartialShape{position_ids_shape[0], -1});
        } else {
            OPENVINO_THROW("Unexpected shape for position_ids input: expected rank 2 or 3, observed ",
                           position_ids_shape.rank().is_static() ? position_ids_shape.rank().get_length() : -1);
        }
    }
    const auto position_ids_target_inputs = position_ids->output(0).get_target_inputs();
    auto unsqueezed_position_ids =
        std::make_shared<v0::Unsqueeze>(position_ids, v0::Constant::create(element::i32, Shape{}, {-1}));
    for (const auto& target : position_ids_target_inputs) {
        target.replace_source_output(unsqueezed_position_ids);
    }

    // Refresh shapes after the input_ids/position_ids changes above; per-pass validation is disabled below.
    model->validate_nodes_and_infer_types();

    ov::pass::Manager manager("SDPA to PA");
    manager.set_per_pass_validation(false);
    manager.register_pass<EliminateDropBatch>();
    manager.register_pass<RoPEUnsqueezeAxisReplacer>();
    manager.register_pass<ov::pass::GatedDeltaNetFusion>();  // This pass is required to ensure that all GatedDeltaNet
                                                             // nodes are in the expected form before running
                                                             // PagedGatedDeltaNetFusion.
    auto ssm_fusion = manager.register_pass<SelectiveSSMFusion>();
    if (m_options.is_draft_model) {
        manager.register_pass<Gemma4MTPStateManagementPattern>(m_params, params_to_remove, m_options);
    } else {
        manager.register_pass<StateManagementPattern>(m_params, m_results, m_options, var_ids_to_remove);
    }
    manager.register_pass<EliminateConvPaddingMaskGating>();
    auto paged_ssm_fusion = manager.register_pass<PagedSelectiveSSMFusion>(m_params, var_ids_to_remove);
    manager.register_pass<PagedCausalConv1DFusion>(m_params, var_ids_to_remove);
    manager.register_pass<PagedGatedDeltaNetFusion>(m_params, var_ids_to_remove);
    manager.register_pass<PrevSequenceLengthPattern>(processed_input_ids, max_context_len, position_ids);
    manager.register_pass<TotalSequenceLengthPattern>(max_context_len);
    manager.register_pass<TotalSequenceLengthPatternQwen>(max_context_len);
    manager.register_pass<TotalSequenceLengthPatternCodeGen2>(max_context_len);
    manager.register_pass<PositionIDsReplacer>(unsqueezed_position_ids);
    manager.register_pass<PositionIDsReplacerQwen>(unsqueezed_position_ids);
    manager.register_pass<PositionIDsReplacerCodeGen2>(position_ids);
    manager.register_pass<PositionIDsReplacerLFM2>(position_ids);
    manager.register_pass<PositionIDsReplacerGranite>(unsqueezed_position_ids);
    manager.run_passes(model);

    OPENVINO_ASSERT(ssm_fusion->get_fused_count() == paged_ssm_fusion->get_fused_count(),
                    "SelectiveSSMFusion fused ",
                    ssm_fusion->get_fused_count(),
                    " SelectiveSSM node(s), but PagedSelectiveSSMFusion converted ",
                    paged_ssm_fusion->get_fused_count(),
                    ". Stateful SSM nodes cannot be left in the graph.");

    {
        // Remove all Assigns aggressively, the path from the kv-cache concat to Assign can be complicated,
        // but there is no reason to track it and reject part of the Assigns, because the model will remain
        // in incorrect form anyway.
        auto sinks = model->get_sinks();

        for (auto& sink : sinks) {
            if (auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink)) {
                if (var_ids_to_remove.count(assign->get_variable_id())) {
                    model->remove_sink(sink);
                }
            }
        }
    }

    for (const auto& param_name : params_to_remove) {
        if (auto param = get_parameter(model, param_name)) {
            model->remove_parameter(param);

            if (param->output(0).get_target_inputs().size() == 0) {
                std::stringstream consumers;
                consumers << std::endl;
                for (auto& input : param->output(0).get_target_inputs()) {
                    consumers << *input.get_node() << std::endl;
                }
                OPENVINO_ASSERT(param->output(0).get_target_inputs().size() == 0,
                                "PagedAttention transformation failed: couldn't remove ",
                                param->output(0).get_target_inputs().size(),
                                " inputs of ",
                                param_name,
                                " input: ",
                                consumers.str());
            }
        }
    }

    model->add_results(m_results.items());
    const auto model_parameters = model->get_parameters();
    ov::ParameterVector parameters_to_add;
    for (const auto& parameter : m_params.items()) {
        if (std::find(model_parameters.begin(), model_parameters.end(), parameter) == model_parameters.end()) {
            parameters_to_add.push_back(parameter);
        }
    }
    model->add_parameters(parameters_to_add);
    model->validate_nodes_and_infer_types();

    return true;
}
