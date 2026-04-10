// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sequence_loop_decomposer.hpp"

#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/util/framework_node.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pass {
namespace {

bool is_framework_node_type(const std::shared_ptr<ov::Node>& node, const std::string& type) {
    auto fw = ov::as_type_ptr<ov::op::util::FrameworkNode>(node);
    if (!fw)
        return false;
    return fw->get_attrs().get_type_name() == type;
}

int64_t get_position_attr(const std::shared_ptr<ov::Node>& node) {
    auto fw = ov::as_type_ptr<ov::op::util::FrameworkNode>(node);
    if (fw) {
        auto it = fw->get_attrs().find("position");
        if (it != fw->get_attrs().end())
            return std::stoll(it->second);
    }
    if (node->get_input_size() > 1) {
        auto c = ov::as_type_ptr<v0::Constant>(node->input_value(1).get_node_shared_ptr());
        if (c)
            return c->cast_vector<int64_t>()[0];
    }
    return -1;
}

int64_t get_sequence_size(const ov::Output<ov::Node>& output) {
    if (auto sm = ov::as_type_ptr<ov::frontend::SequenceMark>(output.get_node_shared_ptr()))
        return static_cast<int64_t>(sm->get_input_size());

    // Trace through If nodes: check both branches for SequenceMark
    if (auto ifn = ov::as_type_ptr<v8::If>(output.get_node_shared_ptr())) {
        for (int bi : {0, 1}) {
            for (const auto& d : ifn->get_output_descriptions(bi)) {
                if (d->m_output_index == output.get_index()) {
                    auto body = ifn->get_function(bi);
                    auto bres = body->get_results();
                    if (d->m_body_value_index < bres.size()) {
                        int64_t sz = get_sequence_size(bres[d->m_body_value_index]->input_value(0));
                        if (sz > 0)
                            return sz;
                    }
                }
            }
        }
    }

    return -1;
}

// Find the maximum SequenceMark element count in the model, including nested bodies.
int64_t find_max_sequence_size(const std::shared_ptr<ov::Model>& model) {
    int64_t max_n = 0;
    for (auto& op : model->get_ordered_ops()) {
        if (auto sm = ov::as_type_ptr<ov::frontend::SequenceMark>(op)) {
            max_n = std::max(max_n, static_cast<int64_t>(sm->get_input_size()));
        } else if (auto loop = ov::as_type_ptr<v5::Loop>(op)) {
            max_n = std::max(max_n, find_max_sequence_size(loop->get_function()));
        } else if (auto ifn = ov::as_type_ptr<v8::If>(op)) {
            max_n = std::max(max_n, find_max_sequence_size(ifn->get_then_body()));
            max_n = std::max(max_n, find_max_sequence_size(ifn->get_else_body()));
        }
    }
    return max_n;
}

void replace_sequence_consumers(const ov::Output<ov::Node>& seq_output,
                                const ov::OutputVector& elements,
                                int64_t n) {
    // Copy targets since we modify edges
    auto consumers = seq_output.get_target_inputs();
    for (const auto& consumer : consumers) {
        auto cnode = consumer.get_node()->shared_from_this();

        if (is_framework_node_type(cnode, "SequenceAt")) {
            int64_t pos = get_position_attr(cnode);
            if (pos < 0)
                pos += n;
            if (pos >= 0 && pos < n)
                cnode->output(0).replace(elements[pos]);
        } else if (is_framework_node_type(cnode, "SequenceLength")) {
            auto c = v0::Constant::create(ov::element::i64, ov::Shape{}, {n});
            cnode->output(0).replace(c);
        } else if (is_framework_node_type(cnode, "SequenceErase")) {
            int64_t pos = -1;
            if (cnode->get_input_size() > 1) {
                auto pc = ov::as_type_ptr<v0::Constant>(cnode->input_value(1).get_node_shared_ptr());
                if (pc)
                    pos = pc->cast_vector<int64_t>()[0];
            }
            if (pos < 0)
                pos += n;
            if (pos >= 0 && pos < n) {
                ov::OutputVector erased;
                for (int64_t i = 0; i < n; ++i)
                    if (i != pos)
                        erased.push_back(elements[i]);
                auto new_sm = std::make_shared<ov::frontend::SequenceMark>(erased);
                cnode->output(0).replace(new_sm);
            }
        }
    }
}

bool decompose_sequences_in_model(const std::shared_ptr<ov::Model>& model);

// Split an If output that carries a SequenceMark into N individual outputs.
// Uses v8::If::set_input()/set_output() API to build a clean new If.
ov::OutputVector split_if_sequence_output(const std::shared_ptr<v8::If>& if_node,
                                          size_t output_idx,
                                          int64_t n_elements) {
    size_t then_ridx = std::numeric_limits<size_t>::max();
    size_t else_ridx = std::numeric_limits<size_t>::max();
    for (const auto& d : if_node->get_output_descriptions(v8::If::THEN_BODY_INDEX))
        if (d->m_output_index == output_idx)
            then_ridx = d->m_body_value_index;
    for (const auto& d : if_node->get_output_descriptions(v8::If::ELSE_BODY_INDEX))
        if (d->m_output_index == output_idx)
            else_ridx = d->m_body_value_index;

    if (then_ridx == std::numeric_limits<size_t>::max() ||
        else_ridx == std::numeric_limits<size_t>::max())
        return {};

    auto then_body = if_node->get_then_body();
    auto else_body = if_node->get_else_body();

    auto then_result_src = then_body->get_results()[then_ridx]->input_value(0).get_node_shared_ptr();
    auto else_result_src = else_body->get_results()[else_ridx]->input_value(0).get_node_shared_ptr();

    auto then_sm = ov::as_type_ptr<ov::frontend::SequenceMark>(then_result_src);
    auto else_sm = ov::as_type_ptr<ov::frontend::SequenceMark>(else_result_src);

    bool then_full = then_sm && static_cast<int64_t>(then_sm->get_input_size()) == n_elements;
    bool else_full = else_sm && static_cast<int64_t>(else_sm->get_input_size()) == n_elements;
    bool then_empty = then_sm && then_sm->get_input_size() == 0;
    bool else_empty = else_sm && else_sm->get_input_size() == 0;

    // Treat a non-SequenceMark result (e.g. passthrough Identity(Parameter)) as "dummy":
    // the branch doesn't modify the sequence, so we fill it with zeros.
    // This is safe because the If condition ensures the other branch is used when
    // the sequence data is actually needed.
    bool then_passthrough = !then_sm;
    bool else_passthrough = !else_sm;

    bool then_ok = then_full || then_empty || then_passthrough;
    bool else_ok = else_full || else_empty || else_passthrough;

    if (!then_ok || !else_ok)
        return {};
    if (!then_full && !else_full)
        return {};

    auto ref_sm = then_full ? then_sm : else_sm;

    // Expand: replace SequenceMark result with N individual results
    auto expand_body = [&](std::shared_ptr<ov::Model>& body,
                           size_t ridx,
                           const std::shared_ptr<ov::frontend::SequenceMark>& sm,
                           bool full) {
        auto results = body->get_results();
        ov::ResultVector new_results;
        for (size_t i = 0; i < results.size(); ++i) {
            if (i == ridx) {
                for (int64_t j = 0; j < n_elements; ++j) {
                    if (full) {
                        new_results.push_back(std::make_shared<v0::Result>(sm->input_value(j)));
                    } else {
                        auto ref_out = ref_sm->input_value(j);
                        auto et = ref_out.get_element_type();
                        if (et == ov::element::dynamic)
                            et = ov::element::f32;
                        new_results.push_back(
                            std::make_shared<v0::Result>(v0::Constant::create(et, ov::Shape{}, {0})));
                    }
                }
            } else {
                new_results.push_back(results[i]);
            }
        }
        body = std::make_shared<ov::Model>(new_results, body->get_parameters());
    };

    expand_body(then_body, then_ridx, then_sm, then_full);
    expand_body(else_body, else_ridx, else_sm, else_full);

    // Build new If
    auto new_if = std::make_shared<v8::If>(if_node->input_value(0));
    new_if->set_then_body(then_body);
    new_if->set_else_body(else_body);
    new_if->set_friendly_name(if_node->get_friendly_name());

    // Re-create input connections using set_input()
    auto new_then_params = then_body->get_parameters();
    auto new_else_params = else_body->get_parameters();

    // Build sets of covered parameters for each branch
    std::unordered_set<size_t> then_param_covered, else_param_covered;

    // First pass: match inputs that appear in both branches at the same input_index
    for (const auto& td : if_node->get_input_descriptions(v8::If::THEN_BODY_INDEX)) {
        for (const auto& ed : if_node->get_input_descriptions(v8::If::ELSE_BODY_INDEX)) {
            if (td->m_input_index == ed->m_input_index) {
                new_if->set_input(if_node->input_value(td->m_input_index),
                                  new_then_params[td->m_body_parameter_index],
                                  new_else_params[ed->m_body_parameter_index]);
                then_param_covered.insert(td->m_body_parameter_index);
                else_param_covered.insert(ed->m_body_parameter_index);
            }
        }
    }

    // Then-only inputs
    for (const auto& d : if_node->get_input_descriptions(v8::If::THEN_BODY_INDEX)) {
        if (then_param_covered.count(d->m_body_parameter_index))
            continue;
        new_if->set_input(if_node->input_value(d->m_input_index),
                          new_then_params[d->m_body_parameter_index],
                          nullptr);
        then_param_covered.insert(d->m_body_parameter_index);
    }

    // Else-only inputs
    for (const auto& d : if_node->get_input_descriptions(v8::If::ELSE_BODY_INDEX)) {
        if (else_param_covered.count(d->m_body_parameter_index))
            continue;
        new_if->set_input(if_node->input_value(d->m_input_index),
                          nullptr,
                          new_else_params[d->m_body_parameter_index]);
        else_param_covered.insert(d->m_body_parameter_index);
    }

    // Set outputs using set_output()
    auto new_then_results = then_body->get_results();
    auto new_else_results = else_body->get_results();

    size_t offset = static_cast<size_t>(n_elements - 1);
    ov::OutputVector sequence_outputs;

    for (size_t old_out = 0; old_out < if_node->get_output_size(); ++old_out) {
        // Find original result indices
        size_t orig_tri = std::numeric_limits<size_t>::max();
        size_t orig_eri = std::numeric_limits<size_t>::max();
        for (const auto& d : if_node->get_output_descriptions(v8::If::THEN_BODY_INDEX))
            if (d->m_output_index == old_out)
                orig_tri = d->m_body_value_index;
        for (const auto& d : if_node->get_output_descriptions(v8::If::ELSE_BODY_INDEX))
            if (d->m_output_index == old_out)
                orig_eri = d->m_body_value_index;

        if (old_out == output_idx) {
            for (int64_t j = 0; j < n_elements; ++j) {
                auto out = new_if->set_output(new_then_results[then_ridx + j], new_else_results[else_ridx + j]);
                sequence_outputs.push_back(out);
            }
        } else {
            size_t new_tri = orig_tri > then_ridx ? orig_tri + offset : orig_tri;
            size_t new_eri = orig_eri > else_ridx ? orig_eri + offset : orig_eri;
            auto out = new_if->set_output(new_then_results[new_tri], new_else_results[new_eri]);
            if_node->output(old_out).replace(out);
        }
    }

    new_if->validate_and_infer_types();
    ov::copy_runtime_info(if_node, new_if);
    return sequence_outputs;
}

bool decompose_sequences_in_model(const std::shared_ptr<ov::Model>& model) {
    bool modified = false;

    // Depth-first: process subgraph bodies first
    for (auto& op : model->get_ordered_ops()) {
        if (auto loop = ov::as_type_ptr<v5::Loop>(op)) {
            if (decompose_sequences_in_model(loop->get_function())) {
                loop->validate_and_infer_types();
                modified = true;
            }
        } else if (auto ifn = ov::as_type_ptr<v8::If>(op)) {
            bool m1 = decompose_sequences_in_model(ifn->get_then_body());
            bool m2 = decompose_sequences_in_model(ifn->get_else_body());
            if (m1 || m2) {
                ifn->validate_and_infer_types();
                modified = true;
            }
        }
    }

    // Phase 1: Resolve known SequenceMark nodes
    for (auto& op : model->get_ordered_ops()) {
        auto sm = ov::as_type_ptr<ov::frontend::SequenceMark>(op);
        if (!sm || sm->empty())
            continue;
        int64_t n = static_cast<int64_t>(sm->get_input_size());
        ov::OutputVector elems;
        for (int64_t i = 0; i < n; ++i)
            elems.push_back(sm->input_value(i));
        replace_sequence_consumers(sm->output(0), elems, n);
        modified = true;
    }

    // Phase 2: Split If nodes that produce SequenceMark outputs
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& op : model->get_ordered_ops()) {
            auto ifn = ov::as_type_ptr<v8::If>(op);
            if (!ifn)
                continue;

            for (size_t oi = 0; oi < ifn->get_output_size(); ++oi) {
                auto output = ifn->output(oi);
                bool needs_split = false;
                for (const auto& t : output.get_target_inputs()) {
                    auto c = t.get_node()->shared_from_this();
                    if (is_framework_node_type(c, "SequenceAt") || is_framework_node_type(c, "SequenceLength") ||
                        is_framework_node_type(c, "SequenceErase") ||
                        ov::as_type_ptr<ov::frontend::SequenceMark>(c)) {
                        needs_split = true;
                        break;
                    }
                }
                if (!needs_split)
                    continue;

                int64_t n = -1;
                for (auto bi : {v8::If::THEN_BODY_INDEX, v8::If::ELSE_BODY_INDEX}) {
                    for (const auto& d : ifn->get_output_descriptions(bi)) {
                        if (d->m_output_index == oi) {
                            auto body = ifn->get_function(bi);
                            auto bres = body->get_results();
                            if (d->m_body_value_index < bres.size()) {
                                int64_t sz = get_sequence_size(bres[d->m_body_value_index]->input_value(0));
                                if (sz > 0)
                                    n = sz;
                            }
                        }
                    }
                    if (n > 0)
                        break;
                }
                if (n <= 0)
                    continue;

                auto elem_outputs = split_if_sequence_output(ifn, oi, n);
                if (elem_outputs.empty())
                    continue;

                replace_sequence_consumers(output, elem_outputs, n);

                // Replace direct SequenceMark consumers
                auto targets = output.get_target_inputs();
                for (const auto& t : targets) {
                    auto csm = ov::as_type_ptr<ov::frontend::SequenceMark>(t.get_node()->shared_from_this());
                    if (csm) {
                        auto new_sm = std::make_shared<ov::frontend::SequenceMark>(elem_outputs);
                        csm->output(0).replace(new_sm);
                    }
                }

                changed = true;
                modified = true;
                break;
            }
            if (changed)
                break;
        }
    }

    // Phase 3: Handle SequenceLength consuming a Parameter (Loop merged state)
    // The sequence length depends on the iteration: 0 on first, N on subsequent.
    // Replace with Select(Greater(iter_counter, 0), N, 0).
    for (auto& op : model->get_ordered_ops()) {
        if (!is_framework_node_type(op, "SequenceLength"))
            continue;

        // Check if it consumes a Parameter
        auto param = ov::as_type_ptr<v0::Parameter>(op->input_value(0).get_node_shared_ptr());
        if (!param)
            continue;

        // Find the max sequence size across all SequenceMarks in the entire model
        // (including nested Loop/If bodies)
        int64_t n = find_max_sequence_size(model);
        if (n <= 0)
            continue;

        // Use the iteration counter (parameter 0 in Loop bodies) to distinguish
        // first iteration (length=0) from subsequent (length=N)
        auto params = model->get_parameters();
        if (params.size() > 0) {
            auto iter = params[0]->output(0);
            auto zero_i64 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto n_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {n});
            auto iter_i64 = std::make_shared<v0::Convert>(iter, ov::element::i64);
            auto gt = std::make_shared<ov::op::v1::Greater>(iter_i64, zero_i64);
            auto select = std::make_shared<ov::op::v1::Select>(gt, n_const, zero_i64);
            select->set_friendly_name(op->get_friendly_name());
            op->output(0).replace(select->output(0));
            modified = true;
        }
    }

    return modified;
}

}  // namespace

bool SequenceLoopDecomposer::run_on_model(const std::shared_ptr<ov::Model>& model) {
    return decompose_sequences_in_model(model);
}

}  // namespace pass
}  // namespace frontend
}  // namespace ov
