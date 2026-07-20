// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/lower_set_rows_stateful.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

namespace {
std::shared_ptr<ov::op::v0::Parameter> find_param(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& p : model->get_parameters()) {
        if (p->get_friendly_name() == name || p->get_output_tensor(0).get_names().count(name)) {
            return p;
        }
    }
    return nullptr;
}
}  // namespace

bool LowerSetRowsStateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // beam_idx reorders the past KV cache along the batch axis. It is added as a model input by
    // the .gguf builder; with batch=1 / beam_idx=[0] the Gather is an identity, but emitting it
    // is what lets the CPU stateful_sdpa_fusion match.
    auto beam_idx = find_param(model, "beam_idx");

    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        auto set_rows = ov::as_type_ptr<SetRows>(node);
        if (!set_rows) {
            continue;
        }
        auto data = set_rows->input_value(0);  // reshaped to [1, 1, seq, emb]
        auto dst = set_rows->input_value(2);   // destination KV cache (a Parameter at this stage)

        // The cache layout is [1, ctx, n_head_kv, head_size]. Reshape the new rows from the
        // [1, 1, seq, emb] placeholder layout into [1, seq, n_head_kv, head_size] before the
        // Concat along the sequence axis (axis 1). emb == n_head_kv * head_size.
        const auto& dst_ps = dst.get_partial_shape();
        OPENVINO_ASSERT(dst_ps.rank().is_static() && dst_ps.rank().get_length() == 4,
                        "LowerSetRowsStateful expects a rank-4 KV-cache destination");
        int64_t dim2 = dst_ps[2].get_length();
        int64_t dim3 = dst_ps[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data,
            ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t)1, (int64_t)-1, dim2, dim3}),
            false);

        Output<Node> past = dst;
        if (beam_idx) {
            auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            past = std::make_shared<ov::op::v8::Gather>(dst, beam_idx, axis0);
        }
        std::shared_ptr<ov::Node> res = std::make_shared<ov::op::v0::Concat>(OutputVector{past, data}, 1);

        res->set_friendly_name(set_rows->get_friendly_name());
        ov::copy_runtime_info(set_rows, res);
        ov::replace_node(set_rows, res);
        changed = true;
    }
    return changed;
}

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
