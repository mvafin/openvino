// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>
#include "node_context.hpp"

using namespace std;
using namespace ngraph;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateEluOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto alpha = 1.0;  // node.get_attribute<float>("alpha");
    return {ConstructNgNode<opset::Elu>(node.get_name(), input, alpha)};
}
}  // namespace ngraph_bridge
}  // namespace tensorflow
