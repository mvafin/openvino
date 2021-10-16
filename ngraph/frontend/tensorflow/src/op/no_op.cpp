// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector NoOp(const NodeContext& node) {
    if (node.get_ng_input_size() == 0) {
        return OutputVector{};
    }
    if (node.get_ng_input_size() != 1) {
        throw errors::InvalidArgument("NoOp has " + to_string(node.get_ng_input_size()) + " inputs, should have 1");
    }
    return OutputVector{node.get_ng_input(0)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
