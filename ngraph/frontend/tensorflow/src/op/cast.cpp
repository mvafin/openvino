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

OutputVector TranslateCastOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);

    auto ng_et = node.get_attribute<element::Type>("DstT");
    return {ConstructNgNode<Convert>(node.get_name(), ng_input, ng_et)};
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
