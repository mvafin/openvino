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

OutputVector PlaceholderOp(const NodeContext& node) {
    auto ng_et = node.get_attribute<ov::element::Type>("dtype");
    auto ng_shape = node.get_attribute<ov::PartialShape>("shape", ov::PartialShape());
    return {ConstructNgNode<Parameter>(node.get_name(), ng_et, ng_shape)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
