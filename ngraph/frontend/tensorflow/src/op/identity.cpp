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

OutputVector TranslateIdentityOp(const NodeContext& node) {
    return {node.get_ng_input(0)};
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
