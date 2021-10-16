// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <memory>
#include <queue>

#include <ngraph/dimension.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/pass.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/op/util/framework_node.hpp>

#include "ie_common.h"

#include "test_common.hpp"

#include "graph_comparator.hpp"
#include "test_tools.hpp"

#define DYN ngraph::Dimension::dynamic()

using TransformationTests = CommonTestUtils::TestsCommon;
