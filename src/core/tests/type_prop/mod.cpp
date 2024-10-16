// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mod.hpp"

#include "arithmetic_ops.hpp"

using Type = ::testing::Types<ov::op::v1::Mod>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_mod, ArithmeticOperator, Type);
