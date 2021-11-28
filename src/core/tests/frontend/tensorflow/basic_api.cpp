// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "basic_api.hpp"

#include "tf_utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

using TFBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models{
    std::string("conv2d_s/conv2d.pb"),
    std::string("conv2d_relu/conv2d_relu.pb"),
    std::string("2in_2out/2in_2out.pb"),
    std::string("2in_2out_dynbatch/2in_2out_dynbatch.pb"),
};

INSTANTIATE_TEST_SUITE_P(TFBasicTest,
                         FrontEndBasicTest,
                         ::testing::Combine(::testing::Values(TF_FE),
                                            ::testing::Values(std::string(TEST_TENSORFLOW_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         FrontEndBasicTest::getTestCaseName);
