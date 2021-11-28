// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "set_element_type.hpp"

#include "tf_utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

using TFCutTest = FrontEndElementTypeTest;

static SetTypeFEParam getTestData_relu() {
    SetTypeFEParam res;
    res.m_frontEndName = TF_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "relu/relu.pb";
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFCutTest,
                         FrontEndElementTypeTest,
                         ::testing::Values(getTestData_relu()),
                         FrontEndElementTypeTest::getTestCaseName);
