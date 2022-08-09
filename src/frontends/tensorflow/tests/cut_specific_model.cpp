// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cut_specific_model.hpp"

#include "tf_utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

using TFCutTest = FrontEndCutModelTest;

static CutModelParam getTestData_2in_2out() {
    CutModelParam res;
    res.m_frontEndName = TF_FE;
    res.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    res.m_modelName = "2in_2out/2in_2out.pb";
    res.m_oldInputs = {"inputX1", "inputX2"};
    res.m_newInputs = {"add1:0"};
    res.m_oldOutputs = {"relu3a", "relu3b"};
    res.m_newOutputs = {"add2"};
    res.m_tensorValueName = "inputX2";
    res.m_tensorValue = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    return res;
}

INSTANTIATE_TEST_SUITE_P(TFCutTest,
                         FrontEndCutModelTest,
                         ::testing::Values(getTestData_2in_2out()),
                         FrontEndCutModelTest::getTestCaseName);
