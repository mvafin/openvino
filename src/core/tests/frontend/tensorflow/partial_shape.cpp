// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partial_shape.hpp"

#include "tf_utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

using TFPartialShapeTest = FrontEndPartialShapeTest;

static PartShape getTestShape_2in_2out() {
    PartShape res;
    res.m_modelName = "2in_2out/2in_2out.pb";
    res.m_tensorName = "inputX1";
    res.m_oldPartialShape = PartialShape{1, 3, 3, 1};
    res.m_newPartialShape = PartialShape{2, 3, 3, 1};
    return res;
}

static PartShape getTestShape_2in_2out_dynbatch() {
    PartShape res;
    res.m_modelName = "2in_2out_dynbatch/2in_2out_dynbatch.pb";
    res.m_tensorName = "inputX1";
    res.m_oldPartialShape = PartialShape{Dimension::dynamic(), 3, 3, 1};
    res.m_newPartialShape = PartialShape{2, 3, 3, 1};
    return res;
}

static PartShape getTestShape_conv2d() {
    PartShape res;
    res.m_modelName = "conv2d_s/conv2d.pb";
    res.m_tensorName = "x";
    res.m_oldPartialShape = PartialShape{1, 5, 5, 1};
    res.m_newPartialShape = PartialShape{1, 10, 10, 1};
    return res;
}

static PartShape getTestShape_conv2d_setDynamicBatch() {
    PartShape res;
    res.m_modelName = "conv2d_s/conv2d.pb";
    res.m_tensorName = "x";
    res.m_oldPartialShape = PartialShape{1, 5, 5, 1};
    res.m_newPartialShape = PartialShape{Dimension::dynamic(), 10, 10, 1};
    return res;
}

static PartShape getTestShape_conv2d_relu() {
    PartShape res;
    res.m_modelName = "conv2d_relu/conv2d_relu.pb";
    res.m_tensorName = "xxx";
    res.m_oldPartialShape = PartialShape{1, 5, 5, 1};
    res.m_newPartialShape = PartialShape{5, 5, 5, 1};
    return res;
}

INSTANTIATE_TEST_SUITE_P(
    PDPDPartialShapeTest,
    FrontEndPartialShapeTest,
    ::testing::Combine(::testing::Values(BaseFEParam{TF_FE, std::string(TEST_TENSORFLOW_MODELS_DIRNAME)}),
                       ::testing::ValuesIn(std::vector<PartShape>{getTestShape_2in_2out(),
                                                                  getTestShape_conv2d_relu(),
                                                                  getTestShape_conv2d(),
                                                                  getTestShape_conv2d_setDynamicBatch(),
                                                                  getTestShape_2in_2out_dynbatch()})),
    FrontEndPartialShapeTest::getTestCaseName);
