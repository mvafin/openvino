// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/interpolate.hpp>
#include "cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<cpu_memory_format_t>,
        std::vector<cpu_memory_format_t>,
        std::vector<std::string>,
        std::string> InterpolateCPUSpecificParams;

typedef std::tuple<
        LayerTestsDefinitions::InterpolateLayerTestParams,
        InterpolateCPUSpecificParams> InterpolateLayerCPUTestParamsSet;

class InterpolateLayerCPUTest : public testing::WithParamInterface<InterpolateLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::InterpolateLayerTestParams basicParamsSet;
        InterpolateCPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::InterpolateLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::InterpolateLayerTestParams>(
                basicParamsSet, 0));

        std::vector<cpu_memory_format_t> inFmts, outFmts;
        std::vector<std::string> priority;
        std::string selectedType;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        result << "_inFmts=" << CPUTestUtils::fmts2str(inFmts);
        result << "_outFmts=" << CPUTestUtils::fmts2str(outFmts);
        result << "_primitive=" << selectedType;

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::InterpolateLayerTestParams basicParamsSet;
        InterpolateCPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        LayerTestsDefinitions::InterpolateSpecificParams interpolateParams;
        std::vector<size_t> inputShape;
        std::vector<size_t> targetShape;
        auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(interpolateParams, netPrecision, inputShape, targetShape, targetDevice) = basicParamsSet;

        ngraph::op::v4::Interpolate::InterpolateMode mode;
        ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode;
        ngraph::op::v4::Interpolate::NearestMode nearestMode;
        bool antialias;
        std::vector<size_t> padBegin, padEnd;
        double cubeCoef;
        std:tie(mode, coordinateTransformMode, nearestMode, antialias, padBegin, padEnd, cubeCoef) = interpolateParams;

        using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;
        ShapeCalcMode shape_calc_mode = ShapeCalcMode::sizes;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

        auto constant = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {targetShape.size()}, targetShape);

        std::vector<float> scales(targetShape.size(), 1.0f);
        auto scales_const = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {scales.size()}, scales);

        auto scalesInput = std::make_shared<ngraph::opset3::Constant>(scales_const);

        auto secondaryInput = std::make_shared<ngraph::opset3::Constant>(constant);

        ngraph::op::v4::Interpolate::InterpolateAttrs interpolateAttributes{mode, shape_calc_mode, padBegin,
            padEnd, coordinateTransformMode, nearestMode, antialias, cubeCoef};
        auto interpolate = std::make_shared<ngraph::op::v4::Interpolate>(params[0],
                                                                         secondaryInput,
                                                                         scalesInput,
                                                                         interpolateAttributes);
        interpolate->get_rt_info() = CPUTestUtils::setCPUInfo(inFmts, outFmts, priority);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(interpolate)};
        function = std::make_shared<ngraph::Function>(results, params, "interpolate");
    }

    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
};
/**/

TEST_P(InterpolateLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CPUTestUtils::CheckCPUImpl(executableNetwork, "interpolate", inFmts, outFmts, selectedType);
}

namespace {

/* CPU PARAMS */
const auto cpuParams_nChw16c = InterpolateCPUSpecificParams {{nChw16c}, {nChw16c}, {}, "unknown"};
const auto cpuParams_nChw8c = InterpolateCPUSpecificParams {{nChw8c}, {nChw8c}, {}, "unknown"};
const auto cpuParams_nhwc = InterpolateCPUSpecificParams {{nhwc}, {nhwc}, {}, "unknown"};

const std::vector<InterpolateCPUSpecificParams> CPUParams = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nhwc,
};
/* ========== */

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::FP32
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modes = {
        ngraph::op::v4::Interpolate::InterpolateMode::nearest,
        ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::simple,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
        ngraph::op::v4::Interpolate::NearestMode::floor,
        ngraph::op::v4::Interpolate::NearestMode::ceil,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<std::vector<size_t>> pads = {
        {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const auto interpolateCases = ::testing::Combine(
        ::testing::ValuesIn(modes),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_CASE_P(Interpolate_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCases,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(std::vector<size_t>({1, 4, 40, 40})),
                ::testing::Values(std::vector<size_t>({1, 4, 50, 60})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(CPUParams)),
    InterpolateLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
