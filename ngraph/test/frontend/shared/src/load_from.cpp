// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/load_from.hpp"
#include <fstream>
#include <ngraph/variant.hpp>
#include "../include/load_from.hpp"
#include "../include/utils.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

std::string
    FrontEndLoadFromTest::getTestCaseName(const testing::TestParamInfo<LoadFromFEParam>& obj)
{
    std::string res = obj.param.m_frontEndName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndLoadFromTest::SetUp()
{
    FrontEndTestUtils::setupTestEnv();
    m_fem = FrontEndManager(); // re-initialize after setting up environment
    m_param = GetParam();
}

///////////////////////////////////////////////////////////////////

/*TEST_P(FrontEndLoadFromTest, testLoadFromFile)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel =
                        m_frontEnd->load_from_file(m_param.m_modelsPath + m_param.m_file));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}*/

///////////////////load from Variants//////////////////////

TEST_P(FrontEndLoadFromTest, testLoadFromFilePath)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelsPath + m_param.m_file));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromTwoFiles)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(m_param.m_modelsPath + m_param.m_files[0],
                                                    m_param.m_modelsPath + m_param.m_files[1]));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    function = m_frontEnd->convert(m_inputModel);
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromStream)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    auto is = std::make_shared<std::ifstream>(m_param.m_modelsPath + m_param.m_stream, std::ios::in | std::ifstream::binary);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(std::dynamic_pointer_cast<std::istream>(is)));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}

TEST_P(FrontEndLoadFromTest, testLoadFromTwoStreams)
{
    std::vector<std::string> frontends;
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(frontends = m_fem.get_available_front_ends());
    ASSERT_NO_THROW(m_frontEnd = m_fem.load_by_framework(m_param.m_frontEndName));
    ASSERT_NE(m_frontEnd, nullptr);

    std::vector<std::shared_ptr<std::ifstream>> is_vec;
    auto p1 = std::make_shared<std::ifstream>(m_param.m_modelsPath + m_param.m_streams[0], std::ios::in | std::ifstream::binary);
    auto p2 = std::make_shared<std::ifstream>(m_param.m_modelsPath + m_param.m_streams[1], std::ios::in | std::ifstream::binary);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->load(std::dynamic_pointer_cast<std::istream>(p1),
                                                    std::dynamic_pointer_cast<std::istream>(p2)));
    ASSERT_NE(m_inputModel, nullptr);

    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}
