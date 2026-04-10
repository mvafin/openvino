// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov::frontend::onnx::tests;

namespace {
std::shared_ptr<ov::op::util::FrameworkNode> get_framework_node_with_out_name(const std::shared_ptr<ov::Model>& model,
                                                                              const std::string& out_name) {
    for (const auto& op : model->get_ops()) {
        if (auto framework_node = ov::as_type_ptr<ov::op::util::FrameworkNode>(op)) {
            for (const auto& out : op->outputs()) {
                if (out.get_any_name() == out_name) {
                    return framework_node;
                }
            }
        }
    }
    return nullptr;
}
}  // namespace

TEST(ONNXFeConvertPartially, insert_framework_node_if_unsupported) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("unsupported_ops/add_unsupported.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::opset11::Add>(model), 1);
    const auto unsupported_add = get_framework_node_with_out_name(model, "Y");
    ASSERT_TRUE(unsupported_add);
    EXPECT_EQ(unsupported_add->get_attrs().get_type_name(), "UnsupportedAdd");
    EXPECT_EQ(unsupported_add->get_attrs().get_opset_name(), "test_domain");
}

TEST(ONNXFeConvertPartially, insert_more_framework_nodes_if_unsupported) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("unsupported_ops/two_unsupported_nodes.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::opset11::Add>(model), 1);
    const auto unsupported_add = get_framework_node_with_out_name(model, "X");
    ASSERT_TRUE(unsupported_add);
    EXPECT_EQ(unsupported_add->get_attrs().get_type_name(), "UnsupportedAdd");

    const auto unsupported_abs = get_framework_node_with_out_name(model, "Y_out");
    ASSERT_TRUE(unsupported_abs);
    EXPECT_EQ(unsupported_abs->get_attrs().get_type_name(), "UnsupportedAbs");
}

// validation error - onnx/instance_norm_bad_scale_type.onnx
TEST(ONNXFeConvertPartially, insert_framework_node_if_onnx_validation_exception) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("instance_norm_bad_scale_type.onnx"));
    ASSERT_TRUE(model);

    const auto incorrect_instance_norm = get_framework_node_with_out_name(model, "y");
    ASSERT_TRUE(incorrect_instance_norm);
    EXPECT_EQ(incorrect_instance_norm->get_attrs().get_type_name(), "InstanceNormalization");
}

TEST(ONNXFeConvertPartially, insert_framework_node_if_other_translation_exception) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("depth_to_space_bad_mode.onnx"));
    ASSERT_TRUE(model);

    const auto incorrect_dts = get_framework_node_with_out_name(model, "B");
    ASSERT_TRUE(incorrect_dts);
    EXPECT_EQ(incorrect_dts->get_attrs().get_type_name(), "DepthToSpace");
}

TEST(ONNXFeConvertPartially, insert_framework_nodes_if_both_unsupported_and_other_translation_exception) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("unsupported_ops/unsupported_add_and_incorrect_dts.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::opset11::Abs>(model), 1);
    const auto incorrect_dts = get_framework_node_with_out_name(model, "B");
    ASSERT_TRUE(incorrect_dts);
    EXPECT_EQ(incorrect_dts->get_attrs().get_type_name(), "DepthToSpace");

    const auto unsupported_add = get_framework_node_with_out_name(model, "Y");
    ASSERT_TRUE(unsupported_add);
    EXPECT_EQ(unsupported_add->get_attrs().get_type_name(), "UnsupportedAdd");
}

/// @brief Test that ONNXFrameworkNode clone_with_new_inputs works for custom ops in Loop body.
/// When convert_partially processes a Loop, the body is cloned during validation.
/// Without the clone-safe fix, this crashes due to dangling ONNX decoder references.
TEST(ONNXFeConvertPartially, custom_op_in_loop_clone_safe) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("controlflow/custom_op_in_loop.onnx"));
    ASSERT_TRUE(model);

    // The model should contain a Loop node
    EXPECT_EQ(count_ops_of_type<ov::opset11::Loop>(model), 1);

    // Clone the model — triggers clone_with_new_inputs on all nodes including
    // ONNXFrameworkNode inside the Loop body
    std::shared_ptr<ov::Model> cloned;
    OV_ASSERT_NO_THROW(cloned = model->clone());
    ASSERT_TRUE(cloned);
    EXPECT_EQ(count_ops_of_type<ov::opset11::Loop>(cloned), 1);
}

/// @brief Test that FusedMatMul handles dynamic element types from unsupported predecessor ops.
/// Without the fix, CHECK_VALID_NODE fails when input element type is dynamic.
TEST(ONNXFeConvertPartially, fusedmatmul_with_dynamic_input_type) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("com.microsoft/fusedmatmul_dynamic_type.onnx"));
    ASSERT_TRUE(model);

    // FusedMatMul should be translated to MatMul+Multiply despite dynamic input type
    EXPECT_EQ(count_ops_of_type<ov::opset11::MatMul>(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::opset11::Multiply>(model), 1);
}
