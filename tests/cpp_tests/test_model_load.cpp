/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <LightGBM/c_api.h>

#include <string>

namespace {

std::string MakeTreeSection(const std::string& left_child, const std::string& right_child,
                            const std::string& split_feature) {
  return
      "Tree=0\n"
      "num_leaves=3\n"
      "num_cat=0\n"
      "split_feature=" + split_feature + "\n"
      "split_gain=1 1\n"
      "threshold=0.5 0.5\n"
      "decision_type=2 2\n"
      "left_child=" + left_child + "\n"
      "right_child=" + right_child + "\n"
      "leaf_value=0.1 0.2 0.3\n"
      "leaf_weight=1 1 1\n"
      "\n";
}

std::string MakeModel(const std::string& left_child, const std::string& right_child,
                      const std::string& split_feature, bool with_tree_sizes = false) {
  std::string tree_section = MakeTreeSection(left_child, right_child, split_feature);
  std::string model =
      "tree\n"
      "version=v4\n"
      "num_class=1\n"
      "num_tree_per_iteration=1\n"
      "label_index=0\n"
      "max_feature_idx=2\n"
      "objective=regression\n"
      "feature_names=f0 f1 f2\n"
      "feature_infos=[0:1] [0:1] [0:1]\n";
  if (with_tree_sizes) {
    model += "tree_sizes=" + std::to_string(tree_section.size()) + "\n";
  }
  model += "\n" + tree_section + "end of trees\n";
  return model;
}

std::string MakeLinearModel(const std::string& leaf_features) {
  std::string model =
      "tree\n"
      "version=v4\n"
      "num_class=1\n"
      "num_tree_per_iteration=1\n"
      "label_index=0\n"
      "max_feature_idx=2\n"
      "objective=regression\n"
      "feature_names=f0 f1 f2\n"
      "feature_infos=[0:1] [0:1] [0:1]\n"
      "\n"
      "Tree=0\n"
      "num_leaves=3\n"
      "num_cat=0\n"
      "split_feature=0 1\n"
      "split_gain=1 1\n"
      "threshold=0.5 0.5\n"
      "decision_type=2 2\n"
      "left_child=1 -1\n"
      "right_child=-2 -3\n"
      "leaf_value=0.1 0.2 0.3\n"
      "leaf_weight=1 1 1\n"
      "is_linear=1\n"
      "leaf_const=0.01 0.02 0.03\n"
      "num_features=2 1 1\n"
      "leaf_features=" + leaf_features + "\n"
      "leaf_coeff=0.5 0.5 0.5 0.5\n"
      "\n"
      "end of trees\n";
  return model;
}

int LoadModel(const std::string& model_text, BoosterHandle* out) {
  int num_iterations = 0;
  return LGBM_BoosterLoadModelFromString(model_text.c_str(), &num_iterations, out);
}

void ExpectLoadError(const std::string& model_text, const std::string& error_substr) {
  BoosterHandle booster = nullptr;
  int result = LoadModel(model_text, &booster);
  EXPECT_NE(0, result) << "model unexpectedly loaded";
  std::string error = LGBM_GetLastError();
  EXPECT_NE(std::string::npos, error.find(error_substr))
      << "expected error containing '" << error_substr << "', got: " << error;
}

}  // namespace

TEST(ModelLoad, ValidTextModelLoads) {
  BoosterHandle booster = nullptr;
  int result = LoadModel(MakeModel("1 -1", "-2 -3", "0 1"), &booster);
  ASSERT_EQ(0, result) << LGBM_GetLastError();
  EXPECT_EQ(0, LGBM_BoosterFree(booster));
}

TEST(ModelLoad, RejectsOutOfRangeLeftChildIndex) {
  // the first variant is the out-of-bounds write case from the security report
  for (const std::string left_child : {"-1000000000 -1", "1000000000 -1", "0 -1"}) {
    ExpectLoadError(MakeModel(left_child, "-2 -3", "0 1"), "child index out of range");
  }
}

TEST(ModelLoad, RejectsOutOfRangeRightChildIndex) {
  ExpectLoadError(MakeModel("1 -1", "-2000000000 -3", "0 1"), "child index out of range");
}

TEST(ModelLoad, RejectsOutOfRangeSplitFeature) {
  // the huge positive variant is the out-of-bounds read case from the security report
  for (const std::string split_feature : {"2000000000 1", "-1 1"}) {
    ExpectLoadError(MakeModel("1 -1", "-2 -3", split_feature), "split_feature out of range");
  }
}

TEST(ModelLoad, RejectsOutOfRangeChildIndexWithTreeSizes) {
  // same rejection must happen via the parallel loading path used when the
  // model file contains tree_sizes, and must return an error (not abort)
  ExpectLoadError(MakeModel("-1000000000 -1", "-2 -3", "0 1", true), "child index out of range");
  ExpectLoadError(MakeModel("1 -1", "-2 -3", "2000000000 1", true), "split_feature out of range");
}

TEST(ModelLoad, RejectsCyclicTree) {
  // node 1 splits on itself
  ExpectLoadError(MakeModel("1 1", "-2 -3", "0 1"), "cycle detected");
}

TEST(ModelLoad, RejectsNodeWithMultipleParents) {
  // node 1 is the left and the right child of the root
  ExpectLoadError(MakeModel("1 -1", "1 -2", "0 1"), "multiple parents");
}

TEST(ModelLoad, RejectsUnreachableNode) {
  // node 1 is never referenced as a child
  ExpectLoadError(MakeModel("-1 -1", "-2 -2", "0 1"), "not reachable");
}

TEST(ModelLoad, ValidLinearModelLoads) {
  BoosterHandle booster = nullptr;
  int result = LoadModel(MakeLinearModel("0 1 2 0"), &booster);
  ASSERT_EQ(0, result) << LGBM_GetLastError();
  EXPECT_EQ(0, LGBM_BoosterFree(booster));
}

TEST(ModelLoad, RejectsOutOfRangeLeafFeatures) {
  ExpectLoadError(MakeLinearModel("2000000000 1 2 0"), "leaf_features out of range");
  ExpectLoadError(MakeLinearModel("-1 1 2 0"), "leaf_features out of range");
}
