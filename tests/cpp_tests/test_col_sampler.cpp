/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <LightGBM/c_api.h>
#include <LightGBM/tree.h>

#include <algorithm>
#include <memory>

#include "../../src/treelearner/col_sampler.hpp"

namespace {

class ColSamplerTest : public testing::Test {
 protected:
  void SetUp() override {
    double features[20 * 8];
    for (int row = 0; row < 20; ++row) {
      for (int col = 0; col < 8; ++col) {
        features[row * 8 + col] = row * (col + 1);
      }
    }
    DatasetHandle handle;
    ASSERT_EQ(0, LGBM_DatasetCreateFromMat(features, C_API_DTYPE_FLOAT64, 20, 8, 1,
        "min_data_in_bin=1 min_data_in_leaf=1 num_threads=1", nullptr, &handle));
    data_.reset(static_cast<LightGBM::Dataset*>(handle));
    config_.feature_fraction = 0.5;
    config_.feature_fraction_seed = 123;
  }

  LightGBM::Config config_;
  std::unique_ptr<LightGBM::Dataset> data_;
};

TEST_F(ColSamplerTest, FullFractionRestoresAllFeatures) {
  LightGBM::ColSampler sampler(&config_);
  sampler.SetTrainingData(data_.get());
  const auto& mask = sampler.is_feature_used_bytree();
  ASSERT_EQ(4, std::count(mask.begin(), mask.end(), 1));

  config_.feature_fraction = 1.0;
  sampler.SetConfig(&config_);
  EXPECT_EQ(std::vector<int8_t>(8, 1), mask);
  sampler.ResetByTree();
  EXPECT_EQ(std::vector<int8_t>(8, 1), mask);

  config_.feature_fraction = 0.25;
  sampler.SetConfig(&config_);
  EXPECT_EQ(2, std::count(mask.begin(), mask.end(), 1));
}

TEST_F(ColSamplerTest, ResetTrainingDataRestoresAllFeatures) {
  config_.feature_fraction = 1.0;
  LightGBM::ColSampler sampler(&config_);
  sampler.SetTrainingData(data_.get());
  // Feature-parallel training restricts this mask to the local features.
  sampler.SetIsFeatureUsedByTree(0, false);
  sampler.SetTrainingData(data_.get());
  EXPECT_EQ(std::vector<int8_t>(8, 1), sampler.is_feature_used_bytree());
}

}  // namespace
