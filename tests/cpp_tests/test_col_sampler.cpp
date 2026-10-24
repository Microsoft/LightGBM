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

TEST_F(ColSamplerTest, UnrelatedConfigPreservesSampling) {
  LightGBM::ColSampler control(&config_);
  LightGBM::ColSampler updated(&config_);
  control.SetTrainingData(data_.get());
  updated.SetTrainingData(data_.get());
  config_.learning_rate = 0.2;
  updated.SetConfig(&config_);
  EXPECT_EQ(control.is_feature_used_bytree(), updated.is_feature_used_bytree());
  control.ResetByTree();
  updated.ResetByTree();
  EXPECT_EQ(control.is_feature_used_bytree(), updated.is_feature_used_bytree());
}

TEST_F(ColSamplerTest, SamplingConfigStillUpdates) {
  LightGBM::ColSampler sampler(&config_);
  sampler.SetTrainingData(data_.get());
  config_.feature_fraction = 0.25;
  sampler.SetConfig(&config_);
  const auto& mask = sampler.is_feature_used_bytree();
  EXPECT_EQ(2, std::count(mask.begin(), mask.end(), 1));

  config_.feature_fraction_bynode = 0.5;
  sampler.SetConfig(&config_);
  const auto node_mask = sampler.GetByNode(nullptr, 0);
  EXPECT_EQ(1, std::count(node_mask.begin(), node_mask.end(), 1));

  config_.feature_fraction_seed = 456;
  sampler.SetConfig(&config_);
  LightGBM::ColSampler fresh(&config_);
  fresh.SetTrainingData(data_.get());
  EXPECT_EQ(fresh.is_feature_used_bytree(), sampler.is_feature_used_bytree());
}

}  // namespace
