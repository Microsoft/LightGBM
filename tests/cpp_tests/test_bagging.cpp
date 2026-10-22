/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <LightGBM/c_api.h>
#include <LightGBM/sample_strategy.h>

#include <algorithm>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace {

class BaggingTest : public testing::Test {
 protected:
  void SetUp() override {
    const double features[] = {0.0, 1.0, 2.0, 3.0};
    DatasetHandle handle;
    ASSERT_EQ(0, LGBM_DatasetCreateFromMat(features, C_API_DTYPE_FLOAT64, 4, 1, 1,
        "min_data_in_bin=1 min_data_in_leaf=1 num_threads=1", nullptr, &handle));
    data_.reset(static_cast<LightGBM::Dataset*>(handle));
    const float labels[] = {0.0f, 1.0f, 0.0f, 1.0f};
    ASSERT_TRUE(data_->SetFloatField("label", labels, 4));
    config_.bagging_freq = 1;
    config_.bagging_seed = 5008;  // The first four draws exceed 0.5.
    config_.force_col_wise = true;
    config_.min_data_in_leaf = 1;
  }

  void SetSeed(bool randomized) {
    if (randomized) {
      std::random_device random;
      config_.bagging_seed = std::uniform_int_distribution<int>(0, 1000000)(random);
    }
  }

  std::unique_ptr<LightGBM::SampleStrategy> Sample() {
    objective_.reset(LightGBM::ObjectiveFunction::CreateObjectiveFunction("binary", config_));
    objective_->Init(data_->metadata(), data_->num_data());
    learner_.reset(LightGBM::TreeLearner::CreateTreeLearner("serial", "cpu", &config_, false));
    learner_->Init(data_.get(), objective_->IsConstantHessian());
    std::unique_ptr<LightGBM::SampleStrategy> strategy(
        LightGBM::SampleStrategy::CreateSampleStrategy(&config_, data_.get(), objective_.get(), 1));
    strategy->ResetSampleConfig(&config_, true);
    strategy->Bagging(0, learner_.get(), nullptr, nullptr);
    return strategy;
  }

  LightGBM::Config config_;
  std::unique_ptr<LightGBM::Dataset> data_;
  std::unique_ptr<LightGBM::ObjectiveFunction> objective_;
  std::unique_ptr<LightGBM::TreeLearner> learner_;
};

class RowBaggingTest : public BaggingTest, public testing::WithParamInterface<std::tuple<bool, double, bool>> {};

TEST_P(RowBaggingTest, NonEmptySamplePreservesIndices) {
  const bool balanced = std::get<0>(GetParam());
  const double fraction = std::get<1>(GetParam());
  const bool randomized = std::get<2>(GetParam());
  SetSeed(randomized);
  SCOPED_TRACE(testing::Message() << "bagging_seed=" << config_.bagging_seed);
  if (balanced) {
    config_.pos_bagging_fraction = fraction;
    config_.neg_bagging_fraction = fraction;
  } else {
    config_.bagging_fraction = fraction;
  }
  auto strategy = Sample();
  ASSERT_GE(strategy->bag_data_cnt(), 1);
  ASSERT_LE(strategy->bag_data_cnt(), data_->num_data());
  if (!randomized) {
    EXPECT_EQ(1, strategy->bag_data_cnt());
  }
  const auto& sampled_indices = strategy->bag_data_indices();
  std::vector<LightGBM::data_size_t> indices(sampled_indices.begin(), sampled_indices.end());
  std::sort(indices.begin(), indices.end());
  EXPECT_EQ((std::vector<LightGBM::data_size_t>{0, 1, 2, 3}), indices);
}

INSTANTIATE_TEST_SUITE_P(FixedSeed, RowBaggingTest,
                        testing::Combine(testing::Bool(), testing::Values(0.5, 1e-6), testing::Values(false)));
INSTANTIATE_TEST_SUITE_P(Randomized, RowBaggingTest,
                        testing::Combine(testing::Bool(), testing::Values(0.5, 1e-6), testing::Values(true)));

class QueryBaggingTest : public BaggingTest, public testing::WithParamInterface<bool> {};

TEST_P(QueryBaggingTest, NonEmptySamplePreservesWholeQueries) {
  const bool randomized = GetParam();
  SetSeed(randomized);
  SCOPED_TRACE(testing::Message() << "bagging_seed=" << config_.bagging_seed);
  const int groups[] = {1, 3};
  ASSERT_TRUE(data_->SetIntField("group", groups, 2));
  config_.bagging_by_query = true;
  config_.bagging_fraction = 0.5;
  auto strategy = Sample();
  const auto query_count = strategy->num_sampled_queries();
  ASSERT_GE(query_count, 1);
  ASSERT_LE(query_count, 2);
  if (!randomized) {
    EXPECT_EQ(1, query_count);
  }
  const auto queries = strategy->sampled_query_indices();
  if (query_count == 2) {
    EXPECT_NE(queries[0], queries[1]);
  }
  const auto boundaries = data_->metadata().query_boundaries();
  std::vector<LightGBM::data_size_t> expected;
  for (LightGBM::data_size_t i = 0; i < query_count; ++i) {
    ASSERT_GE(queries[i], 0);
    ASSERT_LT(queries[i], 2);
    for (auto row = boundaries[queries[i]]; row < boundaries[queries[i] + 1]; ++row) {
      expected.push_back(row);
    }
  }
  ASSERT_EQ(expected.size(), strategy->bag_data_cnt());
  const auto& indices = strategy->bag_data_indices();
  EXPECT_EQ(expected, (std::vector<LightGBM::data_size_t>(indices.begin(), indices.begin() + expected.size())));
}

INSTANTIATE_TEST_SUITE_P(FixedSeed, QueryBaggingTest, testing::Values(false));
INSTANTIATE_TEST_SUITE_P(Randomized, QueryBaggingTest, testing::Values(true));

}  // namespace
