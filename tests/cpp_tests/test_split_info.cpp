/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <limits>

#include "../../src/treelearner/split_info.hpp"

namespace {

template <typename SplitInfoType>
void ExpectNaNGainTreatedAsMinScore() {
  const double nan = std::numeric_limits<double>::quiet_NaN();

  SplitInfoType finite;
  finite.feature = 3;
  finite.gain = 0.0;

  SplitInfoType nan_smaller_feature;
  nan_smaller_feature.feature = 1;
  nan_smaller_feature.gain = nan;

  SplitInfoType nan_larger_feature;
  nan_larger_feature.feature = 2;
  nan_larger_feature.gain = nan;

  SplitInfoType nan_same_feature;
  nan_same_feature.feature = nan_smaller_feature.feature;
  nan_same_feature.gain = nan;

  SplitInfoType min_score;
  min_score.feature = nan_smaller_feature.feature;
  min_score.gain = LightGBM::kMinScore;

  EXPECT_TRUE(finite > nan_smaller_feature);
  EXPECT_FALSE(nan_smaller_feature > finite);
  EXPECT_TRUE(nan_smaller_feature > nan_larger_feature);
  EXPECT_FALSE(nan_larger_feature > nan_smaller_feature);
  EXPECT_TRUE(nan_smaller_feature == nan_same_feature);
  EXPECT_TRUE(nan_smaller_feature == min_score);
  EXPECT_FALSE(nan_smaller_feature == nan_larger_feature);
}

TEST(SplitInfo, NaNGainIsTreatedAsMinScore) {
  ExpectNaNGainTreatedAsMinScore<LightGBM::SplitInfo>();
}

TEST(LightSplitInfo, NaNGainIsTreatedAsMinScore) {
  ExpectNaNGainTreatedAsMinScore<LightGBM::LightSplitInfo>();
}

}  // namespace
