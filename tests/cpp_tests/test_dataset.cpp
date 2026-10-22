/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <LightGBM/dataset.h>

#include <memory>
#include <vector>

TEST(Dataset, BundlingWithNoLocalSampleColumns) {
  const int num_rows = 20;
  double value = 1.0;
  std::vector<std::unique_ptr<LightGBM::BinMapper>> bin_mappers;
  bin_mappers.emplace_back(new LightGBM::BinMapper());
  bin_mappers.back()->FindBin(&value, 1, num_rows, 255, 1, 1, false,
                              LightGBM::BinType::NumericalBin, false, false, {});
  ASSERT_FALSE(bin_mappers.back()->is_trivial());

  LightGBM::Config config;
  config.enable_bundle = true;
  config.is_enable_sparse = true;
  LightGBM::Dataset dataset(num_rows);
  // A worker can receive a useful bin mapper while its local sample arrays are empty.
  dataset.Construct(&bin_mappers, 1, {{}}, nullptr, nullptr, nullptr, 0, num_rows, config);

  EXPECT_EQ(dataset.num_features(), 1);
  EXPECT_EQ(dataset.num_feature_groups(), 1);
}
