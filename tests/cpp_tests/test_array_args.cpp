/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/array_args.h>

#include <random>
#include <vector>

using LightGBM::data_size_t;
using LightGBM::score_t;
using LightGBM::ArrayArgs;


TEST(Partition, JustWorks) {
  std::vector<score_t> gradients({0.5f, 5.0f, 1.0f, 2.0f, 2.0f});
  data_size_t middle_begin, middle_end;

  ArrayArgs<score_t>::Partition(&gradients, 0, static_cast<int>(gradients.size()), &middle_begin, &middle_end);

  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
  EXPECT_GT(gradients[0], gradients[middle_begin + 1]);
  EXPECT_GT(gradients[middle_begin + 1], gradients.back());
}

TEST(Partition, PartitionOneElement) {
  std::vector<score_t> gradients({0.5f});
  data_size_t middle_begin, middle_end;
  ArrayArgs<score_t>::Partition(&gradients, 0, static_cast<int>(gradients.size()), &middle_begin, &middle_end);
  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
}

TEST(Partition, Empty) {
  std::vector<score_t> gradients;
  data_size_t middle_begin, middle_end;
  ArrayArgs<score_t>::Partition(&gradients, 0, static_cast<int>(gradients.size()), &middle_begin, &middle_end);

  EXPECT_EQ(middle_begin, -1);
  EXPECT_EQ(middle_end, 0);
}

TEST(Partition, AllEqual) {
  std::vector<score_t> gradients({0.5f, 0.5f, 0.5f});
  data_size_t middle_begin, middle_end;
  ArrayArgs<score_t>::Partition(&gradients, 0, static_cast<int>(gradients.size()), &middle_begin, &middle_end);

  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
  EXPECT_EQ(middle_begin, -1);
  EXPECT_EQ(middle_end, 3);
}
