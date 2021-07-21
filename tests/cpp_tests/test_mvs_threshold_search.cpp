/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/array_args.h>

#include <random>

using LightGBM::data_size_t;
using LightGBM::score_t;
using LightGBM::ArrayArgs;

template<class Value>
double ComputeExpectationOfMVS(const std::vector<Value> &grads, double threshold) {
  double expectation = 0.0;
  for (const auto &value : grads) {
    if (value >= threshold) {
      expectation += 1.;
    } else {
      expectation += value / threshold;
    }
  }
  return expectation;
}

void ComputeSamplingRate(std::vector<score_t> gradients,
                         const double sampling_fraction,
                         double *expected_sample_size,
                         double *resulting_sample_size) {
  EXPECT_TRUE(expected_sample_size);
  EXPECT_TRUE(resulting_sample_size);

  *expected_sample_size = sampling_fraction * static_cast<double>(gradients.size());

  double threshold = ArrayArgs<score_t>::CalculateThresholdMVS(&gradients, 0, gradients.size(), *expected_sample_size);

  *resulting_sample_size = ComputeExpectationOfMVS(gradients, threshold);
}

template<class VAL_T>
std::vector<VAL_T> GenerateRandomVector(std::mt19937_64 *rng, size_t size) {
  std::uniform_real_distribution<VAL_T> distribution(1., 2.0f);
  std::vector<VAL_T> result;
  for (size_t i = 0; i < size; ++i) {
    result.emplace_back(distribution(*rng));
  }
  return result;
}

TEST(SearchThresholdMVS, Basic) {
  std::vector<score_t> gradients({0.5f, 5.0f, 1.0f, 2.0f, 2.0f});
  double expected, resulting;
  ComputeSamplingRate(gradients, 0.5, &expected, &resulting);
  EXPECT_DOUBLE_EQ(expected, resulting);
}

TEST(SearchThresholdMVS, SameGradientValue) {
  std::vector<score_t> gradients;

  for (size_t i = 0; i < 10; ++i) {
    gradients.emplace_back(1.);
  }

  double expected, resulting;
  ComputeSamplingRate(gradients, 0.5, &expected, &resulting);
  EXPECT_DOUBLE_EQ(expected, resulting);
  EXPECT_DOUBLE_EQ(resulting, 5.);
}

TEST(SearchThresholdMVS, LargeTest) {
  std::mt19937_64 rng(42);
  const size_t number_of_iterations = 100;
  for (size_t i = 0; i < number_of_iterations; ++i) {
    std::vector<score_t> grad = GenerateRandomVector<score_t>(&rng, 10000);

    double expected, resulting;
    ComputeSamplingRate(std::move(grad), 0.01 + (0.98 * i) / number_of_iterations, &expected, &resulting);
    EXPECT_NEAR(expected, resulting, 1e-3);
  }
}

TEST(ArrayArgs, Partition) {
  std::vector<score_t> gradients({0.5f, 5.0f, 1.0f, 2.0f, 2.0f});
  data_size_t middle_begin = -1, middle_end = gradients.size();

  ArrayArgs<score_t>::Partition(&gradients, 0, gradients.size(), &middle_begin, &middle_end);

  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
  EXPECT_GT(gradients[0], gradients[middle_begin + 1]);
  EXPECT_GT(gradients[middle_begin + 1], gradients.back());
}

TEST(ArrayArgs, PartitionOneElement) {
  std::vector<score_t> gradients({0.5f});
  data_size_t middle_begin = -1, middle_end = gradients.size();
  ArrayArgs<score_t>::Partition(&gradients, 0, gradients.size(), &middle_begin, &middle_end);
  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
}
