/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_RANDOM_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_RANDOM_H_

#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace LightGBM {

/*!
* \brief A wrapper for random generator
*/
class Random {
 public:
  /*!
  * \brief Constructor, with random seed
  */
  Random() {
    std::random_device rd;
    auto generator = std::mt19937(rd());
    std::uniform_int_distribution<int> distribution(0, x);
    x = distribution(generator);
  }
  /*!
  * \brief Constructor, with specific seed
  */
  explicit Random(int seed) {
    x = seed;
  }
  /*!
  * \brief Generate random integer, int16 range. [0, 65536]
  * \param lower_bound lower bound
  * \param upper_bound upper bound
  * \return The random integer between [lower_bound, upper_bound)
  */
  inline int NextShort(int lower_bound, int upper_bound) {
    return (RandInt16()) % (upper_bound - lower_bound) + lower_bound;
  }

  /*!
  * \brief Generate random integer, int32 range
  * \param lower_bound lower bound
  * \param upper_bound upper bound
  * \return The random integer between [lower_bound, upper_bound)
  */
  inline int NextInt(int lower_bound, int upper_bound) {
    return (RandInt32()) % (upper_bound - lower_bound) + lower_bound;
  }

  /*!
  * \brief Generate random float data
  * \return The random float between [0.0, 1.0)
  */
  inline float NextFloat() {
    // get random float in [0,1)
    return static_cast<float>(RandInt16()) / (32768.0f);
  }
  /*!
  * \brief Sample K data from {0,1,...,N-1}
  * \param N
  * \param K
  * \return K Ordered sampled data from {0,1,...,N-1}
  */
  inline std::vector<int> Sample(int N, int K) {
    std::vector<int> ret;
    ret.reserve(K);
    if (K > N || K <= 0) {
      return ret;
    } else if (K == N) {
      for (int i = 0; i < N; ++i) {
        ret.push_back(i);
      }
    } else if (K > 1 && K > (N / std::log2(K))) {
      for (int i = 0; i < N; ++i) {
        double prob = (K - ret.size()) / static_cast<double>(N - i);
        if (NextFloat() < prob) {
          ret.push_back(i);
        }
      }
    } else {
      std::set<int> sample_set;
      for (int r = N - K; r < N; ++r) {
        int v = NextInt(0, r + 1);
        if (!sample_set.insert(v).second) {
          sample_set.insert(r);
        }
      }
      for (auto iter = sample_set.begin(); iter != sample_set.end(); ++iter) {
        ret.push_back(*iter);
      }
    }
    return ret;
  }

 private:
  inline int RandInt16() {
    x = (214013 * x + 2531011);
    return static_cast<int>((x >> 16) & 0x7FFF);
  }

  inline int RandInt32() {
    x = (214013 * x + 2531011);
    return static_cast<int>(x & 0x7FFFFFFF);
  }

  unsigned int x = 123456789;
};


}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_RANDOM_H_
