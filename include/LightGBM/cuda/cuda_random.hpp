/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_RANDOM_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_RANDOM_HPP_

#ifdef USE_CUDA

#ifndef USE_ROCM
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace LightGBM {

/*!
* \brief A wrapper for random generator
*/
class CUDARandom {
 public:
  /*!
  * \brief Set specific seed
  */
  __device__ void SetSeed(int seed) {
    x = seed;
  }
  /*!
  * \brief Generate random integer, int16 range. [0, 65536]
  * \param lower_bound lower bound
  * \param upper_bound upper bound
  * \return The random integer between [lower_bound, upper_bound)
  */
  __device__ inline int NextShort(int lower_bound, int upper_bound) {
    return (RandInt16()) % (upper_bound - lower_bound) + lower_bound;
  }

  /*!
  * \brief Generate random integer, int32 range
  * \param lower_bound lower bound
  * \param upper_bound upper bound
  * \return The random integer between [lower_bound, upper_bound)
  */
  __device__ inline int NextInt(int lower_bound, int upper_bound) {
    return (RandInt32()) % (upper_bound - lower_bound) + lower_bound;
  }

  /*!
  * \brief Generate random float data
  * \return The random float between [0.0, 1.0)
  */
  __device__ inline float NextFloat() {
    // get random float in [0,1)
    return static_cast<float>(RandInt16()) / (32768.0f);
  }

 private:
  __device__ inline int RandInt16() {
    x = (214013 * x + 2531011);
    return static_cast<int>((x >> 16) & 0x7FFF);
  }

  __device__ inline int RandInt32() {
    x = (214013 * x + 2531011);
    return static_cast<int>(x & 0x7FFFFFFF);
  }

  unsigned int x = 123456789;
};


}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_RANDOM_HPP_
