/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_score_updater.hpp"

#ifdef USE_CUDA

namespace LightGBM {

__global__ void AddScoreConstantKernel(
  const double val,
  const data_size_t num_data,
  double* score) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    score[data_index] += val;
  }
}

void CUDAScoreUpdater::LaunchAddScoreConstantKernel(const double val, const size_t offset) {
  const int num_blocks = (num_data_ + num_threads_per_block_) / num_threads_per_block_;
  Log::Debug("Adding init score = %lf", val);
  AddScoreConstantKernel<<<num_blocks, num_threads_per_block_>>>(val, num_data_, cuda_score_.RawData() + offset);
}

__global__ void MultiplyScoreConstantKernel(
  const double val,
  const data_size_t num_data,
  double* score) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    score[data_index] *= val;
  }
}

void CUDAScoreUpdater::LaunchMultiplyScoreConstantKernel(const double val, const size_t offset) {
  const int num_blocks = (num_data_ + num_threads_per_block_) / num_threads_per_block_;
  MultiplyScoreConstantKernel<<<num_blocks, num_threads_per_block_>>>(val, num_data_, cuda_score_.RawData() + offset);
}

}  // namespace LightGBM

#endif  // USE_CUDA
