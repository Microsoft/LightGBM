/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"

namespace LightGBM {

CUDALeafSplits::CUDALeafSplits(const data_size_t num_data):
num_data_(num_data) {}

CUDALeafSplits::~CUDALeafSplits() {}

void CUDALeafSplits::Init(const bool use_quantized_grad) {
  num_blocks_init_from_gradients_ = (num_data_ + NUM_THREADS_PER_BLOCK_LEAF_SPLITS - 1) / NUM_THREADS_PER_BLOCK_LEAF_SPLITS;

  // allocate more memory for sum reduction in CUDA
  // only the first element records the final sum
  cuda_sum_of_gradients_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  cuda_sum_of_hessians_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  if (use_quantized_grad) {
    cuda_sum_of_gradients_hessians_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  }

  cuda_struct_.Resize(1);
}

void CUDALeafSplits::InitValues() {
  LaunchInitValuesEmptyKernel();
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDALeafSplits::InitValues(
  const double lambda_l1, const double lambda_l2,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t* cuda_bagging_data_indices, const data_size_t* cuda_data_indices_in_leaf,
  const data_size_t num_used_indices, hist_t* cuda_hist_in_leaf,
  double* root_sum_gradients, double* root_sum_hessians) {
  cuda_gradients_ = cuda_gradients;
  cuda_hessians_ = cuda_hessians;
  cuda_sum_of_gradients_buffer_.SetValue(0);
  cuda_sum_of_hessians_buffer_.SetValue(0);
  LaunchInitValuesKernel(lambda_l1, lambda_l2, cuda_bagging_data_indices, cuda_data_indices_in_leaf, num_used_indices, cuda_hist_in_leaf);
  CopyFromCUDADeviceToHost<double>(root_sum_gradients, cuda_sum_of_gradients_buffer_.RawData(), 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(root_sum_hessians, cuda_sum_of_hessians_buffer_.RawData(), 1, __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDALeafSplits::InitValues(
  const double lambda_l1, const double lambda_l2,
  const int16_t* cuda_gradients_and_hessians,
  const data_size_t* cuda_bagging_data_indices,
  const data_size_t* cuda_data_indices_in_leaf, const data_size_t num_used_indices,
  hist_t* cuda_hist_in_leaf, double* root_sum_gradients, double* root_sum_hessians,
  const score_t* grad_scale, const score_t* hess_scale) {
  cuda_gradients_ = reinterpret_cast<const score_t*>(cuda_gradients_and_hessians);
  cuda_hessians_ = nullptr;
  LaunchInitValuesKernel(lambda_l1, lambda_l2, cuda_bagging_data_indices, cuda_data_indices_in_leaf, num_used_indices, cuda_hist_in_leaf, grad_scale, hess_scale);
  CopyFromCUDADeviceToHost<double>(root_sum_gradients, cuda_sum_of_gradients_buffer_.RawData(), 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(root_sum_hessians, cuda_sum_of_hessians_buffer_.RawData(), 1, __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDALeafSplits::Resize(const data_size_t num_data) {
  num_data_ = num_data;
  num_blocks_init_from_gradients_ = (num_data + NUM_THREADS_PER_BLOCK_LEAF_SPLITS - 1) / NUM_THREADS_PER_BLOCK_LEAF_SPLITS;
  cuda_sum_of_gradients_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  cuda_sum_of_hessians_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  cuda_sum_of_gradients_hessians_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
}

}  // namespace LightGBM

#endif  // USE_CUDA
