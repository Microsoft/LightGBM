/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */


#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"
#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_rocm_interop.h>

namespace LightGBM {

template <bool USE_INDICES>
__global__ void CUDAInitValuesKernel1(const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t num_data, const data_size_t* cuda_bagging_data_indices,
  double* cuda_sum_of_gradients, double* cuda_sum_of_hessians) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double gradient = 0.0f;
  double hessian = 0.0f;
  if (data_index < num_data) {
    gradient = USE_INDICES ? cuda_gradients[cuda_bagging_data_indices[data_index]] : cuda_gradients[data_index];
    hessian = USE_INDICES ? cuda_hessians[cuda_bagging_data_indices[data_index]] : cuda_hessians[data_index];
  }
  const double block_sum_gradient = ShuffleReduceSum<double>(gradient, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const double block_sum_hessian = ShuffleReduceSum<double>(hessian, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_of_gradients[blockIdx.x] += block_sum_gradient;
    cuda_sum_of_hessians[blockIdx.x] += block_sum_hessian;
  }
}

__global__ void CUDAInitValuesKernel2(
  const double lambda_l1,
  const double lambda_l2,
  const int num_blocks_to_reduce,
  double* cuda_sum_of_gradients,
  double* cuda_sum_of_hessians,
  const data_size_t num_data,
  const data_size_t* cuda_data_indices_in_leaf,
  hist_t* cuda_hist_in_leaf,
  CUDALeafSplitsStruct* cuda_struct) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  double thread_sum_of_gradients = 0.0f;
  double thread_sum_of_hessians = 0.0f;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks_to_reduce; block_index += static_cast<int>(blockDim.x)) {
    thread_sum_of_gradients += cuda_sum_of_gradients[block_index];
    thread_sum_of_hessians += cuda_sum_of_hessians[block_index];
  }
  const double sum_of_gradients = ShuffleReduceSum<double>(thread_sum_of_gradients, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const double sum_of_hessians = ShuffleReduceSum<double>(thread_sum_of_hessians, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_of_hessians[0] = sum_of_hessians;
    cuda_struct->leaf_index = 0;
    cuda_struct->sum_of_gradients = sum_of_gradients;
    cuda_struct->sum_of_hessians = sum_of_hessians;
    cuda_struct->num_data_in_leaf = num_data;
    const bool use_l1 = lambda_l1 > 0.0f;
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    cuda_struct->data_indices_in_leaf = cuda_data_indices_in_leaf;
    cuda_struct->hist_in_leaf = cuda_hist_in_leaf;
  }
}

template <bool USE_INDICES>
__global__ void CUDAInitValuesKernel3(const int16_t* cuda_gradients_and_hessians,
  const data_size_t num_data, const data_size_t* cuda_bagging_data_indices,
  double* cuda_sum_of_gradients, double* cuda_sum_of_hessians, int64_t* cuda_sum_of_hessians_hessians,
  const score_t* grad_scale_pointer, const score_t* hess_scale_pointer) {
  const score_t grad_scale = *grad_scale_pointer;
  const score_t hess_scale = *hess_scale_pointer;
  __shared__ int64_t shared_mem_buffer[WARPSIZE];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  int64_t int_gradient = 0;
  int64_t int_hessian = 0;
  if (data_index < num_data) {
    int_gradient = USE_INDICES ? cuda_gradients_and_hessians[2 * cuda_bagging_data_indices[data_index] + 1] :
      cuda_gradients_and_hessians[2 * data_index + 1];
    int_hessian = USE_INDICES ? cuda_gradients_and_hessians[2 * cuda_bagging_data_indices[data_index]] :
      cuda_gradients_and_hessians[2 * data_index];
  }
  const int64_t block_sum_gradient = ShuffleReduceSum<int64_t>(int_gradient, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const int64_t block_sum_hessian = ShuffleReduceSum<int64_t>(int_hessian, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_of_gradients[blockIdx.x] = block_sum_gradient * grad_scale;
    cuda_sum_of_hessians[blockIdx.x] = block_sum_hessian * hess_scale;
    cuda_sum_of_hessians_hessians[blockIdx.x] = ((block_sum_gradient << 32) | block_sum_hessian);
  }
}

__global__ void CUDAInitValuesKernel4(
  const double lambda_l1,
  const double lambda_l2,
  const int num_blocks_to_reduce,
  double* cuda_sum_of_gradients,
  double* cuda_sum_of_hessians,
  int64_t* cuda_sum_of_gradients_hessians,
  const data_size_t num_data,
  const data_size_t* cuda_data_indices_in_leaf,
  hist_t* cuda_hist_in_leaf,
  CUDALeafSplitsStruct* cuda_struct) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  double thread_sum_of_gradients = 0.0f;
  double thread_sum_of_hessians = 0.0f;
  int64_t thread_sum_of_gradients_hessians = 0;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks_to_reduce; block_index += static_cast<int>(blockDim.x)) {
    thread_sum_of_gradients += cuda_sum_of_gradients[block_index];
    thread_sum_of_hessians += cuda_sum_of_hessians[block_index];
    thread_sum_of_gradients_hessians += cuda_sum_of_gradients_hessians[block_index];
  }
  const double sum_of_gradients = ShuffleReduceSum<double>(thread_sum_of_gradients, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const double sum_of_hessians = ShuffleReduceSum<double>(thread_sum_of_hessians, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const double sum_of_gradients_hessians = ShuffleReduceSum<int64_t>(
    thread_sum_of_gradients_hessians,
    reinterpret_cast<int64_t*>(shared_mem_buffer),
    blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_of_hessians[0] = sum_of_hessians;
    cuda_struct->leaf_index = 0;
    cuda_struct->sum_of_gradients = sum_of_gradients;
    cuda_struct->sum_of_hessians = sum_of_hessians;
    cuda_struct->sum_of_gradients_hessians = sum_of_gradients_hessians;
    cuda_struct->num_data_in_leaf = num_data;
    const bool use_l1 = lambda_l1 > 0.0f;
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    cuda_struct->data_indices_in_leaf = cuda_data_indices_in_leaf;
    cuda_struct->hist_in_leaf = cuda_hist_in_leaf;
  }
}

__global__ void InitValuesEmptyKernel(CUDALeafSplitsStruct* cuda_struct) {
  cuda_struct->leaf_index = -1;
  cuda_struct->sum_of_gradients = 0.0f;
  cuda_struct->sum_of_hessians = 0.0f;
  cuda_struct->num_data_in_leaf = 0;
  cuda_struct->gain = 0.0f;
  cuda_struct->leaf_value = 0.0f;
  cuda_struct->data_indices_in_leaf = nullptr;
  cuda_struct->hist_in_leaf = nullptr;
}

__global__ void ReduceGradKernel(
  const int num_blocks_to_reduce,
  double* cuda_sum_of_gradients,
  double* cuda_sum_of_hessians,
  const data_size_t num_data) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  double thread_sum_of_gradients = 0.0f;
  double thread_sum_of_hessians = 0.0f;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks_to_reduce; block_index += static_cast<int>(blockDim.x)) {
    thread_sum_of_gradients += cuda_sum_of_gradients[block_index];
    thread_sum_of_hessians += cuda_sum_of_hessians[block_index];
  }
  const double sum_of_gradients = ShuffleReduceSum<double>(thread_sum_of_gradients, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const double sum_of_hessians = ShuffleReduceSum<double>(thread_sum_of_hessians, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_of_gradients[0] = sum_of_gradients;
    cuda_sum_of_hessians[0] = sum_of_hessians;
  }
}

__global__ void ReduceGradKernel(
  const int num_blocks_to_reduce,
  double* cuda_sum_of_gradients,
  double* cuda_sum_of_hessians,
  int64_t* cuda_sum_of_gradients_hessians,
  const data_size_t num_data) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  double thread_sum_of_gradients = 0.0f;
  double thread_sum_of_hessians = 0.0f;
  int64_t thread_sum_of_gradients_hessians = 0;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks_to_reduce; block_index += static_cast<int>(blockDim.x)) {
    thread_sum_of_gradients += cuda_sum_of_gradients[block_index];
    thread_sum_of_hessians += cuda_sum_of_hessians[block_index];
    thread_sum_of_gradients_hessians += cuda_sum_of_gradients_hessians[block_index];
  }
  const double sum_of_gradients = ShuffleReduceSum<double>(thread_sum_of_gradients, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const double sum_of_hessians = ShuffleReduceSum<double>(thread_sum_of_hessians, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const int64_t sum_of_gradients_hessians = ShuffleReduceSum<int64_t>(
    thread_sum_of_gradients_hessians,
    reinterpret_cast<int64_t*>(shared_mem_buffer),
    blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_of_gradients[0] = sum_of_gradients;
    cuda_sum_of_hessians[0] = sum_of_hessians;
    cuda_sum_of_gradients_hessians[0] = sum_of_gradients_hessians;
  }
}

__global__ void CUDAInitSetValuesKernel(
  const double lambda_l1,
  const double lambda_l2,
  double* cuda_sum_of_gradients,
  double* cuda_sum_of_hessians,
  const data_size_t num_data,
  const data_size_t* cuda_data_indices_in_leaf,
  hist_t* cuda_hist_in_leaf,
  CUDALeafSplitsStruct* cuda_struct) {
  if (threadIdx.x == 0) {
    const double sum_of_gradients = cuda_sum_of_gradients[0];
    const double sum_of_hessians = cuda_sum_of_hessians[0];
    cuda_struct->leaf_index = 0;
    cuda_struct->sum_of_gradients = sum_of_gradients;
    cuda_struct->sum_of_hessians = sum_of_hessians;
    cuda_struct->num_data_in_leaf = num_data;
    const bool use_l1 = lambda_l1 > 0.0f;
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    cuda_struct->data_indices_in_leaf = cuda_data_indices_in_leaf;
    cuda_struct->hist_in_leaf = cuda_hist_in_leaf;
  }
}

__global__ void CUDAInitSetValuesKernel(
  const double lambda_l1,
  const double lambda_l2,
  double* cuda_sum_of_gradients,
  double* cuda_sum_of_hessians,
  int64_t* cuda_sum_of_gradients_hessians,
  const data_size_t num_data,
  const data_size_t* cuda_data_indices_in_leaf,
  hist_t* cuda_hist_in_leaf,
  CUDALeafSplitsStruct* cuda_struct) {
  if (threadIdx.x == 0) {
    const double sum_of_gradients = cuda_sum_of_gradients[0];
    const double sum_of_hessians = cuda_sum_of_hessians[0];
    const int64_t sum_of_gradients_hessians = cuda_sum_of_gradients_hessians[0];
    cuda_struct->leaf_index = 0;
    cuda_struct->sum_of_gradients = sum_of_gradients;
    cuda_struct->sum_of_hessians = sum_of_hessians;
    cuda_struct->sum_of_gradients_hessians = sum_of_gradients_hessians;
    cuda_struct->num_data_in_leaf = num_data;
    const bool use_l1 = lambda_l1 > 0.0f;
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    cuda_struct->data_indices_in_leaf = cuda_data_indices_in_leaf;
    cuda_struct->hist_in_leaf = cuda_hist_in_leaf;
  }
}

void CUDALeafSplits::LaunchInitValuesEmptyKernel() {
  InitValuesEmptyKernel<<<1, 1>>>(cuda_struct_.RawData());
}

void CUDALeafSplits::LaunchInitValuesKernel(
  const double lambda_l1, const double lambda_l2,
  const data_size_t* cuda_bagging_data_indices,
  const data_size_t* cuda_data_indices_in_leaf,
  const data_size_t num_used_indices,
  hist_t* cuda_hist_in_leaf) {
  if (cuda_bagging_data_indices == nullptr) {
    CUDAInitValuesKernel1<false><<<num_blocks_init_from_gradients_, NUM_THREADS_PER_BLOCK_LEAF_SPLITS>>>(
      cuda_gradients_, cuda_hessians_, num_used_indices, nullptr, cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData());
  } else {
    CUDAInitValuesKernel1<true><<<num_blocks_init_from_gradients_, NUM_THREADS_PER_BLOCK_LEAF_SPLITS>>>(
      cuda_gradients_, cuda_hessians_, num_used_indices, cuda_bagging_data_indices, cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData());
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);

  if (nccl_communicator_ != nullptr) {
    ReduceGradKernel<<<1, NUM_THREADS_PER_BLOCK_LEAF_SPLITS>>>(num_blocks_init_from_gradients_, cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData(), num_used_indices);
    SynchronizeCUDADevice(__FILE__, __LINE__);
    cudaStream_t cuda_stream = CUDAStreamCreate();
    NCCLGroupStart();
    NCCLAllReduce<double>(cuda_sum_of_gradients_buffer_.RawData(), cuda_sum_of_gradients_buffer_.RawData(), 1, ncclFloat64, ncclSum, nccl_communicator_, cuda_stream);
    NCCLAllReduce<double>(cuda_sum_of_hessians_buffer_.RawData(), cuda_sum_of_hessians_buffer_.RawData(), 1, ncclFloat64, ncclSum, nccl_communicator_, cuda_stream);
    NCCLGroupEnd();
    SynchronizeCUDAStream(cuda_stream, __FILE__, __LINE__);
    CUDAStreamDestroy(cuda_stream);
    CUDAInitSetValuesKernel<<<1, 1>>>(lambda_l1, lambda_l2, cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData(), num_used_indices,
      cuda_data_indices_in_leaf, cuda_hist_in_leaf, cuda_struct_.RawData());
  } else {
    CUDAInitValuesKernel2<<<1, NUM_THREADS_PER_BLOCK_LEAF_SPLITS>>>(
      lambda_l1, lambda_l2,
      num_blocks_init_from_gradients_,
      cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData(),
      num_used_indices,
      cuda_data_indices_in_leaf,
      cuda_hist_in_leaf,
      cuda_struct_.RawData());
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDALeafSplits::LaunchInitValuesKernel(
  const double lambda_l1, const double lambda_l2,
  const data_size_t* cuda_bagging_data_indices,
  const data_size_t* cuda_data_indices_in_leaf,
  const data_size_t num_used_indices,
  hist_t* cuda_hist_in_leaf,
  const score_t* grad_scale,
  const score_t* hess_scale) {
  if (cuda_bagging_data_indices == nullptr) {
    CUDAInitValuesKernel3<false><<<num_blocks_init_from_gradients_, NUM_THREADS_PER_BLOCK_LEAF_SPLITS>>>(
      reinterpret_cast<const int16_t*>(cuda_gradients_), num_used_indices, nullptr, cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData(), cuda_sum_of_gradients_hessians_buffer_.RawData(), grad_scale, hess_scale);
  } else {
    CUDAInitValuesKernel3<true><<<num_blocks_init_from_gradients_, NUM_THREADS_PER_BLOCK_LEAF_SPLITS>>>(
      reinterpret_cast<const int16_t*>(cuda_gradients_), num_used_indices, cuda_bagging_data_indices, cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData(), cuda_sum_of_gradients_hessians_buffer_.RawData(), grad_scale, hess_scale);
  }

  SynchronizeCUDADevice(__FILE__, __LINE__);

  if (nccl_communicator_ != nullptr) {
    ReduceGradKernel<<<1, NUM_THREADS_PER_BLOCK_LEAF_SPLITS>>>(num_blocks_init_from_gradients_,
      cuda_sum_of_gradients_buffer_.RawData(), cuda_sum_of_hessians_buffer_.RawData(), cuda_sum_of_gradients_hessians_buffer_.RawData(),
      num_used_indices);
    SynchronizeCUDADevice(__FILE__, __LINE__);

    cudaStream_t cuda_stream = CUDAStreamCreate();
    NCCLGroupStart();
    NCCLAllReduce<double>(cuda_sum_of_gradients_buffer_.RawData(), cuda_sum_of_gradients_buffer_.RawData(), 1, ncclFloat64, ncclSum, nccl_communicator_, cuda_stream);
    NCCLAllReduce<double>(cuda_sum_of_hessians_buffer_.RawData(), cuda_sum_of_hessians_buffer_.RawData(), 1, ncclFloat64, ncclSum, nccl_communicator_, cuda_stream);
    NCCLAllReduce<int64_t>(cuda_sum_of_gradients_hessians_buffer_.RawData(), cuda_sum_of_gradients_hessians_buffer_.RawData(), 1, ncclInt64, ncclSum, nccl_communicator_, cuda_stream);
    NCCLGroupEnd();
    SynchronizeCUDAStream(cuda_stream, __FILE__, __LINE__);
    CUDAStreamDestroy(cuda_stream);
    CUDAInitSetValuesKernel<<<1, 1>>>(lambda_l1, lambda_l2, cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData(), cuda_sum_of_gradients_hessians_buffer_.RawData(), num_used_indices,
      cuda_data_indices_in_leaf, cuda_hist_in_leaf, cuda_struct_.RawData());
  } else {
    CUDAInitValuesKernel4<<<1, NUM_THREADS_PER_BLOCK_LEAF_SPLITS>>>(
      lambda_l1, lambda_l2,
      num_blocks_init_from_gradients_,
      cuda_sum_of_gradients_buffer_.RawData(),
      cuda_sum_of_hessians_buffer_.RawData(),
      cuda_sum_of_gradients_hessians_buffer_.RawData(),
      num_used_indices,
      cuda_data_indices_in_leaf,
      cuda_hist_in_leaf,
      cuda_struct_.RawData());
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
