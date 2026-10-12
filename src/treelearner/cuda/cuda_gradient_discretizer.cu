/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <algorithm>

#include <LightGBM/cuda/cuda_algorithms.hpp>

#include "cuda_gradient_discretizer.hpp"

namespace LightGBM {

__global__ void ReduceMinMaxKernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  score_t* grad_min_block_buffer,
  score_t* grad_max_block_buffer,
  score_t* hess_min_block_buffer,
  score_t* hess_max_block_buffer) {
  __shared__ score_t shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  score_t grad_max_val = kMinScore;
  score_t grad_min_val = kMaxScore;
  score_t hess_max_val = kMinScore;
  score_t hess_min_val = kMaxScore;
  if (index < num_data) {
    grad_max_val = input_gradients[index];
    grad_min_val = input_gradients[index];
    hess_max_val = input_hessians[index];
    hess_min_val = input_hessians[index];
  }
  grad_min_val = ShuffleReduceMin<score_t>(grad_min_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  grad_max_val = ShuffleReduceMax<score_t>(grad_max_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_min_val = ShuffleReduceMin<score_t>(hess_min_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_max_val = ShuffleReduceMax<score_t>(hess_max_val, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    grad_min_block_buffer[blockIdx.x] = grad_min_val;
    grad_max_block_buffer[blockIdx.x] = grad_max_val;
    hess_min_block_buffer[blockIdx.x] = hess_min_val;
    hess_max_block_buffer[blockIdx.x] = hess_max_val;
  }
}

__global__ void ReduceBlockMinMaxKernel(
  const int num_blocks,
  const int grad_discretize_bins,
  score_t* grad_min_block_buffer,
  score_t* grad_max_block_buffer,
  score_t* hess_min_block_buffer,
  score_t* hess_max_block_buffer) {
  __shared__ score_t shared_mem_buffer[WARPSIZE];
  score_t grad_max_val = kMinScore;
  score_t grad_min_val = kMaxScore;
  score_t hess_max_val = kMinScore;
  score_t hess_min_val = kMaxScore;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks; block_index += static_cast<int>(blockDim.x)) {
    grad_min_val = min(grad_min_val, grad_min_block_buffer[block_index]);
    grad_max_val = max(grad_max_val, grad_max_block_buffer[block_index]);
    hess_min_val = min(hess_min_val, hess_min_block_buffer[block_index]);
    hess_max_val = max(hess_max_val, hess_max_block_buffer[block_index]);
  }
  grad_min_val = ShuffleReduceMin<score_t>(grad_min_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  grad_max_val = ShuffleReduceMax<score_t>(grad_max_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_max_val = ShuffleReduceMax<score_t>(hess_max_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_max_val = ShuffleReduceMax<score_t>(hess_max_val, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    const score_t grad_abs_max = max(fabs(grad_min_val), fabs(grad_max_val));
    const score_t hess_abs_max = max(fabs(hess_min_val), fabs(hess_max_val));
    grad_min_block_buffer[0] = 1.0f / (grad_abs_max / (grad_discretize_bins / 2));
    grad_max_block_buffer[0] = (grad_abs_max / (grad_discretize_bins / 2));
    hess_min_block_buffer[0] = 1.0f / (hess_abs_max / (grad_discretize_bins));
    hess_max_block_buffer[0] = (hess_abs_max / (grad_discretize_bins));
  }
}

template <bool STOCHASTIC_ROUNDING>
__global__ void DiscretizeGradientsKernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  const score_t* grad_scale_ptr,
  const score_t* hess_scale_ptr,
  const int iter,
  const int* random_values_use_start,
  const score_t* gradient_random_values,
  const score_t* hessian_random_values,
  const int grad_discretize_bins,
  int8_t* output_gradients_and_hessians) {
  const int start = random_values_use_start[iter];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const score_t grad_scale = *grad_scale_ptr;
  const score_t hess_scale = *hess_scale_ptr;
  int16_t* output_gradients_and_hessians_ptr = reinterpret_cast<int16_t*>(output_gradients_and_hessians);
  if (index < num_data) {
    if (STOCHASTIC_ROUNDING) {
      const data_size_t index_offset = (index + start) % num_data;
      const score_t gradient = input_gradients[index];
      const score_t hessian = input_hessians[index];
      const score_t gradient_random_value = gradient_random_values[index_offset];
      const score_t hessian_random_value = hessian_random_values[index_offset];
      output_gradients_and_hessians_ptr[2 * index + 1] = gradient > 0.0f ?
        static_cast<int16_t>(gradient * grad_scale + gradient_random_value) :
        static_cast<int16_t>(gradient * grad_scale - gradient_random_value);
      output_gradients_and_hessians_ptr[2 * index] = static_cast<int16_t>(hessian * hess_scale + hessian_random_value);
    } else {
      const score_t gradient = input_gradients[index];
      const score_t hessian = input_hessians[index];
      output_gradients_and_hessians_ptr[2 * index + 1] = gradient > 0.0f ?
        static_cast<int16_t>(gradient * grad_scale + 0.5) :
        static_cast<int16_t>(gradient * grad_scale - 0.5);
      output_gradients_and_hessians_ptr[2 * index] = static_cast<int16_t>(hessian * hess_scale + 0.5);
    }
  }
}

void CUDAGradientDiscretizer::DiscretizeGradients(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians) {
  ReduceMinMaxKernel<<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(
    num_data, input_gradients, input_hessians,
    grad_min_block_buffer_.RawData(),
    grad_max_block_buffer_.RawData(),
    hess_min_block_buffer_.RawData(),
    hess_max_block_buffer_.RawData());
    SynchronizeCUDADevice(__FILE__, __LINE__);
  ReduceBlockMinMaxKernel<<<1, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(
    num_reduce_blocks_,
    num_grad_quant_bins_,
    grad_min_block_buffer_.RawData(),
    grad_max_block_buffer_.RawData(),
    hess_min_block_buffer_.RawData(),
    hess_max_block_buffer_.RawData());
    SynchronizeCUDADevice(__FILE__, __LINE__);

  if (nccl_communicator_ != nullptr) {
    SynchronizeCUDADevice(__FILE__, __LINE__);
    cudaStream_t cuda_stream = CUDAStreamCreate();
    NCCLGroupStart();
    NCCLAllReduce<score_t>(grad_min_block_buffer_.RawDataReadOnly(), grad_min_block_buffer_.RawData(), 1, ncclFloat32, ncclMin, nccl_communicator_, cuda_stream);
    NCCLAllReduce<score_t>(hess_min_block_buffer_.RawDataReadOnly(), hess_min_block_buffer_.RawData(), 1, ncclFloat32, ncclMin, nccl_communicator_, cuda_stream);
    NCCLAllReduce<score_t>(grad_max_block_buffer_.RawDataReadOnly(), grad_max_block_buffer_.RawData(), 1, ncclFloat32, ncclMax, nccl_communicator_, cuda_stream);
    NCCLAllReduce<score_t>(hess_max_block_buffer_.RawDataReadOnly(), hess_max_block_buffer_.RawData(), 1, ncclFloat32, ncclMax, nccl_communicator_, cuda_stream);
    NCCLGroupEnd();
    SynchronizeCUDAStream(cuda_stream, __FILE__, __LINE__);
    CUDAStreamDestroy(cuda_stream);
  }

  #define DiscretizeGradientsKernel_ARGS \
    num_data, \
    input_gradients, \
    input_hessians, \
    grad_min_block_buffer_.RawData(), \
    hess_min_block_buffer_.RawData(), \
    iter_, \
    random_values_use_start_.RawData(), \
    gradient_random_values_.RawData(), \
    hessian_random_values_.RawData(), \
    num_grad_quant_bins_, \
    discretized_gradients_and_hessians_.RawData()

  if (stochastic_rounding_) {
    DiscretizeGradientsKernel<true><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
  } else {
    DiscretizeGradientsKernel<false><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  ++iter_;
}

}  // namespace LightGBM

#endif  // USE_CUDA
