/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_single_gpu_tree_learner.hpp"

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_rocm_interop.h>

#include <algorithm>

namespace LightGBM {

__global__ void ReduceLeafStatKernel_SharedMemory(
  const score_t* gradients,
  const score_t* hessians,
  const int num_leaves,
  const data_size_t num_data,
  const int* data_index_to_leaf_index,
  double* leaf_grad_stat_buffer,
  double* leaf_hess_stat_buffer) {
  extern __shared__ double shared_mem[];
  double* shared_grad_sum = shared_mem;
  double* shared_hess_sum = shared_mem + num_leaves;
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    shared_grad_sum[leaf_index] = 0.0f;
    shared_hess_sum[leaf_index] = 0.0f;
  }
  __syncthreads();
  if (data_index < num_data) {
    const int leaf_index = data_index_to_leaf_index[data_index];
    atomicAdd_block(shared_grad_sum + leaf_index, gradients[data_index]);
    atomicAdd_block(shared_hess_sum + leaf_index, hessians[data_index]);
  }
  __syncthreads();
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    atomicAdd_system(leaf_grad_stat_buffer + leaf_index, shared_grad_sum[leaf_index]);
    atomicAdd_system(leaf_hess_stat_buffer + leaf_index, shared_hess_sum[leaf_index]);
  }
}

__global__ void ReduceLeafStatKernel_GlobalMemory(
  const score_t* gradients,
  const score_t* hessians,
  const int num_leaves,
  const data_size_t num_data,
  const int* data_index_to_leaf_index,
  double* leaf_grad_stat_buffer,
  double* leaf_hess_stat_buffer) {
  const size_t offset = static_cast<size_t>(num_leaves) * (blockIdx.x + 1);
  double* grad_sum = leaf_grad_stat_buffer + offset;
  double* hess_sum = leaf_hess_stat_buffer + offset;
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    grad_sum[leaf_index] = 0.0f;
    hess_sum[leaf_index] = 0.0f;
  }
  __syncthreads();
  if (data_index < num_data) {
    const int leaf_index = data_index_to_leaf_index[data_index];
    atomicAdd_block(grad_sum + leaf_index, gradients[data_index]);
    atomicAdd_block(hess_sum + leaf_index, hessians[data_index]);
  }
  __syncthreads();
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    atomicAdd_system(leaf_grad_stat_buffer + leaf_index, grad_sum[leaf_index]);
    atomicAdd_system(leaf_hess_stat_buffer + leaf_index, hess_sum[leaf_index]);
  }
}

template <bool USE_L1, bool USE_SMOOTHING>
__global__ void CalcRefitLeafOutputKernel(
  const int num_leaves,
  const double* leaf_grad_stat_buffer,
  const double* leaf_hess_stat_buffer,
  const data_size_t* num_data_in_leaf,
  const int* leaf_parent,
  const int* left_child,
  const int* right_child,
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const double shrinkage_rate,
  const double refit_decay_rate,
  double* leaf_value) {
  const int leaf_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (leaf_index < num_leaves) {
    const double sum_gradients = leaf_grad_stat_buffer[leaf_index];
    const double sum_hessians = leaf_hess_stat_buffer[leaf_index];
    const data_size_t num_data = num_data_in_leaf[leaf_index];
    const double old_leaf_value = leaf_value[leaf_index];
    double new_leaf_value = 0.0f;
    if (!USE_SMOOTHING) {
      new_leaf_value = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_gradients, sum_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      const int parent = leaf_parent[leaf_index];
      if (parent >= 0) {
        const int sibliing = left_child[parent] == leaf_index ? right_child[parent] : left_child[parent];
        const double sum_gradients_of_parent = sum_gradients + leaf_grad_stat_buffer[sibliing];
        const double sum_hessians_of_parent = sum_hessians + leaf_hess_stat_buffer[sibliing];
        const data_size_t num_data_in_parent = num_data + num_data_in_leaf[sibliing];
        const double parent_output =
          CUDALeafSplits::CalculateSplittedLeafOutput<false, true>(
            sum_gradients_of_parent, sum_hessians_of_parent, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
          new_leaf_value = CUDALeafSplits::CalculateSplittedLeafOutput<false, true>(
          sum_gradients, sum_hessians, lambda_l1, lambda_l2, path_smooth, num_data_in_parent, parent_output);
      } else {
        new_leaf_value = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_gradients, sum_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
      }
    }
    if (isnan(new_leaf_value)) {
      new_leaf_value = 0.0f;
    } else {
      new_leaf_value *= shrinkage_rate;
    }
    leaf_value[leaf_index] = refit_decay_rate * old_leaf_value + (1.0f - refit_decay_rate) * new_leaf_value;
  }
}

void CUDASingleGPUTreeLearner::LaunchReduceLeafStatKernel(
  const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf,
  const int* leaf_parent, const int* left_child, const int* right_child, const int num_leaves,
  const data_size_t num_data, double* cuda_leaf_value, const double shrinkage_rate) const {
  int num_block = (num_data + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  if (num_leaves <= 2048) {
    ReduceLeafStatKernel_SharedMemory<<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE, 2 * num_leaves * sizeof(double)>>>(
      gradients, hessians, num_leaves, num_data, cuda_data_partition_->cuda_data_index_to_leaf_index(),
      cuda_leaf_gradient_stat_buffer_.RawData(), cuda_leaf_hessian_stat_buffer_.RawData());
  } else {
    ReduceLeafStatKernel_GlobalMemory<<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(
      gradients, hessians, num_leaves, num_data, cuda_data_partition_->cuda_data_index_to_leaf_index(),
      cuda_leaf_gradient_stat_buffer_.RawData(), cuda_leaf_hessian_stat_buffer_.RawData());
  }
  const bool use_l1 = config_->lambda_l1 > 0.0f;
  const bool use_smoothing = config_->path_smooth > 0.0f;
  num_block = (num_leaves + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;

  #define CalcRefitLeafOutputKernel_ARGS \
    num_leaves, cuda_leaf_gradient_stat_buffer_.RawData(), cuda_leaf_hessian_stat_buffer_.RawData(), num_data_in_leaf, \
    leaf_parent, left_child, right_child, \
    config_->lambda_l1, config_->lambda_l2, config_->path_smooth, \
    shrinkage_rate, config_->refit_decay_rate, cuda_leaf_value

  if (!use_l1) {
    if (!use_smoothing) {
      CalcRefitLeafOutputKernel<false, false>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    } else {
      CalcRefitLeafOutputKernel<false, true>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    }
  } else {
    if (!use_smoothing) {
      CalcRefitLeafOutputKernel<true, false>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    } else {
      CalcRefitLeafOutputKernel<true, true>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    }
  }
  #undef CalcRefitLeafOutputKernel_ARGS
}

template <typename T, bool IS_INNER>
__global__ void CalcBitsetLenKernel(const CUDASplitInfo* best_split_info, size_t* out_len_buffer) {
  __shared__ size_t shared_mem_buffer[WARPSIZE];
  const T* vals = nullptr;
  if (IS_INNER) {
    vals = reinterpret_cast<const T*>(best_split_info->cat_threshold);
  } else {
    vals = reinterpret_cast<const T*>(best_split_info->cat_threshold_real);
  }
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  size_t len = 0;
  if (i < best_split_info->num_cat_threshold) {
    const T val = vals[i];
    len = (val / WARPSIZE) + 1;
  }
  const size_t block_max_len = ShuffleReduceMax<size_t>(len, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    out_len_buffer[blockIdx.x] = block_max_len;
  }
}

__global__ void ReduceBlockMaxLen(size_t* out_len_buffer, const int num_blocks) {
  __shared__ size_t shared_mem_buffer[WARPSIZE];
  size_t max_len = 0;
  for (int i = static_cast<int>(threadIdx.x); i < num_blocks; i += static_cast<int>(blockDim.x)) {
    max_len = max(out_len_buffer[i], max_len);
  }
  const size_t all_max_len = ShuffleReduceMax<size_t>(max_len, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    out_len_buffer[0] = max_len;
  }
}

template <typename T, bool IS_INNER>
__global__ void CUDAConstructBitsetKernel(const CUDASplitInfo* best_split_info, uint32_t* out, size_t cuda_bitset_len) {
  const T* vals = nullptr;
  if (IS_INNER) {
    vals = reinterpret_cast<const T*>(best_split_info->cat_threshold);
  } else {
    vals = reinterpret_cast<const T*>(best_split_info->cat_threshold_real);
  }
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i < best_split_info->num_cat_threshold) {
    const T val = vals[i];
    // can use add instead of or here, because each bit will only be added once
    atomicAdd_system(out + (val / WARPSIZE), (0x1 << (val % WARPSIZE)));
  }
}

__global__ void SetRealThresholdKernel(
  const CUDASplitInfo* best_split_info,
  const int* categorical_bin_to_value,
  const int* categorical_bin_offsets) {
  const int num_cat_threshold = best_split_info->num_cat_threshold;
  const int* categorical_bin_to_value_ptr = categorical_bin_to_value + categorical_bin_offsets[best_split_info->inner_feature_index];
  int* cat_threshold_real = best_split_info->cat_threshold_real;
  const uint32_t* cat_threshold = best_split_info->cat_threshold;
  const int index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (index < num_cat_threshold) {
    cat_threshold_real[index] = categorical_bin_to_value_ptr[cat_threshold[index]];
  }
}

template <typename T, bool IS_INNER>
void CUDAConstructBitset(const CUDASplitInfo* best_split_info, const int num_cat_threshold, uint32_t* out, size_t bitset_len) {
  const int num_blocks = (num_cat_threshold + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  // clear the bitset vector first
  SetCUDAMemory<uint32_t>(out, 0, bitset_len, __FILE__, __LINE__);
  CUDAConstructBitsetKernel<T, IS_INNER><<<num_blocks, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(best_split_info, out, bitset_len);
}

template <typename T, bool IS_INNER>
size_t CUDABitsetLen(const CUDASplitInfo* best_split_info, const int num_cat_threshold, size_t* out_len_buffer) {
  const int num_blocks = (num_cat_threshold + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  CalcBitsetLenKernel<T, IS_INNER><<<num_blocks, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(best_split_info, out_len_buffer);
  ReduceBlockMaxLen<<<1, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(out_len_buffer, num_blocks);
  size_t host_max_len = 0;
  CopyFromCUDADeviceToHost<size_t>(&host_max_len, out_len_buffer, 1, __FILE__, __LINE__);
  return host_max_len;
}

void CUDASingleGPUTreeLearner::LaunchConstructBitsetForCategoricalSplitKernel(
  const CUDASplitInfo* best_split_info) {
  const int num_blocks = (num_cat_threshold_ + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  SetRealThresholdKernel<<<num_blocks, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>
    (best_split_info, cuda_categorical_bin_to_value_, cuda_categorical_bin_offsets_);
  cuda_bitset_inner_len_ = CUDABitsetLen<uint32_t, true>(best_split_info, num_cat_threshold_, cuda_block_bitset_len_buffer_);
  CUDAConstructBitset<uint32_t, true>(best_split_info, num_cat_threshold_, cuda_bitset_inner_, cuda_bitset_inner_len_);
  cuda_bitset_len_ = CUDABitsetLen<int, false>(best_split_info, num_cat_threshold_, cuda_block_bitset_len_buffer_);
  CUDAConstructBitset<int, false>(best_split_info, num_cat_threshold_, cuda_bitset_, cuda_bitset_len_);
}

void CUDASingleGPUTreeLearner::LaunchCalcLeafValuesGivenGradStat(
  CUDATree* cuda_tree, const data_size_t* num_data_in_leaf) {
  #define CalcRefitLeafOutputKernel_ARGS \
    cuda_tree->num_leaves(), cuda_leaf_gradient_stat_buffer_.RawData(), cuda_leaf_hessian_stat_buffer_.RawData(), num_data_in_leaf, \
    cuda_tree->cuda_leaf_parent(), cuda_tree->cuda_left_child(), cuda_tree->cuda_right_child(), \
    config_->lambda_l1, config_->lambda_l2, config_->path_smooth, \
    1.0f, config_->refit_decay_rate, cuda_tree->cuda_leaf_value_ref()
  const bool use_l1 = config_->lambda_l1 > 0.0f;
  const bool use_smoothing = config_->path_smooth > 0.0f;
  const int num_block = (cuda_tree->num_leaves() + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  if (!use_l1) {
    if (!use_smoothing) {
      CalcRefitLeafOutputKernel<false, false>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    } else {
      CalcRefitLeafOutputKernel<false, true>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    }
  } else {
    if (!use_smoothing) {
      CalcRefitLeafOutputKernel<true, false>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    } else {
      CalcRefitLeafOutputKernel<true, true>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    }
  }

  #undef CalcRefitLeafOutputKernel_ARGS
}

}  // namespace LightGBM

#endif  // USE_CUDA
