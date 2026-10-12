/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_

#ifdef USE_CUDA

#ifndef USE_ROCM
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <stdio.h>

#include <LightGBM/bin.h>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/cuda/cuda_rocm_interop.h>
#include <LightGBM/utils/log.h>

#include <algorithm>

#define GLOBAL_PREFIX_SUM_BLOCK_SIZE (1024)
#define BITONIC_SORT_NUM_ELEMENTS (1024)
#define BITONIC_SORT_DEPTH (11)
#define BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE (10)

namespace LightGBM {

template <typename T>
__device__ __forceinline__ T ShufflePrefixSum(T value, T* shared_mem_buffer) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
    const T other_value = __shfl_up_sync(mask, value, offset);
    if (warpLane >= offset) {
      value += other_value;
    }
  }
  if (warpLane == warpSize - 1) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  if (warpID == 0) {
    T warp_sum = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
      const T other_warp_sum = __shfl_up_sync(mask, warp_sum, offset);
      if (warpLane >= offset) {
        warp_sum += other_warp_sum;
      }
    }
    shared_mem_buffer[warpLane] = warp_sum;
  }
  __syncthreads();
  const T warp_base = warpID == 0 ? 0 : shared_mem_buffer[warpID - 1];
  return warp_base + value;
}

template <typename T>
__device__ __forceinline__ T ShufflePrefixSumExclusive(T value, T* shared_mem_buffer) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
    const T other_value = __shfl_up_sync(mask, value, offset);
    if (warpLane >= offset) {
      value += other_value;
    }
  }
  if (warpLane == warpSize - 1) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  if (warpID == 0) {
    T warp_sum = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
      const T other_warp_sum = __shfl_up_sync(mask, warp_sum, offset);
      if (warpLane >= offset) {
        warp_sum += other_warp_sum;
      }
    }
    shared_mem_buffer[warpLane] = warp_sum;
  }
  __syncthreads();
  const T warp_base = warpID == 0 ? 0 : shared_mem_buffer[warpID - 1];
  const T inclusive_result = warp_base + value;
  if (threadIdx.x % warpSize == warpSize - 1) {
    shared_mem_buffer[warpLane] = inclusive_result;
  }
  __syncthreads();
  T exclusive_result = __shfl_up_sync(mask, inclusive_result, 1);
  if (threadIdx.x == 0) {
    exclusive_result = 0;
  } else if (threadIdx.x % warpSize == 0) {
    exclusive_result = shared_mem_buffer[warpLane - 1];
  }
  return exclusive_result;
}

template <typename T>
void ShufflePrefixSumGlobal(T* values, size_t len, T* block_prefix_sum_buffer);

template <typename VAL_T, typename REDUCE_T, typename INDEX_T>
void GlobalInclusiveArgPrefixSum(const INDEX_T* sorted_indices, const VAL_T* in_values, REDUCE_T* out_values, REDUCE_T* block_buffer, size_t n);

template <typename T>
__device__ __forceinline__ T ShuffleReduceSumWarp(T value, const data_size_t len) {
  if (len > 0) {
    const uint32_t mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(mask, value, offset);
    }
  }
  return value;
}

// reduce values from an 1-dimensional block (block size must be no greater than 1024)
template <typename T>
__device__ __forceinline__ T ShuffleReduceSum(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceSumWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    value = ShuffleReduceSumWarp<T>(value, num_warp);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceMaxWarp(T value, const data_size_t len) {
  if (len > 0) {
    const uint32_t mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value = max(value, __shfl_down_sync(mask, value, offset));
    }
  }
  return value;
}

// reduce values from an 1-dimensional block (block size must be no greater than 1024)
template <typename T>
__device__ __forceinline__ T ShuffleReduceMax(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceMaxWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    value = ShuffleReduceMaxWarp<T>(value, num_warp);
  }
  return value;
}

// calculate prefix sum values within an 1-dimensional block in global memory, exclusively
template <typename T>
__device__ __forceinline__ void GlobalMemoryPrefixSum(T* array, const size_t len) {
  const size_t num_values_per_thread = (len + blockDim.x - 1) / blockDim.x;
  const size_t start = threadIdx.x * num_values_per_thread;
  const size_t end = min(start + num_values_per_thread, len);
  T thread_sum = 0;
  for (size_t index = start; index < end; ++index) {
    thread_sum += array[index];
  }
  __shared__ T shared_mem[WARPSIZE];
  const T thread_base = ShufflePrefixSumExclusive<T>(thread_sum, shared_mem);
  if (start < end) {
    array[start] += thread_base;
  }
  for (size_t index = start + 1; index < end; ++index) {
    array[index] += array[index - 1];
  }
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceMinWarp(T value, const data_size_t len) {
  if (len > 0) {
    const uint32_t mask = (0xffffffff >> (warpSize - len));
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      const T other_value = __shfl_down_sync(mask, value, offset);
      value = (other_value < value) ? other_value : value;
    }
  }
  return value;
}

// reduce values from an 1-dimensional block (block size must be no greater than 1024)
template <typename T>
__device__ __forceinline__ T ShuffleReduceMin(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceMinWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : shared_mem_buffer[0]);
    value = ShuffleReduceMinWarp<T>(value, num_warp);
  }
  return value;
}

template <typename VAL_T, typename REDUCE_T>
void ShuffleReduceMinGlobal(const VAL_T* values, size_t n, REDUCE_T* block_buffer);

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__device__ __forceinline__ void BitonicArgSort_1024(const VAL_T* scores, INDEX_T* indices, const INDEX_T num_items) {
  INDEX_T depth = 1;
  INDEX_T num_items_aligend = 1;
  INDEX_T num_items_ref = num_items - 1;
  while (num_items_ref > 0) {
    num_items_ref >>= 1;
    num_items_aligend <<= 1;
    ++depth;
  }
  for (INDEX_T outer_depth = depth - 1; outer_depth >= 1; --outer_depth) {
    const INDEX_T outer_segment_length = 1 << (depth - outer_depth);
    const INDEX_T outer_segment_index = threadIdx.x / outer_segment_length;
    const bool ascending = ASCENDING ? (outer_segment_index % 2 == 0) : (outer_segment_index % 2 > 0);
    for (INDEX_T inner_depth = outer_depth; inner_depth < depth; ++inner_depth) {
      const INDEX_T segment_length = 1 << (depth - inner_depth);
      const INDEX_T half_segment_length = segment_length >> 1;
      const INDEX_T half_segment_index = threadIdx.x / half_segment_length;
      if (threadIdx.x < num_items_aligend) {
        if (half_segment_index % 2 == 0) {
          const INDEX_T index_to_compare = threadIdx.x + half_segment_length;
          if ((scores[indices[threadIdx.x]] > scores[indices[index_to_compare]]) == ascending) {
            const INDEX_T index = indices[threadIdx.x];
            indices[threadIdx.x] = indices[index_to_compare];
            indices[index_to_compare] = index;
          }
        }
      }
      __syncthreads();
    }
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__device__ __forceinline__ void BitonicArgSort_2048(const VAL_T* scores, INDEX_T* indices) {
  for (INDEX_T base = 0; base < 2048; base += 1024) {
    for (INDEX_T outer_depth = 10; outer_depth >= 1; --outer_depth) {
      const INDEX_T outer_segment_length = 1 << (11 - outer_depth);
      const INDEX_T outer_segment_index = threadIdx.x / outer_segment_length;
      const bool ascending = ((base == 0) ^ ASCENDING) ? (outer_segment_index % 2 > 0) : (outer_segment_index % 2 == 0);
      for (INDEX_T inner_depth = outer_depth; inner_depth < 11; ++inner_depth) {
        const INDEX_T segment_length = 1 << (11 - inner_depth);
        const INDEX_T half_segment_length = segment_length >> 1;
        const INDEX_T half_segment_index = threadIdx.x / half_segment_length;
        if (half_segment_index % 2 == 0) {
          const INDEX_T index_to_compare = threadIdx.x + half_segment_length + base;
          if ((scores[indices[threadIdx.x + base]] > scores[indices[index_to_compare]]) == ascending) {
            const INDEX_T index = indices[threadIdx.x + base];
            indices[threadIdx.x + base] = indices[index_to_compare];
            indices[index_to_compare] = index;
          }
        }
        __syncthreads();
      }
    }
  }
  const unsigned int index_to_compare = threadIdx.x + 1024;
  if (scores[indices[index_to_compare]] > scores[indices[threadIdx.x]]) {
    const INDEX_T temp_index = indices[index_to_compare];
    indices[index_to_compare] = indices[threadIdx.x];
    indices[threadIdx.x] = temp_index;
  }
  __syncthreads();
  for (INDEX_T base = 0; base < 2048; base += 1024) {
    for (INDEX_T inner_depth = 1; inner_depth < 11; ++inner_depth) {
      const INDEX_T segment_length = 1 << (11 - inner_depth);
      const INDEX_T half_segment_length = segment_length >> 1;
      const INDEX_T half_segment_index = threadIdx.x / half_segment_length;
      if (half_segment_index % 2 == 0) {
        const INDEX_T index_to_compare = threadIdx.x + half_segment_length + base;
        if (scores[indices[threadIdx.x + base]] < scores[indices[index_to_compare]]) {
          const INDEX_T index = indices[threadIdx.x + base];
          indices[threadIdx.x + base] = indices[index_to_compare];
          indices[index_to_compare] = index;
        }
      }
      __syncthreads();
    }
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING, uint32_t BLOCK_DIM, uint32_t MAX_DEPTH>
__device__ void BitonicArgSortDevice(const VAL_T* values, INDEX_T* indices, const int len) {
  __shared__ VAL_T shared_values[BLOCK_DIM];
  __shared__ INDEX_T shared_indices[BLOCK_DIM];
  int len_to_shift = len - 1;
  int max_depth = 1;
  while (len_to_shift > 0) {
    len_to_shift >>= 1;
    ++max_depth;
  }
  const int num_blocks = (len + static_cast<int>(BLOCK_DIM) - 1) / static_cast<int>(BLOCK_DIM);
  for (int block_index = 0; block_index < num_blocks; ++block_index) {
    const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
    if (this_index < len) {
      shared_values[threadIdx.x] = values[this_index];
      shared_indices[threadIdx.x] = this_index;
    } else {
      shared_indices[threadIdx.x] = len;
    }
    __syncthreads();
    for (int depth = max_depth - 1; depth > max_depth - static_cast<int>(MAX_DEPTH); --depth) {
      const int segment_length = (1 << (max_depth - depth));
      const int segment_index = this_index / segment_length;
      const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
      {
        const int half_segment_length = (segment_length >> 1);
        const int half_segment_index = this_index / half_segment_length;
        const int num_total_segment = (len + segment_length - 1) / segment_length;
        const int offset = (segment_index == num_total_segment - 1 && ascending == ASCENDING) ?
          (num_total_segment * segment_length - len) : 0;
        if (half_segment_index % 2 == 0) {
          const int segment_start = segment_index * segment_length;
          if (this_index >= offset + segment_start) {
            const int other_index = static_cast<int>(threadIdx.x) + half_segment_length - offset;
            const INDEX_T this_data_index = shared_indices[threadIdx.x];
            const INDEX_T other_data_index = shared_indices[other_index];
            const VAL_T this_value = shared_values[threadIdx.x];
            const VAL_T other_value = shared_values[other_index];
            if (other_data_index < len && (this_value > other_value) == ascending) {
              shared_indices[threadIdx.x] = other_data_index;
              shared_indices[other_index] = this_data_index;
              shared_values[threadIdx.x] = other_value;
              shared_values[other_index] = this_value;
            }
          }
        }
        __syncthreads();
      }
      for (int inner_depth = depth + 1; inner_depth < max_depth; ++inner_depth) {
        const int half_segment_length = (1 << (max_depth - inner_depth - 1));
        const int half_segment_index = this_index / half_segment_length;
        if (half_segment_index % 2 == 0) {
          const int other_index = static_cast<int>(threadIdx.x) + half_segment_length;
          const INDEX_T this_data_index = shared_indices[threadIdx.x];
          const INDEX_T other_data_index = shared_indices[other_index];
          const VAL_T this_value = shared_values[threadIdx.x];
          const VAL_T other_value = shared_values[other_index];
          if (other_data_index < len && (this_value > other_value) == ascending) {
            shared_indices[threadIdx.x] = other_data_index;
            shared_indices[other_index] = this_data_index;
            shared_values[threadIdx.x] = other_value;
            shared_values[other_index] = this_value;
          }
        }
        __syncthreads();
      }
    }
    if (this_index < len) {
      indices[this_index] = shared_indices[threadIdx.x];
    }
    __syncthreads();
  }
  for (int depth = max_depth - static_cast<int>(MAX_DEPTH); depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    {
      const int num_total_segment = (len + segment_length - 1) / segment_length;
      const int half_segment_length = (segment_length >> 1);
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
        const int segment_index = this_index / segment_length;
        const int half_segment_index = this_index / half_segment_length;
        const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
        const int offset = (segment_index == num_total_segment - 1 && ascending == ASCENDING) ?
          (num_total_segment * segment_length - len) : 0;
        if (half_segment_index % 2 == 0) {
          const int segment_start = segment_index * segment_length;
          if (this_index >= offset + segment_start) {
            const int other_index = this_index + half_segment_length - offset;
            if (other_index < len) {
              const INDEX_T this_data_index = indices[this_index];
              const INDEX_T other_data_index = indices[other_index];
              const VAL_T this_value = values[this_data_index];
              const VAL_T other_value = values[other_data_index];
              if ((this_value > other_value) == ascending) {
                indices[this_index] = other_data_index;
                indices[other_index] = this_data_index;
              }
            }
          }
        }
      }
      __syncthreads();
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - static_cast<int>(MAX_DEPTH); ++inner_depth) {
      const int half_segment_length = (1 << (max_depth - inner_depth - 1));
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
        const int segment_index = this_index / segment_length;
        const int half_segment_index = this_index / half_segment_length;
        const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
        if (half_segment_index % 2 == 0) {
          const int other_index = this_index + half_segment_length;
          if (other_index < len) {
            const INDEX_T this_data_index = indices[this_index];
            const INDEX_T other_data_index = indices[other_index];
            const VAL_T this_value = values[this_data_index];
            const VAL_T other_value = values[other_data_index];
            if ((this_value > other_value) == ascending) {
              indices[this_index] = other_data_index;
              indices[other_index] = this_data_index;
            }
          }
        }
        __syncthreads();
      }
    }
    for (int block_index = 0; block_index < num_blocks; ++block_index) {
      const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
      const int segment_index = this_index / segment_length;
      const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
      if (this_index < len) {
        const INDEX_T index = indices[this_index];
        shared_values[threadIdx.x] = values[index];
        shared_indices[threadIdx.x] = index;
      } else {
        shared_indices[threadIdx.x] = len;
      }
      __syncthreads();
      for (int inner_depth = max_depth - static_cast<int>(MAX_DEPTH) + 1; inner_depth < max_depth; ++inner_depth) {
        const int half_segment_length = (1 << (max_depth - inner_depth - 1));
        const int half_segment_index = this_index / half_segment_length;
        if (half_segment_index % 2 == 0) {
          const int other_index = static_cast<int>(threadIdx.x) + half_segment_length;
          const INDEX_T this_data_index = shared_indices[threadIdx.x];
          const INDEX_T other_data_index = shared_indices[other_index];
          const VAL_T this_value = shared_values[threadIdx.x];
          const VAL_T other_value = shared_values[other_index];
          if (other_data_index < len && (this_value > other_value) == ascending) {
            shared_indices[threadIdx.x] = other_data_index;
            shared_indices[other_index] = this_data_index;
            shared_values[threadIdx.x] = other_value;
            shared_values[other_index] = this_value;
          }
        }
        __syncthreads();
      }
      if (this_index < len) {
        indices[this_index] = shared_indices[threadIdx.x];
      }
      __syncthreads();
    }
  }
}

void BitonicArgSortItemsGlobal(
  const double* scores,
  const int num_queries,
  const data_size_t* cuda_query_boundaries,
  data_size_t* out_indices);

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
void BitonicArgSortGlobal(const VAL_T* values, INDEX_T* indices, const size_t len);

template <typename VAL_T, typename REDUCE_T>
void ShuffleReduceSumGlobal(const VAL_T* values, size_t n, REDUCE_T* block_buffer);

template <typename VAL_T, typename REDUCE_T>
void ShuffleReduceDotProdGlobal(const VAL_T* values1, const VAL_T* values2, size_t n, REDUCE_T* block_buffer);

template <typename VAL_T, typename REDUCE_VAL_T, typename INDEX_T>
__device__ void ShuffleSortedPrefixSumDevice(const VAL_T* in_values,
                                const INDEX_T* sorted_indices,
                                REDUCE_VAL_T* out_values,
                                const INDEX_T num_data) {
  __shared__ REDUCE_VAL_T shared_buffer[WARPSIZE];
  const INDEX_T num_data_per_thread = (num_data + static_cast<INDEX_T>(blockDim.x) - 1) / static_cast<INDEX_T>(blockDim.x);
  const INDEX_T start = num_data_per_thread * static_cast<INDEX_T>(threadIdx.x);
  const INDEX_T end = min(start + num_data_per_thread, num_data);
  REDUCE_VAL_T thread_sum = 0;
  for (INDEX_T index = start; index < end; ++index) {
    thread_sum += static_cast<REDUCE_VAL_T>(in_values[sorted_indices[index]]);
  }
  __syncthreads();
  thread_sum = ShufflePrefixSumExclusive<REDUCE_VAL_T>(thread_sum, shared_buffer);
  const REDUCE_VAL_T thread_base = shared_buffer[threadIdx.x];
  for (INDEX_T index = start; index < end; ++index) {
    out_values[index] = thread_base + static_cast<REDUCE_VAL_T>(in_values[sorted_indices[index]]);
  }
  __syncthreads();
}

template <typename VAL_T, typename INDEX_T, typename WEIGHT_T, typename WEIGHT_REDUCE_T, bool ASCENDING, bool USE_WEIGHT>
__global__ void PercentileGlobalKernel(const VAL_T* values,
                                       const WEIGHT_T* weights,
                                       const INDEX_T* sorted_indices,
                                       const WEIGHT_REDUCE_T* weights_prefix_sum,
                                       const double alpha,
                                       const INDEX_T len,
                                       VAL_T* out_value) {
  if (!USE_WEIGHT) {
    const double float_pos = (1.0f - alpha) * len;
    const INDEX_T pos = static_cast<INDEX_T>(float_pos);
    if (pos < 1) {
      *out_value = values[sorted_indices[0]];
    } else if (pos >= len) {
      *out_value = values[sorted_indices[len - 1]];
    } else {
      const double bias = float_pos - static_cast<double>(pos);
      const VAL_T v1 = values[sorted_indices[pos - 1]];
      const VAL_T v2 = values[sorted_indices[pos]];
      *out_value = static_cast<VAL_T>(v1 - (v1 - v2) * bias);
    }
  } else {
    const WEIGHT_REDUCE_T threshold = weights_prefix_sum[len - 1] * (1.0f - alpha);
    __shared__ INDEX_T pos;
    if (threadIdx.x == 0) {
      pos = len;
    }
    __syncthreads();
    for (INDEX_T index = static_cast<INDEX_T>(threadIdx.x); index < len; index += static_cast<INDEX_T>(blockDim.x)) {
      if (weights_prefix_sum[index] > threshold && (index == 0 || weights_prefix_sum[index - 1] <= threshold)) {
        pos = index;
      }
    }
    __syncthreads();
    pos = min(pos, len - 1);
    if (pos == 0 || pos == len - 1) {
      *out_value = values[pos];
    }
    const VAL_T v1 = values[sorted_indices[pos - 1]];
    const VAL_T v2 = values[sorted_indices[pos]];
    *out_value = static_cast<VAL_T>(v1 - (v1 - v2) * (threshold - weights_prefix_sum[pos - 1]) / (weights_prefix_sum[pos] - weights_prefix_sum[pos - 1]));
  }
}

template <typename VAL_T, typename INDEX_T, typename WEIGHT_T, typename WEIGHT_REDUCE_T, bool ASCENDING, bool USE_WEIGHT>
void PercentileGlobal(const VAL_T* values,
                      const WEIGHT_T* weights,
                      INDEX_T* indices,
                      WEIGHT_REDUCE_T* weights_prefix_sum,
                      WEIGHT_REDUCE_T* weights_prefix_sum_buffer,
                      const double alpha,
                      const INDEX_T len,
                      VAL_T* cuda_out_value) {
  if (len <= 1) {
    CopyFromCUDADeviceToCUDADevice<VAL_T>(cuda_out_value, values, 1, __FILE__, __LINE__);
  }
  BitonicArgSortGlobal<VAL_T, INDEX_T, ASCENDING>(values, indices, len);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (USE_WEIGHT) {
    GlobalInclusiveArgPrefixSum<WEIGHT_T, WEIGHT_REDUCE_T, INDEX_T>(indices, weights, weights_prefix_sum, weights_prefix_sum_buffer, static_cast<size_t>(len));
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  PercentileGlobalKernel<VAL_T, INDEX_T, WEIGHT_T, WEIGHT_REDUCE_T, ASCENDING, USE_WEIGHT><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, weights, indices, weights_prefix_sum, alpha, len, cuda_out_value);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

template <typename VAL_T, typename INDEX_T, typename WEIGHT_T, typename REDUCE_WEIGHT_T, bool ASCENDING, bool USE_WEIGHT>
__device__ VAL_T PercentileDevice(const VAL_T* values,
                                  const WEIGHT_T* weights,
                                  INDEX_T* indices,
                                  REDUCE_WEIGHT_T* weights_prefix_sum,
                                  const double alpha,
                                  const INDEX_T len) {
  if (len <= 1) {
    return values[0];
  }
  if (!USE_WEIGHT) {
    BitonicArgSortDevice<VAL_T, INDEX_T, ASCENDING, BITONIC_SORT_NUM_ELEMENTS / 2, 10>(values, indices, len);
    const double float_pos = (1.0f - alpha) * len;
    const INDEX_T pos = static_cast<INDEX_T>(float_pos);
    if (pos < 1) {
      return values[indices[0]];
    } else if (pos >= len) {
      return values[indices[len - 1]];
    } else {
      const double bias = float_pos - pos;
      const VAL_T v1 = values[indices[pos - 1]];
      const VAL_T v2 = values[indices[pos]];
      return static_cast<VAL_T>(v1 - (v1 - v2) * bias);
    }
  } else {
    BitonicArgSortDevice<VAL_T, INDEX_T, ASCENDING, BITONIC_SORT_NUM_ELEMENTS / 4, 9>(values, indices, len);
    ShuffleSortedPrefixSumDevice<WEIGHT_T, REDUCE_WEIGHT_T, INDEX_T>(weights, indices, weights_prefix_sum, len);
    const REDUCE_WEIGHT_T threshold = weights_prefix_sum[len - 1] * (1.0f - alpha);
    __shared__ INDEX_T pos;
    if (threadIdx.x == 0) {
      pos = len;
    }
    __syncthreads();
    for (INDEX_T index = static_cast<INDEX_T>(threadIdx.x); index < len; index += static_cast<INDEX_T>(blockDim.x)) {
      if (weights_prefix_sum[index] > threshold && (index == 0 || weights_prefix_sum[index - 1] <= threshold)) {
        pos = index;
      }
    }
    __syncthreads();
    pos = min(pos, len - 1);
    if (pos == 0 || pos == len - 1) {
      return values[pos];
    }
    const VAL_T v1 = values[indices[pos - 1]];
    const VAL_T v2 = values[indices[pos]];
    return static_cast<VAL_T>(v1 - (v1 - v2) * (threshold - weights_prefix_sum[pos - 1]) / (weights_prefix_sum[pos] - weights_prefix_sum[pos - 1]));
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
