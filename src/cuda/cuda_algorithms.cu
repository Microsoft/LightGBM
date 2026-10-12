/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_rocm_interop.h>

#include <algorithm>

namespace LightGBM {

template <typename T>
__global__ void ShufflePrefixSumGlobalKernel(T* values, size_t len, T* block_prefix_sum_buffer) {
  __shared__ T shared_mem_buffer[WARPSIZE];
  const size_t index = static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  T value = 0;
  if (index < len) {
    value = values[index];
  }
  const T prefix_sum_value = ShufflePrefixSum<T>(value, shared_mem_buffer);
  values[index] = prefix_sum_value;
  if (threadIdx.x == blockDim.x - 1) {
    block_prefix_sum_buffer[blockIdx.x] = prefix_sum_value;
  }
}

template <typename T>
__global__ void ShufflePrefixSumGlobalReduceBlockKernel(T* block_prefix_sum_buffer, int num_blocks) {
  __shared__ T shared_mem_buffer[WARPSIZE];
  const int num_blocks_per_thread = (num_blocks + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 2) / (GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1);
  int thread_block_start = threadIdx.x == 0 ? 0 : (threadIdx.x - 1) * num_blocks_per_thread;
  int thread_block_end = threadIdx.x == 0 ? 0 : min(thread_block_start + num_blocks_per_thread, num_blocks);
  T base = 0;
  for (int block_index = thread_block_start; block_index < thread_block_end; ++block_index) {
    base += block_prefix_sum_buffer[block_index];
  }
  base = ShufflePrefixSum<T>(base, shared_mem_buffer);
  thread_block_start = threadIdx.x == blockDim.x - 1 ? 0 : threadIdx.x * num_blocks_per_thread;
  thread_block_end = threadIdx.x == blockDim.x - 1 ? 0 : min(thread_block_start + num_blocks_per_thread, num_blocks);
  for (int block_index = thread_block_start + 1; block_index < thread_block_end; ++block_index) {
    block_prefix_sum_buffer[block_index] += block_prefix_sum_buffer[block_index - 1];
  }
  for (int block_index = thread_block_start; block_index < thread_block_end; ++block_index) {
    block_prefix_sum_buffer[block_index] += base;
  }
}

template <typename T>
__global__ void ShufflePrefixSumGlobalAddBase(size_t len, const T* block_prefix_sum_buffer, T* values) {
  const T base = blockIdx.x == 0 ? 0 : block_prefix_sum_buffer[blockIdx.x - 1];
  const size_t index = static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (index < len) {
    values[index] += base;
  }
}

template <typename T>
void ShufflePrefixSumGlobal(T* values, size_t len, T* block_prefix_sum_buffer) {
  const int num_blocks = (static_cast<int>(len) + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ShufflePrefixSumGlobalKernel<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, len, block_prefix_sum_buffer);
  ShufflePrefixSumGlobalReduceBlockKernel<<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_prefix_sum_buffer, num_blocks);
  ShufflePrefixSumGlobalAddBase<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(len, block_prefix_sum_buffer, values);
}

template void ShufflePrefixSumGlobal<uint16_t>(uint16_t* values, size_t len, uint16_t* block_prefix_sum_buffer);
template void ShufflePrefixSumGlobal<uint32_t>(uint32_t* values, size_t len, uint32_t* block_prefix_sum_buffer);
template void ShufflePrefixSumGlobal<uint64_t>(uint64_t* values, size_t len, uint64_t* block_prefix_sum_buffer);

__global__ void BitonicArgSortItemsGlobalKernel(const double* scores,
  const int num_queries,
  const data_size_t* cuda_query_boundaries,
  data_size_t* out_indices) {
  const int query_index_start = static_cast<int>(blockIdx.x) * BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE;
  const int query_index_end = min(query_index_start + BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE, num_queries);
  for (int query_index = query_index_start; query_index < query_index_end; ++query_index) {
    const data_size_t query_item_start = cuda_query_boundaries[query_index];
    const data_size_t query_item_end = cuda_query_boundaries[query_index + 1];
    const data_size_t num_items_in_query = query_item_end - query_item_start;
    BitonicArgSortDevice<double, data_size_t, false, BITONIC_SORT_NUM_ELEMENTS, 11>(scores + query_item_start,
          out_indices + query_item_start,
          num_items_in_query);
    __syncthreads();
  }
}

void BitonicArgSortItemsGlobal(
  const double* scores,
  const int num_queries,
  const data_size_t* cuda_query_boundaries,
  data_size_t* out_indices) {
  const int num_blocks = (num_queries + BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE - 1) / BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE;
  BitonicArgSortItemsGlobalKernel<<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
    scores, num_queries, cuda_query_boundaries, out_indices);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

template <typename T>
__global__ void BlockReduceSum(T* block_buffer, const data_size_t num_blocks) {
  __shared__ T shared_buffer[WARPSIZE];
  T thread_sum = 0;
  for (data_size_t block_index = static_cast<data_size_t>(threadIdx.x); block_index < num_blocks; block_index += static_cast<data_size_t>(blockDim.x)) {
    thread_sum += block_buffer[block_index];
  }
  thread_sum = ShuffleReduceSum<T>(thread_sum, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[0] = thread_sum;
  }
}

template <typename VAL_T, typename REDUCE_T>
__global__ void ShuffleReduceSumGlobalKernel(const VAL_T* values, const data_size_t num_value, REDUCE_T* block_buffer) {
  __shared__ REDUCE_T shared_buffer[WARPSIZE];
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const REDUCE_T value = (data_index < num_value ? static_cast<REDUCE_T>(values[data_index]) : 0.0f);
  const REDUCE_T reduce_value = ShuffleReduceSum<REDUCE_T>(value, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = reduce_value;
  }
}

template <typename VAL_T, typename REDUCE_T>
void ShuffleReduceSumGlobal(const VAL_T* values, size_t n, REDUCE_T* block_buffer) {
  const data_size_t num_value = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_value + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ShuffleReduceSumGlobalKernel<VAL_T, REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, num_value, block_buffer);
  BlockReduceSum<REDUCE_T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_buffer, num_blocks);
}

template void ShuffleReduceSumGlobal<label_t, double>(const label_t* values, size_t n, double* block_buffer);
template void ShuffleReduceSumGlobal<double, double>(const double* values, size_t n, double* block_buffer);

template <typename VAL_T, typename REDUCE_T>
__global__ void ShuffleReduceMinGlobalKernel(const VAL_T* values, const data_size_t num_value, REDUCE_T* block_buffer) {
  __shared__ REDUCE_T shared_buffer[WARPSIZE];
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const REDUCE_T value = (data_index < num_value ? static_cast<REDUCE_T>(values[data_index]) : 0.0f);
  const REDUCE_T reduce_value = ShuffleReduceMin<REDUCE_T>(value, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = reduce_value;
  }
}

template <typename T>
__global__ void ShuffleBlockReduceMin(T* block_buffer, const data_size_t num_blocks) {
  __shared__ T shared_buffer[WARPSIZE];
  T thread_min = 0;
  for (data_size_t block_index = static_cast<data_size_t>(threadIdx.x); block_index < num_blocks; block_index += static_cast<data_size_t>(blockDim.x)) {
    const T value = block_buffer[block_index];
    if (value < thread_min) {
      thread_min = value;
    }
  }
  thread_min = ShuffleReduceMin<T>(thread_min, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[0] = thread_min;
  }
}

template <typename VAL_T, typename REDUCE_T>
void ShuffleReduceMinGlobal(const VAL_T* values, size_t n, REDUCE_T* block_buffer) {
  const data_size_t num_value = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_value + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ShuffleReduceMinGlobalKernel<VAL_T, REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, num_value, block_buffer);
  ShuffleBlockReduceMin<REDUCE_T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_buffer, num_blocks);
}

template void ShuffleReduceMinGlobal<label_t, double>(const label_t* values, size_t n, double* block_buffer);

template <typename VAL_T, typename REDUCE_T>
__global__ void ShuffleReduceDotProdGlobalKernel(const VAL_T* values1, const VAL_T* values2, const data_size_t num_value, REDUCE_T* block_buffer) {
  __shared__ REDUCE_T shared_buffer[WARPSIZE];
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const REDUCE_T value1 = (data_index < num_value ? static_cast<REDUCE_T>(values1[data_index]) : 0.0f);
  const REDUCE_T value2 = (data_index < num_value ? static_cast<REDUCE_T>(values2[data_index]) : 0.0f);
  const REDUCE_T reduce_value = ShuffleReduceSum<REDUCE_T>(value1 * value2, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = reduce_value;
  }
}

template <typename VAL_T, typename REDUCE_T>
void ShuffleReduceDotProdGlobal(const VAL_T* values1, const VAL_T* values2, size_t n, REDUCE_T* block_buffer) {
  const data_size_t num_value = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_value + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ShuffleReduceDotProdGlobalKernel<VAL_T, REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values1, values2, num_value, block_buffer);
  BlockReduceSum<REDUCE_T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_buffer, num_blocks);
}

template void ShuffleReduceDotProdGlobal<label_t, double>(const label_t* values1, const label_t* values2, size_t n, double* block_buffer);

template <typename INDEX_T, typename VAL_T, typename REDUCE_T>
__global__ void GlobalInclusiveArgPrefixSumKernel(
  const INDEX_T* sorted_indices, const VAL_T* in_values, REDUCE_T* out_values, REDUCE_T* block_buffer, data_size_t num_data) {
  __shared__ REDUCE_T shared_buffer[WARPSIZE];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  REDUCE_T value = static_cast<REDUCE_T>(data_index < num_data ? in_values[sorted_indices[data_index]] : 0);
  __syncthreads();
  value = ShufflePrefixSum<REDUCE_T>(value, shared_buffer);
  if (data_index < num_data) {
    out_values[data_index] = value;
  }
  if (threadIdx.x == blockDim.x - 1) {
    block_buffer[blockIdx.x + 1] = value;
  }
}

template <typename T>
__global__ void GlobalInclusivePrefixSumReduceBlockKernel(T* block_buffer, data_size_t num_blocks) {
  __shared__ T shared_buffer[WARPSIZE];
  T thread_sum = 0;
  const data_size_t num_blocks_per_thread = (num_blocks + static_cast<data_size_t>(blockDim.x)) / static_cast<data_size_t>(blockDim.x);
  const data_size_t thread_start_block_index = static_cast<data_size_t>(threadIdx.x) * num_blocks_per_thread;
  const data_size_t thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread, num_blocks + 1);
  for (data_size_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    thread_sum += block_buffer[block_index];
  }
  ShufflePrefixSumExclusive<T>(thread_sum, shared_buffer);
  for (data_size_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    block_buffer[block_index] += thread_sum;
  }
}

template <typename T>
__global__ void GlobalInclusivePrefixSumAddBlockBaseKernel(const T* block_buffer, T* values, data_size_t num_data) {
  const T block_sum_base = block_buffer[blockIdx.x];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    values[data_index] += block_sum_base;
  }
}

template <typename VAL_T, typename REDUCE_T, typename INDEX_T>
void GlobalInclusiveArgPrefixSum(const INDEX_T* sorted_indices, const VAL_T* in_values, REDUCE_T* out_values, REDUCE_T* block_buffer, size_t n) {
  const data_size_t num_data = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GlobalInclusiveArgPrefixSumKernel<INDEX_T, VAL_T, REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    sorted_indices, in_values, out_values, block_buffer, num_data);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  GlobalInclusivePrefixSumReduceBlockKernel<REDUCE_T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_buffer, num_blocks);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  GlobalInclusivePrefixSumAddBlockBaseKernel<REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_buffer, out_values, num_data);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

template void GlobalInclusiveArgPrefixSum<label_t, double, data_size_t>(const data_size_t* sorted_indices, const label_t* in_values, double* out_values, double* block_buffer, size_t n);

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__global__ void BitonicArgSortGlobalKernel(const VAL_T* values, INDEX_T* indices, const int num_total_data) {
  const int thread_index = static_cast<int>(threadIdx.x);
  const int low = static_cast<int>(blockIdx.x * BITONIC_SORT_NUM_ELEMENTS);
  const bool outer_ascending = ASCENDING ? (blockIdx.x % 2 == 0) : (blockIdx.x % 2 == 1);
  const VAL_T* values_pointer = values + low;
  INDEX_T* indices_pointer = indices + low;
  const int num_data = min(BITONIC_SORT_NUM_ELEMENTS, num_total_data - low);
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  __shared__ INDEX_T shared_indices[BITONIC_SORT_NUM_ELEMENTS];
  if (thread_index < num_data) {
    shared_values[thread_index] = values_pointer[thread_index];
    shared_indices[thread_index] = static_cast<INDEX_T>(thread_index + blockIdx.x * blockDim.x);
  }
  __syncthreads();
  for (int depth = BITONIC_SORT_DEPTH - 1; depth >= 1; --depth) {
    const int segment_length = 1 << (BITONIC_SORT_DEPTH - depth);
    const int segment_index = thread_index / segment_length;
    const bool ascending = outer_ascending ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
    const int num_total_segment = (num_data + segment_length - 1) / segment_length;
    {
      const int inner_depth = depth;
      const int inner_segment_length_half = 1 << (BITONIC_SORT_DEPTH - 1 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      const int offset = ((inner_segment_index_half >> 1) == num_total_segment - 1 && ascending == outer_ascending) ?
        (num_total_segment * segment_length - num_data) : 0;
      const int segment_start = segment_index * segment_length;
      if (inner_segment_index_half % 2 == 0) {
        if (thread_index >= offset + segment_start) {
          const int index_to_compare = thread_index + inner_segment_length_half - offset;
          const INDEX_T this_index = shared_indices[thread_index];
          const INDEX_T other_index = shared_indices[index_to_compare];
          const VAL_T this_value = shared_values[thread_index];
          const VAL_T other_value = shared_values[index_to_compare];
          if (index_to_compare < num_data && (this_value > other_value) == ascending) {
            shared_indices[thread_index] = other_index;
            shared_indices[index_to_compare] = this_index;
            shared_values[thread_index] = other_value;
            shared_values[index_to_compare] = this_value;
          }
        }
      }
      __syncthreads();
    }
    for (int inner_depth = depth + 1; inner_depth < BITONIC_SORT_DEPTH; ++inner_depth) {
      const int inner_segment_length_half = 1 << (BITONIC_SORT_DEPTH - 1 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      if (inner_segment_index_half % 2 == 0) {
        const int index_to_compare = thread_index + inner_segment_length_half;
        const INDEX_T this_index = shared_indices[thread_index];
        const INDEX_T other_index = shared_indices[index_to_compare];
        const VAL_T this_value = shared_values[thread_index];
        const VAL_T other_value = shared_values[index_to_compare];
        if (index_to_compare < num_data && (this_value > other_value) == ascending) {
          shared_indices[thread_index] = other_index;
          shared_indices[index_to_compare] = this_index;
          shared_values[thread_index] = other_value;
          shared_values[index_to_compare] = this_value;
        }
      }
      __syncthreads();
    }
  }
  if (thread_index < num_data) {
    indices_pointer[thread_index] = shared_indices[thread_index];
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__global__ void BitonicArgSortMergeKernel(const VAL_T* values, INDEX_T* indices, const int segment_length, const int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int segment_index = thread_index / segment_length;
  const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  __shared__ INDEX_T shared_indices[BITONIC_SORT_NUM_ELEMENTS];
  const int offset = static_cast<int>(blockIdx.x * blockDim.x);
  const int local_len = min(BITONIC_SORT_NUM_ELEMENTS, len - offset);
  if (thread_index < len) {
    const INDEX_T index = indices[thread_index];
    shared_values[threadIdx.x] = values[index];
    shared_indices[threadIdx.x] = index;
  }
  __syncthreads();
  int half_segment_length = BITONIC_SORT_NUM_ELEMENTS / 2;
  while (half_segment_length >= 1) {
    const int half_segment_index = static_cast<int>(threadIdx.x) / half_segment_length;
    if (half_segment_index % 2 == 0) {
      const int index_to_compare = static_cast<int>(threadIdx.x) + half_segment_length;
      const INDEX_T this_index = shared_indices[threadIdx.x];
      const INDEX_T other_index = shared_indices[index_to_compare];
      const VAL_T this_value = shared_values[threadIdx.x];
      const VAL_T other_value = shared_values[index_to_compare];
      if (index_to_compare < local_len && ((this_value > other_value) == ascending)) {
        shared_indices[threadIdx.x] = other_index;
        shared_indices[index_to_compare] = this_index;
        shared_values[threadIdx.x] = other_value;
        shared_values[index_to_compare] = this_value;
      }
    }
    __syncthreads();
    half_segment_length >>= 1;
  }
  if (thread_index < len) {
    indices[thread_index] = shared_indices[threadIdx.x];
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING, bool BEGIN>
__global__ void BitonicArgCompareKernel(const VAL_T* values, INDEX_T* indices, const int half_segment_length, const int outer_segment_length, const int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int segment_index = thread_index / outer_segment_length;
  const int half_segment_index = thread_index / half_segment_length;
  const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
  if (half_segment_index % 2 == 0) {
    const int num_total_segment = (len + outer_segment_length - 1) / outer_segment_length;
    if (BEGIN && (half_segment_index >> 1) == num_total_segment - 1 && ascending == ASCENDING) {
      const int offset = num_total_segment * outer_segment_length - len;
      const int segment_start = segment_index * outer_segment_length;
      if (thread_index >= offset + segment_start) {
        const int index_to_compare = thread_index + half_segment_length - offset;
        if (index_to_compare < len) {
          const INDEX_T this_index = indices[thread_index];
          const INDEX_T other_index = indices[index_to_compare];
          if ((values[this_index] > values[other_index]) == ascending) {
            indices[thread_index] = other_index;
            indices[index_to_compare] = this_index;
          }
        }
      }
    } else {
      const int index_to_compare = thread_index + half_segment_length;
      if (index_to_compare < len) {
        const INDEX_T this_index = indices[thread_index];
        const INDEX_T other_index = indices[index_to_compare];
        if ((values[this_index] > values[other_index]) == ascending) {
          indices[thread_index] = other_index;
          indices[index_to_compare] = this_index;
        }
      }
    }
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
void BitonicArgSortGlobalHelper(const VAL_T* values, INDEX_T* indices, const size_t len) {
  int max_depth = 1;
  int len_to_shift = static_cast<int>(len) - 1;
  while (len_to_shift > 0) {
    ++max_depth;
    len_to_shift >>= 1;
  }
  const int num_blocks = (static_cast<int>(len) + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  BitonicArgSortGlobalKernel<VAL_T, INDEX_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, indices, static_cast<int>(len));
  SynchronizeCUDADevice(__FILE__, __LINE__);
  for (int depth = max_depth - 11; depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    int half_segment_length = (segment_length >> 1);
    {
      BitonicArgCompareKernel<VAL_T, INDEX_T, ASCENDING, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        values, indices, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADevice(__FILE__, __LINE__);
      half_segment_length >>= 1;
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 11; ++inner_depth) {
      BitonicArgCompareKernel<VAL_T, INDEX_T, ASCENDING, false><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        values, indices, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADevice(__FILE__, __LINE__);
      half_segment_length >>= 1;
    }
    BitonicArgSortMergeKernel<VAL_T, INDEX_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
      values, indices, segment_length, static_cast<int>(len));
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }
}

template <>
void BitonicArgSortGlobal<double, data_size_t, false>(const double* values, data_size_t* indices, const size_t len) {
  BitonicArgSortGlobalHelper<double, data_size_t, false>(values, indices, len);
}

template <>
void BitonicArgSortGlobal<double, data_size_t, true>(const double* values, data_size_t* indices, const size_t len) {
  BitonicArgSortGlobalHelper<double, data_size_t, true>(values, indices, len);
}

template <>
void BitonicArgSortGlobal<label_t, data_size_t, false>(const label_t* values, data_size_t* indices, const size_t len) {
  BitonicArgSortGlobalHelper<label_t, data_size_t, false>(values, indices, len);
}

template <>
void BitonicArgSortGlobal<data_size_t, int, true>(const data_size_t* values, int* indices, const size_t len) {
  BitonicArgSortGlobalHelper<data_size_t, int, true>(values, indices, len);
}

}  // namespace LightGBM

#endif  // USE_CUDA
