/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_data_partition.hpp"

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_rocm_interop.h>
#include <LightGBM/tree.h>

#include <algorithm>
#include <vector>

namespace LightGBM {

__global__ void FillDataIndicesBeforeTrainKernel(const data_size_t num_data,
  data_size_t* data_indices, int* cuda_data_index_to_leaf_index) {
  const unsigned int data_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (data_index < num_data) {
    data_indices[data_index] = data_index;
    cuda_data_index_to_leaf_index[data_index] = 0;
  }
}

__global__ void FillDataIndexToLeafIndexKernel(
  const data_size_t num_data,
  const data_size_t* data_indices,
  int* data_index_to_leaf_index) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    data_index_to_leaf_index[data_indices[data_index]] = 0;
  }
}

void CUDADataPartition::LaunchFillDataIndicesBeforeTrain() {
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  FillDataIndicesBeforeTrainKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(num_data_in_root, cuda_data_indices_.RawData(), cuda_data_index_to_leaf_index_.RawData());
}

void CUDADataPartition::LaunchFillDataIndexToLeafIndex() {
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  FillDataIndexToLeafIndexKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(num_data_in_root, cuda_data_indices_.RawData(), cuda_data_index_to_leaf_index_.RawData());
}

__device__ __forceinline__ void PrepareOffset(const data_size_t num_data_in_leaf, uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const uint16_t thread_to_left_offset_cnt, uint16_t* shared_mem_buffer) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const uint16_t thread_to_left_offset = ShufflePrefixSum<uint16_t>(thread_to_left_offset_cnt, shared_mem_buffer);
  const data_size_t num_data_in_block = (blockIdx.x + 1) * blockDim_x <= num_data_in_leaf ? static_cast<data_size_t>(blockDim_x) :
    num_data_in_leaf - static_cast<data_size_t>(blockIdx.x * blockDim_x);
  if (static_cast<data_size_t>(threadIdx_x) < num_data_in_block) {
    block_to_left_offset[threadIdx_x] = thread_to_left_offset;
  }
  if (threadIdx_x == blockDim_x - 1) {
    if (num_data_in_block > 0) {
      const data_size_t data_to_left = static_cast<data_size_t>(thread_to_left_offset);
      block_to_left_offset_buffer[blockIdx.x + 1] = data_to_left;
      block_to_right_offset_buffer[blockIdx.x + 1] = num_data_in_block - data_to_left;
    } else {
      block_to_left_offset_buffer[blockIdx.x + 1] = 0;
      block_to_right_offset_buffer[blockIdx.x + 1] = 0;
    }
  }
}

template <typename T>
__device__ bool CUDAFindInBitset(const uint32_t* bits, int n, T pos) {
  int i1 = pos / 32;
  if (i1 >= n) {
    return false;
  }
  int i2 = pos % 32;
  return (bits[i1] >> i2) & 1;
}



#define UpdateDataIndexToLeafIndexKernel_PARAMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin, \
  const uint32_t min_bin, \
  const int left_leaf_index, \
  const int right_leaf_index, \
  const int default_leaf_index, \
  const int missing_default_leaf_index

#define UpdateDataIndexToLeafIndex_ARGS \
  column_data, \
  num_data_in_leaf, \
  data_indices_in_leaf, th, \
  t_zero_bin, \
  max_bin, \
  min_bin, \
  left_leaf_index, \
  right_leaf_index, \
  default_leaf_index, \
  missing_default_leaf_index

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, bool USE_MIN_BIN, typename BIN_TYPE>
__global__ void UpdateDataIndexToLeafIndexKernel(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  int* cuda_data_index_to_leaf_index) {
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (!MIN_IS_MAX) {
      if ((MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
        (MISSING_IS_NA && !MFB_IS_NA && bin == max_bin)) {
        cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
      } else if ((USE_MIN_BIN && (bin < min_bin || bin > max_bin)) ||
                 (!USE_MIN_BIN && bin == 0)) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
        }
      } else if (bin > th) {
        cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
      } else {
        cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
      }
    } else {
      if (MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
      } else if (bin != max_bin) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
        }
      } else {
        if (MISSING_IS_NA && !MFB_IS_NA) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          if (!MAX_TO_LEFT) {
            cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
          } else {
            cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
          }
        }
      }
    }
  }
}

template <typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool missing_is_zero,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (min_bin < max_bin) {
    if (!missing_is_zero) {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<false, false, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
    } else {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<false, true, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
    }
  } else {
    if (!missing_is_zero) {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<true, false, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
    } else {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<true, true, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
    }
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner0(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (!missing_is_na) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner1<MIN_IS_MAX, MISSING_IS_ZERO, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner1<MIN_IS_MAX, MISSING_IS_ZERO, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_zero, mfb_is_na, max_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner1(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (!mfb_is_zero) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_na, max_to_left, is_single_feature_in_column);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_na, max_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner2(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool mfb_is_na,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (!mfb_is_na) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner3<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, max_to_left, is_single_feature_in_column);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner3<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, max_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner3(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool max_to_left,
  const bool is_single_feature_in_column) {
  if (!max_to_left) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner4<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, is_single_feature_in_column);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner4<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner4(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool is_single_feature_in_column) {
  if (!is_single_feature_in_column) {
    UpdateDataIndexToLeafIndexKernel<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, true, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index_.RawData());
  } else {
    UpdateDataIndexToLeafIndexKernel<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, false, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index_.RawData());
  }
}

#define GenDataToLeftBitVectorKernel_PARAMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin, \
  const uint32_t min_bin, \
  const uint8_t split_default_to_left, \
  const uint8_t split_missing_default_to_left

#define GenBitVector_ARGS \
  column_data, \
  num_data_in_leaf, \
  data_indices_in_leaf, \
  th, \
  t_zero_bin, \
  max_bin, \
  min_bin, \
  split_default_to_left,  \
  split_missing_default_to_left

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, bool USE_MIN_BIN, typename BIN_TYPE>
__global__ void GenDataToLeftBitVectorKernel(
  GenDataToLeftBitVectorKernel_PARAMS,
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer) {
  __shared__ uint16_t shared_mem_buffer[WARPSIZE];
  uint16_t thread_to_left_offset_cnt = 0;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (!MIN_IS_MAX) {
      if ((MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
        (MISSING_IS_NA && !MFB_IS_NA && bin == max_bin)) {
        thread_to_left_offset_cnt = split_missing_default_to_left;
      } else if ((USE_MIN_BIN && (bin < min_bin || bin > max_bin)) ||
                 (!USE_MIN_BIN && bin == 0)) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO || MFB_IS_ZERO)) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else {
          thread_to_left_offset_cnt = split_default_to_left;
        }
      } else if (bin <= th) {
        thread_to_left_offset_cnt = 1;
      }
    } else {
      if (MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        thread_to_left_offset_cnt = split_missing_default_to_left;
      } else if (bin != max_bin) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else {
          thread_to_left_offset_cnt = split_default_to_left;
        }
      } else {
        if (MISSING_IS_NA && !MFB_IS_NA) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else if (MAX_TO_LEFT) {
          thread_to_left_offset_cnt = 1;
        }
      }
    }
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, block_to_left_offset + blockIdx.x * blockDim.x, block_to_left_offset_buffer, block_to_right_offset_buffer,
    thread_to_left_offset_cnt, shared_mem_buffer);
}

template <typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool missing_is_zero,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (min_bin < max_bin) {
    if (!missing_is_zero) {
      LaunchGenDataToLeftBitVectorKernelInner0<false, false, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
    } else {
      LaunchGenDataToLeftBitVectorKernelInner0<false, true, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
    }
  } else {
    if (!missing_is_zero) {
      LaunchGenDataToLeftBitVectorKernelInner0<true, false, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
    } else {
      LaunchGenDataToLeftBitVectorKernelInner0<true, true, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
    }
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner0(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (!missing_is_na) {
    LaunchGenDataToLeftBitVectorKernelInner1<MIN_IS_MAX, MISSING_IS_ZERO, false, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner1<MIN_IS_MAX, MISSING_IS_ZERO, true, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_zero, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner1(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (!mfb_is_zero) {
    LaunchGenDataToLeftBitVectorKernelInner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, false, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, true, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_na, max_bin_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner2(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (!mfb_is_na) {
    LaunchGenDataToLeftBitVectorKernelInner3
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, false, BIN_TYPE>
      (GenBitVector_ARGS, max_bin_to_left, is_single_feature_in_column);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner3
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, true, BIN_TYPE>
      (GenBitVector_ARGS, max_bin_to_left, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner3(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool max_bin_to_left,
  const bool is_single_feature_in_column) {
  if (!max_bin_to_left) {
    LaunchGenDataToLeftBitVectorKernelInner4
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, false, BIN_TYPE>
      (GenBitVector_ARGS, is_single_feature_in_column);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner4
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, true, BIN_TYPE>
      (GenBitVector_ARGS, is_single_feature_in_column);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner4(
  GenDataToLeftBitVectorKernel_PARAMS,
  const bool is_single_feature_in_column) {
  if (!is_single_feature_in_column) {
    GenDataToLeftBitVectorKernel
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, true, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_ARGS,
        cuda_block_to_left_offset_.RawData(), cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData());
  } else {
    GenDataToLeftBitVectorKernel
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, MAX_TO_LEFT, false, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_ARGS,
        cuda_block_to_left_offset_.RawData(), cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData());
  }
}

void CUDADataPartition::LaunchGenDataToLeftBitVectorKernel(
  const data_size_t num_data_in_leaf,
  const int split_feature_index,
  const uint32_t split_threshold,
  const uint8_t split_default_left,
  const data_size_t leaf_data_start,
  const int left_leaf_index,
  const int right_leaf_index) {
  const bool missing_is_zero = static_cast<bool>(cuda_column_data_->feature_missing_is_zero(split_feature_index));
  const bool missing_is_na = static_cast<bool>(cuda_column_data_->feature_missing_is_na(split_feature_index));
  const bool mfb_is_zero = static_cast<bool>(cuda_column_data_->feature_mfb_is_zero(split_feature_index));
  const bool mfb_is_na = static_cast<bool>(cuda_column_data_->feature_mfb_is_na(split_feature_index));
  const bool is_single_feature_in_column = is_single_feature_in_column_[split_feature_index];
  const uint32_t default_bin = cuda_column_data_->feature_default_bin(split_feature_index);
  const uint32_t most_freq_bin = cuda_column_data_->feature_most_freq_bin(split_feature_index);
  const uint32_t min_bin = is_single_feature_in_column ? 1 : cuda_column_data_->feature_min_bin(split_feature_index);
  const uint32_t max_bin = cuda_column_data_->feature_max_bin(split_feature_index);
  uint32_t th = split_threshold + min_bin;
  uint32_t t_zero_bin = min_bin + default_bin;
  if (most_freq_bin == 0) {
    --th;
    --t_zero_bin;
  }
  uint8_t split_default_to_left = 0;
  uint8_t split_missing_default_to_left = 0;
  int default_leaf_index = right_leaf_index;
  int missing_default_leaf_index = right_leaf_index;
  if (most_freq_bin <= split_threshold) {
    split_default_to_left = 1;
    default_leaf_index = left_leaf_index;
  }
  if (missing_is_zero || missing_is_na) {
    if (split_default_left) {
      split_missing_default_to_left = 1;
      missing_default_leaf_index = left_leaf_index;
    }
  }
  const int column_index = cuda_column_data_->feature_to_column(split_feature_index);
  const uint8_t bit_type = cuda_column_data_->column_bit_type(column_index);

  const bool max_bin_to_left = (max_bin <= th);

  const data_size_t* data_indices_in_leaf = cuda_data_indices_.RawData() + leaf_data_start;
  const void* column_data_pointer = cuda_column_data_->GetColumnData(column_index);

  if (bit_type == 8) {
    const uint8_t* column_data = reinterpret_cast<const uint8_t*>(column_data_pointer);
    LaunchGenDataToLeftBitVectorKernelInner<uint8_t>(
      GenBitVector_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
    LaunchUpdateDataIndexToLeafIndexKernel<uint8_t>(
      UpdateDataIndexToLeafIndex_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
  } else if (bit_type == 16) {
    const uint16_t* column_data = reinterpret_cast<const uint16_t*>(column_data_pointer);
    LaunchGenDataToLeftBitVectorKernelInner<uint16_t>(
      GenBitVector_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
    LaunchUpdateDataIndexToLeafIndexKernel<uint16_t>(
      UpdateDataIndexToLeafIndex_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
  } else if (bit_type == 32) {
    const uint32_t* column_data = reinterpret_cast<const uint32_t*>(column_data_pointer);
    LaunchGenDataToLeftBitVectorKernelInner<uint32_t>(
      GenBitVector_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
    LaunchUpdateDataIndexToLeafIndexKernel<uint32_t>(
      UpdateDataIndexToLeafIndex_ARGS,
      missing_is_zero,
      missing_is_na,
      mfb_is_zero,
      mfb_is_na,
      max_bin_to_left,
      is_single_feature_in_column);
  }
}

#undef UpdateDataIndexToLeafIndexKernel_PARAMS
#undef UpdateDataIndexToLeafIndex_ARGS
#undef GenDataToLeftBitVectorKernel_PARAMS
#undef GenBitVector_ARGS

template <typename BIN_TYPE, bool USE_MIN_BIN>
__global__ void UpdateDataIndexToLeafIndexKernel_Categorical(
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t* bitset, const int bitset_len, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t max_bin, const uint32_t min_bin, const int8_t mfb_offset,
  int* cuda_data_index_to_leaf_index, const int left_leaf_index, const int right_leaf_index,
  const int default_leaf_index) {
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (USE_MIN_BIN && (bin < min_bin || bin > max_bin)) {
      cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
    } else if (!USE_MIN_BIN && bin == 0) {
      cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
    } else if (CUDAFindInBitset(bitset, bitset_len, bin - min_bin + mfb_offset)) {
      cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
    } else {
      cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
    }
  }
}

// for categorical features
template <typename BIN_TYPE, bool USE_MIN_BIN>
__global__ void GenDataToLeftBitVectorKernel_Categorical(
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t* bitset, int bitset_len, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t max_bin, const uint32_t min_bin, const int8_t mfb_offset,
  const uint8_t split_default_to_left,
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer) {
  __shared__ uint16_t shared_mem_buffer[WARPSIZE];
  uint16_t thread_to_left_offset_cnt = 0;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (USE_MIN_BIN && (bin < min_bin || bin > max_bin)) {
      thread_to_left_offset_cnt = split_default_to_left;
    } else if (!USE_MIN_BIN && bin == 0) {
      thread_to_left_offset_cnt = split_default_to_left;
    } else if (CUDAFindInBitset(bitset, bitset_len, bin - min_bin + mfb_offset)) {
      thread_to_left_offset_cnt = 1;
    }
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, block_to_left_offset + blockIdx.x * blockDim.x, block_to_left_offset_buffer, block_to_right_offset_buffer,
    thread_to_left_offset_cnt, shared_mem_buffer);
}

#define GenBitVector_Categorical_ARGS \
  num_data_in_leaf, data_indices_in_leaf, \
  bitset, bitset_len, \
  column_data, max_bin, min_bin, mfb_offset, split_default_to_left, \
  cuda_block_to_left_offset_.RawData(), cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData()

#define UpdateDataIndexToLeafIndex_Categorical_ARGS \
  num_data_in_leaf, data_indices_in_leaf, \
  bitset, bitset_len, \
  column_data, max_bin, min_bin, mfb_offset,  \
  cuda_data_index_to_leaf_index_.RawData(), left_leaf_index, right_leaf_index, default_leaf_index

void CUDADataPartition::LaunchGenDataToLeftBitVectorCategoricalKernel(
  const data_size_t num_data_in_leaf,
  const int split_feature_index,
  const uint32_t* bitset,
  const int bitset_len,
  const uint8_t split_default_left,
  const data_size_t leaf_data_start,
  const int left_leaf_index,
  const int right_leaf_index) {
  const data_size_t* data_indices_in_leaf = cuda_data_indices_.RawData() + leaf_data_start;
  const int column_index = cuda_column_data_->feature_to_column(split_feature_index);
  const uint8_t bit_type = cuda_column_data_->column_bit_type(column_index);
  const bool is_single_feature_in_column = is_single_feature_in_column_[split_feature_index];
  const uint32_t min_bin = is_single_feature_in_column ? 1 : cuda_column_data_->feature_min_bin(split_feature_index);
  const uint32_t max_bin = cuda_column_data_->feature_max_bin(split_feature_index);
  const uint32_t most_freq_bin = cuda_column_data_->feature_most_freq_bin(split_feature_index);
  const uint32_t default_bin = cuda_column_data_->feature_default_bin(split_feature_index);
  const void* column_data_pointer = cuda_column_data_->GetColumnData(column_index);
  const int8_t mfb_offset = static_cast<int8_t>(most_freq_bin == 0);
  std::vector<uint32_t> host_bitset(bitset_len, 0);
  CopyFromCUDADeviceToHost<uint32_t>(host_bitset.data(), bitset, bitset_len, __FILE__, __LINE__);
  uint8_t split_default_to_left = 0;
  int default_leaf_index = right_leaf_index;
  if (most_freq_bin > 0 && Common::FindInBitset(host_bitset.data(), bitset_len, most_freq_bin)) {
    split_default_to_left = 1;
    default_leaf_index = left_leaf_index;
  }
  if (bit_type == 8) {
    const uint8_t* column_data = reinterpret_cast<const uint8_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint8_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint8_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint8_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint8_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  } else if (bit_type == 16) {
    const uint16_t* column_data = reinterpret_cast<const uint16_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint16_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint16_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint16_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint16_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  } else if (bit_type == 32) {
    const uint32_t* column_data = reinterpret_cast<const uint32_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint32_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint32_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint32_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint32_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  }
}

#undef GenBitVector_Categorical_ARGS
#undef UpdateDataIndexToLeafIndex_Categorical_ARGS

__global__ void AggregateBlockOffsetKernel0(
  const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t num_blocks) {
  __shared__ uint32_t shared_mem_buffer[WARPSIZE];
  __shared__ uint32_t to_left_total_count;
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index];
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int threadIdx_x = threadIdx.x;
  const data_size_t num_blocks_plus_1 = num_blocks + 1;
  const uint32_t num_blocks_per_thread = (num_blocks_plus_1 + blockDim_x - 1) / blockDim_x;
  const uint32_t remain = num_blocks_plus_1 - ((num_blocks_per_thread - 1) * blockDim_x);
  const uint32_t remain_offset = remain * num_blocks_per_thread;
  uint32_t thread_start_block_index = 0;
  uint32_t thread_end_block_index = 0;
  if (threadIdx_x < remain) {
    thread_start_block_index = threadIdx_x * num_blocks_per_thread;
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread, num_blocks_plus_1);
  } else {
    thread_start_block_index = remain_offset + (num_blocks_per_thread - 1) * (threadIdx_x - remain);
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread - 1, num_blocks_plus_1);
  }
  if (threadIdx.x == 0) {
    block_to_right_offset_buffer[0] = 0;
  }
  __syncthreads();
  for (uint32_t block_index = thread_start_block_index + 1; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += block_to_left_offset_buffer[block_index - 1];
    block_to_right_offset_buffer[block_index] += block_to_right_offset_buffer[block_index - 1];
  }
  __syncthreads();
  uint32_t block_to_left_offset = 0;
  uint32_t block_to_right_offset = 0;
  if (thread_start_block_index < thread_end_block_index && thread_start_block_index > 1) {
    block_to_left_offset = block_to_left_offset_buffer[thread_start_block_index - 1];
    block_to_right_offset = block_to_right_offset_buffer[thread_start_block_index - 1];
  }
  block_to_left_offset = ShufflePrefixSum<uint32_t>(block_to_left_offset, shared_mem_buffer);
  __syncthreads();
  block_to_right_offset = ShufflePrefixSum<uint32_t>(block_to_right_offset, shared_mem_buffer);
  if (threadIdx_x == blockDim_x - 1) {
    to_left_total_count = block_to_left_offset + block_to_left_offset_buffer[num_blocks];
  }
  __syncthreads();
  const uint32_t to_left_thread_block_offset = block_to_left_offset;
  const uint32_t to_right_thread_block_offset = block_to_right_offset + to_left_total_count;
  for (uint32_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += to_left_thread_block_offset;
    block_to_right_offset_buffer[block_index] += to_right_thread_block_offset;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_num_data[left_leaf_index] = static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = num_data_in_leaf - static_cast<data_size_t>(to_left_total_count);
  }
}

__global__ void AggregateBlockOffsetKernel1(
  const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t num_blocks) {
  __shared__ uint32_t shared_mem_buffer[WARPSIZE];
  __shared__ uint32_t to_left_total_count;
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index];
  const unsigned int threadIdx_x = threadIdx.x;
  uint32_t block_to_left_offset = 0;
  uint32_t block_to_right_offset = 0;
  if (threadIdx_x < static_cast<unsigned int>(num_blocks)) {
    block_to_left_offset = block_to_left_offset_buffer[threadIdx_x + 1];
    block_to_right_offset = block_to_right_offset_buffer[threadIdx_x + 1];
  }
  block_to_left_offset = ShufflePrefixSum<uint32_t>(block_to_left_offset, shared_mem_buffer);
  __syncthreads();
  block_to_right_offset = ShufflePrefixSum<uint32_t>(block_to_right_offset, shared_mem_buffer);
  if (threadIdx.x == blockDim.x - 1) {
    to_left_total_count = block_to_left_offset;
  }
  __syncthreads();
  if (threadIdx_x < static_cast<unsigned int>(num_blocks)) {
    block_to_left_offset_buffer[threadIdx_x + 1] = block_to_left_offset;
    block_to_right_offset_buffer[threadIdx_x + 1] = block_to_right_offset + to_left_total_count;
  }
  if (threadIdx_x == 0) {
    block_to_right_offset_buffer[0] = to_left_total_count;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_num_data[left_leaf_index] = static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = num_data_in_leaf - static_cast<data_size_t>(to_left_total_count);
  }
}

template <bool USE_NCCL, bool USE_GRAD_DISCRETIZED>
__global__ void SplitTreeStructureKernel(const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const CUDASplitInfo* best_split_info,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  const int num_total_bin,
  hist_t* cuda_hist, hist_t** cuda_hist_pool,
  double* cuda_leaf_output,
  int* cuda_split_info_buffer) {
  const unsigned int to_left_total_cnt = cuda_leaf_num_data[left_leaf_index];
  double* cuda_split_info_buffer_for_hessians = reinterpret_cast<double*>(cuda_split_info_buffer + 8);
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_index == 0) {
    cuda_leaf_output[left_leaf_index] = best_split_info->left_value;
  } else if (global_thread_index == 1) {
    cuda_leaf_output[right_leaf_index] = best_split_info->right_value;
  } else if (global_thread_index == 2) {
    cuda_split_info_buffer[0] = left_leaf_index;
  } else if (global_thread_index == 3) {
    cuda_split_info_buffer[1] = cuda_leaf_num_data[left_leaf_index];
  } else if (global_thread_index == 4) {
    cuda_split_info_buffer[2] = cuda_leaf_data_start[left_leaf_index];
  } else if (global_thread_index == 5) {
    cuda_split_info_buffer[3] = right_leaf_index;
  } else if (global_thread_index == 6) {
    cuda_split_info_buffer[4] = cuda_leaf_num_data[right_leaf_index];
  } else if (global_thread_index == 7) {
    cuda_split_info_buffer[5] = cuda_leaf_data_start[right_leaf_index];
  } else if (global_thread_index == 8) {
    cuda_split_info_buffer_for_hessians[0] = best_split_info->left_sum_hessians;
    cuda_split_info_buffer_for_hessians[2] = best_split_info->left_sum_gradients;
  } else if (global_thread_index == 9) {
    cuda_split_info_buffer_for_hessians[1] = best_split_info->right_sum_hessians;
    cuda_split_info_buffer_for_hessians[3] = best_split_info->right_sum_gradients;
  }

  bool left_is_smaller = USE_NCCL ?
    cuda_split_info_buffer[16] < cuda_split_info_buffer[17] :
    cuda_leaf_num_data[left_leaf_index] < cuda_leaf_num_data[right_leaf_index];

  if (left_is_smaller) {
    if (global_thread_index == 0) {
      hist_t* parent_hist_ptr = cuda_hist_pool[left_leaf_index];
      cuda_hist_pool[right_leaf_index] = parent_hist_ptr;
      cuda_hist_pool[left_leaf_index] = USE_GRAD_DISCRETIZED ?
        cuda_hist + right_leaf_index * num_total_bin :
        cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 1) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
    } else if (global_thread_index == 2) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      smaller_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      smaller_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      smaller_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      smaller_leaf_splits->data_indices_in_leaf = cuda_data_indices;
    } else if (global_thread_index == 7) {
      larger_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      larger_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
    } else if (global_thread_index == 9) {
      larger_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      larger_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      larger_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      larger_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      larger_leaf_splits->data_indices_in_leaf = cuda_data_indices + cuda_leaf_num_data[left_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_split_info_buffer[6] = left_leaf_index;
    } else if (global_thread_index == 15) {
      cuda_split_info_buffer[7] = right_leaf_index;
    } else if (global_thread_index == 16) {
      smaller_leaf_splits->leaf_index = left_leaf_index;
    }
  } else {
    if (global_thread_index == 0) {
      larger_leaf_splits->leaf_index = left_leaf_index;
    } else if (global_thread_index == 1) {
      larger_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
    } else if (global_thread_index == 2) {
      larger_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      larger_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      larger_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      larger_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      larger_leaf_splits->data_indices_in_leaf = cuda_data_indices;
    } else if (global_thread_index == 7) {
      smaller_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
    } else if (global_thread_index == 9) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      smaller_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      smaller_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      smaller_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      smaller_leaf_splits->data_indices_in_leaf = cuda_data_indices + cuda_leaf_num_data[left_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_hist_pool[right_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 15) {
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
    } else if (global_thread_index == 16) {
      cuda_split_info_buffer[6] = right_leaf_index;
    } else if (global_thread_index == 17) {
      cuda_split_info_buffer[7] = left_leaf_index;
    }
  }
}

__global__ void SplitInnerKernel(const int left_leaf_index, const int right_leaf_index,
  const data_size_t* cuda_leaf_data_start, const data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices,
  const data_size_t* block_to_left_offset_buffer, const data_size_t* block_to_right_offset_buffer,
  const uint16_t* block_to_left_offset, data_size_t* out_data_indices_in_leaf) {
  const data_size_t leaf_num_data_offset = cuda_leaf_data_start[left_leaf_index];
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index] + cuda_leaf_num_data[right_leaf_index];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim_x + threadIdx_x;
  const data_size_t* cuda_data_indices_in_leaf = cuda_data_indices + leaf_num_data_offset;
  const uint16_t* block_to_left_offset_ptr = block_to_left_offset + blockIdx.x * blockDim_x;
  const uint32_t to_right_block_offset = block_to_right_offset_buffer[blockIdx.x];
  const uint32_t to_left_block_offset = block_to_left_offset_buffer[blockIdx.x];
  data_size_t* left_out_data_indices_in_leaf = out_data_indices_in_leaf + to_left_block_offset;
  data_size_t* right_out_data_indices_in_leaf = out_data_indices_in_leaf + to_right_block_offset;
  if (static_cast<data_size_t>(global_thread_index) < num_data_in_leaf) {
    const uint32_t thread_to_left_offset = (threadIdx_x == 0 ? 0 : block_to_left_offset_ptr[threadIdx_x - 1]);
    const bool to_left = block_to_left_offset_ptr[threadIdx_x] > thread_to_left_offset;
    if (to_left) {
      left_out_data_indices_in_leaf[thread_to_left_offset] = cuda_data_indices_in_leaf[global_thread_index];
    } else {
      const uint32_t thread_to_right_offset = threadIdx.x - thread_to_left_offset;
      right_out_data_indices_in_leaf[thread_to_right_offset] = cuda_data_indices_in_leaf[global_thread_index];
    }
  }
}

__global__ void CopyDataIndicesKernel(
  const data_size_t num_data_in_leaf,
  const data_size_t* out_data_indices_in_leaf,
  data_size_t* cuda_data_indices) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx_x;
  if (global_thread_index < num_data_in_leaf) {
    cuda_data_indices[global_thread_index] = out_data_indices_in_leaf[global_thread_index];
  }
}

void CUDADataPartition::LaunchSplitInnerKernel(
  const data_size_t num_data_in_leaf,
  const CUDASplitInfo* best_split_info,
  const int left_leaf_index,
  const int right_leaf_index,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  data_size_t* left_leaf_num_data_ref,
  data_size_t* right_leaf_num_data_ref,
  data_size_t* left_leaf_start_ref,
  data_size_t* right_leaf_start_ref,
  double* left_leaf_sum_of_hessians_ref,
  double* right_leaf_sum_of_hessians_ref,
  double* left_leaf_sum_of_gradients_ref,
  double* right_leaf_sum_of_gradients_ref,
  data_size_t* global_left_leaf_num_data,
  data_size_t* global_right_leaf_num_data) {
  int num_blocks_final_ref = grid_dim_ - 1;
  int num_blocks_final_aligned = 1;
  while (num_blocks_final_ref > 0) {
    num_blocks_final_aligned <<= 1;
    num_blocks_final_ref >>= 1;
  }
  global_timer.Start("CUDADataPartition::AggregateBlockOffsetKernel");

  if (grid_dim_ > AGGREGATE_BLOCK_SIZE_DATA_PARTITION) {
    AggregateBlockOffsetKernel0<<<1, AGGREGATE_BLOCK_SIZE_DATA_PARTITION, 0, cuda_streams_[0]>>>(
      left_leaf_index,
      right_leaf_index,
      cuda_block_data_to_left_offset_.RawData(),
      cuda_block_data_to_right_offset_.RawData(), cuda_leaf_data_start_.RawData(), cuda_leaf_data_end_.RawData(),
      cuda_leaf_num_data_.RawData(), cuda_data_indices_.RawData(),
      grid_dim_);
  } else {
    AggregateBlockOffsetKernel1<<<1, num_blocks_final_aligned, 0, cuda_streams_[0]>>>(
      left_leaf_index,
      right_leaf_index,
      cuda_block_data_to_left_offset_.RawData(),
      cuda_block_data_to_right_offset_.RawData(), cuda_leaf_data_start_.RawData(), cuda_leaf_data_end_.RawData(),
      cuda_leaf_num_data_.RawData(), cuda_data_indices_.RawData(),
      grid_dim_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::AggregateBlockOffsetKernel");

  if (nccl_communicator_ != nullptr) {
    NCCLGroupStart();
    NCCLAllReduce<data_size_t>(
      cuda_leaf_num_data_.RawData() + left_leaf_index,
      cuda_split_info_buffer_.RawData() + 16,
      1, ncclInt32, ncclSum, nccl_communicator_, cuda_streams_[0]);
      NCCLAllReduce<data_size_t>(
        cuda_leaf_num_data_.RawData() + right_leaf_index,
        cuda_split_info_buffer_.RawData() + 17,
        1, ncclInt32, ncclSum, nccl_communicator_, cuda_streams_[0]);
    NCCLGroupEnd();
  }

  global_timer.Start("CUDADataPartition::SplitInnerKernel");
  SplitInnerKernel<<<grid_dim_, block_dim_, 0, cuda_streams_[1]>>>(
    left_leaf_index, right_leaf_index, cuda_leaf_data_start_.RawData(), cuda_leaf_num_data_.RawData(), cuda_data_indices_.RawData(),
    cuda_block_data_to_left_offset_.RawData(), cuda_block_data_to_right_offset_.RawData(), cuda_block_to_left_offset_.RawData(),
    cuda_out_data_indices_in_leaf_.RawData());
  global_timer.Stop("CUDADataPartition::SplitInnerKernel");
  SynchronizeCUDADevice(__FILE__, __LINE__);

  global_timer.Start("CUDADataPartition::SplitTreeStructureKernel");

#define SPLIT_TREE_ARGS \
  left_leaf_index, right_leaf_index, \
  cuda_block_data_to_left_offset_.RawData(), \
  cuda_block_data_to_right_offset_.RawData(), cuda_leaf_data_start_.RawData(), cuda_leaf_data_end_.RawData(), \
  cuda_leaf_num_data_.RawData(), cuda_out_data_indices_in_leaf_.RawData(), \
  best_split_info, \
  smaller_leaf_splits, \
  larger_leaf_splits, \
  num_total_bin_, \
  cuda_hist_, \
  cuda_hist_pool_.RawData(), \
  cuda_leaf_output_.RawData(), cuda_split_info_buffer_.RawData()

  if (nccl_communicator_ != nullptr) {
    if (use_quantized_grad_) {
      SplitTreeStructureKernel<true, true><<<4, 5, 0, cuda_streams_[0]>>>(SPLIT_TREE_ARGS);
    } else {
      SplitTreeStructureKernel<true, false><<<4, 5, 0, cuda_streams_[0]>>>(SPLIT_TREE_ARGS);
    }
  } else {
    if (use_quantized_grad_) {
      SplitTreeStructureKernel<false, true><<<4, 5, 0, cuda_streams_[0]>>>(SPLIT_TREE_ARGS);
    } else {
      SplitTreeStructureKernel<false, false><<<4, 5, 0, cuda_streams_[0]>>>(SPLIT_TREE_ARGS);
    }
  }

#undef SPLIT_TREE_ARGS
  global_timer.Stop("CUDADataPartition::SplitTreeStructureKernel");
  std::vector<int> cpu_split_info_buffer(18);
  const double* cpu_sum_hessians_info = reinterpret_cast<const double*>(cpu_split_info_buffer.data() + 8);
  global_timer.Start("CUDADataPartition::CopyFromCUDADeviceToHostAsync");
  CopyFromCUDADeviceToHostAsync<int>(cpu_split_info_buffer.data(), cuda_split_info_buffer_.RawData(), 18, cuda_streams_[0], __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::CopyFromCUDADeviceToHostAsync");
  const data_size_t left_leaf_num_data = cpu_split_info_buffer[1];
  const data_size_t left_leaf_data_start = cpu_split_info_buffer[2];
  const data_size_t right_leaf_num_data = cpu_split_info_buffer[4];
  global_timer.Start("CUDADataPartition::CopyDataIndicesKernel");
  CopyDataIndicesKernel<<<grid_dim_, block_dim_, 0, cuda_streams_[2]>>>(
    left_leaf_num_data + right_leaf_num_data, cuda_out_data_indices_in_leaf_.RawData(), cuda_data_indices_.RawData() + left_leaf_data_start);
  global_timer.Stop("CUDADataPartition::CopyDataIndicesKernel");
  const data_size_t right_leaf_data_start = cpu_split_info_buffer[5];
  *left_leaf_num_data_ref = left_leaf_num_data;
  *left_leaf_start_ref = left_leaf_data_start;
  *right_leaf_num_data_ref = right_leaf_num_data;
  *right_leaf_start_ref = right_leaf_data_start;
  *left_leaf_sum_of_hessians_ref = cpu_sum_hessians_info[0];
  *right_leaf_sum_of_hessians_ref = cpu_sum_hessians_info[1];
  *left_leaf_sum_of_gradients_ref = cpu_sum_hessians_info[2];
  *right_leaf_sum_of_gradients_ref = cpu_sum_hessians_info[3];
  if (nccl_communicator_ != nullptr) {
    *global_left_leaf_num_data = cpu_split_info_buffer[16];
    *global_right_leaf_num_data = cpu_split_info_buffer[17];
  }
}

template <bool USE_BAGGING>
__global__ void AddPredictionToScoreKernel(
  const data_size_t* data_indices_in_leaf,
  const double* leaf_value, double* cuda_scores,
  const int* cuda_data_index_to_leaf_index, const data_size_t num_data) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockIdx_x = blockIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const data_size_t local_data_index = static_cast<data_size_t>(blockIdx_x * blockDim_x + threadIdx_x);
  if (local_data_index < num_data) {
    if (USE_BAGGING) {
      const data_size_t global_data_index = data_indices_in_leaf[local_data_index];
      const int leaf_index = cuda_data_index_to_leaf_index[global_data_index];
      const double leaf_prediction_value = leaf_value[leaf_index];
      cuda_scores[global_data_index] += leaf_prediction_value;
    } else {
      const int leaf_index = cuda_data_index_to_leaf_index[local_data_index];
      const double leaf_prediction_value = leaf_value[leaf_index];
      cuda_scores[local_data_index] += leaf_prediction_value;
    }
  }
}

void CUDADataPartition::LaunchAddPredictionToScoreKernel(const double* leaf_value, double* cuda_scores) {
  global_timer.Start("CUDADataPartition::AddPredictionToScoreKernel");
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  if (use_bagging_) {
    AddPredictionToScoreKernel<true><<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
      cuda_data_indices_.RawData(), leaf_value, cuda_scores, cuda_data_index_to_leaf_index_.RawData(), num_data_in_root);
  } else {
    AddPredictionToScoreKernel<false><<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
      cuda_data_indices_.RawData(), leaf_value, cuda_scores, cuda_data_index_to_leaf_index_.RawData(), num_data_in_root);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::AddPredictionToScoreKernel");
}

__global__ void RenewDiscretizedTreeLeavesKernel(
  const score_t* gradients,
  const score_t* hessians,
  const data_size_t* data_indices,
  const data_size_t* leaf_data_start,
  const data_size_t* leaf_num_data,
  double* leaf_grad_stat_buffer,
  double* leaf_hess_stat_buffer,
  double* leaf_values) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const int leaf_index = static_cast<int>(blockIdx.x);
  const data_size_t* data_indices_in_leaf = data_indices + leaf_data_start[leaf_index];
  const data_size_t num_data_in_leaf = leaf_num_data[leaf_index];
  double sum_gradients = 0.0f;
  double sum_hessians = 0.0f;
  for (data_size_t inner_data_index = static_cast<int>(threadIdx.x);
    inner_data_index < num_data_in_leaf; inner_data_index += static_cast<int>(blockDim.x)) {
    const data_size_t data_index = data_indices_in_leaf[inner_data_index];
    const score_t gradient = gradients[data_index];
    const score_t hessian = hessians[data_index];
    sum_gradients += static_cast<double>(gradient);
    sum_hessians += static_cast<double>(hessian);
  }
  sum_gradients = ShuffleReduceSum<double>(sum_gradients, shared_mem_buffer, blockDim.x);
  __syncthreads();
  sum_hessians = ShuffleReduceSum<double>(sum_hessians, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    leaf_grad_stat_buffer[leaf_index] = sum_gradients;
    leaf_hess_stat_buffer[leaf_index] = sum_hessians;
  }
}

void CUDADataPartition::LaunchReduceLeafGradStat(
  const score_t* gradients, const score_t* hessians,
  CUDATree* tree, double* leaf_grad_stat_buffer, double* leaf_hess_state_buffer) const {
  const int num_blocks = tree->num_leaves();
  RenewDiscretizedTreeLeavesKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
    gradients,
    hessians,
    cuda_data_indices_.RawData(),
    cuda_leaf_data_start_.RawData(),
    cuda_leaf_num_data_.RawData(),
    leaf_grad_stat_buffer,
    leaf_hess_state_buffer,
    tree->cuda_leaf_value_ref());
}

}  // namespace LightGBM

#endif  // USE_CUDA
