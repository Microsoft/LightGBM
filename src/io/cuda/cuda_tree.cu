/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */


#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_tree.hpp>

namespace LightGBM {

__device__ void SetDecisionTypeCUDA(int8_t* decision_type, bool input, int8_t mask) {
  if (input) {
    (*decision_type) |= mask;
  } else {
    (*decision_type) &= (127 - mask);
  }
}

__device__ void SetMissingTypeCUDA(int8_t* decision_type, int8_t input) {
  (*decision_type) &= 3;
  (*decision_type) |= (input << 2);
}

__device__ bool GetDecisionTypeCUDA(int8_t decision_type, int8_t mask) {
  return (decision_type & mask) > 0;
}

__device__ int8_t GetMissingTypeCUDA(int8_t decision_type) {
  return (decision_type >> 2) & 3;
}

__device__ bool IsZeroCUDA(double fval) {
  return (fval >= -kZeroThreshold && fval <= kZeroThreshold);
}

template<typename T>
__device__ bool FindInBitsetCUDA(const uint32_t* bits, int n, T pos) {
  int i1 = pos / 32;
  if (i1 >= n) {
    return false;
  }
  int i2 = pos % 32;
  return (bits[i1] >> i2) & 1;
}

__global__ void SplitKernel(  // split information
                            const int leaf_index,
                            const int real_feature_index,
                            const double real_threshold,
                            const MissingType missing_type,
                            const CUDASplitInfo* cuda_split_info,
                            // tree structure
                            const int num_leaves,
                            int* leaf_parent,
                            int* leaf_depth,
                            int* left_child,
                            int* right_child,
                            int* split_feature_inner,
                            int* split_feature,
                            float* split_gain,
                            double* internal_weight,
                            double* internal_value,
                            data_size_t* internal_count,
                            double* leaf_weight,
                            double* leaf_value,
                            data_size_t* leaf_count,
                            int8_t* decision_type,
                            uint32_t* threshold_in_bin,
                            double* threshold) {
  const int new_node_index = num_leaves - 1;
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int parent_index = leaf_parent[leaf_index];
  if (thread_index == 0) {
    if (parent_index >= 0) {
      // if cur node is left child
      if (left_child[parent_index] == ~leaf_index) {
        left_child[parent_index] = new_node_index;
      } else {
        right_child[parent_index] = new_node_index;
      }
    }
    left_child[new_node_index] = ~leaf_index;
    right_child[new_node_index] = ~num_leaves;
    leaf_parent[leaf_index] = new_node_index;
    leaf_parent[num_leaves] = new_node_index;
  } else if (thread_index == 1) {
    // add new node
    split_feature_inner[new_node_index] = cuda_split_info->inner_feature_index;
  } else if (thread_index == 2) {
    split_feature[new_node_index] = real_feature_index;
  } else if (thread_index == 3) {
    split_gain[new_node_index] = static_cast<float>(cuda_split_info->gain);
  } else if (thread_index == 4) {
    // save current leaf value to internal node before change
    internal_weight[new_node_index] = cuda_split_info->left_sum_hessians + cuda_split_info->right_sum_hessians;
    leaf_weight[leaf_index] = cuda_split_info->left_sum_hessians;
  } else if (thread_index == 5) {
    internal_value[new_node_index] = leaf_value[leaf_index];
    leaf_value[leaf_index] = isnan(cuda_split_info->left_value) ? 0.0f : cuda_split_info->left_value;
  } else if (thread_index == 6) {
    internal_count[new_node_index] = cuda_split_info->left_count + cuda_split_info->right_count;
  } else if (thread_index == 7) {
    leaf_count[leaf_index] = cuda_split_info->left_count;
  } else if (thread_index == 8) {
    leaf_value[num_leaves] = isnan(cuda_split_info->right_value) ? 0.0f : cuda_split_info->right_value;
  } else if (thread_index == 9) {
    leaf_weight[num_leaves] = cuda_split_info->right_sum_hessians;
  } else if (thread_index == 10) {
    leaf_count[num_leaves] = cuda_split_info->right_count;
  } else if (thread_index == 11) {
    // update leaf depth
    leaf_depth[num_leaves] = leaf_depth[leaf_index] + 1;
    leaf_depth[leaf_index]++;
  } else if (thread_index == 12) {
    decision_type[new_node_index] = 0;
    SetDecisionTypeCUDA(&decision_type[new_node_index], false, kCategoricalMask);
    SetDecisionTypeCUDA(&decision_type[new_node_index], cuda_split_info->default_left, kDefaultLeftMask);
    SetMissingTypeCUDA(&decision_type[new_node_index], static_cast<int8_t>(missing_type));
  } else if (thread_index == 13) {
    threshold_in_bin[new_node_index] = cuda_split_info->threshold;
  } else if (thread_index == 14) {
    threshold[new_node_index] = real_threshold;
  }
}

void CUDATree::LaunchSplitKernel(const int leaf_index,
                                 const int real_feature_index,
                                 const double real_threshold,
                                 const MissingType missing_type,
                                 const CUDASplitInfo* cuda_split_info) {
  SplitKernel<<<3, 5, 0, cuda_stream_>>>(
    // split information
    leaf_index,
    real_feature_index,
    real_threshold,
    missing_type,
    cuda_split_info,
    // tree structure
    num_leaves_,
    cuda_leaf_parent_.RawData(),
    cuda_leaf_depth_.RawData(),
    cuda_left_child_.RawData(),
    cuda_right_child_.RawData(),
    cuda_split_feature_inner_.RawData(),
    cuda_split_feature_.RawData(),
    cuda_split_gain_.RawData(),
    cuda_internal_weight_.RawData(),
    cuda_internal_value_.RawData(),
    cuda_internal_count_.RawData(),
    cuda_leaf_weight_.RawData(),
    cuda_leaf_value_.RawData(),
    cuda_leaf_count_.RawData(),
    cuda_decision_type_.RawData(),
    cuda_threshold_in_bin_.RawData(),
    cuda_threshold_.RawData());
}

__global__ void SplitCategoricalKernel(  // split information
  const int leaf_index,
  const int real_feature_index,
  const MissingType missing_type,
  const CUDASplitInfo* cuda_split_info,
  // tree structure
  const int num_leaves,
  int* leaf_parent,
  int* leaf_depth,
  int* left_child,
  int* right_child,
  int* split_feature_inner,
  int* split_feature,
  float* split_gain,
  double* internal_weight,
  double* internal_value,
  data_size_t* internal_count,
  double* leaf_weight,
  double* leaf_value,
  data_size_t* leaf_count,
  int8_t* decision_type,
  uint32_t* threshold_in_bin,
  double* threshold,
  size_t cuda_bitset_len,
  size_t cuda_bitset_inner_len,
  int num_cat,
  int* cuda_cat_boundaries,
  int* cuda_cat_boundaries_inner) {
  const int new_node_index = num_leaves - 1;
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int parent_index = leaf_parent[leaf_index];
  if (thread_index == 0) {
    if (parent_index >= 0) {
      // if cur node is left child
      if (left_child[parent_index] == ~leaf_index) {
        left_child[parent_index] = new_node_index;
      } else {
        right_child[parent_index] = new_node_index;
      }
    }
    left_child[new_node_index] = ~leaf_index;
    right_child[new_node_index] = ~num_leaves;
    leaf_parent[leaf_index] = new_node_index;
    leaf_parent[num_leaves] = new_node_index;
  } else if (thread_index == 1) {
    // add new node
    split_feature_inner[new_node_index] = cuda_split_info->inner_feature_index;
  } else if (thread_index == 2) {
    split_feature[new_node_index] = real_feature_index;
  } else if (thread_index == 3) {
    split_gain[new_node_index] = static_cast<float>(cuda_split_info->gain);
  } else if (thread_index == 4) {
    // save current leaf value to internal node before change
    internal_weight[new_node_index] = cuda_split_info->left_sum_hessians + cuda_split_info->right_sum_hessians;
    leaf_weight[leaf_index] = cuda_split_info->left_sum_hessians;
  } else if (thread_index == 5) {
    internal_value[new_node_index] = leaf_value[leaf_index];
    leaf_value[leaf_index] = isnan(cuda_split_info->left_value) ? 0.0f : cuda_split_info->left_value;
  } else if (thread_index == 6) {
    internal_count[new_node_index] = cuda_split_info->left_count + cuda_split_info->right_count;
  } else if (thread_index == 7) {
    leaf_count[leaf_index] = cuda_split_info->left_count;
  } else if (thread_index == 8) {
    leaf_value[num_leaves] = isnan(cuda_split_info->right_value) ? 0.0f : cuda_split_info->right_value;
  } else if (thread_index == 9) {
    leaf_weight[num_leaves] = cuda_split_info->right_sum_hessians;
  } else if (thread_index == 10) {
    leaf_count[num_leaves] = cuda_split_info->right_count;
  } else if (thread_index == 11) {
    // update leaf depth
    leaf_depth[num_leaves] = leaf_depth[leaf_index] + 1;
    leaf_depth[leaf_index]++;
  } else if (thread_index == 12) {
    decision_type[new_node_index] = 0;
    SetDecisionTypeCUDA(&decision_type[new_node_index], true, kCategoricalMask);
    SetMissingTypeCUDA(&decision_type[new_node_index], static_cast<int8_t>(missing_type));
  } else if (thread_index == 13) {
    threshold_in_bin[new_node_index] = num_cat;
  } else if (thread_index == 14) {
    threshold[new_node_index] = num_cat;
  } else if (thread_index == 15) {
    if (num_cat == 0) {
      cuda_cat_boundaries[num_cat] = 0;
    }
    cuda_cat_boundaries[num_cat + 1] = cuda_cat_boundaries[num_cat] + cuda_bitset_len;
  } else if (thread_index == 16) {
    if (num_cat == 0) {
      cuda_cat_boundaries_inner[num_cat] = 0;
    }
    cuda_cat_boundaries_inner[num_cat + 1] = cuda_cat_boundaries_inner[num_cat] + cuda_bitset_inner_len;
  }
}

void CUDATree::LaunchSplitCategoricalKernel(const int leaf_index,
  const int real_feature_index,
  const MissingType missing_type,
  const CUDASplitInfo* cuda_split_info,
  size_t cuda_bitset_len,
  size_t cuda_bitset_inner_len) {
  SplitCategoricalKernel<<<3, 6, 0, cuda_stream_>>>(
    // split information
    leaf_index,
    real_feature_index,
    missing_type,
    cuda_split_info,
    // tree structure
    num_leaves_,
    cuda_leaf_parent_.RawData(),
    cuda_leaf_depth_.RawData(),
    cuda_left_child_.RawData(),
    cuda_right_child_.RawData(),
    cuda_split_feature_inner_.RawData(),
    cuda_split_feature_.RawData(),
    cuda_split_gain_.RawData(),
    cuda_internal_weight_.RawData(),
    cuda_internal_value_.RawData(),
    cuda_internal_count_.RawData(),
    cuda_leaf_weight_.RawData(),
    cuda_leaf_value_.RawData(),
    cuda_leaf_count_.RawData(),
    cuda_decision_type_.RawData(),
    cuda_threshold_in_bin_.RawData(),
    cuda_threshold_.RawData(),
    cuda_bitset_len,
    cuda_bitset_inner_len,
    num_cat_,
    cuda_cat_boundaries_.RawData(),
    cuda_cat_boundaries_inner_.RawData());
}

__global__ void ShrinkageKernel(const double rate, double* cuda_leaf_value, const int num_leaves) {
  const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (leaf_index < num_leaves) {
    cuda_leaf_value[leaf_index] *= rate;
  }
}

void CUDATree::LaunchShrinkageKernel(const double rate) {
  const int num_threads_per_block = 1024;
  const int num_blocks = (num_leaves_ + num_threads_per_block - 1) / num_threads_per_block;
  ShrinkageKernel<<<num_blocks, num_threads_per_block>>>(rate, cuda_leaf_value_.RawData(), num_leaves_);
}

__global__ void AddBiasKernel(const double val, double* cuda_leaf_value, const int num_leaves) {
  const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (leaf_index < num_leaves) {
    cuda_leaf_value[leaf_index] += val;
  }
}

void CUDATree::LaunchAddBiasKernel(const double val) {
  const int num_threads_per_block = 1024;
  const int num_blocks = (num_leaves_ + num_threads_per_block - 1) / num_threads_per_block;
  AddBiasKernel<<<num_blocks, num_threads_per_block>>>(val, cuda_leaf_value_.RawData(), num_leaves_);
}

template <bool USE_INDICES>
__global__ void AddPredictionToScoreKernel(
  // dataset information
  const data_size_t num_data,
  uint8_t* const* cuda_data_by_column,
  const uint8_t* cuda_column_bit_type,
  const uint32_t* cuda_feature_min_bin,
  const uint32_t* cuda_feature_max_bin,
  const uint32_t* cuda_feature_offset,
  const uint32_t* cuda_feature_default_bin,
  const uint32_t* cuda_feature_most_freq_bin,
  const int* cuda_feature_to_column,
  const data_size_t* cuda_used_indices,
  // tree information
  const uint32_t* cuda_threshold_in_bin,
  const int8_t* cuda_decision_type,
  const int* cuda_split_feature_inner,
  const int* cuda_left_child,
  const int* cuda_right_child,
  const double* cuda_leaf_value,
  const uint32_t* cuda_bitset_inner,
  const int* cuda_cat_boundaries_inner,
  // output
  double* score) {
  const data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (inner_data_index < num_data) {
    const data_size_t data_index = USE_INDICES ? cuda_used_indices[inner_data_index] : inner_data_index;
    int node = 0;
    while (node >= 0) {
      const int split_feature_inner = cuda_split_feature_inner[node];
      const int column = cuda_feature_to_column[split_feature_inner];
      const uint32_t default_bin = cuda_feature_default_bin[split_feature_inner];
      const uint32_t most_freq_bin = cuda_feature_most_freq_bin[split_feature_inner];
      const uint32_t max_bin = cuda_feature_max_bin[split_feature_inner];
      const uint32_t min_bin = cuda_feature_min_bin[split_feature_inner];
      const uint32_t offset = cuda_feature_offset[split_feature_inner];
      const uint8_t column_bit_type = cuda_column_bit_type[column];
      uint32_t bin = 0;
      if (column_bit_type == 8) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint8_t*>(cuda_data_by_column[column]))[data_index]);
      } else if (column_bit_type == 16) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint16_t*>(cuda_data_by_column[column]))[data_index]);
      } else if (column_bit_type == 32) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint32_t*>(cuda_data_by_column[column]))[data_index]);
      }
      if (bin >= min_bin && bin <= max_bin) {
        bin = bin - min_bin + offset;
      } else {
        bin = most_freq_bin;
      }
      const int8_t decision_type = cuda_decision_type[node];
      if (GetDecisionTypeCUDA(decision_type, kCategoricalMask)) {
        int cat_idx = static_cast<int>(cuda_threshold_in_bin[node]);
        if (FindInBitsetCUDA(cuda_bitset_inner + cuda_cat_boundaries_inner[cat_idx],
                             cuda_cat_boundaries_inner[cat_idx + 1] - cuda_cat_boundaries_inner[cat_idx], bin)) {
          node = cuda_left_child[node];
        } else {
          node = cuda_right_child[node];
        }
      } else {
        const uint32_t threshold_in_bin = cuda_threshold_in_bin[node];
        const int8_t missing_type = GetMissingTypeCUDA(decision_type);
        const bool default_left = ((decision_type & kDefaultLeftMask) > 0);
        if ((missing_type == 1 && bin == default_bin) || (missing_type == 2 && bin == max_bin)) {
          if (default_left) {
            node = cuda_left_child[node];
          } else {
            node = cuda_right_child[node];
          }
        } else {
          if (bin <= threshold_in_bin) {
            node = cuda_left_child[node];
          } else {
            node = cuda_right_child[node];
          }
        }
      }
    }
    score[data_index] += cuda_leaf_value[~node];
  }
}

void CUDATree::LaunchAddPredictionToScoreKernel(
  const Dataset* data,
  const data_size_t* used_data_indices,
  data_size_t num_data,
  double* score) const {
  const CUDAColumnData* cuda_column_data = data->cuda_column_data();
  const int num_blocks = (num_data + num_threads_per_block_add_prediction_to_score_ - 1) / num_threads_per_block_add_prediction_to_score_;
  if (used_data_indices == nullptr) {
    AddPredictionToScoreKernel<false><<<num_blocks, num_threads_per_block_add_prediction_to_score_>>>(
      // dataset information
      num_data,
      cuda_column_data->cuda_data_by_column(),
      cuda_column_data->cuda_column_bit_type(),
      cuda_column_data->cuda_feature_min_bin(),
      cuda_column_data->cuda_feature_max_bin(),
      cuda_column_data->cuda_feature_offset(),
      cuda_column_data->cuda_feature_default_bin(),
      cuda_column_data->cuda_feature_most_freq_bin(),
      cuda_column_data->cuda_feature_to_column(),
      nullptr,
      // tree information
      cuda_threshold_in_bin_.RawData(),
      cuda_decision_type_.RawData(),
      cuda_split_feature_inner_.RawData(),
      cuda_left_child_.RawData(),
      cuda_right_child_.RawData(),
      cuda_leaf_value_.RawData(),
      cuda_bitset_inner_.RawDataReadOnly(),
      cuda_cat_boundaries_inner_.RawDataReadOnly(),
      // output
      score);
  } else {
    AddPredictionToScoreKernel<true><<<num_blocks, num_threads_per_block_add_prediction_to_score_>>>(
      // dataset information
      num_data,
      cuda_column_data->cuda_data_by_column(),
      cuda_column_data->cuda_column_bit_type(),
      cuda_column_data->cuda_feature_min_bin(),
      cuda_column_data->cuda_feature_max_bin(),
      cuda_column_data->cuda_feature_offset(),
      cuda_column_data->cuda_feature_default_bin(),
      cuda_column_data->cuda_feature_most_freq_bin(),
      cuda_column_data->cuda_feature_to_column(),
      used_data_indices,
      // tree information
      cuda_threshold_in_bin_.RawData(),
      cuda_decision_type_.RawData(),
      cuda_split_feature_inner_.RawData(),
      cuda_left_child_.RawData(),
      cuda_right_child_.RawData(),
      cuda_leaf_value_.RawData(),
      cuda_bitset_inner_.RawDataReadOnly(),
      cuda_cat_boundaries_inner_.RawDataReadOnly(),
      // output
      score);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
