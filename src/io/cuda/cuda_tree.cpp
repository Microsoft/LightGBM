/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_tree.hpp>

namespace LightGBM {

CUDATree::CUDATree(int max_leaves, bool track_branch_features, bool is_linear,
  const int gpu_device_id, const bool has_categorical_feature):
Tree(max_leaves, track_branch_features, is_linear),
num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
  if (has_categorical_feature) {
    cuda_cat_boundaries_.Resize(max_leaves);
    cuda_cat_boundaries_inner_.Resize(max_leaves);
  }
  InitCUDAMemory();
}

CUDATree::CUDATree(const Tree* host_tree):
  Tree(*host_tree),
  num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  InitCUDA();
}

CUDATree::~CUDATree() {
  gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
}

void CUDATree::InitCUDAMemory() {
  cuda_left_child_.Resize(static_cast<size_t>(max_leaves_));
  cuda_right_child_.Resize(static_cast<size_t>(max_leaves_));
  cuda_split_feature_inner_.Resize(static_cast<size_t>(max_leaves_));
  cuda_split_feature_.Resize(static_cast<size_t>(max_leaves_));
  cuda_leaf_depth_.Resize(static_cast<size_t>(max_leaves_));
  cuda_leaf_parent_.Resize(static_cast<size_t>(max_leaves_));
  cuda_threshold_in_bin_.Resize(static_cast<size_t>(max_leaves_));
  cuda_threshold_.Resize(static_cast<size_t>(max_leaves_));
  cuda_decision_type_.Resize(static_cast<size_t>(max_leaves_));
  cuda_leaf_value_.Resize(static_cast<size_t>(max_leaves_));
  cuda_internal_weight_.Resize(static_cast<size_t>(max_leaves_));
  cuda_internal_value_.Resize(static_cast<size_t>(max_leaves_));
  cuda_leaf_weight_.Resize(static_cast<size_t>(max_leaves_));
  cuda_leaf_count_.Resize(static_cast<size_t>(max_leaves_));
  cuda_internal_count_.Resize(static_cast<size_t>(max_leaves_));
  cuda_split_gain_.Resize(static_cast<size_t>(max_leaves_));
  SetCUDAMemory<double>(cuda_leaf_value_.RawData(), 0.0f, 1, __FILE__, __LINE__);
  SetCUDAMemory<double>(cuda_leaf_weight_.RawData(), 0.0f, 1, __FILE__, __LINE__);
  SetCUDAMemory<int>(cuda_leaf_depth_.RawData(), 0, 1, __FILE__, __LINE__);
  SetCUDAMemory<int>(cuda_leaf_parent_.RawData(), -1, 1, __FILE__, __LINE__);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDATree::InitCUDA() {
  cuda_left_child_.InitFromHostVector(left_child_);
  cuda_right_child_.InitFromHostVector(right_child_);
  cuda_split_feature_inner_.InitFromHostVector(split_feature_inner_);
  cuda_split_feature_.InitFromHostVector(split_feature_);
  cuda_threshold_in_bin_.InitFromHostVector(threshold_in_bin_);
  cuda_threshold_.InitFromHostVector(threshold_);
  cuda_leaf_depth_.InitFromHostVector(leaf_depth_);
  cuda_decision_type_.InitFromHostVector(decision_type_);
  cuda_internal_weight_.InitFromHostVector(internal_weight_);
  cuda_internal_value_.InitFromHostVector(internal_value_);
  cuda_internal_count_.InitFromHostVector(internal_count_);
  cuda_leaf_count_.InitFromHostVector(leaf_count_);
  cuda_split_gain_.InitFromHostVector(split_gain_);
  cuda_leaf_value_.InitFromHostVector(leaf_value_);
  cuda_leaf_weight_.InitFromHostVector(leaf_weight_);
  cuda_leaf_parent_.InitFromHostVector(leaf_parent_);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

int CUDATree::Split(const int leaf_index,
           const int real_feature_index,
           const double real_threshold,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info) {
  LaunchSplitKernel(leaf_index, real_feature_index, real_threshold, missing_type, cuda_split_info);
  RecordBranchFeatures(leaf_index, num_leaves_, real_feature_index);
  ++num_leaves_;
  return num_leaves_ - 1;
}

int CUDATree::SplitCategorical(const int leaf_index,
           const int real_feature_index,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info,
           uint32_t* cuda_bitset,
           size_t cuda_bitset_len,
           uint32_t* cuda_bitset_inner,
           size_t cuda_bitset_inner_len) {
  LaunchSplitCategoricalKernel(leaf_index, real_feature_index,
    missing_type, cuda_split_info,
    cuda_bitset_len, cuda_bitset_inner_len);
  cuda_bitset_.PushBack(cuda_bitset, cuda_bitset_len);
  cuda_bitset_inner_.PushBack(cuda_bitset_inner, cuda_bitset_inner_len);
  ++num_leaves_;
  ++num_cat_;
  RecordBranchFeatures(leaf_index, num_leaves_, real_feature_index);
  return num_leaves_ - 1;
}

void CUDATree::RecordBranchFeatures(const int left_leaf_index,
                                    const int right_leaf_index,
                                    const int real_feature_index) {
  if (track_branch_features_) {
    branch_features_[right_leaf_index] = branch_features_[left_leaf_index];
    branch_features_[right_leaf_index].push_back(real_feature_index);
    branch_features_[left_leaf_index].push_back(real_feature_index);
  }
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    data_size_t num_data,
                                    double* score) const {
  LaunchAddPredictionToScoreKernel(data, nullptr, num_data, score);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    const data_size_t* used_data_indices,
                                    data_size_t num_data, double* score) const {
  LaunchAddPredictionToScoreKernel(data, used_data_indices, num_data, score);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

inline void CUDATree::Shrinkage(double rate) {
  Tree::Shrinkage(rate);
  LaunchShrinkageKernel(rate);
}

inline void CUDATree::AddBias(double val) {
  Tree::AddBias(val);
  LaunchAddBiasKernel(val);
}

void CUDATree::ToHost() {
  left_child_.resize(max_leaves_ - 1);
  right_child_.resize(max_leaves_ - 1);
  split_feature_inner_.resize(max_leaves_ - 1);
  split_feature_.resize(max_leaves_ - 1);
  threshold_in_bin_.resize(max_leaves_ - 1);
  threshold_.resize(max_leaves_ - 1);
  decision_type_.resize(max_leaves_ - 1, 0);
  split_gain_.resize(max_leaves_ - 1);
  leaf_parent_.resize(max_leaves_);
  leaf_value_.resize(max_leaves_);
  leaf_weight_.resize(max_leaves_);
  leaf_count_.resize(max_leaves_);
  internal_value_.resize(max_leaves_ - 1);
  internal_weight_.resize(max_leaves_ - 1);
  internal_count_.resize(max_leaves_ - 1);
  leaf_depth_.resize(max_leaves_);

  const size_t num_leaves_size = static_cast<size_t>(num_leaves_);
  CopyFromCUDADeviceToHost<int>(left_child_.data(), cuda_left_child_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(right_child_.data(), cuda_right_child_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(split_feature_inner_.data(), cuda_split_feature_inner_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(split_feature_.data(), cuda_split_feature_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<uint32_t>(threshold_in_bin_.data(), cuda_threshold_in_bin_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(threshold_.data(), cuda_threshold_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int8_t>(decision_type_.data(), cuda_decision_type_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<float>(split_gain_.data(), cuda_split_gain_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(leaf_parent_.data(), cuda_leaf_parent_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(leaf_value_.data(), cuda_leaf_value_.RawData(), num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(leaf_weight_.data(), cuda_leaf_weight_.RawData(), num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(leaf_count_.data(), cuda_leaf_count_.RawData(), num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(internal_value_.data(), cuda_internal_value_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(internal_weight_.data(), cuda_internal_weight_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(internal_count_.data(), cuda_internal_count_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(leaf_depth_.data(), cuda_leaf_depth_.RawData(), num_leaves_size, __FILE__, __LINE__);

  if (num_cat_ > 0) {
    cuda_cat_boundaries_inner_.Resize(num_cat_ + 1);
    cuda_cat_boundaries_.Resize(num_cat_ + 1);
    cat_boundaries_ = cuda_cat_boundaries_.ToHost();
    cat_boundaries_inner_ = cuda_cat_boundaries_inner_.ToHost();
    cat_threshold_ = cuda_bitset_.ToHost();
    cat_threshold_inner_ = cuda_bitset_inner_.ToHost();
  }

  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDATree::SyncLeafOutputFromHostToCUDA() {
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_.RawData(), leaf_value_.data(), leaf_value_.size(), __FILE__, __LINE__);
}

void CUDATree::SyncLeafOutputFromCUDAToHost() {
  CopyFromCUDADeviceToHost<double>(leaf_value_.data(), cuda_leaf_value_.RawData(), leaf_value_.size(), __FILE__, __LINE__);
}

void CUDATree::AsConstantTree(double val, int count) {
  Tree::AsConstantTree(val, count);
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_.RawData(), &val, 1, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(cuda_leaf_count_.RawData(), &count, 1, __FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
