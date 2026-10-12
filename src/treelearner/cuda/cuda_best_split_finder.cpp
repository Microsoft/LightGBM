/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <algorithm>
#include <vector>

#include "cuda_best_split_finder.hpp"
#include "cuda_leaf_splits.hpp"

namespace LightGBM {

CUDABestSplitFinder::CUDABestSplitFinder(
  const hist_t* cuda_hist,
  const Dataset* train_data,
  const std::vector<uint32_t>& feature_hist_offsets,
  const bool select_features_by_node,
  const Config* config):
  num_features_(train_data->num_features()),
  num_leaves_(config->num_leaves),
  feature_hist_offsets_(feature_hist_offsets),
  lambda_l1_(config->lambda_l1),
  lambda_l2_(config->lambda_l2),
  min_data_in_leaf_(config->min_data_in_leaf),
  min_sum_hessian_in_leaf_(config->min_sum_hessian_in_leaf),
  min_gain_to_split_(config->min_gain_to_split),
  cat_smooth_(config->cat_smooth),
  cat_l2_(config->cat_l2),
  max_cat_threshold_(config->max_cat_threshold),
  min_data_per_group_(config->min_data_per_group),
  max_cat_to_onehot_(config->max_cat_to_onehot),
  extra_trees_(config->extra_trees),
  extra_seed_(config->extra_seed),
  use_smoothing_(config->path_smooth > 0),
  path_smooth_(config->path_smooth),
  num_total_bin_(feature_hist_offsets.empty() ? 0 : static_cast<int>(feature_hist_offsets.back())),
  select_features_by_node_(select_features_by_node),
  cuda_hist_(cuda_hist) {
  InitFeatureMetaInfo(train_data);
  if (has_categorical_feature_ && config->use_quantized_grad) {
    Log::Fatal("Quantized training on GPU with categorical features is not supported yet.");
  }
}

CUDABestSplitFinder::~CUDABestSplitFinder() {
  gpuAssert(cudaStreamDestroy(cuda_streams_[0]), __FILE__, __LINE__);
  gpuAssert(cudaStreamDestroy(cuda_streams_[1]), __FILE__, __LINE__);
  cuda_streams_.clear();
  cuda_streams_.shrink_to_fit();
}

void CUDABestSplitFinder::InitFeatureMetaInfo(const Dataset* train_data) {
  feature_missing_type_.resize(num_features_);
  feature_mfb_offsets_.resize(num_features_);
  feature_default_bins_.resize(num_features_);
  feature_num_bins_.resize(num_features_);
  max_num_bin_in_feature_ = 0;
  has_categorical_feature_ = false;
  max_num_categorical_bin_ = 0;
  is_categorical_.resize(train_data->num_features(), 0);
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(inner_feature_index);
    if (bin_mapper->bin_type() == BinType::CategoricalBin) {
      has_categorical_feature_ = true;
      is_categorical_[inner_feature_index] = 1;
      if (bin_mapper->num_bin() > max_num_categorical_bin_) {
        max_num_categorical_bin_ = bin_mapper->num_bin();
      }
    }
    const MissingType missing_type = bin_mapper->missing_type();
    feature_missing_type_[inner_feature_index] = missing_type;
    feature_mfb_offsets_[inner_feature_index] = static_cast<int8_t>(bin_mapper->GetMostFreqBin() == 0);
    feature_default_bins_[inner_feature_index] = bin_mapper->GetDefaultBin();
    feature_num_bins_[inner_feature_index] = static_cast<uint32_t>(bin_mapper->num_bin());
    const int num_bin_hist = bin_mapper->num_bin() - feature_mfb_offsets_[inner_feature_index];
    if (num_bin_hist > max_num_bin_in_feature_) {
      max_num_bin_in_feature_ = num_bin_hist;
    }
  }
  if (max_num_bin_in_feature_ > NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER) {
    use_global_memory_ = true;
  } else {
    use_global_memory_ = false;
  }
}

void CUDABestSplitFinder::Init() {
  InitCUDAFeatureMetaInfo();
  cuda_streams_.resize(2);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[1]));
  cuda_best_split_info_buffer_.Resize(8);
  if (use_global_memory_) {
    cuda_feature_hist_grad_buffer_.Resize(static_cast<size_t>(num_total_bin_));
    cuda_feature_hist_hess_buffer_.Resize(static_cast<size_t>(num_total_bin_));
    if (has_categorical_feature_) {
      cuda_feature_hist_stat_buffer_.Resize(static_cast<size_t>(num_total_bin_));
      cuda_feature_hist_index_buffer_.Resize(static_cast<size_t>(num_total_bin_));
    }
  }

  if (select_features_by_node_) {
    is_feature_used_by_smaller_node_.Resize(num_features_);
    is_feature_used_by_larger_node_.Resize(num_features_);
  }
}

void CUDABestSplitFinder::InitCUDAFeatureMetaInfo() {
  cuda_is_feature_used_bytree_.Resize(static_cast<size_t>(num_features_));

  // initialize split find task information (a split find task is one pass through the histogram of a feature)
  num_tasks_ = 0;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const uint32_t num_bin = feature_num_bins_[inner_feature_index];
    const MissingType missing_type = feature_missing_type_[inner_feature_index];
    if (num_bin > 2 && missing_type != MissingType::None && !is_categorical_[inner_feature_index]) {
      num_tasks_ += 2;
    } else {
      ++num_tasks_;
    }
  }
  split_find_tasks_.resize(num_tasks_);
  split_find_tasks_.shrink_to_fit();
  int cur_task_index = 0;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const uint32_t num_bin = feature_num_bins_[inner_feature_index];
    const MissingType missing_type = feature_missing_type_[inner_feature_index];
    if (num_bin > 2 && missing_type != MissingType::None && !is_categorical_[inner_feature_index]) {
      if (missing_type == MissingType::Zero) {
        SplitFindTask* new_task = &split_find_tasks_[cur_task_index];
        new_task->reverse = false;
        new_task->skip_default_bin = true;
        new_task->na_as_missing = false;
        new_task->inner_feature_index = inner_feature_index;
        new_task->assume_out_default_left = false;
        new_task->is_categorical = false;
        uint32_t num_bin = feature_num_bins_[inner_feature_index];
        new_task->is_one_hot = false;
        new_task->hist_offset = feature_hist_offsets_[inner_feature_index];
        new_task->mfb_offset = feature_mfb_offsets_[inner_feature_index];
        new_task->default_bin = feature_default_bins_[inner_feature_index];
        new_task->num_bin = num_bin;
        ++cur_task_index;

        new_task = &split_find_tasks_[cur_task_index];
        new_task->reverse = true;
        new_task->skip_default_bin = true;
        new_task->na_as_missing = false;
        new_task->inner_feature_index = inner_feature_index;
        new_task->assume_out_default_left = true;
        new_task->is_categorical = false;
        num_bin = feature_num_bins_[inner_feature_index];
        new_task->is_one_hot = false;
        new_task->hist_offset = feature_hist_offsets_[inner_feature_index];
        new_task->default_bin = feature_default_bins_[inner_feature_index];
        new_task->mfb_offset = feature_mfb_offsets_[inner_feature_index];
        new_task->num_bin = num_bin;
        ++cur_task_index;
      } else {
        SplitFindTask* new_task = &split_find_tasks_[cur_task_index];
        new_task->reverse = false;
        new_task->skip_default_bin = false;
        new_task->na_as_missing = true;
        new_task->inner_feature_index = inner_feature_index;
        new_task->assume_out_default_left = false;
        new_task->is_categorical = false;
        uint32_t num_bin = feature_num_bins_[inner_feature_index];
        new_task->is_one_hot = false;
        new_task->hist_offset = feature_hist_offsets_[inner_feature_index];
        new_task->mfb_offset = feature_mfb_offsets_[inner_feature_index];
        new_task->default_bin = feature_default_bins_[inner_feature_index];
        new_task->num_bin = num_bin;
        ++cur_task_index;

        new_task = &split_find_tasks_[cur_task_index];
        new_task->reverse = true;
        new_task->skip_default_bin = false;
        new_task->na_as_missing = true;
        new_task->inner_feature_index = inner_feature_index;
        new_task->assume_out_default_left = true;
        new_task->is_categorical = false;
        num_bin = feature_num_bins_[inner_feature_index];
        new_task->is_one_hot = false;
        new_task->hist_offset = feature_hist_offsets_[inner_feature_index];
        new_task->mfb_offset = feature_mfb_offsets_[inner_feature_index];
        new_task->default_bin = feature_default_bins_[inner_feature_index];
        new_task->num_bin = num_bin;
        ++cur_task_index;
      }
    } else {
      SplitFindTask& new_task = split_find_tasks_[cur_task_index];
      const uint32_t num_bin = feature_num_bins_[inner_feature_index];
      if (is_categorical_[inner_feature_index]) {
        new_task.reverse = false;
        new_task.is_categorical = true;
        new_task.is_one_hot = (static_cast<int>(num_bin) <= max_cat_to_onehot_);
      } else {
        new_task.reverse = true;
        new_task.is_categorical = false;
        new_task.is_one_hot = false;
      }
      new_task.skip_default_bin = false;
      new_task.na_as_missing = false;
      new_task.inner_feature_index = inner_feature_index;
      if (missing_type != MissingType::NaN && !is_categorical_[inner_feature_index]) {
        new_task.assume_out_default_left = true;
      } else {
        new_task.assume_out_default_left = false;
      }
      new_task.hist_offset = feature_hist_offsets_[inner_feature_index];
      new_task.mfb_offset = feature_mfb_offsets_[inner_feature_index];
      new_task.default_bin = feature_default_bins_[inner_feature_index];
      new_task.num_bin = num_bin;
      ++cur_task_index;
    }
  }
  CHECK_EQ(cur_task_index, static_cast<int>(split_find_tasks_.size()));

  if (extra_trees_) {
    cuda_randoms_.Resize(num_tasks_ * 2);
    LaunchInitCUDARandomKernel();
  }

  const int num_task_blocks = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  const size_t cuda_best_leaf_split_info_buffer_size = static_cast<size_t>(num_task_blocks) * static_cast<size_t>(num_leaves_);

  cuda_leaf_best_split_info_.Resize(cuda_best_leaf_split_info_buffer_size);

  cuda_split_find_tasks_.Resize(num_tasks_);
  CopyFromHostToCUDADevice<SplitFindTask>(cuda_split_find_tasks_.RawData(),
                                          split_find_tasks_.data(),
                                          split_find_tasks_.size(),
                                          __FILE__,
                                          __LINE__);

  const size_t output_buffer_size = 2 * static_cast<size_t>(num_tasks_);
  cuda_best_split_info_.Resize(output_buffer_size);

  max_num_categories_in_split_ = std::min(max_cat_threshold_, max_num_categorical_bin_ / 2);
  cuda_cat_threshold_feature_.Resize(max_num_categories_in_split_ * output_buffer_size);
  cuda_cat_threshold_real_feature_.Resize(max_num_categories_in_split_ * output_buffer_size);
  cuda_cat_threshold_leaf_.Resize(max_num_categories_in_split_ * cuda_best_leaf_split_info_buffer_size);
  cuda_cat_threshold_real_leaf_.Resize(max_num_categories_in_split_ * cuda_best_leaf_split_info_buffer_size);
  AllocateCatVectors(cuda_leaf_best_split_info_.RawData(), cuda_cat_threshold_leaf_.RawData(), cuda_cat_threshold_real_leaf_.RawData(), cuda_best_leaf_split_info_buffer_size);
  AllocateCatVectors(cuda_best_split_info_.RawData(), cuda_cat_threshold_feature_.RawData(), cuda_cat_threshold_real_feature_.RawData(), output_buffer_size);
}

void CUDABestSplitFinder::ResetTrainingData(
  const hist_t* cuda_hist,
  const Dataset* train_data,
  const std::vector<uint32_t>& feature_hist_offsets) {
  cuda_hist_ = cuda_hist;
  num_features_ = train_data->num_features();
  feature_hist_offsets_ = feature_hist_offsets;
  InitFeatureMetaInfo(train_data);
  cuda_is_feature_used_bytree_.Clear();
  cuda_best_split_info_.Clear();
  InitCUDAFeatureMetaInfo();
}

void CUDABestSplitFinder::ResetConfig(const Config* config, const hist_t* cuda_hist) {
  num_leaves_ = config->num_leaves;
  lambda_l1_ = config->lambda_l1;
  lambda_l2_ = config->lambda_l2;
  min_data_in_leaf_ = config->min_data_in_leaf;
  min_sum_hessian_in_leaf_ = config->min_sum_hessian_in_leaf;
  min_gain_to_split_ = config->min_gain_to_split;
  cat_smooth_ = config->cat_smooth;
  cat_l2_ = config->cat_l2;
  max_cat_threshold_ = config->max_cat_threshold;
  min_data_per_group_ = config->min_data_per_group;
  max_cat_to_onehot_ = config->max_cat_to_onehot;
  extra_trees_ = config->extra_trees;
  extra_seed_ = config->extra_seed;
  use_smoothing_ = (config->path_smooth > 0.0f);
  path_smooth_ = config->path_smooth;
  cuda_hist_ = cuda_hist;

  const int num_task_blocks = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  size_t cuda_best_leaf_split_info_buffer_size = static_cast<size_t>(num_task_blocks) * static_cast<size_t>(num_leaves_);
  cuda_leaf_best_split_info_.Resize(cuda_best_leaf_split_info_buffer_size);
  max_num_categories_in_split_ = std::min(max_cat_threshold_, max_num_categorical_bin_ / 2);
  size_t total_cat_threshold_size = max_num_categories_in_split_ * cuda_best_leaf_split_info_buffer_size;
  cuda_cat_threshold_leaf_.Resize(total_cat_threshold_size);
  cuda_cat_threshold_real_leaf_.Resize(total_cat_threshold_size);
  AllocateCatVectors(cuda_leaf_best_split_info_.RawData(), cuda_cat_threshold_leaf_.RawData(), cuda_cat_threshold_real_leaf_.RawData(), cuda_best_leaf_split_info_buffer_size);

  cuda_best_leaf_split_info_buffer_size = 2 * static_cast<size_t>(num_tasks_);
  total_cat_threshold_size = max_num_categories_in_split_ * cuda_best_leaf_split_info_buffer_size;
  cuda_cat_threshold_feature_.Resize(total_cat_threshold_size);
  cuda_cat_threshold_real_feature_.Resize(total_cat_threshold_size);
  AllocateCatVectors(cuda_best_split_info_.RawData(), cuda_cat_threshold_feature_.RawData(), cuda_cat_threshold_real_feature_.RawData(), cuda_best_leaf_split_info_buffer_size);
}

void CUDABestSplitFinder::BeforeTrain(const std::vector<int8_t>& is_feature_used_bytree) {
  CopyFromHostToCUDADevice<int8_t>(cuda_is_feature_used_bytree_.RawData(),
                                   is_feature_used_bytree.data(),
                                   is_feature_used_bytree.size(), __FILE__, __LINE__);
}

void CUDABestSplitFinder::FindBestSplitsForLeaf(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const CUDALeafSplitsStruct* larger_leaf_splits,
  const int smaller_leaf_index,
  const int larger_leaf_index,
  const data_size_t num_data_in_smaller_leaf,
  const data_size_t num_data_in_larger_leaf,
  const double sum_hessians_in_smaller_leaf,
  const double sum_hessians_in_larger_leaf,
  const score_t* grad_scale,
  const score_t* hess_scale,
  const uint8_t smaller_num_bits_in_histogram_bins,
  const uint8_t larger_num_bits_in_histogram_bins) {
  const bool is_smaller_leaf_valid = (num_data_in_smaller_leaf > min_data_in_leaf_ &&
    sum_hessians_in_smaller_leaf > min_sum_hessian_in_leaf_);
  const bool is_larger_leaf_valid = (num_data_in_larger_leaf > min_data_in_leaf_ &&
    sum_hessians_in_larger_leaf > min_sum_hessian_in_leaf_ && larger_leaf_index >= 0);
  if (grad_scale != nullptr && hess_scale != nullptr) {
    LaunchFindBestSplitsDiscretizedForLeafKernel(smaller_leaf_splits, larger_leaf_splits,
      smaller_leaf_index, larger_leaf_index, is_smaller_leaf_valid, is_larger_leaf_valid,
      grad_scale, hess_scale, smaller_num_bits_in_histogram_bins, larger_num_bits_in_histogram_bins, num_data_in_smaller_leaf, num_data_in_larger_leaf);
  } else {
    LaunchFindBestSplitsForLeafKernel(smaller_leaf_splits, larger_leaf_splits,
      smaller_leaf_index, larger_leaf_index, is_smaller_leaf_valid, is_larger_leaf_valid, num_data_in_smaller_leaf, num_data_in_larger_leaf);
  }
  global_timer.Start("CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel");
  LaunchSyncBestSplitForLeafKernel(smaller_leaf_index, larger_leaf_index, is_smaller_leaf_valid, is_larger_leaf_valid);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel");
}

const CUDASplitInfo* CUDABestSplitFinder::FindBestFromAllSplits(
    const int cur_num_leaves,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left,
    int* best_leaf_index,
    int* num_cat_threshold) {
  LaunchFindBestFromAllSplitsKernel(
    cur_num_leaves,
    smaller_leaf_index,
    larger_leaf_index,
    smaller_leaf_best_split_feature,
    smaller_leaf_best_split_threshold,
    smaller_leaf_best_split_default_left,
    larger_leaf_best_split_feature,
    larger_leaf_best_split_threshold,
    larger_leaf_best_split_default_left,
    best_leaf_index,
    num_cat_threshold);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  return cuda_leaf_best_split_info_.RawData() + (*best_leaf_index);
}

void CUDABestSplitFinder::AllocateCatVectors(CUDASplitInfo* cuda_split_infos, uint32_t* cat_threshold_vec, int* cat_threshold_real_vec, size_t len) {
  LaunchAllocateCatVectorsKernel(cuda_split_infos, cat_threshold_vec, cat_threshold_real_vec, len);
}

void CUDABestSplitFinder::SetUsedFeatureByNode(const std::vector<int8_t>& is_feature_used_by_smaller_node,
                                               const std::vector<int8_t>& is_feature_used_by_larger_node) {
  if (select_features_by_node_) {
    CopyFromHostToCUDADevice<int8_t>(is_feature_used_by_smaller_node_.RawData(),
                                     is_feature_used_by_smaller_node.data(), is_feature_used_by_smaller_node.size(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int8_t>(is_feature_used_by_larger_node_.RawData(),
                                     is_feature_used_by_larger_node.data(), is_feature_used_by_larger_node.size(), __FILE__, __LINE__);
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
