/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <algorithm>
#include <vector>

namespace LightGBM {

CUDAHistogramConstructor::CUDAHistogramConstructor(
  const Dataset* train_data,
  const int num_leaves,
  const int num_threads,
  const std::vector<uint32_t>& feature_hist_offsets,
  const int min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const int gpu_device_id,
  const bool gpu_use_dp,
  const bool use_quantized_grad,
  const int num_grad_quant_bins):
  num_data_(train_data->num_data()),
  num_features_(train_data->num_features()),
  num_leaves_(num_leaves),
  num_threads_(num_threads),
  min_data_in_leaf_(min_data_in_leaf),
  min_sum_hessian_in_leaf_(min_sum_hessian_in_leaf),
  gpu_device_id_(gpu_device_id),
  gpu_use_dp_(gpu_use_dp),
  use_quantized_grad_(use_quantized_grad),
  num_grad_quant_bins_(num_grad_quant_bins) {
  InitFeatureMetaInfo(train_data, feature_hist_offsets);
  cuda_row_data_.reset(nullptr);
}

CUDAHistogramConstructor::~CUDAHistogramConstructor() {
  gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
}

void CUDAHistogramConstructor::InitFeatureMetaInfo(const Dataset* train_data, const std::vector<uint32_t>& feature_hist_offsets) {
  need_fix_histogram_features_.clear();
  need_fix_histogram_features_num_bin_aligend_.clear();
  feature_num_bins_.clear();
  feature_most_freq_bins_.clear();
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    const uint32_t most_freq_bin = bin_mapper->GetMostFreqBin();
    if (most_freq_bin != 0) {
      need_fix_histogram_features_.emplace_back(feature_index);
      uint32_t num_bin_ref = static_cast<uint32_t>(bin_mapper->num_bin()) - 1;
      uint32_t num_bin_aligned = 1;
      while (num_bin_ref > 0) {
        num_bin_aligned <<= 1;
        num_bin_ref >>= 1;
      }
      need_fix_histogram_features_num_bin_aligend_.emplace_back(num_bin_aligned);
    }
    feature_num_bins_.emplace_back(static_cast<uint32_t>(bin_mapper->num_bin()));
    feature_most_freq_bins_.emplace_back(most_freq_bin);
  }
  feature_hist_offsets_.clear();
  for (size_t i = 0; i < feature_hist_offsets.size(); ++i) {
    feature_hist_offsets_.emplace_back(feature_hist_offsets[i]);
  }
  if (feature_hist_offsets.empty()) {
    num_total_bin_ = 0;
  } else {
    num_total_bin_ = static_cast<int>(feature_hist_offsets.back());
  }
}

void CUDAHistogramConstructor::BeforeTrain(const score_t* gradients, const score_t* hessians) {
  cuda_gradients_ = gradients;
  cuda_hessians_ = hessians;
  cuda_hist_.SetValue(0);
}

void CUDAHistogramConstructor::Init(const Dataset* train_data, TrainingShareStates* share_state) {
  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_));
  cuda_hist_.SetValue(0);

  cuda_feature_num_bins_.InitFromHostVector(feature_num_bins_);
  cuda_feature_hist_offsets_.InitFromHostVector(feature_hist_offsets_);
  cuda_feature_most_freq_bins_.InitFromHostVector(feature_most_freq_bins_);

  cuda_row_data_.reset(new CUDARowData(train_data, share_state, gpu_device_id_, gpu_use_dp_));
  cuda_row_data_->Init(train_data, share_state);

  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));

  cuda_need_fix_histogram_features_.InitFromHostVector(need_fix_histogram_features_);
  cuda_need_fix_histogram_features_num_bin_aligned_.InitFromHostVector(need_fix_histogram_features_num_bin_aligend_);

  if (cuda_row_data_->NumLargeBinPartition() > 0) {
    int grid_dim_x = 0, grid_dim_y = 0, block_dim_x = 0, block_dim_y = 0;
    CalcConstructHistogramKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, num_data_);
    const size_t buffer_size = static_cast<size_t>(grid_dim_y) * static_cast<size_t>(num_total_bin_);
    if (!use_quantized_grad_) {
      if (gpu_use_dp_) {
        // need to double the size of histogram buffer in global memory when using double precision in histogram construction
        cuda_hist_buffer_.Resize(buffer_size * 4);
      } else {
        cuda_hist_buffer_.Resize(buffer_size * 2);
      }
    } else {
      // use only half the size of histogram buffer in global memory when quantized training since each gradient and hessian takes only 2 bytes
      cuda_hist_buffer_.Resize(buffer_size);
    }
  }
  hist_buffer_for_num_bit_change_.Resize(num_total_bin_ * 2);
}

void CUDAHistogramConstructor::ConstructHistogramForLeaf(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* /*cuda_larger_leaf_splits*/,
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  const data_size_t num_data_in_smaller_leaf,
  const data_size_t /*num_data_in_larger_leaf*/,
  const double sum_hessians_in_smaller_leaf,
  const double sum_hessians_in_larger_leaf,
  const uint8_t num_bits_in_histogram_bins) {
if ((global_num_data_in_smaller_leaf <= min_data_in_leaf_ || sum_hessians_in_smaller_leaf <= min_sum_hessian_in_leaf_) &&
    (global_num_data_in_larger_leaf <= min_data_in_leaf_ || sum_hessians_in_larger_leaf <= min_sum_hessian_in_leaf_)) {
    return;
  }
  LaunchConstructHistogramKernel(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAHistogramConstructor::SubtractHistogramForLeaf(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool use_quantized_grad,
  const uint8_t parent_num_bits_in_histogram_bins,
  const uint8_t smaller_num_bits_in_histogram_bins,
  const uint8_t larger_num_bits_in_histogram_bins) {
  global_timer.Start("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
  LaunchSubtractHistogramKernel(cuda_smaller_leaf_splits, cuda_larger_leaf_splits, use_quantized_grad,
                                parent_num_bits_in_histogram_bins, smaller_num_bits_in_histogram_bins, larger_num_bits_in_histogram_bins);
  global_timer.Stop("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
}

void CUDAHistogramConstructor::CalcConstructHistogramKernelDim(
  int* grid_dim_x,
  int* grid_dim_y,
  int* block_dim_x,
  int* block_dim_y,
  const data_size_t num_data_in_smaller_leaf) {
  *block_dim_x = cuda_row_data_->max_num_column_per_partition();
  *block_dim_y = NUM_THREADS_PER_BLOCK / cuda_row_data_->max_num_column_per_partition();
  *grid_dim_x = cuda_row_data_->num_feature_partitions();
  *grid_dim_y = std::max(min_grid_dim_y_,
    ((num_data_in_smaller_leaf + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + (*block_dim_y) - 1) / (*block_dim_y));
}

void CUDAHistogramConstructor::ResetTrainingData(const Dataset* train_data, TrainingShareStates* share_states) {
  num_data_ = train_data->num_data();
  num_features_ = train_data->num_features();
  InitFeatureMetaInfo(train_data, share_states->feature_hist_offsets());

  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_));
  cuda_hist_.SetValue(0);
  cuda_feature_num_bins_.InitFromHostVector(feature_num_bins_);
  cuda_feature_hist_offsets_.InitFromHostVector(feature_hist_offsets_);
  cuda_feature_most_freq_bins_.InitFromHostVector(feature_most_freq_bins_);

  cuda_row_data_.reset(new CUDARowData(train_data, share_states, gpu_device_id_, gpu_use_dp_));
  cuda_row_data_->Init(train_data, share_states);

  cuda_need_fix_histogram_features_.InitFromHostVector(need_fix_histogram_features_);
  cuda_need_fix_histogram_features_num_bin_aligned_.InitFromHostVector(need_fix_histogram_features_num_bin_aligend_);
}

void CUDAHistogramConstructor::ResetConfig(const Config* config) {
  num_threads_ = OMP_NUM_THREADS();
  num_leaves_ = config->num_leaves;
  min_data_in_leaf_ = config->min_data_in_leaf;
  min_sum_hessian_in_leaf_ = config->min_sum_hessian_in_leaf;
  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_));
  cuda_hist_.SetValue(0);
}

}  // namespace LightGBM

#endif  // USE_CUDA
