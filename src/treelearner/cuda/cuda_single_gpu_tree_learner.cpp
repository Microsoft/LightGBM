/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_single_gpu_tree_learner.hpp"

#include <LightGBM/cuda/cuda_tree.hpp>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/feature_group.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace LightGBM {

CUDASingleGPUTreeLearner::CUDASingleGPUTreeLearner(const Config* config, const bool boosting_on_cuda): SerialTreeLearner(config), boosting_on_cuda_(boosting_on_cuda) {}

CUDASingleGPUTreeLearner::~CUDASingleGPUTreeLearner() {
  if (nccl_communicator_ != nullptr) {
    CUDAStreamDestroy(nccl_stream_);
  }
}

void CUDASingleGPUTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();
  // use the first gpu by default
  if (nccl_communicator_ == nullptr) {
    gpu_device_id_ = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
  }
  cuda_smaller_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_smaller_leaf_splits_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_smaller_leaf_splits_->Init(config_->use_quantized_grad);
  cuda_larger_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_larger_leaf_splits_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_larger_leaf_splits_->Init(config_->use_quantized_grad);

  cuda_histogram_constructor_.reset(new CUDAHistogramConstructor(train_data_, config_->num_leaves, num_threads_,
    share_state_->feature_hist_offsets(),
    config_->min_data_in_leaf, config_->min_sum_hessian_in_leaf, gpu_device_id_, config_->gpu_use_dp,
    config_->use_quantized_grad, config_->num_grad_quant_bins));
  cuda_histogram_constructor_->Init(train_data_, share_state_.get());

  const auto& feature_hist_offsets = share_state_->feature_hist_offsets();
  num_total_bin_ = feature_hist_offsets.empty() ? 0 : static_cast<int>(feature_hist_offsets.back());
  cuda_data_partition_.reset(new CUDADataPartition(
    train_data_, num_total_bin_, config_->num_leaves, num_threads_, config_->use_quantized_grad,
    cuda_histogram_constructor_->cuda_hist_pointer()));
  cuda_data_partition_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_data_partition_->Init();

  select_features_by_node_ = !config_->interaction_constraints_vector.empty() || config_->feature_fraction_bynode < 1.0;
  cuda_best_split_finder_.reset(new CUDABestSplitFinder(cuda_histogram_constructor_->cuda_hist(),
    train_data_, this->share_state_->feature_hist_offsets(), select_features_by_node_, config_));
  cuda_best_split_finder_->Init();

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_gradients_.resize(config_->num_leaves, 0.0f);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);

  if (!boosting_on_cuda_) {
    cuda_gradients_.Resize(static_cast<size_t>(num_data_));
    cuda_hessians_.Resize(static_cast<size_t>(num_data_));
  }
  AllocateBitset();

  leaf_stat_buffer_size_ = 0;
  num_cat_threshold_ = 0;

  if (config_->use_quantized_grad) {
    cuda_leaf_gradient_stat_buffer_.Resize(config_->num_leaves);
    cuda_leaf_hessian_stat_buffer_.Resize(config_->num_leaves);
    cuda_gradient_discretizer_.reset(new CUDAGradientDiscretizer(
      config_->num_grad_quant_bins, config_->num_iterations, config_->seed, is_constant_hessian, config_->stochastic_rounding));
    cuda_gradient_discretizer_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
    cuda_gradient_discretizer_->Init(num_data_, config_->num_leaves, train_data_->num_features(), train_data_);
  } else {
    cuda_gradient_discretizer_.reset(nullptr);
  }

  #ifdef DEBUG
  host_gradients_.resize(num_data_, 0.0f);
  host_hessians_.resize(num_data_, 0.0f);
  #endif  // DEBUG
}

void CUDASingleGPUTreeLearner::BeforeTrain() {
  const data_size_t root_num_data = cuda_data_partition_->root_num_data();
  if (!boosting_on_cuda_) {
    CopyFromHostToCUDADevice<score_t>(cuda_gradients_.RawData(), gradients_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<score_t>(cuda_hessians_.RawData(), hessians_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    gradients_ = cuda_gradients_.RawData();
    hessians_ = cuda_hessians_.RawData();
  }

  #ifdef DEBUG
  CopyFromCUDADeviceToHost<score_t>(host_gradients_.data(), gradients_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<score_t>(host_hessians_.data(), hessians_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  #endif  // DEBUG

  const data_size_t* leaf_splits_init_indices =
    cuda_data_partition_->use_bagging() ? cuda_data_partition_->cuda_data_indices() : nullptr;
  cuda_data_partition_->BeforeTrain();
  if (config_->use_quantized_grad) {
    cuda_gradient_discretizer_->DiscretizeGradients(num_data_, gradients_, hessians_);
    cuda_histogram_constructor_->BeforeTrain(
      reinterpret_cast<const score_t*>(cuda_gradient_discretizer_->discretized_gradients_and_hessians()), nullptr);
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
      reinterpret_cast<const int16_t*>(cuda_gradient_discretizer_->discretized_gradients_and_hessians()),
      leaf_splits_init_indices,
      cuda_data_partition_->cuda_data_indices(),
      root_num_data,
      cuda_histogram_constructor_->cuda_hist_pointer(),
      &leaf_sum_gradients_[0],
      &leaf_sum_hessians_[0],
      cuda_gradient_discretizer_->grad_scale_ptr(),
      cuda_gradient_discretizer_->hess_scale_ptr());
      cuda_gradient_discretizer_->SetNumBitsInHistogramBin<false>(0, -1, root_num_data, 0);
      if (nccl_communicator_ != nullptr) {
        cuda_gradient_discretizer_->SetNumBitsInHistogramBin<true>(0, -1, global_num_data_, 0);
      }
  } else {
    cuda_histogram_constructor_->BeforeTrain(gradients_, hessians_);
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
      gradients_,
      hessians_,
      leaf_splits_init_indices,
      cuda_data_partition_->cuda_data_indices(),
      root_num_data,
      cuda_histogram_constructor_->cuda_hist_pointer(),
      &leaf_sum_gradients_[0],
      &leaf_sum_hessians_[0]);
  }
  leaf_num_data_[0] = root_num_data;
  cuda_larger_leaf_splits_->InitValues();
  col_sampler_.ResetByTree();
  cuda_best_split_finder_->BeforeTrain(col_sampler_.is_feature_used_bytree());
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;

  if (nccl_communicator_ != nullptr) {
    leaf_to_hist_index_map_.resize(config_->num_leaves, -1);
    leaf_to_hist_index_map_[0] = 0;
    global_num_data_in_leaf_[0] = global_num_data_;
  }
}

void CUDASingleGPUTreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  cuda_data_partition_->UpdateTrainScore(tree, out_score);
}

Tree* CUDASingleGPUTreeLearner::Train(const score_t* gradients,
  const score_t* hessians, bool /*is_first_tree*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  global_timer.Start("CUDASingleGPUTreeLearner::BeforeTrain");
  BeforeTrain();
  global_timer.Stop("CUDASingleGPUTreeLearner::BeforeTrain");
  const bool track_branch_features = !(config_->interaction_constraints_vector.empty());
  std::unique_ptr<CUDATree> tree(new CUDATree(config_->num_leaves, track_branch_features,
    config_->linear_tree, gpu_device_id_, has_categorical_feature_));
  // set the root value by hand, as it is not handled by splits
  tree->SetLeafOutput(0, CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(
    leaf_sum_gradients_[smaller_leaf_index_], leaf_sum_hessians_[smaller_leaf_index_],
    config_->lambda_l1, config_->lambda_l2,  config_->path_smooth,
    static_cast<data_size_t>(num_data_), 0));
  tree->SyncLeafOutputFromHostToCUDA();
  for (int i = 0; i < config_->num_leaves - 1; ++i) {
    global_timer.Start("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");
    const data_size_t global_num_data_in_smaller_leaf = nccl_communicator_ != nullptr ?
      global_num_data_in_leaf_[smaller_leaf_index_] :
      leaf_num_data_[smaller_leaf_index_];
    const data_size_t global_num_data_in_larger_leaf = nccl_communicator_ != nullptr ?
      (larger_leaf_index_ < 0 ? 0 : global_num_data_in_leaf_[larger_leaf_index_]) :
      (larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_]);
    const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hessians_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index_];
    const uint8_t num_bits_in_histogram_bins = config_->use_quantized_grad ? (nccl_communicator_ != nullptr ?
        cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(smaller_leaf_index_) :
        cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_index_)) : 0;
    cuda_histogram_constructor_->ConstructHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      global_num_data_in_smaller_leaf,
      global_num_data_in_larger_leaf,
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf,
      num_bits_in_histogram_bins);
    global_timer.Stop("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");

    global_timer.Start("CUDASingleGPUTreeLearner::NCCLReduceHistogram");
    if (nccl_communicator_ != nullptr) {
      NCCLReduceHistogram();
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::NCCLReduceHistogram");

    global_timer.Start("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
    uint8_t parent_num_bits_bin = 0;
    uint8_t smaller_num_bits_bin = 0;
    uint8_t larger_num_bits_bin = 0;
    if (config_->use_quantized_grad) {
      if (larger_leaf_index_ != -1) {
        const int parent_leaf_index = std::min(smaller_leaf_index_, larger_leaf_index_);
        if (nccl_communicator_ != nullptr) {
          parent_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInNode<true>(parent_leaf_index);
          smaller_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(smaller_leaf_index_);
          larger_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(larger_leaf_index_);
        } else {
          parent_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInNode<false>(parent_leaf_index);
          smaller_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_index_);
          larger_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_index_);
        }
      } else {
        if (nccl_communicator_ != nullptr) {
          parent_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(0);
          smaller_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(0);
          larger_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(0);
        } else {
          parent_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
          smaller_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
          larger_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
        }
      }
    } else {
      parent_num_bits_bin = 0;
      smaller_num_bits_bin = 0;
      larger_num_bits_bin = 0;
    }
    cuda_histogram_constructor_->SubtractHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      config_->use_quantized_grad,
      parent_num_bits_bin,
      smaller_num_bits_bin,
      larger_num_bits_bin);

    SelectFeatureByNode(tree.get());

    if (config_->use_quantized_grad) {
      const uint8_t smaller_leaf_num_bits_bin = nccl_communicator_ == nullptr ?
        cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_index_) :
        cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(smaller_leaf_index_);
      const uint8_t larger_leaf_num_bits_bin = larger_leaf_index_ < 0 ? 32 : (nccl_communicator_ == nullptr ?
        cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_index_) :
        cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(larger_leaf_index_));
      cuda_best_split_finder_->FindBestSplitsForLeaf(
        cuda_smaller_leaf_splits_->GetCUDAStruct(),
        cuda_larger_leaf_splits_->GetCUDAStruct(),
        smaller_leaf_index_, larger_leaf_index_,
        global_num_data_in_smaller_leaf, global_num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
        cuda_gradient_discretizer_->grad_scale_ptr(),
        cuda_gradient_discretizer_->hess_scale_ptr(),
        smaller_leaf_num_bits_bin,
        larger_leaf_num_bits_bin);
    } else {
      cuda_best_split_finder_->FindBestSplitsForLeaf(
        cuda_smaller_leaf_splits_->GetCUDAStruct(),
        cuda_larger_leaf_splits_->GetCUDAStruct(),
        smaller_leaf_index_, larger_leaf_index_,
        global_num_data_in_smaller_leaf, global_num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
        nullptr, nullptr, 0, 0);
    }

    global_timer.Stop("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
    global_timer.Start("CUDASingleGPUTreeLearner::FindBestFromAllSplits");
    const CUDASplitInfo* best_split_info = nullptr;
    if (larger_leaf_index_ >= 0) {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        &leaf_best_split_feature_[larger_leaf_index_],
        &leaf_best_split_threshold_[larger_leaf_index_],
        &leaf_best_split_default_left_[larger_leaf_index_],
        &best_leaf_index_,
        &num_cat_threshold_);
    } else {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        nullptr,
        nullptr,
        nullptr,
        &best_leaf_index_,
        &num_cat_threshold_);
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::FindBestFromAllSplits");

    if (best_leaf_index_ == -1) {
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }

    global_timer.Start("CUDASingleGPUTreeLearner::Split");
    if (num_cat_threshold_ > 0) {
      ConstructBitsetForCategoricalSplit(best_split_info);
    }

    int right_leaf_index = 0;
    if (train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->bin_type() == BinType::CategoricalBin) {
      right_leaf_index = tree->SplitCategorical(best_leaf_index_,
                                       train_data_->RealFeatureIndex(leaf_best_split_feature_[best_leaf_index_]),
                                       train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->missing_type(),
                                       best_split_info,
                                       cuda_bitset_,
                                       cuda_bitset_len_,
                                       cuda_bitset_inner_,
                                       cuda_bitset_inner_len_);
    } else {
      right_leaf_index = tree->Split(best_leaf_index_,
                                       train_data_->RealFeatureIndex(leaf_best_split_feature_[best_leaf_index_]),
                                       train_data_->RealThreshold(leaf_best_split_feature_[best_leaf_index_],
                                        leaf_best_split_threshold_[best_leaf_index_]),
                                       train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->missing_type(),
                                       best_split_info);
    }

    cuda_data_partition_->Split(best_split_info,
                                best_leaf_index_,
                                right_leaf_index,
                                leaf_best_split_feature_[best_leaf_index_],
                                leaf_best_split_threshold_[best_leaf_index_],
                                cuda_bitset_inner_,
                                static_cast<int>(cuda_bitset_inner_len_),
                                leaf_best_split_default_left_[best_leaf_index_],
                                leaf_num_data_[best_leaf_index_],
                                leaf_data_start_[best_leaf_index_],
                                cuda_smaller_leaf_splits_->GetCUDAStructRef(),
                                cuda_larger_leaf_splits_->GetCUDAStructRef(),
                                &leaf_num_data_[best_leaf_index_],
                                &leaf_num_data_[right_leaf_index],
                                &leaf_data_start_[best_leaf_index_],
                                &leaf_data_start_[right_leaf_index],
                                &leaf_sum_hessians_[best_leaf_index_],
                                &leaf_sum_hessians_[right_leaf_index],
                                &leaf_sum_gradients_[best_leaf_index_],
                                &leaf_sum_gradients_[right_leaf_index],
                                global_num_data_in_leaf_.data() + best_leaf_index_,
                                global_num_data_in_leaf_.data() + right_leaf_index);
    #ifdef DEBUG
    CheckSplitValid(best_leaf_index_, right_leaf_index);
    #endif  // DEBUG

    if (nccl_communicator_ != nullptr) {
      smaller_leaf_index_ = (global_num_data_in_leaf_[best_leaf_index_] < global_num_data_in_leaf_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
      larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);
      const int best_leaf_hist_index = leaf_to_hist_index_map_[best_leaf_index_];
      leaf_to_hist_index_map_[smaller_leaf_index_] = right_leaf_index;
      leaf_to_hist_index_map_[larger_leaf_index_] = best_leaf_hist_index;
    } else {
      smaller_leaf_index_ = (leaf_num_data_[best_leaf_index_] < leaf_num_data_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
      larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);
    }

    if (config_->use_quantized_grad) {
      cuda_gradient_discretizer_->SetNumBitsInHistogramBin<false>(
        best_leaf_index_, right_leaf_index, leaf_num_data_[best_leaf_index_], leaf_num_data_[right_leaf_index]);
      if (nccl_communicator_ != nullptr) {
        cuda_gradient_discretizer_->SetNumBitsInHistogramBin<true>(
          best_leaf_index_, right_leaf_index, global_num_data_in_leaf_[best_leaf_index_], global_num_data_in_leaf_[right_leaf_index]);
      }
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::Split");
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (config_->use_quantized_grad && config_->quant_train_renew_leaf) {
    global_timer.Start("CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves");
    RenewDiscretizedTreeLeaves(tree.get());
    global_timer.Stop("CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves");
  }
  tree->ToHost();
  return tree.release();
}

void CUDASingleGPUTreeLearner::ResetTrainingData(
  const Dataset* train_data,
  bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  CHECK_EQ(num_features_, train_data_->num_features());
  cuda_histogram_constructor_->ResetTrainingData(train_data, share_state_.get());
  cuda_data_partition_->ResetTrainingData(train_data,
    static_cast<int>(share_state_->feature_hist_offsets().back()),
    cuda_histogram_constructor_->cuda_hist_pointer());
  cuda_best_split_finder_->ResetTrainingData(
    cuda_histogram_constructor_->cuda_hist(),
    train_data,
    share_state_->feature_hist_offsets());
  cuda_smaller_leaf_splits_->Resize(num_data_);
  cuda_larger_leaf_splits_->Resize(num_data_);
  CHECK_EQ(is_constant_hessian, share_state_->is_constant_hessian);
  if (!boosting_on_cuda_) {
    cuda_gradients_.Resize(static_cast<size_t>(num_data_));
    cuda_hessians_.Resize(static_cast<size_t>(num_data_));
  }
}

void CUDASingleGPUTreeLearner::ResetConfig(const Config* config) {
  const int old_num_leaves = config_->num_leaves;
  SerialTreeLearner::ResetConfig(config);
  if (config_->gpu_device_id >= 0 && config_->gpu_device_id != gpu_device_id_) {
    Log::Fatal("Changing gpu device ID by resetting configuration parameter is not allowed for CUDA tree learner.");
  }
  num_threads_ = OMP_NUM_THREADS();
  if (config_->num_leaves != old_num_leaves) {
    leaf_best_split_feature_.resize(config_->num_leaves, -1);
    leaf_best_split_threshold_.resize(config_->num_leaves, 0);
    leaf_best_split_default_left_.resize(config_->num_leaves, 0);
    leaf_num_data_.resize(config_->num_leaves, 0);
    leaf_data_start_.resize(config_->num_leaves, 0);
    leaf_sum_gradients_.resize(config_->num_leaves, 0.0f);
    leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);
  }
  cuda_histogram_constructor_->ResetConfig(config);
  cuda_best_split_finder_->ResetConfig(config, cuda_histogram_constructor_->cuda_hist());
  cuda_data_partition_->ResetConfig(config, cuda_histogram_constructor_->cuda_hist_pointer());
}

void CUDASingleGPUTreeLearner::SetBaggingData(const Dataset* /*subset*/,
  const data_size_t* used_indices, data_size_t num_data) {
  cuda_data_partition_->SetUsedDataIndices(used_indices, num_data);
}

void CUDASingleGPUTreeLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                                               data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt, const double* train_score) const {
  CHECK(tree->is_cuda_tree());
  CUDATree* cuda_tree = reinterpret_cast<CUDATree*>(tree);
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    CHECK_LE(cuda_tree->num_leaves(), data_partition_->num_leaves());
    if (boosting_on_cuda_) {
      obj->RenewTreeOutputCUDA(train_score, cuda_data_partition_->cuda_data_indices(),
                               cuda_data_partition_->cuda_leaf_num_data(), cuda_data_partition_->cuda_leaf_data_start(),
                               cuda_tree->num_leaves(), cuda_tree->cuda_leaf_value_ref());
      cuda_tree->SyncLeafOutputFromCUDAToHost();
    } else {
      const data_size_t* bag_mapper = nullptr;
      if (total_num_data != num_data_) {
        CHECK_EQ(bag_cnt, num_data_);
        bag_mapper = bag_indices;
      }
      std::vector<int> n_nozeroworker_perleaf(cuda_tree->num_leaves(), 1);
      int num_machines = Network::num_machines();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int i = 0; i < cuda_tree->num_leaves(); ++i) {
        const double output = static_cast<double>(cuda_tree->LeafOutput(i));
        data_size_t cnt_leaf_data = leaf_num_data_[i];
        std::vector<data_size_t> index_mapper(cnt_leaf_data, -1);
        CopyFromCUDADeviceToHost<data_size_t>(index_mapper.data(),
          cuda_data_partition_->cuda_data_indices() + leaf_data_start_[i],
          static_cast<size_t>(cnt_leaf_data), __FILE__, __LINE__);
        if (cnt_leaf_data > 0) {
          const double new_output = obj->RenewTreeOutput(output, residual_getter, index_mapper.data(), bag_mapper, cnt_leaf_data);
          cuda_tree->SetLeafOutput(i, new_output);
        } else {
          CHECK_GT(num_machines, 1);
          cuda_tree->SetLeafOutput(i, 0.0);
          n_nozeroworker_perleaf[i] = 0;
        }
      }
      if (num_machines > 1) {
        std::vector<double> outputs(cuda_tree->num_leaves());
        for (int i = 0; i < cuda_tree->num_leaves(); ++i) {
          outputs[i] = static_cast<double>(cuda_tree->LeafOutput(i));
        }
        outputs = Network::GlobalSum(&outputs);
        n_nozeroworker_perleaf = Network::GlobalSum(&n_nozeroworker_perleaf);
        for (int i = 0; i < cuda_tree->num_leaves(); ++i) {
          cuda_tree->SetLeafOutput(i, outputs[i] / n_nozeroworker_perleaf[i]);
        }
      }
    }
    cuda_tree->SyncLeafOutputFromHostToCUDA();
  }
}

Tree* CUDASingleGPUTreeLearner::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const {
  std::unique_ptr<CUDATree> cuda_tree(new CUDATree(old_tree));
  cuda_leaf_gradient_stat_buffer_.SetValue(0);
  cuda_leaf_hessian_stat_buffer_.SetValue(0);
  ReduceLeafStat(cuda_tree.get(), gradients, hessians, cuda_data_partition_->cuda_data_indices());
  cuda_tree->SyncLeafOutputFromCUDAToHost();
  return cuda_tree.release();
}

Tree* CUDASingleGPUTreeLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                                  const score_t* gradients, const score_t* hessians) const {
  cuda_data_partition_->ResetByLeafPred(leaf_pred, old_tree->num_leaves());
  refit_num_data_ = static_cast<data_size_t>(leaf_pred.size());
  data_size_t buffer_size = static_cast<data_size_t>(old_tree->num_leaves());
  if (old_tree->num_leaves() > 2048) {
    const int num_block = (refit_num_data_ + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
    buffer_size *= static_cast<data_size_t>(num_block + 1);
  }
  if (static_cast<size_t>(buffer_size) > cuda_leaf_gradient_stat_buffer_.Size()) {
    cuda_leaf_gradient_stat_buffer_.Resize(buffer_size);
    cuda_leaf_hessian_stat_buffer_.Resize(buffer_size);
  }
  return FitByExistingTree(old_tree, gradients, hessians);
}

void CUDASingleGPUTreeLearner::ReduceLeafStat(
  CUDATree* old_tree, const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf) const {
  LaunchReduceLeafStatKernel(gradients, hessians, num_data_in_leaf, old_tree->cuda_leaf_parent(),
    old_tree->cuda_left_child(), old_tree->cuda_right_child(),
    old_tree->num_leaves(), refit_num_data_, old_tree->cuda_leaf_value_ref(), old_tree->shrinkage());
}

void CUDASingleGPUTreeLearner::ConstructBitsetForCategoricalSplit(
  const CUDASplitInfo* best_split_info) {
  LaunchConstructBitsetForCategoricalSplitKernel(best_split_info);
}

void CUDASingleGPUTreeLearner::AllocateBitset() {
  has_categorical_feature_ = false;
  categorical_bin_offsets_.clear();
  categorical_bin_offsets_.push_back(0);
  categorical_bin_to_value_.clear();
  for (int i = 0; i < train_data_->num_features(); ++i) {
    const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
    if (bin_mapper->bin_type() == BinType::CategoricalBin) {
      has_categorical_feature_ = true;
      break;
    }
  }
  if (has_categorical_feature_) {
    int max_cat_value = 0;
    int max_cat_num_bin = 0;
    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        max_cat_value = std::max(bin_mapper->MaxCatValue(), max_cat_value);
        max_cat_num_bin = std::max(bin_mapper->num_bin(), max_cat_num_bin);
      }
    }
    // std::max(..., 1UL) to avoid error in the case when there are NaN's in the categorical values
    const size_t cuda_bitset_max_size = std::max(static_cast<size_t>((max_cat_value + 31) / 32), 1UL);
    const size_t cuda_bitset_inner_max_size = std::max(static_cast<size_t>((max_cat_num_bin + 31) / 32), 1UL);
    AllocateCUDAMemory<uint32_t>(&cuda_bitset_, cuda_bitset_max_size, __FILE__, __LINE__);
    AllocateCUDAMemory<uint32_t>(&cuda_bitset_inner_, cuda_bitset_inner_max_size, __FILE__, __LINE__);
    const int max_cat_in_split = std::min(config_->max_cat_threshold, max_cat_num_bin / 2);
    const int num_blocks = (max_cat_in_split + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
    AllocateCUDAMemory<size_t>(&cuda_block_bitset_len_buffer_, num_blocks, __FILE__, __LINE__);

    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        categorical_bin_offsets_.push_back(bin_mapper->num_bin());
      } else {
        categorical_bin_offsets_.push_back(0);
      }
    }
    for (size_t i = 1; i < categorical_bin_offsets_.size(); ++i) {
      categorical_bin_offsets_[i] += categorical_bin_offsets_[i - 1];
    }
    categorical_bin_to_value_.resize(categorical_bin_offsets_.back(), 0);
    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        const int offset = categorical_bin_offsets_[i];
        for (int bin = 0; bin < bin_mapper->num_bin(); ++bin) {
          categorical_bin_to_value_[offset + bin] = static_cast<int>(bin_mapper->BinToValue(bin));
        }
      }
    }
    InitCUDAMemoryFromHostMemory<int>(&cuda_categorical_bin_offsets_, categorical_bin_offsets_.data(), categorical_bin_offsets_.size(), __FILE__, __LINE__);
    InitCUDAMemoryFromHostMemory<int>(&cuda_categorical_bin_to_value_, categorical_bin_to_value_.data(), categorical_bin_to_value_.size(), __FILE__, __LINE__);
  } else {
    cuda_bitset_ = nullptr;
    cuda_bitset_inner_ = nullptr;
  }
  cuda_bitset_len_ = 0;
  cuda_bitset_inner_len_ = 0;
}

void CUDASingleGPUTreeLearner::ResetBoostingOnGPU(const bool boosting_on_cuda) {
  boosting_on_cuda_ = boosting_on_cuda;
  cuda_gradients_.Clear();
  cuda_hessians_.Clear();
  if (!boosting_on_cuda_) {
    cuda_gradients_.Resize(static_cast<size_t>(num_data_));
    cuda_hessians_.Resize(static_cast<size_t>(num_data_));
  }
}

void CUDASingleGPUTreeLearner::SelectFeatureByNode(const Tree* tree) {
  if (select_features_by_node_) {
    // use feature interaction constraint or sample features by node
    const std::vector<int8_t>& is_feature_used_by_smaller_node = col_sampler_.GetByNode(tree, smaller_leaf_index_);
    std::vector<int8_t> is_feature_used_by_larger_node;
    if (larger_leaf_index_ >= 0) {
      is_feature_used_by_larger_node = col_sampler_.GetByNode(tree, larger_leaf_index_);
    }
    cuda_best_split_finder_->SetUsedFeatureByNode(is_feature_used_by_smaller_node, is_feature_used_by_larger_node);
  }
}

#ifdef DEBUG
void CUDASingleGPUTreeLearner::CheckSplitValid(
  const int left_leaf,
  const int right_leaf) {
  std::vector<data_size_t> left_data_indices(leaf_num_data_[left_leaf]);
  std::vector<data_size_t> right_data_indices(leaf_num_data_[right_leaf]);
  CopyFromCUDADeviceToHost<data_size_t>(left_data_indices.data(),
    cuda_data_partition_->cuda_data_indices() + leaf_data_start_[left_leaf],
    leaf_num_data_[left_leaf], __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(right_data_indices.data(),
    cuda_data_partition_->cuda_data_indices() + leaf_data_start_[right_leaf],
    leaf_num_data_[right_leaf], __FILE__, __LINE__);
  double sum_left_gradients = 0.0f, sum_left_hessians = 0.0f;
  double sum_right_gradients = 0.0f, sum_right_hessians = 0.0f;
  for (size_t i = 0; i < left_data_indices.size(); ++i) {
    const data_size_t index = left_data_indices[i];
    sum_left_gradients += host_gradients_[index];
    sum_left_hessians += host_hessians_[index];
  }
  for (size_t i = 0; i < right_data_indices.size(); ++i) {
    const data_size_t index = right_data_indices[i];
    sum_right_gradients += host_gradients_[index];
    sum_right_hessians += host_hessians_[index];
  }
  if (nccl_communicator_ != nullptr) {
    sum_left_gradients = NCCLAllReduce<double>(sum_left_gradients, ncclFloat64, ncclSum, nccl_communicator_);
    sum_left_hessians = NCCLAllReduce<double>(sum_left_hessians, ncclFloat64, ncclSum, nccl_communicator_);
    sum_right_gradients = NCCLAllReduce<double>(sum_right_gradients, ncclFloat64, ncclSum, nccl_communicator_);
    sum_right_hessians = NCCLAllReduce<double>(sum_right_hessians, ncclFloat64, ncclSum, nccl_communicator_);
  }
  CHECK_LE(std::fabs(sum_left_gradients - leaf_sum_gradients_[left_leaf]), 1e-6f);
  CHECK_LE(std::fabs(sum_left_hessians - leaf_sum_hessians_[left_leaf]), 1e-6f);
  CHECK_LE(std::fabs(sum_right_gradients - leaf_sum_gradients_[right_leaf]), 1e-6f);
  CHECK_LE(std::fabs(sum_right_hessians - leaf_sum_hessians_[right_leaf]), 1e-6f);
}
#endif  // DEBUG

void CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves(CUDATree* cuda_tree) {
  cuda_data_partition_->ReduceLeafGradStat(
    gradients_, hessians_, cuda_tree,
    cuda_leaf_gradient_stat_buffer_.RawData(),
    cuda_leaf_hessian_stat_buffer_.RawData());
  LaunchCalcLeafValuesGivenGradStat(cuda_tree, cuda_data_partition_->cuda_data_indices());
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDASingleGPUTreeLearner::SetNCCLInfo(
  ncclComm_t nccl_communicator,
  int nccl_gpu_rank,
  int local_gpu_rank,
  int gpu_device_id,
  data_size_t global_num_data) {
  NCCLInfo::SetNCCLInfo(nccl_communicator, nccl_gpu_rank, local_gpu_rank, gpu_device_id, global_num_data);
  leaf_to_hist_index_map_.resize(config_->num_leaves - 1);
  global_num_data_in_leaf_.resize(config_->num_leaves, 0);
  nccl_stream_ = CUDAStreamCreate();
}

void CUDASingleGPUTreeLearner::NCCLReduceHistogram() {
  if (config_->use_quantized_grad) {
    hist_t* smaller_leaf_hist_pointer = cuda_histogram_constructor_->cuda_hist_pointer() +
      leaf_to_hist_index_map_[smaller_leaf_index_] * num_total_bin_;
    const int8_t bit_size = cuda_gradient_discretizer_->GetHistBitsInNode<true>(smaller_leaf_index_);
    if (bit_size == 32) {
      NCCLAllReduce<int64_t>(
        reinterpret_cast<const int64_t*>(smaller_leaf_hist_pointer),
        reinterpret_cast<int64_t*>(smaller_leaf_hist_pointer),
        static_cast<size_t>(num_total_bin_),
        ncclInt64,
        ncclSum,
        nccl_communicator_,
        nccl_stream_);
    } else if (bit_size <= 16) {
      NCCLAllReduce<int32_t>(
        reinterpret_cast<const int32_t*>(smaller_leaf_hist_pointer),
        reinterpret_cast<int32_t*>(smaller_leaf_hist_pointer),
        static_cast<size_t>(num_total_bin_),
        ncclInt32,
        ncclSum,
        nccl_communicator_,
        nccl_stream_);
    }
  } else {
    hist_t* smaller_leaf_hist_pointer = cuda_histogram_constructor_->cuda_hist_pointer() +
        leaf_to_hist_index_map_[smaller_leaf_index_] * num_total_bin_ * 2;
    NCCLAllReduce<hist_t>(
      smaller_leaf_hist_pointer,
      smaller_leaf_hist_pointer,
      static_cast<size_t>(num_total_bin_) * 2,
      ncclFloat64,
      ncclSum,
      nccl_communicator_,
      nccl_stream_);
  }
  SynchronizeCUDAStream(nccl_stream_, __FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
