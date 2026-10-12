/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "serial_tree_learner.h"

#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/common.h>

#include <algorithm>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cost_effective_gradient_boosting.hpp"

namespace LightGBM {

SerialTreeLearner::SerialTreeLearner(const Config* config)
    : config_(config), col_sampler_(config) {
  gradient_discretizer_ = nullptr;
}

SerialTreeLearner::~SerialTreeLearner() {
}

void SerialTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();
  int max_cache_size = 0;
  // Get the max size of pool
  if (config_->histogram_pool_size <= 0) {
    max_cache_size = config_->num_leaves;
  } else {
    size_t total_histogram_size = 0;
    for (int i = 0; i < train_data_->num_features(); ++i) {
      total_histogram_size += kHistEntrySize * train_data_->FeatureNumBin(i);
    }
    max_cache_size = static_cast<int>(config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
  }
  // at least need 2 leaves
  max_cache_size = std::max(2, max_cache_size);
  max_cache_size = std::min(max_cache_size, config_->num_leaves);

  // push split information for all leaves
  best_split_per_leaf_.resize(config_->num_leaves);
  constraints_.reset(LeafConstraintsBase::Create(config_, config_->num_leaves, train_data_->num_features()));

  // initialize splits for leaf
  smaller_leaf_splits_.reset(new LeafSplits(train_data_->num_data(), config_));
  larger_leaf_splits_.reset(new LeafSplits(train_data_->num_data(), config_));

  // initialize data partition
  data_partition_.reset(new DataPartition(num_data_, config_->num_leaves));
  col_sampler_.SetTrainingData(train_data_);
  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);

  if (config_->use_quantized_grad) {
    gradient_discretizer_.reset(new GradientDiscretizer(config_->num_grad_quant_bins, config_->num_iterations, config_->seed, is_constant_hessian, config_->stochastic_rounding));
    gradient_discretizer_->Init(num_data_, config_->num_leaves, num_features_, train_data_);
  }

  GetShareStates(train_data_, is_constant_hessian, true);
  histogram_pool_.DynamicChangeSize(train_data_,
  share_state_->num_hist_total_bin(),
  share_state_->feature_hist_offsets(),
  config_, max_cache_size, config_->num_leaves);
  Log::Info("Number of data points in the train set: %d, number of used features: %d", num_data_, num_features_);
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    cegb_.reset(new CostEfficientGradientBoosting(this));
    cegb_->Init();
  }
}

void SerialTreeLearner::GetShareStates(const Dataset* dataset,
                                       bool is_constant_hessian,
                                       bool is_first_time) {
  if (is_first_time) {
    if (config_->use_quantized_grad) {
      share_state_.reset(dataset->GetShareStates<true, 32>(
        reinterpret_cast<score_t*>(gradient_discretizer_->ordered_int_gradients_and_hessians()), nullptr,
        col_sampler_.is_feature_used_bytree(), is_constant_hessian,
        config_->force_col_wise, config_->force_row_wise, config_->num_grad_quant_bins));
    } else {
      share_state_.reset(dataset->GetShareStates<false, 0>(
          ordered_gradients_.data(), ordered_hessians_.data(),
          col_sampler_.is_feature_used_bytree(), is_constant_hessian,
          config_->force_col_wise, config_->force_row_wise, config_->num_grad_quant_bins));
    }
  } else {
    CHECK_NOTNULL(share_state_);
    // cannot change is_hist_col_wise during training
    if (config_->use_quantized_grad) {
      share_state_.reset(dataset->GetShareStates<true, 32>(
          reinterpret_cast<score_t*>(gradient_discretizer_->ordered_int_gradients_and_hessians()), nullptr,
          col_sampler_.is_feature_used_bytree(), is_constant_hessian,
          share_state_->is_col_wise, !share_state_->is_col_wise, config_->num_grad_quant_bins));
    } else {
      share_state_.reset(dataset->GetShareStates<false, 0>(
          ordered_gradients_.data(), ordered_hessians_.data(), col_sampler_.is_feature_used_bytree(),
          is_constant_hessian, share_state_->is_col_wise,
          !share_state_->is_col_wise, config_->num_grad_quant_bins));
    }
  }
  CHECK_NOTNULL(share_state_);
}

void SerialTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
                                               bool is_constant_hessian,
                                               bool reset_multi_val_bin) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  CHECK_EQ(num_features_, train_data_->num_features());

  // initialize splits for leaf
  smaller_leaf_splits_->ResetNumData(num_data_);
  larger_leaf_splits_->ResetNumData(num_data_);

  // initialize data partition
  data_partition_->ResetNumData(num_data_);
  if (reset_multi_val_bin) {
    col_sampler_.SetTrainingData(train_data_);
    GetShareStates(train_data_, is_constant_hessian, false);
  }

  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  if (cegb_ != nullptr) {
    cegb_->Init();
  }
}

void SerialTreeLearner::ResetConfig(const Config* config) {
  if (config_->num_leaves != config->num_leaves) {
    config_ = config;
    int max_cache_size = 0;
    // Get the max size of pool
    if (config->histogram_pool_size <= 0) {
      max_cache_size = config_->num_leaves;
    } else {
      size_t total_histogram_size = 0;
      for (int i = 0; i < train_data_->num_features(); ++i) {
        total_histogram_size += kHistEntrySize * train_data_->FeatureNumBin(i);
      }
      max_cache_size = static_cast<int>(config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
    }
    // at least need 2 leaves
    max_cache_size = std::max(2, max_cache_size);
    max_cache_size = std::min(max_cache_size, config_->num_leaves);
    histogram_pool_.DynamicChangeSize(train_data_,
    share_state_->num_hist_total_bin(),
    share_state_->feature_hist_offsets(),
    config_, max_cache_size, config_->num_leaves);

    // push split information for all leaves
    best_split_per_leaf_.resize(config_->num_leaves);
    data_partition_->ResetLeaves(config_->num_leaves);
  } else {
    config_ = config;
  }
  col_sampler_.SetConfig(config_);
  histogram_pool_.ResetConfig(train_data_, config_);
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    if (cegb_ == nullptr) {
      cegb_.reset(new CostEfficientGradientBoosting(this));
    }
    cegb_->Init();
  }
  constraints_.reset(LeafConstraintsBase::Create(config_, config_->num_leaves, train_data_->num_features()));
}

Tree* SerialTreeLearner::Train(const score_t* gradients, const score_t *hessians, bool /*is_first_tree*/) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::Train", global_timer);
  gradients_ = gradients;
  hessians_ = hessians;
  int num_threads = OMP_NUM_THREADS();
  if (share_state_->num_threads != num_threads && share_state_->num_threads > 0) {
    Log::Warning(
        "Detected that num_threads changed during training (from %d to %d), "
        "it may cause unexpected errors.",
        share_state_->num_threads, num_threads);
  }
  share_state_->num_threads = num_threads;

  if (config_->use_quantized_grad) {
    gradient_discretizer_->DiscretizeGradients(num_data_, gradients_, hessians_);
  }

  // some initial works before training
  BeforeTrain();

  bool track_branch_features = !(config_->interaction_constraints_vector.empty());
  auto tree = std::unique_ptr<Tree>(new Tree(config_->num_leaves, track_branch_features, false));
  auto tree_ptr = tree.get();
  constraints_->ShareTreePointer(tree_ptr);

  // set the root value by hand, as it is not handled by splits
  tree->SetLeafOutput(0, FeatureHistogram::CalculateSplittedLeafOutput<true, true, true, false>(
    smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
    config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
    BasicConstraint(), config_->path_smooth, static_cast<data_size_t>(num_data_), 0));

  // root leaf
  int left_leaf = 0;
  int cur_depth = 1;
  // only root leaf can be splitted on first time
  int right_leaf = -1;

  int init_splits = ForceSplits(tree_ptr, &left_leaf, &right_leaf, &cur_depth);

  for (int split = init_splits; split < config_->num_leaves - 1; ++split) {
    // some initial works before finding best split
    if (BeforeFindBestSplit(tree_ptr, left_leaf, right_leaf)) {
      // find best threshold for every feature
      FindBestSplits(tree_ptr);
    }
    // Get a leaf with max split gain
    int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
    // Get split information for best leaf
    const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
    // cannot split, quit
    if (best_leaf_SplitInfo.gain <= 0.0) {
      Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_SplitInfo.gain);
      break;
    }
    // split tree with best leaf
    Split(tree_ptr, best_leaf, &left_leaf, &right_leaf);
    cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));
  }

  if (config_->use_quantized_grad && config_->quant_train_renew_leaf) {
    gradient_discretizer_->RenewIntGradTreeOutput(tree.get(), config_, data_partition_.get(), gradients_, hessians_,
      [this] (int leaf_index) { return GetGlobalDataCountInLeaf(leaf_index); });
  }

  Log::Debug("Trained a tree with leaves = %d and depth = %d", tree->num_leaves(), cur_depth);
  return tree.release();
}

Tree* SerialTreeLearner::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t *hessians) const {
  auto tree = std::unique_ptr<Tree>(new Tree(*old_tree));
  CHECK_GE(data_partition_->num_leaves(), tree->num_leaves());
  OMP_INIT_EX();
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int i = 0; i < tree->num_leaves(); ++i) {
    OMP_LOOP_EX_BEGIN();
    data_size_t cnt_leaf_data = 0;
    auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
    double sum_grad = 0.0f;
    double sum_hess = kEpsilon;
    for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
      auto idx = tmp_idx[j];
      sum_grad += gradients[idx];
      sum_hess += hessians[idx];
    }
    double output;
    if ((config_->path_smooth > kEpsilon) & (i > 0)) {
      output = FeatureHistogram::CalculateSplittedLeafOutput<true, true, true>(
          sum_grad, sum_hess, config_->lambda_l1, config_->lambda_l2,
          config_->max_delta_step, config_->path_smooth, cnt_leaf_data, tree->leaf_parent(i));
    } else {
      output = FeatureHistogram::CalculateSplittedLeafOutput<true, true, false>(
          sum_grad, sum_hess, config_->lambda_l1, config_->lambda_l2,
          config_->max_delta_step, config_->path_smooth, cnt_leaf_data, 0);
    }
    auto old_leaf_output = tree->LeafOutput(i);
    auto new_leaf_output = output * tree->shrinkage();
    tree->SetLeafOutput(i, config_->refit_decay_rate * old_leaf_output + (1.0 - config_->refit_decay_rate) * new_leaf_output);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  return tree.release();
}

Tree* SerialTreeLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                           const score_t* gradients, const score_t *hessians) const {
  data_partition_->ResetByLeafPred(leaf_pred, old_tree->num_leaves());
  return FitByExistingTree(old_tree, gradients, hessians);
}

void SerialTreeLearner::BeforeTrain() {
  Common::FunctionTimer fun_timer("SerialTreeLearner::BeforeTrain", global_timer);
  // reset histogram pool
  histogram_pool_.ResetMap();

  col_sampler_.ResetByTree();
  train_data_->InitTrain(col_sampler_.is_feature_used_bytree(), share_state_.get());
  // initialize data partition
  data_partition_->Init();

  constraints_->Reset();

  // reset the splits for leaves
  for (int i = 0; i < config_->num_leaves; ++i) {
    best_split_per_leaf_[i].Reset();
  }

  // Sumup for root
  if (data_partition_->leaf_count(0) == num_data_) {
    // use all data
    if (!config_->use_quantized_grad) {
      smaller_leaf_splits_->Init(gradients_, hessians_);
    } else {
      smaller_leaf_splits_->Init(
        gradient_discretizer_->discretized_gradients_and_hessians(),
        gradient_discretizer_->grad_scale(),
        gradient_discretizer_->hess_scale());
    }
  } else {
    // use bagging, only use part of data
    if (!config_->use_quantized_grad) {
      smaller_leaf_splits_->Init(0, data_partition_.get(), gradients_, hessians_);
    } else {
      smaller_leaf_splits_->Init(
        0, data_partition_.get(),
        gradient_discretizer_->discretized_gradients_and_hessians(),
        static_cast<score_t>(gradient_discretizer_->grad_scale()),
        static_cast<score_t>(gradient_discretizer_->hess_scale()));
    }
  }

  larger_leaf_splits_->Init();

  if (cegb_ != nullptr) {
    cegb_->BeforeTrain();
  }

  if (config_->use_quantized_grad && config_->tree_learner != std::string("data")) {
    gradient_discretizer_->SetNumBitsInHistogramBin<false>(0, -1, data_partition_->leaf_count(0), 0);
  }
}

bool SerialTreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::BeforeFindBestSplit", global_timer);
  // check depth of current leaf
  if (config_->max_depth > 0) {
    // only need to check left leaf, since right leaf is in same level of left leaf
    if (tree->leaf_depth(left_leaf) >= config_->max_depth) {
      best_split_per_leaf_[left_leaf].gain = kMinScore;
      if (right_leaf >= 0) {
        best_split_per_leaf_[right_leaf].gain = kMinScore;
      }
      return false;
    }
  }
  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
  // no enough data to continue
  if (num_data_in_right_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)
      && num_data_in_left_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)) {
    best_split_per_leaf_[left_leaf].gain = kMinScore;
    if (right_leaf >= 0) {
      best_split_per_leaf_[right_leaf].gain = kMinScore;
    }
    return false;
  }
  parent_leaf_histogram_array_ = nullptr;
  // only have root
  if (right_leaf < 0) {
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
    larger_leaf_histogram_array_ = nullptr;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    // put parent(left) leaf's histograms into larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) {
      parent_leaf_histogram_array_ = larger_leaf_histogram_array_;
    }
    histogram_pool_.Move(left_leaf, right_leaf);
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
  } else {
    // put parent(left) leaf's histograms to larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) {
      parent_leaf_histogram_array_ = larger_leaf_histogram_array_;
    }
    histogram_pool_.Get(right_leaf, &smaller_leaf_histogram_array_);
  }
  return true;
}

void SerialTreeLearner::FindBestSplits(const Tree* tree) {
  FindBestSplits(tree, nullptr);
}

void SerialTreeLearner::FindBestSplits(const Tree* tree, const std::set<int>* force_features) {
  std::vector<int8_t> is_feature_used(num_features_, 0);
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 256) if (num_features_ >= 512)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index] && (force_features == nullptr || force_features->find(feature_index) == force_features->end())) continue;
    if (parent_leaf_histogram_array_ != nullptr
        && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
  }
  bool use_subtract = parent_leaf_histogram_array_ != nullptr;

  ConstructHistograms(is_feature_used, use_subtract);
  FindBestSplitsFromHistograms(is_feature_used, use_subtract, tree);
}

void SerialTreeLearner::ConstructHistograms(
    const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::ConstructHistograms",
                                  global_timer);
  // construct smaller leaf
  if (config_->use_quantized_grad) {
    const uint8_t smaller_leaf_num_bits = gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_splits_->leaf_index());
    hist_t* ptr_smaller_leaf_hist_data =
        smaller_leaf_num_bits <= 16 ?
        reinterpret_cast<hist_t*>(smaller_leaf_histogram_array_[0].RawDataInt16() - kHistOffset) :
        reinterpret_cast<hist_t*>(smaller_leaf_histogram_array_[0].RawDataInt32() - kHistOffset);
    #define SMALLER_LEAF_ARGS \
      is_feature_used, smaller_leaf_splits_->data_indices(), \
      smaller_leaf_splits_->num_data_in_leaf(), \
      reinterpret_cast<const score_t*>(gradient_discretizer_->discretized_gradients_and_hessians()), \
      nullptr, \
      reinterpret_cast<score_t*>(gradient_discretizer_->ordered_int_gradients_and_hessians()), \
      nullptr, \
      share_state_.get(), \
      reinterpret_cast<hist_t*>(ptr_smaller_leaf_hist_data)
    if (smaller_leaf_num_bits <= 16) {
      train_data_->ConstructHistograms<true, 16>(SMALLER_LEAF_ARGS);
    } else {
      train_data_->ConstructHistograms<true, 32>(SMALLER_LEAF_ARGS);
    }
    #undef SMALLER_LEAF_ARGS
    if (larger_leaf_histogram_array_ && !use_subtract) {
      const uint8_t larger_leaf_num_bits = gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_splits_->leaf_index());
      hist_t* ptr_larger_leaf_hist_data =
        larger_leaf_num_bits <= 16 ?
        reinterpret_cast<hist_t*>(larger_leaf_histogram_array_[0].RawDataInt16() - kHistOffset) :
        reinterpret_cast<hist_t*>(larger_leaf_histogram_array_[0].RawDataInt32() - kHistOffset);
      #define LARGER_LEAF_ARGS \
        is_feature_used, larger_leaf_splits_->data_indices(), \
        larger_leaf_splits_->num_data_in_leaf(), \
        reinterpret_cast<const score_t*>(gradient_discretizer_->discretized_gradients_and_hessians()), \
        nullptr, \
        reinterpret_cast<score_t*>(gradient_discretizer_->ordered_int_gradients_and_hessians()), \
        nullptr, \
        share_state_.get(), \
        reinterpret_cast<hist_t*>(ptr_larger_leaf_hist_data)
      if (larger_leaf_num_bits <= 16) {
        train_data_->ConstructHistograms<true, 16>(LARGER_LEAF_ARGS);
      } else {
        train_data_->ConstructHistograms<true, 32>(LARGER_LEAF_ARGS);
      }
      #undef LARGER_LEAF_ARGS
    }
  } else {
    hist_t* ptr_smaller_leaf_hist_data =
        smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
    train_data_->ConstructHistograms<false, 0>(
        is_feature_used, smaller_leaf_splits_->data_indices(),
        smaller_leaf_splits_->num_data_in_leaf(), gradients_, hessians_,
        ordered_gradients_.data(), ordered_hessians_.data(), share_state_.get(),
        ptr_smaller_leaf_hist_data);
    if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
      // construct larger leaf
      hist_t* ptr_larger_leaf_hist_data =
          larger_leaf_histogram_array_[0].RawData() - kHistOffset;
      train_data_->ConstructHistograms<false, 0>(
          is_feature_used, larger_leaf_splits_->data_indices(),
          larger_leaf_splits_->num_data_in_leaf(), gradients_, hessians_,
          ordered_gradients_.data(), ordered_hessians_.data(), share_state_.get(),
          ptr_larger_leaf_hist_data);
    }
  }
}

void SerialTreeLearner::FindBestSplitsFromHistograms(
    const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree) {
  Common::FunctionTimer fun_timer(
      "SerialTreeLearner::FindBestSplitsFromHistograms", global_timer);
  std::vector<SplitInfo> smaller_best(share_state_->num_threads);
  std::vector<SplitInfo> larger_best(share_state_->num_threads);
  std::vector<int8_t> smaller_node_used_features = col_sampler_.GetByNode(tree, smaller_leaf_splits_->leaf_index());
  std::vector<int8_t> larger_node_used_features;
  double smaller_leaf_parent_output = GetParentOutput(tree, smaller_leaf_splits_.get());
  double larger_leaf_parent_output = 0;
  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->leaf_index() >= 0) {
    larger_leaf_parent_output = GetParentOutput(tree, larger_leaf_splits_.get());
  }
  if (larger_leaf_splits_->leaf_index() >= 0) {
    larger_node_used_features = col_sampler_.GetByNode(tree, larger_leaf_splits_->leaf_index());
  }

  if (use_subtract && config_->use_quantized_grad) {
    const int parent_index = std::min(smaller_leaf_splits_->leaf_index(), larger_leaf_splits_->leaf_index());
    const uint8_t parent_hist_bits = gradient_discretizer_->GetHistBitsInNode<false>(parent_index);
    const uint8_t larger_hist_bits = gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_splits_->leaf_index());
    if (parent_hist_bits > 16 && larger_hist_bits <= 16) {
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) num_threads(share_state_->num_threads)
      for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
        OMP_LOOP_EX_BEGIN();
        if (!is_feature_used[feature_index]) {
          continue;
        }
        larger_leaf_histogram_array_[feature_index].CopyToBuffer(gradient_discretizer_->GetChangeHistBitsBuffer(feature_index));
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    }
  }

  OMP_INIT_EX();
// find splits
#pragma omp parallel for schedule(static) num_threads(share_state_->num_threads)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_used[feature_index]) {
      continue;
    }
    const int tid = omp_get_thread_num();
    if (config_->use_quantized_grad) {
      const uint8_t hist_bits_bin = gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_splits_->leaf_index());
      const int64_t int_sum_gradient_and_hessian = smaller_leaf_splits_->int_sum_gradients_and_hessians();
      if (hist_bits_bin <= 16) {
        train_data_->FixHistogramInt<int32_t, int32_t, 16, 16>(
            feature_index, int_sum_gradient_and_hessian,
            reinterpret_cast<hist_t*>(smaller_leaf_histogram_array_[feature_index].RawDataInt16()));
      } else {
        train_data_->FixHistogramInt<int64_t, int64_t, 32, 32>(
            feature_index, int_sum_gradient_and_hessian,
            reinterpret_cast<hist_t*>(smaller_leaf_histogram_array_[feature_index].RawDataInt32()));
      }
    } else {
      train_data_->FixHistogram(
          feature_index, smaller_leaf_splits_->sum_gradients(),
          smaller_leaf_splits_->sum_hessians(),
          smaller_leaf_histogram_array_[feature_index].RawData());
    }
    int real_fidx = train_data_->RealFeatureIndex(feature_index);

    ComputeBestSplitForFeature(smaller_leaf_histogram_array_, feature_index,
                               real_fidx,
                               smaller_node_used_features[feature_index],
                               smaller_leaf_splits_->num_data_in_leaf(),
                               smaller_leaf_splits_.get(), &smaller_best[tid],
                               smaller_leaf_parent_output);

    // only has root leaf
    if (larger_leaf_splits_ == nullptr ||
        larger_leaf_splits_->leaf_index() < 0) {
      continue;
    }

    if (use_subtract) {
      if (config_->use_quantized_grad) {
        const int parent_index = std::min(smaller_leaf_splits_->leaf_index(), larger_leaf_splits_->leaf_index());
        const uint8_t parent_hist_bits = gradient_discretizer_->GetHistBitsInNode<false>(parent_index);
        const uint8_t smaller_hist_bits = gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_splits_->leaf_index());
        const uint8_t larger_hist_bits = gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_splits_->leaf_index());
        if (parent_hist_bits <= 16) {
          CHECK_LE(smaller_hist_bits, 16);
          CHECK_LE(larger_hist_bits, 16);
          larger_leaf_histogram_array_[feature_index].Subtract<true, int32_t, int32_t, int32_t, 16, 16, 16>(
              smaller_leaf_histogram_array_[feature_index]);
        } else if (larger_hist_bits <= 16) {
          CHECK_LE(smaller_hist_bits, 16);
          larger_leaf_histogram_array_[feature_index].Subtract<true, int64_t, int32_t, int32_t, 32, 16, 16>(
              smaller_leaf_histogram_array_[feature_index], gradient_discretizer_->GetChangeHistBitsBuffer(feature_index));
        } else if (smaller_hist_bits <= 16) {
          larger_leaf_histogram_array_[feature_index].Subtract<true, int64_t, int32_t, int64_t, 32, 16, 32>(
              smaller_leaf_histogram_array_[feature_index]);
        } else {
          larger_leaf_histogram_array_[feature_index].Subtract<true, int64_t, int64_t, int64_t, 32, 32, 32>(
              smaller_leaf_histogram_array_[feature_index]);
        }
      } else {
        larger_leaf_histogram_array_[feature_index].Subtract<false>(
            smaller_leaf_histogram_array_[feature_index]);
      }
    } else {
      if (config_->use_quantized_grad) {
        const int64_t int_sum_gradient_and_hessian = larger_leaf_splits_->int_sum_gradients_and_hessians();
        const uint8_t hist_bits_bin = gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_splits_->leaf_index());
        if (hist_bits_bin <= 16) {
          train_data_->FixHistogramInt<int32_t, int32_t, 16, 16>(
              feature_index, int_sum_gradient_and_hessian,
              reinterpret_cast<hist_t*>(larger_leaf_histogram_array_[feature_index].RawDataInt16()));
        } else {
          train_data_->FixHistogramInt<int64_t, int64_t, 32, 32>(
              feature_index, int_sum_gradient_and_hessian,
              reinterpret_cast<hist_t*>(larger_leaf_histogram_array_[feature_index].RawDataInt32()));
        }
      } else {
        train_data_->FixHistogram(
            feature_index, larger_leaf_splits_->sum_gradients(),
            larger_leaf_splits_->sum_hessians(),
            larger_leaf_histogram_array_[feature_index].RawData());
      }
    }

    ComputeBestSplitForFeature(larger_leaf_histogram_array_, feature_index,
                               real_fidx,
                               larger_node_used_features[feature_index],
                               larger_leaf_splits_->num_data_in_leaf(),
                               larger_leaf_splits_.get(), &larger_best[tid],
                               larger_leaf_parent_output);

    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_best);
  int leaf = smaller_leaf_splits_->leaf_index();
  best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx];

  if (larger_leaf_splits_ != nullptr &&
      larger_leaf_splits_->leaf_index() >= 0) {
    leaf = larger_leaf_splits_->leaf_index();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_best);
    best_split_per_leaf_[leaf] = larger_best[larger_best_idx];
  }
}

int32_t SerialTreeLearner::ForceSplits(Tree* tree, int* left_leaf,
                                       int* right_leaf, int *cur_depth) {
  bool abort_last_forced_split = false;
  if (forced_split_json_ == nullptr) {
    return 0;
  }
  int32_t result_count = 0;
  // start at root leaf
  *left_leaf = 0;
  std::queue<std::pair<Json, int>> q;
  Json left = *forced_split_json_;
  Json right;
  bool left_smaller = true;
  std::unordered_map<int, SplitInfo> forceSplitMap;
  q.push(std::make_pair(left, *left_leaf));

  // Histogram construction require parent features.
  std::set<int> force_split_features = FindAllForceFeatures(*forced_split_json_);
  while (!q.empty()) {
    if (BeforeFindBestSplit(tree, *left_leaf, *right_leaf)) {
      FindBestSplits(tree, &force_split_features);
    }

    // then, compute own splits
    SplitInfo left_split;
    SplitInfo right_split;

    if (!left.is_null()) {
      const int left_feature = left["feature"].int_value();
      const double left_threshold_double = left["threshold"].number_value();
      const int left_inner_feature_index = train_data_->InnerFeatureIndex(left_feature);
      const uint32_t left_threshold = train_data_->BinThreshold(
              left_inner_feature_index, left_threshold_double);
      auto leaf_histogram_array = (left_smaller) ? smaller_leaf_histogram_array_ : larger_leaf_histogram_array_;
      auto left_leaf_splits = (left_smaller) ? smaller_leaf_splits_.get() : larger_leaf_splits_.get();
      leaf_histogram_array[left_inner_feature_index].GatherInfoForThreshold(
              left_leaf_splits->sum_gradients(),
              left_leaf_splits->sum_hessians(),
              left_threshold,
              left_leaf_splits->num_data_in_leaf(),
              left_leaf_splits->weight(),
              &left_split);
      left_split.feature = left_feature;
      forceSplitMap[*left_leaf] = left_split;
      if (left_split.gain < 0) {
        forceSplitMap.erase(*left_leaf);
      }
    }

    if (!right.is_null()) {
      const int right_feature = right["feature"].int_value();
      const double right_threshold_double = right["threshold"].number_value();
      const int right_inner_feature_index = train_data_->InnerFeatureIndex(right_feature);
      const uint32_t right_threshold = train_data_->BinThreshold(
              right_inner_feature_index, right_threshold_double);
      auto leaf_histogram_array = (left_smaller) ? larger_leaf_histogram_array_ : smaller_leaf_histogram_array_;
      auto right_leaf_splits = (left_smaller) ? larger_leaf_splits_.get() : smaller_leaf_splits_.get();
      leaf_histogram_array[right_inner_feature_index].GatherInfoForThreshold(
        right_leaf_splits->sum_gradients(),
        right_leaf_splits->sum_hessians(),
        right_threshold,
        right_leaf_splits->num_data_in_leaf(),
        right_leaf_splits->weight(),
        &right_split);
      right_split.feature = right_feature;
      forceSplitMap[*right_leaf] = right_split;
      if (right_split.gain < 0) {
        forceSplitMap.erase(*right_leaf);
      }
    }

    std::pair<Json, int> pair = q.front();
    q.pop();
    int current_leaf = pair.second;
    // split info should exist because searching in bfs fashion - should have added from parent
    if (forceSplitMap.find(current_leaf) == forceSplitMap.end()) {
        abort_last_forced_split = true;
        break;
    }
    best_split_per_leaf_[current_leaf] = forceSplitMap[current_leaf];
    Split(tree, current_leaf, left_leaf, right_leaf);
    left_smaller = best_split_per_leaf_[current_leaf].left_count <
                   best_split_per_leaf_[current_leaf].right_count;
    left = Json();
    right = Json();
    if ((pair.first).object_items().count("left") > 0) {
      left = (pair.first)["left"];
      if (left.object_items().count("feature") > 0 && left.object_items().count("threshold") > 0) {
        q.push(std::make_pair(left, *left_leaf));
      }
    }
    if ((pair.first).object_items().count("right") > 0) {
      right = (pair.first)["right"];
      if (right.object_items().count("feature") > 0 && right.object_items().count("threshold") > 0) {
        q.push(std::make_pair(right, *right_leaf));
      }
    }
    result_count++;
    *(cur_depth) = std::max(*(cur_depth), tree->leaf_depth(*left_leaf));
  }
  if (abort_last_forced_split) {
    int best_leaf =
        static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
    const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
    if (best_leaf_SplitInfo.gain <= 0.0) {
      Log::Warning("No further splits with positive gain, best gain: %f",
                   best_leaf_SplitInfo.gain);
      return config_->num_leaves;
    }
    Split(tree, best_leaf, left_leaf, right_leaf);
    *(cur_depth) = std::max(*(cur_depth), tree->leaf_depth(*left_leaf));
    ++result_count;
  }
  return result_count;
}

std::set<int> SerialTreeLearner::FindAllForceFeatures(Json force_split_leaf_setting) {
  std::set<int> force_features;
  std::queue<Json> force_split_leaves;

  force_split_leaves.push(force_split_leaf_setting);

  while (!force_split_leaves.empty()) {
    Json split_leaf = force_split_leaves.front();
    force_split_leaves.pop();

    const int feature_index = split_leaf["feature"].int_value();
    const int feature_inner_index = train_data_->InnerFeatureIndex(feature_index);
    force_features.insert(feature_inner_index);

    if (split_leaf.object_items().count("left") > 0) {
      force_split_leaves.push(split_leaf["left"]);
    }

    if (split_leaf.object_items().count("right") > 0) {
      force_split_leaves.push(split_leaf["right"]);
    }
  }

  return force_features;
}

void SerialTreeLearner::SplitInner(Tree* tree, int best_leaf, int* left_leaf,
                                   int* right_leaf, bool update_cnt) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::SplitInner", global_timer);
  SplitInfo& best_split_info = best_split_per_leaf_[best_leaf];
  const int inner_feature_index =
      train_data_->InnerFeatureIndex(best_split_info.feature);
  if (cegb_ != nullptr) {
    cegb_->UpdateLeafBestSplits(tree, best_leaf, &best_split_info,
                                &best_split_per_leaf_);
  }
  *left_leaf = best_leaf;
  auto next_leaf_id = tree->NextLeafId();

  // update before tree split
  constraints_->BeforeSplit(best_leaf, next_leaf_id,
                            best_split_info.monotone_type);

  bool is_numerical_split =
      train_data_->FeatureBinMapper(inner_feature_index)->bin_type() ==
      BinType::NumericalBin;
  if (is_numerical_split) {
    auto threshold_double = train_data_->RealThreshold(
        inner_feature_index, best_split_info.threshold);
    data_partition_->Split(best_leaf, train_data_, inner_feature_index,
                           &best_split_info.threshold, 1,
                           best_split_info.default_left, next_leaf_id);
    if (update_cnt) {
      // don't need to update this in data-based parallel model
      best_split_info.left_count = data_partition_->leaf_count(*left_leaf);
      best_split_info.right_count = data_partition_->leaf_count(next_leaf_id);
    }
    // split tree, will return right leaf
    *right_leaf = tree->Split(
        best_leaf, inner_feature_index, best_split_info.feature,
        best_split_info.threshold, threshold_double,
        static_cast<double>(best_split_info.left_output),
        static_cast<double>(best_split_info.right_output),
        static_cast<data_size_t>(best_split_info.left_count),
        static_cast<data_size_t>(best_split_info.right_count),
        static_cast<double>(best_split_info.left_sum_hessian),
        static_cast<double>(best_split_info.right_sum_hessian),
        // store the true split gain in tree model
        static_cast<float>(best_split_info.gain + config_->min_gain_to_split),
        train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
        best_split_info.default_left);
  } else {
    std::vector<uint32_t> cat_bitset_inner =
        Common::ConstructBitset(best_split_info.cat_threshold.data(),
                                best_split_info.num_cat_threshold);
    std::vector<int> threshold_int(best_split_info.num_cat_threshold);
    for (int i = 0; i < best_split_info.num_cat_threshold; ++i) {
      threshold_int[i] = static_cast<int>(train_data_->RealThreshold(
          inner_feature_index, best_split_info.cat_threshold[i]));
    }
    std::vector<uint32_t> cat_bitset = Common::ConstructBitset(
        threshold_int.data(), best_split_info.num_cat_threshold);

    data_partition_->Split(best_leaf, train_data_, inner_feature_index,
                           cat_bitset_inner.data(),
                           static_cast<int>(cat_bitset_inner.size()),
                           best_split_info.default_left, next_leaf_id);

    if (update_cnt) {
      // don't need to update this in data-based parallel model
      best_split_info.left_count = data_partition_->leaf_count(*left_leaf);
      best_split_info.right_count = data_partition_->leaf_count(next_leaf_id);
    }

    *right_leaf = tree->SplitCategorical(
        best_leaf, inner_feature_index, best_split_info.feature,
        cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()),
        cat_bitset.data(), static_cast<int>(cat_bitset.size()),
        static_cast<double>(best_split_info.left_output),
        static_cast<double>(best_split_info.right_output),
        static_cast<data_size_t>(best_split_info.left_count),
        static_cast<data_size_t>(best_split_info.right_count),
        static_cast<double>(best_split_info.left_sum_hessian),
        static_cast<double>(best_split_info.right_sum_hessian),
        // store the true split gain in tree model
        static_cast<float>(best_split_info.gain + config_->min_gain_to_split),
        train_data_->FeatureBinMapper(inner_feature_index)->missing_type());
  }

#ifdef DEBUG
  CHECK(*right_leaf == next_leaf_id);
#endif

  // init the leaves that used on next iteration
  if (!config_->use_quantized_grad) {
    if (best_split_info.left_count < best_split_info.right_count) {
      CHECK_GT(best_split_info.left_count, 0);
      smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                 best_split_info.left_sum_gradient,
                                 best_split_info.left_sum_hessian,
                                 best_split_info.left_output);
      larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                best_split_info.right_sum_gradient,
                                best_split_info.right_sum_hessian,
                                best_split_info.right_output);
    } else {
      CHECK_GT(best_split_info.right_count, 0);
      smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                 best_split_info.right_sum_gradient,
                                 best_split_info.right_sum_hessian,
                                 best_split_info.right_output);
      larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                best_split_info.left_sum_gradient,
                                best_split_info.left_sum_hessian,
                                best_split_info.left_output);
    }
  } else {
    if (best_split_info.left_count < best_split_info.right_count) {
      CHECK_GT(best_split_info.left_count, 0);
      smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                 best_split_info.left_sum_gradient,
                                 best_split_info.left_sum_hessian,
                                 best_split_info.left_sum_gradient_and_hessian,
                                 best_split_info.left_output);
      larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                 best_split_info.right_sum_gradient,
                                 best_split_info.right_sum_hessian,
                                 best_split_info.right_sum_gradient_and_hessian,
                                 best_split_info.right_output);
    } else {
      CHECK_GT(best_split_info.right_count, 0);
      smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                 best_split_info.right_sum_gradient,
                                 best_split_info.right_sum_hessian,
                                 best_split_info.right_sum_gradient_and_hessian,
                                 best_split_info.right_output);
      larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                best_split_info.left_sum_gradient,
                                best_split_info.left_sum_hessian,
                                best_split_info.left_sum_gradient_and_hessian,
                                best_split_info.left_output);
    }
  }
  if (config_->use_quantized_grad && config_->tree_learner != std::string("data")) {
    gradient_discretizer_->SetNumBitsInHistogramBin<false>(*left_leaf, *right_leaf,
                                                    data_partition_->leaf_count(*left_leaf),
                                                    data_partition_->leaf_count(*right_leaf));
  }

  #ifdef DEBUG
  CheckSplit(best_split_info, *left_leaf, *right_leaf);
  #endif

  auto leaves_need_update = constraints_->Update(
      is_numerical_split, *left_leaf, *right_leaf,
      best_split_info.monotone_type, best_split_info.right_output,
      best_split_info.left_output, inner_feature_index, best_split_info,
      best_split_per_leaf_);
  // update leave outputs if needed
  for (auto leaf : leaves_need_update) {
    RecomputeBestSplitForLeaf(tree, leaf, &best_split_per_leaf_[leaf]);
  }
}

void SerialTreeLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                                        data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt, const double* /*train_score*/) const {
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
    const data_size_t* bag_mapper = nullptr;
    if (total_num_data != num_data_) {
      CHECK_EQ(bag_cnt, num_data_);
      bag_mapper = bag_indices;
    }
    std::vector<int> n_nozeroworker_perleaf(tree->num_leaves(), 1);
    int num_machines = Network::num_machines();
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      const double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto index_mapper = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      if (cnt_leaf_data > 0) {
        // bag_mapper[index_mapper[i]]
        const double new_output = obj->RenewTreeOutput(output, residual_getter, index_mapper, bag_mapper, cnt_leaf_data);
        tree->SetLeafOutput(i, new_output);
      } else {
        CHECK_GT(num_machines, 1);
        tree->SetLeafOutput(i, 0.0);
        n_nozeroworker_perleaf[i] = 0;
      }
    }
    if (num_machines > 1) {
      std::vector<double> outputs(tree->num_leaves());
      for (int i = 0; i < tree->num_leaves(); ++i) {
        outputs[i] = static_cast<double>(tree->LeafOutput(i));
      }
      outputs = Network::GlobalSum(&outputs);
      n_nozeroworker_perleaf = Network::GlobalSum(&n_nozeroworker_perleaf);
      for (int i = 0; i < tree->num_leaves(); ++i) {
        tree->SetLeafOutput(i, outputs[i] / n_nozeroworker_perleaf[i]);
      }
    }
  }
}

void SerialTreeLearner::ComputeBestSplitForFeature(
    FeatureHistogram* histogram_array_, int feature_index, int real_fidx,
    int8_t is_feature_used, int num_data, const LeafSplits* leaf_splits,
    SplitInfo* best_split, double parent_output) {
  bool is_feature_numerical = train_data_->FeatureBinMapper(feature_index)
                                  ->bin_type() == BinType::NumericalBin;
  if (is_feature_numerical & !config_->monotone_constraints.empty()) {
    constraints_->RecomputeConstraintsIfNeeded(
        constraints_.get(), feature_index, ~(leaf_splits->leaf_index()),
        train_data_->FeatureNumBin(feature_index));
  }
  SplitInfo new_split;
  if (config_->use_quantized_grad) {
    const uint8_t hist_bits_bin = gradient_discretizer_->GetHistBitsInLeaf<false>(leaf_splits->leaf_index());
    histogram_array_[feature_index].FindBestThresholdInt(
        leaf_splits->int_sum_gradients_and_hessians(),
        gradient_discretizer_->grad_scale(),
        gradient_discretizer_->hess_scale(),
        hist_bits_bin,
        hist_bits_bin,
        num_data,
        constraints_->GetFeatureConstraint(leaf_splits->leaf_index(), feature_index), parent_output, &new_split);
  } else {
    histogram_array_[feature_index].FindBestThreshold(
        leaf_splits->sum_gradients(), leaf_splits->sum_hessians(), num_data,
        constraints_->GetFeatureConstraint(leaf_splits->leaf_index(), feature_index), parent_output, &new_split);
  }
  new_split.feature = real_fidx;
  if (cegb_ != nullptr) {
    new_split.gain -=
        cegb_->DeltaGain(feature_index, real_fidx, leaf_splits->leaf_index(),
                         num_data, new_split);
  }
  if (new_split.monotone_type != 0) {
    double penalty = constraints_->ComputeMonotoneSplitGainPenalty(
        leaf_splits->leaf_index(), config_->monotone_penalty);
    new_split.gain *= penalty;
  }
  // it is needed to filter the features after the above code.
  // Otherwise, the `is_splittable` in `FeatureHistogram` will be wrong, and cause some features being accidentally filtered in the later nodes.
  if (new_split > *best_split && is_feature_used) {
    *best_split = new_split;
  }
}

double SerialTreeLearner::GetParentOutput(const Tree* tree, const LeafSplits* leaf_splits) const {
  double parent_output;
  if (tree->num_leaves() == 1) {
    // for root leaf the "parent" output is its own output because we don't apply any smoothing to the root
    parent_output = FeatureHistogram::CalculateSplittedLeafOutput<true, true, true, false>(
      leaf_splits->sum_gradients(), leaf_splits->sum_hessians(), config_->lambda_l1,
      config_->lambda_l2, config_->max_delta_step, BasicConstraint(),
      config_->path_smooth, static_cast<data_size_t>(leaf_splits->num_data_in_leaf()), 0);
  } else {
    parent_output = leaf_splits->weight();
  }
  return parent_output;
}

void SerialTreeLearner::RecomputeBestSplitForLeaf(Tree* tree, int leaf, SplitInfo* split) {
  FeatureHistogram* histogram_array_;
  if (!histogram_pool_.Get(leaf, &histogram_array_)) {
    Log::Warning(
        "Get historical Histogram for leaf %d failed, will skip the "
        "``RecomputeBestSplitForLeaf``",
        leaf);
    return;
  }
  double sum_gradients = split->left_sum_gradient + split->right_sum_gradient;
  double sum_hessians = split->left_sum_hessian + split->right_sum_hessian;
  int num_data = split->left_count + split->right_count;

  std::vector<SplitInfo> bests(share_state_->num_threads);
  LeafSplits leaf_splits(num_data, config_);
  leaf_splits.Init(leaf, sum_gradients, sum_hessians);

  // can't use GetParentOutput because leaf_splits doesn't have weight property set
  double parent_output = 0;
  if (config_->path_smooth > kEpsilon) {
    parent_output = FeatureHistogram::CalculateSplittedLeafOutput<true, true, true, false>(
      sum_gradients, sum_hessians, config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
      BasicConstraint(), config_->path_smooth, static_cast<data_size_t>(num_data), 0);
  }

  OMP_INIT_EX();
// find splits
std::vector<int8_t> node_used_features = col_sampler_.GetByNode(tree, leaf);
#pragma omp parallel for schedule(static) num_threads(share_state_->num_threads)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!col_sampler_.is_feature_used_bytree()[feature_index] ||
        !histogram_array_[feature_index].is_splittable()) {
      continue;
    }
    const int tid = omp_get_thread_num();
    int real_fidx = train_data_->RealFeatureIndex(feature_index);
    ComputeBestSplitForFeature(histogram_array_, feature_index, real_fidx, node_used_features[feature_index],
                               num_data, &leaf_splits, &bests[tid], parent_output);

    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  auto best_idx = ArrayArgs<SplitInfo>::ArgMax(bests);
  *split = bests[best_idx];
}

#ifdef DEBUG
void SerialTreeLearner::CheckSplit(const SplitInfo& best_split_info, const int left_leaf_index, const int right_leaf_index) {
  data_size_t num_data_in_left = 0;
  data_size_t num_data_in_right = 0;
  const data_size_t* data_indices_in_left = data_partition_->GetIndexOnLeaf(left_leaf_index, &num_data_in_left);
  const data_size_t* data_indices_in_right = data_partition_->GetIndexOnLeaf(right_leaf_index, &num_data_in_right);
  if (config_->use_quantized_grad) {
    int32_t sum_left_gradient = 0;
    int32_t sum_left_hessian = 0;
    int32_t sum_right_gradient = 0;
    int32_t sum_right_hessian = 0;
    const int8_t* discretized_grad_and_hess = gradient_discretizer_->discretized_gradients_and_hessians();
    for (data_size_t i = 0; i < num_data_in_left; ++i) {
      const data_size_t index = data_indices_in_left[i];
      sum_left_gradient += discretized_grad_and_hess[2 * index + 1];
      sum_left_hessian += discretized_grad_and_hess[2 * index];
    }
    for (data_size_t i = 0; i < num_data_in_right; ++i) {
      const data_size_t index = data_indices_in_right[i];
      sum_right_gradient += discretized_grad_and_hess[2 * index + 1];
      sum_right_hessian += discretized_grad_and_hess[2 * index];
    }
    Log::Warning("============================ start leaf split info ============================");
    Log::Warning("left_leaf_index = %d, right_leaf_index = %d", left_leaf_index, right_leaf_index);
    Log::Warning("num_data_in_left = %d, num_data_in_right = %d", num_data_in_left, num_data_in_right);
    Log::Warning("sum_left_gradient = %d, best_split_info->left_sum_gradient_and_hessian.gradient = %d", sum_left_gradient,
      static_cast<int32_t>(best_split_info.left_sum_gradient_and_hessian >> 32));
    Log::Warning("sum_left_hessian = %d, best_split_info->left_sum_gradient_and_hessian.hessian = %d", sum_left_hessian,
      static_cast<int32_t>(best_split_info.left_sum_gradient_and_hessian & 0x00000000ffffffff));
    Log::Warning("sum_right_gradient = %d, best_split_info->right_sum_gradient_and_hessian.gradient = %d", sum_right_gradient,
      static_cast<int32_t>(best_split_info.right_sum_gradient_and_hessian >> 32));
    Log::Warning("sum_right_hessian = %d, best_split_info->right_sum_gradient_and_hessian.hessian = %d", sum_right_hessian,
      static_cast<int32_t>(best_split_info.right_sum_gradient_and_hessian & 0x00000000ffffffff));
    CHECK_EQ(num_data_in_left, best_split_info.left_count);
    CHECK_EQ(num_data_in_right, best_split_info.right_count);
    CHECK_EQ(sum_left_gradient, static_cast<int32_t>(best_split_info.left_sum_gradient_and_hessian >> 32))
    CHECK_EQ(sum_left_hessian, static_cast<int32_t>(best_split_info.left_sum_gradient_and_hessian & 0x00000000ffffffff));
    CHECK_EQ(sum_right_gradient, static_cast<int32_t>(best_split_info.right_sum_gradient_and_hessian >> 32));
    CHECK_EQ(sum_right_hessian, static_cast<int32_t>(best_split_info.right_sum_gradient_and_hessian & 0x00000000ffffffff));
    Log::Warning("============================ end leaf split info ============================");
  }
}
#endif

}  // namespace LightGBM
