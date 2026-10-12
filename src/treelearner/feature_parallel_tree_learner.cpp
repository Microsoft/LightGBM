/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cstring>
#include <vector>

#include "parallel_tree_learner.h"

namespace LightGBM {


template <typename TREELEARNER_T>
FeatureParallelTreeLearner<TREELEARNER_T>::FeatureParallelTreeLearner(const Config* config):TREELEARNER_T(config) {
}

template <typename TREELEARNER_T>
FeatureParallelTreeLearner<TREELEARNER_T>::~FeatureParallelTreeLearner() {
}

template <typename TREELEARNER_T>
void FeatureParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  auto max_cat_threshold = this->config_->max_cat_threshold;
  // need to be able to hold smaller and larger best splits in SyncUpGlobalBestSplit
  int split_info_size = SplitInfo::Size(max_cat_threshold) * 2;

  input_buffer_.resize(split_info_size);
  output_buffer_.resize(split_info_size);
}


template <typename TREELEARNER_T>
void FeatureParallelTreeLearner<TREELEARNER_T>::BeforeTrain() {
  TREELEARNER_T::BeforeTrain();
  // get feature partition
  std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  std::vector<int> num_bins_distributed(num_machines_, 0);
  for (int i = 0; i < this->train_data_->num_total_features(); ++i) {
    int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
    if (inner_feature_index == -1) {
      continue;
    }
    if (this->col_sampler_.is_feature_used_bytree()[inner_feature_index]) {
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
      feature_distribution[cur_min_machine].push_back(inner_feature_index);
      num_bins_distributed[cur_min_machine] += this->train_data_->FeatureNumBin(inner_feature_index);
      this->col_sampler_.SetIsFeatureUsedByTree(inner_feature_index, false);
    }
  }
  // get local used features
  for (auto fid : feature_distribution[rank_]) {
    this->col_sampler_.SetIsFeatureUsedByTree(fid, true);
  }
}

template <typename TREELEARNER_T>
void FeatureParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(
      const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree) {
  TREELEARNER_T::FindBestSplitsFromHistograms(is_feature_used, use_subtract, tree);
  SplitInfo smaller_best_split, larger_best_split;
  // get best split at smaller leaf
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->leaf_index() >= 0) {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()];
  }
  // sync global best info
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);
  // update best split
  this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()] = smaller_best_split;
  if (this->larger_leaf_splits_->leaf_index() >= 0) {
    this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()] = larger_best_split;
  }
}

// instantiate template classes, otherwise linker cannot find the code
template class FeatureParallelTreeLearner<GPUTreeLearner>;
template class FeatureParallelTreeLearner<SerialTreeLearner>;
}  // namespace LightGBM
