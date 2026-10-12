/*!
 * Copyright (c) 2019-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2019-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_COST_EFFECTIVE_GRADIENT_BOOSTING_HPP_
#define LIGHTGBM_SRC_TREELEARNER_COST_EFFECTIVE_GRADIENT_BOOSTING_HPP_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/threading.h>

#include <vector>

#include "data_partition.hpp"
#include "serial_tree_learner.h"
#include "split_info.hpp"

namespace LightGBM {

class CostEfficientGradientBoosting {
 public:
  explicit CostEfficientGradientBoosting(const SerialTreeLearner* tree_learner)
      : init_(false), tree_learner_(tree_learner) {}
  static bool IsEnable(const Config* config) {
    if (config->cegb_tradeoff >= 1.0f && config->cegb_penalty_split <= 0.0f &&
        config->cegb_penalty_feature_coupled.empty() &&
        config->cegb_penalty_feature_lazy.empty()) {
      return false;
    } else {
      return true;
    }
  }

  void Init() {
    auto train_data = tree_learner_->train_data_;
    if (!init_) {
      splits_per_leaf_.resize(
          static_cast<size_t>(tree_learner_->config_->num_leaves) *
          train_data->num_features());
      is_feature_used_in_split_.clear();
      is_feature_used_in_split_.resize(train_data->num_features());
    }

    if (!tree_learner_->config_->cegb_penalty_feature_coupled.empty() &&
        tree_learner_->config_->cegb_penalty_feature_coupled.size() !=
            static_cast<size_t>(train_data->num_total_features())) {
      Log::Fatal(
          "cegb_penalty_feature_coupled should be the same size as feature "
          "number.");
    }
    if (!tree_learner_->config_->cegb_penalty_feature_lazy.empty()) {
      if (tree_learner_->config_->cegb_penalty_feature_lazy.size() !=
          static_cast<size_t>(train_data->num_total_features())) {
        Log::Fatal(
            "cegb_penalty_feature_lazy should be the same size as feature "
            "number.");
      }
      if (!init_) {
        feature_used_in_data_ = Common::EmptyBitset(train_data->num_features() *
                                                    tree_learner_->num_data_);
      }
    }
    init_ = true;
  }

  void BeforeTrain() {
    // clear the splits in splits_per_leaf_
    Threading::For<size_t>(0, splits_per_leaf_.size(), 1024,
      [this] (int /*thread_index*/, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        splits_per_leaf_[i].Reset();
      }
    });
  }

  double DeltaGain(int feature_index, int real_fidx, int leaf_index,
                   int num_data_in_leaf, SplitInfo split_info) {
    auto config = tree_learner_->config_;
    double delta =
        config->cegb_tradeoff * config->cegb_penalty_split * num_data_in_leaf;
    if (!config->cegb_penalty_feature_coupled.empty() &&
        !is_feature_used_in_split_[feature_index]) {
      delta += config->cegb_tradeoff *
               config->cegb_penalty_feature_coupled[real_fidx];
    }
    if (!config->cegb_penalty_feature_lazy.empty()) {
      delta += config->cegb_tradeoff *
               CalculateOndemandCosts(feature_index, real_fidx, leaf_index);
    }
    splits_per_leaf_[static_cast<size_t>(leaf_index) *
                         tree_learner_->train_data_->num_features() +
                     feature_index] = split_info;
    return delta;
  }

  void UpdateLeafBestSplits(Tree* tree, int best_leaf,
                            const SplitInfo* best_split_info,
                            std::vector<SplitInfo>* best_split_per_leaf) {
    auto config = tree_learner_->config_;
    auto train_data = tree_learner_->train_data_;
    const int inner_feature_index =
        train_data->InnerFeatureIndex(best_split_info->feature);
    auto& ref_best_split_per_leaf = *best_split_per_leaf;
    if (!config->cegb_penalty_feature_coupled.empty() &&
        !is_feature_used_in_split_[inner_feature_index]) {
      is_feature_used_in_split_[inner_feature_index] = true;
      for (int i = 0; i < tree->num_leaves(); ++i) {
        if (i == best_leaf) continue;
        auto split = &splits_per_leaf_[static_cast<size_t>(i) *
                                           train_data->num_features() +
                                       inner_feature_index];
        split->gain +=
            config->cegb_tradeoff *
            config->cegb_penalty_feature_coupled[best_split_info->feature];
        // Avoid to update the leaf that cannot split
        if (ref_best_split_per_leaf[i].gain > kMinScore &&
            *split > ref_best_split_per_leaf[i]) {
          ref_best_split_per_leaf[i] = *split;
        }
      }
    }
    if (!config->cegb_penalty_feature_lazy.empty()) {
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = tree_learner_->data_partition_->GetIndexOnLeaf(
          best_leaf, &cnt_leaf_data);
      for (data_size_t i_input = 0; i_input < cnt_leaf_data; ++i_input) {
        int real_idx = tmp_idx[i_input];
        Common::InsertBitset(
            &feature_used_in_data_,
            train_data->num_data() * inner_feature_index + real_idx);
      }
    }
  }

 private:
  double CalculateOndemandCosts(int feature_index, int real_fidx,
                                int leaf_index) const {
    if (tree_learner_->config_->cegb_penalty_feature_lazy.empty()) {
      return 0.0f;
    }
    auto train_data = tree_learner_->train_data_;
    double penalty =
        tree_learner_->config_->cegb_penalty_feature_lazy[real_fidx];

    double total = 0.0f;
    data_size_t cnt_leaf_data = 0;
    auto tmp_idx = tree_learner_->data_partition_->GetIndexOnLeaf(
        leaf_index, &cnt_leaf_data);

    for (data_size_t i_input = 0; i_input < cnt_leaf_data; ++i_input) {
      int real_idx = tmp_idx[i_input];
      if (Common::FindInBitset(
              feature_used_in_data_.data(),
              train_data->num_data() * train_data->num_features(),
              train_data->num_data() * feature_index + real_idx)) {
        continue;
      }
      total += penalty;
    }
    return total;
  }
  bool init_;
  const SerialTreeLearner* tree_learner_;
  std::vector<SplitInfo> splits_per_leaf_;
  std::vector<bool> is_feature_used_in_split_;
  std::vector<uint32_t> feature_used_in_data_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_COST_EFFECTIVE_GRADIENT_BOOSTING_HPP_
