/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_COL_SAMPLER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_COL_SAMPLER_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/random.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace LightGBM {
class ColSampler {
 public:
  explicit ColSampler(const Config* config)
      : fraction_bytree_(config->feature_fraction),
        fraction_bynode_(config->feature_fraction_bynode),
        seed_(config->feature_fraction_seed),
        random_(config->feature_fraction_seed) {
    for (auto constraint : config->interaction_constraints_vector) {
      std::unordered_set<int> constraint_set(constraint.begin(), constraint.end());
      interaction_constraints_.push_back(constraint_set);
    }
  }

  static int GetCnt(size_t total_cnt, double fraction) {
    const int min = std::min(1, static_cast<int>(total_cnt));
    int used_feature_cnt = static_cast<int>(Common::RoundInt(total_cnt * fraction));
    return std::max(used_feature_cnt, min);
  }

  void SetTrainingData(const Dataset* train_data) {
    train_data_ = train_data;
    is_feature_used_.resize(train_data_->num_features(), 1);
    valid_feature_indices_ = train_data->ValidFeatureIndices();
    if (fraction_bytree_ >= 1.0f) {
      need_reset_bytree_ = false;
      used_cnt_bytree_ = static_cast<int>(valid_feature_indices_.size());
    } else {
      need_reset_bytree_ = true;
      used_cnt_bytree_ =
          GetCnt(valid_feature_indices_.size(), fraction_bytree_);
    }
    ResetByTree();
  }

  void SetConfig(const Config* config) {
    fraction_bytree_ = config->feature_fraction;
    fraction_bynode_ = config->feature_fraction_bynode;
    is_feature_used_.resize(train_data_->num_features(), 1);
    // seed is changed
    if (seed_ != config->feature_fraction_seed) {
      seed_ = config->feature_fraction_seed;
      random_ = Random(seed_);
    }
    if (fraction_bytree_ >= 1.0f) {
      need_reset_bytree_ = false;
      used_cnt_bytree_ = static_cast<int>(valid_feature_indices_.size());
    } else {
      need_reset_bytree_ = true;
      used_cnt_bytree_ =
          GetCnt(valid_feature_indices_.size(), fraction_bytree_);
    }
    ResetByTree();
  }

  void ResetByTree() {
    if (need_reset_bytree_) {
      std::memset(is_feature_used_.data(), 0,
                  sizeof(int8_t) * is_feature_used_.size());
      used_feature_indices_ = random_.Sample(
          static_cast<int>(valid_feature_indices_.size()), used_cnt_bytree_);
      int omp_loop_size = static_cast<int>(used_feature_indices_.size());

#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature = valid_feature_indices_[used_feature_indices_[i]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        is_feature_used_[inner_feature_index] = 1;
      }
    }
  }

  std::vector<int8_t> GetByNode(const Tree* tree, int leaf) {
    // get interaction constraints for current branch
    std::unordered_set<int> allowed_features;
    if (!interaction_constraints_.empty()) {
      std::vector<int> branch_features = tree->branch_features(leaf);
      allowed_features.insert(branch_features.begin(), branch_features.end());
      for (auto constraint : interaction_constraints_) {
        int num_feat_found = 0;
        if (branch_features.size() == 0) {
          allowed_features.insert(constraint.begin(), constraint.end());
        }
        for (int feat : branch_features) {
          if (constraint.count(feat) == 0) {
            break;
          }
          ++num_feat_found;
          if (num_feat_found == static_cast<int>(branch_features.size())) {
            allowed_features.insert(constraint.begin(), constraint.end());
            break;
          }
        }
      }
    }

    std::vector<int8_t> ret(train_data_->num_features(), 0);
    if (fraction_bynode_ >= 1.0f) {
      if (interaction_constraints_.empty()) {
        return std::vector<int8_t>(train_data_->num_features(), 1);
      } else {
        for (int feat : allowed_features) {
          int inner_feat = train_data_->InnerFeatureIndex(feat);
          if (inner_feat >= 0) {
            ret[inner_feat] = 1;
          }
        }
        return ret;
      }
    }
    if (need_reset_bytree_) {
      auto used_feature_cnt = GetCnt(used_feature_indices_.size(), fraction_bynode_);
      std::vector<int>* allowed_used_feature_indices;
      std::vector<int> filtered_feature_indices;
      if (interaction_constraints_.empty()) {
        allowed_used_feature_indices = &used_feature_indices_;
      } else {
        for (int feat_ind : used_feature_indices_) {
          if (allowed_features.count(valid_feature_indices_[feat_ind]) == 1) {
            filtered_feature_indices.push_back(feat_ind);
          }
        }
        used_feature_cnt = std::min(used_feature_cnt, static_cast<int>(filtered_feature_indices.size()));
        allowed_used_feature_indices = &filtered_feature_indices;
      }
      auto sampled_indices = random_.Sample(
          static_cast<int>((*allowed_used_feature_indices).size()), used_feature_cnt);
      int omp_loop_size = static_cast<int>(sampled_indices.size());
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature =
            valid_feature_indices_[(*allowed_used_feature_indices)[sampled_indices[i]]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        ret[inner_feature_index] = 1;
      }
    } else {
      auto used_feature_cnt =
          GetCnt(valid_feature_indices_.size(), fraction_bynode_);
      std::vector<int>* allowed_valid_feature_indices;
      std::vector<int> filtered_feature_indices;
      if (interaction_constraints_.empty()) {
        allowed_valid_feature_indices = &valid_feature_indices_;
      } else {
        for (int feat : valid_feature_indices_) {
          if (allowed_features.count(feat) == 1) {
            filtered_feature_indices.push_back(feat);
          }
        }
        allowed_valid_feature_indices = &filtered_feature_indices;
        used_feature_cnt = std::min(used_feature_cnt, static_cast<int>(filtered_feature_indices.size()));
      }
      auto sampled_indices = random_.Sample(
          static_cast<int>((*allowed_valid_feature_indices).size()), used_feature_cnt);
      int omp_loop_size = static_cast<int>(sampled_indices.size());
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature = (*allowed_valid_feature_indices)[sampled_indices[i]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        ret[inner_feature_index] = 1;
      }
    }
    return ret;
  }

  const std::vector<int8_t>& is_feature_used_bytree() const {
    return is_feature_used_;
  }

  void SetIsFeatureUsedByTree(int fid, bool val) {
    is_feature_used_[fid] = val;
  }

 private:
  const Dataset* train_data_;
  double fraction_bytree_;
  double fraction_bynode_;
  bool need_reset_bytree_;
  int used_cnt_bytree_;
  int seed_;
  Random random_;
  std::vector<int8_t> is_feature_used_;
  std::vector<int> used_feature_indices_;
  std::vector<int> valid_feature_indices_;
  /*! \brief interaction constraints index in original (raw data) features */
  std::vector<std::unordered_set<int>> interaction_constraints_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_TREELEARNER_COL_SAMPLER_HPP_
