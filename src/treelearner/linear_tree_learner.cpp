/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "linear_tree_learner.h"

#include <Eigen/Dense>

#include <algorithm>
#include <memory>
#include <vector>

namespace LightGBM {

template <typename TREE_LEARNER_TYPE>
void LinearTreeLearner<TREE_LEARNER_TYPE>::Init(const Dataset* train_data, bool is_constant_hessian) {
  TREE_LEARNER_TYPE::Init(train_data, is_constant_hessian);
  LinearTreeLearner::InitLinear(train_data, this->config_->num_leaves);
}

template <typename TREE_LEARNER_TYPE>
void LinearTreeLearner<TREE_LEARNER_TYPE>::InitLinear(const Dataset* train_data, const int max_leaves) {
  leaf_map_ = std::vector<int>(train_data->num_data(), -1);
  contains_nan_ = std::vector<int8_t>(train_data->num_features(), 0);
  // identify features containing nans
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int feat = 0; feat < train_data->num_features(); ++feat) {
    auto bin_mapper = this->train_data_->FeatureBinMapper(feat);
    if (bin_mapper->bin_type() == BinType::NumericalBin) {
      const float* feat_ptr = this->train_data_->raw_index(feat);
      for (int i = 0; i < train_data->num_data(); ++i) {
        if (std::isnan(feat_ptr[i])) {
          contains_nan_[feat] = 1;
          break;
        }
      }
    }
  }
  any_nan_ = false;
  for (int feat = 0; feat < train_data->num_features(); ++feat) {
    if (contains_nan_[feat]) {
      any_nan_ = true;
      break;
    }
  }
  // preallocate the matrix used to calculate linear model coefficients
  int max_num_feat = std::min(max_leaves, this->train_data_->num_numeric_features());
  XTHX_.clear();
  XTg_.clear();
  for (int i = 0; i < max_leaves; ++i) {
    // store only upper triangular half of matrix as an array, in row-major order
    // this requires (max_num_feat + 1) * (max_num_feat + 2) / 2 entries (including the constant terms of the regression)
    // we add another 8 to ensure cache lines are not shared among processors
    XTHX_.push_back(std::vector<double>((max_num_feat + 1) * (max_num_feat + 2) / 2 + 8, 0));
    XTg_.push_back(std::vector<double>(max_num_feat + 9, 0.0));
  }
  XTHX_by_thread_.clear();
  XTg_by_thread_.clear();
  int max_threads = OMP_NUM_THREADS();
  for (int i = 0; i < max_threads; ++i) {
    XTHX_by_thread_.push_back(XTHX_);
    XTg_by_thread_.push_back(XTg_);
  }
}

template <typename TREE_LEARNER_TYPE>
Tree* LinearTreeLearner<TREE_LEARNER_TYPE>::Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::Train", global_timer);
  this->gradients_ = gradients;
  this->hessians_ = hessians;
  int num_threads = OMP_NUM_THREADS();
  if (this->share_state_->num_threads != num_threads && this->share_state_->num_threads > 0) {
    Log::Warning(
        "Detected that num_threads changed during training (from %d to %d), "
        "it may cause unexpected errors.",
        this->share_state_->num_threads, num_threads);
  }
  this->share_state_->num_threads = num_threads;

  // some initial works before training
  this->BeforeTrain();

  auto tree = std::unique_ptr<Tree>(new Tree(this->config_->num_leaves, true, true));
  auto tree_ptr = tree.get();
  this->constraints_->ShareTreePointer(tree_ptr);

  // root leaf
  int left_leaf = 0;
  int cur_depth = 1;
  // only root leaf can be splitted on first time
  int right_leaf = -1;

  int init_splits = this->ForceSplits(tree_ptr, &left_leaf, &right_leaf, &cur_depth);

  for (int split = init_splits; split < this->config_->num_leaves - 1; ++split) {
    // some initial works before finding best split
    if (this->BeforeFindBestSplit(tree_ptr, left_leaf, right_leaf)) {
      // find best threshold for every feature
      this->FindBestSplits(tree_ptr);
    }
    // Get a leaf with max split gain
    int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(this->best_split_per_leaf_));
    // Get split information for best leaf
    const SplitInfo& best_leaf_SplitInfo = this->best_split_per_leaf_[best_leaf];
    // cannot split, quit
    if (best_leaf_SplitInfo.gain <= 0.0) {
      Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_SplitInfo.gain);
      break;
    }
    // split tree with best leaf
    this->Split(tree_ptr, best_leaf, &left_leaf, &right_leaf);
    cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));
  }

  bool has_nan = false;
  if (any_nan_) {
    for (int i = 0; i < tree->num_leaves() - 1 ; ++i) {
      if (contains_nan_[tree_ptr->split_feature_inner(i)]) {
        has_nan = true;
        break;
      }
    }
  }

  GetLeafMap(tree_ptr);

  if (has_nan) {
    CalculateLinear<true>(tree_ptr, false, this->gradients_, this->hessians_, is_first_tree);
  } else {
    CalculateLinear<false>(tree_ptr, false, this->gradients_, this->hessians_, is_first_tree);
  }

  Log::Debug("Trained a tree with leaves = %d and depth = %d", tree->num_leaves(), cur_depth);
  return tree.release();
}

template <typename TREE_LEARNER_TYPE>
Tree* LinearTreeLearner<TREE_LEARNER_TYPE>::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t *hessians) const {
  auto tree = TREE_LEARNER_TYPE::FitByExistingTree(old_tree, gradients, hessians);
  bool has_nan = false;
  if (any_nan_) {
    for (int i = 0; i < tree->num_leaves() - 1 ; ++i) {
      if (contains_nan_[this->train_data_->InnerFeatureIndex(tree->split_feature(i))]) {
        has_nan = true;
        break;
      }
    }
  }
  GetLeafMap(tree);
  if (has_nan) {
    CalculateLinear<true>(tree, true, gradients, hessians, false);
  } else {
    CalculateLinear<false>(tree, true, gradients, hessians, false);
  }
  return tree;
}

template <typename TREE_LEARNER_TYPE>
Tree* LinearTreeLearner<TREE_LEARNER_TYPE>::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                           const score_t* gradients, const score_t *hessians) const {
  this->data_partition_->ResetByLeafPred(leaf_pred, old_tree->num_leaves());
  return LinearTreeLearner::FitByExistingTree(old_tree, gradients, hessians);
}

template <typename TREE_LEARNER_TYPE>
void LinearTreeLearner<TREE_LEARNER_TYPE>::GetLeafMap(Tree* tree) const {
  std::fill(leaf_map_.begin(), leaf_map_.end(), -1);
  // map data to leaf number
  const data_size_t* ind = this->data_partition_->indices();
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(dynamic)
  for (int i = 0; i < tree->num_leaves(); ++i) {
    data_size_t idx = this->data_partition_->leaf_begin(i);
    for (int j = 0; j < this->data_partition_->leaf_count(i); ++j) {
      leaf_map_[ind[idx + j]] = i;
    }
  }
}


template<typename TREE_LEARNER_TYPE>
template <bool HAS_NAN>
void LinearTreeLearner<TREE_LEARNER_TYPE>::CalculateLinear(Tree* tree, bool is_refit, const score_t* gradients, const score_t* hessians, bool is_first_tree) const {
  tree->SetIsLinear(true);
  int num_leaves = tree->num_leaves();
  int num_threads = OMP_NUM_THREADS();
  if (is_first_tree) {
    for (int leaf_num = 0; leaf_num < num_leaves; ++leaf_num) {
      tree->SetLeafConst(leaf_num, tree->LeafOutput(leaf_num));
    }
    return;
  }

  // calculate coefficients using the method described in Eq 3 of https://arxiv.org/abs/1802.05640
  // the coefficients vector is given by
  // - (X_T * H * X + lambda) ^ (-1) * (X_T * g)
  // where:
  // X is the matrix where the first column is the feature values and the second is all ones,
  // H is the diagonal matrix of the hessian,
  // lambda is the diagonal matrix with diagonal entries equal to the regularisation term linear_lambda
  // g is the vector of gradients
  // the subscript _T denotes the transpose

  // create array of pointers to raw data, and coefficient matrices, for each leaf
  std::vector<std::vector<int>> leaf_features;
  std::vector<int> leaf_num_features;
  std::vector<std::vector<const float*>> raw_data_ptr;
  size_t max_num_features = 0;
  for (int i = 0; i < num_leaves; ++i) {
    std::vector<int> raw_features;
    if (is_refit) {
      raw_features = tree->LeafFeatures(i);
    } else {
      raw_features = tree->branch_features(i);
    }
    std::sort(raw_features.begin(), raw_features.end());
    auto new_end = std::unique(raw_features.begin(), raw_features.end());
    raw_features.erase(new_end, raw_features.end());
    std::vector<int> numerical_features;
    std::vector<const float*> data_ptr;
    for (size_t j = 0; j < raw_features.size(); ++j) {
      int feat = this->train_data_->InnerFeatureIndex(raw_features[j]);
      auto bin_mapper = this->train_data_->FeatureBinMapper(feat);
      if (bin_mapper->bin_type() == BinType::NumericalBin) {
        numerical_features.push_back(feat);
        data_ptr.push_back(this->train_data_->raw_index(feat));
      }
    }
    leaf_features.push_back(numerical_features);
    raw_data_ptr.push_back(data_ptr);
    leaf_num_features.push_back(static_cast<int>(numerical_features.size()));
    if (numerical_features.size() > max_num_features) {
      max_num_features = numerical_features.size();
    }
  }
  // clear the coefficient matrices
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int i = 0; i < num_threads; ++i) {
    for (int leaf_num = 0; leaf_num < num_leaves; ++leaf_num) {
      size_t num_feat = leaf_features[leaf_num].size();
      std::fill(XTHX_by_thread_[i][leaf_num].begin(), XTHX_by_thread_[i][leaf_num].begin() + (num_feat + 1) * (num_feat + 2) / 2, 0.0f);
      std::fill(XTg_by_thread_[i][leaf_num].begin(), XTg_by_thread_[i][leaf_num].begin() + num_feat + 1, 0.0f);
    }
  }
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int leaf_num = 0; leaf_num < num_leaves; ++leaf_num) {
    size_t num_feat = leaf_features[leaf_num].size();
    std::fill(XTHX_[leaf_num].begin(), XTHX_[leaf_num].begin() + (num_feat + 1) * (num_feat + 2) / 2, 0.0f);
    std::fill(XTg_[leaf_num].begin(), XTg_[leaf_num].begin() + num_feat + 1, 0.0f);
  }
  std::vector<std::vector<int>> num_nonzero;
  for (int i = 0; i < num_threads; ++i) {
    if (HAS_NAN) {
      num_nonzero.push_back(std::vector<int>(num_leaves, 0));
    }
  }
  OMP_INIT_EX();
#pragma omp parallel num_threads(OMP_NUM_THREADS()) if (this->num_data_ > 1024)
  {
    std::vector<float> curr_row(max_num_features + 1);
    int tid = omp_get_thread_num();
#pragma omp for schedule(static)
    for (int i = 0; i < this->num_data_; ++i) {
      OMP_LOOP_EX_BEGIN();
      int leaf_num = leaf_map_[i];
      if (leaf_num < 0) {
        continue;
      }
      bool nan_found = false;
      int num_feat = leaf_num_features[leaf_num];
      for (int feat = 0; feat < num_feat; ++feat) {
        if (HAS_NAN) {
          float val = raw_data_ptr[leaf_num][feat][i];
          if (std::isnan(val)) {
            nan_found = true;
            break;
          }
          num_nonzero[tid][leaf_num] += 1;
          curr_row[feat] = val;
        } else {
          curr_row[feat] = raw_data_ptr[leaf_num][feat][i];
        }
      }
      if (HAS_NAN) {
        if (nan_found) {
          continue;
        }
      }
      curr_row[num_feat] = 1.0;
      float h = static_cast<float>(hessians[i]);
      float g = static_cast<float>(gradients[i]);
      int j = 0;
      for (int feat1 = 0; feat1 < num_feat + 1; ++feat1) {
        double f1_val = static_cast<double>(curr_row[feat1]);
        XTg_by_thread_[tid][leaf_num][feat1] += f1_val * g;
        f1_val *= h;
        for (int feat2 = feat1; feat2 < num_feat + 1; ++feat2) {
          XTHX_by_thread_[tid][leaf_num][j] += f1_val * curr_row[feat2];
          ++j;
        }
      }
      OMP_LOOP_EX_END();
    }
  }
  OMP_THROW_EX();
  auto total_nonzero = std::vector<int>(tree->num_leaves());
  // aggregate results from different threads
  for (int tid = 0; tid < num_threads; ++tid) {
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int leaf_num = 0; leaf_num < num_leaves; ++leaf_num) {
      size_t num_feat = leaf_features[leaf_num].size();
      for (size_t j = 0; j < (num_feat + 1) * (num_feat + 2) / 2; ++j) {
        XTHX_[leaf_num][j] += XTHX_by_thread_[tid][leaf_num][j];
      }
      for (size_t feat1 = 0; feat1 < num_feat + 1; ++feat1) {
        XTg_[leaf_num][feat1] += XTg_by_thread_[tid][leaf_num][feat1];
      }
      if (HAS_NAN) {
        total_nonzero[leaf_num] += num_nonzero[tid][leaf_num];
      }
    }
  }
  if (!HAS_NAN) {
    for (int leaf_num = 0; leaf_num < num_leaves; ++leaf_num) {
      total_nonzero[leaf_num] = this->data_partition_->leaf_count(leaf_num);
    }
  }
  double shrinkage = tree->shrinkage();
  double decay_rate = this->config_->refit_decay_rate;
  // copy into eigen matrices and solve
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int leaf_num = 0; leaf_num < num_leaves; ++leaf_num) {
    if (total_nonzero[leaf_num] < static_cast<int>(leaf_features[leaf_num].size()) + 1) {
      if (is_refit) {
        double old_const = tree->LeafConst(leaf_num);
        tree->SetLeafConst(leaf_num, decay_rate * old_const + (1.0 - decay_rate) * tree->LeafOutput(leaf_num) * shrinkage);
        tree->SetLeafCoeffs(leaf_num, std::vector<double>(leaf_features[leaf_num].size(), 0));
        tree->SetLeafFeaturesInner(leaf_num, leaf_features[leaf_num]);
      } else {
        tree->SetLeafConst(leaf_num, tree->LeafOutput(leaf_num));
      }
      continue;
    }
    size_t num_feat = leaf_features[leaf_num].size();
    Eigen::MatrixXd XTHX_mat(num_feat + 1, num_feat + 1);
    Eigen::MatrixXd XTg_mat(num_feat + 1, 1);
    size_t j = 0;
    for (size_t feat1 = 0; feat1 < num_feat + 1; ++feat1) {
      for (size_t feat2 = feat1; feat2 < num_feat + 1; ++feat2) {
        XTHX_mat(feat1, feat2) = XTHX_[leaf_num][j];
        XTHX_mat(feat2, feat1) = XTHX_mat(feat1, feat2);
        if ((feat1 == feat2) && (feat1 < num_feat)) {
          XTHX_mat(feat1, feat2) += this->config_->linear_lambda;
        }
        ++j;
      }
      XTg_mat(feat1) = XTg_[leaf_num][feat1];
    }
    Eigen::MatrixXd coeffs = - XTHX_mat.fullPivLu().inverse() * XTg_mat;
    std::vector<double> coeffs_vec;
    std::vector<int> features_new;
    std::vector<double> old_coeffs = tree->LeafCoeffs(leaf_num);
    for (size_t i = 0; i < leaf_features[leaf_num].size(); ++i) {
      if (is_refit) {
        features_new.push_back(leaf_features[leaf_num][i]);
        coeffs_vec.push_back(decay_rate * old_coeffs[i] + (1.0 - decay_rate) * coeffs(i) * shrinkage);
      } else {
        if (coeffs(i) < -kZeroThreshold || coeffs(i) > kZeroThreshold) {
          coeffs_vec.push_back(coeffs(i));
          int feat = leaf_features[leaf_num][i];
          features_new.push_back(feat);
        }
      }
    }
    // update the tree properties
    tree->SetLeafFeaturesInner(leaf_num, features_new);
    std::vector<int> features_raw(features_new.size());
    for (size_t i = 0; i < features_new.size(); ++i) {
      features_raw[i] = this->train_data_->RealFeatureIndex(features_new[i]);
    }
    tree->SetLeafFeatures(leaf_num, features_raw);
    tree->SetLeafCoeffs(leaf_num, coeffs_vec);
    if (is_refit) {
      double old_const = tree->LeafConst(leaf_num);
      tree->SetLeafConst(leaf_num, decay_rate * old_const + (1.0 - decay_rate) * coeffs(num_feat) * shrinkage);
    } else {
      tree->SetLeafConst(leaf_num, coeffs(num_feat));
    }
  }
}

template void LinearTreeLearner<SerialTreeLearner>::Init(const Dataset* train_data, bool is_constant_hessian);
template void LinearTreeLearner<SerialTreeLearner>::InitLinear(const Dataset* train_data, const int max_leaves);
template Tree* LinearTreeLearner<SerialTreeLearner>::Train(const score_t* gradients, const score_t *hessians, bool is_first_tree);
template Tree* LinearTreeLearner<SerialTreeLearner>::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t *hessians) const;
template Tree* LinearTreeLearner<SerialTreeLearner>::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                           const score_t* gradients, const score_t *hessians) const;

template void LinearTreeLearner<GPUTreeLearner>::Init(const Dataset* train_data, bool is_constant_hessian);
template void LinearTreeLearner<GPUTreeLearner>::InitLinear(const Dataset* train_data, const int max_leaves);
template Tree* LinearTreeLearner<GPUTreeLearner>::Train(const score_t* gradients, const score_t *hessians, bool is_first_tree);
template Tree* LinearTreeLearner<GPUTreeLearner>::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t *hessians) const;
template Tree* LinearTreeLearner<GPUTreeLearner>::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                           const score_t* gradients, const score_t *hessians) const;

}  // namespace LightGBM
