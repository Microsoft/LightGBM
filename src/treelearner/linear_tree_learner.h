/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_LINEAR_TREE_LEARNER_H_
#define LIGHTGBM_SRC_TREELEARNER_LINEAR_TREE_LEARNER_H_

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "gpu_tree_learner.h"
#include "serial_tree_learner.h"

namespace LightGBM {

template <typename TREE_LEARNER_TYPE>
class LinearTreeLearner: public TREE_LEARNER_TYPE {
 public:
  explicit LinearTreeLearner(const Config* config) : TREE_LEARNER_TYPE(config) {}

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void InitLinear(const Dataset* train_data, const int max_leaves) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  /*! \brief Create array mapping dataset to leaf index, used for linear trees */
  void GetLeafMap(Tree* tree) const;

  template<bool HAS_NAN>
  void CalculateLinear(Tree* tree, bool is_refit, const score_t* gradients, const score_t* hessians, bool is_first_tree) const;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) const override;

  void AddPredictionToScore(const Tree* tree,
                            double* out_score) const override {
    CHECK_LE(tree->num_leaves(), this->data_partition_->num_leaves());
    bool has_nan = false;
    if (any_nan_) {
      for (int i = 0; i < tree->num_leaves() - 1 ; ++i) {
        // use split_feature because split_feature_inner doesn't work when refitting existing tree
        if (contains_nan_[this->train_data_->InnerFeatureIndex(tree->split_feature(i))]) {
          has_nan = true;
          break;
        }
      }
    }
    if (has_nan) {
      AddPredictionToScoreInner<true>(tree, out_score);
    } else {
      AddPredictionToScoreInner<false>(tree, out_score);
    }
  }

  template<bool HAS_NAN>
  void AddPredictionToScoreInner(const Tree* tree, double* out_score) const {
    int num_leaves = tree->num_leaves();
    std::vector<double> leaf_const(num_leaves);
    std::vector<std::vector<double>> leaf_coeff(num_leaves);
    std::vector<std::vector<const float*>> feat_ptr(num_leaves);
    std::vector<double> leaf_output(num_leaves);
    std::vector<int> leaf_num_features(num_leaves);
    for (int leaf_num = 0; leaf_num < num_leaves; ++leaf_num) {
      leaf_const[leaf_num] = tree->LeafConst(leaf_num);
      leaf_coeff[leaf_num] = tree->LeafCoeffs(leaf_num);
      leaf_output[leaf_num] = tree->LeafOutput(leaf_num);
      for (int feat : tree->LeafFeaturesInner(leaf_num)) {
        feat_ptr[leaf_num].push_back(this->train_data_->raw_index(feat));
      }
      leaf_num_features[leaf_num] = static_cast<int>(feat_ptr[leaf_num].size());
    }
    OMP_INIT_EX();
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) if (this->num_data_ > 1024)
    for (int i = 0; i < this->num_data_; ++i) {
      OMP_LOOP_EX_BEGIN();
      int leaf_num = leaf_map_[i];
      if (leaf_num < 0) {
        continue;
      }
      double output = leaf_const[leaf_num];
      int num_feat = leaf_num_features[leaf_num];
      if (HAS_NAN) {
        bool nan_found = false;
        for (int feat_ind = 0; feat_ind < num_feat; ++feat_ind) {
          float val = feat_ptr[leaf_num][feat_ind][i];
          if (std::isnan(val)) {
            nan_found = true;
            break;
          }
          output += val * leaf_coeff[leaf_num][feat_ind];
        }
        if (nan_found) {
          out_score[i] += leaf_output[leaf_num];
        } else {
          out_score[i] += output;
        }
      } else {
        for (int feat_ind = 0; feat_ind < num_feat; ++feat_ind) {
          output += feat_ptr[leaf_num][feat_ind][i] * leaf_coeff[leaf_num][feat_ind];
        }
        out_score[i] += output;
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }

 protected:
  /*! \brief whether numerical features contain any nan values */
  std::vector<int8_t> contains_nan_;
  /*! whether any numerical feature contains a nan value */
  bool any_nan_;
  /*! \brief map dataset to leaves */
  mutable std::vector<int> leaf_map_;
  /*! \brief temporary storage for calculating linear model coefficients */
  mutable std::vector<std::vector<double>> XTHX_;
  mutable std::vector<std::vector<double>> XTg_;
  mutable std::vector<std::vector<std::vector<double>>> XTHX_by_thread_;
  mutable std::vector<std::vector<std::vector<double>>> XTg_by_thread_;
};

}  // namespace LightGBM
#endif   // LIGHTGBM_SRC_TREELEARNER_LINEAR_TREE_LEARNER_H_
