/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_BOOSTING_DART_HPP_
#define LIGHTGBM_SRC_BOOSTING_DART_HPP_

#include <LightGBM/boosting.h>

#include <string>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

#include "gbdt.h"
#include "score_updater.hpp"

namespace LightGBM {
/*!
* \brief DART algorithm implementation. including Training, prediction, bagging.
*/
class DART: public GBDT {
 public:
  /*!
  * \brief Constructor
  */
  DART() : GBDT() { }
  /*!
  * \brief Destructor
  */
  ~DART() { }
  /*!
  * \brief Initialization logic
  * \param config Config for boosting
  * \param train_data Training data
  * \param objective_function Training objective function
  * \param training_metrics Training metrics
  */
  void Init(const Config* config, const Dataset* train_data,
            const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {
    GBDT::Init(config, train_data, objective_function, training_metrics);
    random_for_drop_ = Random(config_->drop_seed);
    sum_weight_ = 0.0f;
  }

  void ResetConfig(const Config* config) override {
    GBDT::ResetConfig(config);
    random_for_drop_ = Random(config_->drop_seed);
    sum_weight_ = 0.0f;
  }

  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian) override {
    is_update_score_cur_iter_ = false;
    bool ret = GBDT::TrainOneIter(gradient, hessian);
    if (ret) {
      return ret;
    }
    // normalize
    Normalize();
    if (!config_->uniform_drop) {
      tree_weight_.push_back(shrinkage_rate_);
      sum_weight_ += shrinkage_rate_;
    }
    return false;
  }

  /*!
  * \brief Get current training score
  * \param out_len length of returned score
  * \return training score
  */
  const double* GetTrainingScore(int64_t* out_len) override {
    if (!is_update_score_cur_iter_) {
      // only drop one time in one iteration
      DroppingTrees();
      is_update_score_cur_iter_ = true;
    }
    *out_len = static_cast<int64_t>(train_score_updater_->num_data()) * num_class_;
    return train_score_updater_->score();
  }

  bool EvalAndCheckEarlyStopping() override {
    GBDT::OutputMetric(iter_);
    return false;
  }

 private:
  /*!
  * \brief drop trees based on drop_rate
  */
  void DroppingTrees() {
    drop_index_.clear();
    bool is_skip = random_for_drop_.NextFloat() < config_->skip_drop;
    // select dropping tree indices based on drop_rate and tree weights
    if (!is_skip) {
      double drop_rate = config_->drop_rate;
      if (!config_->uniform_drop) {
        double inv_average_weight = static_cast<double>(tree_weight_.size()) / sum_weight_;
        if (config_->max_drop > 0) {
          drop_rate = std::min(drop_rate, config_->max_drop * inv_average_weight / sum_weight_);
        }
        for (int i = 0; i < iter_; ++i) {
          if (random_for_drop_.NextFloat() < drop_rate * tree_weight_[i] * inv_average_weight) {
            drop_index_.push_back(num_init_iteration_ + i);
            if (drop_index_.size() >= static_cast<size_t>(config_->max_drop)) {
              break;
            }
          }
        }
      } else {
        if (config_->max_drop > 0) {
          drop_rate = std::min(drop_rate, config_->max_drop / static_cast<double>(iter_));
        }
        for (int i = 0; i < iter_; ++i) {
          if (random_for_drop_.NextFloat() < drop_rate) {
            drop_index_.push_back(num_init_iteration_ + i);
            if (drop_index_.size() >= static_cast<size_t>(config_->max_drop)) {
              break;
            }
          }
        }
      }
    }
    // drop trees
    for (auto i : drop_index_) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
        models_[curr_tree]->Shrinkage(-1.0);
        train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
    }
    if (!config_->xgboost_dart_mode) {
      shrinkage_rate_ = config_->learning_rate / (1.0f + static_cast<double>(drop_index_.size()));
    } else {
      if (drop_index_.empty()) {
        shrinkage_rate_ = config_->learning_rate;
      } else {
        shrinkage_rate_ = config_->learning_rate / (config_->learning_rate + static_cast<double>(drop_index_.size()));
      }
    }
  }
  /*!
  * \brief normalize dropped trees
  * NOTE: num_drop_tree(k), learning_rate(lr), shrinkage_rate_ = lr / (k + 1)
  *       step 1: shrink tree to -1 -> drop tree
  *       step 2: shrink tree to k / (k + 1) - 1 from -1, by 1/(k+1)
  *               -> normalize for valid data
  *       step 3: shrink tree to k / (k + 1) from k / (k + 1) - 1, by -k
  *               -> normalize for train data
  *       end with tree weight = (k / (k + 1)) * old_weight
  */
  void Normalize() {
    double k = static_cast<double>(drop_index_.size());
    if (!config_->xgboost_dart_mode) {
      for (auto i : drop_index_) {
        for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
          auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
          // update validation score
          models_[curr_tree]->Shrinkage(1.0f / (k + 1.0f));
          for (auto& score_updater : valid_score_updater_) {
            score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
          }
          // update training score
          models_[curr_tree]->Shrinkage(-k);
          train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
        }
        if (!config_->uniform_drop) {
          sum_weight_ -= tree_weight_[i - num_init_iteration_] * (1.0f / (k + 1.0f));
          tree_weight_[i - num_init_iteration_] *= (k / (k + 1.0f));
        }
      }
    } else {
      for (auto i : drop_index_) {
        for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
          auto curr_tree = i * num_tree_per_iteration_ + cur_tree_id;
          // update validation score
          models_[curr_tree]->Shrinkage(shrinkage_rate_);
          for (auto& score_updater : valid_score_updater_) {
            score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
          }
          // update training score
          models_[curr_tree]->Shrinkage(-k / config_->learning_rate);
          train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
        }
        if (!config_->uniform_drop) {
          sum_weight_ -= tree_weight_[i - num_init_iteration_] * (1.0f / (k + config_->learning_rate));;
          tree_weight_[i - num_init_iteration_] *= (k / (k + config_->learning_rate));
        }
      }
    }
  }
  /*! \brief The weights of all trees, used to choose drop trees */
  std::vector<double> tree_weight_;
  /*! \brief sum weights of all trees */
  double sum_weight_;
  /*! \brief The indices of dropping trees */
  std::vector<int> drop_index_;
  /*! \brief Random generator, used to select dropping trees */
  Random random_for_drop_;
  /*! \brief Flag that the score is update on current iter or not*/
  bool is_update_score_cur_iter_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_BOOSTING_DART_HPP_
