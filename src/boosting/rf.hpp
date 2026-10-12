/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_BOOSTING_RF_HPP_
#define LIGHTGBM_SRC_BOOSTING_RF_HPP_

#include <LightGBM/boosting.h>
#include <LightGBM/metric.h>

#include <string>
#include <cstdio>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "gbdt.h"
#include "score_updater.hpp"

namespace LightGBM {
/*!
* \brief Random Forest implementation
*/
class RF : public GBDT {
 public:
  RF() : GBDT() {
    average_output_ = true;
  }

  ~RF() {}

  void Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
    const std::vector<const Metric*>& training_metrics) override {
    if (config->data_sample_strategy == std::string("bagging")) {
      CHECK((config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f) ||
            (config->feature_fraction < 1.0f && config->feature_fraction > 0.0f));
    } else {
      CHECK_EQ(config->data_sample_strategy, std::string("goss"));
    }
    GBDT::Init(config, train_data, objective_function, training_metrics);

    if (num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        MultiplyScore(cur_tree_id, 1.0f / num_init_iteration_);
      }
    } else {
      CHECK_EQ(train_data->metadata().init_score(), nullptr);
    }
    CHECK_EQ(num_tree_per_iteration_, num_class_);
    // not shrinkage rate for the RF
    shrinkage_rate_ = 1.0f;
    // only boosting one time
    Boosting();
    if (data_sample_strategy_->is_use_subset() && data_sample_strategy_->bag_data_cnt() < num_data_) {
      tmp_grad_.resize(num_data_);
      tmp_hess_.resize(num_data_);
    }
  }

  void ResetConfig(const Config* config) override {
    if (config->data_sample_strategy == std::string("bagging")) {
      CHECK((config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f) ||
            (config->feature_fraction < 1.0f && config->feature_fraction > 0.0f));
    } else {
      CHECK_EQ(config->data_sample_strategy, std::string("goss"));
    }
    GBDT::ResetConfig(config);
    // not shrinkage rate for the RF
    shrinkage_rate_ = 1.0f;
  }

  void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
    const std::vector<const Metric*>& training_metrics) override {
    GBDT::ResetTrainingData(train_data, objective_function, training_metrics);
    if (iter_ + num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        train_score_updater_->MultiplyScore(1.0f / (iter_ + num_init_iteration_), cur_tree_id);
      }
    }
    CHECK_EQ(num_tree_per_iteration_, num_class_);
    // only boosting one time
    Boosting();
    if (data_sample_strategy_->is_use_subset() && data_sample_strategy_->bag_data_cnt() < num_data_) {
      tmp_grad_.resize(num_data_);
      tmp_hess_.resize(num_data_);
    }
  }

  void Boosting() override {
    if (objective_function_ == nullptr) {
      Log::Fatal("RF mode do not support custom objective function, please use built-in objectives.");
    }
    init_scores_.resize(num_tree_per_iteration_, 0.0);
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      init_scores_[cur_tree_id] = BoostFromAverage(cur_tree_id, false);
    }
    size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
    std::vector<double> tmp_scores(total_size, 0.0f);
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int j = 0; j < num_tree_per_iteration_; ++j) {
      size_t offset = static_cast<size_t>(j)* num_data_;
      for (data_size_t i = 0; i < num_data_; ++i) {
        tmp_scores[offset + i] = init_scores_[j];
      }
    }
    objective_function_->
      GetGradients(tmp_scores.data(), gradients_.data(), hessians_.data());
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override {
    // bagging logic
    data_sample_strategy_ ->Bagging(iter_, tree_learner_.get(), gradients_.data(), hessians_.data());
    const bool is_use_subset = data_sample_strategy_->is_use_subset();
    const data_size_t bag_data_cnt = data_sample_strategy_->bag_data_cnt();
    const std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>>& bag_data_indices = data_sample_strategy_->bag_data_indices();

    // GOSSStrategy->Bagging may modify value of bag_data_cnt_
    if (is_use_subset && bag_data_cnt < num_data_) {
      tmp_grad_.resize(num_data_);
      tmp_hess_.resize(num_data_);
    }

    CHECK_EQ(gradients, nullptr);
    CHECK_EQ(hessians, nullptr);

    gradients = gradients_.data();
    hessians = hessians_.data();
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      std::unique_ptr<Tree> new_tree(new Tree(2, false, false));
      size_t offset = static_cast<size_t>(cur_tree_id)* num_data_;
      if (class_need_train_[cur_tree_id]) {
        auto grad = gradients + offset;
        auto hess = hessians + offset;

        if (is_use_subset && bag_data_cnt < num_data_ && !boosting_on_gpu_) {
          for (int i = 0; i < bag_data_cnt; ++i) {
            tmp_grad_[i] = grad[bag_data_indices[i]];
            tmp_hess_[i] = hess[bag_data_indices[i]];
          }
          grad = tmp_grad_.data();
          hess = tmp_hess_.data();
        }

        new_tree.reset(tree_learner_->Train(grad, hess, false));
      }

      if (new_tree->num_leaves() > 1) {
        double pred = init_scores_[cur_tree_id];
        auto residual_getter = [pred](const label_t* label, int i) {return static_cast<double>(label[i]) - pred; };
        tree_learner_->RenewTreeOutput(new_tree.get(), objective_function_, residual_getter,
          num_data_, bag_data_indices.data(), bag_data_cnt, train_score_updater_->score());
        if (std::fabs(init_scores_[cur_tree_id]) > kEpsilon) {
          new_tree->AddBias(init_scores_[cur_tree_id]);
        }
        // update score
        MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
        UpdateScore(new_tree.get(), cur_tree_id);
        MultiplyScore(cur_tree_id, 1.0 / (iter_ + num_init_iteration_ + 1));
      } else {
        // only add default score one-time
        if (models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
          double output = 0.0;
          if (!class_need_train_[cur_tree_id]) {
            if (objective_function_ != nullptr) {
              output = objective_function_->BoostFromScore(cur_tree_id);
            } else {
              output = init_scores_[cur_tree_id];
            }
          }
          new_tree->AsConstantTree(output, num_data_);
          MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
          UpdateScore(new_tree.get(), cur_tree_id);
          MultiplyScore(cur_tree_id, 1.0 / (iter_ + num_init_iteration_ + 1));
        }
      }
      // add model
      models_.push_back(std::move(new_tree));
    }
    ++iter_;
    return false;
  }

  void RollbackOneIter() override {
    if (iter_ <= 0) {
      return;
    }
    int cur_iter = iter_ + num_init_iteration_ - 1;
    // reset score
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      auto curr_tree = cur_iter * num_tree_per_iteration_ + cur_tree_id;
      models_[curr_tree]->Shrinkage(-1.0);
      MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
      train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      for (auto& score_updater : valid_score_updater_) {
        score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
      MultiplyScore(cur_tree_id, 1.0f / (iter_ + num_init_iteration_ - 1));
    }
    // remove model
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      models_.pop_back();
    }
    --iter_;
  }

  void MultiplyScore(const int cur_tree_id, double val) {
    train_score_updater_->MultiplyScore(val, cur_tree_id);
    for (auto& score_updater : valid_score_updater_) {
      score_updater->MultiplyScore(val, cur_tree_id);
    }
  }

  void AddValidDataset(const Dataset* valid_data,
    const std::vector<const Metric*>& valid_metrics) override {
    GBDT::AddValidDataset(valid_data, valid_metrics);
    if (iter_ + num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        valid_score_updater_.back()->MultiplyScore(1.0f / (iter_ + num_init_iteration_), cur_tree_id);
      }
    }
  }

  bool NeedAccuratePrediction() const override {
    // No early stopping for prediction
    return true;
  };

 private:
  std::vector<score_t> tmp_grad_;
  std::vector<score_t> tmp_hess_;
  std::vector<double> init_scores_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_BOOSTING_RF_HPP_
