/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_BOOSTING_MVS_H_
#define LIGHTGBM_BOOSTING_MVS_H_

#include <LightGBM/boosting.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>

#include "gbdt.h"
#include "score_updater.hpp"

namespace LightGBM {

class MVS : public GBDT {
 public:
  MVS();

  ~MVS() {
  }

  void Init(const Config *config, const Dataset *train_data, const ObjectiveFunction *objective_function,
            const std::vector<const Metric *> &training_metrics) override {
    GBDT::Init(config, train_data, objective_function, training_metrics);
    mvs_lambda_ = config_->mvs_lambda;
    mvs_adaptive_ = config_->mvs_adaptive;
    ResetMVS();
    if (objective_function_ == nullptr) {
      // use customized objective function
      size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
      gradients_.resize(total_size, 0.0f);
      hessians_.resize(total_size, 0.0f);
    }
  }

  void ResetTrainingData(const Dataset *train_data, const ObjectiveFunction *objective_function,
                         const std::vector<const Metric *> &training_metrics) override {
    GBDT::ResetTrainingData(train_data, objective_function, training_metrics);
    ResetMVS();
  }

  void ResetConfig(const Config *config) override {
    GBDT::ResetConfig(config);
    need_re_bagging_ = mvs_adaptive_ != config->mvs_adaptive
        || (mvs_lambda_ != config->mvs_lambda && !mvs_adaptive_ && !config->mvs_adaptive);
    mvs_lambda_ = config_->mvs_lambda;
    mvs_adaptive_ = config_->mvs_adaptive;
    ResetMVS();
  }

  void ResetMVS();

  bool TrainOneIter(const score_t *gradients, const score_t *hessians) override {
    if (gradients != nullptr) {
      // use customized objective function
      CHECK(hessians != nullptr && objective_function_ == nullptr);
      int64_t total_size = static_cast<int64_t>(num_data_) * num_tree_per_iteration_;
      #pragma omp parallel for schedule(static, 1024)
      for (int64_t i = 0; i < total_size; ++i) {
        gradients_[i] = gradients[i];
        hessians_[i] = hessians[i];
      }
      return GBDT::TrainOneIter(gradients_.data(), hessians_.data());
    } else {
      CHECK(hessians == nullptr);
      return GBDT::TrainOneIter(nullptr, nullptr);
    }
  }

  data_size_t BaggingHelper(data_size_t start, data_size_t cnt, data_size_t *buffer) override;


  void Bagging(int iter) override;

  static constexpr double kMVSEps = 1e-20;

 protected:
  bool GetIsConstHessian(const ObjectiveFunction *) override {
    return false;
  }

  double GetThreshold(data_size_t begin, data_size_t end);

  double GetLambda();

  double mvs_lambda_;
  double threshold_{0.0};
  std::vector<score_t> tmp_derivatives_;
  bool mvs_adaptive_;
};
}  // namespace LightGBM
#endif   // LIGHTGBM_BOOSTING_MVS_H_
