/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "mvs.hpp"

#include <memory>

namespace LightGBM {

using ConstTreeIterator = std::vector<std::unique_ptr<Tree>>::const_iterator;

MVS::MVS() : GBDT() {}

static double ComputeLeavesMeanSquaredValue(ConstTreeIterator begin,
                                            ConstTreeIterator end,
                                            const data_size_t num_leaves) {
  double sum_values = 0.0;
#pragma omp parallel for schedule(static, 2048) reduction(+ : sum_values)
  for (data_size_t leaf_idx = 0; leaf_idx < num_leaves; ++leaf_idx) {
    double leave_value = 0.0;
    for (ConstTreeIterator it = begin; it != end; ++it) {
      if (leaf_idx < (**it).num_leaves()) {
        const double value = (*it)->LeafOutput(leaf_idx);
        leave_value += value * value;
      }
    }
    sum_values += std::sqrt(leave_value);
  }
  return sum_values / num_leaves;
}

static double ComputeMeanGradValues(score_t *gradients, score_t *hessians,
                                    data_size_t size,
                                    data_size_t num_tree_per_iteration) {
  double sum = 0.0;
#pragma omp parallel for schedule(static, 1024) reduction(+ : sum)
  for (data_size_t i = 0; i < size; ++i) {
    double local_hessians = 0.0, local_gradients = 0.0;
    for (data_size_t j = 0; j < num_tree_per_iteration; ++j) {
      size_t idx = static_cast<size_t>(size) * j + i;
      local_hessians += hessians[idx] * hessians[idx];
      local_gradients += gradients[idx] * gradients[idx];
    }
    sum += std::sqrt(local_gradients / local_hessians);
  }
  return sum / size;
}

double MVS::GetLambda() {
  if (!mvs_adaptive_) {
    return mvs_lambda_;
  }
  if (this->iter_ > 0) {
    return ComputeLeavesMeanSquaredValue(models_.cend() - num_tree_per_iteration_,
                                         models_.cend(), config_->num_leaves);
  }
  return ComputeMeanGradValues(gradients_.data(), hessians_.data(), num_data_,
                               num_tree_per_iteration_);
}

void MVS::Bagging(int iter) {
  if (iter % config_->bagging_freq != 0 && !need_re_bagging_) {
    return;
  }
  need_re_bagging_ = false;
  bag_data_cnt_ = num_data_;
  mvs_lambda_ = GetLambda();

  //#pragma omp parallel for schedule(static, 1024)
  for (data_size_t i = 0; i < num_data_; ++i) {
    tmp_derivatives_[i] = 0.0f;
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + i;
      tmp_derivatives_[i] += gradients_[idx] * gradients_[idx] + mvs_lambda_ * hessians_[idx] * hessians_[idx];
    }
    tmp_derivatives_[i] = std::sqrt(tmp_derivatives_[i]);
  }

  if (num_data_ <= config_->mvs_max_sequential_size) {
    threshold_ = GetThreshold(0, num_data_);
  }

  auto left_cnt = bagging_runner_.Run<true>(
      num_data_,
      [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t *left,
          data_size_t *) {
        data_size_t left_count = BaggingHelper(cur_start, cur_cnt, left);
        return left_count;
      },
      bag_data_indices_.data());

  bag_data_cnt_ = left_cnt;
  if (!is_use_subset_) {
    tree_learner_->SetBaggingData(nullptr, bag_data_indices_.data(), bag_data_cnt_);
  } else {
    tmp_subset_->ReSize(bag_data_cnt_);
    tmp_subset_->CopySubrow(train_data_, bag_data_indices_.data(),
                            bag_data_cnt_, false);
    tree_learner_->SetBaggingData(tmp_subset_.get(), bag_data_indices_.data(),
                                  bag_data_cnt_);
  }
  threshold_ = 0.0;
  Log::Debug("MVS Sample size %d %d", left_cnt, static_cast<data_size_t>(config_->bagging_fraction * num_data_));
}

data_size_t MVS::BaggingHelper(data_size_t start, data_size_t cnt, data_size_t *buffer) {
  if (cnt <= 0) {
    return 0;
  }

  const double threshold = GetThreshold(start, cnt);

  data_size_t left_cnt = 0;
  data_size_t right_pos = cnt;
  data_size_t big_weight_cnt = 0;
  for (data_size_t i = 0; i < cnt; ++i) {
    data_size_t position = start + i;

    double derivative = 0.0;
    for (data_size_t j = 0; j < num_tree_per_iteration_; ++j) {
      size_t idx = static_cast<size_t>(j) * num_data_ + position;
      derivative += gradients_[idx] * gradients_[idx] + mvs_lambda_ * hessians_[idx] * hessians_[idx];
    }
    derivative = std::sqrt(derivative);

    if (derivative >= threshold) {
      buffer[left_cnt++] = position;
      ++big_weight_cnt;
    } else {
      const double proba_threshold = derivative / threshold;
      const double proba = bagging_rands_[position / bagging_rand_block_].NextFloat();
      if (proba < proba_threshold) {
        buffer[left_cnt++] = position;
        for (data_size_t tree_id = 0; tree_id < num_tree_per_iteration_; ++tree_id) {
          size_t idx = static_cast<size_t>(num_data_) * tree_id + position;
          gradients_[idx] /= proba_threshold;
          hessians_[idx] /= proba_threshold;
        }
      } else {
        buffer[--right_pos] = position;
      }
    }
  }

  return left_cnt;
}

double MVS::GetThreshold(data_size_t begin, data_size_t cnt) {
  if (num_data_ <= config_->mvs_max_sequential_size && threshold_ != 0.0) {
    return threshold_;
  }

  double threshold = ArrayArgs<score_t>::CalculateThresholdMVS(&tmp_derivatives_, begin, begin + cnt,
                                                  cnt * config_->bagging_fraction);
  return threshold;
}

void MVS::ResetMVS() {
  CHECK(config_->bagging_fraction > 0.0f && config_->bagging_fraction < 1.0f && config_->bagging_freq > 0);
  CHECK(config_->mvs_lambda >= 0.0f);
  CHECK(!balanced_bagging_);
  bag_data_indices_.resize(num_data_);
  tmp_derivatives_.resize(num_data_);
  Log::Info("Using MVS");
}

}  // namespace LightGBM
