/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_SRC_BOOSTING_GOSS_HPP_
#define LIGHTGBM_SRC_BOOSTING_GOSS_HPP_

#include <LightGBM/utils/array_args.h>
#include <LightGBM/sample_strategy.h>

#include <algorithm>
#include <string>
#include <vector>

namespace LightGBM {

class GOSSStrategy : public SampleStrategy {
 public:
  GOSSStrategy(const Config* config, const Dataset* train_data, int num_tree_per_iteration) {
    config_ = config;
    train_data_ = train_data;
    num_tree_per_iteration_ = num_tree_per_iteration;
    num_data_ = train_data->num_data();
  }

  ~GOSSStrategy() {
  }

  void Bagging(int iter, TreeLearner* tree_learner, score_t* gradients, score_t* hessians) override {
    bag_data_cnt_ = num_data_;
    // not subsample for first iterations
    if (iter < static_cast<int>(1.0f / config_->learning_rate)) {
      return;
    }
    auto left_cnt = bagging_runner_.Run<true>(
        num_data_,
        [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t* left,
            data_size_t*) {
          data_size_t cur_left_count = 0;
          cur_left_count = Helper(cur_start, cur_cnt, left, gradients, hessians);
          return cur_left_count;
        },
        bag_data_indices_.data());
    bag_data_cnt_ = left_cnt;
    // set bagging data to tree learner
    if (!is_use_subset_) {
      #ifdef USE_CUDA
      if (config_->device_type == std::string("cuda")) {
        CopyFromHostToCUDADevice<data_size_t>(cuda_bag_data_indices_.RawData(), bag_data_indices_.data(), static_cast<size_t>(num_data_), __FILE__, __LINE__);
        tree_learner->SetBaggingData(nullptr, cuda_bag_data_indices_.RawData(), bag_data_cnt_);
      } else {
      #endif  // USE_CUDA
        tree_learner->SetBaggingData(nullptr, bag_data_indices_.data(), bag_data_cnt_);
      #ifdef USE_CUDA
      }
      #endif  // USE_CUDA
    } else {
      // get subset
      tmp_subset_->ReSize(bag_data_cnt_);
      tmp_subset_->CopySubrow(train_data_, bag_data_indices_.data(),
                              bag_data_cnt_, false);
      #ifdef USE_CUDA
      if (config_->device_type == std::string("cuda")) {
        CopyFromHostToCUDADevice<data_size_t>(cuda_bag_data_indices_.RawData(), bag_data_indices_.data(), static_cast<size_t>(num_data_), __FILE__, __LINE__);
        tree_learner->SetBaggingData(tmp_subset_.get(), cuda_bag_data_indices_.RawData(),
                                      bag_data_cnt_);
      } else {
      #endif  // USE_CUDA
        tree_learner->SetBaggingData(tmp_subset_.get(), bag_data_indices_.data(),
                                     bag_data_cnt_);
      #ifdef USE_CUDA
      }
      #endif  // USE_CUDA
    }
  }

  void ResetSampleConfig(const Config* config, bool /*is_change_dataset*/) override {
    // Cannot use bagging in GOSS
    config_ = config;
    need_resize_gradients_ = false;
    if (objective_function_ == nullptr) {
      // resize gradient vectors to copy the customized gradients for goss
      need_resize_gradients_ = true;
    }

    CHECK_LE(config_->top_rate + config_->other_rate, 1.0f);
    CHECK(config_->top_rate > 0.0f && config_->other_rate > 0.0f);
    if (config_->bagging_freq > 0 && config_->bagging_fraction != 1.0f) {
      Log::Fatal("Cannot use bagging in GOSS");
    }
    Log::Info("Using GOSS");
    balanced_bagging_ = false;
    bag_data_indices_.resize(num_data_);
    bagging_runner_.ReSize(num_data_);
    bagging_rands_.clear();
    for (int i = 0;
         i < (num_data_ + bagging_rand_block_ - 1) / bagging_rand_block_; ++i) {
      bagging_rands_.emplace_back(config_->bagging_seed + i);
    }
    is_use_subset_ = false;
    if (config_->top_rate + config_->other_rate <= 0.5) {
      auto bag_data_cnt = static_cast<data_size_t>((config_->top_rate + config_->other_rate) * num_data_);
      bag_data_cnt = std::max(1, bag_data_cnt);
      tmp_subset_.reset(new Dataset(bag_data_cnt));
      tmp_subset_->CopyFeatureMapperFrom(train_data_);
      is_use_subset_ = true;
    }
    // flag to not bagging first
    bag_data_cnt_ = num_data_;
  }

  bool IsHessianChange() const override {
    return true;
  }

 private:
  data_size_t Helper(data_size_t start, data_size_t cnt, data_size_t* buffer, score_t* gradients, score_t* hessians) {
    if (cnt <= 0) {
      return 0;
    }
    std::vector<score_t> tmp_gradients(cnt, 0.0f);
    for (data_size_t i = 0; i < cnt; ++i) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
        tmp_gradients[i] += std::fabs(gradients[idx] * hessians[idx]);
      }
    }
    data_size_t top_k = static_cast<data_size_t>(cnt * config_->top_rate);
    data_size_t other_k = static_cast<data_size_t>(cnt * config_->other_rate);
    top_k = std::max(1, top_k);
    ArrayArgs<score_t>::ArgMaxAtK(&tmp_gradients, 0, static_cast<int>(tmp_gradients.size()), top_k - 1);
    score_t threshold = tmp_gradients[top_k - 1];

    score_t multiply = static_cast<score_t>(cnt - top_k) / other_k;
    data_size_t cur_left_cnt = 0;
    data_size_t cur_right_pos = cnt;
    data_size_t big_weight_cnt = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      auto cur_idx = start + i;
      score_t grad = 0.0f;
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + cur_idx;
        grad += std::fabs(gradients[idx] * hessians[idx]);
      }
      if (grad >= threshold) {
        buffer[cur_left_cnt++] = cur_idx;
        ++big_weight_cnt;
      } else {
        data_size_t sampled = cur_left_cnt - big_weight_cnt;
        data_size_t rest_need = other_k - sampled;
        data_size_t rest_all = (cnt - i) - (top_k - big_weight_cnt);
        double prob = (rest_need) / static_cast<double>(rest_all);
        if (bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() < prob) {
          buffer[cur_left_cnt++] = cur_idx;
          for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
            size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + cur_idx;
            gradients[idx] *= multiply;
            hessians[idx] *= multiply;
          }
        } else {
          buffer[--cur_right_pos] = cur_idx;
        }
      }
    }
    return cur_left_cnt;
  }
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_BOOSTING_GOSS_HPP_
