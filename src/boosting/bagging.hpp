/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_SRC_BOOSTING_BAGGING_HPP_
#define LIGHTGBM_SRC_BOOSTING_BAGGING_HPP_

#include <string>
#include <vector>

namespace LightGBM {

class BaggingSampleStrategy : public SampleStrategy {
 public:
  BaggingSampleStrategy(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function, int num_tree_per_iteration)
    : need_re_bagging_(false) {
    config_ = config;
    train_data_ = train_data;
    num_data_ = train_data->num_data();
    num_queries_ = train_data->metadata().num_queries();
    query_boundaries_ = train_data->metadata().query_boundaries();
    objective_function_ = objective_function;
    num_tree_per_iteration_ = num_tree_per_iteration;
    num_threads_ = OMP_NUM_THREADS();
  }

  ~BaggingSampleStrategy() {}

  void Bagging(int iter, TreeLearner* tree_learner, score_t* /*gradients*/, score_t* /*hessians*/) override {
    Common::FunctionTimer fun_timer("GBDT::Bagging", global_timer);
    // if need bagging
    if ((bag_data_cnt_ < num_data_ && iter % config_->bagging_freq == 0) ||
      need_re_bagging_) {
      need_re_bagging_ = false;
      if (!config_->bagging_by_query) {
        auto left_cnt = bagging_runner_.Run<true>(
          num_data_,
          [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t* left,
              data_size_t*) {
            data_size_t cur_left_count = 0;
            if (balanced_bagging_) {
              cur_left_count =
                  BalancedBaggingHelper(cur_start, cur_cnt, left);
            } else {
              cur_left_count = BaggingHelper(cur_start, cur_cnt, left);
            }
            return cur_left_count;
          },
          bag_data_indices_.data());
        bag_data_cnt_ = left_cnt;
      } else {
        num_sampled_queries_ = bagging_runner_.Run<true>(
          num_queries_,
          [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t* left,
              data_size_t*) {
            data_size_t cur_left_count = 0;
            cur_left_count = BaggingHelper(cur_start, cur_cnt, left);
            return cur_left_count;
          }, bag_query_indices_.data());

        sampled_query_boundaries_[0] = 0;
        OMP_INIT_EX();
        #pragma omp parallel for schedule(static) num_threads(num_threads_)
        for (data_size_t i = 0; i < num_sampled_queries_; ++i) {
          OMP_LOOP_EX_BEGIN();
          sampled_query_boundaries_[i + 1] = query_boundaries_[bag_query_indices_[i] + 1] - query_boundaries_[bag_query_indices_[i]];
          OMP_LOOP_EX_END();
        }
        OMP_THROW_EX();

        const int num_blocks = Threading::For<data_size_t>(0, num_sampled_queries_ + 1, 128, [this](int thread_index, data_size_t start_index, data_size_t end_index) {
          for (data_size_t i = start_index + 1; i < end_index; ++i) {
            sampled_query_boundaries_[i] += sampled_query_boundaries_[i - 1];
          }
          sampled_query_boundaries_thread_buffer_[thread_index] = sampled_query_boundaries_[end_index - 1];
         });

        for (int thread_index = 1; thread_index < num_blocks; ++thread_index) {
          sampled_query_boundaries_thread_buffer_[thread_index] += sampled_query_boundaries_thread_buffer_[thread_index - 1];
        }

        Threading::For<data_size_t>(0, num_sampled_queries_ + 1, 128, [this](int thread_index, data_size_t start_index, data_size_t end_index) {
          if (thread_index > 0) {
            for (data_size_t i = start_index; i < end_index; ++i) {
              sampled_query_boundaries_[i] += sampled_query_boundaries_thread_buffer_[thread_index - 1];
            }
          }
        });

        bag_data_cnt_ = sampled_query_boundaries_[num_sampled_queries_];

        Threading::For<data_size_t>(0, num_sampled_queries_, 1, [this](int /*thread_index*/, data_size_t start_index, data_size_t end_index) {
          for (data_size_t sampled_query_id = start_index; sampled_query_id < end_index; ++sampled_query_id) {
            const data_size_t query_index = bag_query_indices_[sampled_query_id];
            const data_size_t data_index_start = query_boundaries_[query_index];
            const data_size_t data_index_end = query_boundaries_[query_index + 1];
            const data_size_t sampled_query_start = sampled_query_boundaries_[sampled_query_id];
            for (data_size_t i = data_index_start; i < data_index_end; ++i) {
              bag_data_indices_[sampled_query_start + i - data_index_start] = i;
            }
          }
        });
      }
      Log::Debug("Re-bagging, using %d data to train", bag_data_cnt_);
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
  }

  void ResetSampleConfig(const Config* config, bool is_change_dataset) override {
    need_resize_gradients_ = false;
    // if need bagging, create buffer
    data_size_t num_pos_data = 0;
    if (objective_function_ != nullptr) {
      num_pos_data = objective_function_->NumPositiveData();
    }
    bool balance_bagging_cond = (config->pos_bagging_fraction < 1.0 || config->neg_bagging_fraction < 1.0) && (num_pos_data > 0);
    if ((config->bagging_fraction < 1.0 || balance_bagging_cond) && config->bagging_freq > 0) {
      need_re_bagging_ = false;
      if (!is_change_dataset &&
        config_ != nullptr && config_->bagging_fraction == config->bagging_fraction && config_->bagging_freq == config->bagging_freq
        && config_->pos_bagging_fraction == config->pos_bagging_fraction && config_->neg_bagging_fraction == config->neg_bagging_fraction) {
        config_ = config;
        return;
      }
      config_ = config;
      if (balance_bagging_cond) {
        balanced_bagging_ = true;
        bag_data_cnt_ = static_cast<data_size_t>(num_pos_data * config_->pos_bagging_fraction)
                        + static_cast<data_size_t>((num_data_ - num_pos_data) * config_->neg_bagging_fraction);
      } else {
        bag_data_cnt_ = static_cast<data_size_t>(config_->bagging_fraction * num_data_);
      }
      bag_data_indices_.resize(num_data_);
      #ifdef USE_CUDA
      if (config_->device_type == std::string("cuda")) {
        cuda_bag_data_indices_.Resize(num_data_);
      }
      #endif  // USE_CUDA
      if (!config_->bagging_by_query) {
        bagging_runner_.ReSize(num_data_);
      } else {
        bagging_runner_.ReSize(num_queries_);
        sampled_query_boundaries_.resize(num_queries_ + 1, 0);
        sampled_query_boundaries_thread_buffer_.resize(num_threads_, 0);
        bag_query_indices_.resize(num_data_);
      }
      bagging_rands_.clear();
      for (int i = 0;
          i < (num_data_ + bagging_rand_block_ - 1) / bagging_rand_block_; ++i) {
        bagging_rands_.emplace_back(config_->bagging_seed + i);
      }

      double average_bag_rate =
          (static_cast<double>(bag_data_cnt_) / num_data_) / config_->bagging_freq;
      is_use_subset_ = false;
      if (config_->device_type != std::string("cuda")) {
        const int group_threshold_usesubset = 100;
        const double average_bag_rate_threshold = 0.5;
        if (average_bag_rate <= average_bag_rate_threshold
            && (train_data_->num_feature_groups() < group_threshold_usesubset)) {
          if (tmp_subset_ == nullptr || is_change_dataset) {
            tmp_subset_.reset(new Dataset(bag_data_cnt_));
            tmp_subset_->CopyFeatureMapperFrom(train_data_);
          }
          is_use_subset_ = true;
          Log::Debug("Use subset for bagging");
        }
      }

      need_re_bagging_ = true;

      if (is_use_subset_ && bag_data_cnt_ < num_data_) {
        // resize gradient vectors to copy the customized gradients for using subset data
        need_resize_gradients_ = true;
      }
    } else {
      bag_data_cnt_ = num_data_;
      bag_data_indices_.clear();
      #ifdef USE_CUDA
      cuda_bag_data_indices_.Clear();
      #endif  // USE_CUDA
      bagging_runner_.ReSize(0);
      is_use_subset_ = false;
    }
  }

  bool IsHessianChange() const override {
    return false;
  }

  data_size_t num_sampled_queries() const override {
    return num_sampled_queries_;
  }

  const data_size_t* sampled_query_indices() const override {
    return bag_query_indices_.data();
  }

 private:
  data_size_t BaggingHelper(data_size_t start, data_size_t cnt, data_size_t* buffer) {
    if (cnt <= 0) {
      return 0;
    }
    data_size_t cur_left_cnt = 0;
    data_size_t cur_right_pos = cnt;
    // random bagging, minimal unit is one record
    for (data_size_t i = 0; i < cnt; ++i) {
      auto cur_idx = start + i;
      if (bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() < config_->bagging_fraction) {
        buffer[cur_left_cnt++] = cur_idx;
      } else {
        buffer[--cur_right_pos] = cur_idx;
      }
    }
    return cur_left_cnt;
  }

  data_size_t BalancedBaggingHelper(data_size_t start, data_size_t cnt, data_size_t* buffer) {
    if (cnt <= 0) {
      return 0;
    }
    auto label_ptr = train_data_->metadata().label();
    data_size_t cur_left_cnt = 0;
    data_size_t cur_right_pos = cnt;
    // random bagging, minimal unit is one record
    for (data_size_t i = 0; i < cnt; ++i) {
      auto cur_idx = start + i;
      bool is_pos = label_ptr[start + i] > 0;
      bool is_in_bag = false;
      if (is_pos) {
        is_in_bag = bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() <
                    config_->pos_bagging_fraction;
      } else {
        is_in_bag = bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() <
                    config_->neg_bagging_fraction;
      }
      if (is_in_bag) {
        buffer[cur_left_cnt++] = cur_idx;
      } else {
        buffer[--cur_right_pos] = cur_idx;
      }
    }
    return cur_left_cnt;
  }

  /*! \brief whether need restart bagging in continued training */
  bool need_re_bagging_;
  /*! \brief number of threads */
  int num_threads_;
  /*! \brief query boundaries of the in-bag queries */
  std::vector<data_size_t> sampled_query_boundaries_;
  /*! \brief buffer for calculating sampled_query_boundaries_ */
  std::vector<data_size_t> sampled_query_boundaries_thread_buffer_;
  /*! \brief in-bag query indices */
  std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>> bag_query_indices_;
  /*! \brief number of queries in the training dataset */
  data_size_t num_queries_;
  /*! \brief number of in-bag queries */
  data_size_t num_sampled_queries_;
  /*! \brief query boundaries of the whole training dataset */
  const data_size_t* query_boundaries_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_BOOSTING_BAGGING_HPP_
