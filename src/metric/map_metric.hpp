/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_MAP_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_MAP_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

namespace LightGBM {

class MapMetric:public Metric {
 public:
  explicit MapMetric(const Config& config) {
    // get eval position
    eval_at_ = config.eval_at;
    DCGCalculator::DefaultEvalAt(&eval_at_);
  }

  ~MapMetric() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    for (auto k : eval_at_) {
      name_.emplace_back(std::string("map@") + std::to_string(k));
    }
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get query boundaries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("For MAP metric, there should be query information");
    }
    num_queries_ = metadata.num_queries();
    Log::Info("Total groups: %d, total data: %d", num_queries_, num_data_);
    // get query weights
    query_weights_ = metadata.query_weights();
    if (query_weights_ == nullptr) {
      sum_query_weights_ = static_cast<double>(num_queries_);
    } else {
      sum_query_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_queries_; ++i) {
        sum_query_weights_ += query_weights_[i];
      }
    }

    npos_per_query_.resize(num_queries_, 0);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      for (data_size_t j = query_boundaries_[i]; j < query_boundaries_[i + 1]; ++j) {
        if (label_[j] > 0.5f) {
          ++npos_per_query_[i];
        }
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  void CalMapAtK(std::vector<int> ks, data_size_t npos, const label_t* label,
                 const double* score, data_size_t num_data, std::vector<double>* out) const {
    // get sorted indices by score
    std::vector<data_size_t> sorted_idx;
    for (data_size_t i = 0; i < num_data; ++i) {
      sorted_idx.emplace_back(i);
    }
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                     [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });

    int num_hit = 0;
    double sum_ap = 0.0f;
    data_size_t cur_left = 0;
    for (size_t i = 0; i < ks.size(); ++i) {
      data_size_t cur_k = static_cast<data_size_t>(ks[i]);
      if (cur_k > num_data) {
        cur_k = num_data;
      }
      for (data_size_t j = cur_left; j < cur_k; ++j) {
        data_size_t idx = sorted_idx[j];
        if (label[idx] > 0.5f) {
          ++num_hit;
          sum_ap += static_cast<double>(num_hit) / (j + 1.0f);
        }
      }
      if (npos > 0) {
        (*out)[i] = sum_ap / std::min(npos, cur_k);
      } else {
        (*out)[i] = 1.0f;
      }
      cur_left = cur_k;
    }
  }
  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    // some buffers for multi-threading sum up
    int num_threads = OMP_NUM_THREADS();
    std::vector<std::vector<double>> result_buffer_;
    for (int i = 0; i < num_threads; ++i) {
      result_buffer_.emplace_back(eval_at_.size(), 0.0f);
    }
    std::vector<double> tmp_map(eval_at_.size(), 0.0f);
    if (query_weights_ == nullptr) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided) firstprivate(tmp_map)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        const int tid = omp_get_thread_num();
        CalMapAtK(eval_at_, npos_per_query_[i], label_ + query_boundaries_[i],
                  score + query_boundaries_[i], query_boundaries_[i + 1] - query_boundaries_[i], &tmp_map);
        for (size_t j = 0; j < eval_at_.size(); ++j) {
          result_buffer_[tid][j] += tmp_map[j];
        }
      }
    } else {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided) firstprivate(tmp_map)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        const int tid = omp_get_thread_num();
        CalMapAtK(eval_at_, npos_per_query_[i], label_ + query_boundaries_[i],
                  score + query_boundaries_[i], query_boundaries_[i + 1] - query_boundaries_[i], &tmp_map);
        for (size_t j = 0; j < eval_at_.size(); ++j) {
          result_buffer_[tid][j] += tmp_map[j] * query_weights_[i];
        }
      }
    }
    // Get final average MAP
    std::vector<double> result(eval_at_.size(), 0.0f);
    for (size_t j = 0; j < result.size(); ++j) {
      for (int i = 0; i < num_threads; ++i) {
        result[j] += result_buffer_[i][j];
      }
      result[j] /= sum_query_weights_;
    }
    return result;
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Query boundaries information */
  const data_size_t* query_boundaries_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Weights of queries */
  const label_t* query_weights_;
  /*! \brief Sum weights of queries */
  double sum_query_weights_;
  /*! \brief Evaluate position of Nmap */
  std::vector<data_size_t> eval_at_;
  std::vector<std::string> name_;
  std::vector<data_size_t> npos_per_query_;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_SRC_METRIC_MAP_METRIC_HPP_
