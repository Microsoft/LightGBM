/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_RANK_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_RANK_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <string>
#include <sstream>
#include <vector>

namespace LightGBM {

class NDCGMetric:public Metric {
 public:
  explicit NDCGMetric(const Config& config) {
    // get eval position
    eval_at_ = config.eval_at;
    auto label_gain = config.label_gain;
    DCGCalculator::DefaultEvalAt(&eval_at_);
    DCGCalculator::DefaultLabelGain(&label_gain);
    // initialize DCG calculator
    DCGCalculator::Init(label_gain);
  }

  ~NDCGMetric() {
  }
  void Init(const Metadata& metadata, data_size_t num_data) override {
    for (auto k : eval_at_) {
      name_.emplace_back(std::string("ndcg@") + std::to_string(k));
    }
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    num_queries_ = metadata.num_queries();
    DCGCalculator::CheckMetadata(metadata, num_queries_);
    DCGCalculator::CheckLabel(label_, num_data_);
    // get query boundaries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("The NDCG metric requires query information");
    }
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
    inverse_max_dcgs_.resize(num_queries_);
    // cache the inverse max DCG for all queries, used to calculate NDCG
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i].resize(eval_at_.size(), 0.0f);
      DCGCalculator::CalMaxDCG(eval_at_, label_ + query_boundaries_[i],
                               query_boundaries_[i + 1] - query_boundaries_[i],
                               &inverse_max_dcgs_[i]);
      for (size_t j = 0; j < inverse_max_dcgs_[i].size(); ++j) {
        if (inverse_max_dcgs_[i][j] > 0.0f) {
          inverse_max_dcgs_[i][j] = 1.0f / inverse_max_dcgs_[i][j];
        } else {
          // marking negative for all negative queries.
          // if one meet this query, it's ndcg will be set as -1.
          inverse_max_dcgs_[i][j] = -1.0f;
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

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    int num_threads = OMP_NUM_THREADS();
    // some buffers for multi-threading sum up
    std::vector<std::vector<double>> result_buffer_;
    for (int i = 0; i < num_threads; ++i) {
      result_buffer_.emplace_back(eval_at_.size(), 0.0f);
    }
    std::vector<double> tmp_dcg(eval_at_.size(), 0.0f);
    if (query_weights_ == nullptr) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) firstprivate(tmp_dcg)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        const int tid = omp_get_thread_num();
        // if all doc in this query are all negative, let its NDCG=1
        if (inverse_max_dcgs_[i][0] <= 0.0f) {
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += 1.0f;
          }
        } else {
          // calculate DCG
          DCGCalculator::CalDCG(eval_at_, label_ + query_boundaries_[i],
                                score + query_boundaries_[i],
                                query_boundaries_[i + 1] - query_boundaries_[i], &tmp_dcg);
          // calculate NDCG
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += tmp_dcg[j] * inverse_max_dcgs_[i][j];
          }
        }
      }
    } else {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) firstprivate(tmp_dcg)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        const int tid = omp_get_thread_num();
        // if all doc in this query are all negative, let its NDCG=1
        if (inverse_max_dcgs_[i][0] <= 0.0f) {
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += 1.0f;
          }
        } else {
          // calculate DCG
          DCGCalculator::CalDCG(eval_at_, label_ + query_boundaries_[i],
                                score + query_boundaries_[i],
                                query_boundaries_[i + 1] - query_boundaries_[i], &tmp_dcg);
          // calculate NDCG
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += tmp_dcg[j] * inverse_max_dcgs_[i][j] * query_weights_[i];
          }
        }
      }
    }
    // Get final average NDCG
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
  /*! \brief Name of test set */
  std::vector<std::string> name_;
  /*! \brief Query boundaries information */
  const data_size_t* query_boundaries_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Weights of queries */
  const label_t* query_weights_;
  /*! \brief Sum weights of queries */
  double sum_query_weights_;
  /*! \brief Evaluate position of NDCG */
  std::vector<data_size_t> eval_at_;
  /*! \brief Cache the inverse max dcg for all queries */
  std::vector<std::vector<double>> inverse_max_dcgs_;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_SRC_METRIC_RANK_METRIC_HPP_
