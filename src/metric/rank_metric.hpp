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
#include <LightGBM/objective_function.h>

#include <string>
#include <sstream>
#include <vector>
#include <numeric>

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
    pairwise_scores_ = config.objective == std::string("pairwise_lambdarank");
    sigmoid_ = config.sigmoid;
    truncation_level_ = config.lambdarank_truncation_level;
    model_indirect_comparison_ = config.pairwise_lambdarank_model_indirect_comparison;
    model_conditional_rel_ = config.pairwise_lambdarank_model_conditional_rel;
    indirect_comparison_above_only_ = config.pairwise_lambdarank_indirect_comparison_above_only;
    logarithmic_discounts_ = config.pairwise_lambdarank_logarithmic_discounts;
    hard_pairwise_preference_ = config.pairwise_lambdarank_hard_pairwise_preference;
    indirect_comparison_max_rank_ = config.pairwise_lambdarank_indirect_comparison_max_rank;
    indirect_comparison_weight_ = model_indirect_comparison_ ? config.pairwise_lambdarank_indirect_comparison_weight : 0.0;
    l2_pairwise_diff_weight_ = config.pairwise_lambdarank_l2_pairwise_diff_weight;
  }

  ~NDCGMetric() {
  }
  void Init(const Metadata& metadata, data_size_t num_data) override {
    for (auto k : eval_at_) {
      name_.emplace_back(std::string("ndcg@") + std::to_string(k));
    }
    num_data_ = metadata.query_boundaries()[metadata.num_queries()];
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
    if (pairwise_scores_) {
      paired_index_map_ = metadata.paired_ranking_item_index_map();
      scores_pointwise_.resize(num_data_, 0.0);
      num_data_pairwise_ = metadata.pairwise_query_boundaries()[metadata.num_queries()];
      query_boundaries_pairwise_ = metadata.pairwise_query_boundaries();

      right2left2pair_map_byquery_.resize(num_queries_);
      left2right2pair_map_byquery_.resize(num_queries_);
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
      for (data_size_t q = 0; q < num_queries_; ++q) {
        const data_size_t start_pairwise = query_boundaries_pairwise_[q];
        const data_size_t cnt_pairwise = query_boundaries_pairwise_[q + 1] - query_boundaries_pairwise_[q];
        const data_size_t cnt_pointwise = query_boundaries_[q + 1] - query_boundaries_[q];

        std::vector<std::vector<std::pair<short, data_size_t>>> left2right2pair_map(cnt_pointwise);
        std::vector<std::vector<std::pair<short, data_size_t>>> right2left2pair_map(cnt_pointwise);

        for (data_size_t i = 0; i < cnt_pairwise; ++i) {
          auto tmp = paired_index_map_[i + start_pairwise];
          auto index_left_full = tmp.first;
          auto index_right_full = tmp.second;

          // Check range before narrowing
          if (index_left_full >= 32768 || index_right_full >= 32768) {
            Log::Fatal("Index exceeds 32768: index_left_full = %d, index_right_full = %d, cnt_pairwise = %d, query idx = %d", index_left_full, index_right_full, cnt_pairwise, q);
          }

          short index_left = static_cast<short>(index_left_full);
          short index_right = static_cast<short>(index_right_full);

          left2right2pair_map[index_left].emplace_back(index_right, i);
          right2left2pair_map[index_right].emplace_back(index_left, i);
        }

        for (auto& vec : left2right2pair_map) {
          std::sort(vec.begin(), vec.end(), [](auto& a, auto& b) { return a.first < b.first; });
          vec.shrink_to_fit();
        }
        for (auto& vec : right2left2pair_map) {
          std::sort(vec.begin(), vec.end(), [](auto& a, auto& b) { return a.first < b.first; });
          vec.shrink_to_fit();
        }

        left2right2pair_map_byquery_[q] = std::move(left2right2pair_map);
        right2left2pair_map_byquery_[q] = std::move(right2left2pair_map);
      }
    }
    sigmoid_cache_.Init(sigmoid_);
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
          if (pairwise_scores_) {
            const data_size_t start_pointwise = query_boundaries_[i];
            const data_size_t cnt_pointwise = query_boundaries_[i + 1] - query_boundaries_[i];
            const data_size_t start_pairwise = query_boundaries_pairwise_[i];
            const data_size_t cnt_pairwise = query_boundaries_pairwise_[i + 1] - query_boundaries_pairwise_[i];
            UpdatePointwiseScoresForOneQuery(scores_pointwise_.data() + start_pointwise, score + start_pairwise, cnt_pointwise,
              paired_index_map_ + start_pairwise, right2left2pair_map_byquery_[i], left2right2pair_map_byquery_[i], truncation_level_,
              sigmoid_, sigmoid_cache_, model_indirect_comparison_, model_conditional_rel_, indirect_comparison_above_only_, logarithmic_discounts_,
              hard_pairwise_preference_, indirect_comparison_max_rank_, indirect_comparison_weight_, l2_pairwise_diff_weight_);
          }

          // calculate DCG
          DCGCalculator::CalDCG(eval_at_, label_ + query_boundaries_[i],
                                (pairwise_scores_? scores_pointwise_.data(): score) + query_boundaries_[i],
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
          if (pairwise_scores_) {
            const data_size_t start_pointwise = query_boundaries_[i];
            const data_size_t cnt_pointwise = query_boundaries_[i + 1] - query_boundaries_[i];
            const data_size_t start_pairwise = query_boundaries_pairwise_[i];
            const data_size_t cnt_pairwise = query_boundaries_pairwise_[i + 1] - query_boundaries_pairwise_[i];
            UpdatePointwiseScoresForOneQuery(scores_pointwise_.data() + start_pointwise, score + start_pairwise, cnt_pointwise, 
              paired_index_map_ + start_pairwise, right2left2pair_map_byquery_[i], left2right2pair_map_byquery_[i], truncation_level_,
              sigmoid_, sigmoid_cache_, model_indirect_comparison_, model_conditional_rel_, indirect_comparison_above_only_, logarithmic_discounts_,
              hard_pairwise_preference_, indirect_comparison_max_rank_, indirect_comparison_weight_, l2_pairwise_diff_weight_);
          }
          // calculate DCG
          DCGCalculator::CalDCG(eval_at_, label_ + query_boundaries_[i],
                                (pairwise_scores_ ? scores_pointwise_.data() : score) + query_boundaries_[i],
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
  bool pairwise_scores_;
  double sigmoid_;
  CommonC::SigmoidCache sigmoid_cache_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  mutable std::vector<double> scores_pointwise_;
  const std::pair<data_size_t, data_size_t>* paired_index_map_;
  std::vector<std::vector<std::vector<std::pair<short, data_size_t>>>> left2right2pair_map_byquery_;
  std::vector<std::vector<std::vector<std::pair<short, data_size_t>>>> right2left2pair_map_byquery_;
  /*! \brief Number of data */
  data_size_t num_data_pairwise_;
  const data_size_t* query_boundaries_pairwise_;
  bool model_indirect_comparison_;
  bool model_conditional_rel_;
  bool indirect_comparison_above_only_;
  bool logarithmic_discounts_;
  bool hard_pairwise_preference_;
  int indirect_comparison_max_rank_;
  double indirect_comparison_weight_;
  double l2_pairwise_diff_weight_;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_SRC_METRIC_RANK_METRIC_HPP_
