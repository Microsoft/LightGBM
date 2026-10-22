/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <numeric>

namespace LightGBM {
  template <
    typename Case1Handler, typename Case2Handler,
    typename Case3Handler, typename Case4Handler>
  inline void process_indirect_comparisons_optimized(
    short indexLeft,
    short indexRight,
    const std::vector<int>& ranks,
    const std::vector<std::vector<std::pair<short, data_size_t>>>& left2right2pair_map,
    const std::vector<std::vector<std::pair<short, data_size_t>>>& right2left2pair_map,
    int indirect_comparison_max_rank,
    bool indirect_comparison_above_only,
    bool model_conditional_rel,
    Case1Handler&& case1,
    Case2Handler&& case2,
    Case3Handler&& case3,
    Case4Handler&& case4) noexcept
  {
    int rankLeft = ranks[indexLeft];
    int rankRight = ranks[indexRight];

    // ----------- First loop: right2left2pair_map neighbors -----------
    for (const auto& p : right2left2pair_map[indexLeft]) {
      short indexO = p.first;
      data_size_t indexOLeft = p.second;

      int rankO = ranks[indexO];
      if (rankO >= indirect_comparison_max_rank) continue;

      bool isOAboveBoth = (rankO < rankRight) && (rankO < rankLeft);

      // Case 1
      if (isOAboveBoth || (!indirect_comparison_above_only && !model_conditional_rel)) {
        data_size_t indexORight = get_pair_index(left2right2pair_map[indexO], indexRight);
        if (indexORight != static_cast<data_size_t>(-1))
          case1(indexORight, indexOLeft);
      }

      // Case 2
      if ((!indirect_comparison_above_only || isOAboveBoth) && !model_conditional_rel) {
        data_size_t indexRightO = get_pair_index(left2right2pair_map[indexRight], indexO);
        if (indexRightO != static_cast<data_size_t>(-1))
          case2(indexRightO, indexOLeft);
      }
    }

    // ----------- Second loop: left2right2pair_map neighbors -----------
    if (!model_conditional_rel) {
      const auto& left_pairs = left2right2pair_map[indexLeft];
      for (const auto& p : left_pairs) {
        short indexO = p.first;
        data_size_t indexLeftO = p.second;

        int rankO = ranks[indexO];
        if (rankO >= indirect_comparison_max_rank) continue;

        bool isOAboveBoth = (rankO < rankRight) && (rankO < rankLeft);
        if (!indirect_comparison_above_only || isOAboveBoth) {
          // Case 3
          data_size_t indexORight = get_pair_index(left2right2pair_map[indexO], indexRight);
          if (indexORight != static_cast<data_size_t>(-1))
            case3(indexORight, indexLeftO);

          // Case 4
          data_size_t indexRightO = get_pair_index(left2right2pair_map[indexRight], indexO);
          if (indexRightO != static_cast<data_size_t>(-1))
            case4(indexRightO, indexLeftO);
        }
      }
    }
  }

  void UpdatePointwiseScoresForOneQuery(double* score_pointwise, const double* score_pairwise, data_size_t cnt_pointwise,
    const std::pair<data_size_t, data_size_t>* paired_index_map,
    const std::vector<std::vector<std::pair<short, data_size_t>>>& right2left2pair_map,
    const std::vector<std::vector<std::pair<short, data_size_t>>>& left2right2pair_map,
    int truncation_level, double sigma, const CommonC::SigmoidCache& sigmoid_cache, bool model_indirect_comparison, bool model_conditional_rel,
    bool indirect_comparison_above_only, bool logarithmic_discounts, bool hard_pairwise_preference, int indirect_comparison_max_rank,
    double indirect_comparison_weight, double l2_pairwise_diff_weight);

/*!
 * \brief Objective function for Ranking
 */
class RankingObjective : public ObjectiveFunction {
 public:
  explicit RankingObjective(const Config& config)
      : seed_(config.objective_seed) {
    learning_rate_ = config.learning_rate;
    position_bias_regularization_ = config.lambdarank_position_bias_regularization;
  }

  explicit RankingObjective(const std::vector<std::string>&) : seed_(0) {}

  ~RankingObjective() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    // get positions
    positions_ = metadata.positions();
    // get position ids
    position_ids_ = metadata.position_ids();
    // get number of different position ids
    num_position_ids_ = static_cast<data_size_t>(metadata.num_position_ids());
    // get boundaries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Ranking tasks require query information");
    }
    num_queries_ = metadata.num_queries();
    // initialize position bias vectors
    pos_biases_.resize(num_position_ids_, 0.0);
  }

  void GetGradientsWithSampledQueries(const double* score, const data_size_t num_sampled_queries, const data_size_t* sampled_query_indices,
                    score_t* gradients, score_t* hessians) const override {
    const data_size_t num_queries = (sampled_query_indices == nullptr ? num_queries_ : num_sampled_queries);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
    for (data_size_t i = 0; i < num_queries; ++i) {
      const data_size_t query_index = (sampled_query_indices == nullptr ? i : sampled_query_indices[i]);
      const data_size_t start = query_boundaries_[query_index];
      const data_size_t cnt = query_boundaries_[query_index + 1] - query_boundaries_[query_index];
      std::vector<double> score_adjusted;
      if (num_position_ids_ > 0) {
        for (data_size_t j = 0; j < cnt; ++j) {
          score_adjusted.push_back(score[start + j] + pos_biases_[positions_[start + j]]);
        }
      }
      GetGradientsForOneQuery(query_index, cnt, label_ + start, num_position_ids_ > 0 ? score_adjusted.data() : score + start,
                              gradients + start, hessians + start);
      if (weights_ != nullptr) {
        for (data_size_t j = 0; j < cnt; ++j) {
          gradients[start + j] =
              static_cast<score_t>(gradients[start + j] * weights_[start + j]);
          hessians[start + j] =
              static_cast<score_t>(hessians[start + j] * weights_[start + j]);
        }
      }
    }
    if (num_position_ids_ > 0) {
      UpdatePositionBiasFactors(gradients, hessians);
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    GetGradientsWithSampledQueries(score, num_queries_, nullptr, gradients, hessians);
  }

  virtual void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                       const label_t* label,
                                       const double* score, score_t* lambdas,
                                       score_t* hessians) const = 0;

  virtual void UpdatePositionBiasFactors(const score_t* /*lambdas*/, const score_t* /*hessians*/) const {}

  const char* GetName() const override = 0;

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

 protected:
  int seed_;
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Pointer of positions */
  const data_size_t* positions_;
  /*! \brief Pointer of position IDs */
  const std::string* position_ids_;
  /*! \brief Pointer of label */
  data_size_t num_position_ids_;
  /*! \brief Query boundaries */
  const data_size_t* query_boundaries_;
  /*! \brief Position bias factors */
  mutable std::vector<label_t> pos_biases_;
  /*! \brief Learning rate to update position bias factors */
  double learning_rate_;
  /*! \brief Position bias regularization */
  double position_bias_regularization_;
};

/*!
 * \brief Objective function for LambdaRank with NDCG
 */
class LambdarankNDCG : public RankingObjective {
 public:
  explicit LambdarankNDCG(const Config& config)
      : RankingObjective(config),
        sigmoid_(config.sigmoid),
        norm_(config.lambdarank_norm),
        truncation_level_(config.lambdarank_truncation_level) {
    label_gain_ = config.label_gain;
    // initialize DCG calculator
    DCGCalculator::DefaultLabelGain(&label_gain_);
    DCGCalculator::Init(label_gain_);
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }
  }

  explicit LambdarankNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~LambdarankNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    DCGCalculator::CheckMetadata(metadata, num_queries_);
    DCGCalculator::CheckLabel(label_, num_data_);
    inverse_max_dcgs_.resize(num_queries_);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(
          truncation_level_, label_ + query_boundaries_[i],
          query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_dcgs_[i] > 0.0) {
        inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
      }
    }
    // construct Sigmoid table to speed up Sigmoid transform
    ConstructSigmoidTable();
    sigmoid_cache_.Init(sigmoid_);
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // get max DCG on current query
    const double inverse_max_dcg = inverse_max_dcgs_[query_id];
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(),
        [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;
    // start accumulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) {
        continue;
      }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) {
          continue;
        }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) {
          continue;
        }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        } else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
  }

  inline double GetSigmoid(double score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too large, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
                                                sigmoid_table_idx_factor_)];
    }
  }

  void ConstructSigmoidTable() {
    // get boundary
    min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2;
    max_sigmoid_input_ = -min_sigmoid_input_;
    sigmoid_table_.resize(_sigmoid_bins);
    // get score to bin factor
    sigmoid_table_idx_factor_ =
        _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_);
    // cache
    for (size_t i = 0; i < _sigmoid_bins; ++i) {
      const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
      sigmoid_table_[i] = 1.0f / (1.0f + std::exp(score * sigmoid_));
    }
  }

  void UpdatePositionBiasFactors(const score_t* lambdas, const score_t* hessians) const override {
    /// get number of threads
    int num_threads = OMP_NUM_THREADS();
    // create per-thread buffers for first and second derivatives of utility w.r.t. position bias factors
    std::vector<double> bias_first_derivatives(num_position_ids_ * num_threads, 0.0);
    std::vector<double> bias_second_derivatives(num_position_ids_ * num_threads, 0.0);
    std::vector<int> instance_counts(num_position_ids_ * num_threads, 0);
    #pragma omp parallel for schedule(guided) num_threads(num_threads)
    for (data_size_t i = 0; i < num_data_; i++) {
      // get thread ID
      const int tid = omp_get_thread_num();
      size_t offset = static_cast<size_t>(positions_[i] + tid * num_position_ids_);
      // accumulate first derivatives of utility w.r.t. position bias factors, for each position
      bias_first_derivatives[offset] -= lambdas[i];
      // accumulate second derivatives of utility w.r.t. position bias factors, for each position
      bias_second_derivatives[offset] -= hessians[i];
      instance_counts[offset]++;
    }
    #pragma omp parallel for schedule(guided) num_threads(num_threads)
    for (data_size_t i = 0; i < num_position_ids_; i++) {
      double bias_first_derivative = 0.0;
      double bias_second_derivative = 0.0;
      int instance_count = 0;
      // aggregate derivatives from per-thread buffers
      for (int tid = 0; tid < num_threads; tid++) {
        size_t offset = static_cast<size_t>(i + tid * num_position_ids_);
        bias_first_derivative += bias_first_derivatives[offset];
        bias_second_derivative += bias_second_derivatives[offset];
        instance_count += instance_counts[offset];
      }
      // L2 regularization on position bias factors
      bias_first_derivative -= pos_biases_[i] * position_bias_regularization_ * instance_count;
      bias_second_derivative -= position_bias_regularization_ * instance_count;
      // do Newton-Raphson step to update position bias factors
      pos_biases_[i] += learning_rate_ * bias_first_derivative / (std::abs(bias_second_derivative) + 0.001);
    }
    LogDebugPositionBiasFactors();
  }

  const char* GetName() const override { return "lambdarank"; }

 protected:
  void LogDebugPositionBiasFactors() const {
    std::stringstream message_stream;
    message_stream << std::setw(15) << "position"
      << std::setw(15) << "bias_factor"
      << std::endl;
    Log::Debug(message_stream.str().c_str());
    message_stream.str("");
    for (int i = 0; i < num_position_ids_; ++i) {
      message_stream << std::setw(15) << position_ids_[i]
        << std::setw(15) << pos_biases_[i];
      Log::Debug(message_stream.str().c_str());
      message_stream.str("");
    }
  }
  /*! \brief Sigmoid param */
  double sigmoid_;
  /*! \brief Normalize the lambdas or not */
  bool norm_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<double> sigmoid_table_;
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of Sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in Sigmoid table */
  double sigmoid_table_idx_factor_;
  CommonC::SigmoidCache sigmoid_cache_;
};

/*!
 * \brief Implementation of the learning-to-rank objective function, XE_NDCG
 * [arxiv.org/abs/1911.09798].
 */
class RankXENDCG : public RankingObjective {
 public:
  explicit RankXENDCG(const Config& config) : RankingObjective(config) {}

  explicit RankXENDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~RankXENDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      rands_.emplace_back(seed_ + i);
    }
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // Skip groups with too few items.
    if (cnt <= 1) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = 0.0f;
        hessians[i] = 0.0f;
      }
      return;
    }

    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho(cnt, 0.0);
    Common::Softmax(score, rho.data(), cnt);

    // An auxiliary buffer of parameters used to form the ground-truth
    // distribution and compute the loss.
    std::vector<double> params(cnt);

    double inv_denominator = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      params[i] = Phi(label[i], rands_[query_id].NextFloat());
      inv_denominator += params[i];
    }
    // sum_labels will always be positive number
    inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);

    // Approximate gradients and inverse Hessian.
    // First order terms.
    double sum_l1 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = -params[i] * inv_denominator + rho[i];
      lambdas[i] = static_cast<score_t>(term);
      // Params will now store terms needed to compute second-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l1 += params[i];
    }
    // Second order terms.
    double sum_l2 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = rho[i] * (sum_l1 - params[i]);
      lambdas[i] += static_cast<score_t>(term);
      // Params will now store terms needed to compute third-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l2 += params[i];
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
  }

  double Phi(const label_t l, double g) const {
    return Common::Pow(2, static_cast<int>(l)) - g;
  }

  const char* GetName() const override { return "rank_xendcg"; }

 protected:
  mutable std::vector<Random> rands_;
};


class PairwiseLambdarankNDCG: public LambdarankNDCG {
 public:
  explicit PairwiseLambdarankNDCG(const Config& config): LambdarankNDCG(config) {
    model_indirect_comparison_ = config.pairwise_lambdarank_model_indirect_comparison;
    model_conditional_rel_ = config.pairwise_lambdarank_model_conditional_rel;
    indirect_comparison_above_only_ = config.pairwise_lambdarank_indirect_comparison_above_only;
    logarithmic_discounts_ = config.pairwise_lambdarank_logarithmic_discounts;
    hard_pairwise_preference_ = config.pairwise_lambdarank_hard_pairwise_preference;
    indirect_comparison_max_rank_ = config.pairwise_lambdarank_indirect_comparison_max_rank;
    indirect_comparison_weight_ = model_indirect_comparison_? config.pairwise_lambdarank_indirect_comparison_weight: 0.0;
    l2_pairwise_diff_weight_ = config.pairwise_lambdarank_l2_pairwise_diff_weight;
  }

  explicit PairwiseLambdarankNDCG(const std::vector<std::string>& strs): LambdarankNDCG(strs) {}

  ~PairwiseLambdarankNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data_pairwise) override {
    data_size_t num_data_pointwise = metadata.query_boundaries()[metadata.num_queries()];
    Log::Info("!!! num_data_pointwise %d", num_data_pointwise);
    LambdarankNDCG::Init(metadata, num_data_pointwise);
    num_data_pairwise_ = num_data_pairwise;
    query_boundaries_pairwise_ = metadata.pairwise_query_boundaries();
    if (query_boundaries_pairwise_ == nullptr) {
      Log::Fatal("Ranking tasks require query information");
    }
    paired_index_map_ = metadata.paired_ranking_item_index_map();
    scores_pointwise_.resize(num_data_pointwise, 0.0);

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

  void GetGradientsWithSampledQueries(const double* score_pairwise, const data_size_t num_sampled_queries, const data_size_t* sampled_query_indices,
                    score_t* gradients_pairwise, score_t* hessians_pairwise) const override {
    const data_size_t num_queries = (sampled_query_indices == nullptr ? num_queries_ : num_sampled_queries);
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
    for (data_size_t i = 0; i < num_queries; ++i) {
      global_timer.Start("pairwise_lambdarank::GetGradients part 0");
      const data_size_t query_index = (sampled_query_indices == nullptr ? i : sampled_query_indices[i]);
      const data_size_t start_pointwise = query_boundaries_[query_index];
      const data_size_t cnt_pointwise = query_boundaries_[query_index + 1] - query_boundaries_[query_index];
      const data_size_t start_pairwise = query_boundaries_pairwise_[query_index];
      const data_size_t cnt_pairwise = query_boundaries_pairwise_[query_index + 1] - query_boundaries_pairwise_[query_index];
      std::vector<double> score_adjusted_pairwise;
      if (num_position_ids_ > 0) {
        for (data_size_t j = 0; j < cnt_pairwise; ++j) {
          score_adjusted_pairwise.push_back(score_pairwise[start_pairwise + j] + pos_biases_[positions_[start_pointwise + paired_index_map_[start_pairwise + j].first]] -
            pos_biases_[positions_[start_pointwise + paired_index_map_[start_pairwise + j].second]]);
        }
      }
      global_timer.Stop("pairwise_lambdarank::GetGradients part 0");
      global_timer.Start("pairwise_lambdarank::GetGradients part 1");
      GetGradientsForOneQuery(query_index, cnt_pointwise, cnt_pairwise, label_ + start_pointwise, scores_pointwise_.data() + start_pointwise, num_position_ids_ > 0 ? score_adjusted_pairwise.data() : score_pairwise + start_pairwise,
        right2left2pair_map_byquery_[query_index], left2right2pair_map_byquery_[query_index],
        gradients_pairwise + start_pairwise, hessians_pairwise + start_pairwise);
      global_timer.Stop("pairwise_lambdarank::GetGradients part 1");
      global_timer.Start("pairwise_lambdarank::GetGradients part 2");
      UpdatePointwiseScoresForOneQuery(scores_pointwise_.data() + start_pointwise, score_pairwise + start_pairwise, cnt_pointwise,
        paired_index_map_ + start_pairwise, right2left2pair_map_byquery_[query_index], left2right2pair_map_byquery_[query_index], truncation_level_, sigmoid_, sigmoid_cache_,
        model_indirect_comparison_, model_conditional_rel_, indirect_comparison_above_only_, logarithmic_discounts_, hard_pairwise_preference_, indirect_comparison_max_rank_,
        indirect_comparison_weight_, l2_pairwise_diff_weight_);
      global_timer.Stop("pairwise_lambdarank::GetGradients part 2");
    }
    if (num_position_ids_ > 0) {
      std::vector<score_t> gradients_pointwise(num_data_);
      std::vector<score_t> hessians_pointwise(num_data_);
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        const data_size_t query_index = (sampled_query_indices == nullptr ? i : sampled_query_indices[i]);
        const data_size_t cnt_pointwise = query_boundaries_[query_index + 1] - query_boundaries_[query_index];
        const data_size_t cnt_pairwise = query_boundaries_pairwise_[query_index + 1] - query_boundaries_pairwise_[query_index];
        TransformGradientsPairwiseIntoPointwiseForOneQuery(query_index, cnt_pointwise, cnt_pairwise, gradients_pairwise, hessians_pairwise, gradients_pointwise.data(), hessians_pointwise.data());
      }
      UpdatePositionBiasFactors(gradients_pointwise.data(), hessians_pointwise.data());
    }
  }

  inline void TransformGradientsPairwiseIntoPointwiseForOneQuery(data_size_t query_id, data_size_t cnt_pointwise, data_size_t cnt,
    const score_t* gradients, const score_t* hessians, score_t* gradients_pointwise, score_t* hessians_pointwise) const {
    // initialize with zero
    for (data_size_t i = 0; i < cnt_pointwise; ++i) {
      gradients_pointwise[i] = 0.0f;
      hessians_pointwise[i] = 0.0f;
    }
    const data_size_t start = query_boundaries_[query_id];
    for (data_size_t i = 0; i < cnt; i++) {
      int indexLeft = paired_index_map_[i + start].first;
      int indexRight = paired_index_map_[i + start].second;
      gradients_pointwise[indexLeft] += gradients[i];
      gradients_pointwise[indexRight] -= gradients[i];
      hessians_pointwise[indexLeft] += hessians[i];
      hessians_pointwise[indexRight] += hessians[i];
    }
  }


  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt_pointwise, data_size_t cnt_pairwise,
    const label_t* label, const double* score_pointwise, const double* score_pairwise,
    const std::vector<std::vector<std::pair<short, data_size_t>>>& right2left2pair_map,
    const std::vector<std::vector<std::pair<short, data_size_t>>>& left2right2pair_map,
    score_t* lambdas_pairwise,
    score_t* hessians_pairwise) const {

    const data_size_t start_pointwise = query_boundaries_[query_id];
    const data_size_t start_pairwise = query_boundaries_pairwise_[query_id];

    // get max DCG on current query
    const double inverse_max_dcg = inverse_max_dcgs_[query_id];
    // initialize with zero
    for (data_size_t i = 0; i < cnt_pairwise; ++i) {
      lambdas_pairwise[i] = 0.0f;
      hessians_pairwise[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt_pointwise);
    for (data_size_t i = 0; i < cnt_pointwise; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
      sorted_idx.begin(), sorted_idx.end(),
      [score_pointwise](data_size_t a, data_size_t b) { return score_pointwise[a] > score_pointwise[b]; });
    // get ranks when sorted by scores
    std::vector<int> ranks(cnt_pointwise);
    for (int i = 0; i < cnt_pointwise; i++) {
      ranks[sorted_idx.at(i)] = i;
    }
    // get best and worst score
    const double best_score = score_pointwise[sorted_idx[0]];
    data_size_t worst_idx = cnt_pointwise - 1;
    if (worst_idx > 0 && score_pointwise[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score_pointwise[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;


    // start accumulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt_pointwise - 1 && i < truncation_level_; ++i) {
      if (score_pointwise[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt_pointwise; ++j) {
        if (score_pointwise[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        }
        else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        data_size_t indexLeft = high;
        const int high_label = static_cast<int>(label[high]);
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        data_size_t indexRight = low;
        const int low_label = static_cast<int>(label[low]);
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);


        double delta_score = 0.0;
        double delta_score_total_weight = 0.0;
        int comparisons_direct = 0;
        double delta_score_indirect = 0.0;
        int comparisons_indirect = 0;

        data_size_t pair = get_pair_index(left2right2pair_map[indexLeft], indexRight);
        if (pair != static_cast<data_size_t>(-1)) {
          delta_score += score_pairwise[pair];
          ++comparisons_direct;
        }

        data_size_t pair_inverse = get_pair_index(left2right2pair_map[indexRight], indexLeft);
        if (pair_inverse != static_cast<data_size_t>(-1)) {
          delta_score -= score_pairwise[pair_inverse];
          ++comparisons_direct;
        }

        if (comparisons_direct > 0) {
          delta_score /= comparisons_direct;
          delta_score *= (1.0 - indirect_comparison_weight_);
          delta_score_total_weight += (1.0 - indirect_comparison_weight_);
        }

        if (model_indirect_comparison_) {
          auto apply_score = [&](data_size_t a, data_size_t b, score_t signA, score_t signB) noexcept {
            delta_score_indirect += signA * score_pairwise[a] + signB * score_pairwise[b];
            ++comparisons_indirect;
            };

          process_indirect_comparisons_optimized(
            indexLeft, indexRight, ranks, left2right2pair_map, right2left2pair_map,
            indirect_comparison_max_rank_, indirect_comparison_above_only_, model_conditional_rel_,
            // Case 1: (+, −)
            [&](data_size_t idxA, data_size_t idxB) noexcept { apply_score(idxA, idxB, +1, -1); },
            // Case 2: (−, −)
            [&](data_size_t idxA, data_size_t idxB) noexcept { apply_score(idxA, idxB, -1, -1); },
            // Case 3: (+, +)
            [&](data_size_t idxA, data_size_t idxB) noexcept { apply_score(idxA, idxB, +1, +1); },
            // Case 4: (−, +)
            [&](data_size_t idxA, data_size_t idxB) noexcept { apply_score(idxA, idxB, -1, +1); }
          );

          if (comparisons_indirect > 0) {
            delta_score_indirect /= comparisons_indirect;
            delta_score += delta_score_indirect * indirect_comparison_weight_;
            delta_score_total_weight += indirect_comparison_weight_;
          }
        }


        if (delta_score_total_weight <= 0.0) { continue; }
        delta_score /= delta_score_total_weight;

        if (delta_score == kMinScore || -delta_score == kMinScore) { continue; }


        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair	
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regularize the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        if (weights_ != nullptr) {
          p_lambda *= weights_[start_pointwise + high] * weights_[start_pointwise + low];
          p_hessian *= weights_[start_pointwise + high] * weights_[start_pointwise + low];
        }

        if (pair >= 0) {
          lambdas_pairwise[pair] += static_cast<score_t>(((1.0 - indirect_comparison_weight_) / delta_score_total_weight) * p_lambda / comparisons_direct);
          hessians_pairwise[pair] += static_cast<score_t>(((1.0 - indirect_comparison_weight_) / delta_score_total_weight) * p_hessian / comparisons_direct);
        }
        if (pair_inverse >= 0) {
          lambdas_pairwise[pair_inverse] -= static_cast<score_t>(((1.0 - indirect_comparison_weight_) / delta_score_total_weight) * p_lambda / comparisons_direct);
          hessians_pairwise[pair_inverse] += static_cast<score_t>(((1.0 - indirect_comparison_weight_) / delta_score_total_weight) * p_hessian / comparisons_direct);
        }

        if (model_indirect_comparison_ && comparisons_indirect > 0) {
          const score_t l = static_cast<score_t>((indirect_comparison_weight_ / delta_score_total_weight) * p_lambda / comparisons_indirect);
          const score_t h = static_cast<score_t>((indirect_comparison_weight_ / delta_score_total_weight ) * p_hessian / comparisons_indirect);

          auto apply_update = [&](data_size_t a, data_size_t b, score_t signA, score_t signB) noexcept {
            lambdas_pairwise[a] += signA * l;
            hessians_pairwise[a] += h;
            lambdas_pairwise[b] += signB * l;
            hessians_pairwise[b] += h;
            };

          process_indirect_comparisons_optimized(
            indexLeft, indexRight, ranks, left2right2pair_map, right2left2pair_map,
            indirect_comparison_max_rank_, indirect_comparison_above_only_, model_conditional_rel_,
            // Case 1: (+, −)
            [&](data_size_t idxA, data_size_t idxB) noexcept { apply_update(idxA, idxB, +1, -1); },
            // Case 2: (−, −)
            [&](data_size_t idxA, data_size_t idxB) noexcept { apply_update(idxA, idxB, -1, -1); },
            // Case 3: (+, +)
            [&](data_size_t idxA, data_size_t idxB) noexcept { apply_update(idxA, idxB, +1, +1); },
            // Case 4: (−, +)
            [&](data_size_t idxA, data_size_t idxB) noexcept { apply_update(idxA, idxB, -1, +1); }
          );
        }
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }

    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t pair = 0; pair < cnt_pairwise; ++pair) {
        lambdas_pairwise[pair] = static_cast<score_t>(lambdas_pairwise[pair] * norm_factor);
        hessians_pairwise[pair] = static_cast<score_t>(hessians_pairwise[pair] * norm_factor);
      }
    }
  }

  inline double GetSigmoid(double score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    }
    else if (score >= max_sigmoid_input_) {
      // too large, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    }
    else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
        sigmoid_table_idx_factor_)];
    }
  }

  const char* GetName() const override { return "pairwise_lambdarank"; }

 protected:
   /*! \brief Query boundaries for pairwise data instances */
   const data_size_t* query_boundaries_pairwise_;
   /*! \brief Number of pairwise data */
   data_size_t num_data_pairwise_;
   mutable std::vector<double> scores_pointwise_;
   bool model_indirect_comparison_;
   bool model_conditional_rel_;
   bool indirect_comparison_above_only_;
   bool logarithmic_discounts_;
   bool hard_pairwise_preference_;
   int indirect_comparison_max_rank_;
   double indirect_comparison_weight_;
   double l2_pairwise_diff_weight_;

 private:
  const std::pair<data_size_t, data_size_t>* paired_index_map_;
  std::vector<std::vector<std::vector<std::pair<short, data_size_t>>>> right2left2pair_map_byquery_;
  std::vector<std::vector<std::vector<std::pair<short, data_size_t>>>> left2right2pair_map_byquery_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_OBJECTIVE_RANK_OBJECTIVE_HPP_
