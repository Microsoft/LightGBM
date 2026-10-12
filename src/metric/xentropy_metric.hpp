/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_XENTROPY_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_XENTROPY_METRIC_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

/*
 * Implements three related metrics:
 *
 * (1) standard cross-entropy that can be used for continuous labels in [0, 1]
 * (2) "intensity-weighted" cross-entropy, also for continuous labels in [0, 1]
 * (3) Kullback-Leibler divergence, also for continuous labels in [0, 1]
 *
 * (3) adds an offset term to (1); the entropy of the label
 *
 * See xentropy_objective.hpp for further details.
 *
 */

namespace LightGBM {

// label should be in interval [0, 1];
// prob should be in interval (0, 1); prob is clipped if needed
inline static double XentLoss(label_t label, double prob) {
  const double log_arg_epsilon = 1.0e-12;
  double a = label;
  if (prob > log_arg_epsilon) {
    a *= std::log(prob);
  } else {
    a *= std::log(log_arg_epsilon);
  }
  double b = 1.0f - label;
  if (1.0f - prob > log_arg_epsilon) {
    b *= std::log(1.0f - prob);
  } else {
    b *= std::log(log_arg_epsilon);
  }
  return - (a + b);
}

// hhat >(=) 0 assumed; and weight > 0 required; but not checked here
inline static double XentLambdaLoss(label_t label, label_t weight, double hhat) {
  return XentLoss(label, 1.0f - std::exp(-weight * hhat));
}

// Computes the (negative) entropy for label p; p should be in interval [0, 1];
// This is used to presum the KL-divergence offset term (to be _added_ to the cross-entropy loss).
// NOTE: x*log(x) = 0 for x=0,1; so only add when in (0, 1); avoid log(0)*0
inline static double YentLoss(double p) {
  double hp = 0.0;
  if (p > 0) hp += p * std::log(p);
  double q = 1.0f - p;
  if (q > 0) hp += q * std::log(q);
  return hp;
}

//
// CrossEntropyMetric : "xentropy" : (optional) weights are used linearly
//
class CrossEntropyMetric : public Metric {
 public:
  explicit CrossEntropyMetric(const Config&) {}
  virtual ~CrossEntropyMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("cross_entropy");
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    CHECK_NOTNULL(label_);

    // ensure that labels are in interval [0, 1], interval ends included
    Common::CheckElementsIntervalClosed<label_t>(label_, 0.0f, 1.0f, num_data_, GetName()[0].c_str());
    Log::Info("[%s:%s]: (metric) labels passed interval [0, 1] check",  GetName()[0].c_str(), __func__);

    // check that weights are non-negative and sum is positive
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      label_t minw;
      Common::ObtainMinMaxSum(weights_, num_data_, &minw, static_cast<label_t*>(nullptr), &sum_weights_);
      if (minw < 0.0f) {
        Log::Fatal("[%s:%s]: (metric) weights not allowed to be negative", GetName()[0].c_str(), __func__);
      }
    }

    // check weight sum (may fail to be zero)
    if (sum_weights_ <= 0.0f) {
      Log::Fatal("[%s:%s]: sum-of-weights = %f is non-positive", __func__, GetName()[0].c_str(), sum_weights_);
    }
    Log::Info("[%s:%s]: sum-of-weights = %f", GetName()[0].c_str(), __func__, sum_weights_);
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          sum_loss += XentLoss(label_[i], score[i]);  // NOTE: does not work unless score is a probability
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          sum_loss += XentLoss(label_[i], score[i]) * weights_[i];  // NOTE: does not work unless score is a probability
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double p = 0;
          objective->ConvertOutput(&score[i], &p);
          sum_loss += XentLoss(label_[i], p);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double p = 0;
          objective->ConvertOutput(&score[i], &p);
          sum_loss += XentLoss(label_[i], p) * weights_[i];
        }
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;  // negative means smaller loss is better, positive means larger loss is better
  }

 private:
  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer to label */
  const label_t* label_;
  /*! \brief Pointer to weights */
  const label_t* weights_;
  /*! \brief Sum of weights */
  double sum_weights_;
  /*! \brief Name of this metric */
  std::vector<std::string> name_;
};

//
// CrossEntropyLambdaMetric : "xentlambda" : (optional) weights have a different meaning than for "xentropy"
// ATTENTION: Supposed to be used when the objective also is "xentlambda"
//
class CrossEntropyLambdaMetric : public Metric {
 public:
  explicit CrossEntropyLambdaMetric(const Config&) {}
  virtual ~CrossEntropyLambdaMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("cross_entropy_lambda");
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    CHECK_NOTNULL(label_);
    Common::CheckElementsIntervalClosed<label_t>(label_, 0.0f, 1.0f, num_data_, GetName()[0].c_str());
    Log::Info("[%s:%s]: (metric) labels passed interval [0, 1] check",  GetName()[0].c_str(), __func__);

    // check all weights are strictly positive; throw error if not
    if (weights_ != nullptr) {
      label_t minw;
      Common::ObtainMinMaxSum(weights_, num_data_, &minw, static_cast<label_t*>(nullptr), static_cast<label_t*>(nullptr));
      if (minw <= 0.0f) {
        Log::Fatal("[%s:%s]: (metric) all weights must be positive", GetName()[0].c_str(), __func__);
      }
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double hhat = std::log1p(std::exp(score[i]));  // auto-convert
          sum_loss += XentLambdaLoss(label_[i], 1.0f, hhat);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double hhat = std::log1p(std::exp(score[i]));  // auto-convert
          sum_loss += XentLambdaLoss(label_[i], weights_[i], hhat);
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double hhat = 0;
          objective->ConvertOutput(&score[i], &hhat);  // NOTE: this only works if objective = "xentlambda"
          sum_loss += XentLambdaLoss(label_[i], 1.0f, hhat);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double hhat = 0;
          objective->ConvertOutput(&score[i], &hhat);  // NOTE: this only works if objective = "xentlambda"
          sum_loss += XentLambdaLoss(label_[i], weights_[i], hhat);
        }
      }
    }
    return std::vector<double>(1, sum_loss / static_cast<double>(num_data_));
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

 private:
  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer to label */
  const label_t* label_;
  /*! \brief Pointer to weights */
  const label_t* weights_;
  /*! \brief Name of this metric */
  std::vector<std::string> name_;
};

//
// KullbackLeiblerDivergence : "kldiv" : (optional) weights are used linearly
//
class KullbackLeiblerDivergence : public Metric {
 public:
  explicit KullbackLeiblerDivergence(const Config&) {}
  virtual ~KullbackLeiblerDivergence() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("kullback_leibler");
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    CHECK_NOTNULL(label_);
    Common::CheckElementsIntervalClosed<label_t>(label_, 0.0f, 1.0f, num_data_, GetName()[0].c_str());
    Log::Info("[%s:%s]: (metric) labels passed interval [0, 1] check",  GetName()[0].c_str(), __func__);

    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      label_t minw;
      Common::ObtainMinMaxSum(weights_, num_data_, &minw, static_cast<label_t*>(nullptr), &sum_weights_);
      if (minw < 0.0f) {
        Log::Fatal("[%s:%s]: (metric) at least one weight is negative", GetName()[0].c_str(), __func__);
      }
    }

    // check weight sum
    if (sum_weights_ <= 0.0f) {
      Log::Fatal("[%s:%s]: sum-of-weights = %f is non-positive", GetName()[0].c_str(), __func__, sum_weights_);
    }

    Log::Info("[%s:%s]: sum-of-weights = %f", GetName()[0].c_str(), __func__, sum_weights_);

    // evaluate offset term
    presum_label_entropy_ = 0.0f;
    if (weights_ == nullptr) {
      for (data_size_t i = 0; i < num_data; ++i) {
        presum_label_entropy_ += YentLoss(label_[i]);
      }
    } else {
      for (data_size_t i = 0; i < num_data; ++i) {
        presum_label_entropy_ += YentLoss(label_[i]) * weights_[i];
      }
    }
    presum_label_entropy_ /= sum_weights_;

    // communicate the value of the offset term to be added
    Log::Info("%s offset term = %f", GetName()[0].c_str(), presum_label_entropy_);
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          sum_loss += XentLoss(label_[i], score[i]);  // NOTE: does not work unless score is a probability
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          sum_loss += XentLoss(label_[i], score[i]) * weights_[i];  // NOTE: does not work unless score is a probability
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double p = 0;
          objective->ConvertOutput(&score[i], &p);
          sum_loss += XentLoss(label_[i], p);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double p = 0;
          objective->ConvertOutput(&score[i], &p);
          sum_loss += XentLoss(label_[i], p) * weights_[i];
        }
      }
    }
    double loss = presum_label_entropy_ + sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

 private:
  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer to label */
  const label_t* label_;
  /*! \brief Pointer to weights */
  const label_t* weights_;
  /*! \brief Sum of weights */
  double sum_weights_;
  /*! \brief Offset term to cross-entropy; precomputed during init */
  double presum_label_entropy_;
  /*! \brief Name of this metric */
  std::vector<std::string> name_;
};

}  // end namespace LightGBM

#endif  // LIGHTGBM_SRC_METRIC_XENTROPY_METRIC_HPP_
