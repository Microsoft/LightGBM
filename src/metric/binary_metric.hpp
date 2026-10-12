/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_BINARY_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_BINARY_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

namespace LightGBM {

/*!
* \brief Metric for binary classification task.
* Use static class "PointWiseLossCalculator" to calculate loss point-wise
*/
template<typename PointWiseLossCalculator>
class BinaryMetric: public Metric {
 public:
  explicit BinaryMetric(const Config&) {
  }

  virtual ~BinaryMetric() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back(PointWiseLossCalculator::Name());

    num_data_ = num_data;
    // get label
    label_ = metadata.label();

    // get weights
    weights_ = metadata.weights();

    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i]);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i]) * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double prob = 0;
          objective->ConvertOutput(&score[i], &prob);
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double prob = 0;
          objective->ConvertOutput(&score[i], &prob);
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob) * weights_[i];
        }
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

 protected:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of test set */
  std::vector<std::string> name_;
};

/*!
* \brief Log loss metric for binary classification task.
*/
class BinaryLoglossMetric: public BinaryMetric<BinaryLoglossMetric> {
 public:
  explicit BinaryLoglossMetric(const Config& config) :BinaryMetric<BinaryLoglossMetric>(config) {}

  inline static double LossOnPoint(label_t label, double prob) {
    if (label <= 0) {
      if (1.0f - prob > kEpsilon) {
        return -std::log(1.0f - prob);
      }
    } else {
      if (prob > kEpsilon) {
        return -std::log(prob);
      }
    }
    return -std::log(kEpsilon);
  }

  inline static const char* Name() {
    return "binary_logloss";
  }
};
/*!
* \brief Error rate metric for binary classification task.
*/
class BinaryErrorMetric: public BinaryMetric<BinaryErrorMetric> {
 public:
  explicit BinaryErrorMetric(const Config& config) :BinaryMetric<BinaryErrorMetric>(config) {}

  inline static double LossOnPoint(label_t label, double prob) {
    if (prob <= 0.5f) {
      return label > 0;
    } else {
      return label <= 0;
    }
  }

  inline static const char* Name() {
    return "binary_error";
  }
};

/*!
* \brief Auc Metric for binary classification task.
*/
class AUCMetric: public Metric {
 public:
  explicit AUCMetric(const Config&) {
  }

  virtual ~AUCMetric() {
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("auc");

    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();

    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    // get indices sorted by score, descent order
    std::vector<data_size_t> sorted_idx;
    for (data_size_t i = 0; i < num_data_; ++i) {
      sorted_idx.emplace_back(i);
    }
    Common::ParallelSort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });
    // temp sum of positive label
    double cur_pos = 0.0f;
    // total sum of positive label
    double sum_pos = 0.0f;
    // accumulate of AUC
    double accum = 0.0f;
    // temp sum of negative label
    double cur_neg = 0.0f;
    double threshold = score[sorted_idx[0]];
    if (weights_ == nullptr) {  // no weights
      for (data_size_t i = 0; i < num_data_; ++i) {
        const label_t cur_label = label_[sorted_idx[i]];
        const double cur_score = score[sorted_idx[i]];
        // new threshold
        if (cur_score != threshold) {
          threshold = cur_score;
          // accumulate
          accum += cur_neg*(cur_pos * 0.5f + sum_pos);
          sum_pos += cur_pos;
          // reset
          cur_neg = cur_pos = 0.0f;
        }
        cur_neg += (cur_label <= 0);
        cur_pos += (cur_label > 0);
      }
    } else {  // has weights
      for (data_size_t i = 0; i < num_data_; ++i) {
        const label_t cur_label = label_[sorted_idx[i]];
        const double cur_score = score[sorted_idx[i]];
        const label_t cur_weight = weights_[sorted_idx[i]];
        // new threshold
        if (cur_score != threshold) {
          threshold = cur_score;
          // accumulate
          accum += cur_neg*(cur_pos * 0.5f + sum_pos);
          sum_pos += cur_pos;
          // reset
          cur_neg = cur_pos = 0.0f;
        }
        cur_neg += (cur_label <= 0)*cur_weight;
        cur_pos += (cur_label > 0)*cur_weight;
      }
    }
    accum += cur_neg*(cur_pos * 0.5f + sum_pos);
    sum_pos += cur_pos;
    double auc = 1.0f;
    if (sum_pos > 0.0f && sum_pos != sum_weights_) {
      auc = accum / (sum_pos *(sum_weights_ - sum_pos));
    }
    return std::vector<double>(1, auc);
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of test set */
  std::vector<std::string> name_;
};


/*!
* \brief Average Precision Metric for binary classification task.
*/
class AveragePrecisionMetric: public Metric {
 public:
  explicit AveragePrecisionMetric(const Config&) {
  }

  virtual ~AveragePrecisionMetric() {
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("average_precision");

    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();

    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    // get indices sorted by score, descending order
    std::vector<data_size_t> sorted_idx;
    for (data_size_t i = 0; i < num_data_; ++i) {
      sorted_idx.emplace_back(i);
    }
    Common::ParallelSort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });
    // temp sum of positive label
    double cur_actual_pos = 0.0f;
    // total sum of positive label
    double sum_actual_pos = 0.0f;
    // total sum of predicted positive
    double sum_pred_pos = 0.0f;
    // accumulated precision
    double accum_prec = 1.0f;
    // accumulated pr-auc
    double accum = 0.0f;
    // temp sum of negative label
    double cur_neg = 0.0f;
    double threshold = score[sorted_idx[0]];
    if (weights_ == nullptr) {  // no weights
      for (data_size_t i = 0; i < num_data_; ++i) {
        const label_t cur_label = label_[sorted_idx[i]];
        const double cur_score = score[sorted_idx[i]];
        // new threshold
        if (cur_score != threshold) {
          threshold = cur_score;
          // accumulate
          sum_actual_pos += cur_actual_pos;
          sum_pred_pos += cur_actual_pos + cur_neg;
          accum_prec = sum_actual_pos / sum_pred_pos;
          accum += cur_actual_pos * accum_prec;
          // reset
          cur_neg = cur_actual_pos = 0.0f;
        }
        cur_neg += (cur_label <= 0);
        cur_actual_pos += (cur_label > 0);
      }
    } else {  // has weights
      for (data_size_t i = 0; i < num_data_; ++i) {
        const label_t cur_label = label_[sorted_idx[i]];
        const double cur_score = score[sorted_idx[i]];
        const label_t cur_weight = weights_[sorted_idx[i]];
        // new threshold
        if (cur_score != threshold) {
          threshold = cur_score;
          // accumulate
          sum_actual_pos += cur_actual_pos;
          sum_pred_pos += cur_actual_pos + cur_neg;
          accum_prec = sum_actual_pos / sum_pred_pos;
          accum += cur_actual_pos * accum_prec;
          // reset
          cur_neg = cur_actual_pos = 0.0f;
        }
        cur_neg += (cur_label <= 0) * cur_weight;
        cur_actual_pos += (cur_label > 0) * cur_weight;
      }
    }
    sum_actual_pos += cur_actual_pos;
    sum_pred_pos += cur_actual_pos + cur_neg;
    accum_prec = sum_actual_pos / sum_pred_pos;
    accum += cur_actual_pos * accum_prec;
    double ap = 1.0f;
    if (sum_actual_pos > 0.0f && sum_actual_pos != sum_weights_) {
      ap = accum / sum_actual_pos;
    }
    return std::vector<double>(1, ap);
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of test set */
  std::vector<std::string> name_;
};

}  // namespace LightGBM
#endif   // LIGHTGBM_SRC_METRIC_BINARY_METRIC_HPP_
