/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_SRC_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_metric.hpp>
#include <LightGBM/cuda/cuda_utils.hu>

#include <vector>

#include "cuda_pointwise_metric.hpp"
#include "../regression_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
class CUDARegressionMetricInterface: public CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC> {
 public:
  explicit CUDARegressionMetricInterface(const Config& config):
    CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>(config) {}

  virtual ~CUDARegressionMetricInterface() {}

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;
};

class CUDARMSEMetric: public CUDARegressionMetricInterface<RMSEMetric, CUDARMSEMetric> {
 public:
  explicit CUDARMSEMetric(const Config& config);

  virtual ~CUDARMSEMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score, double /*alpha*/) {
    return (score - label) * (score - label);
  }
};

class CUDAL2Metric : public CUDARegressionMetricInterface<L2Metric, CUDAL2Metric> {
 public:
  explicit CUDAL2Metric(const Config& config);

  virtual ~CUDAL2Metric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score, double /*alpha*/) {
    return (score - label) * (score - label);
  }
};

class CUDAQuantileMetric : public CUDARegressionMetricInterface<QuantileMetric, CUDAQuantileMetric> {
 public:
  explicit CUDAQuantileMetric(const Config& config);

  virtual ~CUDAQuantileMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score, double alpha) {
    double delta = label - score;
    if (delta < 0) {
      return (alpha - 1.0f) * delta;
    } else {
      return alpha * delta;
    }
  }

  double GetParamFromConfig() const override {
    return alpha_;
  }

 private:
  const double alpha_;
};

class CUDAL1Metric : public CUDARegressionMetricInterface<L1Metric, CUDAL1Metric> {
 public:
  explicit CUDAL1Metric(const Config& config);

  virtual ~CUDAL1Metric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double /*alpha*/) {
    return std::fabs(score - label);
  }
};

class CUDAHuberLossMetric : public CUDARegressionMetricInterface<HuberLossMetric, CUDAHuberLossMetric> {
 public:
  explicit CUDAHuberLossMetric(const Config& config);

  virtual ~CUDAHuberLossMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double alpha) {
    const double diff = score - label;
    if (std::abs(diff) <= alpha) {
      return 0.5f * diff * diff;
    } else {
      return alpha * (std::abs(diff) - 0.5f * alpha);
    }
  }

  double GetParamFromConfig() const override {
    return alpha_;
  }
 private:
  const double alpha_;
};

class CUDAFairLossMetric : public CUDARegressionMetricInterface<FairLossMetric, CUDAFairLossMetric> {
 public:
  explicit CUDAFairLossMetric(const Config& config);

  virtual ~CUDAFairLossMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double fair_c) {
    const double x = std::fabs(score - label);
    const double c =  fair_c;
    return c * x - c * c * std::log1p(x / c);
  }

  double GetParamFromConfig() const override {
    return fair_c_;
  }

 private:
  const double fair_c_;
};

class CUDAPoissonMetric : public CUDARegressionMetricInterface<PoissonMetric, CUDAPoissonMetric> {
 public:
  explicit CUDAPoissonMetric(const Config& config);

  virtual ~CUDAPoissonMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double /*alpha*/) {
    const double eps = 1e-10f;
    if (score < eps) {
      score = eps;
    }
    return score - label * std::log(score);
  }
};

class CUDAMAPEMetric : public CUDARegressionMetricInterface<MAPEMetric, CUDAMAPEMetric> {
 public:
  explicit CUDAMAPEMetric(const Config& config);

  virtual ~CUDAMAPEMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double /*alpha*/) {
    return std::fabs((label - score)) / fmax(1.0f, std::fabs(label));
  }
};

class CUDAGammaMetric : public CUDARegressionMetricInterface<GammaMetric, CUDAGammaMetric> {
 public:
  explicit CUDAGammaMetric(const Config& config);

  virtual ~CUDAGammaMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double /*alpha*/) {
    const double psi = 1.0;
    const double theta = -1.0 / score;
    const double a = psi;
    const double b = -SafeLog(-theta);
    const double c = 1. / psi * SafeLog(label / psi) - SafeLog(label) - 0;  // 0 = std::lgamma(1.0 / psi) = std::lgamma(1.0);
    return -((label * theta - b) / a + c);
  }
};

class CUDAGammaDevianceMetric : public CUDARegressionMetricInterface<GammaDevianceMetric, CUDAGammaDevianceMetric> {
 public:
  explicit CUDAGammaDevianceMetric(const Config& config);

  virtual ~CUDAGammaDevianceMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double /*alpha*/) {
    const double epsilon = 1.0e-9;
    const double tmp = label / (score + epsilon);
    return tmp - SafeLog(tmp) - 1;
  }
};

class CUDATweedieMetric : public CUDARegressionMetricInterface<TweedieMetric, CUDATweedieMetric> {
 public:
  explicit CUDATweedieMetric(const Config& config);

  virtual ~CUDATweedieMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double tweedie_variance_power) {
    const double rho = tweedie_variance_power;
    const double eps = 1e-10f;
    if (score < eps) {
      score = eps;
    }
    const double a = label * std::exp((1 - rho) * std::log(score)) / (1 - rho);
    const double b = std::exp((2 - rho) * std::log(score)) / (2 - rho);
    return -a + b;
  }

  double GetParamFromConfig() const override {
    return tweedie_variance_power_;
  }

 private:
  const double tweedie_variance_power_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_SRC_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_
