/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_SRC_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_metric.hpp>
#include <LightGBM/cuda/cuda_utils.hu>

#include <vector>

#include "cuda_regression_metric.hpp"
#include "../binary_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
class CUDABinaryMetricInterface: public CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC> {
 public:
  explicit CUDABinaryMetricInterface(const Config& config): CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>(config) {}

  virtual ~CUDABinaryMetricInterface() {}

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;
};

class CUDABinaryLoglossMetric: public CUDABinaryMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric> {
 public:
  explicit CUDABinaryLoglossMetric(const Config& config);

  virtual ~CUDABinaryLoglossMetric() {}

  __device__ static double MetricOnPointCUDA(label_t label, double score, const double /*param*/) {
    // score should have been converted to probability
    if (label <= 0) {
      if (1.0f - score > kEpsilon) {
        return -log(1.0f - score);
      }
    } else {
      if (score > kEpsilon) {
        return -log(score);
      }
    }
    return -log(kEpsilon);
  }
};

class CUDABinaryErrorMetric: public CUDABinaryMetricInterface<BinaryErrorMetric, CUDABinaryErrorMetric> {
 public:
  explicit CUDABinaryErrorMetric(const Config& config);

  virtual ~CUDABinaryErrorMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score, const double /*param*/) {
    if (score <= 0.5f) {
      return label > 0;
    } else {
      return label <= 0;
    }
  }
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_SRC_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_
