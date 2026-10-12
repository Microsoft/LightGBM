/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <vector>

#include "cuda_regression_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
std::vector<double> CUDARegressionMetricInterface<HOST_METRIC, CUDA_METRIC>::Eval(const double* score, const ObjectiveFunction* objective) const {
  const double* score_convert = score;
  if (objective != nullptr && objective->NeedConvertOutputCUDA()) {
    this->score_convert_buffer_.Resize(static_cast<size_t>(this->num_data_) * static_cast<size_t>(this->num_class_));
    score_convert = objective->ConvertOutputCUDA(this->num_data_, score, this->score_convert_buffer_.RawData());
  }
  double sum_loss = 0.0, sum_weight = 0.0;
  this->LaunchEvalKernel(score_convert, &sum_loss, &sum_weight);
  const double eval_score = this->AverageLoss(sum_loss, sum_weight);
  return std::vector<double>{eval_score};
}

CUDARMSEMetric::CUDARMSEMetric(const Config& config): CUDARegressionMetricInterface<RMSEMetric, CUDARMSEMetric>(config) {}

CUDAL2Metric::CUDAL2Metric(const Config& config): CUDARegressionMetricInterface<L2Metric, CUDAL2Metric>(config) {}

CUDAQuantileMetric::CUDAQuantileMetric(const Config& config): CUDARegressionMetricInterface<QuantileMetric, CUDAQuantileMetric>(config), alpha_(config.alpha) {}

CUDAL1Metric::CUDAL1Metric(const Config& config): CUDARegressionMetricInterface<L1Metric, CUDAL1Metric>(config) {}

CUDAHuberLossMetric::CUDAHuberLossMetric(const Config& config): CUDARegressionMetricInterface<HuberLossMetric, CUDAHuberLossMetric>(config), alpha_(config.alpha) {}

CUDAFairLossMetric::CUDAFairLossMetric(const Config& config): CUDARegressionMetricInterface<FairLossMetric, CUDAFairLossMetric>(config) , fair_c_(config.fair_c) {}

CUDAPoissonMetric::CUDAPoissonMetric(const Config& config): CUDARegressionMetricInterface<PoissonMetric, CUDAPoissonMetric>(config) {}

CUDAMAPEMetric::CUDAMAPEMetric(const Config& config): CUDARegressionMetricInterface<MAPEMetric, CUDAMAPEMetric>(config) {}

CUDAGammaMetric::CUDAGammaMetric(const Config& config): CUDARegressionMetricInterface<GammaMetric, CUDAGammaMetric>(config) {}

CUDAGammaDevianceMetric::CUDAGammaDevianceMetric(const Config& config): CUDARegressionMetricInterface<GammaDevianceMetric, CUDAGammaDevianceMetric>(config) {}

CUDATweedieMetric::CUDATweedieMetric(const Config& config): CUDARegressionMetricInterface<TweedieMetric, CUDATweedieMetric>(config) , tweedie_variance_power_(config.tweedie_variance_power) {}

}  // namespace LightGBM

#endif  // USE_CUDA
