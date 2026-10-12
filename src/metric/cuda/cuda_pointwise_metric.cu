/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/cuda/cuda_rocm_interop.h>

#include "cuda_binary_metric.hpp"
#include "cuda_pointwise_metric.hpp"
#include "cuda_regression_metric.hpp"

namespace LightGBM {

template <typename CUDA_METRIC, bool USE_WEIGHTS>
__global__ void EvalKernel(const data_size_t num_data, const label_t* labels, const label_t* weights,
                           const double* scores, double* reduce_block_buffer, const double param) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    point_metric = USE_WEIGHTS ?
      CUDA_METRIC::MetricOnPointCUDA(labels[index], scores[index], param) * weights[index] :
      CUDA_METRIC::MetricOnPointCUDA(labels[index], scores[index], param);
  }
  const double block_sum_point_metric = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, NUM_DATA_PER_EVAL_THREAD);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum_point_metric;
  }
  if (USE_WEIGHTS) {
    double weight = 0.0;
    if (index < num_data) {
      weight = static_cast<double>(weights[index]);
      const double block_sum_weight = ShuffleReduceSum<double>(weight, shared_mem_buffer, NUM_DATA_PER_EVAL_THREAD);
      if (threadIdx.x == 0) {
        reduce_block_buffer[blockIdx.x + gridDim.x] = block_sum_weight;
      }
    }
  }
}

template <typename HOST_METRIC, typename CUDA_METRIC>
void CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const {
  const int num_blocks = (this->num_data_ + NUM_DATA_PER_EVAL_THREAD - 1) / NUM_DATA_PER_EVAL_THREAD;
  if (this->cuda_weights_ != nullptr) {
    EvalKernel<CUDA_METRIC, true><<<num_blocks, NUM_DATA_PER_EVAL_THREAD>>>(
      this->num_data_, this->cuda_labels_, this->cuda_weights_, score, reduce_block_buffer_.RawData(), GetParamFromConfig());
  } else {
    EvalKernel<CUDA_METRIC, false><<<num_blocks, NUM_DATA_PER_EVAL_THREAD>>>(
      this->num_data_, this->cuda_labels_, this->cuda_weights_, score, reduce_block_buffer_.RawData(), GetParamFromConfig());
  }
  ShuffleReduceSumGlobal<double, double>(reduce_block_buffer_.RawData(), num_blocks, reduce_block_buffer_inner_.RawData());
  CopyFromCUDADeviceToHost<double>(sum_loss, reduce_block_buffer_inner_.RawData(), 1, __FILE__, __LINE__);
  *sum_weight = static_cast<double>(this->num_data_);
  if (this->cuda_weights_ != nullptr) {
    ShuffleReduceSumGlobal<double, double>(reduce_block_buffer_.RawData() + num_blocks, num_blocks, reduce_block_buffer_inner_.RawData());
    CopyFromCUDADeviceToHost<double>(sum_weight, reduce_block_buffer_inner_.RawData(), 1, __FILE__, __LINE__);
  }
}

// Regression metrics
template void CUDAPointwiseMetricInterface<RMSEMetric, CUDARMSEMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<L2Metric, CUDAL2Metric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<QuantileMetric, CUDAQuantileMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<L1Metric, CUDAL1Metric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<HuberLossMetric, CUDAHuberLossMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<FairLossMetric, CUDAFairLossMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<PoissonMetric, CUDAPoissonMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<MAPEMetric, CUDAMAPEMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<GammaMetric, CUDAGammaMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<GammaDevianceMetric, CUDAGammaDevianceMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<TweedieMetric, CUDATweedieMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;

// Binary metrics
template void CUDAPointwiseMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<BinaryErrorMetric, CUDABinaryErrorMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;

}  // namespace LightGBM

#endif  // USE_CUDA
