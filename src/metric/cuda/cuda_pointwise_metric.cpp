/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_binary_metric.hpp"
#include "cuda_pointwise_metric.hpp"
#include "cuda_regression_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
void CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAMetricInterface<HOST_METRIC>::Init(metadata, num_data);
  const int max_num_reduce_blocks = (this->num_data_ + NUM_DATA_PER_EVAL_THREAD - 1) / NUM_DATA_PER_EVAL_THREAD;
  if (this->cuda_weights_ == nullptr) {
    reduce_block_buffer_.Resize(max_num_reduce_blocks);
  } else {
    reduce_block_buffer_.Resize(max_num_reduce_blocks * 2);
  }
  const int max_num_reduce_blocks_inner = (max_num_reduce_blocks + NUM_DATA_PER_EVAL_THREAD - 1) / NUM_DATA_PER_EVAL_THREAD;
  if (this->cuda_weights_ == nullptr) {
    reduce_block_buffer_inner_.Resize(max_num_reduce_blocks_inner);
  } else {
    reduce_block_buffer_inner_.Resize(max_num_reduce_blocks_inner * 2);
  }
}

// Regression metrics
template void CUDAPointwiseMetricInterface<RMSEMetric, CUDARMSEMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<L2Metric, CUDAL2Metric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<QuantileMetric, CUDAQuantileMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<L1Metric, CUDAL1Metric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<HuberLossMetric, CUDAHuberLossMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<FairLossMetric, CUDAFairLossMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<PoissonMetric, CUDAPoissonMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<MAPEMetric, CUDAMAPEMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<GammaMetric, CUDAGammaMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<GammaDevianceMetric, CUDAGammaDevianceMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<TweedieMetric, CUDATweedieMetric>::Init(const Metadata& metadata, data_size_t num_data);

// Binary metrics
template void CUDAPointwiseMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDAPointwiseMetricInterface<BinaryErrorMetric, CUDABinaryErrorMetric>::Init(const Metadata& metadata, data_size_t num_data);

}  // namespace LightGBM

#endif  // USE_CUDA
