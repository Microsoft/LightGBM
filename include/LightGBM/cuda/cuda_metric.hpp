/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_METRIC_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_METRIC_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/metric.h>

namespace LightGBM {

template <typename HOST_METRIC>
class CUDAMetricInterface: public HOST_METRIC {
 public:
  explicit CUDAMetricInterface(const Config& config): HOST_METRIC(config) {
    cuda_labels_ = nullptr;
    cuda_weights_ = nullptr;
    const int gpu_device_id = config.gpu_device_id >= 0 ? config.gpu_device_id : 0;
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    HOST_METRIC::Init(metadata, num_data);
    cuda_labels_ = metadata.cuda_metadata()->cuda_label();
    cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  }

  bool IsCUDAMetric() const { return true; }

 protected:
  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_METRIC_HPP_
