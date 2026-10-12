/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_metadata.hpp>

#include <vector>

namespace LightGBM {

CUDAMetadata::CUDAMetadata(const int gpu_device_id) {
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
}

CUDAMetadata::~CUDAMetadata() {}

void CUDAMetadata::Init(const std::vector<label_t>& label,
                        const std::vector<label_t>& weight,
                        const std::vector<data_size_t>& query_boundaries,
                        const std::vector<label_t>& query_weights,
                        const std::vector<double>& init_score) {
  if (label.size() == 0) {
    cuda_label_.Clear();
  } else {
    cuda_label_.InitFromHostVector(label);
  }
  if (weight.size() == 0) {
    cuda_weights_.Clear();
  } else {
    cuda_weights_.InitFromHostVector(weight);
  }
  if (query_boundaries.size() == 0) {
    cuda_query_boundaries_.Clear();
  } else {
    cuda_query_boundaries_.InitFromHostVector(query_boundaries);
  }
  if (query_weights.size() == 0) {
    cuda_query_weights_.Clear();
  } else {
    cuda_query_weights_.InitFromHostVector(query_weights);
  }
  if (init_score.size() == 0) {
    cuda_init_score_.Clear();
  } else {
    cuda_init_score_.InitFromHostVector(init_score);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAMetadata::SetLabel(const label_t* label, data_size_t len) {
  cuda_label_.InitFromHostMemory(label, static_cast<size_t>(len));
}

void CUDAMetadata::SetWeights(const label_t* weights, data_size_t len) {
  cuda_weights_.InitFromHostMemory(weights, static_cast<size_t>(len));
}

void CUDAMetadata::SetQuery(const data_size_t* query_boundaries, const label_t* query_weights, data_size_t num_queries) {
  cuda_query_boundaries_.InitFromHostMemory(query_boundaries, static_cast<size_t>(num_queries) + 1);
  if (query_weights != nullptr) {
    cuda_query_weights_.InitFromHostMemory(query_weights, static_cast<size_t>(num_queries));
  }
}

void CUDAMetadata::SetInitScore(const double* init_score, data_size_t len) {
  cuda_init_score_.InitFromHostMemory(init_score, len);
}

}  // namespace LightGBM

#endif  // USE_CUDA
