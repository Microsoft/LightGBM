/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include "cuda_multiclass_objective.hpp"

#include <string>
#include <vector>

namespace LightGBM {

CUDAMulticlassSoftmax::CUDAMulticlassSoftmax(const Config& config): CUDAObjectiveInterface<MulticlassSoftmax>(config) {}

CUDAMulticlassSoftmax::CUDAMulticlassSoftmax(const std::vector<std::string>& strs): CUDAObjectiveInterface<MulticlassSoftmax>(strs) {}

CUDAMulticlassSoftmax::~CUDAMulticlassSoftmax() {}

void CUDAMulticlassSoftmax::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAObjectiveInterface<MulticlassSoftmax>::Init(metadata, num_data);
  cuda_softmax_buffer_.Resize(static_cast<size_t>(num_data) * static_cast<size_t>(num_class_));
  SynchronizeCUDADevice(__FILE__, __LINE__);
}


CUDAMulticlassOVA::CUDAMulticlassOVA(const Config& config): CUDAObjectiveInterface<MulticlassOVA>(config) {
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_.emplace_back(new CUDABinaryLogloss(config, i));
  }
}

CUDAMulticlassOVA::CUDAMulticlassOVA(const std::vector<std::string>& strs): CUDAObjectiveInterface<MulticlassOVA>(strs) {}

CUDAMulticlassOVA::~CUDAMulticlassOVA() {}

void CUDAMulticlassOVA::Init(const Metadata& metadata, data_size_t num_data) {
  MulticlassOVA::Init(metadata, num_data);
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_[i]->Init(metadata, num_data);
  }
}

void CUDAMulticlassOVA::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  for (int i = 0; i < num_class_; ++i) {
    int64_t offset = static_cast<int64_t>(num_data_) * i;
    cuda_binary_loss_[i]->GetGradients(score + offset, gradients + offset, hessians + offset);
  }
}

const double* CUDAMulticlassOVA::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  for (int i = 0; i < num_class_; ++i) {
    cuda_binary_loss_[i]->ConvertOutputCUDA(num_data, input + i * num_data, output + i * num_data);
  }
  return output;
}


}  // namespace LightGBM

#endif  // USE_CUDA
