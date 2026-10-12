/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_binary_objective.hpp"

#include <string>
#include <vector>

namespace LightGBM {

CUDABinaryLogloss::CUDABinaryLogloss(const Config& config):
CUDAObjectiveInterface<BinaryLogloss>(config), ova_class_id_(-1) {
  cuda_label_ = nullptr;
  cuda_weights_ = nullptr;
}

CUDABinaryLogloss::CUDABinaryLogloss(const Config& config, const int ova_class_id):
CUDAObjectiveInterface<BinaryLogloss>(config), ova_class_id_(ova_class_id) {
  is_pos_ = [ova_class_id](label_t label) { return static_cast<int>(label) == ova_class_id; };
}

CUDABinaryLogloss::CUDABinaryLogloss(const std::vector<std::string>& strs): CUDAObjectiveInterface<BinaryLogloss>(strs) {}

CUDABinaryLogloss::~CUDABinaryLogloss() {}

void CUDABinaryLogloss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAObjectiveInterface<BinaryLogloss>::Init(metadata, num_data);
  if (ova_class_id_ == -1) {
    cuda_label_ = metadata.cuda_metadata()->cuda_label();
    cuda_ova_label_.Clear();
  } else {
    cuda_ova_label_.Resize(static_cast<size_t>(num_data));
    CopyFromHostToCUDADevice<label_t>(cuda_ova_label_.RawData(), metadata.cuda_metadata()->cuda_label(), static_cast<size_t>(num_data), __FILE__, __LINE__);
    LaunchResetOVACUDALabelKernel();
    cuda_label_ = cuda_ova_label_.RawData();
  }
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  cuda_boost_from_score_.Resize(1);
  SetCUDAMemory<double>(cuda_boost_from_score_.RawData(), 0, 1, __FILE__, __LINE__);
  cuda_sum_weights_.Resize(1);
  SetCUDAMemory<double>(cuda_sum_weights_.RawData(), 0, 1, __FILE__, __LINE__);
  if (label_weights_[0] != 1.0f || label_weights_[1] != 1.0f) {
    cuda_label_weights_.Resize(2);
    CopyFromHostToCUDADevice<double>(cuda_label_weights_.RawData(), label_weights_, 2, __FILE__, __LINE__);
  } else {
    cuda_label_weights_.Clear();
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
