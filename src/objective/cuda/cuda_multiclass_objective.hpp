/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_MULTICLASS_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_MULTICLASS_OBJECTIVE_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_objective_function.hpp>

#include <memory>
#include <string>
#include <vector>

#include "cuda_binary_objective.hpp"

#include "../multiclass_objective.hpp"

#define GET_GRADIENTS_BLOCK_SIZE_MULTICLASS (1024)

namespace LightGBM {

class CUDAMulticlassSoftmax: public CUDAObjectiveInterface<MulticlassSoftmax> {
 public:
  explicit CUDAMulticlassSoftmax(const Config& config);

  explicit CUDAMulticlassSoftmax(const std::vector<std::string>& strs);

  ~CUDAMulticlassSoftmax();

  void Init(const Metadata& metadata, data_size_t num_data) override;

 private:
  void LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const;

  const double* LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const;

  // CUDA memory, held by this object
  CUDAVector<double> cuda_softmax_buffer_;
};


class CUDAMulticlassOVA: public CUDAObjectiveInterface<MulticlassOVA> {
 public:
  explicit CUDAMulticlassOVA(const Config& config);

  explicit CUDAMulticlassOVA(const std::vector<std::string>& strs);

  void Init(const Metadata& metadata, data_size_t num_data) override;

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override;

  const double* ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const override;

  double BoostFromScore(int class_id) const override {
    return cuda_binary_loss_[class_id]->BoostFromScore(0);
  }

  bool ClassNeedTrain(int class_id) const override {
    return cuda_binary_loss_[class_id]->ClassNeedTrain(0);
  }

  ~CUDAMulticlassOVA();

  bool IsCUDAObjective() const override { return true; }

 private:
  void LaunchGetGradientsKernel(const double* /*scores*/, score_t* /*gradients*/, score_t* /*hessians*/) const {}

  std::vector<std::unique_ptr<CUDABinaryLogloss>> cuda_binary_loss_;
};


}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_MULTICLASS_OBJECTIVE_HPP_
