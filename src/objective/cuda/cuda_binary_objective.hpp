/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_BINARY_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_BINARY_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define GET_GRADIENTS_BLOCK_SIZE_BINARY (1024)
#define CALC_INIT_SCORE_BLOCK_SIZE_BINARY (1024)

#include <LightGBM/cuda/cuda_objective_function.hpp>

#include <string>
#include <vector>

#include "../binary_objective.hpp"

namespace LightGBM {

class CUDABinaryLogloss : public CUDAObjectiveInterface<BinaryLogloss> {
 public:
  explicit CUDABinaryLogloss(const Config& config);

  explicit CUDABinaryLogloss(const Config& config, const int ova_class_id);

  explicit CUDABinaryLogloss(const std::vector<std::string>& strs);

  ~CUDABinaryLogloss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  bool NeedConvertOutputCUDA() const override { return true; }

 private:
  void LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const override;

  double LaunchCalcInitScoreKernel(const int class_id) const override;

  const double* LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const override;

  void LaunchResetOVACUDALabelKernel() const;

  // CUDA memory, held by other objects
  const label_t* cuda_label_;
  CUDAVector<label_t> cuda_ova_label_;
  const label_t* cuda_weights_;

  // CUDA memory, held by this object
  CUDAVector<double> cuda_boost_from_score_;
  CUDAVector<double> cuda_sum_weights_;
  CUDAVector<double> cuda_label_weights_;
  const int ova_class_id_ = -1;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_BINARY_OBJECTIVE_HPP_
