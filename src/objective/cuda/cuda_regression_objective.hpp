/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define GET_GRADIENTS_BLOCK_SIZE_REGRESSION (1024)

#include <LightGBM/cuda/cuda_objective_function.hpp>

#include <string>
#include <vector>

#include "../regression_objective.hpp"

namespace LightGBM {

template <typename HOST_OBJECTIVE>
class CUDARegressionObjectiveInterface: public CUDAObjectiveInterface<HOST_OBJECTIVE> {
 public:
  explicit CUDARegressionObjectiveInterface(const Config& config): CUDAObjectiveInterface<HOST_OBJECTIVE>(config) {}

  explicit CUDARegressionObjectiveInterface(const std::vector<std::string>& strs): CUDAObjectiveInterface<HOST_OBJECTIVE>(strs) {}

  void Init(const Metadata& metadata, data_size_t num_data) override;

 protected:
  double LaunchCalcInitScoreKernel(const int class_id) const override;

  CUDAVector<double> cuda_block_buffer_;
  CUDAVector<label_t> cuda_trans_label_;
};

class CUDARegressionL2loss : public CUDARegressionObjectiveInterface<RegressionL2loss> {
 public:
  explicit CUDARegressionL2loss(const Config& config);

  explicit CUDARegressionL2loss(const std::vector<std::string>& strs);

  ~CUDARegressionL2loss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

 protected:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  const double* LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const override;

  bool NeedConvertOutputCUDA() const override { return sqrt_; }
};


class CUDARegressionL1loss : public CUDARegressionObjectiveInterface<RegressionL1loss> {
 public:
  explicit CUDARegressionL1loss(const Config& config);

  explicit CUDARegressionL1loss(const std::vector<std::string>& strs);

  ~CUDARegressionL1loss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

 protected:
  CUDAVector<data_size_t> cuda_data_indices_buffer_;
  CUDAVector<double> cuda_weights_prefix_sum_;
  CUDAVector<double> cuda_weights_prefix_sum_buffer_;
  CUDAVector<double> cuda_residual_buffer_;
  CUDAVector<label_t> cuda_weight_by_leaf_buffer_;
  CUDAVector<label_t> cuda_percentile_result_;

  double LaunchCalcInitScoreKernel(const int class_id) const override;

  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  void LaunchRenewTreeOutputCUDAKernel(
    const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;
};


class CUDARegressionHuberLoss : public CUDARegressionObjectiveInterface<RegressionHuberLoss> {
 public:
  explicit CUDARegressionHuberLoss(const Config& config);

  explicit CUDARegressionHuberLoss(const std::vector<std::string>& strs);

  ~CUDARegressionHuberLoss();

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;
};


// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class CUDARegressionFairLoss : public CUDARegressionObjectiveInterface<RegressionFairLoss> {
 public:
  explicit CUDARegressionFairLoss(const Config& config);

  explicit CUDARegressionFairLoss(const std::vector<std::string>& strs);

  ~CUDARegressionFairLoss();

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;
};


class CUDARegressionPoissonLoss : public CUDARegressionObjectiveInterface<RegressionPoissonLoss> {
 public:
  explicit CUDARegressionPoissonLoss(const Config& config);

  explicit CUDARegressionPoissonLoss(const std::vector<std::string>& strs);

  ~CUDARegressionPoissonLoss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  const double* LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const override;

  bool NeedConvertOutputCUDA() const override { return true; }

  double LaunchCalcInitScoreKernel(const int class_id) const override;

  void LaunchCheckLabelKernel() const;
};


class CUDARegressionQuantileloss : public CUDARegressionObjectiveInterface<RegressionQuantileloss> {
 public:
  explicit CUDARegressionQuantileloss(const Config& config);

  explicit CUDARegressionQuantileloss(const std::vector<std::string>& strs);

  ~CUDARegressionQuantileloss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

 protected:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  double LaunchCalcInitScoreKernel(const int class_id) const override;

  void LaunchRenewTreeOutputCUDAKernel(
    const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;

  CUDAVector<data_size_t> cuda_data_indices_buffer_;
  CUDAVector<double> cuda_weights_prefix_sum_;
  CUDAVector<double> cuda_weights_prefix_sum_buffer_;
  CUDAVector<double> cuda_residual_buffer_;
  CUDAVector<label_t> cuda_weight_by_leaf_buffer_;
  CUDAVector<label_t> cuda_percentile_result_;
};


}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_
