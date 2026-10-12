/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/objective_function.h>
#include <LightGBM/meta.h>

#include <string>
#include <vector>

namespace LightGBM {

template <typename HOST_OBJECTIVE>
class CUDAObjectiveInterface: public HOST_OBJECTIVE, public NCCLInfo {
 public:
  explicit CUDAObjectiveInterface(const Config& config): HOST_OBJECTIVE(config) {
    if (config.num_gpu <= 1) {
      const int gpu_device_id = config.gpu_device_id >= 0 ? config.gpu_device_id : 0;
      SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
    }
  }

  explicit CUDAObjectiveInterface(const std::vector<std::string>& strs): HOST_OBJECTIVE(strs) {}

  void Init(const Metadata& metadata, data_size_t num_data) {
    HOST_OBJECTIVE::Init(metadata, num_data);
    cuda_labels_ = metadata.cuda_metadata()->cuda_label();
    cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  }

  void SetNCCLInfo(
    ncclComm_t nccl_communicator,
    int nccl_gpu_rank,
    int local_gpu_rank,
    int gpu_device_id,
    data_size_t global_num_data) override {
    NCCLInfo::SetNCCLInfo(nccl_communicator, nccl_gpu_rank, local_gpu_rank, gpu_device_id, global_num_data);
  }

  virtual const double* ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
    return LaunchConvertOutputCUDAKernel(num_data, input, output);
  }

  double BoostFromScore(int class_id) const override {
    return LaunchCalcInitScoreKernel(class_id);
  }

  bool IsCUDAObjective() const override { return true; }

  void GetGradients(const double* scores, score_t* gradients, score_t* hessians) const override {
    LaunchGetGradientsKernel(scores, gradients, hessians);
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }

  void GetGradientsWithSampledQueries(const double* scores, const data_size_t /*num_sampled_queries*/, const data_size_t* /*sampled_query_indices*/, score_t* gradients, score_t* hessians) const override {
    LaunchGetGradientsKernel(scores, gradients, hessians);
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }

  void RenewTreeOutputCUDA(const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override {
    global_timer.Start("CUDAObjectiveInterface::LaunchRenewTreeOutputCUDAKernel");
    LaunchRenewTreeOutputCUDAKernel(score, data_indices_in_leaf, num_data_in_leaf, data_start_in_leaf, num_leaves, leaf_value);
    SynchronizeCUDADevice(__FILE__, __LINE__);
    global_timer.Stop("CUDAObjectiveInterface::LaunchRenewTreeOutputCUDAKernel");
  }

 protected:
  virtual void LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const = 0;

  virtual double LaunchCalcInitScoreKernel(const int class_id) const {
    return HOST_OBJECTIVE::BoostFromScore(class_id);
  }

  virtual const double* LaunchConvertOutputCUDAKernel(const data_size_t /*num_data*/, const double* input, double* /*output*/) const { return input; }

  virtual void LaunchRenewTreeOutputCUDAKernel(
    const double* /*score*/, const data_size_t* /*data_indices_in_leaf*/, const data_size_t* /*num_data_in_leaf*/,
    const data_size_t* /*data_start_in_leaf*/, const int /*num_leaves*/, double* /*leaf_value*/) const {}

  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_
