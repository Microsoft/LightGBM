/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_SRC_BOOSTING_CUDA_NCCL_GBDT_COMPONENT_HPP_
#define LIGHTGBM_SRC_BOOSTING_CUDA_NCCL_GBDT_COMPONENT_HPP_

#ifdef USE_CUDA

#include <LightGBM/objective_function.h>
#include <LightGBM/tree.h>

#include <algorithm>
#include <vector>
#include <memory>

#include <LightGBM/cuda/cuda_objective_function.hpp>
#include "cuda_score_updater.hpp"
#include "../../treelearner/cuda/cuda_single_gpu_tree_learner.hpp"

namespace LightGBM {

class NCCLGBDTComponent: public NCCLInfo {
 public:
  NCCLGBDTComponent() {}

  ~NCCLGBDTComponent() {}

  void Init(const Config* config, const Dataset* train_data, const int num_tree_per_iteration, const bool boosting_on_gpu, const bool is_constant_hessian) {
    CUDASUCCESS_OR_FATAL(cudaGetDeviceCount(&num_gpu_in_node_));
    const data_size_t num_data_per_gpu = (train_data->num_data() + num_gpu_in_node_ - 1) / num_gpu_in_node_;
    data_start_index_ = num_data_per_gpu * local_gpu_rank_;
    data_end_index_ = std::min<data_size_t>(data_start_index_ + num_data_per_gpu, train_data->num_data());
    num_data_in_gpu_ = data_end_index_ - data_start_index_;

    dataset_.reset(new Dataset(num_data_in_gpu_));
    dataset_->ReSize(num_data_in_gpu_);
    dataset_->CopyFeatureMapperFrom(train_data);
    std::vector<data_size_t> used_indices(num_data_in_gpu_);
    for (data_size_t data_index = data_start_index_; data_index < data_end_index_; ++data_index) {
      used_indices[data_index - data_start_index_] = data_index;
    }
    dataset_->CopySubrowToDevice(train_data, used_indices.data(), num_data_in_gpu_, true, gpu_device_id_);

    objective_function_.reset(ObjectiveFunction::CreateObjectiveFunctionCUDA(config->objective, *config));
    objective_function_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, train_data->num_data());
    train_score_updater_.reset(new CUDAScoreUpdater(dataset_.get(), num_tree_per_iteration, boosting_on_gpu));
    gradients_.reset(new CUDAVector<score_t>(num_data_in_gpu_));
    hessians_.reset(new CUDAVector<score_t>(num_data_in_gpu_));
    tree_learner_.reset(new CUDASingleGPUTreeLearner(config, boosting_on_gpu));

    tree_learner_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, train_data->num_data());

    objective_function_->Init(dataset_->metadata(), dataset_->num_data());
    tree_learner_->Init(dataset_.get(), is_constant_hessian);
  }

  ObjectiveFunction* objective_function() { return objective_function_.get(); }

  ScoreUpdater* train_score_updater() { return train_score_updater_.get(); }

  score_t* gradients() { return gradients_->RawData(); }

  score_t* hessians() { return hessians_->RawData(); }

  data_size_t num_data_in_gpu() const { return num_data_in_gpu_; }

  CUDASingleGPUTreeLearner* tree_learner() { return tree_learner_.get(); }

  void SetTree(Tree* tree) {
    new_tree_.reset(tree);
  }

  data_size_t data_start_index() const { return data_start_index_; }

  data_size_t data_end_index() const { return data_end_index_; }

  Tree* new_tree() { return new_tree_.get(); }

  Tree* release_new_tree() { return new_tree_.release(); }

  void clear_new_tree() { new_tree_.reset(nullptr); }

 private:
  std::unique_ptr<ObjectiveFunction> objective_function_;
  std::unique_ptr<ScoreUpdater> train_score_updater_;
  std::unique_ptr<CUDAVector<score_t>> gradients_;
  std::unique_ptr<CUDAVector<score_t>> hessians_;
  std::unique_ptr<Dataset> dataset_;
  std::unique_ptr<CUDASingleGPUTreeLearner> tree_learner_;
  std::unique_ptr<Tree> new_tree_;

  data_size_t data_start_index_;
  data_size_t data_end_index_;
  data_size_t num_data_in_gpu_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_SRC_BOOSTING_CUDA_NCCL_GBDT_COMPONENT_HPP_
