/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_SRC_BOOSTING_CUDA_NCCL_GBDT_HPP_
#define LIGHTGBM_SRC_BOOSTING_CUDA_NCCL_GBDT_HPP_

#ifdef USE_CUDA

#include <pthread.h>

#include <LightGBM/objective_function.h>
#include <LightGBM/network.h>

#include <memory>
#include <vector>

#include <LightGBM/cuda/cuda_nccl_topology.hpp>

#include "cuda_score_updater.hpp"
#include "nccl_gbdt_component.hpp"

#include "../gbdt.h"

namespace LightGBM {

template <typename GBDT_T>
class NCCLGBDT: public GBDT_T {
 public:
  NCCLGBDT();

  ~NCCLGBDT();

  void Init(const Config* gbdt_config, const Dataset* train_data,
            const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override;

  void Boosting() override;

  void RefitTree(const int* /*tree_leaf_prediction*/, const size_t /*nrow*/, const size_t /*ncol*/) override {
    Log::Fatal("RefitTree is not supported for NCCLGBDT.");
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override;

  const double* GetTrainingScore(int64_t* /*out_len*/) override {
    Log::Fatal("GetTrainingScore is not supported for NCCLGBDT.");
  }

  void ResetTrainingData(const Dataset* /*train_data*/, const ObjectiveFunction* /*objective_function*/,
                         const std::vector<const Metric*>& /*training_metrics*/) override {
    Log::Fatal("ResetTrainingData is not supported for NCCLGBDT.");
  }

  void ResetConfig(const Config* /*gbdt_config*/) override {
    Log::Fatal("ResetConfig is not supported for NCCLGBDT.");
  }

 private:
  struct BoostingThreadData {
    int gpu_index;
    ObjectiveFunction* gpu_objective_function;
    score_t* gradients;
    score_t* hessians;
    const double* score;

    BoostingThreadData() {
      gpu_index = 0;
      gpu_objective_function = nullptr;
    }
  };

  struct TrainTreeLearnerThreadData {
    int gpu_index;
    TreeLearner* gpu_tree_learner;
    const score_t* gradients;
    const score_t* hessians;
    bool is_first_time;
    int class_id;
    data_size_t num_data_in_gpu;
    std::unique_ptr<Tree> tree;

    TrainTreeLearnerThreadData() {
      gpu_index = 0;
      gpu_tree_learner = nullptr;
      gradients = nullptr;
      hessians = nullptr;
      is_first_time = false;
      class_id = 0;
      num_data_in_gpu = 0;
      tree.reset(nullptr);
    }
  };

  struct UpdateScoreThreadData {
    int gpu_index;
    ScoreUpdater* gpu_score_updater;
    TreeLearner* gpu_tree_learner;
    Tree* tree;
    int cur_tree_id;

    UpdateScoreThreadData() {
      gpu_index = 0;
      gpu_score_updater = nullptr;
      gpu_tree_learner = nullptr;
      tree = nullptr;
      cur_tree_id = 0;
    }
  };

  static void BoostingThread(NCCLGBDTComponent* thread_data);

  static void TrainTreeLearnerThread(NCCLGBDTComponent* thread_data, const int class_id, const bool is_first_tree);

  static void UpdateScoreThread(NCCLGBDTComponent* thread_data, const int cur_tree_id, const double shrinkage_rate, const double init_score);

  double BoostFromAverage(int class_id, bool update_scorer) override;

  void UpdateScore(const std::vector<std::unique_ptr<Tree>>& tree, const int cur_tree_id);

  void UpdateScore(const Tree* /*tree*/, const int /*cur_tree_id*/) {
    Log::Fatal("UpdateScore is not supported for NCCLGBDT.");
  }

  void RollbackOneIter() override {
    Log::Fatal("RollbackOneIter is not supported for NCCLGBDT.");
  }

  std::vector<double> EvalOneMetric(const Metric* metric, const double* score, const data_size_t num_data) const override;


  int num_threads_;
  std::unique_ptr<NCCLTopology> nccl_topology_;

  std::vector<int> nccl_gpu_rank_;
  std::vector<ncclComm_t> nccl_communicators_;

  std::vector<std::unique_ptr<NCCLGBDTComponent>> nccl_gbdt_components_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_BOOSTING_CUDA_NCCL_GBDT_HPP_
