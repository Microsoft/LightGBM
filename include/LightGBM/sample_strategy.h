/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_SAMPLE_STRATEGY_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_SAMPLE_STRATEGY_H_

#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/utils/random.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree_learner.h>
#include <LightGBM/objective_function.h>

#include <memory>
#include <vector>

namespace LightGBM {

class SampleStrategy {
 public:
  SampleStrategy() : balanced_bagging_(false), bagging_runner_(0, bagging_rand_block_), need_resize_gradients_(false) {}

  virtual ~SampleStrategy() {}

  static SampleStrategy* CreateSampleStrategy(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function, int num_tree_per_iteration);

  virtual void Bagging(int iter, TreeLearner* tree_learner, score_t* gradients, score_t* hessians) = 0;

  virtual void ResetSampleConfig(const Config* config, bool is_change_dataset) = 0;

  bool is_use_subset() const { return is_use_subset_; }

  data_size_t bag_data_cnt() const { return bag_data_cnt_; }

  std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>>& bag_data_indices() { return bag_data_indices_; }

  #ifdef USE_CUDA
  CUDAVector<data_size_t>& cuda_bag_data_indices() { return cuda_bag_data_indices_; }
  #endif  // USE_CUDA

  void UpdateObjectiveFunction(const ObjectiveFunction* objective_function) {
    objective_function_ = objective_function;
  }

  void UpdateTrainingData(const Dataset* train_data) {
    train_data_ = train_data;
    num_data_ = train_data->num_data();
  }

  virtual bool IsHessianChange() const = 0;

  bool NeedResizeGradients() const { return need_resize_gradients_; }

  virtual data_size_t num_sampled_queries() const { return 0; }

  virtual const data_size_t* sampled_query_indices() const { return nullptr; }

 protected:
  const Config* config_;
  const Dataset* train_data_;
  const ObjectiveFunction* objective_function_;
  std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>> bag_data_indices_;
  data_size_t bag_data_cnt_;
  data_size_t num_data_;
  int num_tree_per_iteration_;
  std::unique_ptr<Dataset> tmp_subset_;
  bool is_use_subset_;
  bool balanced_bagging_;
  const int bagging_rand_block_ = 1024;
  std::vector<Random> bagging_rands_;
  ParallelPartitionRunner<data_size_t, false> bagging_runner_;
  /*! \brief whether need to resize the gradient vectors */
  bool need_resize_gradients_;

  #ifdef USE_CUDA
  /*! \brief Buffer for bag_data_indices_ on GPU, used only with cuda */
  CUDAVector<data_size_t> cuda_bag_data_indices_;
  #endif  // USE_CUDA
};

}  // namespace LightGBM

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_SAMPLE_STRATEGY_H_
