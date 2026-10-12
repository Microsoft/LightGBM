/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_SRC_BOOSTING_CUDA_CUDA_SCORE_UPDATER_HPP_
#define LIGHTGBM_SRC_BOOSTING_CUDA_CUDA_SCORE_UPDATER_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.hu>

#include "../score_updater.hpp"

namespace LightGBM {

class CUDAScoreUpdater: public ScoreUpdater {
 public:
  CUDAScoreUpdater(const Dataset* data, int num_tree_per_iteration, const bool boosting_on_cuda);

  ~CUDAScoreUpdater();

  void AddScore(double val, int cur_tree_id) override;

  inline void AddScore(const Tree* tree, int cur_tree_id) override;

  void AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) override;

  inline void AddScore(const Tree* tree, const data_size_t* data_indices,
                       data_size_t data_cnt, int cur_tree_id) override;

  inline void MultiplyScore(double val, int cur_tree_id) override;

  inline const double* score() const override {
    if (boosting_on_cuda_) {
      return cuda_score_.RawData();
    } else {
      return score_.data();
    }
  }

  /*! \brief Disable copy */
  CUDAScoreUpdater& operator=(const CUDAScoreUpdater&) = delete;

  CUDAScoreUpdater(const CUDAScoreUpdater&) = delete;

 private:
  void InitCUDA(const size_t total_size);

  void LaunchAddScoreConstantKernel(const double val, const size_t offset);

  void LaunchMultiplyScoreConstantKernel(const double val, const size_t offset);

  CUDAVector<double> cuda_score_;

  const int num_threads_per_block_;

  const bool boosting_on_cuda_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_SRC_BOOSTING_CUDA_CUDA_SCORE_UPDATER_HPP_
