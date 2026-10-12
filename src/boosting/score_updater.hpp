/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_BOOSTING_SCORE_UPDATER_HPP_
#define LIGHTGBM_SRC_BOOSTING_SCORE_UPDATER_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/tree.h>
#include <LightGBM/tree_learner.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <cstring>
#include <vector>

namespace LightGBM {
/*!
* \brief Used to store and update score for data
*/
class ScoreUpdater {
 public:
  /*!
  * \brief Constructor, will pass a const pointer of dataset
  * \param data This class will bind with this data set
  */
  ScoreUpdater(const Dataset* data, int num_tree_per_iteration) : data_(data) {
    num_data_ = data->num_data();
    int64_t total_size = static_cast<int64_t>(num_data_) * num_tree_per_iteration;
    score_.resize(total_size);
    // default start score is zero
    std::memset(score_.data(), 0, total_size * sizeof(double));
    has_init_score_ = false;
    const double* init_score = data->metadata().init_score();
    // if exists initial score, will start from it
    if (init_score != nullptr) {
      if ((data->metadata().num_init_score() % num_data_) != 0
          || (data->metadata().num_init_score() / num_data_) != num_tree_per_iteration) {
        Log::Fatal("Number of class for initial score error");
      }
      has_init_score_ = true;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (total_size >= 1024)
      for (int64_t i = 0; i < total_size; ++i) {
        score_[i] = init_score[i];
      }
    }
  }
  /*! \brief Destructor */
  virtual ~ScoreUpdater() {
  }

  inline bool has_init_score() const { return has_init_score_; }

  virtual inline void AddScore(double val, int cur_tree_id) {
    Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
    const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_data_ >= 1024)
    for (int i = 0; i < num_data_; ++i) {
      score_[offset + i] += val;
    }
  }

  virtual inline void MultiplyScore(double val, int cur_tree_id) {
    const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_data_ >= 1024)
    for (int i = 0; i < num_data_; ++i) {
      score_[offset + i] *= val;
    }
  }
  /*!
  * \brief Using tree model to get prediction number, then adding to scores for all data
  *        Note: this function generally will be used on validation data too.
  * \param tree Trained tree model
  * \param cur_tree_id Current tree for multiclass training
  */
  virtual inline void AddScore(const Tree* tree, int cur_tree_id) {
    Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
    const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
    tree->AddPredictionToScore(data_, num_data_, score_.data() + offset);
  }
  /*!
  * \brief Adding prediction score, only used for training data.
  *        The training data is partitioned into tree leaves after training
  *        Based on which We can get prediction quickly.
  * \param tree_learner
  * \param cur_tree_id Current tree for multiclass training
  */
  virtual inline void AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) {
    Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
    const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
    tree_learner->AddPredictionToScore(tree, score_.data() + offset);
  }
  /*!
  * \brief Using tree model to get prediction number, then adding to scores for parts of data
  *        Used for prediction of training out-of-bag data
  * \param tree Trained tree model
  * \param data_indices Indices of data that will be processed
  * \param data_cnt Number of data that will be processed
  * \param cur_tree_id Current tree for multiclass training
  */
  virtual inline void AddScore(const Tree* tree, const data_size_t* data_indices,
                       data_size_t data_cnt, int cur_tree_id) {
    Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
    const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
    tree->AddPredictionToScore(data_, data_indices, data_cnt, score_.data() + offset);
  }
  /*! \brief Pointer of score */
  virtual inline const double* score() const { return score_.data(); }

  inline data_size_t num_data() const { return num_data_; }

  /*! \brief Disable copy */
  ScoreUpdater& operator=(const ScoreUpdater&) = delete;
  /*! \brief Disable copy */
  ScoreUpdater(const ScoreUpdater&) = delete;

 protected:
  /*! \brief Number of total data */
  data_size_t num_data_;
  /*! \brief Pointer of data set */
  const Dataset* data_;
  /*! \brief Scores for data set */
  std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>> score_;
  bool has_init_score_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_BOOSTING_SCORE_UPDATER_HPP_
