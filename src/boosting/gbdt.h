/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_BOOSTING_GBDT_H_
#define LIGHTGBM_BOOSTING_GBDT_H_

#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/cuda/vector_cudahost.h>
#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/threading.h>

#include <string>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "score_updater.hpp"

namespace LightGBM {

using json11::Json;

/*!
* \brief GBDT algorithm implementation. including Training, prediction, bagging.
*/
class GBDT : public GBDTBase {
 public:
  /*!
  * \brief Constructor
  */
  GBDT();

  /*!
  * \brief Destructor
  */
  ~GBDT();


  /*!
  * \brief Initialization logic
  * \param gbdt_config Config for boosting
  * \param train_data Training data
  * \param objective_function Training objective function
  * \param training_metrics Training metrics
  */
  void Init(const Config* gbdt_config, const Dataset* train_data,
            const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override;

  /*!
  * \brief Merge model from other boosting object. Will insert to the front of current boosting object
  * \param other
  */
  void MergeFrom(const Boosting* other) override {
    auto other_gbdt = reinterpret_cast<const GBDT*>(other);
    // tmp move to other vector
    auto original_models = std::move(models_);
    models_ = std::vector<std::unique_ptr<Tree>>();
    // push model from other first
    for (const auto& tree : other_gbdt->models_) {
      auto new_tree = std::unique_ptr<Tree>(new Tree(*(tree.get())));
      models_.push_back(std::move(new_tree));
    }
    num_init_iteration_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
    // push model in current object
    for (const auto& tree : original_models) {
      auto new_tree = std::unique_ptr<Tree>(new Tree(*(tree.get())));
      models_.push_back(std::move(new_tree));
    }
    num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
  }

  void ShuffleModels(int start_iter, int end_iter) override {
    int total_iter = static_cast<int>(models_.size()) / num_tree_per_iteration_;
    start_iter = std::max(0, start_iter);
    if (end_iter <= 0) {
      end_iter = total_iter;
    }
    end_iter = std::min(total_iter, end_iter);
    auto original_models = std::move(models_);
    std::vector<int> indices(total_iter);
    for (int i = 0; i < total_iter; ++i) {
      indices[i] = i;
    }
    Random tmp_rand(17);
    for (int i = start_iter; i < end_iter - 1; ++i) {
      int j = tmp_rand.NextShort(i + 1, end_iter);
      std::swap(indices[i], indices[j]);
    }
    models_ = std::vector<std::unique_ptr<Tree>>();
    for (int i = 0; i < total_iter; ++i) {
      for (int j = 0; j < num_tree_per_iteration_; ++j) {
        int tree_idx = indices[i] * num_tree_per_iteration_ + j;
        auto new_tree = std::unique_ptr<Tree>(new Tree(*(original_models[tree_idx].get())));
        models_.push_back(std::move(new_tree));
      }
    }
  }

  /*!
  * \brief Reset the training data
  * \param train_data New Training data
  * \param objective_function Training objective function
  * \param training_metrics Training metrics
  */
  void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                         const std::vector<const Metric*>& training_metrics) override;

  /*!
  * \brief Reset Boosting Config
  * \param gbdt_config Config for boosting
  */
  void ResetConfig(const Config* gbdt_config) override;

  /*!
  * \brief Adding a validation dataset
  * \param valid_data Validation dataset
  * \param valid_metrics Metrics for validation dataset
  */
  void AddValidDataset(const Dataset* valid_data,
                       const std::vector<const Metric*>& valid_metrics) override;

  /*!
  * \brief Perform a full training procedure
  * \param snapshot_freq frequency of snapshot
  * \param model_output_path path of model file
  */
  void Train(int snapshot_freq, const std::string& model_output_path) override;

  void RefitTree(const std::vector<std::vector<int>>& tree_leaf_prediction) override;

  /*!
  * \brief Training logic
  * \param gradients nullptr for using default objective, otherwise use self-defined boosting
  * \param hessians nullptr for using default objective, otherwise use self-defined boosting
  * \return True if cannot train any more
  */
  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override;

  /*!
  * \brief Rollback one iteration
  */
  void RollbackOneIter() override;

  /*!
  * \brief Get current iteration
  */
  int GetCurrentIteration() const override { return static_cast<int>(models_.size()) / num_tree_per_iteration_; }

  /*!
  * \brief Can use early stopping for prediction or not
  * \return True if cannot use early stopping for prediction
  */
  bool NeedAccuratePrediction() const override {
    if (objective_function_ == nullptr) {
      return true;
    } else {
      return objective_function_->NeedAccuratePrediction();
    }
  }

  /*!
  * \brief Get evaluation result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \return evaluation result
  */
  std::vector<double> GetEvalAt(int data_idx) const override;

  /*!
  * \brief Get current training score
  * \param out_len length of returned score
  * \return training score
  */
  const double* GetTrainingScore(int64_t* out_len) override;

  /*!
  * \brief Get size of prediction at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \return The size of prediction
  */
  int64_t GetNumPredictAt(int data_idx) const override {
    CHECK(data_idx >= 0 && data_idx <= static_cast<int>(valid_score_updater_.size()));
    data_size_t num_data = train_data_->num_data();
    if (data_idx > 0) {
      num_data = valid_score_updater_[data_idx - 1]->num_data();
    }
    return num_data * num_class_;
  }

  /*!
  * \brief Get prediction result at data_idx data
  * \param data_idx 0: training data, 1: 1st validation data
  * \param result used to store prediction result, should allocate memory before call this function
  * \param out_len length of returned score
  */
  void GetPredictAt(int data_idx, double* out_result, int64_t* out_len) override;

  /*!
  * \brief Get number of prediction for one data
  * \param start_iteration Start index of the iteration to predict
  * \param num_iteration number of used iterations
  * \param is_pred_leaf True if predicting leaf index
  * \param is_pred_contrib True if predicting feature contribution
  * \return number of prediction
  */
  inline int NumPredictOneRow(int start_iteration, int num_iteration, bool is_pred_leaf, bool is_pred_contrib) const override {
    int num_pred_in_one_row = num_class_;
    if (is_pred_leaf) {
      int max_iteration = GetCurrentIteration();
      start_iteration = std::max(start_iteration, 0);
      start_iteration = std::min(start_iteration, max_iteration);
      if (num_iteration > 0) {
        num_pred_in_one_row *= static_cast<int>(std::min(max_iteration - start_iteration, num_iteration));
      } else {
        num_pred_in_one_row *= (max_iteration - start_iteration);
      }
    } else if (is_pred_contrib) {
      num_pred_in_one_row = num_tree_per_iteration_ * (max_feature_idx_ + 2);  // +1 for 0-based indexing, +1 for baseline
    }
    return num_pred_in_one_row;
  }

  void PredictRaw(const double* features, double* output,
                  const PredictionEarlyStopInstance* earlyStop) const override;

  void PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                       const PredictionEarlyStopInstance* early_stop) const override;

  void Predict(const double* features, double* output,
               const PredictionEarlyStopInstance* earlyStop) const override;

  void PredictByMap(const std::unordered_map<int, double>& features, double* output,
                    const PredictionEarlyStopInstance* early_stop) const override;

  void PredictLeafIndex(const double* features, double* output) const override;

  void PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const override;

  void PredictContrib(const double* features, double* output) const override;

  void PredictContribByMap(const std::unordered_map<int, double>& features,
                           std::vector<std::unordered_map<int, double>>* output) const override;

  /*!
  * \brief Dump model to json format string
  * \param start_iteration The model will be saved start from
  * \param num_iteration Number of iterations that want to dump, -1 means dump all
  * \param feature_importance_type Type of feature importance, 0: split, 1: gain
  * \return Json format string of model
  */
  std::string DumpModel(int start_iteration, int num_iteration,
                        int feature_importance_type) const override;

  /*!
  * \brief Translate model to if-else statement
  * \param num_iteration Number of iterations that want to translate, -1 means translate all
  * \return if-else format codes of model
  */
  std::string ModelToIfElse(int num_iteration) const override;

  /*!
  * \brief Translate model to if-else statement
  * \param num_iteration Number of iterations that want to translate, -1 means translate all
  * \param filename Filename that want to save to
  * \return is_finish Is training finished or not
  */
  bool SaveModelToIfElse(int num_iteration, const char* filename) const override;

  /*!
  * \brief Save model to file
  * \param start_iteration The model will be saved start from
  * \param num_iterations Number of model that want to save, -1 means save all
  * \param feature_importance_type Type of feature importance, 0: split, 1: gain
  * \param filename Filename that want to save to
  * \return is_finish Is training finished or not
  */
  bool SaveModelToFile(int start_iteration, int num_iterations,
                       int feature_importance_type,
                       const char* filename) const override;

  /*!
  * \brief Save model to string
  * \param start_iteration The model will be saved start from
  * \param num_iterations Number of model that want to save, -1 means save all
  * \param feature_importance_type Type of feature importance, 0: split, 1: gain
  * \return Non-empty string if succeeded
  */
  std::string SaveModelToString(int start_iteration, int num_iterations, int feature_importance_type) const override;

  /*!
  * \brief Restore from a serialized buffer
  */
  bool LoadModelFromString(const char* buffer, size_t len) override;

  /*!
  * \brief Calculate feature importances
  * \param num_iteration Number of model that want to use for feature importance, -1 means use all
  * \param importance_type: 0 for split, 1 for gain
  * \return vector of feature_importance
  */
  std::vector<double> FeatureImportance(int num_iteration, int importance_type) const override;

  /*!
  * \brief Calculate upper bound value
  * \return upper bound value
  */
  double GetUpperBoundValue() const override;

  /*!
  * \brief Calculate lower bound value
  * \return lower bound value
  */
  double GetLowerBoundValue() const override;

  /*!
  * \brief Get max feature index of this model
  * \return Max feature index of this model
  */
  inline int MaxFeatureIdx() const override { return max_feature_idx_; }

  /*!
  * \brief Get feature names of this model
  * \return Feature names of this model
  */
  inline std::vector<std::string> FeatureNames() const override { return feature_names_; }

  /*!
  * \brief Get index of label column
  * \return index of label column
  */
  inline int LabelIdx() const override { return label_idx_; }

  /*!
  * \brief Get number of weak sub-models
  * \return Number of weak sub-models
  */
  inline int NumberOfTotalModel() const override { return static_cast<int>(models_.size()); }

  /*!
  * \brief Get number of tree per iteration
  * \return number of tree per iteration
  */
  inline int NumModelPerIteration() const override { return num_tree_per_iteration_; }

  /*!
  * \brief Get number of classes
  * \return Number of classes
  */
  inline int NumberOfClasses() const override { return num_class_; }

  inline void InitPredict(int start_iteration, int num_iteration, bool is_pred_contrib) override {
    num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
    start_iteration = std::max(start_iteration, 0);
    start_iteration = std::min(start_iteration, num_iteration_for_pred_);
    if (num_iteration > 0) {
      num_iteration_for_pred_ = std::min(num_iteration, num_iteration_for_pred_ - start_iteration);
    } else {
      num_iteration_for_pred_ = num_iteration_for_pred_ - start_iteration;
    }
    start_iteration_for_pred_ = start_iteration;
    if (is_pred_contrib) {
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(models_.size()); ++i) {
        models_[i]->RecomputeMaxDepth();
      }
    }
  }

  inline double GetLeafValue(int tree_idx, int leaf_idx) const override {
    CHECK(tree_idx >= 0 && static_cast<size_t>(tree_idx) < models_.size());
    CHECK(leaf_idx >= 0 && leaf_idx < models_[tree_idx]->num_leaves());
    return models_[tree_idx]->LeafOutput(leaf_idx);
  }

  inline void SetLeafValue(int tree_idx, int leaf_idx, double val) override {
    CHECK(tree_idx >= 0 && static_cast<size_t>(tree_idx) < models_.size());
    CHECK(leaf_idx >= 0 && leaf_idx < models_[tree_idx]->num_leaves());
    models_[tree_idx]->SetLeafOutput(leaf_idx, val);
  }

  /*!
  * \brief Get Type name of this boosting object
  */
  const char* SubModelName() const override { return "tree"; }

  const CategoryEncodingProvider* category_encoding_provider() const override { return category_encoding_provider_.get(); }

  bool IsLinear() const override { return linear_tree_; }

 protected:
  virtual bool GetIsConstHessian(const ObjectiveFunction* objective_function) {
    if (objective_function != nullptr) {
      return objective_function->IsConstantHessian();
    } else {
      return false;
    }
  }
  /*!
  * \brief Print eval result and check early stopping
  */
  virtual bool EvalAndCheckEarlyStopping();

  /*!
  * \brief reset config for bagging
  */
  void ResetBaggingConfig(const Config* config, bool is_change_dataset);

  /*!
  * \brief Implement bagging logic
  * \param iter Current interation
  */
  virtual void Bagging(int iter);

  virtual data_size_t BaggingHelper(data_size_t start, data_size_t cnt,
                                    data_size_t* buffer);

  data_size_t BalancedBaggingHelper(data_size_t start, data_size_t cnt,
                                    data_size_t* buffer);

  /*!
  * \brief calculate the object function
  */
  virtual void Boosting();

  /*!
  * \brief updating score after tree was trained
  * \param tree Trained tree of this iteration
  * \param cur_tree_id Current tree for multiclass training
  */
  virtual void UpdateScore(const Tree* tree, const int cur_tree_id);

  /*!
  * \brief eval results for one metric

  */
  virtual std::vector<double> EvalOneMetric(const Metric* metric, const double* score) const;

  /*!
  * \brief Print metric result of current iteration
  * \param iter Current iteration
  * \return best_msg if met early_stopping
  */
  std::string OutputMetric(int iter);

  double BoostFromAverage(int class_id, bool update_scorer);

  /*! \brief current iteration */
  int iter_;
  /*! \brief Pointer to training data */
  const Dataset* train_data_;
  /*! \brief Config of gbdt */
  std::unique_ptr<Config> config_;
  /*! \brief Tree learner, will use this class to learn trees */
  std::unique_ptr<TreeLearner> tree_learner_;
  /*! \brief Objective function */
  const ObjectiveFunction* objective_function_;
  /*! \brief Store and update training data's score */
  std::unique_ptr<ScoreUpdater> train_score_updater_;
  /*! \brief Metrics for training data */
  std::vector<const Metric*> training_metrics_;
  /*! \brief Store and update validation data's scores */
  std::vector<std::unique_ptr<ScoreUpdater>> valid_score_updater_;
  /*! \brief Metric for validation data */
  std::vector<std::vector<const Metric*>> valid_metrics_;
  /*! \brief Number of rounds for early stopping */
  int early_stopping_round_;
  /*! \brief Only use first metric for early stopping */
  bool es_first_metric_only_;
  /*! \brief Best iteration(s) for early stopping */
  std::vector<std::vector<int>> best_iter_;
  /*! \brief Best score(s) for early stopping */
  std::vector<std::vector<double>> best_score_;
  /*! \brief output message of best iteration */
  std::vector<std::vector<std::string>> best_msg_;
  /*! \brief Trained models(trees) */
  std::vector<std::unique_ptr<Tree>> models_;
  /*! \brief Max feature index of training data*/
  int max_feature_idx_;

#ifdef USE_CUDA
  /*! \brief First order derivative of training data */
  std::vector<score_t, CHAllocator<score_t>> gradients_;
  /*! \brief Second order derivative of training data */
  std::vector<score_t, CHAllocator<score_t>> hessians_;
#else
  /*! \brief First order derivative of training data */
  std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>> gradients_;
  /*! \brief Second order derivative of training data */
  std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>> hessians_;
#endif

  /*! \brief Store the indices of in-bag data */
  std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>> bag_data_indices_;
  /*! \brief Number of in-bag data */
  data_size_t bag_data_cnt_;
  /*! \brief Number of training data */
  data_size_t num_data_;
  /*! \brief Number of trees per iterations */
  int num_tree_per_iteration_;
  /*! \brief Number of class */
  int num_class_;
  /*! \brief Index of label column */
  data_size_t label_idx_;
  /*! \brief number of used model */
  int num_iteration_for_pred_;
  /*! \brief Start iteration of used model */
  int start_iteration_for_pred_;
  /*! \brief Shrinkage rate for one iteration */
  double shrinkage_rate_;
  /*! \brief Number of loaded initial models */
  int num_init_iteration_;
  /*! \brief Feature names */
  std::vector<std::string> feature_names_;
  std::vector<std::string> feature_infos_;
  std::unique_ptr<Dataset> tmp_subset_;
  bool is_use_subset_;
  std::vector<bool> class_need_train_;
  bool is_constant_hessian_;
  std::unique_ptr<ObjectiveFunction> loaded_objective_;
  bool average_output_;
  bool need_re_bagging_;
  bool balanced_bagging_;
  std::string loaded_parameter_;
  std::vector<int8_t> monotone_constraints_;
  const int bagging_rand_block_ = 1024;
  std::vector<Random> bagging_rands_;
  ParallelPartitionRunner<data_size_t, false> bagging_runner_;
  Json forced_splits_json_;
  std::unique_ptr<const CategoryEncodingProvider> category_encoding_provider_;
  bool linear_tree_;
};

}  // namespace LightGBM
#endif   // LightGBM_BOOSTING_GBDT_H_
