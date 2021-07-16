/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/c_api.h>

#include <LightGBM/boosting.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/metric.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/utils/threading.h>

#include <string>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "application/predictor.hpp"
#include <LightGBM/utils/yamc/alternate_shared_mutex.hpp>
#include <LightGBM/utils/yamc/yamc_shared_lock.hpp>

namespace LightGBM {

inline int LGBM_APIHandleException(const std::exception& ex) {
  LGBM_SetLastError(ex.what());
  return -1;
}
inline int LGBM_APIHandleException(const std::string& ex) {
  LGBM_SetLastError(ex.c_str());
  return -1;
}

#define API_BEGIN() try {
#define API_END() } \
catch(std::exception& ex) { return LGBM_APIHandleException(ex); } \
catch(std::string& ex) { return LGBM_APIHandleException(ex); } \
catch(...) { return LGBM_APIHandleException("unknown exception"); } \
return 0;

#define UNIQUE_LOCK(mtx) \
std::unique_lock<yamc::alternate::shared_mutex> lock(mtx);

#define SHARED_LOCK(mtx) \
yamc::shared_lock<yamc::alternate::shared_mutex> lock(&mtx);

const int PREDICTOR_TYPES = 4;

// Single row predictor to abstract away caching logic
class SingleRowPredictor {
 public:
  PredictFunction predict_function;
  int64_t num_pred_in_one_row;

  SingleRowPredictor(int predict_type, Boosting* boosting, const Config& config, int start_iter, int num_iter) {
    bool is_predict_leaf = false;
    bool is_raw_score = false;
    bool predict_contrib = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_RAW_SCORE) {
      is_raw_score = true;
    } else if (predict_type == C_API_PREDICT_CONTRIB) {
      predict_contrib = true;
    }
    early_stop_ = config.pred_early_stop;
    early_stop_freq_ = config.pred_early_stop_freq;
    early_stop_margin_ = config.pred_early_stop_margin;
    iter_ = num_iter;
    predictor_.reset(new Predictor(boosting, start_iter, iter_, is_raw_score, is_predict_leaf, predict_contrib,
                                   early_stop_, early_stop_freq_, early_stop_margin_));
    num_pred_in_one_row = boosting->NumPredictOneRow(start_iter, iter_, is_predict_leaf, predict_contrib);
    predict_function = predictor_->GetPredictFunction();
    num_total_model_ = boosting->NumberOfTotalModel();
  }

  ~SingleRowPredictor() {}

  bool IsPredictorEqual(const Config& config, int iter, Boosting* boosting) {
    return early_stop_ == config.pred_early_stop &&
      early_stop_freq_ == config.pred_early_stop_freq &&
      early_stop_margin_ == config.pred_early_stop_margin &&
      iter_ == iter &&
      num_total_model_ == boosting->NumberOfTotalModel();
  }

 private:
  std::unique_ptr<Predictor> predictor_;
  bool early_stop_;
  int early_stop_freq_;
  double early_stop_margin_;
  int iter_;
  int num_total_model_;
};

class Booster {
 public:
  explicit Booster(const char* filename) {
    boosting_.reset(Boosting::CreateBoosting("gbdt", filename));
  }

  Booster(const Dataset* train_data,
          const char* parameters) {
    auto param = Config::Str2Map(parameters);
    config_.Set(param);
    if (config_.num_threads > 0) {
      omp_set_num_threads(config_.num_threads);
    }
    // create boosting
    if (config_.input_model.size() > 0) {
      Log::Warning("Continued train from model is not supported for c_api,\n"
                   "please use continued train with input score");
    }

    boosting_.reset(Boosting::CreateBoosting(config_.boosting, nullptr));

    train_data_ = train_data;
    CreateObjectiveAndMetrics();
    // initialize the boosting
    if (config_.tree_learner == std::string("feature")) {
      Log::Fatal("Do not support feature parallel in c api");
    }
    if (Network::num_machines() == 1 && config_.tree_learner != std::string("serial")) {
      Log::Warning("Only find one worker, will switch to serial tree learner");
      config_.tree_learner = "serial";
    }
    boosting_->Init(&config_, train_data_, objective_fun_.get(),
                    Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
  }

  void MergeFrom(const Booster* other) {
    UNIQUE_LOCK(mutex_)
    boosting_->MergeFrom(other->boosting_.get());
  }

  ~Booster() {
  }

  void CreateObjectiveAndMetrics() {
    // create objective function
    objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective,
                                                                    config_));
    if (objective_fun_ == nullptr) {
      Log::Warning("Using self-defined objective function");
    }
    // initialize the objective function
    if (objective_fun_ != nullptr) {
      objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
    }

    // create training metric
    train_metric_.clear();
    for (auto metric_type : config_.metric) {
      auto metric = std::unique_ptr<Metric>(
        Metric::CreateMetric(metric_type, config_));
      if (metric == nullptr) { continue; }
      metric->Init(train_data_->metadata(), train_data_->num_data());
      train_metric_.push_back(std::move(metric));
    }
    train_metric_.shrink_to_fit();
  }

  void ResetTrainingData(const Dataset* train_data) {
    if (train_data != train_data_) {
      UNIQUE_LOCK(mutex_)
      train_data_ = train_data;
      CreateObjectiveAndMetrics();
      // reset the boosting
      boosting_->ResetTrainingData(train_data_,
                                   objective_fun_.get(), Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
    }
  }

  static void CheckDatasetResetConfig(
      const Config& old_config,
      const std::unordered_map<std::string, std::string>& new_param) {
    Config new_config;
    new_config.Set(new_param);
    if (new_param.count("data_random_seed") &&
        new_config.data_random_seed != old_config.data_random_seed) {
      Log::Fatal("Cannot change data_random_seed after constructed Dataset handle.");
    }
    if (new_param.count("max_bin") &&
        new_config.max_bin != old_config.max_bin) {
      Log::Fatal("Cannot change max_bin after constructed Dataset handle.");
    }
    if (new_param.count("max_bin_by_feature") &&
        new_config.max_bin_by_feature != old_config.max_bin_by_feature) {
      Log::Fatal(
          "Cannot change max_bin_by_feature after constructed Dataset handle.");
    }
    if (new_param.count("bin_construct_sample_cnt") &&
        new_config.bin_construct_sample_cnt !=
            old_config.bin_construct_sample_cnt) {
      Log::Fatal(
          "Cannot change bin_construct_sample_cnt after constructed Dataset "
          "handle.");
    }
    if (new_param.count("min_data_in_bin") &&
        new_config.min_data_in_bin != old_config.min_data_in_bin) {
      Log::Fatal(
          "Cannot change min_data_in_bin after constructed Dataset handle.");
    }
    if (new_param.count("use_missing") &&
        new_config.use_missing != old_config.use_missing) {
      Log::Fatal("Cannot change use_missing after constructed Dataset handle.");
    }
    if (new_param.count("zero_as_missing") &&
        new_config.zero_as_missing != old_config.zero_as_missing) {
      Log::Fatal(
          "Cannot change zero_as_missing after constructed Dataset handle.");
    }
    if (new_param.count("categorical_feature") &&
        new_config.categorical_feature != old_config.categorical_feature) {
      Log::Fatal(
          "Cannot change categorical_feature after constructed Dataset "
          "handle.");
    }
    if (new_param.count("feature_pre_filter") &&
        new_config.feature_pre_filter != old_config.feature_pre_filter) {
      Log::Fatal(
          "Cannot change feature_pre_filter after constructed Dataset handle.");
    }
    if (new_param.count("is_enable_sparse") &&
        new_config.is_enable_sparse != old_config.is_enable_sparse) {
      Log::Fatal(
          "Cannot change is_enable_sparse after constructed Dataset handle.");
    }
    if (new_param.count("pre_partition") &&
        new_config.pre_partition != old_config.pre_partition) {
      Log::Fatal(
          "Cannot change pre_partition after constructed Dataset handle.");
    }
    if (new_param.count("enable_bundle") &&
        new_config.enable_bundle != old_config.enable_bundle) {
      Log::Fatal(
          "Cannot change enable_bundle after constructed Dataset handle.");
    }
    if (new_param.count("header") && new_config.header != old_config.header) {
      Log::Fatal("Cannot change header after constructed Dataset handle.");
    }
    if (new_param.count("two_round") &&
        new_config.two_round != old_config.two_round) {
      Log::Fatal("Cannot change two_round after constructed Dataset handle.");
    }
    if (new_param.count("label_column") &&
        new_config.label_column != old_config.label_column) {
      Log::Fatal(
          "Cannot change label_column after constructed Dataset handle.");
    }
    if (new_param.count("weight_column") &&
        new_config.weight_column != old_config.weight_column) {
      Log::Fatal(
          "Cannot change weight_column after constructed Dataset handle.");
    }
    if (new_param.count("group_column") &&
        new_config.group_column != old_config.group_column) {
      Log::Fatal(
          "Cannot change group_column after constructed Dataset handle.");
    }
    if (new_param.count("ignore_column") &&
        new_config.ignore_column != old_config.ignore_column) {
      Log::Fatal(
          "Cannot change ignore_column after constructed Dataset handle.");
    }
    if (new_param.count("forcedbins_filename")) {
      Log::Fatal("Cannot change forced bins after constructed Dataset handle.");
    }
    if (new_param.count("min_data_in_leaf") &&
        new_config.min_data_in_leaf < old_config.min_data_in_leaf &&
        old_config.feature_pre_filter) {
      Log::Fatal(
          "Reducing `min_data_in_leaf` with `feature_pre_filter=true` may "
          "cause unexpected behaviour "
          "for features that were pre-filtered by the larger "
          "`min_data_in_leaf`.\n"
          "You need to set `feature_pre_filter=false` to dynamically change "
          "the `min_data_in_leaf`.");
    }
    if (new_param.count("linear_tree") && new_config.linear_tree != old_config.linear_tree) {
      Log::Fatal("Cannot change linear_tree after constructed Dataset handle.");
    }
    if (new_param.count("precise_float_parser") &&
        new_config.precise_float_parser != old_config.precise_float_parser) {
      Log::Fatal("Cannot change precise_float_parser after constructed Dataset handle.");
    }
  }

  void ResetConfig(const char* parameters) {
    UNIQUE_LOCK(mutex_)
    auto param = Config::Str2Map(parameters);
    Config new_config;
    new_config.Set(param);
    if (param.count("num_class") && new_config.num_class != config_.num_class) {
      Log::Fatal("Cannot change num_class during training");
    }
    if (param.count("boosting") && new_config.boosting != config_.boosting) {
      Log::Fatal("Cannot change boosting during training");
    }
    if (param.count("metric") && new_config.metric != config_.metric) {
      Log::Fatal("Cannot change metric during training");
    }
    CheckDatasetResetConfig(config_, param);

    config_.Set(param);

    if (config_.num_threads > 0) {
      omp_set_num_threads(config_.num_threads);
    }

    if (param.count("objective")) {
      // create objective function
      objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective,
                                                                      config_));
      if (objective_fun_ == nullptr) {
        Log::Warning("Using self-defined objective function");
      }
      // initialize the objective function
      if (objective_fun_ != nullptr) {
        objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
      }
      boosting_->ResetTrainingData(train_data_,
                                   objective_fun_.get(), Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
    }

    boosting_->ResetConfig(&config_);
  }

  void AddValidData(const Dataset* valid_data) {
    UNIQUE_LOCK(mutex_)
    valid_metrics_.emplace_back();
    for (auto metric_type : config_.metric) {
      auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_));
      if (metric == nullptr) { continue; }
      metric->Init(valid_data->metadata(), valid_data->num_data());
      valid_metrics_.back().push_back(std::move(metric));
    }
    valid_metrics_.back().shrink_to_fit();
    boosting_->AddValidDataset(valid_data,
                               Common::ConstPtrInVectorWrapper<Metric>(valid_metrics_.back()));
  }

  bool TrainOneIter() {
    UNIQUE_LOCK(mutex_)
    return boosting_->TrainOneIter(nullptr, nullptr);
  }

  void Refit(const int32_t* leaf_preds, int32_t nrow, int32_t ncol) {
    UNIQUE_LOCK(mutex_)
    std::vector<std::vector<int32_t>> v_leaf_preds(nrow, std::vector<int32_t>(ncol, 0));
    for (int i = 0; i < nrow; ++i) {
      for (int j = 0; j < ncol; ++j) {
        v_leaf_preds[i][j] = leaf_preds[static_cast<size_t>(i) * static_cast<size_t>(ncol) + static_cast<size_t>(j)];
      }
    }
    boosting_->RefitTree(v_leaf_preds);
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) {
    UNIQUE_LOCK(mutex_)
    return boosting_->TrainOneIter(gradients, hessians);
  }

  void RollbackOneIter() {
    UNIQUE_LOCK(mutex_)
    boosting_->RollbackOneIter();
  }

  void SetSingleRowPredictor(int start_iteration, int num_iteration, int predict_type, const Config& config) {
      UNIQUE_LOCK(mutex_)
      if (single_row_predictor_[predict_type].get() == nullptr ||
          !single_row_predictor_[predict_type]->IsPredictorEqual(config, num_iteration, boosting_.get())) {
        single_row_predictor_[predict_type].reset(new SingleRowPredictor(predict_type, boosting_.get(),
                                                                         config, start_iteration, num_iteration));
      }
  }

  void PredictSingleRow(int predict_type, int ncol,
               std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
               const Config& config,
               double* out_result, int64_t* out_len) const {
    if (!config.predict_disable_shape_check && ncol != boosting_->MaxFeatureIdx() + 1) {
      Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n"\
                 "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.", ncol, boosting_->MaxFeatureIdx() + 1);
    }
    UNIQUE_LOCK(mutex_)
    const auto& single_row_predictor = single_row_predictor_[predict_type];
    auto one_row = get_row_fun(0);
    auto pred_wrt_ptr = out_result;
    single_row_predictor->predict_function(one_row, pred_wrt_ptr);

    *out_len = single_row_predictor->num_pred_in_one_row;
  }

  Predictor CreatePredictor(int start_iteration, int num_iteration, int predict_type, int ncol, const Config& config) const {
    if (!config.predict_disable_shape_check && ncol != boosting_->MaxFeatureIdx() + 1) {
      Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n" \
                 "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.", ncol, boosting_->MaxFeatureIdx() + 1);
    }
    bool is_predict_leaf = false;
    bool is_raw_score = false;
    bool predict_contrib = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_RAW_SCORE) {
      is_raw_score = true;
    } else if (predict_type == C_API_PREDICT_CONTRIB) {
      predict_contrib = true;
    } else {
      is_raw_score = false;
    }

    return Predictor(boosting_.get(), start_iteration, num_iteration, is_raw_score, is_predict_leaf, predict_contrib,
                        config.pred_early_stop, config.pred_early_stop_freq, config.pred_early_stop_margin);
  }

  void Predict(int start_iteration, int num_iteration, int predict_type, int nrow, int ncol,
               std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
               const Config& config,
               double* out_result, int64_t* out_len) const {
    SHARED_LOCK(mutex_);
    auto predictor = CreatePredictor(start_iteration, num_iteration, predict_type, ncol, config);
    bool is_predict_leaf = false;
    bool predict_contrib = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_CONTRIB) {
      predict_contrib = true;
    }
    int64_t num_pred_in_one_row = boosting_->NumPredictOneRow(start_iteration, num_iteration, is_predict_leaf, predict_contrib);
    auto pred_fun = predictor.GetPredictFunction();
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrow; ++i) {
      OMP_LOOP_EX_BEGIN();
      auto one_row = get_row_fun(i);
      auto pred_wrt_ptr = out_result + static_cast<size_t>(num_pred_in_one_row) * i;
      pred_fun(one_row, pred_wrt_ptr);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    *out_len = num_pred_in_one_row * nrow;
  }

  void PredictSparse(int start_iteration, int num_iteration, int predict_type, int64_t nrow, int ncol,
                     std::function<std::vector<std::pair<int, double>>(int64_t row_idx)> get_row_fun,
                     const Config& config, int64_t* out_elements_size,
                     std::vector<std::vector<std::unordered_map<int, double>>>* agg_ptr,
                     int32_t** out_indices, void** out_data, int data_type,
                     bool* is_data_float32_ptr, int num_matrices) const {
    auto predictor = CreatePredictor(start_iteration, num_iteration, predict_type, ncol, config);
    auto pred_sparse_fun = predictor.GetPredictSparseFunction();
    std::vector<std::vector<std::unordered_map<int, double>>>& agg = *agg_ptr;
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < nrow; ++i) {
      OMP_LOOP_EX_BEGIN();
      auto one_row = get_row_fun(i);
      agg[i] = std::vector<std::unordered_map<int, double>>(num_matrices);
      pred_sparse_fun(one_row, &agg[i]);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    // calculate the nonzero data and indices size
    int64_t elements_size = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(agg.size()); ++i) {
      auto row_vector = agg[i];
      for (int j = 0; j < static_cast<int>(row_vector.size()); ++j) {
        elements_size += static_cast<int64_t>(row_vector[j].size());
      }
    }
    *out_elements_size = elements_size;
    *is_data_float32_ptr = false;
    // allocate data and indices arrays
    if (data_type == C_API_DTYPE_FLOAT32) {
      *out_data = new float[elements_size];
      *is_data_float32_ptr = true;
    } else if (data_type == C_API_DTYPE_FLOAT64) {
      *out_data = new double[elements_size];
    } else {
      Log::Fatal("Unknown data type in PredictSparse");
      return;
    }
    *out_indices = new int32_t[elements_size];
  }

  void PredictSparseCSR(int start_iteration, int num_iteration, int predict_type, int64_t nrow, int ncol,
                        std::function<std::vector<std::pair<int, double>>(int64_t row_idx)> get_row_fun,
                        const Config& config,
                        int64_t* out_len, void** out_indptr, int indptr_type,
                        int32_t** out_indices, void** out_data, int data_type) const {
    SHARED_LOCK(mutex_);
    // Get the number of trees per iteration (for multiclass scenario we output multiple sparse matrices)
    int num_matrices = boosting_->NumModelPerIteration();
    bool is_indptr_int32 = false;
    bool is_data_float32 = false;
    int64_t indptr_size = (nrow + 1) * num_matrices;
    if (indptr_type == C_API_DTYPE_INT32) {
      *out_indptr = new int32_t[indptr_size];
      is_indptr_int32 = true;
    } else if (indptr_type == C_API_DTYPE_INT64) {
      *out_indptr = new int64_t[indptr_size];
    } else {
      Log::Fatal("Unknown indptr type in PredictSparseCSR");
      return;
    }
    // aggregated per row feature contribution results
    std::vector<std::vector<std::unordered_map<int, double>>> agg(nrow);
    int64_t elements_size = 0;
    PredictSparse(start_iteration, num_iteration, predict_type, nrow, ncol, get_row_fun, config, &elements_size, &agg,
                  out_indices, out_data, data_type, &is_data_float32, num_matrices);
    std::vector<int> row_sizes(num_matrices * nrow);
    std::vector<int64_t> row_matrix_offsets(num_matrices * nrow);
    std::vector<int64_t> matrix_offsets(num_matrices);
    int64_t row_vector_cnt = 0;
    for (int m = 0; m < num_matrices; ++m) {
      for (int64_t i = 0; i < static_cast<int64_t>(agg.size()); ++i) {
        auto row_vector = agg[i];
        auto row_vector_size = row_vector[m].size();
        // keep track of the row_vector sizes for parallelization
        row_sizes[row_vector_cnt] = static_cast<int>(row_vector_size);
        if (i == 0) {
          row_matrix_offsets[row_vector_cnt] = 0;
        } else {
          row_matrix_offsets[row_vector_cnt] = static_cast<int64_t>(row_sizes[row_vector_cnt - 1] + row_matrix_offsets[row_vector_cnt - 1]);
        }
        row_vector_cnt++;
      }
      if (m == 0) {
        matrix_offsets[m] = 0;
      }
      if (m + 1 < num_matrices) {
        matrix_offsets[m + 1] = static_cast<int64_t>(matrix_offsets[m] + row_matrix_offsets[row_vector_cnt - 1] + row_sizes[row_vector_cnt - 1]);
      }
    }
    // copy vector results to output for each row
    int64_t indptr_index = 0;
    for (int m = 0; m < num_matrices; ++m) {
      if (is_indptr_int32) {
        (reinterpret_cast<int32_t*>(*out_indptr))[indptr_index] = 0;
      } else {
        (reinterpret_cast<int64_t*>(*out_indptr))[indptr_index] = 0;
      }
      indptr_index++;
      int64_t matrix_start_index = m * static_cast<int64_t>(agg.size());
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < static_cast<int64_t>(agg.size()); ++i) {
        OMP_LOOP_EX_BEGIN();
        auto row_vector = agg[i];
        int64_t row_start_index = matrix_start_index + i;
        int64_t element_index = row_matrix_offsets[row_start_index] + matrix_offsets[m];
        int64_t indptr_loop_index = indptr_index + i;
        for (auto it = row_vector[m].begin(); it != row_vector[m].end(); ++it) {
          (*out_indices)[element_index] = it->first;
          if (is_data_float32) {
            (reinterpret_cast<float*>(*out_data))[element_index] = static_cast<float>(it->second);
          } else {
            (reinterpret_cast<double*>(*out_data))[element_index] = it->second;
          }
          element_index++;
        }
        int64_t indptr_value = row_matrix_offsets[row_start_index] + row_sizes[row_start_index];
        if (is_indptr_int32) {
          (reinterpret_cast<int32_t*>(*out_indptr))[indptr_loop_index] = static_cast<int32_t>(indptr_value);
        } else {
          (reinterpret_cast<int64_t*>(*out_indptr))[indptr_loop_index] = indptr_value;
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      indptr_index += static_cast<int64_t>(agg.size());
    }
    out_len[0] = elements_size;
    out_len[1] = indptr_size;
  }

  void PredictSparseCSC(int start_iteration, int num_iteration, int predict_type, int64_t nrow, int ncol,
                        std::function<std::vector<std::pair<int, double>>(int64_t row_idx)> get_row_fun,
                        const Config& config,
                        int64_t* out_len, void** out_col_ptr, int col_ptr_type,
                        int32_t** out_indices, void** out_data, int data_type) const {
    SHARED_LOCK(mutex_);
    // Get the number of trees per iteration (for multiclass scenario we output multiple sparse matrices)
    int num_matrices = boosting_->NumModelPerIteration();
    auto predictor = CreatePredictor(start_iteration, num_iteration, predict_type, ncol, config);
    auto pred_sparse_fun = predictor.GetPredictSparseFunction();
    bool is_col_ptr_int32 = false;
    bool is_data_float32 = false;
    int num_output_cols = ncol + 1;
    int col_ptr_size = (num_output_cols + 1) * num_matrices;
    if (col_ptr_type == C_API_DTYPE_INT32) {
      *out_col_ptr = new int32_t[col_ptr_size];
      is_col_ptr_int32 = true;
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      *out_col_ptr = new int64_t[col_ptr_size];
    } else {
      Log::Fatal("Unknown col_ptr type in PredictSparseCSC");
      return;
    }
    // aggregated per row feature contribution results
    std::vector<std::vector<std::unordered_map<int, double>>> agg(nrow);
    int64_t elements_size = 0;
    PredictSparse(start_iteration, num_iteration, predict_type, nrow, ncol, get_row_fun, config, &elements_size, &agg,
                  out_indices, out_data, data_type, &is_data_float32, num_matrices);
    // calculate number of elements per column to construct
    // the CSC matrix with random access
    std::vector<std::vector<int64_t>> column_sizes(num_matrices);
    for (int m = 0; m < num_matrices; ++m) {
      column_sizes[m] = std::vector<int64_t>(num_output_cols, 0);
      for (int64_t i = 0; i < static_cast<int64_t>(agg.size()); ++i) {
        auto row_vector = agg[i];
        for (auto it = row_vector[m].begin(); it != row_vector[m].end(); ++it) {
          column_sizes[m][it->first] += 1;
        }
      }
    }
    // keep track of column counts
    std::vector<std::vector<int64_t>> column_counts(num_matrices);
    // keep track of beginning index for each column
    std::vector<std::vector<int64_t>> column_start_indices(num_matrices);
    // keep track of beginning index for each matrix
    std::vector<int64_t> matrix_start_indices(num_matrices, 0);
    int col_ptr_index = 0;
    for (int m = 0; m < num_matrices; ++m) {
      int64_t col_ptr_value = 0;
      column_start_indices[m] = std::vector<int64_t>(num_output_cols, 0);
      column_counts[m] = std::vector<int64_t>(num_output_cols, 0);
      if (is_col_ptr_int32) {
        (reinterpret_cast<int32_t*>(*out_col_ptr))[col_ptr_index] = static_cast<int32_t>(col_ptr_value);
      } else {
        (reinterpret_cast<int64_t*>(*out_col_ptr))[col_ptr_index] = col_ptr_value;
      }
      col_ptr_index++;
      for (int64_t i = 1; i < static_cast<int64_t>(column_sizes[m].size()); ++i) {
        column_start_indices[m][i] = column_sizes[m][i - 1] + column_start_indices[m][i - 1];
        if (is_col_ptr_int32) {
          (reinterpret_cast<int32_t*>(*out_col_ptr))[col_ptr_index] = static_cast<int32_t>(column_start_indices[m][i]);
        } else {
          (reinterpret_cast<int64_t*>(*out_col_ptr))[col_ptr_index] = column_start_indices[m][i];
        }
        col_ptr_index++;
      }
      int64_t last_elem_index = static_cast<int64_t>(column_sizes[m].size()) - 1;
      int64_t last_column_start_index = column_start_indices[m][last_elem_index];
      int64_t last_column_size = column_sizes[m][last_elem_index];
      if (is_col_ptr_int32) {
        (reinterpret_cast<int32_t*>(*out_col_ptr))[col_ptr_index] = static_cast<int32_t>(last_column_start_index + last_column_size);
      } else {
        (reinterpret_cast<int64_t*>(*out_col_ptr))[col_ptr_index] = last_column_start_index + last_column_size;
      }
      if (m + 1 < num_matrices) {
        matrix_start_indices[m + 1] = matrix_start_indices[m] + last_column_start_index + last_column_size;
      }
      col_ptr_index++;
    }
    // Note: we parallelize across matrices instead of rows because of the column_counts[m][col_idx] increment inside the loop
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < num_matrices; ++m) {
      OMP_LOOP_EX_BEGIN();
      for (int64_t i = 0; i < static_cast<int64_t>(agg.size()); ++i) {
        auto row_vector = agg[i];
        for (auto it = row_vector[m].begin(); it != row_vector[m].end(); ++it) {
          int64_t col_idx = it->first;
          int64_t element_index = column_start_indices[m][col_idx] +
            matrix_start_indices[m] +
            column_counts[m][col_idx];
          // store the row index
          (*out_indices)[element_index] = static_cast<int32_t>(i);
          // update column count
          column_counts[m][col_idx]++;
          if (is_data_float32) {
            (reinterpret_cast<float*>(*out_data))[element_index] = static_cast<float>(it->second);
          } else {
            (reinterpret_cast<double*>(*out_data))[element_index] = it->second;
          }
        }
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    out_len[0] = elements_size;
    out_len[1] = col_ptr_size;
  }

  void Predict(int start_iteration, int num_iteration, int predict_type, const char* data_filename,
               int data_has_header, const Config& config,
               const char* result_filename) const {
    SHARED_LOCK(mutex_)
    bool is_predict_leaf = false;
    bool is_raw_score = false;
    bool predict_contrib = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_RAW_SCORE) {
      is_raw_score = true;
    } else if (predict_type == C_API_PREDICT_CONTRIB) {
      predict_contrib = true;
    } else {
      is_raw_score = false;
    }
    Predictor predictor(boosting_.get(), start_iteration, num_iteration, is_raw_score, is_predict_leaf, predict_contrib,
                        config.pred_early_stop, config.pred_early_stop_freq, config.pred_early_stop_margin);
    bool bool_data_has_header = data_has_header > 0 ? true : false;
    predictor.Predict(data_filename, result_filename, bool_data_has_header, config.predict_disable_shape_check,
                      config.precise_float_parser);
  }

  void GetPredictAt(int data_idx, double* out_result, int64_t* out_len) const {
    boosting_->GetPredictAt(data_idx, out_result, out_len);
  }

  void SaveModelToFile(int start_iteration, int num_iteration, int feature_importance_type, const char* filename) const {
    boosting_->SaveModelToFile(start_iteration, num_iteration, feature_importance_type, filename);
  }

  void LoadModelFromString(const char* model_str) {
    size_t len = std::strlen(model_str);
    boosting_->LoadModelFromString(model_str, len);
  }

  std::string SaveModelToString(int start_iteration, int num_iteration,
                                int feature_importance_type) const {
    return boosting_->SaveModelToString(start_iteration,
                                        num_iteration, feature_importance_type);
  }

  std::string DumpModel(int start_iteration, int num_iteration,
                        int feature_importance_type) const {
    return boosting_->DumpModel(start_iteration, num_iteration,
                                feature_importance_type);
  }

  std::vector<double> FeatureImportance(int num_iteration, int importance_type) const {
    return boosting_->FeatureImportance(num_iteration, importance_type);
  }

  double UpperBoundValue() const {
    SHARED_LOCK(mutex_)
    return boosting_->GetUpperBoundValue();
  }

  double LowerBoundValue() const {
    SHARED_LOCK(mutex_)
    return boosting_->GetLowerBoundValue();
  }

  double GetLeafValue(int tree_idx, int leaf_idx) const {
    SHARED_LOCK(mutex_)
    return dynamic_cast<GBDTBase*>(boosting_.get())->GetLeafValue(tree_idx, leaf_idx);
  }

  void SetLeafValue(int tree_idx, int leaf_idx, double val) {
    UNIQUE_LOCK(mutex_)
    dynamic_cast<GBDTBase*>(boosting_.get())->SetLeafValue(tree_idx, leaf_idx, val);
  }

  void ShuffleModels(int start_iter, int end_iter) {
    UNIQUE_LOCK(mutex_)
    boosting_->ShuffleModels(start_iter, end_iter);
  }

  int GetEvalCounts() const {
    SHARED_LOCK(mutex_)
    int ret = 0;
    for (const auto& metric : train_metric_) {
      ret += static_cast<int>(metric->GetName().size());
    }
    return ret;
  }

  int GetEvalNames(char** out_strs, const int len, const size_t buffer_len, size_t *out_buffer_len) const {
    SHARED_LOCK(mutex_)
    *out_buffer_len = 0;
    int idx = 0;
    for (const auto& metric : train_metric_) {
      for (const auto& name : metric->GetName()) {
        if (idx < len) {
          std::memcpy(out_strs[idx], name.c_str(), std::min(name.size() + 1, buffer_len));
          out_strs[idx][buffer_len - 1] = '\0';
        }
        *out_buffer_len = std::max(name.size() + 1, *out_buffer_len);
        ++idx;
      }
    }
    return idx;
  }

  int GetFeatureNames(char** out_strs, const int len, const size_t buffer_len, size_t *out_buffer_len) const {
    SHARED_LOCK(mutex_)
    *out_buffer_len = 0;
    int idx = 0;
    for (const auto& name : boosting_->FeatureNames()) {
      if (idx < len) {
        std::memcpy(out_strs[idx], name.c_str(), std::min(name.size() + 1, buffer_len));
        out_strs[idx][buffer_len - 1] = '\0';
      }
      *out_buffer_len = std::max(name.size() + 1, *out_buffer_len);
      ++idx;
    }
    return idx;
  }

  const Boosting* GetBoosting() const { return boosting_.get(); }

 private:
  const Dataset* train_data_;
  std::unique_ptr<Boosting> boosting_;
  std::unique_ptr<SingleRowPredictor> single_row_predictor_[PREDICTOR_TYPES];

  /*! \brief All configs */
  Config config_;
  /*! \brief Metric for training data */
  std::vector<std::unique_ptr<Metric>> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_;
  /*! \brief Training objective function */
  std::unique_ptr<ObjectiveFunction> objective_fun_;
  /*! \brief mutex for threading safe call */
  mutable yamc::alternate::shared_mutex mutex_;
};

}  // namespace LightGBM

// explicitly declare symbols from LightGBM namespace
using LightGBM::AllgatherFunction;
using LightGBM::Booster;
using LightGBM::Common::CheckElementsIntervalClosed;
using LightGBM::Common::RemoveQuotationSymbol;
using LightGBM::Common::Vector2Ptr;
using LightGBM::Common::VectorSize;
using LightGBM::Config;
using LightGBM::data_size_t;
using LightGBM::Dataset;
using LightGBM::DatasetLoader;
using LightGBM::kZeroThreshold;
using LightGBM::LGBM_APIHandleException;
using LightGBM::Log;
using LightGBM::Network;
using LightGBM::Random;
using LightGBM::ReduceScatterFunction;

// some help functions used to convert data

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseRows(const void** data, int num_col, int data_type);

template<typename T>
std::function<std::vector<std::pair<int, double>>(T idx)>
RowFunctionFromCSR(const void* indptr, int indptr_type, const int32_t* indices,
                   const void* data, int data_type, int64_t nindptr, int64_t nelem);

// Row iterator of on column for CSC matrix
class CSC_RowIterator {
 public:
  CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx);
  ~CSC_RowIterator() {}
  // return value at idx, only can access by ascent order
  double Get(int idx);
  // return next non-zero pair, if index < 0, means no more data
  std::pair<int, double> NextNonZero();

 private:
  int nonzero_idx_ = 0;
  int cur_idx_ = -1;
  double cur_val_ = 0.0f;
  bool is_end_ = false;
  std::function<std::pair<int, double>(int idx)> iter_fun_;
};

// start of c_api functions

const char* LGBM_GetLastError() {
  return LastErrorMsg();
}

int LGBM_RegisterLogCallback(void (*callback)(const char*)) {
  API_BEGIN();
  Log::ResetCallBack(callback);
  API_END();
}

static inline int SampleCount(int32_t total_nrow, const Config& config) {
  return static_cast<int>(total_nrow < config.bin_construct_sample_cnt ? total_nrow : config.bin_construct_sample_cnt);
}

static inline std::vector<int32_t> CreateSampleIndices(int32_t total_nrow, const Config& config) {
  Random rand(config.data_random_seed);
  int sample_cnt = SampleCount(total_nrow, config);
  return rand.Sample(total_nrow, sample_cnt);
}

int LGBM_GetSampleCount(int32_t num_total_row,
                        const char* parameters,
                        int* out) {
  API_BEGIN();
  if (out == nullptr) {
    Log::Fatal("LGBM_GetSampleCount output is nullptr");
  }
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);

  *out = SampleCount(num_total_row, config);
  API_END();
}

int LGBM_SampleIndices(int32_t num_total_row,
                       const char* parameters,
                       void* out,
                       int32_t* out_len) {
  // This API is to keep python binding's behavior the same with C++ implementation.
  // Sample count, random seed etc. should be provided in parameters.
  API_BEGIN();
  if (out == nullptr) {
    Log::Fatal("LGBM_SampleIndices output is nullptr");
  }
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);

  auto sample_indices = CreateSampleIndices(num_total_row, config);
  memcpy(out, sample_indices.data(), sizeof(int32_t) * sample_indices.size());
  *out_len = static_cast<int32_t>(sample_indices.size());
  API_END();
}

int LGBM_DatasetCreateFromFile(const char* filename,
                               const char* parameters,
                               const DatasetHandle reference,
                               DatasetHandle* out) {
  API_BEGIN();
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  DatasetLoader loader(config, nullptr, 1, filename);
  if (reference == nullptr) {
    if (Network::num_machines() == 1) {
      *out = loader.LoadFromFile(filename);
    } else {
      *out = loader.LoadFromFile(filename, Network::rank(), Network::num_machines());
    }
  } else {
    *out = loader.LoadFromFileAlignWithOtherDataset(filename,
                                                    reinterpret_cast<const Dataset*>(reference));
  }
  API_END();
}


int LGBM_DatasetCreateFromSampledColumn(double** sample_data,
                                        int** sample_indices,
                                        int32_t ncol,
                                        const int* num_per_col,
                                        int32_t num_sample_row,
                                        int32_t num_total_row,
                                        const char* parameters,
                                        DatasetHandle* out) {
  API_BEGIN();
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  DatasetLoader loader(config, nullptr, 1, nullptr);
  *out = loader.ConstructFromSampleData(sample_data, sample_indices, ncol, num_per_col,
                                        num_sample_row,
                                        static_cast<data_size_t>(num_total_row));
  API_END();
}


int LGBM_DatasetCreateByReference(const DatasetHandle reference,
                                  int64_t num_total_row,
                                  DatasetHandle* out) {
  API_BEGIN();
  std::unique_ptr<Dataset> ret;
  ret.reset(new Dataset(static_cast<data_size_t>(num_total_row)));
  ret->CreateValid(reinterpret_cast<const Dataset*>(reference));
  *out = ret.release();
  API_END();
}

int LGBM_DatasetPushRows(DatasetHandle dataset,
                         const void* data,
                         int data_type,
                         int32_t nrow,
                         int32_t ncol,
                         int32_t start_row) {
  API_BEGIN();
  auto p_dataset = reinterpret_cast<Dataset*>(dataset);
  auto get_row_fun = RowFunctionFromDenseMatric(data, nrow, ncol, data_type, 1);
  if (p_dataset->has_raw()) {
    p_dataset->ResizeRaw(p_dataset->num_numeric_features() + nrow);
  }
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nrow; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    p_dataset->PushOneRow(tid, start_row + i, one_row);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  if (start_row + nrow == p_dataset->num_data()) {
    p_dataset->FinishLoad();
  }
  API_END();
}

int LGBM_DatasetPushRowsByCSR(DatasetHandle dataset,
                              const void* indptr,
                              int indptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t nindptr,
                              int64_t nelem,
                              int64_t,
                              int64_t start_row) {
  API_BEGIN();
  auto p_dataset = reinterpret_cast<Dataset*>(dataset);
  auto get_row_fun = RowFunctionFromCSR<int>(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int32_t nrow = static_cast<int32_t>(nindptr - 1);
  if (p_dataset->has_raw()) {
    p_dataset->ResizeRaw(p_dataset->num_numeric_features() + nrow);
  }
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nrow; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    p_dataset->PushOneRow(tid, static_cast<data_size_t>(start_row + i), one_row);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  if (start_row + nrow == static_cast<int64_t>(p_dataset->num_data())) {
    p_dataset->FinishLoad();
  }
  API_END();
}

int LGBM_DatasetCreateFromMat(const void* data,
                              int data_type,
                              int32_t nrow,
                              int32_t ncol,
                              int is_row_major,
                              const char* parameters,
                              const DatasetHandle reference,
                              DatasetHandle* out) {
  return LGBM_DatasetCreateFromMats(1,
                                    &data,
                                    data_type,
                                    &nrow,
                                    ncol,
                                    is_row_major,
                                    parameters,
                                    reference,
                                    out);
}

int LGBM_DatasetCreateFromMats(int32_t nmat,
                               const void** data,
                               int data_type,
                               int32_t* nrow,
                               int32_t ncol,
                               int is_row_major,
                               const char* parameters,
                               const DatasetHandle reference,
                               DatasetHandle* out) {
  API_BEGIN();
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  std::unique_ptr<Dataset> ret;
  int32_t total_nrow = 0;
  for (int j = 0; j < nmat; ++j) {
    total_nrow += nrow[j];
  }

  std::vector<std::function<std::vector<double>(int row_idx)>> get_row_fun;
  for (int j = 0; j < nmat; ++j) {
    get_row_fun.push_back(RowFunctionFromDenseMatric(data[j], nrow[j], ncol, data_type, is_row_major));
  }

  if (reference == nullptr) {
    // sample data first
    auto sample_indices = CreateSampleIndices(total_nrow, config);
    int sample_cnt = static_cast<int>(sample_indices.size());
    std::vector<std::vector<double>> sample_values(ncol);
    std::vector<std::vector<int>> sample_idx(ncol);

    int offset = 0;
    int j = 0;
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      while ((idx - offset) >= nrow[j]) {
        offset += nrow[j];
        ++j;
      }

      auto row = get_row_fun[j](static_cast<int>(idx - offset));
      for (size_t k = 0; k < row.size(); ++k) {
        if (std::fabs(row[k]) > kZeroThreshold || std::isnan(row[k])) {
          sample_values[k].emplace_back(row[k]);
          sample_idx[k].emplace_back(static_cast<int>(i));
        }
      }
    }
    DatasetLoader loader(config, nullptr, 1, nullptr);
    ret.reset(loader.ConstructFromSampleData(Vector2Ptr<double>(&sample_values).data(),
                                             Vector2Ptr<int>(&sample_idx).data(),
                                             ncol,
                                             VectorSize<double>(sample_values).data(),
                                             sample_cnt, total_nrow));
  } else {
    ret.reset(new Dataset(total_nrow));
    ret->CreateValid(
      reinterpret_cast<const Dataset*>(reference));
    if (ret->has_raw()) {
      ret->ResizeRaw(total_nrow);
    }
  }
  int32_t start_row = 0;
  for (int j = 0; j < nmat; ++j) {
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrow[j]; ++i) {
      OMP_LOOP_EX_BEGIN();
      const int tid = omp_get_thread_num();
      auto one_row = get_row_fun[j](i);
      ret->PushOneRow(tid, start_row + i, one_row);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();

    start_row += nrow[j];
  }
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

int LGBM_DatasetCreateFromCSR(const void* indptr,
                              int indptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t nindptr,
                              int64_t nelem,
                              int64_t num_col,
                              const char* parameters,
                              const DatasetHandle reference,
                              DatasetHandle* out) {
  API_BEGIN();
  if (num_col <= 0) {
    Log::Fatal("The number of columns should be greater than zero.");
  } else if (num_col >= INT32_MAX) {
    Log::Fatal("The number of columns should be smaller than INT32_MAX.");
  }
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  std::unique_ptr<Dataset> ret;
  auto get_row_fun = RowFunctionFromCSR<int>(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int32_t nrow = static_cast<int32_t>(nindptr - 1);
  if (reference == nullptr) {
    // sample data first
    auto sample_indices = CreateSampleIndices(nrow, config);
    int sample_cnt = static_cast<int>(sample_indices.size());
    std::vector<std::vector<double>> sample_values(num_col);
    std::vector<std::vector<int>> sample_idx(num_col);
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      for (std::pair<int, double>& inner_data : row) {
        CHECK_LT(inner_data.first, num_col);
        if (std::fabs(inner_data.second) > kZeroThreshold || std::isnan(inner_data.second)) {
          sample_values[inner_data.first].emplace_back(inner_data.second);
          sample_idx[inner_data.first].emplace_back(static_cast<int>(i));
        }
      }
    }
    DatasetLoader loader(config, nullptr, 1, nullptr);
    ret.reset(loader.ConstructFromSampleData(Vector2Ptr<double>(&sample_values).data(),
                                             Vector2Ptr<int>(&sample_idx).data(),
                                             static_cast<int>(num_col),
                                             VectorSize<double>(sample_values).data(),
                                             sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow));
    ret->CreateValid(
      reinterpret_cast<const Dataset*>(reference));
    if (ret->has_raw()) {
      ret->ResizeRaw(nrow);
    }
  }
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nindptr - 1; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

int LGBM_DatasetCreateFromCSRFunc(void* get_row_funptr,
                                  int num_rows,
                                  int64_t num_col,
                                  const char* parameters,
                                  const DatasetHandle reference,
                                  DatasetHandle* out) {
  API_BEGIN();
  if (num_col <= 0) {
    Log::Fatal("The number of columns should be greater than zero.");
  } else if (num_col >= INT32_MAX) {
    Log::Fatal("The number of columns should be smaller than INT32_MAX.");
  }
  auto get_row_fun = *static_cast<std::function<void(int idx, std::vector<std::pair<int, double>>&)>*>(get_row_funptr);
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  std::unique_ptr<Dataset> ret;
  int32_t nrow = num_rows;
  if (reference == nullptr) {
    // sample data first
    auto sample_indices = CreateSampleIndices(nrow, config);
    int sample_cnt = static_cast<int>(sample_indices.size());
    std::vector<std::vector<double>> sample_values(num_col);
    std::vector<std::vector<int>> sample_idx(num_col);
    // local buffer to re-use memory
    std::vector<std::pair<int, double>> buffer;
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      get_row_fun(static_cast<int>(idx), buffer);
      for (std::pair<int, double>& inner_data : buffer) {
        CHECK_LT(inner_data.first, num_col);
        if (std::fabs(inner_data.second) > kZeroThreshold || std::isnan(inner_data.second)) {
          sample_values[inner_data.first].emplace_back(inner_data.second);
          sample_idx[inner_data.first].emplace_back(static_cast<int>(i));
        }
      }
    }
    DatasetLoader loader(config, nullptr, 1, nullptr);
    ret.reset(loader.ConstructFromSampleData(Vector2Ptr<double>(&sample_values).data(),
                                             Vector2Ptr<int>(&sample_idx).data(),
                                             static_cast<int>(num_col),
                                             VectorSize<double>(sample_values).data(),
                                             sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow));
    ret->CreateValid(
      reinterpret_cast<const Dataset*>(reference));
    if (ret->has_raw()) {
      ret->ResizeRaw(nrow);
    }
  }

  OMP_INIT_EX();
  std::vector<std::pair<int, double>> thread_buffer;
  #pragma omp parallel for schedule(static) private(thread_buffer)
  for (int i = 0; i < num_rows; ++i) {
    OMP_LOOP_EX_BEGIN();
    {
      const int tid = omp_get_thread_num();
      get_row_fun(i, thread_buffer);
      ret->PushOneRow(tid, i, thread_buffer);
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

int LGBM_DatasetCreateFromCSC(const void* col_ptr,
                              int col_ptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t ncol_ptr,
                              int64_t nelem,
                              int64_t num_row,
                              const char* parameters,
                              const DatasetHandle reference,
                              DatasetHandle* out) {
  API_BEGIN();
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  std::unique_ptr<Dataset> ret;
  int32_t nrow = static_cast<int32_t>(num_row);
  if (reference == nullptr) {
    // sample data first
    auto sample_indices = CreateSampleIndices(nrow, config);
    int sample_cnt = static_cast<int>(sample_indices.size());
    std::vector<std::vector<double>> sample_values(ncol_ptr - 1);
    std::vector<std::vector<int>> sample_idx(ncol_ptr - 1);
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
      OMP_LOOP_EX_BEGIN();
      CSC_RowIterator col_it(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, i);
      for (int j = 0; j < sample_cnt; j++) {
        auto val = col_it.Get(sample_indices[j]);
        if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
          sample_values[i].emplace_back(val);
          sample_idx[i].emplace_back(j);
        }
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    DatasetLoader loader(config, nullptr, 1, nullptr);
    ret.reset(loader.ConstructFromSampleData(Vector2Ptr<double>(&sample_values).data(),
                                             Vector2Ptr<int>(&sample_idx).data(),
                                             static_cast<int>(sample_values.size()),
                                             VectorSize<double>(sample_values).data(),
                                             sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow));
    ret->CreateValid(
      reinterpret_cast<const Dataset*>(reference));
  }
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < ncol_ptr - 1; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    int feature_idx = ret->InnerFeatureIndex(i);
    if (feature_idx < 0) { continue; }
    int group = ret->Feature2Group(feature_idx);
    int sub_feature = ret->Feture2SubFeature(feature_idx);
    CSC_RowIterator col_it(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, i);
    auto bin_mapper = ret->FeatureBinMapper(feature_idx);
    if (bin_mapper->GetDefaultBin() == bin_mapper->GetMostFreqBin()) {
      int row_idx = 0;
      while (row_idx < nrow) {
        auto pair = col_it.NextNonZero();
        row_idx = pair.first;
        // no more data
        if (row_idx < 0) { break; }
        ret->PushOneData(tid, row_idx, group, feature_idx, sub_feature, pair.second);
      }
    } else {
      for (int row_idx = 0; row_idx < nrow; ++row_idx) {
        auto val = col_it.Get(row_idx);
        ret->PushOneData(tid, row_idx, group, feature_idx, sub_feature, val);
      }
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

int LGBM_DatasetGetSubset(
  const DatasetHandle handle,
  const int32_t* used_row_indices,
  int32_t num_used_row_indices,
  const char* parameters,
  DatasetHandle* out) {
  API_BEGIN();
  auto param = Config::Str2Map(parameters);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  auto full_dataset = reinterpret_cast<const Dataset*>(handle);
  CHECK_GT(num_used_row_indices, 0);
  const int32_t lower = 0;
  const int32_t upper = full_dataset->num_data() - 1;
  CheckElementsIntervalClosed(used_row_indices, lower, upper, num_used_row_indices, "Used indices of subset");
  if (!std::is_sorted(used_row_indices, used_row_indices + num_used_row_indices)) {
    Log::Fatal("used_row_indices should be sorted in Subset");
  }
  auto ret = std::unique_ptr<Dataset>(new Dataset(num_used_row_indices));
  ret->CopyFeatureMapperFrom(full_dataset);
  ret->CopySubrow(full_dataset, used_row_indices, num_used_row_indices, true);
  *out = ret.release();
  API_END();
}

int LGBM_DatasetSetFeatureNames(
  DatasetHandle handle,
  const char** feature_names,
  int num_feature_names) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  std::vector<std::string> feature_names_str;
  for (int i = 0; i < num_feature_names; ++i) {
    feature_names_str.emplace_back(feature_names[i]);
  }
  dataset->set_feature_names(feature_names_str);
  API_END();
}

int LGBM_DatasetGetFeatureNames(
    DatasetHandle handle,
    const int len,
    int* num_feature_names,
    const size_t buffer_len,
    size_t* out_buffer_len,
    char** feature_names) {
  API_BEGIN();
  *out_buffer_len = 0;
  auto dataset = reinterpret_cast<Dataset*>(handle);
  auto inside_feature_name = dataset->feature_names();
  *num_feature_names = static_cast<int>(inside_feature_name.size());
  for (int i = 0; i < *num_feature_names; ++i) {
    if (i < len) {
      std::memcpy(feature_names[i], inside_feature_name[i].c_str(), std::min(inside_feature_name[i].size() + 1, buffer_len));
      feature_names[i][buffer_len - 1] = '\0';
    }
    *out_buffer_len = std::max(inside_feature_name[i].size() + 1, *out_buffer_len);
  }
  API_END();
}

#ifdef _MSC_VER
  #pragma warning(disable : 4702)
#endif
int LGBM_DatasetFree(DatasetHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Dataset*>(handle);
  API_END();
}

int LGBM_DatasetSaveBinary(DatasetHandle handle,
                           const char* filename) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->SaveBinaryFile(filename);
  API_END();
}

int LGBM_DatasetDumpText(DatasetHandle handle,
                         const char* filename) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->DumpTextFile(filename);
  API_END();
}

int LGBM_DatasetSetField(DatasetHandle handle,
                         const char* field_name,
                         const void* field_data,
                         int num_element,
                         int type) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  bool is_success = false;
  if (type == C_API_DTYPE_FLOAT32) {
    is_success = dataset->SetFloatField(field_name, reinterpret_cast<const float*>(field_data), static_cast<int32_t>(num_element));
  } else if (type == C_API_DTYPE_INT32) {
    is_success = dataset->SetIntField(field_name, reinterpret_cast<const int*>(field_data), static_cast<int32_t>(num_element));
  } else if (type == C_API_DTYPE_FLOAT64) {
    is_success = dataset->SetDoubleField(field_name, reinterpret_cast<const double*>(field_data), static_cast<int32_t>(num_element));
  }
  if (!is_success) { Log::Fatal("Input data type error or field not found"); }
  API_END();
}

int LGBM_DatasetGetField(DatasetHandle handle,
                         const char* field_name,
                         int* out_len,
                         const void** out_ptr,
                         int* out_type) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  bool is_success = false;
  if (dataset->GetFloatField(field_name, out_len, reinterpret_cast<const float**>(out_ptr))) {
    *out_type = C_API_DTYPE_FLOAT32;
    is_success = true;
  } else if (dataset->GetIntField(field_name, out_len, reinterpret_cast<const int**>(out_ptr))) {
    *out_type = C_API_DTYPE_INT32;
    is_success = true;
  } else if (dataset->GetDoubleField(field_name, out_len, reinterpret_cast<const double**>(out_ptr))) {
    *out_type = C_API_DTYPE_FLOAT64;
    is_success = true;
  }
  if (!is_success) { Log::Fatal("Field not found"); }
  if (*out_ptr == nullptr) { *out_len = 0; }
  API_END();
}

int LGBM_DatasetUpdateParamChecking(const char* old_parameters, const char* new_parameters) {
  API_BEGIN();
  auto old_param = Config::Str2Map(old_parameters);
  Config old_config;
  old_config.Set(old_param);
  auto new_param = Config::Str2Map(new_parameters);
  Booster::CheckDatasetResetConfig(old_config, new_param);
  API_END();
}

int LGBM_DatasetGetNumData(DatasetHandle handle,
                           int* out) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_data();
  API_END();
}

int LGBM_DatasetGetNumFeature(DatasetHandle handle,
                              int* out) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_total_features();
  API_END();
}

int LGBM_DatasetAddFeaturesFrom(DatasetHandle target,
                                DatasetHandle source) {
  API_BEGIN();
  auto target_d = reinterpret_cast<Dataset*>(target);
  auto source_d = reinterpret_cast<Dataset*>(source);
  target_d->AddFeaturesFrom(source_d);
  API_END();
}

// ---- start of booster

int LGBM_BoosterCreate(const DatasetHandle train_data,
                       const char* parameters,
                       BoosterHandle* out) {
  API_BEGIN();
  const Dataset* p_train_data = reinterpret_cast<const Dataset*>(train_data);
  auto ret = std::unique_ptr<Booster>(new Booster(p_train_data, parameters));
  *out = ret.release();
  API_END();
}

int LGBM_BoosterCreateFromModelfile(
  const char* filename,
  int* out_num_iterations,
  BoosterHandle* out) {
  API_BEGIN();
  auto ret = std::unique_ptr<Booster>(new Booster(filename));
  *out_num_iterations = ret->GetBoosting()->GetCurrentIteration();
  *out = ret.release();
  API_END();
}

int LGBM_BoosterLoadModelFromString(
  const char* model_str,
  int* out_num_iterations,
  BoosterHandle* out) {
  API_BEGIN();
  auto ret = std::unique_ptr<Booster>(new Booster(nullptr));
  ret->LoadModelFromString(model_str);
  *out_num_iterations = ret->GetBoosting()->GetCurrentIteration();
  *out = ret.release();
  API_END();
}

#ifdef _MSC_VER
  #pragma warning(disable : 4702)
#endif
int LGBM_BoosterFree(BoosterHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Booster*>(handle);
  API_END();
}

int LGBM_BoosterShuffleModels(BoosterHandle handle, int start_iter, int end_iter) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->ShuffleModels(start_iter, end_iter);
  API_END();
}

int LGBM_BoosterMerge(BoosterHandle handle,
                      BoosterHandle other_handle) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  Booster* ref_other_booster = reinterpret_cast<Booster*>(other_handle);
  ref_booster->MergeFrom(ref_other_booster);
  API_END();
}

int LGBM_BoosterAddValidData(BoosterHandle handle,
                             const DatasetHandle valid_data) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  const Dataset* p_dataset = reinterpret_cast<const Dataset*>(valid_data);
  ref_booster->AddValidData(p_dataset);
  API_END();
}

int LGBM_BoosterResetTrainingData(BoosterHandle handle,
                                  const DatasetHandle train_data) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  const Dataset* p_dataset = reinterpret_cast<const Dataset*>(train_data);
  ref_booster->ResetTrainingData(p_dataset);
  API_END();
}

int LGBM_BoosterResetParameter(BoosterHandle handle, const char* parameters) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->ResetConfig(parameters);
  API_END();
}

int LGBM_BoosterGetNumClasses(BoosterHandle handle, int* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetBoosting()->NumberOfClasses();
  API_END();
}

int LGBM_BoosterGetLinear(BoosterHandle handle, bool* out) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out = ref_booster->GetBoosting()->IsLinear();
  API_END();
}

int LGBM_BoosterRefit(BoosterHandle handle, const int32_t* leaf_preds, int32_t nrow, int32_t ncol) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->Refit(leaf_preds, nrow, ncol);
  API_END();
}

int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  if (ref_booster->TrainOneIter()) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  API_END();
}

int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
                                    const float* grad,
                                    const float* hess,
                                    int* is_finished) {
  API_BEGIN();
  #ifdef SCORE_T_USE_DOUBLE
  (void) handle;       // UNUSED VARIABLE
  (void) grad;         // UNUSED VARIABLE
  (void) hess;         // UNUSED VARIABLE
  (void) is_finished;  // UNUSED VARIABLE
  Log::Fatal("Don't support custom loss function when SCORE_T_USE_DOUBLE is enabled");
  #else
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  if (ref_booster->TrainOneIter(grad, hess)) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  #endif
  API_END();
}

int LGBM_BoosterRollbackOneIter(BoosterHandle handle) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->RollbackOneIter();
  API_END();
}

int LGBM_BoosterGetCurrentIteration(BoosterHandle handle, int* out_iteration) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_iteration = ref_booster->GetBoosting()->GetCurrentIteration();
  API_END();
}

int LGBM_BoosterNumModelPerIteration(BoosterHandle handle, int* out_tree_per_iteration) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_tree_per_iteration = ref_booster->GetBoosting()->NumModelPerIteration();
  API_END();
}

int LGBM_BoosterNumberOfTotalModel(BoosterHandle handle, int* out_models) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_models = ref_booster->GetBoosting()->NumberOfTotalModel();
  API_END();
}

int LGBM_BoosterGetEvalCounts(BoosterHandle handle, int* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetEvalCounts();
  API_END();
}

int LGBM_BoosterGetEvalNames(BoosterHandle handle,
                             const int len,
                             int* out_len,
                             const size_t buffer_len,
                             size_t* out_buffer_len,
                             char** out_strs) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetEvalNames(out_strs, len, buffer_len, out_buffer_len);
  API_END();
}

int LGBM_BoosterGetFeatureNames(BoosterHandle handle,
                                const int len,
                                int* out_len,
                                const size_t buffer_len,
                                size_t* out_buffer_len,
                                char** out_strs) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetFeatureNames(out_strs, len, buffer_len, out_buffer_len);
  API_END();
}

int LGBM_BoosterGetNumFeature(BoosterHandle handle, int* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetBoosting()->MaxFeatureIdx() + 1;
  API_END();
}

int LGBM_BoosterGetEval(BoosterHandle handle,
                        int data_idx,
                        int* out_len,
                        double* out_results) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto boosting = ref_booster->GetBoosting();
  auto result_buf = boosting->GetEvalAt(data_idx);
  *out_len = static_cast<int>(result_buf.size());
  for (size_t i = 0; i < result_buf.size(); ++i) {
    (out_results)[i] = static_cast<double>(result_buf[i]);
  }
  API_END();
}

int LGBM_BoosterGetNumPredict(BoosterHandle handle,
                              int data_idx,
                              int64_t* out_len) {
  API_BEGIN();
  auto boosting = reinterpret_cast<Booster*>(handle)->GetBoosting();
  *out_len = boosting->GetNumPredictAt(data_idx);
  API_END();
}

int LGBM_BoosterGetPredict(BoosterHandle handle,
                           int data_idx,
                           int64_t* out_len,
                           double* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->GetPredictAt(data_idx, out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictForFile(BoosterHandle handle,
                               const char* data_filename,
                               int data_has_header,
                               int predict_type,
                               int start_iteration,
                               int num_iteration,
                               const char* parameter,
                               const char* result_filename) {
  API_BEGIN();
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->Predict(start_iteration, num_iteration, predict_type, data_filename, data_has_header,
                       config, result_filename);
  API_END();
}

int LGBM_BoosterCalcNumPredict(BoosterHandle handle,
                               int num_row,
                               int predict_type,
                               int start_iteration,
                               int num_iteration,
                               int64_t* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = static_cast<int64_t>(num_row) * ref_booster->GetBoosting()->NumPredictOneRow(start_iteration,
    num_iteration, predict_type == C_API_PREDICT_LEAF_INDEX, predict_type == C_API_PREDICT_CONTRIB);
  API_END();
}

/*!
 * \brief Object to store resources meant for single-row Fast Predict methods.
 *
 * Meant to be used as a basic struct by the *Fast* predict methods only.
 * It stores the configuration resources for reuse during prediction.
 *
 * Even the row function is stored. We score the instance at the same memory
 * address all the time. One just replaces the feature values at that address
 * and scores again with the *Fast* methods.
 */
struct FastConfig {
  FastConfig(Booster *const booster_ptr,
             const char *parameter,
             const int predict_type_,
             const int data_type_,
             const int32_t num_cols) : booster(booster_ptr), predict_type(predict_type_), data_type(data_type_), ncol(num_cols) {
    config.Set(Config::Str2Map(parameter));
  }

  Booster* const booster;
  Config config;
  const int predict_type;
  const int data_type;
  const int32_t ncol;
};

int LGBM_FastConfigFree(FastConfigHandle fastConfig) {
  API_BEGIN();
  delete reinterpret_cast<FastConfig*>(fastConfig);
  API_END();
}

int LGBM_BoosterPredictForCSR(BoosterHandle handle,
                              const void* indptr,
                              int indptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t nindptr,
                              int64_t nelem,
                              int64_t num_col,
                              int predict_type,
                              int start_iteration,
                              int num_iteration,
                              const char* parameter,
                              int64_t* out_len,
                              double* out_result) {
  API_BEGIN();
  if (num_col <= 0) {
    Log::Fatal("The number of columns should be greater than zero.");
  } else if (num_col >= INT32_MAX) {
    Log::Fatal("The number of columns should be smaller than INT32_MAX.");
  }
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowFunctionFromCSR<int>(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int nrow = static_cast<int>(nindptr - 1);
  ref_booster->Predict(start_iteration, num_iteration, predict_type, nrow, static_cast<int>(num_col), get_row_fun,
                       config, out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictSparseOutput(BoosterHandle handle,
                                    const void* indptr,
                                    int indptr_type,
                                    const int32_t* indices,
                                    const void* data,
                                    int data_type,
                                    int64_t nindptr,
                                    int64_t nelem,
                                    int64_t num_col_or_row,
                                    int predict_type,
                                    int start_iteration,
                                    int num_iteration,
                                    const char* parameter,
                                    int matrix_type,
                                    int64_t* out_len,
                                    void** out_indptr,
                                    int32_t** out_indices,
                                    void** out_data) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  if (matrix_type == C_API_MATRIX_TYPE_CSR) {
    if (num_col_or_row <= 0) {
      Log::Fatal("The number of columns should be greater than zero.");
    } else if (num_col_or_row >= INT32_MAX) {
      Log::Fatal("The number of columns should be smaller than INT32_MAX.");
    }
    auto get_row_fun = RowFunctionFromCSR<int64_t>(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
    int64_t nrow = nindptr - 1;
    ref_booster->PredictSparseCSR(start_iteration, num_iteration, predict_type, nrow, static_cast<int>(num_col_or_row), get_row_fun,
                                  config, out_len, out_indptr, indptr_type, out_indices, out_data, data_type);
  } else if (matrix_type == C_API_MATRIX_TYPE_CSC) {
    int num_threads = OMP_NUM_THREADS();
    int ncol = static_cast<int>(nindptr - 1);
    std::vector<std::vector<CSC_RowIterator>> iterators(num_threads, std::vector<CSC_RowIterator>());
    for (int i = 0; i < num_threads; ++i) {
      for (int j = 0; j < ncol; ++j) {
        iterators[i].emplace_back(indptr, indptr_type, indices, data, data_type, nindptr, nelem, j);
      }
    }
    std::function<std::vector<std::pair<int, double>>(int64_t row_idx)> get_row_fun =
      [&iterators, ncol](int64_t i) {
      std::vector<std::pair<int, double>> one_row;
      one_row.reserve(ncol);
      const int tid = omp_get_thread_num();
      for (int j = 0; j < ncol; ++j) {
        auto val = iterators[tid][j].Get(static_cast<int>(i));
        if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
          one_row.emplace_back(j, val);
        }
      }
      return one_row;
    };
    ref_booster->PredictSparseCSC(start_iteration, num_iteration, predict_type, num_col_or_row, ncol, get_row_fun, config,
                                  out_len, out_indptr, indptr_type, out_indices, out_data, data_type);
  } else {
    Log::Fatal("Unknown matrix type in LGBM_BoosterPredictSparseOutput");
  }
  API_END();
}

int LGBM_BoosterFreePredictSparse(void* indptr, int32_t* indices, void* data, int indptr_type, int data_type) {
  API_BEGIN();
  if (indptr_type == C_API_DTYPE_INT32) {
    delete reinterpret_cast<int32_t*>(indptr);
  } else if (indptr_type == C_API_DTYPE_INT64) {
    delete reinterpret_cast<int64_t*>(indptr);
  } else {
    Log::Fatal("Unknown indptr type in LGBM_BoosterFreePredictSparse");
  }
  delete indices;
  if (data_type == C_API_DTYPE_FLOAT32) {
    delete reinterpret_cast<float*>(data);
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    delete reinterpret_cast<double*>(data);
  } else {
    Log::Fatal("Unknown data type in LGBM_BoosterFreePredictSparse");
  }
  API_END();
}

int LGBM_BoosterPredictForCSRSingleRow(BoosterHandle handle,
                                       const void* indptr,
                                       int indptr_type,
                                       const int32_t* indices,
                                       const void* data,
                                       int data_type,
                                       int64_t nindptr,
                                       int64_t nelem,
                                       int64_t num_col,
                                       int predict_type,
                                       int start_iteration,
                                       int num_iteration,
                                       const char* parameter,
                                       int64_t* out_len,
                                       double* out_result) {
  API_BEGIN();
  if (num_col <= 0) {
    Log::Fatal("The number of columns should be greater than zero.");
  } else if (num_col >= INT32_MAX) {
    Log::Fatal("The number of columns should be smaller than INT32_MAX.");
  }
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowFunctionFromCSR<int>(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  ref_booster->SetSingleRowPredictor(start_iteration, num_iteration, predict_type, config);
  ref_booster->PredictSingleRow(predict_type, static_cast<int32_t>(num_col), get_row_fun, config, out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictForCSRSingleRowFastInit(BoosterHandle handle,
                                               const int predict_type,
                                               const int start_iteration,
                                               const int num_iteration,
                                               const int data_type,
                                               const int64_t num_col,
                                               const char* parameter,
                                               FastConfigHandle *out_fastConfig) {
  API_BEGIN();
  if (num_col <= 0) {
    Log::Fatal("The number of columns should be greater than zero.");
  } else if (num_col >= INT32_MAX) {
    Log::Fatal("The number of columns should be smaller than INT32_MAX.");
  }

  auto fastConfig_ptr = std::unique_ptr<FastConfig>(new FastConfig(
    reinterpret_cast<Booster*>(handle),
    parameter,
    predict_type,
    data_type,
    static_cast<int32_t>(num_col)));

  if (fastConfig_ptr->config.num_threads > 0) {
    omp_set_num_threads(fastConfig_ptr->config.num_threads);
  }

  fastConfig_ptr->booster->SetSingleRowPredictor(start_iteration, num_iteration, predict_type, fastConfig_ptr->config);

  *out_fastConfig = fastConfig_ptr.release();
  API_END();
}

int LGBM_BoosterPredictForCSRSingleRowFast(FastConfigHandle fastConfig_handle,
                                           const void* indptr,
                                           const int indptr_type,
                                           const int32_t* indices,
                                           const void* data,
                                           const int64_t nindptr,
                                           const int64_t nelem,
                                           int64_t* out_len,
                                           double* out_result) {
  API_BEGIN();
  FastConfig *fastConfig = reinterpret_cast<FastConfig*>(fastConfig_handle);
  auto get_row_fun = RowFunctionFromCSR<int>(indptr, indptr_type, indices, data, fastConfig->data_type, nindptr, nelem);
  fastConfig->booster->PredictSingleRow(fastConfig->predict_type, fastConfig->ncol,
                                        get_row_fun, fastConfig->config, out_result, out_len);
  API_END();
}


int LGBM_BoosterPredictForCSC(BoosterHandle handle,
                              const void* col_ptr,
                              int col_ptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t ncol_ptr,
                              int64_t nelem,
                              int64_t num_row,
                              int predict_type,
                              int start_iteration,
                              int num_iteration,
                              const char* parameter,
                              int64_t* out_len,
                              double* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  int num_threads = OMP_NUM_THREADS();
  int ncol = static_cast<int>(ncol_ptr - 1);
  std::vector<std::vector<CSC_RowIterator>> iterators(num_threads, std::vector<CSC_RowIterator>());
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < ncol; ++j) {
      iterators[i].emplace_back(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, j);
    }
  }
  std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun =
      [&iterators, ncol](int i) {
        std::vector<std::pair<int, double>> one_row;
        one_row.reserve(ncol);
        const int tid = omp_get_thread_num();
        for (int j = 0; j < ncol; ++j) {
          auto val = iterators[tid][j].Get(i);
          if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
            one_row.emplace_back(j, val);
          }
        }
        return one_row;
      };
  ref_booster->Predict(start_iteration, num_iteration, predict_type, static_cast<int>(num_row), ncol, get_row_fun, config,
                       out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictForMat(BoosterHandle handle,
                              const void* data,
                              int data_type,
                              int32_t nrow,
                              int32_t ncol,
                              int is_row_major,
                              int predict_type,
                              int start_iteration,
                              int num_iteration,
                              const char* parameter,
                              int64_t* out_len,
                              double* out_result) {
  API_BEGIN();
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowPairFunctionFromDenseMatric(data, nrow, ncol, data_type, is_row_major);
  ref_booster->Predict(start_iteration, num_iteration, predict_type, nrow, ncol, get_row_fun,
                       config, out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictForMatSingleRow(BoosterHandle handle,
                                       const void* data,
                                       int data_type,
                                       int32_t ncol,
                                       int is_row_major,
                                       int predict_type,
                                       int start_iteration,
                                       int num_iteration,
                                       const char* parameter,
                                       int64_t* out_len,
                                       double* out_result) {
  API_BEGIN();
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowPairFunctionFromDenseMatric(data, 1, ncol, data_type, is_row_major);
  ref_booster->SetSingleRowPredictor(start_iteration, num_iteration, predict_type, config);
  ref_booster->PredictSingleRow(predict_type, ncol, get_row_fun, config, out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictForMatSingleRowFastInit(BoosterHandle handle,
                                               const int predict_type,
                                               const int start_iteration,
                                               const int num_iteration,
                                               const int data_type,
                                               const int32_t ncol,
                                               const char* parameter,
                                               FastConfigHandle *out_fastConfig) {
  API_BEGIN();
  auto fastConfig_ptr = std::unique_ptr<FastConfig>(new FastConfig(
    reinterpret_cast<Booster*>(handle),
    parameter,
    predict_type,
    data_type,
    ncol));

  if (fastConfig_ptr->config.num_threads > 0) {
    omp_set_num_threads(fastConfig_ptr->config.num_threads);
  }

  fastConfig_ptr->booster->SetSingleRowPredictor(start_iteration, num_iteration, predict_type, fastConfig_ptr->config);

  *out_fastConfig = fastConfig_ptr.release();
  API_END();
}

int LGBM_BoosterPredictForMatSingleRowFast(FastConfigHandle fastConfig_handle,
                                           const void* data,
                                           int64_t* out_len,
                                           double* out_result) {
  API_BEGIN();
  FastConfig *fastConfig = reinterpret_cast<FastConfig*>(fastConfig_handle);
  // Single row in row-major format:
  auto get_row_fun = RowPairFunctionFromDenseMatric(data, 1, fastConfig->ncol, fastConfig->data_type, 1);
  fastConfig->booster->PredictSingleRow(fastConfig->predict_type, fastConfig->ncol,
                                        get_row_fun, fastConfig->config,
                                        out_result, out_len);
  API_END();
}


int LGBM_BoosterPredictForMats(BoosterHandle handle,
                               const void** data,
                               int data_type,
                               int32_t nrow,
                               int32_t ncol,
                               int predict_type,
                               int start_iteration,
                               int num_iteration,
                               const char* parameter,
                               int64_t* out_len,
                               double* out_result) {
  API_BEGIN();
  auto param = Config::Str2Map(parameter);
  Config config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowPairFunctionFromDenseRows(data, ncol, data_type);
  ref_booster->Predict(start_iteration, num_iteration, predict_type, nrow, ncol, get_row_fun, config, out_result, out_len);
  API_END();
}

int LGBM_BoosterSaveModel(BoosterHandle handle,
                          int start_iteration,
                          int num_iteration,
                          int feature_importance_type,
                          const char* filename) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->SaveModelToFile(start_iteration, num_iteration,
                               feature_importance_type, filename);
  API_END();
}

int LGBM_BoosterSaveModelToString(BoosterHandle handle,
                                  int start_iteration,
                                  int num_iteration,
                                  int feature_importance_type,
                                  int64_t buffer_len,
                                  int64_t* out_len,
                                  char* out_str) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::string model = ref_booster->SaveModelToString(
      start_iteration, num_iteration, feature_importance_type);
  *out_len = static_cast<int64_t>(model.size()) + 1;
  if (*out_len <= buffer_len) {
    std::memcpy(out_str, model.c_str(), *out_len);
  }
  API_END();
}

int LGBM_BoosterDumpModel(BoosterHandle handle,
                          int start_iteration,
                          int num_iteration,
                          int feature_importance_type,
                          int64_t buffer_len,
                          int64_t* out_len,
                          char* out_str) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::string model = ref_booster->DumpModel(start_iteration, num_iteration,
                                             feature_importance_type);
  *out_len = static_cast<int64_t>(model.size()) + 1;
  if (*out_len <= buffer_len) {
    std::memcpy(out_str, model.c_str(), *out_len);
  }
  API_END();
}

int LGBM_BoosterGetLeafValue(BoosterHandle handle,
                             int tree_idx,
                             int leaf_idx,
                             double* out_val) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_val = static_cast<double>(ref_booster->GetLeafValue(tree_idx, leaf_idx));
  API_END();
}

int LGBM_BoosterSetLeafValue(BoosterHandle handle,
                             int tree_idx,
                             int leaf_idx,
                             double val) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->SetLeafValue(tree_idx, leaf_idx, val);
  API_END();
}

int LGBM_BoosterFeatureImportance(BoosterHandle handle,
                                  int num_iteration,
                                  int importance_type,
                                  double* out_results) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::vector<double> feature_importances = ref_booster->FeatureImportance(num_iteration, importance_type);
  for (size_t i = 0; i < feature_importances.size(); ++i) {
    (out_results)[i] = feature_importances[i];
  }
  API_END();
}

int LGBM_BoosterGetUpperBoundValue(BoosterHandle handle,
                                   double* out_results) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  double max_value = ref_booster->UpperBoundValue();
  *out_results = max_value;
  API_END();
}

int LGBM_BoosterGetLowerBoundValue(BoosterHandle handle,
                                   double* out_results) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  double min_value = ref_booster->LowerBoundValue();
  *out_results = min_value;
  API_END();
}

int LGBM_NetworkInit(const char* machines,
                     int local_listen_port,
                     int listen_time_out,
                     int num_machines) {
  API_BEGIN();
  Config config;
  config.machines = RemoveQuotationSymbol(std::string(machines));
  config.local_listen_port = local_listen_port;
  config.num_machines = num_machines;
  config.time_out = listen_time_out;
  if (num_machines > 1) {
    Network::Init(config);
  }
  API_END();
}

int LGBM_NetworkFree() {
  API_BEGIN();
  Network::Dispose();
  API_END();
}

int LGBM_NetworkInitWithFunctions(int num_machines, int rank,
                                  void* reduce_scatter_ext_fun,
                                  void* allgather_ext_fun) {
  API_BEGIN();
  if (num_machines > 1) {
    Network::Init(num_machines, rank, (ReduceScatterFunction)reduce_scatter_ext_fun, (AllgatherFunction)allgather_ext_fun);
  }
  API_END();
}

// ---- start of some help functions


template<typename T>
std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric_helper(const void* data, int num_row, int num_col, int is_row_major) {
  const T* data_ptr = reinterpret_cast<const T*>(data);
  if (is_row_major) {
    return [=] (int row_idx) {
      std::vector<double> ret(num_col);
      auto tmp_ptr = data_ptr + static_cast<size_t>(num_col) * row_idx;
      for (int i = 0; i < num_col; ++i) {
        ret[i] = static_cast<double>(*(tmp_ptr + i));
      }
      return ret;
    };
  } else {
    return [=] (int row_idx) {
      std::vector<double> ret(num_col);
      for (int i = 0; i < num_col; ++i) {
        ret[i] = static_cast<double>(*(data_ptr + static_cast<size_t>(num_row) * i + row_idx));
      }
      return ret;
    };
  }
}

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    return RowFunctionFromDenseMatric_helper<float>(data, num_row, num_col, is_row_major);
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    return RowFunctionFromDenseMatric_helper<double>(data, num_row, num_col, is_row_major);
  }
  Log::Fatal("Unknown data type in RowFunctionFromDenseMatric");
  return nullptr;
}

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  auto inner_function = RowFunctionFromDenseMatric(data, num_row, num_col, data_type, is_row_major);
  if (inner_function != nullptr) {
    return [inner_function] (int row_idx) {
      auto raw_values = inner_function(row_idx);
      std::vector<std::pair<int, double>> ret;
      ret.reserve(raw_values.size());
      for (int i = 0; i < static_cast<int>(raw_values.size()); ++i) {
        if (std::fabs(raw_values[i]) > kZeroThreshold || std::isnan(raw_values[i])) {
          ret.emplace_back(i, raw_values[i]);
        }
      }
      return ret;
    };
  }
  return nullptr;
}

// data is array of pointers to individual rows
std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseRows(const void** data, int num_col, int data_type) {
  return [=](int row_idx) {
    auto inner_function = RowFunctionFromDenseMatric(data[row_idx], 1, num_col, data_type, /* is_row_major */ true);
    auto raw_values = inner_function(0);
    std::vector<std::pair<int, double>> ret;
    ret.reserve(raw_values.size());
    for (int i = 0; i < static_cast<int>(raw_values.size()); ++i) {
      if (std::fabs(raw_values[i]) > kZeroThreshold || std::isnan(raw_values[i])) {
        ret.emplace_back(i, raw_values[i]);
      }
    }
    return ret;
  };
}

template<typename T, typename T1, typename T2>
std::function<std::vector<std::pair<int, double>>(T idx)>
RowFunctionFromCSR_helper(const void* indptr, const int32_t* indices, const void* data) {
  const T1* data_ptr = reinterpret_cast<const T1*>(data);
  const T2* ptr_indptr = reinterpret_cast<const T2*>(indptr);
  return [=] (T idx) {
    std::vector<std::pair<int, double>> ret;
    int64_t start = ptr_indptr[idx];
    int64_t end = ptr_indptr[idx + 1];
    if (end - start > 0)  {
      ret.reserve(end - start);
    }
    for (int64_t i = start; i < end; ++i) {
      ret.emplace_back(indices[i], data_ptr[i]);
    }
    return ret;
  };
}

template<typename T>
std::function<std::vector<std::pair<int, double>>(T idx)>
RowFunctionFromCSR(const void* indptr, int indptr_type, const int32_t* indices, const void* data, int data_type, int64_t , int64_t ) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    if (indptr_type == C_API_DTYPE_INT32) {
     return RowFunctionFromCSR_helper<T, float, int32_t>(indptr, indices, data);
    } else if (indptr_type == C_API_DTYPE_INT64) {
     return RowFunctionFromCSR_helper<T, float, int64_t>(indptr, indices, data);
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    if (indptr_type == C_API_DTYPE_INT32) {
     return RowFunctionFromCSR_helper<T, double, int32_t>(indptr, indices, data);
    } else if (indptr_type == C_API_DTYPE_INT64) {
     return RowFunctionFromCSR_helper<T, double, int64_t>(indptr, indices, data);
    }
  }
  Log::Fatal("Unknown data type in RowFunctionFromCSR");
  return nullptr;
}



template <typename T1, typename T2>
std::function<std::pair<int, double>(int idx)> IterateFunctionFromCSC_helper(const void* col_ptr, const int32_t* indices, const void* data, int col_idx) {
  const T1* data_ptr = reinterpret_cast<const T1*>(data);
  const T2* ptr_col_ptr = reinterpret_cast<const T2*>(col_ptr);
  int64_t start = ptr_col_ptr[col_idx];
  int64_t end = ptr_col_ptr[col_idx + 1];
  return [=] (int offset) {
    int64_t i = static_cast<int64_t>(start + offset);
    if (i >= end) {
      return std::make_pair(-1, 0.0);
    }
    int idx = static_cast<int>(indices[i]);
    double val = static_cast<double>(data_ptr[i]);
    return std::make_pair(idx, val);
  };
}

std::function<std::pair<int, double>(int idx)>
IterateFunctionFromCSC(const void* col_ptr, int col_ptr_type, const int32_t* indices, const void* data, int data_type, int64_t ncol_ptr, int64_t , int col_idx) {
  CHECK(col_idx < ncol_ptr && col_idx >= 0);
  if (data_type == C_API_DTYPE_FLOAT32) {
    if (col_ptr_type == C_API_DTYPE_INT32) {
      return IterateFunctionFromCSC_helper<float, int32_t>(col_ptr, indices, data, col_idx);
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      return IterateFunctionFromCSC_helper<float, int64_t>(col_ptr, indices, data, col_idx);
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    if (col_ptr_type == C_API_DTYPE_INT32) {
      return IterateFunctionFromCSC_helper<double, int32_t>(col_ptr, indices, data, col_idx);
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      return IterateFunctionFromCSC_helper<double, int64_t>(col_ptr, indices, data, col_idx);
    }
  }
  Log::Fatal("Unknown data type in CSC matrix");
  return nullptr;
}

CSC_RowIterator::CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                                 const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx) {
  iter_fun_ = IterateFunctionFromCSC(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, col_idx);
}

double CSC_RowIterator::Get(int idx) {
  while (idx > cur_idx_ && !is_end_) {
    auto ret = iter_fun_(nonzero_idx_);
    if (ret.first < 0) {
      is_end_ = true;
      break;
    }
    cur_idx_ = ret.first;
    cur_val_ = ret.second;
    ++nonzero_idx_;
  }
  if (idx == cur_idx_) {
    return cur_val_;
  } else {
    return 0.0f;
  }
}

std::pair<int, double> CSC_RowIterator::NextNonZero() {
  if (!is_end_) {
    auto ret = iter_fun_(nonzero_idx_);
    ++nonzero_idx_;
    if (ret.first < 0) {
      is_end_ = true;
    }
    return ret;
  } else {
    return std::make_pair(-1, 0.0);
  }
}
