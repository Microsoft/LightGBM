/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "gbdt.h"

#include <LightGBM/metric.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <chrono>
#include <ctime>
#include <sstream>

namespace LightGBM {

Common::Timer global_timer;

int LGBM_config_::current_device = lgbm_device_cpu;
int LGBM_config_::current_learner = use_cpu_learner;

GBDT::GBDT()
    : iter_(0),
      train_data_(nullptr),
      config_(nullptr),
      objective_function_(nullptr),
      early_stopping_round_(0),
      es_first_metric_only_(false),
      max_feature_idx_(0),
      num_tree_per_iteration_(1),
      num_class_(1),
      num_iteration_for_pred_(0),
      shrinkage_rate_(0.1f),
      num_init_iteration_(0),
      need_re_bagging_(false),
      balanced_bagging_(false),
      bagging_runner_(0, bagging_rand_block_) {
  average_output_ = false;
  tree_learner_ = nullptr;
  linear_tree_ = false;
}

GBDT::~GBDT() {
}

void GBDT::Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
                const std::vector<const Metric*>& training_metrics) {
  CHECK_NOTNULL(train_data);
  train_data_ = train_data;
  if (!config->monotone_constraints.empty()) {
    CHECK_EQ(static_cast<size_t>(train_data_->num_total_features()), config->monotone_constraints.size());
  }
  if (!config->feature_contri.empty()) {
    CHECK_EQ(static_cast<size_t>(train_data_->num_total_features()), config->feature_contri.size());
  }
  iter_ = 0;
  num_iteration_for_pred_ = 0;
  max_feature_idx_ = 0;
  num_class_ = config->num_class;
  config_ = std::unique_ptr<Config>(new Config(*config));
  early_stopping_round_ = config_->early_stopping_round;
  es_first_metric_only_ = config_->first_metric_only;
  shrinkage_rate_ = config_->learning_rate;

  if (train_data_->category_encoding_provider() != nullptr) {
    category_encoding_provider_.reset(CategoryEncodingProvider::RecoverFromModelString(train_data_->category_encoding_provider()->DumpToString()));
  }

  if (config_->device_type == std::string("cuda")) {
    LGBM_config_::current_learner = use_cuda_learner;
  }

  // load forced_splits file
  if (!config->forcedsplits_filename.empty()) {
    std::ifstream forced_splits_file(config->forcedsplits_filename.c_str());
    std::stringstream buffer;
    buffer << forced_splits_file.rdbuf();
    std::string err;
    forced_splits_json_ = Json::parse(buffer.str(), &err);
  }

  objective_function_ = objective_function;
  num_tree_per_iteration_ = num_class_;
  if (objective_function_ != nullptr) {
    num_tree_per_iteration_ = objective_function_->NumModelPerIteration();
    if (objective_function_->IsRenewTreeOutput() && !config->monotone_constraints.empty()) {
      Log::Fatal("Cannot use ``monotone_constraints`` in %s objective, please disable it.", objective_function_->GetName());
    }
  }

  is_constant_hessian_ = GetIsConstHessian(objective_function);

  tree_learner_ = std::unique_ptr<TreeLearner>(TreeLearner::CreateTreeLearner(config_->tree_learner, config_->device_type,
                                                                              config_.get()));

  // init tree learner
  tree_learner_->Init(train_data_, is_constant_hessian_);
  tree_learner_->SetForcedSplit(&forced_splits_json_);

  // push training metrics
  training_metrics_.clear();
  for (const auto& metric : training_metrics) {
    training_metrics_.push_back(metric);
  }
  training_metrics_.shrink_to_fit();

  train_score_updater_.reset(new ScoreUpdater(train_data_, num_tree_per_iteration_));

  num_data_ = train_data_->num_data();
  // create buffer for gradients and Hessians
  if (objective_function_ != nullptr) {
    size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
    gradients_.resize(total_size);
    hessians_.resize(total_size);
  }
  // get max feature index
  max_feature_idx_ = train_data_->num_total_features() - 1;
  // get label index
  label_idx_ = train_data_->label_idx();
  // get feature names
  feature_names_ = train_data_->feature_names();
  feature_infos_ = train_data_->feature_infos();
  monotone_constraints_ = config->monotone_constraints;

  // if need bagging, create buffer
  ResetBaggingConfig(config_.get(), true);

  class_need_train_ = std::vector<bool>(num_tree_per_iteration_, true);
  if (objective_function_ != nullptr && objective_function_->SkipEmptyClass()) {
    CHECK_EQ(num_tree_per_iteration_, num_class_);
    for (int i = 0; i < num_class_; ++i) {
      class_need_train_[i] = objective_function_->ClassNeedTrain(i);
    }
  }

  if (config_->linear_tree) {
    linear_tree_ = true;
  }
}

void GBDT::AddValidDataset(const Dataset* valid_data,
                           const std::vector<const Metric*>& valid_metrics) {
  if (!train_data_->CheckAlign(*valid_data)) {
    Log::Fatal("Cannot add validation data, since it has different bin mappers with training data");
  }
  // for a validation dataset, we need its score and metric
  auto new_score_updater = std::unique_ptr<ScoreUpdater>(new ScoreUpdater(valid_data, num_tree_per_iteration_));
  // update score
  for (int i = 0; i < iter_; ++i) {
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      auto curr_tree = (i + num_init_iteration_) * num_tree_per_iteration_ + cur_tree_id;
      new_score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
    }
  }
  valid_score_updater_.push_back(std::move(new_score_updater));
  valid_metrics_.emplace_back();
  for (const auto& metric : valid_metrics) {
    valid_metrics_.back().push_back(metric);
  }
  valid_metrics_.back().shrink_to_fit();

  if (early_stopping_round_ > 0) {
    auto num_metrics = valid_metrics.size();
    if (es_first_metric_only_) { num_metrics = 1; }
    best_iter_.emplace_back(num_metrics, 0);
    best_score_.emplace_back(num_metrics, kMinScore);
    best_msg_.emplace_back(num_metrics);
  }
}

void GBDT::Boosting() {
  Common::FunctionTimer fun_timer("GBDT::Boosting", global_timer);
  if (objective_function_ == nullptr) {
    Log::Fatal("No object function provided");
  }
  // objective function will calculate gradients and hessians
  int64_t num_score = 0;
  objective_function_->
    GetGradients(GetTrainingScore(&num_score), gradients_.data(), hessians_.data());
}

data_size_t GBDT::BaggingHelper(data_size_t start, data_size_t cnt, data_size_t* buffer) {
  if (cnt <= 0) {
    return 0;
  }
  data_size_t cur_left_cnt = 0;
  data_size_t cur_right_pos = cnt;
  // random bagging, minimal unit is one record
  for (data_size_t i = 0; i < cnt; ++i) {
    auto cur_idx = start + i;
    if (bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() < config_->bagging_fraction) {
      buffer[cur_left_cnt++] = cur_idx;
    } else {
      buffer[--cur_right_pos] = cur_idx;
    }
  }
  return cur_left_cnt;
}

data_size_t GBDT::BalancedBaggingHelper(data_size_t start, data_size_t cnt,
                                        data_size_t* buffer) {
  if (cnt <= 0) {
    return 0;
  }
  auto label_ptr = train_data_->metadata().label();
  data_size_t cur_left_cnt = 0;
  data_size_t cur_right_pos = cnt;
  // random bagging, minimal unit is one record
  for (data_size_t i = 0; i < cnt; ++i) {
    auto cur_idx = start + i;
    bool is_pos = label_ptr[start + i] > 0;
    bool is_in_bag = false;
    if (is_pos) {
      is_in_bag = bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() <
                  config_->pos_bagging_fraction;
    } else {
      is_in_bag = bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() <
                  config_->neg_bagging_fraction;
    }
    if (is_in_bag) {
      buffer[cur_left_cnt++] = cur_idx;
    } else {
      buffer[--cur_right_pos] = cur_idx;
    }
  }
  return cur_left_cnt;
}

void GBDT::Bagging(int iter) {
  Common::FunctionTimer fun_timer("GBDT::Bagging", global_timer);
  // if need bagging
  if ((bag_data_cnt_ < num_data_ && iter % config_->bagging_freq == 0) ||
      need_re_bagging_) {
    need_re_bagging_ = false;
    auto left_cnt = bagging_runner_.Run<true>(
        num_data_,
        [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t* left,
            data_size_t*) {
          data_size_t cur_left_count = 0;
          if (balanced_bagging_) {
            cur_left_count =
                BalancedBaggingHelper(cur_start, cur_cnt, left);
          } else {
            cur_left_count = BaggingHelper(cur_start, cur_cnt, left);
          }
          return cur_left_count;
        },
        bag_data_indices_.data());
    bag_data_cnt_ = left_cnt;
    Log::Debug("Re-bagging, using %d data to train", bag_data_cnt_);
    // set bagging data to tree learner
    if (!is_use_subset_) {
      tree_learner_->SetBaggingData(nullptr, bag_data_indices_.data(), bag_data_cnt_);
    } else {
      // get subset
      tmp_subset_->ReSize(bag_data_cnt_);
      tmp_subset_->CopySubrow(train_data_, bag_data_indices_.data(),
                              bag_data_cnt_, false);
      tree_learner_->SetBaggingData(tmp_subset_.get(), bag_data_indices_.data(),
                                    bag_data_cnt_);
    }
  }
}

void GBDT::Train(int snapshot_freq, const std::string& model_output_path) {
  Common::FunctionTimer fun_timer("GBDT::Train", global_timer);
  bool is_finished = false;
  auto start_time = std::chrono::steady_clock::now();
  for (int iter = 0; iter < config_->num_iterations && !is_finished; ++iter) {
    is_finished = TrainOneIter(nullptr, nullptr);
    if (!is_finished) {
      is_finished = EvalAndCheckEarlyStopping();
    }
    auto end_time = std::chrono::steady_clock::now();
    // output used time per iteration
    Log::Info("%f seconds elapsed, finished iteration %d", std::chrono::duration<double,
              std::milli>(end_time - start_time) * 1e-3, iter + 1);
    if (snapshot_freq > 0
        && (iter + 1) % snapshot_freq == 0) {
      std::string snapshot_out = model_output_path + ".snapshot_iter_" + std::to_string(iter + 1);
      SaveModelToFile(0, -1, config_->saved_feature_importance_type, snapshot_out.c_str());
    }
  }
}

void GBDT::RefitTree(const std::vector<std::vector<int>>& tree_leaf_prediction) {
  CHECK_GT(tree_leaf_prediction.size(), 0);
  CHECK_EQ(static_cast<size_t>(num_data_), tree_leaf_prediction.size());
  CHECK_EQ(static_cast<size_t>(models_.size()), tree_leaf_prediction[0].size());
  int num_iterations = static_cast<int>(models_.size() / num_tree_per_iteration_);
  std::vector<int> leaf_pred(num_data_);
  if (linear_tree_) {
    std::vector<int> max_leaves_by_thread = std::vector<int>(OMP_NUM_THREADS(), 0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(tree_leaf_prediction.size()); ++i) {
      int tid = omp_get_thread_num();
      for (size_t j = 0; j < tree_leaf_prediction[i].size(); ++j) {
        max_leaves_by_thread[tid] = std::max(max_leaves_by_thread[tid], tree_leaf_prediction[i][j]);
      }
    }
    int max_leaves = *std::max_element(max_leaves_by_thread.begin(), max_leaves_by_thread.end());
    max_leaves += 1;
    tree_learner_->InitLinear(train_data_, max_leaves);
  }
  for (int iter = 0; iter < num_iterations; ++iter) {
    Boosting();
    for (int tree_id = 0; tree_id < num_tree_per_iteration_; ++tree_id) {
      int model_index = iter * num_tree_per_iteration_ + tree_id;
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < num_data_; ++i) {
        leaf_pred[i] = tree_leaf_prediction[i][model_index];
        CHECK_LT(leaf_pred[i], models_[model_index]->num_leaves());
      }
      size_t offset = static_cast<size_t>(tree_id) * num_data_;
      auto grad = gradients_.data() + offset;
      auto hess = hessians_.data() + offset;
      auto new_tree = tree_learner_->FitByExistingTree(models_[model_index].get(), leaf_pred, grad, hess);
      train_score_updater_->AddScore(tree_learner_.get(), new_tree, tree_id);
      models_[model_index].reset(new_tree);
    }
  }
}

/* If the custom "average" is implemented it will be used in place of the label average (if enabled)
*
* An improvement to this is to have options to explicitly choose
* (i) standard average
* (ii) custom average if available
* (iii) any user defined scalar bias (e.g. using a new option "init_score" that overrides (i) and (ii) )
*
* (i) and (ii) could be selected as say "auto_init_score" = 0 or 1 etc..
*
*/
double ObtainAutomaticInitialScore(const ObjectiveFunction* fobj, int class_id) {
  double init_score = 0.0;
  if (fobj != nullptr) {
    init_score = fobj->BoostFromScore(class_id);
  }
  if (Network::num_machines() > 1) {
    init_score = Network::GlobalSyncUpByMean(init_score);
  }
  return init_score;
}

double GBDT::BoostFromAverage(int class_id, bool update_scorer) {
  Common::FunctionTimer fun_timer("GBDT::BoostFromAverage", global_timer);
  // boosting from average label; or customized "average" if implemented for the current objective
  if (models_.empty() && !train_score_updater_->has_init_score() && objective_function_ != nullptr) {
    if (config_->boost_from_average || (train_data_ != nullptr && train_data_->num_features() == 0)) {
      double init_score = ObtainAutomaticInitialScore(objective_function_, class_id);
      if (std::fabs(init_score) > kEpsilon) {
        if (update_scorer) {
          train_score_updater_->AddScore(init_score, class_id);
          for (auto& score_updater : valid_score_updater_) {
            score_updater->AddScore(init_score, class_id);
          }
        }
        Log::Info("Start training from score %lf", init_score);
        return init_score;
      }
    } else if (std::string(objective_function_->GetName()) == std::string("regression_l1")
               || std::string(objective_function_->GetName()) == std::string("quantile")
               || std::string(objective_function_->GetName()) == std::string("mape")) {
      Log::Warning("Disabling boost_from_average in %s may cause the slow convergence", objective_function_->GetName());
    }
  }
  return 0.0f;
}

bool GBDT::TrainOneIter(const score_t* gradients, const score_t* hessians) {
  Common::FunctionTimer fun_timer("GBDT::TrainOneIter", global_timer);
  std::vector<double> init_scores(num_tree_per_iteration_, 0.0);
  // boosting first
  if (gradients == nullptr || hessians == nullptr) {
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      init_scores[cur_tree_id] = BoostFromAverage(cur_tree_id, true);
    }
    Boosting();
    gradients = gradients_.data();
    hessians = hessians_.data();
  }
  // bagging logic
  Bagging(iter_);

  bool should_continue = false;
  for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
    const size_t offset = static_cast<size_t>(cur_tree_id) * num_data_;
    std::unique_ptr<Tree> new_tree(new Tree(2, false, false));
    if (class_need_train_[cur_tree_id] && train_data_->num_features() > 0) {
      auto grad = gradients + offset;
      auto hess = hessians + offset;
      // need to copy gradients for bagging subset.
      if (is_use_subset_ && bag_data_cnt_ < num_data_) {
        for (int i = 0; i < bag_data_cnt_; ++i) {
          gradients_[offset + i] = grad[bag_data_indices_[i]];
          hessians_[offset + i] = hess[bag_data_indices_[i]];
        }
        grad = gradients_.data() + offset;
        hess = hessians_.data() + offset;
      }
      bool is_first_tree = models_.size() < static_cast<size_t>(num_tree_per_iteration_);
      new_tree.reset(tree_learner_->Train(grad, hess, is_first_tree));
    }

    if (new_tree->num_leaves() > 1) {
      should_continue = true;
      auto score_ptr = train_score_updater_->score() + offset;
      auto residual_getter = [score_ptr](const label_t* label, int i) {return static_cast<double>(label[i]) - score_ptr[i]; };
      tree_learner_->RenewTreeOutput(new_tree.get(), objective_function_, residual_getter,
                                     num_data_, bag_data_indices_.data(), bag_data_cnt_);
      // shrinkage by learning rate
      new_tree->Shrinkage(shrinkage_rate_);
      // update score
      UpdateScore(new_tree.get(), cur_tree_id);
      if (std::fabs(init_scores[cur_tree_id]) > kEpsilon) {
        new_tree->AddBias(init_scores[cur_tree_id]);
      }
    } else {
      // only add default score one-time
      if (models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
        double output = 0.0;
        if (!class_need_train_[cur_tree_id]) {
          if (objective_function_ != nullptr) {
            output = objective_function_->BoostFromScore(cur_tree_id);
          }
        } else {
          output = init_scores[cur_tree_id];
        }
        new_tree->AsConstantTree(output);
        // updates scores
        train_score_updater_->AddScore(output, cur_tree_id);
        for (auto& score_updater : valid_score_updater_) {
          score_updater->AddScore(output, cur_tree_id);
        }
      }
    }
    // add model
    models_.push_back(std::move(new_tree));
  }

  if (!should_continue) {
    Log::Warning("Stopped training because there are no more leaves that meet the split requirements");
    if (models_.size() > static_cast<size_t>(num_tree_per_iteration_)) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        models_.pop_back();
      }
    }
    return true;
  }

  ++iter_;
  return false;
}

void GBDT::RollbackOneIter() {
  if (iter_ <= 0) { return; }
  // reset score
  for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
    auto curr_tree = models_.size() - num_tree_per_iteration_ + cur_tree_id;
    models_[curr_tree]->Shrinkage(-1.0);
    train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
    for (auto& score_updater : valid_score_updater_) {
      score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
    }
  }
  // remove model
  for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
    models_.pop_back();
  }
  --iter_;
}

bool GBDT::EvalAndCheckEarlyStopping() {
  bool is_met_early_stopping = false;
  // print message for metric
  auto best_msg = OutputMetric(iter_);


  is_met_early_stopping = !best_msg.empty();
  if (is_met_early_stopping) {
    Log::Info("Early stopping at iteration %d, the best iteration round is %d",
              iter_, iter_ - early_stopping_round_);
    Log::Info("Output of best iteration round:\n%s", best_msg.c_str());
    // pop last early_stopping_round_ models
    for (int i = 0; i < early_stopping_round_ * num_tree_per_iteration_; ++i) {
      models_.pop_back();
    }
  }
  return is_met_early_stopping;
}

void GBDT::UpdateScore(const Tree* tree, const int cur_tree_id) {
  Common::FunctionTimer fun_timer("GBDT::UpdateScore", global_timer);
  // update training score
  if (!is_use_subset_) {
    train_score_updater_->AddScore(tree_learner_.get(), tree, cur_tree_id);

    // we need to predict out-of-bag scores of data for boosting
    if (num_data_ - bag_data_cnt_ > 0) {
      train_score_updater_->AddScore(tree, bag_data_indices_.data() + bag_data_cnt_, num_data_ - bag_data_cnt_, cur_tree_id);
    }

  } else {
    train_score_updater_->AddScore(tree, cur_tree_id);
  }


  // update validation score
  for (auto& score_updater : valid_score_updater_) {
    score_updater->AddScore(tree, cur_tree_id);
  }
}

std::vector<double> GBDT::EvalOneMetric(const Metric* metric, const double* score) const {
  return metric->Eval(score, objective_function_);
}

std::string GBDT::OutputMetric(int iter) {
  bool need_output = (iter % config_->metric_freq) == 0;
  std::string ret = "";
  std::stringstream msg_buf;
  std::vector<std::pair<size_t, size_t>> meet_early_stopping_pairs;
  // print training metric
  if (need_output) {
    for (auto& sub_metric : training_metrics_) {
      auto name = sub_metric->GetName();
      auto scores = EvalOneMetric(sub_metric, train_score_updater_->score());
      for (size_t k = 0; k < name.size(); ++k) {
        std::stringstream tmp_buf;
        tmp_buf << "Iteration:" << iter
          << ", training " << name[k]
          << " : " << scores[k];
        Log::Info(tmp_buf.str().c_str());
        if (early_stopping_round_ > 0) {
          msg_buf << tmp_buf.str() << '\n';
        }
      }
    }
  }
  // print validation metric
  if (need_output || early_stopping_round_ > 0) {
    for (size_t i = 0; i < valid_metrics_.size(); ++i) {
      for (size_t j = 0; j < valid_metrics_[i].size(); ++j) {
        auto test_scores = EvalOneMetric(valid_metrics_[i][j], valid_score_updater_[i]->score());
        auto name = valid_metrics_[i][j]->GetName();
        for (size_t k = 0; k < name.size(); ++k) {
          std::stringstream tmp_buf;
          tmp_buf << "Iteration:" << iter
            << ", valid_" << i + 1 << " " << name[k]
            << " : " << test_scores[k];
          if (need_output) {
            Log::Info(tmp_buf.str().c_str());
          }
          if (early_stopping_round_ > 0) {
            msg_buf << tmp_buf.str() << '\n';
          }
        }
        if (es_first_metric_only_ && j > 0) { continue; }
        if (ret.empty() && early_stopping_round_ > 0) {
          auto cur_score = valid_metrics_[i][j]->factor_to_bigger_better() * test_scores.back();
          if (cur_score > best_score_[i][j]) {
            best_score_[i][j] = cur_score;
            best_iter_[i][j] = iter;
            meet_early_stopping_pairs.emplace_back(i, j);
          } else {
            if (iter - best_iter_[i][j] >= early_stopping_round_) { ret = best_msg_[i][j]; }
          }
        }
      }
    }
  }
  for (auto& pair : meet_early_stopping_pairs) {
    best_msg_[pair.first][pair.second] = msg_buf.str();
  }
  return ret;
}

/*! \brief Get eval result */
std::vector<double> GBDT::GetEvalAt(int data_idx) const {
  CHECK(data_idx >= 0 && data_idx <= static_cast<int>(valid_score_updater_.size()));
  std::vector<double> ret;
  if (data_idx == 0) {
    for (auto& sub_metric : training_metrics_) {
      auto scores = EvalOneMetric(sub_metric, train_score_updater_->score());
      for (auto score : scores) {
        ret.push_back(score);
      }
    }
  } else {
    auto used_idx = data_idx - 1;
    for (size_t j = 0; j < valid_metrics_[used_idx].size(); ++j) {
      auto test_scores = EvalOneMetric(valid_metrics_[used_idx][j], valid_score_updater_[used_idx]->score());
      for (auto score : test_scores) {
        ret.push_back(score);
      }
    }
  }
  return ret;
}

/*! \brief Get training scores result */
const double* GBDT::GetTrainingScore(int64_t* out_len) {
  *out_len = static_cast<int64_t>(train_score_updater_->num_data()) * num_class_;
  return train_score_updater_->score();
}

void GBDT::PredictContrib(const double* features, double* output) const {
  // set zero
  const int num_features = max_feature_idx_ + 1;
  std::memset(output, 0, sizeof(double) * num_tree_per_iteration_ * (num_features + 1));
  const int end_iteration_for_pred = start_iteration_for_pred_ + num_iteration_for_pred_;
  for (int i = start_iteration_for_pred_; i < end_iteration_for_pred; ++i) {
    // predict all the trees for one iteration
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      models_[i * num_tree_per_iteration_ + k]->PredictContrib(features, num_features, output + k*(num_features + 1));
    }
  }
}

void GBDT::PredictContribByMap(const std::unordered_map<int, double>& features,
                               std::vector<std::unordered_map<int, double>>* output) const {
  const int num_features = max_feature_idx_ + 1;
  const int end_iteration_for_pred = start_iteration_for_pred_ + num_iteration_for_pred_;
  for (int i = start_iteration_for_pred_; i < end_iteration_for_pred; ++i) {
    // predict all the trees for one iteration
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      models_[i * num_tree_per_iteration_ + k]->PredictContribByMap(features, num_features, &((*output)[k]));
    }
  }
}

void GBDT::GetPredictAt(int data_idx, double* out_result, int64_t* out_len) {
  CHECK(data_idx >= 0 && data_idx <= static_cast<int>(valid_score_updater_.size()));

  const double* raw_scores = nullptr;
  data_size_t num_data = 0;
  if (data_idx == 0) {
    raw_scores = GetTrainingScore(out_len);
    num_data = train_score_updater_->num_data();
  } else {
    auto used_idx = data_idx - 1;
    raw_scores = valid_score_updater_[used_idx]->score();
    num_data = valid_score_updater_[used_idx]->num_data();
    *out_len = static_cast<int64_t>(num_data) * num_class_;
  }
  if (objective_function_ != nullptr) {
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data; ++i) {
      std::vector<double> tree_pred(num_tree_per_iteration_);
      for (int j = 0; j < num_tree_per_iteration_; ++j) {
        tree_pred[j] = raw_scores[j * num_data + i];
      }
      std::vector<double> tmp_result(num_class_);
      objective_function_->ConvertOutput(tree_pred.data(), tmp_result.data());
      for (int j = 0; j < num_class_; ++j) {
        out_result[j * num_data + i] = static_cast<double>(tmp_result[j]);
      }
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data; ++i) {
      for (int j = 0; j < num_tree_per_iteration_; ++j) {
        out_result[j * num_data + i] = static_cast<double>(raw_scores[j * num_data + i]);
      }
    }
  }
}

double GBDT::GetUpperBoundValue() const {
  double max_value = 0.0;
  for (const auto &tree : models_) {
    max_value += tree->GetUpperBoundValue();
  }
  return max_value;
}

double GBDT::GetLowerBoundValue() const {
  double min_value = 0.0;
  for (const auto &tree : models_) {
    min_value += tree->GetLowerBoundValue();
  }
  return min_value;
}

void GBDT::ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                             const std::vector<const Metric*>& training_metrics) {
  if (train_data != train_data_ && !train_data_->CheckAlign(*train_data)) {
    Log::Fatal("Cannot reset training data, since new training data has different bin mappers");
  }

  objective_function_ = objective_function;
  if (objective_function_ != nullptr) {
    CHECK_EQ(num_tree_per_iteration_, objective_function_->NumModelPerIteration());
    if (objective_function_->IsRenewTreeOutput() && !config_->monotone_constraints.empty()) {
      Log::Fatal("Cannot use ``monotone_constraints`` in %s objective, please disable it.", objective_function_->GetName());
    }
  }
  is_constant_hessian_ = GetIsConstHessian(objective_function);

  // push training metrics
  training_metrics_.clear();
  for (const auto& metric : training_metrics) {
    training_metrics_.push_back(metric);
  }
  training_metrics_.shrink_to_fit();

  if (train_data != train_data_) {
    train_data_ = train_data;
    // not same training data, need reset score and others
    // create score tracker
    train_score_updater_.reset(new ScoreUpdater(train_data_, num_tree_per_iteration_));

    // update score
    for (int i = 0; i < iter_; ++i) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        auto curr_tree = (i + num_init_iteration_) * num_tree_per_iteration_ + cur_tree_id;
        train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
    }

    num_data_ = train_data_->num_data();

    // create buffer for gradients and hessians
    if (objective_function_ != nullptr) {
      size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
      gradients_.resize(total_size);
      hessians_.resize(total_size);
    }

    max_feature_idx_ = train_data_->num_total_features() - 1;
    label_idx_ = train_data_->label_idx();
    feature_names_ = train_data_->feature_names();
    feature_infos_ = train_data_->feature_infos();

    tree_learner_->ResetTrainingData(train_data, is_constant_hessian_);
    ResetBaggingConfig(config_.get(), true);
  } else {
    tree_learner_->ResetIsConstantHessian(is_constant_hessian_);
  }
}

void GBDT::ResetConfig(const Config* config) {
  auto new_config = std::unique_ptr<Config>(new Config(*config));
  if (!config->monotone_constraints.empty()) {
    CHECK_EQ(static_cast<size_t>(train_data_->num_total_features()), config->monotone_constraints.size());
  }
  if (!config->feature_contri.empty()) {
    CHECK_EQ(static_cast<size_t>(train_data_->num_total_features()), config->feature_contri.size());
  }
  if (objective_function_ != nullptr && objective_function_->IsRenewTreeOutput() && !config->monotone_constraints.empty()) {
    Log::Fatal("Cannot use ``monotone_constraints`` in %s objective, please disable it.", objective_function_->GetName());
  }
  early_stopping_round_ = new_config->early_stopping_round;
  shrinkage_rate_ = new_config->learning_rate;
  if (tree_learner_ != nullptr) {
    tree_learner_->ResetConfig(new_config.get());
  }
  if (train_data_ != nullptr) {
    ResetBaggingConfig(new_config.get(), false);
  }
  if (config_.get() != nullptr && config_->forcedsplits_filename != new_config->forcedsplits_filename) {
    // load forced_splits file
    if (!new_config->forcedsplits_filename.empty()) {
      std::ifstream forced_splits_file(
          new_config->forcedsplits_filename.c_str());
      std::stringstream buffer;
      buffer << forced_splits_file.rdbuf();
      std::string err;
      forced_splits_json_ = Json::parse(buffer.str(), &err);
      tree_learner_->SetForcedSplit(&forced_splits_json_);
    } else {
      forced_splits_json_ = Json();
      tree_learner_->SetForcedSplit(nullptr);
    }
  }
  config_.reset(new_config.release());
}

void GBDT::ResetBaggingConfig(const Config* config, bool is_change_dataset) {
  // if need bagging, create buffer
  data_size_t num_pos_data = 0;
  if (objective_function_ != nullptr) {
    num_pos_data = objective_function_->NumPositiveData();
  }
  bool balance_bagging_cond = (config->pos_bagging_fraction < 1.0 || config->neg_bagging_fraction < 1.0) && (num_pos_data > 0);
  if ((config->bagging_fraction < 1.0 || balance_bagging_cond) && config->bagging_freq > 0) {
    need_re_bagging_ = false;
    if (!is_change_dataset &&
      config_.get() != nullptr && config_->bagging_fraction == config->bagging_fraction && config_->bagging_freq == config->bagging_freq
      && config_->pos_bagging_fraction == config->pos_bagging_fraction && config_->neg_bagging_fraction == config->neg_bagging_fraction) {
      return;
    }
    if (balance_bagging_cond) {
      balanced_bagging_ = true;
      bag_data_cnt_ = static_cast<data_size_t>(num_pos_data * config->pos_bagging_fraction)
                      + static_cast<data_size_t>((num_data_ - num_pos_data) * config->neg_bagging_fraction);
    } else {
      bag_data_cnt_ = static_cast<data_size_t>(config->bagging_fraction * num_data_);
    }
    bag_data_indices_.resize(num_data_);
    bagging_runner_.ReSize(num_data_);
    bagging_rands_.clear();
    for (int i = 0;
         i < (num_data_ + bagging_rand_block_ - 1) / bagging_rand_block_; ++i) {
      bagging_rands_.emplace_back(config_->bagging_seed + i);
    }

    double average_bag_rate =
        (static_cast<double>(bag_data_cnt_) / num_data_) / config->bagging_freq;
    is_use_subset_ = false;
    const int group_threshold_usesubset = 100;
    if (average_bag_rate <= 0.5
        && (train_data_->num_feature_groups() < group_threshold_usesubset)) {
      if (tmp_subset_ == nullptr || is_change_dataset) {
        tmp_subset_.reset(new Dataset(bag_data_cnt_));
        tmp_subset_->CopyFeatureMapperFrom(train_data_);
      }
      is_use_subset_ = true;
      Log::Debug("Use subset for bagging");
    }

    need_re_bagging_ = true;

    if (is_use_subset_ && bag_data_cnt_ < num_data_) {
      if (objective_function_ == nullptr) {
        size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
        gradients_.resize(total_size);
        hessians_.resize(total_size);
      }
    }
  } else {
    bag_data_cnt_ = num_data_;
    bag_data_indices_.clear();
    bagging_runner_.ReSize(0);
    is_use_subset_ = false;
  }
}

}  // namespace LightGBM
