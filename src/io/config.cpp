/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/config.h>

#include <LightGBM/cuda/vector_cudahost.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/random.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace LightGBM {

void Config::KV2Map(std::unordered_map<std::string, std::vector<std::string>>* params, const char* kv) {
  std::vector<std::string> tmp_strs = Common::Split(kv, '=');
  if (tmp_strs.size() == 2 || tmp_strs.size() == 1) {
    std::string key = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[0]));
    std::string value = "";
    if (tmp_strs.size() == 2) {
      value = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[1]));
    }
    if (key.size() > 0) {
      params->operator[](key).emplace_back(value);
    }
  } else {
    Log::Warning("Unknown parameter %s", kv);
  }
}

void GetFirstValueAsInt(const std::unordered_map<std::string, std::vector<std::string>>& params, std::string key, int* out) {
  const auto pair = params.find(key);
  if (pair != params.end()) {
    auto candidate = pair->second[0].c_str();
    if (!Common::AtoiAndCheck(candidate, out)) {
      Log::Fatal("Parameter %s should be of type int, got \"%s\"", key.c_str(), candidate);
    }
  }
}

void Config::SetVerbosity(const std::unordered_map<std::string, std::vector<std::string>>& params) {
  int verbosity = 1;

  // if "verbosity" was found in params, prefer that to any other aliases
  const auto verbosity_iter = params.find("verbosity");
  if (verbosity_iter != params.end()) {
    GetFirstValueAsInt(params, "verbosity", &verbosity);
  } else {
    // if "verbose" was found in params and "verbosity" was not, use that value
    const auto verbose_iter = params.find("verbose");
    if (verbose_iter != params.end()) {
      GetFirstValueAsInt(params, "verbose", &verbosity);
    } else {
      // if "verbosity" and "verbose" were both missing from params, don't modify LightGBM's log level
      return;
    }
  }

  // otherwise, update LightGBM's log level based on the passed-in value
  if (verbosity < 0) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
  } else if (verbosity == 0) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Warning);
  } else if (verbosity == 1) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Info);
  } else {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Debug);
  }
}

void Config::KeepFirstValues(const std::unordered_map<std::string, std::vector<std::string>>& params, std::unordered_map<std::string, std::string>* out) {
  for (auto pair = params.begin(); pair != params.end(); ++pair) {
    auto name = pair->first.c_str();
    auto values = pair->second;
    out->emplace(name, values[0]);
    for (size_t i = 1; i < pair->second.size(); ++i) {
      Log::Warning("%s is set=%s, %s=%s will be ignored. Current value: %s=%s",
        name, values[0].c_str(),
        name, values[i].c_str(),
        name, values[0].c_str());
    }
  }
}

std::unordered_map<std::string, std::string> Config::Str2Map(const char* parameters) {
  std::unordered_map<std::string, std::vector<std::string>> all_params;
  std::unordered_map<std::string, std::string> params;
  auto args = Common::Split(parameters, " \t\n\r");
  for (auto arg : args) {
    KV2Map(&all_params, Common::Trim(arg).c_str());
  }
  SetVerbosity(all_params);
  KeepFirstValues(all_params, &params);
  ParameterAlias::KeyAliasTransform(&params);
  return params;
}

void GetBoostingType(const std::unordered_map<std::string, std::string>& params, std::string* boosting) {
  std::string value;
  if (Config::GetString(params, "boosting", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
    if (value == std::string("gbdt") || value == std::string("gbrt")) {
      *boosting = "gbdt";
    } else if (value == std::string("dart")) {
      *boosting = "dart";
    } else if (value == std::string("goss")) {
      *boosting = "goss";
    } else if (value == std::string("rf") || value == std::string("random_forest")) {
      *boosting = "rf";
    } else {
      Log::Fatal("Unknown boosting type %s", value.c_str());
    }
  }
}

void GetDataSampleStrategy(const std::unordered_map<std::string, std::string>& params, std::string* strategy) {
  std::string value;
  if (Config::GetString(params, "data_sample_strategy", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
    if (value == std::string("goss")) {
      *strategy = "goss";
    } else if (value == std::string("bagging")) {
      *strategy = "bagging";
    } else {
      Log::Fatal("Unknown sample strategy %s", value.c_str());
    }
  }
}

void ParseMetrics(const std::string& value, std::vector<std::string>* out_metric) {
  std::unordered_set<std::string> metric_sets;
  out_metric->clear();
  std::vector<std::string> metrics = Common::Split(value.c_str(), ',');
  for (auto& met : metrics) {
    auto type = ParseMetricAlias(met);
    if (metric_sets.count(type) <= 0) {
      out_metric->push_back(type);
      metric_sets.insert(type);
    }
  }
}

void GetObjectiveType(const std::unordered_map<std::string, std::string>& params, std::string* objective) {
  std::string value;
  if (Config::GetString(params, "objective", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
    *objective = ParseObjectiveAlias(value);
  }
}

void GetMetricType(const std::unordered_map<std::string, std::string>& params, const std::string& objective, std::vector<std::string>* metric) {
  std::string value;
  if (Config::GetString(params, "metric", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
    ParseMetrics(value, metric);
  }
  // add names of objective function if not providing metric
  if (metric->empty() && value.size() == 0) {
    ParseMetrics(objective, metric);
  }
}

void GetTaskType(const std::unordered_map<std::string, std::string>& params, TaskType* task) {
  std::string value;
  if (Config::GetString(params, "task", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
    if (value == std::string("train") || value == std::string("training")) {
      *task = TaskType::kTrain;
    } else if (value == std::string("predict") || value == std::string("prediction")
               || value == std::string("test")) {
      *task = TaskType::kPredict;
    } else if (value == std::string("convert_model")) {
      *task = TaskType::kConvertModel;
    } else if (value == std::string("refit") || value == std::string("refit_tree")) {
      *task = TaskType::KRefitTree;
    } else if (value == std::string("save_binary")) {
      *task = TaskType::kSaveBinary;
    } else {
      Log::Fatal("Unknown task type %s", value.c_str());
    }
  }
}

void GetDeviceType(const std::unordered_map<std::string, std::string>& params, std::string* device_type) {
  std::string value;
  if (Config::GetString(params, "device_type", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
    if (value == std::string("cpu")) {
      *device_type = "cpu";
    } else if (value == std::string("gpu")) {
      *device_type = "gpu";
    } else if (value == std::string("cuda")) {
      *device_type = "cuda";
    } else {
      Log::Fatal("Unknown device type %s", value.c_str());
    }
  }
}

void GetTreeLearnerType(const std::unordered_map<std::string, std::string>& params, std::string* tree_learner) {
  std::string value;
  if (Config::GetString(params, "tree_learner", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return std::tolower(c); });
    if (value == std::string("serial")) {
      *tree_learner = "serial";
    } else if (value == std::string("feature") || value == std::string("feature_parallel")) {
      *tree_learner = "feature";
    } else if (value == std::string("data") || value == std::string("data_parallel")) {
      *tree_learner = "data";
    } else if (value == std::string("voting") || value == std::string("voting_parallel")) {
      *tree_learner = "voting";
    } else {
      Log::Fatal("Unknown tree learner type %s", value.c_str());
    }
  }
}

void Config::GetAucMuWeights() {
  if (auc_mu_weights.empty()) {
    // equal weights for all classes
    auc_mu_weights_matrix = std::vector<std::vector<double>> (num_class, std::vector<double>(num_class, 1));
    for (size_t i = 0; i < static_cast<size_t>(num_class); ++i) {
      auc_mu_weights_matrix[i][i] = 0;
    }
  } else {
    auc_mu_weights_matrix = std::vector<std::vector<double>> (num_class, std::vector<double>(num_class, 0));
    if (auc_mu_weights.size() != static_cast<size_t>(num_class * num_class)) {
      Log::Fatal("auc_mu_weights must have %d elements, but found %zu", num_class * num_class, auc_mu_weights.size());
    }
    for (size_t i = 0; i < static_cast<size_t>(num_class); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(num_class); ++j) {
        if (i == j) {
          auc_mu_weights_matrix[i][j] = 0;
          if (std::fabs(auc_mu_weights[i * num_class + j]) > kZeroThreshold) {
            Log::Info("AUC-mu matrix must have zeros on diagonal. Overwriting value in position %zu of auc_mu_weights with 0.", i * num_class + j);
          }
        } else {
          if (std::fabs(auc_mu_weights[i * num_class + j]) < kZeroThreshold) {
            Log::Fatal("AUC-mu matrix must have non-zero values for non-diagonal entries. Found zero value in position %zu of auc_mu_weights.", i * num_class + j);
          }
          auc_mu_weights_matrix[i][j] = auc_mu_weights[i * num_class + j];
        }
      }
    }
  }
}

void Config::GetInteractionConstraints() {
  if (interaction_constraints == "") {
    interaction_constraints_vector = std::vector<std::vector<int>>();
  } else {
    interaction_constraints_vector = Common::StringToArrayofArrays<int>(interaction_constraints, '[', ']', ',');
  }
}

void Config::Set(const std::unordered_map<std::string, std::string>& params) {
  // generate seeds by seed.
  if (GetInt(params, "seed", &seed)) {
    Random rand(seed);
    int int_max = std::numeric_limits<int16_t>::max();
    data_random_seed = static_cast<int>(rand.NextShort(0, int_max));
    bagging_seed = static_cast<int>(rand.NextShort(0, int_max));
    drop_seed = static_cast<int>(rand.NextShort(0, int_max));
    feature_fraction_seed = static_cast<int>(rand.NextShort(0, int_max));
    objective_seed = static_cast<int>(rand.NextShort(0, int_max));
    extra_seed = static_cast<int>(rand.NextShort(0, int_max));
  }

  GetTaskType(params, &task);
  GetBoostingType(params, &boosting);
  GetDataSampleStrategy(params, &data_sample_strategy);
  GetObjectiveType(params, &objective);
  GetMetricType(params, objective, &metric);
  GetDeviceType(params, &device_type);
  if (device_type == std::string("cuda")) {
    LGBM_config_::current_device = lgbm_device_cuda;
  }
  GetTreeLearnerType(params, &tree_learner);

  GetMembersFromString(params);

  GetAucMuWeights();

  GetInteractionConstraints();

  // sort eval_at
  std::sort(eval_at.begin(), eval_at.end());

  std::vector<std::string> new_valid;
  for (size_t i = 0; i < valid.size(); ++i) {
    if (valid[i] != data) {
      // Only push the non-training data
      new_valid.push_back(valid[i]);
    } else {
      is_provide_training_metric = true;
    }
  }
  valid = new_valid;

  if ((task == TaskType::kSaveBinary) && !save_binary) {
    Log::Info("save_binary parameter set to true because task is save_binary");
    save_binary = true;
  }

  // check for conflicts
  CheckParamConflict(params);
}

bool CheckMultiClassObjective(const std::string& objective) {
  return (objective == std::string("multiclass") || objective == std::string("multiclassova"));
}

void Config::CheckParamConflict(const std::unordered_map<std::string, std::string>& params) {
  // check if objective, metric, and num_class match
  int num_class_check = num_class;
  bool objective_type_multiclass = CheckMultiClassObjective(objective) || (objective == std::string("custom") && num_class_check > 1);

  if (objective_type_multiclass) {
    if (num_class_check <= 1) {
      Log::Fatal("Number of classes should be specified and greater than 1 for multiclass training");
    }
  } else {
    if (task == TaskType::kTrain && num_class_check != 1) {
      Log::Fatal("Number of classes must be 1 for non-multiclass training");
    }
  }
  for (std::string metric_type : metric) {
    bool metric_type_multiclass = (CheckMultiClassObjective(metric_type)
                                   || metric_type == std::string("multi_logloss")
                                   || metric_type == std::string("multi_error")
                                   || metric_type == std::string("auc_mu")
                                   || (metric_type == std::string("custom") && num_class_check > 1));
    if ((objective_type_multiclass && !metric_type_multiclass)
        || (!objective_type_multiclass && metric_type_multiclass)) {
      Log::Fatal("Multiclass objective and metrics don't match");
    }
  }

  if (num_machines > 1) {
    is_parallel = true;
  } else {
    is_parallel = false;
    tree_learner = "serial";
  }

  bool is_single_tree_learner = tree_learner == std::string("serial");

  if (is_single_tree_learner) {
    is_parallel = false;
    num_machines = 1;
  }

  if (is_single_tree_learner || tree_learner == std::string("feature")) {
    is_data_based_parallel = false;
  } else if (tree_learner == std::string("data")
             || tree_learner == std::string("voting")) {
    is_data_based_parallel = true;
    if (histogram_pool_size >= 0
        && tree_learner == std::string("data")) {
      Log::Warning("Histogram LRU queue was enabled (histogram_pool_size=%f).\n"
                   "Will disable this to reduce communication costs",
                   histogram_pool_size);
      // Change pool size to -1 (no limit) when using data parallel to reduce communication costs
      histogram_pool_size = -1;
    }
  }
  if (is_data_based_parallel) {
    if (!forcedsplits_filename.empty()) {
      Log::Fatal("Don't support forcedsplits in %s tree learner",
                 tree_learner.c_str());
    }
  }

  // max_depth defaults to -1, so max_depth>0 implies "you explicitly overrode the default"
  //
  // Changing max_depth while leaving num_leaves at its default (31) can lead to 2 undesirable situations:
  //
  //   * (0 <= max_depth <= 4) it's not possible to produce a tree with 31 leaves
  //     - this block reduces num_leaves to 2^max_depth
  //   * (max_depth > 4) 31 leaves is less than a full depth-wise tree, which might lead to underfitting
  //     - this block warns about that
  // ref: https://github.com/lightgbm-org/LightGBM/issues/2898#issuecomment-1002860601
  if (max_depth > 0 && (params.count("num_leaves") == 0 || params.at("num_leaves").empty())) {
    double full_num_leaves = std::pow(2, max_depth);
    if (full_num_leaves > num_leaves) {
      Log::Warning("Provided parameters constrain tree depth (max_depth=%d) without explicitly setting 'num_leaves'. "
                   "This can lead to underfitting. To resolve this warning, pass 'num_leaves' (<=%.0f) in params. "
                   "Alternatively, pass (max_depth=-1) and just use 'num_leaves' to constrain model complexity.",
                   max_depth,
                   full_num_leaves);
    }

    if (full_num_leaves < num_leaves) {
      // Fits in an int, and is more restrictive than the current num_leaves
      num_leaves = static_cast<int>(full_num_leaves);
    }
  }
  if (device_type == std::string("gpu")) {
    // force col-wise for gpu version
    force_col_wise = true;
    force_row_wise = false;
    if (deterministic) {
      Log::Warning("Although \"deterministic\" is set, the results ran by GPU may be non-deterministic.");
    }
    if (use_quantized_grad) {
      Log::Warning("Quantized training is not supported by GPU tree learner. Switch to full precision training.");
      use_quantized_grad = false;
    }
  } else if (device_type == std::string("cuda")) {
    // force row-wise for cuda version
    force_col_wise = false;
    force_row_wise = true;
    if (deterministic) {
      Log::Warning("Although \"deterministic\" is set, the results ran by GPU may be non-deterministic.");
    }
  }
  // linear tree learner must be serial type and run on CPU device
  if (linear_tree) {
    if (device_type != std::string("cpu") && device_type != std::string("gpu")) {
      device_type = "cpu";
      Log::Warning("Linear tree learner only works with CPU and GPU. Falling back to CPU now.");
    }
    if (tree_learner != std::string("serial")) {
      tree_learner = "serial";
      Log::Warning("Linear tree learner must be serial.");
    }
    if (zero_as_missing) {
      Log::Fatal("zero_as_missing must be false when fitting linear trees.");
    }
    if (objective == std::string("regression_l1")) {
      Log::Fatal("Cannot use regression_l1 objective when fitting linear trees.");
    }
  }
  // min_data_in_leaf must be at least 2 if path smoothing is active. This is because when the split is calculated
  // the count is calculated using the proportion of hessian in the leaf which is rounded up to nearest int, so it can
  // be 1 when there is actually no data in the leaf. In rare cases this can cause a bug because with path smoothing the
  // calculated split gain can be positive even with zero gradient and hessian.
  if (path_smooth > kEpsilon && min_data_in_leaf < 2) {
    min_data_in_leaf = 2;
    Log::Warning("min_data_in_leaf has been increased to 2 because this is required when path smoothing is active.");
  }
  if (is_parallel && (monotone_constraints_method == std::string("intermediate") || monotone_constraints_method == std::string("advanced"))) {
    // In distributed mode, local node doesn't have histograms on all features, cannot perform "intermediate" monotone constraints.
    Log::Warning("Cannot use \"intermediate\" or \"advanced\" monotone constraints in distributed learning, auto set to \"basic\" method.");
    monotone_constraints_method = "basic";
  }
  if (feature_fraction_bynode != 1.0 && (monotone_constraints_method == std::string("intermediate") || monotone_constraints_method == std::string("advanced"))) {
    // "intermediate" monotone constraints need to recompute splits. If the features are sampled when computing the
    // split initially, then the sampling needs to be recorded or done once again, which is currently not supported
    Log::Warning("Cannot use \"intermediate\" or \"advanced\" monotone constraints with feature fraction different from 1, auto set monotone constraints to \"basic\" method.");
    monotone_constraints_method = "basic";
  }
  if (max_depth > 0 && monotone_penalty >= max_depth) {
    Log::Warning("Monotone penalty greater than tree depth. Monotone features won't be used.");
  }
  if (min_data_in_leaf <= 0 && min_sum_hessian_in_leaf <= kEpsilon) {
    Log::Warning(
        "Cannot set both min_data_in_leaf and min_sum_hessian_in_leaf to 0. "
        "Will set min_data_in_leaf to 1.");
    min_data_in_leaf = 1;
  }
  if (boosting == std::string("goss")) {
    boosting = std::string("gbdt");
    data_sample_strategy = std::string("goss");
    Log::Warning("Found boosting=goss. For backwards compatibility reasons, LightGBM interprets this as boosting=gbdt, data_sample_strategy=goss."
                 "To suppress this warning, set data_sample_strategy=goss instead.");
  }

  if (bagging_by_query && data_sample_strategy != std::string("bagging")) {
    Log::Warning("bagging_by_query=true is only compatible with data_sample_strategy=bagging. Setting bagging_by_query=false.");
    bagging_by_query = false;
  }
}

std::string Config::ToString() const {
  std::stringstream str_buf;
  str_buf << "[boosting: " << boosting << "]\n";
  str_buf << "[objective: " << objective << "]\n";
  str_buf << "[metric: " << Common::Join(metric, ",") << "]\n";
  str_buf << "[tree_learner: " << tree_learner << "]\n";
  str_buf << "[device_type: " << device_type << "]\n";
  str_buf << SaveMembersToString();
  return str_buf.str();
}

const std::string Config::DumpAliases() {
  auto map = Config::parameter2aliases();
  for (auto& pair : map) {
    std::sort(pair.second.begin(), pair.second.end(), SortAlias);
  }
  std::stringstream str_buf;
  str_buf << "{\n";
  bool first = true;
  for (const auto& pair : map) {
    if (first) {
      str_buf << "   \"";
      first = false;
    } else {
      str_buf << "   , \"";
    }
    str_buf << pair.first << "\": [";
    if (pair.second.size() > 0) {
      str_buf << "\"" << CommonC::Join(pair.second, "\", \"") << "\"";
    }
    str_buf << "]\n";
  }
  str_buf << "}\n";
  return str_buf.str();
}

}  // namespace LightGBM
