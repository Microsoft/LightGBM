/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>

#include <LightGBM/boosting.h>
#include <LightGBM/dataset.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/metric.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/cuda/vector_cudahost.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>

#include <string>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <utility>

#include "predictor.hpp"

namespace LightGBM {

Application::Application(int argc, char** argv) {
  LoadParameters(argc, argv);
  // set number of threads for openmp
  if (config_.num_threads > 0) {
    omp_set_num_threads(config_.num_threads);
  }
  if (config_.data.size() == 0 && config_.task != TaskType::kConvertModel) {
    Log::Fatal("No training/prediction data, application quit");
  }

  if (config_.device_type == std::string("cuda")) {
      LGBM_config_::current_device = lgbm_device_cuda;
  }
}

Application::~Application() {
  if (config_.is_parallel) {
    Network::Dispose();
  }
}

void Application::LoadParameters(int argc, char** argv) {
  std::unordered_map<std::string, std::string> params;
  for (int i = 1; i < argc; ++i) {
    Config::KV2Map(&params, argv[i]);
  }
  // check for alias
  ParameterAlias::KeyAliasTransform(&params);
  // read parameters from config file
  if (params.count("config") > 0) {
    TextReader<size_t> config_reader(params["config"].c_str(), false);
    config_reader.ReadAllLines();
    if (!config_reader.Lines().empty()) {
      for (auto& line : config_reader.Lines()) {
        // remove str after "#"
        if (line.size() > 0 && std::string::npos != line.find_first_of("#")) {
          line.erase(line.find_first_of("#"));
        }
        line = Common::Trim(line);
        if (line.size() == 0) {
          continue;
        }
        Config::KV2Map(&params, line.c_str());
      }
    } else {
      Log::Warning("Config file %s doesn't exist, will ignore",
                   params["config"].c_str());
    }
  }
  // check for alias again
  ParameterAlias::KeyAliasTransform(&params);
  // load configs
  config_.Set(params);
  Log::Info("Finished loading parameters");
}

void Application::LoadData() {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::unique_ptr<Predictor> predictor;
  // prediction is needed if using input initial model(continued train)
  PredictFunction predict_fun = nullptr;
  // need to continue training
  if (boosting_->NumberOfTotalModel() > 0 && config_.task != TaskType::KRefitTree) {
    predictor.reset(new Predictor(boosting_.get(), 0, -1, true, false, false, false, -1, -1));
    predict_fun = predictor->GetPredictFunction();
  }

  // sync up random seed for data partition
  if (config_.is_data_based_parallel) {
    config_.data_random_seed = Network::GlobalSyncUpByMin(config_.data_random_seed);
  }

  Log::Debug("Loading train file...");
  std::unique_ptr<CategoryEncodingProvider> category_encoding_provider(nullptr);
  if (!config_.category_encoders.empty()) {
    category_encoding_provider.reset(CategoryEncodingProvider::CreateCategoryEncodingProvider(&config_));
  }
  DatasetLoader dataset_loader(&config_, predict_fun,
                               config_.num_class, config_.data.c_str());
  // load Training data
  if (config_.is_data_based_parallel) {
    // load data for distributed training
    train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(),
                                                  Network::rank(), Network::num_machines(), category_encoding_provider.get()));
  } else {
    // load data for single machine
    train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(), 0, 1, category_encoding_provider.get()));
  }
  // need save binary file
  if (config_.save_binary) {
    train_data_->SaveBinaryFile(nullptr);
  }
  // create training metric
  if (config_.is_provide_training_metric) {
    for (auto metric_type : config_.metric) {
      auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_));
      if (metric == nullptr) { continue; }
      metric->Init(train_data_->metadata(), train_data_->num_data());
      train_metric_.push_back(std::move(metric));
    }
  }
  train_metric_.shrink_to_fit();

  if (!config_.metric.empty()) {
    // only when have metrics then need to construct validation data

    // Add validation data, if it exists
    for (size_t i = 0; i < config_.valid.size(); ++i) {
      Log::Debug("Loading validation file #%zu...", (i + 1));
      // add
      auto new_dataset = std::unique_ptr<Dataset>(
        dataset_loader.LoadFromFileAlignWithOtherDataset(
          config_.valid[i].c_str(),
          train_data_.get()));
      valid_datas_.push_back(std::move(new_dataset));
      // need save binary file
      if (config_.save_binary) {
        valid_datas_.back()->SaveBinaryFile(nullptr);
      }

      // add metric for validation data
      valid_metrics_.emplace_back();
      for (auto metric_type : config_.metric) {
        auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_));
        if (metric == nullptr) { continue; }
        metric->Init(valid_datas_.back()->metadata(),
                     valid_datas_.back()->num_data());
        valid_metrics_.back().push_back(std::move(metric));
      }
      valid_metrics_.back().shrink_to_fit();
    }
    valid_datas_.shrink_to_fit();
    valid_metrics_.shrink_to_fit();
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  // output used time on each iteration
  Log::Info("Finished loading data in %f seconds",
            std::chrono::duration<double, std::milli>(end_time - start_time) * 1e-3);
}

void Application::InitTrain() {
  if (config_.is_parallel) {
    // need init network
    Network::Init(config_);
    Log::Info("Finished initializing network");
    config_.feature_fraction_seed =
      Network::GlobalSyncUpByMin(config_.feature_fraction_seed);
    config_.feature_fraction =
      Network::GlobalSyncUpByMin(config_.feature_fraction);
    config_.drop_seed =
      Network::GlobalSyncUpByMin(config_.drop_seed);
  }

  // create boosting
  boosting_.reset(
    Boosting::CreateBoosting(config_.boosting,
                             config_.input_model.c_str()));
  // create objective function
  objective_fun_.reset(
    ObjectiveFunction::CreateObjectiveFunction(config_.objective,
                                               config_));
  // load training data
  LoadData();
  if (config_.task == TaskType::kSaveBinary) {
    Log::Info("Save data as binary finished, exit");
    exit(0);
  }
  // initialize the objective function
  objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
  // initialize the boosting
  boosting_->Init(&config_, train_data_.get(), objective_fun_.get(),
                  Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
  // add validation data into boosting
  for (size_t i = 0; i < valid_datas_.size(); ++i) {
    boosting_->AddValidDataset(valid_datas_[i].get(),
                               Common::ConstPtrInVectorWrapper<Metric>(valid_metrics_[i]));
    Log::Debug("Number of data points in validation set #%zu: %zu", i + 1, valid_datas_[i]->num_data());
  }
  Log::Info("Finished initializing training");
}

void Application::Train() {
  Log::Info("Started training...");
  boosting_->Train(config_.snapshot_freq, config_.output_model);
  boosting_->SaveModelToFile(0, -1, config_.saved_feature_importance_type,
                             config_.output_model.c_str());
  // convert model to if-else statement code
  if (config_.convert_model_language == std::string("cpp")) {
    boosting_->SaveModelToIfElse(-1, config_.convert_model.c_str());
  }
  Log::Info("Finished training");
}

void Application::Predict() {
  if (config_.task == TaskType::KRefitTree) {
    // create predictor
    Predictor predictor(boosting_.get(), 0, -1, false, true, false, false, 1, 1);
    predictor.Predict(config_.data.c_str(), config_.output_result.c_str(), config_.header, config_.predict_disable_shape_check,
                      config_.precise_float_parser);
    TextReader<int> result_reader(config_.output_result.c_str(), false);
    result_reader.ReadAllLines();
    std::vector<std::vector<int>> pred_leaf(result_reader.Lines().size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(result_reader.Lines().size()); ++i) {
      pred_leaf[i] = Common::StringToArray<int>(result_reader.Lines()[i], '\t');
      // Free memory
      result_reader.Lines()[i].clear();
    }
    DatasetLoader dataset_loader(&config_, nullptr,
                                 config_.num_class, config_.data.c_str());
    std::unique_ptr<CategoryEncodingProvider> category_encoding_provider(nullptr);
    if (!config_.category_encoders.empty()) {
      category_encoding_provider.reset(CategoryEncodingProvider::CreateCategoryEncodingProvider(&config_));
    }
    train_data_.reset(dataset_loader.LoadFromFile(config_.data.c_str(), 0, 1, category_encoding_provider.get()));
    train_metric_.clear();
    objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective,
                                                                    config_));
    objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
    boosting_->Init(&config_, train_data_.get(), objective_fun_.get(),
                    Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
    boosting_->RefitTree(pred_leaf);
    boosting_->SaveModelToFile(0, -1, config_.saved_feature_importance_type,
                               config_.output_model.c_str());
    Log::Info("Finished RefitTree");
  } else {
    // create predictor
    Predictor predictor(boosting_.get(), config_.start_iteration_predict, config_.num_iteration_predict, config_.predict_raw_score,
                        config_.predict_leaf_index, config_.predict_contrib,
                        config_.pred_early_stop, config_.pred_early_stop_freq,
                        config_.pred_early_stop_margin);
    predictor.Predict(config_.data.c_str(),
                      config_.output_result.c_str(), config_.header, config_.predict_disable_shape_check,
                      config_.precise_float_parser);
    Log::Info("Finished prediction");
  }
}

void Application::InitPredict() {
  boosting_.reset(
    Boosting::CreateBoosting("gbdt", config_.input_model.c_str()));
  Log::Info("Finished initializing prediction, total used %d iterations", boosting_->GetCurrentIteration());
}

void Application::ConvertModel() {
  boosting_.reset(
    Boosting::CreateBoosting(config_.boosting, config_.input_model.c_str()));
  boosting_->SaveModelToIfElse(-1, config_.convert_model.c_str());
}


}  // namespace LightGBM
