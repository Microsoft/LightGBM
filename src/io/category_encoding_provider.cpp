/*!
  * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */

#include <LightGBM/category_encoding_provider.hpp>

#include <set>

namespace LightGBM {

CategoryEncodingProvider::CategoryEncodingProvider(Config* config) {
  accumulated_from_file_ = true;
  SetConfig(config);
}

CategoryEncodingProvider::CategoryEncodingProvider(Config* config,
  const std::vector<std::function<std::vector<double>(int row_idx)>>& get_row_fun,
  const std::function<label_t(int row_idx)>& get_label_fun, const int32_t nmat,
  const int32_t* nrow, const int32_t ncol) {
  accumulated_from_file_ = false;
  SetConfig(config);
  num_original_features_ = ncol;
  ParseMetaInfo(nullptr, config);
  PrepareCategoryEncodingStatVectors();
  if (category_encoders_.size() == 0) { return; }
  if (get_label_fun == nullptr) {
    Log::Fatal("Please specify the label before the dataset is constructed to use category encoding");
  }
  int32_t mat_offset = 0;
  num_data_ = 0;
  for (int32_t i = 0; i < nmat; ++i) {
    num_data_ += nrow[i];
  }
  std::vector<std::mt19937> mt_generators;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    mt_generators.emplace_back(config_.seed + thread_id);
  }
  const std::vector<double> fold_probs(config_.num_target_encoding_folds, 1.0 / config_.num_target_encoding_folds);
  std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
  training_data_fold_id_.resize(num_data_, 0);
  for (int32_t i_mat = 0; i_mat < nmat; ++i_mat) {
    const int32_t mat_nrow = nrow[i_mat];
    const auto& mat_get_row_fun = get_row_fun[i_mat];
    Threading::For<int32_t>(0, mat_nrow, 1024,
      [this, &mat_get_row_fun, &get_label_fun, &mat_offset, &fold_distribution, &mt_generators]
      (int thread_id, int32_t start, int32_t end) {
      for (int32_t j = start; j < end; ++j) {
        const std::vector<double>& oneline_features = mat_get_row_fun(j);
        const int32_t row_idx = j + mat_offset;
        const double label = static_cast<double>(get_label_fun(row_idx));
        const int fold_id = fold_distribution(mt_generators[thread_id]);
        training_data_fold_id_[row_idx] = fold_id;
        ProcessOneLine(oneline_features, label, row_idx, thread_id, fold_id);
      }
    });
    mat_offset += mat_nrow;
  }
  FinishProcess(1, config);
}


CategoryEncodingProvider::CategoryEncodingProvider(Config* config,
  const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
  const std::function<label_t(int row_idx)>& get_label_fun,
  const int64_t nrow, const int64_t ncol) {
  accumulated_from_file_ = false;
  SetConfig(config);
  num_original_features_ = ncol;
  ParseMetaInfo(nullptr, config);
  num_data_ = nrow;
  training_data_fold_id_.resize(num_data_);
  PrepareCategoryEncodingStatVectors();
  if (category_encoders_.size() == 0) { return; }
  if (get_label_fun == nullptr) {
    Log::Fatal("Please specify the label before the dataset is constructed to use category encoding");
  }
  std::vector<std::mt19937> mt_generators;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    mt_generators.emplace_back(config_.seed + thread_id);
  }
  const std::vector<double> fold_probs(config_.num_target_encoding_folds, 1.0 / config_.num_target_encoding_folds);
  std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
  std::vector<std::vector<bool>> is_feature_processed(num_threads_);
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    is_feature_processed[thread_id].resize(num_original_features_, false);
  }
  Threading::For<int64_t>(0, nrow, 1024,
  [this, &get_row_fun, &get_label_fun, &fold_distribution, &mt_generators, &is_feature_processed]
  (int thread_id, int64_t start, int64_t end) {
    for (int64_t j = start; j < end; ++j) {
      const std::vector<std::pair<int, double>>& oneline_features = get_row_fun(j);
      const int32_t row_idx = j;
      const double label = static_cast<double>(get_label_fun(row_idx));
      const int fold_id = fold_distribution(mt_generators[thread_id]);
      training_data_fold_id_[row_idx] = fold_id;
      ProcessOneLine(oneline_features, label, row_idx, &is_feature_processed[thread_id], thread_id, fold_id);
    }
  });
  FinishProcess(1, config);
}

CategoryEncodingProvider::CategoryEncodingProvider(Config* config,
  const std::vector<std::unique_ptr<CSC_RowIterator>>& csc_iters,
  const std::function<label_t(int row_idx)>& get_label_fun,
  const int64_t nrow, const int64_t ncol) {
  accumulated_from_file_ = false;
  SetConfig(config);
  num_original_features_ = ncol;
  ParseMetaInfo(nullptr, config);
  num_data_ = nrow;
  training_data_fold_id_.resize(num_data_);
  PrepareCategoryEncodingStatVectors();
  if (category_encoders_.size() == 0) { return; }
  if (get_label_fun == nullptr) {
    Log::Fatal("Please specify the label before the dataset is constructed to use category encoding");
  }
  std::vector<std::mt19937> mt_generators;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    mt_generators.emplace_back(config_.seed + thread_id);
  }
  const std::vector<double> fold_probs(config_.num_target_encoding_folds, 1.0 / config_.num_target_encoding_folds);
  std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
  std::vector<std::vector<std::unique_ptr<CSC_RowIterator>>> thread_csc_iters(num_threads_);
  // copy csc row iterators for each thread
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    for (size_t i = 0; i < csc_iters.size(); ++i) {
      thread_csc_iters[thread_id].emplace_back(new CSC_RowIterator(*csc_iters[i].get()));
    }
  }
  Threading::For<int32_t>(0, nrow, 1024,
    [this, &thread_csc_iters, &get_label_fun, ncol, &fold_distribution, &mt_generators]
    (int thread_id, int32_t start, int32_t end) {
    std::vector<double> oneline_features(ncol, 0.0f);
    for (int32_t row_idx = start; row_idx < end; ++row_idx) {
      for (int32_t col_idx = 0; col_idx < ncol; ++col_idx) {
        oneline_features[col_idx] = thread_csc_iters[thread_id][col_idx]->Get(row_idx);
      }
      const int fold_id = fold_distribution(mt_generators[thread_id]);
      training_data_fold_id_[row_idx] = fold_id;
      const double label = static_cast<double>(get_label_fun(row_idx));
      ProcessOneLine(oneline_features, label, row_idx, thread_id, fold_id);
    }
  });
  FinishProcess(1, config);
}

void CategoryEncodingProvider::SetConfig(const Config* config) {
  config_ = *config;
  num_threads_ = config_.num_threads > 0 ? config_.num_threads : OMP_NUM_THREADS();
  keep_raw_cat_method_ = false;
  const std::string target_encoding_string = std::string("target");
  category_encoders_.clear();
  if (config_.category_encoders.size() > 0) {
    for (auto token : Common::Split(config_.category_encoders.c_str(), ',')) {
      if (Common::StartsWith(token, "target")) {
        if (token.size() == target_encoding_string.size()) {
          category_encoders_.emplace_back(new CategoryEncodingProvider::TargetEncoderLabelMean());
        } else {
          double prior = 0.0f;
          if (token[target_encoding_string.size()] != ':' ||
              !Common::AtofAndCheck(token.c_str() + target_encoding_string.size() + 1, &prior)) {
            Log::Fatal("Target encoding prior of cat_converter specification %s is not a valid float value.", token.c_str());
          }
          category_encoders_.emplace_back(new CategoryEncodingProvider::TargetEncoder(prior));
        }
      } else if (token == std::string("count")) {
        category_encoders_.emplace_back(new CategoryEncodingProvider::CountEncoder());
      } else if (token == std::string("raw")) {
        keep_raw_cat_method_ = true;
      } else {
        Log::Fatal("Unknown category_encoders specification %s.", token.c_str());
      }
    }
  }

  prior_weight_ = config_.prior_weight;
  tmp_parser_ = nullptr;
}

std::string CategoryEncodingProvider::DumpToJSON() const {
  if (category_encoders_.size() > 0) {
    json11::Json::object json_map;
    json_map["keep_raw_cat_method"] = json11::Json(static_cast<int>(keep_raw_cat_method_));
    json_map["num_original_features"] = json11::Json(num_original_features_);
    json_map["num_total_features"] = json11::Json(num_total_features_);
    json_map["prior_weight"] = json11::Json(prior_weight_);
    json_map["num_categorical_features"] = json11::Json(static_cast<int>(categorical_features_.size()));

    json11::Json::array category_infos;
    for (const int cat_fid : categorical_features_) {
      json11::Json::array label_and_count_info;
      const auto& feature_label_info = label_info_.at(cat_fid).back();
      const auto& feature_count_info = count_info_.at(cat_fid).back();
      for (const auto& pair : feature_label_info) {
        const int feature_value = pair.first;
        const double label_sum = pair.second;
        const int count = feature_count_info.at(feature_value);
        label_and_count_info.emplace_back(json11::Json::object {
          {"categorical_feature_value", json11::Json(feature_value)},
          {"label_sum", json11::Json(label_sum)},
          {"count", json11::Json(count)}
        });
      }
      category_infos.emplace_back(
        json11::Json::object {
          {"categorical_feature_index", json11::Json(cat_fid)},
          {"label_and_count_info", json11::Json(label_and_count_info)}
        });
    }
    json_map["category_infos"] = json11::Json(category_infos);
    json_map["num_categorical_encoders"] = json11::Json(static_cast<int>(category_encoders_.size()));

    json11::Json::array category_encoders;
    for (const auto& category_encoder : category_encoders_) {
      category_encoders.push_back(category_encoder->DumpToJSONObject());
    }
    json_map["category_encoders"] = json11::Json(category_encoders);
    auto ret = json11::Json(json_map).dump();
    Log::Warning(ret.c_str());
    return ret;
  } else {
    return json11::Json().dump();
  }
}

std::string CategoryEncodingProvider::DumpToString() const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  if (category_encoders_.size() > 0) {
    str_buf << "keep_raw_cat_method=" << static_cast<int>(keep_raw_cat_method_) << "\n";
    str_buf << "num_original_features=" << num_original_features_ << "\n";
    str_buf << "num_total_features=" << num_total_features_ << "\n";
    str_buf << "prior_weight=" << prior_weight_ << "\n";
    str_buf << "num_categorical_features=" << categorical_features_.size() << "\n";
    for (const int cat_fid : categorical_features_) {
      str_buf << "categorical_feature_index=" << cat_fid << "\n";
      const auto& feature_label_info = label_info_.at(cat_fid).back();
      const auto& feature_count_info = count_info_.at(cat_fid).back();
      str_buf << "num_categorical_feature_values=" << feature_label_info.size() << "\n";
      str_buf << "categorical_feature_info=";
      for (auto iter = feature_label_info.begin(); iter != feature_label_info.end(); ++iter) {
        if (iter != feature_label_info.begin()) {
          str_buf << " ";
        }
        const int categorical_feature_value = iter->first;
        str_buf << categorical_feature_value << ":" << iter->second << ":" << feature_count_info.at(categorical_feature_value);
      }
      str_buf << "\n";
    }
    str_buf << "num_category_encoders=" << category_encoders_.size() << "\n";
    for (size_t i = 0; i < category_encoders_.size(); ++i) {
      str_buf << "category_encoder=" << i << "\n";
      str_buf << category_encoders_[i]->DumpToString();
    }
  }
  return str_buf.str();
}

CategoryEncodingProvider::CategoryEncodingProvider(const std::string model_string):
  CategoryEncodingProvider(model_string.c_str(), nullptr) {}

CategoryEncodingProvider::CategoryEncodingProvider(const char* str, size_t* used_len) {
  accumulated_from_file_ = false;
  const char* str_ptr = str;
  size_t line_len = 0;
  std::string line = std::string("");

  line_len = Common::GetLine(str_ptr);
  line = std::string(str_ptr, line_len);
  if (!Common::StartsWith(line, "keep_raw_cat_method=")) {
    Log::Fatal("CategoryEncodingProvider model format error.");
  } else {
    int8_t keep_raw_cat_method = 0;
    Common::Atoi(Common::Split(line.c_str(), "=")[1].c_str(), &keep_raw_cat_method);
    keep_raw_cat_method_ = static_cast<bool>(keep_raw_cat_method);
  }
  str_ptr += line_len;
  str_ptr = Common::SkipNewLine(str_ptr);

  line_len = Common::GetLine(str_ptr);
  line = std::string(str_ptr, line_len);
  if (!Common::StartsWith(line, "num_original_features=")) {
    Log::Fatal("CategoryEncodingProvider model format error.");
  } else {
    Common::Atoi(Common::Split(line.c_str(), "=")[1].c_str(), &num_original_features_);
  }
  str_ptr += line_len;
  str_ptr = Common::SkipNewLine(str_ptr);

  line_len = Common::GetLine(str_ptr);
  line = std::string(str_ptr, line_len);
  if (!Common::StartsWith(line, "num_total_features=")) {
    Log::Fatal("CategoryEncodingProvider model format error.");
  } else {
    Common::Atoi(Common::Split(line.c_str(), "=")[1].c_str(), &num_total_features_);
  }
  str_ptr += line_len;
  str_ptr = Common::SkipNewLine(str_ptr);

  line_len = Common::GetLine(str_ptr);
  line = std::string(str_ptr, line_len);
  if (!Common::StartsWith(line, "prior_weight=")) {
    Log::Fatal("CategoryEncodingProvider model format error.");
  } else {
    Common::Atof(Common::Split(line.c_str(), "=")[1].c_str(), &prior_weight_);
  }
  str_ptr += line_len;
  str_ptr = Common::SkipNewLine(str_ptr);

  line_len = Common::GetLine(str_ptr);
  line = std::string(str_ptr, line_len);
  if (!Common::StartsWith(line, "num_categorical_features=")) {
    Log::Fatal("CategoryEncodingProvider model format error.");
  }
  int num_categorical_features = 0;
  Common::Atoi(Common::Split(line.c_str(), "=")[1].c_str(), &num_categorical_features);
  str_ptr += line_len;
  str_ptr = Common::SkipNewLine(str_ptr);

  categorical_features_.clear();
  is_categorical_feature_.clear();
  is_categorical_feature_.resize(num_original_features_, false);
  label_info_.clear();
  count_info_.clear();
  for (int i = 0; i < num_categorical_features; ++i) {
    line_len = Common::GetLine(str_ptr);
    line = std::string(str_ptr, line_len);
    if (!Common::StartsWith(line, "categorical_feature_index=")) {
      Log::Fatal("CategoryEncodingProvider model format error.");
    }
    int categorical_feature_index = 0;
    Common::Atoi(Common::Split(line.c_str(), "=")[1].c_str(), &categorical_feature_index);
    categorical_features_.emplace_back(categorical_feature_index);
    is_categorical_feature_[categorical_feature_index] = true;
    label_info_[categorical_feature_index].resize(1);
    count_info_[categorical_feature_index].resize(1);
    std::unordered_map<int, double>& label_info_ref = label_info_[categorical_feature_index][0];
    std::unordered_map<int, int>& count_info_ref = count_info_[categorical_feature_index][0];
    str_ptr += line_len;
    str_ptr = Common::SkipNewLine(str_ptr);

    line_len = Common::GetLine(str_ptr);
    line = std::string(str_ptr, line_len);
    if (!Common::StartsWith(line, "num_categorical_feature_values=")) {
      Log::Fatal("CategoryEncodingProvider model format error.");
    }
    size_t num_categorical_feature_values = 0;
    Common::Atoi(Common::Split(line.c_str(), "=")[1].c_str(), &num_categorical_feature_values);
    str_ptr += line_len;
    str_ptr = Common::SkipNewLine(str_ptr);

    line_len = Common::GetLine(str_ptr);
    line = std::string(str_ptr, line_len);
    if (!Common::StartsWith(line, "categorical_feature_info=")) {
      Log::Fatal("CategoryEncodingProvider model format error.");
    }
    std::vector<std::string> categorical_feature_info = Common::Split(Common::Split(line.c_str(), "=")[1].c_str(), " ");
    if (categorical_feature_info.size() == num_categorical_feature_values) {
      for (const auto& triplet : categorical_feature_info) {
        std::vector<std::string> value_label_count = Common::Split(triplet.c_str(), ":");
        int value = 0;
        Common::Atoi(value_label_count[0].c_str(), &value);
        double label_sum = 0.0f;
        Common::Atof(value_label_count[1].c_str(), &label_sum);
        int count_sum = 0;
        Common::Atoi(value_label_count[2].c_str(), &count_sum);

        label_info_ref[value] = label_sum;
        count_info_ref[value] = count_sum;
      }
    } else {
      Log::Fatal("CategoryEncodingProvider model format error.");
    }
    str_ptr += line_len;
    str_ptr = Common::SkipNewLine(str_ptr);
  }

  std::sort(categorical_features_.begin(), categorical_features_.end());

  line_len = Common::GetLine(str_ptr);
  line = std::string(str_ptr, line_len);
  if (!Common::StartsWith(line, "num_category_encoders=")) {
    Log::Fatal("CategoryEncodingProvider model format error.");
  }
  int num_category_encoders = 0;
  Common::Atoi(Common::Split(line.c_str(), "=")[1].c_str(), &num_category_encoders);
  str_ptr += line_len;
  str_ptr = Common::SkipNewLine(str_ptr);

  category_encoders_.clear();
  for (int i = 0; i < num_category_encoders; ++i) {
    line_len = Common::GetLine(str_ptr);
    line = std::string(str_ptr, line_len);
    if (!Common::StartsWith(line, "category_encoder=")) {
      Log::Fatal("CategoryEncodingProvider model format error.");
    }
    int category_encoder_index = 0;
    Common::Atoi(Common::Split(line.c_str(), "=")[1].c_str(), &category_encoder_index);
    if (category_encoder_index != i) {
      Log::Fatal("CategoryEncodingProvider model format error.");
    }
    str_ptr += line_len;
    str_ptr = Common::SkipNewLine(str_ptr);

    size_t category_encoder_used_len = 0;
    category_encoders_.emplace_back(CatConverter::CreateFromCharPointer(str_ptr, &category_encoder_used_len, prior_weight_));
    str_ptr += category_encoder_used_len;
    if (used_len != nullptr) {
      *used_len = static_cast<size_t>(str_ptr - str);
    }
  }
}

CategoryEncodingProvider* CategoryEncodingProvider::RecoverFromModelString(const std::string model_string) {
  if (!model_string.empty()) {
    std::unique_ptr<CategoryEncodingProvider> ret(new CategoryEncodingProvider(model_string));
    if (ret->category_encoders_.size() > 0) {
      return ret.release();
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

CategoryEncodingProvider* CategoryEncodingProvider::RecoverFromCharPointer(const char* model_char_pointer, size_t* used_len) {
  if (model_char_pointer != nullptr) {
    std::unique_ptr<CategoryEncodingProvider> ret(new CategoryEncodingProvider(model_char_pointer, used_len));
    if (ret->category_encoders_.size() > 0) {
      return ret.release();
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

void CategoryEncodingProvider::ExtendFeatureNames(std::vector<std::string>* feature_names_ptr) const {
  auto& feature_names = *feature_names_ptr;
  if (feature_names.empty()) {
    for (int i = 0; i < num_original_features_; ++i) {
      std::stringstream str_buf;
      str_buf << "Column_" << i;
      feature_names.push_back(str_buf.str());
    }
  }
  int feature_names_size = static_cast<int>(feature_names.size());
  const std::vector<std::string> old_feature_names = feature_names;
  std::vector<std::string> new_feature_names(num_total_features_);
  for (int fid = 0; fid < num_original_features_; ++fid) {
    new_feature_names[fid] = old_feature_names[fid];
    if (is_categorical_feature_[fid]) {
      for (const auto& cat_converter : category_encoders_) {
        const int convert_fid = cat_converter->GetConvertFid(fid);
        std::string cat_converter_name = cat_converter->FeatureName();
        new_feature_names[convert_fid] = old_feature_names[fid] + std::string("_") + cat_converter_name;
      }
    }
  }
  if (feature_names_size == num_original_features_) {
    feature_names = new_feature_names;
  } else if (feature_names_size == num_original_features_ +
    static_cast<int>(category_encoders_.size()) * static_cast<int>(categorical_features_.size())) {
    for (size_t i = 0; i < new_feature_names.size(); ++i) {
      CHECK_EQ(new_feature_names[i], feature_names[i]);
    }
  } else {
    Log::Fatal("wrong length of feature_names");
  }
}

int CategoryEncodingProvider::ParseMetaInfo(const char* filename, Config* config) {
  std::unordered_set<int> ignore_features;
  std::unordered_map<std::string, int> name2idx;
  std::string name_prefix("name:");
  int label_idx = 0;
  if (filename != nullptr) {
    TextReader<data_size_t> text_reader(filename, config->header);
    // get column names
    std::vector<std::string> feature_names;
    if (config->header) {
      std::string first_line = text_reader.first_line();
      feature_names = Common::Split(first_line.c_str(), "\t,");
    }
    // load label idx first
    if (config->label_column.size() > 0) {
      if (Common::StartsWith(config->label_column, name_prefix)) {
        std::string name = config->label_column.substr(name_prefix.size());
        label_idx = -1;
        for (int i = 0; i < static_cast<int>(feature_names.size()); ++i) {
          if (name == feature_names[i]) {
            label_idx = i;
            break;
          }
        }
        if (label_idx >= 0) {
          Log::Info("Using column %s as label", name.c_str());
        } else {
          Log::Fatal("Could not find label column %s in data file \n"
                    "or data file doesn't contain header", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(config->label_column.c_str(), &label_idx)) {
          Log::Fatal("label_column is not a number,\n"
                    "if you want to use a column name,\n"
                    "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as label", label_idx);
      }
    }

    if (!feature_names.empty()) {
      // erase label column name
      feature_names.erase(feature_names.begin() + label_idx);
      for (size_t i = 0; i < feature_names.size(); ++i) {
        name2idx[feature_names[i]] = static_cast<int>(i);
      }
    }

    // load ignore columns
    if (config->ignore_column.size() > 0) {
      if (Common::StartsWith(config->ignore_column, name_prefix)) {
        std::string names = config->ignore_column.substr(name_prefix.size());
        for (auto name : Common::Split(names.c_str(), ',')) {
          if (name2idx.count(name) > 0) {
            int tmp = name2idx[name];
            ignore_features.emplace(tmp);
          } else {
            Log::Fatal("Could not find ignore column %s in data file", name.c_str());
          }
        }
      } else {
        for (auto token : Common::Split(config->ignore_column.c_str(), ',')) {
          int tmp = 0;
          if (!Common::AtoiAndCheck(token.c_str(), &tmp)) {
            Log::Fatal("ignore_column is not a number,\n"
                      "if you want to use a column name,\n"
                      "please add the prefix \"name:\" to the column name");
          }
          ignore_features.emplace(tmp);
        }
      }
    }
  }
  std::unordered_set<int> categorical_features;
  if (config->categorical_feature.size() > 0) {
    if (Common::StartsWith(config->categorical_feature, name_prefix)) {
      std::string names = config->categorical_feature.substr(name_prefix.size());
      for (auto name : Common::Split(names.c_str(), ',')) {
        if (name2idx.count(name) > 0) {
          int tmp = name2idx[name];
          categorical_features.emplace(tmp);
        } else {
          Log::Fatal("Could not find categorical_feature %s in data file", name.c_str());
        }
      }
    } else {
      for (auto token : Common::Split(config->categorical_feature.c_str(), ',')) {
        int tmp = 0;
        if (!Common::AtoiAndCheck(token.c_str(), &tmp)) {
          Log::Fatal("categorical_feature is not a number,\n"
                    "if you want to use a column name,\n"
                    "please add the prefix \"name:\" to the column name");
        }
        categorical_features.emplace(tmp);
      }
    }
  }

  for (const int fid : ignore_features) {
    if (categorical_features.count(fid)) {
      categorical_features.erase(fid);
    }
  }

  // reset categorical features for Config
  if (config->categorical_feature.size() > 0) {
    if (!keep_raw_cat_method_) {
      config->categorical_feature.clear();
      config->categorical_feature.shrink_to_fit();
    }
  }

  categorical_features_.clear();
  for (const int fid : categorical_features) {
    if (fid < num_original_features_) {
      categorical_features_.push_back(fid);
    } else {
      Log::Warning("Categorical feature index %d is no less than the actual feature number %d, and will be ignored.",
        fid, num_original_features_);
    }
  }
  std::sort(categorical_features_.begin(), categorical_features_.end());
  is_categorical_feature_.clear();

  return label_idx;
}

void CategoryEncodingProvider::PrepareCategoryEncodingStatVectors() {
  is_categorical_feature_.clear();
  is_categorical_feature_.resize(num_original_features_, false);
  for (const int fid : categorical_features_) {
    if (fid < num_original_features_) {
      is_categorical_feature_[fid] = true;
    }
  }
  fold_prior_.resize(config_.num_target_encoding_folds + 1, 0.0f);
  if (category_encoders_.size() > 0) {
    // prepare to accumulate target encoding statistics
    fold_label_sum_.resize(config_.num_target_encoding_folds + 1, 0.0f);
    fold_num_data_.resize(config_.num_target_encoding_folds + 1, 0);
    if (!accumulated_from_file_) {
      thread_fold_label_sum_.resize(num_threads_);
      thread_fold_num_data_.resize(num_threads_);
      thread_count_info_.resize(num_threads_);
      thread_label_info_.resize(num_threads_);
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        thread_fold_label_sum_[thread_id].resize(config_.num_target_encoding_folds + 1, 0.0f);
        thread_fold_num_data_[thread_id].resize(config_.num_target_encoding_folds + 1, 0.0f);
      }
    }
    for (const int fid : categorical_features_) {
      if (fid < num_original_features_) {
        count_info_[fid].resize(config_.num_target_encoding_folds + 1);
        label_info_[fid].resize(config_.num_target_encoding_folds + 1);
        if (!accumulated_from_file_) {
          for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
            thread_count_info_[thread_id][fid].resize(config_.num_target_encoding_folds + 1);
            thread_label_info_[thread_id][fid].resize(config_.num_target_encoding_folds + 1);
          }
        }
      }
    }
  }
}

void CategoryEncodingProvider::SyncEncodingStat(std::vector<std::unordered_map<int, double>>* fold_label_sum_ptr,
    std::vector<std::unordered_map<int, int>>* fold_total_count_ptr, const int num_machines) const {
  auto& fold_label_sum = *fold_label_sum_ptr;
  auto& fold_total_count = *fold_total_count_ptr;
  if (num_machines > 1) {
    std::string target_encoding_stat_string;
    for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
      target_encoding_stat_string += CommonC::UnorderedMapToString(fold_label_sum[fold_id], ':', ' ') + "@";
      target_encoding_stat_string += CommonC::UnorderedMapToString(fold_total_count[fold_id], ':', ' ') + "@";
    }
    const size_t max_target_encoding_values_string_size = Network::GlobalSyncUpByMax(target_encoding_stat_string.size()) + 1;
    std::vector<char> input_buffer(max_target_encoding_values_string_size), output_buffer(max_target_encoding_values_string_size * num_machines);
    std::memcpy(input_buffer.data(), target_encoding_stat_string.c_str(), target_encoding_stat_string.size() * sizeof(char));
    input_buffer[target_encoding_stat_string.size()] = '\0';

    Network::Allgather(input_buffer.data(), sizeof(char) * max_target_encoding_values_string_size, output_buffer.data());

    int feature_value = 0;
    int count_value = 0;
    double label_sum = 0;

    for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
      fold_label_sum[fold_id].clear();
      fold_total_count[fold_id].clear();
    }

    size_t cur_str_pos = 0;
    int check_num_machines = 0;
    while (cur_str_pos < output_buffer.size()) {
      std::string all_target_encoding_stat_string(output_buffer.data() + cur_str_pos);
      cur_str_pos += max_target_encoding_values_string_size;
      ++check_num_machines;
      std::stringstream sin(all_target_encoding_stat_string);
      for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
        auto& this_fold_label_sum = fold_label_sum[fold_id];
        auto& this_fold_total_count = fold_total_count[fold_id];
        char ending_char = ' ';
        while (ending_char != '@') {
          CHECK_EQ(ending_char, ' ');
          sin >> feature_value;
          CHECK_EQ(sin.get(), ':');
          sin >> label_sum;
          if (this_fold_label_sum.count(feature_value) > 0) {
            this_fold_label_sum[feature_value] += label_sum;
          } else {
            this_fold_label_sum[feature_value] = label_sum;
          }
          ending_char = sin.get();
        }
        ending_char = ' ';
        while (ending_char != '@') {
          CHECK_EQ(ending_char, ' ');
          sin >> feature_value;
          CHECK_EQ(sin.get(), ':');
          sin >> count_value;
          if (this_fold_total_count.count(feature_value) > 0) {
            this_fold_total_count[feature_value] += count_value;
          } else {
            this_fold_total_count[feature_value] = count_value;
          }
          ending_char = sin.get();
        }
      }
    }
    CHECK_EQ(check_num_machines, num_machines);
    CHECK_EQ(cur_str_pos, output_buffer.size());
  }
}

void CategoryEncodingProvider::SyncEncodingPrior(const double label_sum, const int local_num_data,
  double* all_label_sum_ptr, int* all_num_data_ptr, int num_machines) const {
  if (num_machines > 1) {
    *all_label_sum_ptr = Network::GlobalSyncUpBySum(label_sum);
    *all_num_data_ptr = Network::GlobalSyncUpBySum(local_num_data);
  } else {
    *all_label_sum_ptr = label_sum;
    *all_num_data_ptr = local_num_data;
  }
}

void CategoryEncodingProvider::ProcessOneLine(const std::vector<double>& one_line, double label,
  int /*line_idx*/, const int thread_id, const int fold_id) {
  auto& count_info = thread_count_info_[thread_id];
  auto& label_info = thread_label_info_[thread_id];
  for (int fid = 0; fid < num_original_features_; ++fid) {
    if (is_categorical_feature_[fid]) {
      const int value = static_cast<int>(one_line[fid]);
      AddCountAndLabel(&count_info[fid][fold_id], &label_info[fid][fold_id],
        value, 1, static_cast<label_t>(label));
    }
  }
  thread_fold_label_sum_[thread_id][fold_id] += label;
  ++thread_fold_num_data_[thread_id][fold_id];
}

void CategoryEncodingProvider::ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label, int line_idx,
  std::vector<bool>* is_feature_processed_ptr, const int thread_id, const int fold_id) {
  ProcessOneLineInner<false>(one_line, label, line_idx, is_feature_processed_ptr, &thread_count_info_[thread_id],
    &thread_label_info_[thread_id], &thread_fold_label_sum_[thread_id], &thread_fold_num_data_[thread_id], fold_id);
}

void CategoryEncodingProvider::ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
  int line_idx, std::vector<bool>* is_feature_processed_ptr, const int fold_id) {
  ProcessOneLineInner<true>(one_line, label, line_idx, is_feature_processed_ptr,
    &count_info_, &label_info_, &fold_label_sum_, &fold_num_data_, fold_id);
}

template <bool ACCUMULATE_FROM_FILE>
void CategoryEncodingProvider::ProcessOneLineInner(const std::vector<std::pair<int, double>>& one_line,
  double label, int /*line_idx*/,
  std::vector<bool>* is_feature_processed_ptr,
  std::unordered_map<int, std::vector<std::unordered_map<int, int>>>* count_info_ptr,
  std::unordered_map<int, std::vector<std::unordered_map<int, double>>>* label_info_ptr,
  std::vector<double>* label_sum_ptr,
  std::vector<int>* num_data_ptr,
  const int fold_id) {
  auto& is_feature_processed = *is_feature_processed_ptr;
  auto& count_info = *count_info_ptr;
  auto& label_info = *label_info_ptr;
  auto& label_sum = *label_sum_ptr;
  auto& num_data = *num_data_ptr;
  for (size_t i = 0; i < is_feature_processed.size(); ++i) {
    is_feature_processed[i] = false;
  }
  ++num_data[fold_id];
  for (const auto& pair : one_line) {
    const int fid = pair.first;
    if (ACCUMULATE_FROM_FILE) {
      if (fid >= num_original_features_) {
        ExpandNumFeatureWhileAccumulate(fid);
      }
    }
    if (is_categorical_feature_[fid]) {
      is_feature_processed[fid] = true;
      const int value = static_cast<int>(pair.second);
      AddCountAndLabel(&count_info[fid][fold_id], &label_info[fid][fold_id], value, 1, static_cast<label_t>(label));
    }
  }
  // pad the missing values with zeros
  for (const int fid : categorical_features_) {
    if (!is_feature_processed[fid]) {
      AddCountAndLabel(&count_info[fid][fold_id], &label_info[fid][fold_id], 0, 1, static_cast<label_t>(label));
    }
  }
  label_sum[fold_id] += label;
}

Parser* CategoryEncodingProvider::FinishProcess(const int num_machines, Config* config_from_loader) {
  num_total_features_ = num_original_features_;

  auto categorical_features = categorical_features_;
  categorical_features_.clear();
  for (const int fid : categorical_features) {
    if (fid < num_original_features_) {
      categorical_features_.emplace_back(fid);
    } else {
      Log::Warning("Categorical feature index %d is no less than the actual feature number %d, and will be ignored.",
        fid, num_original_features_);
    }
  }

  if (categorical_features_.size() == 0) {
    category_encoders_.clear();
    Log::Warning("category_encoders is specified but no categorical feature is specified. Ignoring category_encoders.");
    if (tmp_parser_ != nullptr) {
      return tmp_parser_.release();
    } else {
      return nullptr;
    }
  }

  const bool is_max_bin_by_feature_set = !config_from_loader->max_bin_by_feature.empty();
  auto& max_bin_by_feature = config_from_loader->max_bin_by_feature;
  if (is_max_bin_by_feature_set) {
    const int max_bin_by_feature_size = static_cast<int>(max_bin_by_feature.size());
    if (max_bin_by_feature_size < num_original_features_) {
      Log::Warning("Size of max_bin_by_feature is smaller than the number of features.");
      Log::Warning("Padding the unspecified max_bin_by_feature by max_bin=%d.", config_.max_bin);
      for (int fid = max_bin_by_feature_size; fid < num_original_features_; ++fid) {
        max_bin_by_feature.emplace_back(config_.max_bin);
      }
    } else {
      Log::Warning("Size of max_bin_by_feature is larger than the number of features.");
      Log::Warning("Ignoring the extra max_bin_by_feature values.");
      max_bin_by_feature.resize(num_original_features_);
      max_bin_by_feature.shrink_to_fit();
    }
  }

  size_t append_from = 0;
  if (!keep_raw_cat_method_) {
    auto& cat_converter = category_encoders_[0];
    for (int fid : categorical_features_) {
      cat_converter->RegisterConvertFid(fid, fid);
    }
    append_from = 1;
  }
  for (size_t i = append_from; i < category_encoders_.size(); ++i) {
    auto& cat_converter = category_encoders_[i];
    for (const int& fid : categorical_features_) {
      cat_converter->RegisterConvertFid(fid, num_total_features_);
      if (is_max_bin_by_feature_set) {
        max_bin_by_feature.push_back(max_bin_by_feature[fid]);
      }
      ++num_total_features_;
    }
  }

  if (!accumulated_from_file_) {
    // gather from threads
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < static_cast<int>(categorical_features_.size()); ++i) {
      const int fid = categorical_features_[i];
      for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
        auto& feature_fold_count_info = count_info_.at(fid)[fold_id];
        auto& feature_fold_label_info = label_info_.at(fid)[fold_id];
        for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
          const auto& thread_feature_fold_count_info = thread_count_info_[thread_id].at(fid)[fold_id];
          const auto& thread_feature_fold_label_info = thread_label_info_[thread_id].at(fid)[fold_id];
          for (const auto& pair : thread_feature_fold_count_info) {
            AddCountAndLabel(&feature_fold_count_info, &feature_fold_label_info,
              pair.first, pair.second, thread_feature_fold_label_info.at(pair.first));
          }
        }
      }
    }
  }

  for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
    if (!accumulated_from_file_) {
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        fold_num_data_[fold_id] += thread_fold_num_data_[thread_id][fold_id];
      }
    }
    fold_num_data_.back() += fold_num_data_[fold_id];
  }
  for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
    if (!accumulated_from_file_) {
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        fold_label_sum_[fold_id] += thread_fold_label_sum_[thread_id][fold_id];
      }
    }
    fold_label_sum_.back() += fold_label_sum_[fold_id];
  }
  thread_count_info_.clear();
  thread_label_info_.clear();
  thread_count_info_.shrink_to_fit();
  thread_label_info_.shrink_to_fit();
  thread_fold_label_sum_.clear();
  thread_fold_label_sum_.shrink_to_fit();

  // gather from machines
  if (num_machines > 1) {
    for (size_t i = 0; i < categorical_features_.size(); ++i) {
      SyncEncodingStat(&label_info_.at(categorical_features_[i]),
        &count_info_.at(categorical_features_[i]), num_machines);
    }
    for (int fold_id = 0; fold_id < config_.num_target_encoding_folds + 1; ++fold_id) {
      const double local_label_sum = fold_label_sum_[fold_id];
      const int local_num_data = static_cast<int>(fold_num_data_[fold_id]);
      int global_num_data = 0;
      double global_label_sum = 0.0f;
      SyncEncodingPrior(local_label_sum, local_num_data,
        &global_label_sum, &global_num_data, num_machines);
    }
  }
  for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
    fold_label_sum_[fold_id] = fold_label_sum_.back() - fold_label_sum_[fold_id];
    fold_num_data_[fold_id] = fold_num_data_.back() - fold_num_data_[fold_id];
  }
  for (int fold_id = 0; fold_id < config_.num_target_encoding_folds + 1; ++fold_id) {
    fold_prior_[fold_id] = fold_label_sum_[fold_id] * 1.0f / fold_num_data_[fold_id];
  }
  // set prior for label mean target encoder
  for (size_t i = 0; i < category_encoders_.size(); ++i) {
    category_encoders_[i]->SetPrior(fold_prior_.back(), config_.prior_weight);
  }

  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < static_cast<int>(categorical_features_.size()); ++i) {
    const int fid = categorical_features_[i];
    auto& total_count_info = count_info_.at(fid).at(config_.num_target_encoding_folds);
    auto& label_count_info = label_info_.at(fid).at(config_.num_target_encoding_folds);
    // gather from folds
    for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
      const auto& fold_count_info = count_info_.at(fid).at(fold_id);
      const auto& fold_label_info = label_info_.at(fid).at(fold_id);
      for (const auto& pair : fold_count_info) {
        AddCountAndLabel(&total_count_info, &label_count_info,
          pair.first, pair.second, fold_label_info.at(pair.first));
      }
    }
    // replace "fold sum" with "total sum - fold sum", for the convenience of value calculation
    for (const auto& pair : total_count_info) {
      for (int fold_id = 0; fold_id < config_.num_target_encoding_folds; ++fold_id) {
        if (count_info_.at(fid).at(fold_id).count(pair.first) == 0) {
          count_info_[fid][fold_id][pair.first] = total_count_info.at(pair.first);
          label_info_[fid][fold_id][pair.first] = label_count_info.at(pair.first);
        } else {
          count_info_[fid][fold_id][pair.first] =
            total_count_info.at(pair.first) - count_info_[fid][fold_id][pair.first];
          label_info_[fid][fold_id][pair.first] =
            label_count_info.at(pair.first) - label_info_[fid][fold_id][pair.first];
        }
      }
    }
  }
  if (tmp_parser_ != nullptr) {
    return tmp_parser_.release();
  } else {
    return nullptr;
  }
}

void CategoryEncodingProvider::IterateOverCatConverters(int fid, double fval, int line_idx,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process) const {
  const int fold_id = training_data_fold_id_[line_idx];
  IterateOverCatConvertersInner<true>(fid, fval, fold_id, write_func, post_process);
}

void CategoryEncodingProvider::IterateOverCatConverters(int fid, double fval,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process) const {
  IterateOverCatConvertersInner<false>(fid, fval, -1, write_func, post_process);
}

void CategoryEncodingProvider::ConvertCatToEncodingValues(std::vector<double>* features, int line_idx) const {
  if (category_encoders_.size() == 0) { return; }
  auto& features_ref = *features;
  features_ref.resize(num_total_features_);
  for (const auto& pair : label_info_) {
    IterateOverCatConverters(pair.first, features_ref[pair.first], line_idx,

      [&features_ref] (int convert_fid, int, double convert_value) {
        features_ref[convert_fid] = convert_value;
      },

      [] (int) {});
  }
}

void CategoryEncodingProvider::ConvertCatToEncodingValues(std::vector<double>* features) const {
  if (category_encoders_.size() == 0) { return; }
  auto& features_ref = *features;
  features_ref.resize(num_total_features_);
  features->resize(num_total_features_);
  for (const auto& pair : label_info_) {
    IterateOverCatConverters(pair.first, features_ref[pair.first],

      [&features_ref] (int convert_fid, int, double convert_value) {
        features_ref[convert_fid] = convert_value;
      },

      [] (int) {});
  }
}

void CategoryEncodingProvider::ConvertCatToEncodingValues(std::vector<std::pair<int, double>>* features_ptr,
  const int line_idx) const {
  auto& features_ref = *features_ptr;
  std::vector<bool> feature_processed(num_original_features_, false);
  for (const int fid : categorical_features_) {
    feature_processed[fid] = false;
  }
  const size_t n_pairs = features_ref.size();
  for (size_t i = 0; i < n_pairs; ++i) {
    auto& pair = features_ref[i];
    const int fid = pair.first;
    if (is_categorical_feature_[fid]) {
      IterateOverCatConverters(fid, pair.second, line_idx,

        [&features_ref, &pair] (int convert_fid, int fid, double convert_value) {
          if (convert_fid == fid) {
            pair.second = convert_value;
          } else {
            // assert that convert_fid in this case is larger than all the original feature indices
            features_ref.emplace_back(convert_fid, convert_value);
          }
        },

        [&feature_processed] (int fid) { feature_processed[fid] =  true; });
    }
  }
  for (const int fid : categorical_features_) {
    if (!feature_processed[fid]) {
      IterateOverCatConverters(fid, 0.0f, line_idx,

        [&features_ref] (int convert_fid, int, double convert_value) {
          // assert that convert_fid in this case is larger than all the original feature indices
          features_ref.emplace_back(convert_fid, convert_value);
        },

        [] (int) {});
    }
  }
}

void CategoryEncodingProvider::ConvertCatToEncodingValues(std::vector<std::pair<int, double>>* features_ptr) const {
  auto& features_ref = *features_ptr;
  std::vector<bool> feature_processed(num_original_features_, false);
  for (const int fid : categorical_features_) {
    feature_processed[fid] = false;
  }
  const size_t n_pairs = features_ref.size();
  for (size_t i = 0; i < n_pairs; ++i) {
    auto& pair = features_ref[i];
    const int fid = pair.first;
    if (is_categorical_feature_[fid]) {
      IterateOverCatConverters(fid, pair.second,

        [&features_ref, &pair] (int convert_fid, int fid, double convert_value) {
          if (convert_fid == fid) {
            pair.second = convert_value;
          } else {
            // assert that convert_fid in this case is larger than all the original feature indices
            features_ref.emplace_back(convert_fid, convert_value);
          }
        },

        [&feature_processed] (int fid) { feature_processed[fid] = true; });
    }
  }
  for (const int fid : categorical_features_) {
    if (!feature_processed[fid]) {
      IterateOverCatConverters(fid, 0.0f,

        [&features_ref] (int convert_fid, int, double convert_value) {
          // assert that convert_fid in this case is larger than all the original feature indices
          features_ref.emplace_back(convert_fid, convert_value);
        },

        [] (int) {});
    }
  }
}

double CategoryEncodingProvider::ConvertCatToEncodingValues(double fval, const CategoryEncodingProvider::CatConverter* cat_converter,
  int col_idx, int line_idx) const {
  const int fold_id = training_data_fold_id_[line_idx];
  return HandleOneCatConverter<true>(col_idx, fval, fold_id, cat_converter);
}

double CategoryEncodingProvider::ConvertCatToEncodingValues(double fval, const CategoryEncodingProvider::CatConverter* cat_converter,
  int col_idx) const {
  return HandleOneCatConverter<false>(col_idx, fval, -1, cat_converter);
}

void CategoryEncodingProvider::WrapColIters(
  std::vector<std::unique_ptr<CSC_RowIterator>>* col_iters,
  int64_t* ncol_ptr, bool is_valid, int64_t num_row) const {
  int old_num_col = static_cast<int>(col_iters->size());
  std::vector<std::unique_ptr<CSC_RowIterator>> old_col_iters(col_iters->size());
  for (int i = 0; i < old_num_col; ++i) {
    old_col_iters[i].reset(col_iters->operator[](i).release());
  }
  col_iters->resize(num_total_features_);
  CHECK((*ncol_ptr) - 1 == old_num_col);
  for (int i = 0; i < (*ncol_ptr) - 1; ++i) {
    if (is_categorical_feature_[i]) {
      for (const auto& cat_converter : category_encoders_) {
        const int convert_fid = cat_converter->GetConvertFid(i);
        col_iters->operator[](convert_fid).reset(new Category_Encoding_CSC_RowIterator(
          old_col_iters[i].get(), i, cat_converter.get(), this, is_valid, num_row));
      }
      if (keep_raw_cat_method_) {
        col_iters->operator[](i).reset(old_col_iters[i].release());
      }
    } else {
      col_iters->operator[](i).reset(old_col_iters[i].release());
    }
  }
  *ncol_ptr = static_cast<int64_t>(col_iters->size()) + 1;
}

void CategoryEncodingProvider::InitFromParser(Config* config_from_loader, Parser* parser, const int num_machines,
  std::unordered_set<int>* categorical_features_from_loader) {
  if (category_encoders_.size() == 0) { return; }
  num_original_features_ = parser->NumFeatures();
  if (num_machines > 1) {
    num_original_features_ = Network::GlobalSyncUpByMax(num_original_features_);
  }
  categorical_features_.clear();
  auto& categorical_features_from_loader_ref = *categorical_features_from_loader;
  for (const int fid : categorical_features_from_loader_ref) {
    if (fid < num_original_features_) {
      categorical_features_.push_back(fid);
    }
  }
  std::sort(categorical_features_.begin(), categorical_features_.end());
  PrepareCategoryEncodingStatVectors();
  tmp_parser_.reset(parser);
  training_data_fold_id_.clear();
  tmp_oneline_features_.clear();
  tmp_mt_generator_ = std::mt19937(config_.seed);
  tmp_fold_probs_.resize(config_.num_target_encoding_folds, 1.0f / config_.num_target_encoding_folds);
  tmp_fold_distribution_ = std::discrete_distribution<int>(tmp_fold_probs_.begin(), tmp_fold_probs_.end());
  num_data_ = 0;
  tmp_is_feature_processed_.clear();
  tmp_is_feature_processed_.resize(num_original_features_, false);
  if (config_from_loader->categorical_feature.size() > 0) {
    if (!keep_raw_cat_method_) {
      config_from_loader->categorical_feature.clear();
      config_from_loader->categorical_feature.shrink_to_fit();
      categorical_features_from_loader_ref.clear();
    }
  }
}

void CategoryEncodingProvider::AccumulateOneLineStat(const char* buffer, const size_t size, const data_size_t row_idx) {
  tmp_oneline_features_.clear();
  std::string oneline_feature_str(buffer, size);
  double label = 0.0f;
  tmp_parser_->ParseOneLine(oneline_feature_str.data(), &tmp_oneline_features_, &label);
  const int fold_id = tmp_fold_distribution_(tmp_mt_generator_);
  training_data_fold_id_.emplace_back(fold_id);
  ++num_data_;
  ProcessOneLine(tmp_oneline_features_, label, row_idx, &tmp_is_feature_processed_, fold_id);
}

void CategoryEncodingProvider::ExpandNumFeatureWhileAccumulate(const int new_largest_fid) {
  num_original_features_ = new_largest_fid + 1;
  is_categorical_feature_.resize(num_original_features_, false);
  for (const int fid : categorical_features_) {
    if (fid < num_original_features_) {
      is_categorical_feature_[fid] = true;
      count_info_[fid].resize(config_.num_target_encoding_folds + 1);
      label_info_[fid].resize(config_.num_target_encoding_folds + 1);
    }
  }
}

}  // namespace LightGBM
