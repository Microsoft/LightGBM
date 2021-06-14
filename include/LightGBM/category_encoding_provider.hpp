/*!
  * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */
#ifndef LIGHTGBM_CATEGORY_ENCODING_PROVIDER_HPP_
#define LIGHTGBM_CATEGORY_ENCODING_PROVIDER_HPP_

#include <LightGBM/config.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/parser_base.h>

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace LightGBM {

// transform categorical features to encoded numerical values before the bin construction process
class CategoryEncodingProvider {
 public:
  class CatConverter {
   protected:
    std::unordered_map<int, int> cat_fid_to_convert_fid_;

   public:
    virtual ~CatConverter() {}

    virtual double CalcValue(const double sum_label, const double sum_count,
      const double all_fold_sum_count, const double prior) const = 0;

    virtual double CalcValue(const double sum_label, const double sum_count,
      const double all_fold_sum_count) const = 0;

    virtual std::string DumpToString() const = 0;

    virtual json11::Json DumpToJSONObject() const = 0;

    virtual std::string FeatureName() const = 0;

    virtual void SetPrior(const double /*prior*/, const double /*prior_weight*/) {}

    void SetCatFidToConvertFid(const std::unordered_map<int, int>& cat_fid_to_convert_fid) {
      cat_fid_to_convert_fid_ = cat_fid_to_convert_fid;
    }

    void RegisterConvertFid(const int cat_fid, const int convert_fid) {
      cat_fid_to_convert_fid_[cat_fid] = convert_fid;
    }

    int GetConvertFid(const int cat_fid) const {
      return cat_fid_to_convert_fid_.at(cat_fid);
    }

    static CatConverter* CreateFromCharPointer(const char* char_pointer, size_t* used_len, double prior_weight) {
      const char* char_pointer_start = char_pointer;
      size_t line_len = Common::GetLine(char_pointer);
      std::string line(char_pointer, line_len);
      char_pointer += line_len;
      char_pointer = Common::SkipNewLine(char_pointer);
      CatConverter* ret = nullptr;
      if (Common::StartsWith(line, "type=")) {
        std::string type = Common::Split(line.c_str(), "=")[1];

        if (type == std::string("target_encoder") || type == std::string("target_encoder_label_mean")) {
          line_len = Common::GetLine(char_pointer);
          line = std::string(char_pointer, line_len);
          char_pointer += line_len;
          char_pointer = Common::SkipNewLine(char_pointer);
          if (!Common::StartsWith(line.c_str(), "prior=")) {
            Log::Fatal("CatConverter model format error");
          }
          double prior = 0.0f;
          Common::Atof(Common::Split(line.c_str(), "=")[1].c_str(), &prior);
          if (type == std::string("target_encoder")) {
            ret = new TargetEncoder(prior);
          } else {
            ret = new TargetEncoderLabelMean();
          }
          ret->SetPrior(prior, prior_weight);
        } else if (type == std::string("count_encoder")) {
          ret = new CountEncoder();
        } else {
          Log::Fatal("Unknown CatConverter type %s", type.c_str());
        }

        line_len = Common::GetLine(char_pointer);
        line = std::string(char_pointer, line_len);
        char_pointer += line_len;
        char_pointer = Common::SkipNewLine(char_pointer);
        if (!Common::StartsWith(line.c_str(), "categorical_feature_index_to_encoded_feature_index=")) {
          Log::Fatal("CatConverter model format error");
        }
        std::vector<std::string> feature_index_pair = Common::Split(Common::Split(line.c_str(), "=")[1].c_str(), " ");
        ret->cat_fid_to_convert_fid_.clear();
        for (auto& pair_string : feature_index_pair) {
          std::vector<std::string> cat_fid_and_convert_fid_string = Common::Split(pair_string.c_str(), ":");
          int cat_fid = 0;
          Common::Atoi(cat_fid_and_convert_fid_string[0].c_str(), &cat_fid);
          int convert_fid = 0;
          Common::Atoi(cat_fid_and_convert_fid_string[1].c_str(), &convert_fid);
          ret->cat_fid_to_convert_fid_[cat_fid] = convert_fid;
        }
        *used_len = static_cast<size_t>(char_pointer - char_pointer_start);
      } else {
        Log::Fatal("CatConverter model format error");
      }
      return ret;
    }
  };

  class TargetEncoder: public CatConverter {
   public:
    explicit TargetEncoder(const double prior): prior_(prior) {}

    inline double CalcValue(const double sum_label, const double sum_count,
      const double /*all_fold_sum_count*/) const override {
      return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
    }

    inline double CalcValue(const double sum_label, const double sum_count,
      const double /*all_fold_sum_count*/, const double /*prior*/) const override {
      return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
    }

    void SetPrior(const double /*prior*/, const double prior_weight) override {
      prior_weight_ = prior_weight;
    }

    std::string FeatureName() const override {
      std::stringstream str_stream;
      Common::C_stringstream(str_stream);
      str_stream << "target_encoding_" << prior_;
      return str_stream.str();
    }

    json11::Json DumpToJSONObject() const override {
      json11::Json::array cat_fid_to_convert_fid_array;
      for (const auto& pair : cat_fid_to_convert_fid_) {
        cat_fid_to_convert_fid_array.emplace_back(
          json11::Json::object{
            {"cat_fid", json11::Json(pair.first)},
            {"convert_fid", json11::Json(pair.second)
          }
        });
      }

      json11::Json ret(json11::Json::object {
        {"name", json11::Json("target_encoder")},
        {"prior", json11::Json(prior_)},
        {"categorical_feature_index_to_encoded_feature_index", json11::Json(cat_fid_to_convert_fid_array)}
      });
      return ret;
    }

    std::string DumpToString() const override {
      std::stringstream str_stream;
      Common::C_stringstream(str_stream);
      str_stream << "type=target_encoder\n";
      str_stream << "prior=" << prior_ << "\n";
      str_stream << "categorical_feature_index_to_encoded_feature_index=" <<
        CommonC::UnorderedMapToString<false, false, int, int>(cat_fid_to_convert_fid_, ':', ' ') << "\n";
      return str_stream.str();
    }

   private:
    const double prior_;
    double prior_weight_;
  };

  class CountEncoder: public CatConverter {
   public:
    CountEncoder() {}

    inline double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
      const double all_fold_sum_count) const override {
      return all_fold_sum_count;
    }

    inline double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
      const double all_fold_sum_count, const double /*prior*/) const override {
      return all_fold_sum_count;
    }

    std::string FeatureName() const override {
      return std::string("count_encoding");
    }

    json11::Json DumpToJSONObject() const override {
      json11::Json::array cat_fid_to_convert_fid_array;
      for (const auto& pair : cat_fid_to_convert_fid_) {
        cat_fid_to_convert_fid_array.emplace_back(
          json11::Json::object{
            {"cat_fid", json11::Json(pair.first)},
            {"convert_fid", json11::Json(pair.second)
          }
        });
      }

      json11::Json ret(json11::Json::object {
        {"name", json11::Json("count_encoder")},
        {"categorical_feature_index_to_encoded_feature_index", json11::Json(cat_fid_to_convert_fid_array)}
      });
      return ret;
    }

    std::string DumpToString() const override {
      std::stringstream str_stream;
      Common::C_stringstream(str_stream);
      str_stream << "type=count_encoder\n";
      str_stream << "categorical_feature_index_to_encoded_feature_index=" <<
        CommonC::UnorderedMapToString<false, false, int, int>(cat_fid_to_convert_fid_, ':', ' ') << "\n";
      return str_stream.str();
    }
  };

  class TargetEncoderLabelMean: public CatConverter {
   public:
    TargetEncoderLabelMean() { prior_set_ = false; }

    void SetPrior(const double prior, const double prior_weight) override {
      prior_ = prior;
      prior_weight_ = prior_weight;
      prior_set_ = true;
    }

    inline double CalcValue(const double sum_label, const double sum_count,
      const double /*all_fold_sum_count*/) const override {
      if (!prior_set_) {
        Log::Fatal("TargetEncoderLabelMean is not ready since the prior value is not set.");
      }
      return (sum_label + prior_weight_ * prior_) / (sum_count + prior_weight_);
    }

    inline double CalcValue(const double sum_label, const double sum_count,
      const double /*all_fold_sum_count*/, const double prior) const override {
      if (!prior_set_) {
        Log::Fatal("TargetEncoderLabelMean is not ready since the prior value is not set.");
      }
      return (sum_label + prior * prior_weight_) / (sum_count + prior_weight_);
    }

    std::string FeatureName() const override {
      std::stringstream str_stream;
      Common::C_stringstream(str_stream);
      str_stream << "label_mean_prior_target_encoding_" << prior_;
      return str_stream.str();
    }

    json11::Json DumpToJSONObject() const override {
      json11::Json::array cat_fid_to_convert_fid_array;
      for (const auto& pair : cat_fid_to_convert_fid_) {
        cat_fid_to_convert_fid_array.emplace_back(
          json11::Json::object{
            {"cat_fid", json11::Json(pair.first)},
            {"convert_fid", json11::Json(pair.second)
          }
        });
      }

      json11::Json ret(json11::Json::object {
        {"name", json11::Json("target_encoder")},
        {"prior", json11::Json(prior_)},
        {"categorical_feature_index_to_encoded_feature_index", json11::Json(cat_fid_to_convert_fid_array)}
      });
      return ret;
    }

    std::string DumpToString() const override {
      std::stringstream str_stream;
      Common::C_stringstream(str_stream);
      str_stream << "type=target_encoder_label_mean\n";
      str_stream << "prior=" << prior_ << "\n";
      str_stream << "categorical_feature_index_to_encoded_feature_index=" <<
        CommonC::UnorderedMapToString<false, false, int, int>(cat_fid_to_convert_fid_, ':', ' ') << "\n";
      return str_stream.str();
    }

   private:
    double prior_;
    double prior_weight_;
    bool prior_set_;
  };

  ~CategoryEncodingProvider() {
    training_data_fold_id_.clear();
    training_data_fold_id_.shrink_to_fit();
    fold_prior_.clear();
    fold_prior_.shrink_to_fit();
    is_categorical_feature_.clear();
    is_categorical_feature_.shrink_to_fit();
    count_info_.clear();
    label_info_.clear();
    category_encoders_.clear();
    category_encoders_.shrink_to_fit();
  }

  // for file data input and accumulating statistics when sampling from file
  static CategoryEncodingProvider* CreateCategoryEncodingProvider(Config* config) {
    std::unique_ptr<CategoryEncodingProvider> category_encoding_provider(new CategoryEncodingProvider(config));
    if (category_encoding_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return category_encoding_provider.release();
    }
  }

  // for pandas/numpy array data input
  static CategoryEncodingProvider* CreateCategoryEncodingProvider(Config* config,
    const std::vector<std::function<std::vector<double>(int row_idx)>>& get_row_fun,
    const std::function<label_t(int row_idx)>& get_label_fun,
    int32_t nmat, int32_t* nrow, int32_t ncol) {
    std::unique_ptr<CategoryEncodingProvider> category_encoding_provider(new CategoryEncodingProvider(config, get_row_fun, get_label_fun, nmat, nrow, ncol));
    if (category_encoding_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return category_encoding_provider.release();
    }
  }

  // for csr sparse matrix data input
  static CategoryEncodingProvider* CreateCategoryEncodingProvider(Config* config,
    const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
    const std::function<label_t(int row_idx)>& get_label_fun,
    int64_t nrow, int64_t ncol) {
    std::unique_ptr<CategoryEncodingProvider> category_encoding_provider(new CategoryEncodingProvider(config, get_row_fun, get_label_fun, nrow, ncol));
    if (category_encoding_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return category_encoding_provider.release();
    }
  }

  // for csc sparse matrix data input
  static CategoryEncodingProvider* CreateCategoryEncodingProvider(Config* config,
    const std::vector<std::unique_ptr<CSC_RowIterator>>& csc_func,
    const std::function<label_t(int row_idx)>& get_label_fun,
    int64_t nrow, int64_t ncol) {
    std::unique_ptr<CategoryEncodingProvider> category_encoding_provider(new CategoryEncodingProvider(config, csc_func, get_label_fun, nrow, ncol));
    if (category_encoding_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return category_encoding_provider.release();
    }
  }

  void PrepareCategoryEncodingStatVectors();

  void ProcessOneLine(const std::vector<double>& one_line, double label,
    int line_idx, int thread_id, const int fold_id);

  void ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
    int line_idx, std::vector<bool>* is_feature_processed, int thread_id, const int fold_id);

  void ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
    int line_idx, std::vector<bool>* is_feature_processed, const int fold_id);

  std::string DumpToJSON() const;

  std::string DumpToString() const;

  static CategoryEncodingProvider* RecoverFromCharPointer(const char* model_char_pointer, size_t* used_len);

  static CategoryEncodingProvider* RecoverFromModelString(const std::string model_string);

  static CategoryEncodingProvider* RecoverFromJSONString(const std::string json_model_string);

  bool IsCategorical(const int fid) const {
    if (fid < num_original_features_) {
      return is_categorical_feature_[fid];
    } else {
      return false;
    }
  }

  inline int GetNumOriginalFeatures() const {
    return num_original_features_;
  }

  inline int GetNumTotalFeatures() const {
    return num_total_features_;
  }

  inline int GetNumCatConverters() const {
    return static_cast<int>(category_encoders_.size());
  }

  void IterateOverCatConverters(int fid, double fval, int line_idx,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const;

  void IterateOverCatConverters(int fid, double fval,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const;

  template <bool IS_TRAIN>
  inline void GetCategoryEncodingStatForOneCatValue(int fid, double fval, int fold_id,
    double* out_label_sum, double* out_total_count, double* out_all_fold_total_count) const {
    *out_label_sum = 0.0f;
    *out_total_count = 0.0f;
    *out_all_fold_total_count = 0.0f;
    const int int_fval = static_cast<int>(fval);
    const auto& fold_label_info = IS_TRAIN ?
      label_info_.at(fid).at(fold_id) : label_info_.at(fid).back();
    const auto& fold_count_info = IS_TRAIN ?
      count_info_.at(fid).at(fold_id) : count_info_.at(fid).back();
    if (fold_count_info.count(int_fval) > 0) {
      *out_label_sum = fold_label_info.at(int_fval);
      *out_total_count = fold_count_info.at(int_fval);
    }
    if (IS_TRAIN) {
      const auto& all_fold_count_info = count_info_.at(fid).back();
      if (all_fold_count_info.count(int_fval) > 0) {
        *out_all_fold_total_count = all_fold_count_info.at(int_fval);
      }
    } else {
      *out_all_fold_total_count = *out_total_count;
    }
  }

  template <bool IS_TRAIN>
  void IterateOverCatConvertersInner(int fid, double fval, int fold_id,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const {
    double label_sum = 0.0f, total_count = 0.0f, all_fold_total_count = 0.0f;
    GetCategoryEncodingStatForOneCatValue<IS_TRAIN>(fid, fval, fold_id, &label_sum, &total_count, &all_fold_total_count);
    for (const auto& cat_converter : category_encoders_) {
      const double convert_value = IS_TRAIN ?
        cat_converter->CalcValue(label_sum, total_count, all_fold_total_count, fold_prior_[fold_id]) :
        cat_converter->CalcValue(label_sum, total_count, all_fold_total_count);
      const int convert_fid = cat_converter->GetConvertFid(fid);
      write_func(convert_fid, fid, convert_value);
    }
    post_process_func(fid);
  }

  template <bool IS_TRAIN>
  double HandleOneCatConverter(int fid, double fval, int fold_id,
    const CategoryEncodingProvider::CatConverter* cat_converter) const {
    double label_sum = 0.0f, total_count = 0.0f, all_fold_total_count = 0.0f;
    GetCategoryEncodingStatForOneCatValue<IS_TRAIN>(fid, fval, fold_id, &label_sum, &total_count, &all_fold_total_count);
    if (IS_TRAIN) {
      return cat_converter->CalcValue(label_sum, total_count, all_fold_total_count, fold_prior_[fold_id]);
    } else {
      return cat_converter->CalcValue(label_sum, total_count, all_fold_total_count);
    }
  }

  void ConvertCatToEncodingValues(std::vector<double>* features, int line_idx) const;

  void ConvertCatToEncodingValues(std::vector<double>* features) const;

  void ConvertCatToEncodingValues(std::vector<std::pair<int, double>>* features_ptr, const int fold_id) const;

  void ConvertCatToEncodingValues(std::vector<std::pair<int, double>>* features_ptr) const;

  double ConvertCatToEncodingValues(double fval, const CategoryEncodingProvider::CatConverter* cat_converter,
    int col_idx, int line_idx) const;

  double ConvertCatToEncodingValues(double fval, const CategoryEncodingProvider::CatConverter* cat_converter,
    int col_idx) const;

  void ExtendFeatureNames(std::vector<std::string>* feature_names_ptr) const;

  template <typename INDEX_T>
  void WrapRowFunctions(
    std::vector<std::function<std::vector<double>(INDEX_T row_idx)>>* get_row_fun,
    int32_t* ncol, bool is_valid) const {
    const std::vector<std::function<std::vector<double>(INDEX_T row_idx)>> old_get_row_fun = *get_row_fun;
    get_row_fun->clear();
    for (size_t i = 0; i < old_get_row_fun.size(); ++i) {
      get_row_fun->push_back(WrapRowFunctionInner<double, INDEX_T>(&old_get_row_fun[i], is_valid));
    }
    *ncol = static_cast<int32_t>(num_total_features_);
  }

  template <typename INDEX_T>
  void WrapRowFunction(
    std::function<std::vector<std::pair<int, double>>(INDEX_T row_idx)>* get_row_fun,
    int64_t* ncol, bool is_valid) const {
    *get_row_fun = WrapRowFunctionInner<std::pair<int, double>, INDEX_T>(get_row_fun, is_valid);
    *ncol = static_cast<int64_t>(num_total_features_);
  }

  template <typename T, typename INDEX_T>
  std::function<std::vector<T>(INDEX_T row_idx)> WrapRowFunctionInner(
    const std::function<std::vector<T>(INDEX_T row_idx)>* get_row_fun, bool is_valid) const {
  std::function<std::vector<T>(INDEX_T row_idx)> old_get_row_fun = *get_row_fun;
    if (is_valid) {
      return [old_get_row_fun, this] (INDEX_T row_idx) {
        std::vector<T> row = old_get_row_fun(row_idx);
        ConvertCatToEncodingValues(&row);
        return row;
      };
    } else {
      return [old_get_row_fun, this] (INDEX_T row_idx) {
        std::vector<T> row = old_get_row_fun(row_idx);
        ConvertCatToEncodingValues(&row, row_idx);
        return row;
      };
    }
  }

  void WrapColIters(
    std::vector<std::unique_ptr<CSC_RowIterator>>* col_iters,
    int64_t* ncol_ptr, bool is_valid, int64_t num_row) const;

  Parser* FinishProcess(const int num_machines, Config* config);

  void InitFromParser(Config* config_from_loader, Parser* parser, const int num_machines,
    std::unordered_set<int>* categorical_features_from_loader);

  void AccumulateOneLineStat(const char* buffer, const size_t size, const data_size_t row_idx);

 private:
  void SetConfig(const Config* config);

  explicit CategoryEncodingProvider(const std::string model_string);

  CategoryEncodingProvider(const char* str, size_t* used_len);

  explicit CategoryEncodingProvider(Config* config);

  CategoryEncodingProvider(Config* config,
    const std::vector<std::function<std::vector<double>(int row_idx)>>& get_row_fun,
    const std::function<label_t(int row_idx)>& get_label_fun, const int32_t nmat,
    const int32_t* nrow, const int32_t ncol);

  CategoryEncodingProvider(Config* config,
    const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
    const std::function<label_t(int row_idx)>& get_label_fun,
    const int64_t nrow, const int64_t ncol);

  CategoryEncodingProvider(Config* config,
    const std::vector<std::unique_ptr<CSC_RowIterator>>& csc_iters,
    const std::function<label_t(int row_idx)>& get_label_fun,
    const int64_t nrow, const int64_t ncol);

  template <bool ACCUMULATE_FROM_FILE>
  void ProcessOneLineInner(const std::vector<std::pair<int, double>>& one_line,
    double label, int line_idx, std::vector<bool>* is_feature_processed_ptr,
    std::unordered_map<int, std::vector<std::unordered_map<int, int>>>* count_info_ptr,
    std::unordered_map<int, std::vector<std::unordered_map<int, double>>>* label_info_ptr,
    std::vector<double>* label_sum_ptr, std::vector<int>* num_data_ptr, const int fold_id);

  // sync up encoding values by gathering statistics from all machines in distributed scenario
  void SyncEncodingStat(std::vector<std::unordered_map<int, double>>* fold_label_sum_ptr,
    std::vector<std::unordered_map<int, int>>* fold_total_count_ptr, const int num_machines) const;

  // sync up statistics to calculate the encoding prior by gathering statistics from all machines in distributed scenario
  void SyncEncodingPrior(const double label_sum, const int local_num_data, double* all_label_sum_ptr,
    int* all_num_data_ptr, int num_machines) const;

  int ParseMetaInfo(const char* filename, Config* config);

  void ExpandNumFeatureWhileAccumulate(const int new_largest_fid);

  inline void AddCountAndLabel(std::unordered_map<int, int>* count_map,
    std::unordered_map<int, double>* label_map,
    const int cat_value, const int count_value, const label_t label_value) {
    if (count_map->count(cat_value) == 0) {
      count_map->operator[](cat_value) = count_value;
      label_map->operator[](cat_value) = static_cast<double>(label_value);
    } else {
      count_map->operator[](cat_value) += count_value;
      label_map->operator[](cat_value) += static_cast<double>(label_value);
    }
  }

  // parameter configuration
  Config config_;

  // size of training data
  data_size_t num_data_;
  // list of categorical feature indices (real index, not inner index of Dataset)
  std::vector<int> categorical_features_;

  // maps training data index to fold index
  std::vector<int> training_data_fold_id_;
  // prior used by per fold
  std::vector<double> fold_prior_;
  // weight of the prior in category encoding calculation
  double prior_weight_;
  // record whether a feature is categorical in the original data
  std::vector<bool> is_categorical_feature_;

  // number of features in the original dataset, without adding count features
  int num_original_features_;
  // number of features after converting categorical features
  int num_total_features_;

  // number of threads used for category encoding
  int num_threads_;

  // the accumulated count information for category encoding
  std::unordered_map<int, std::vector<std::unordered_map<int, int>>> count_info_;
  // the accumulated label sum information for category encoding
  std::unordered_map<int, std::vector<std::unordered_map<int, double>>> label_info_;
  // the accumulated count information for category encoding per thread
  std::vector<std::unordered_map<int, std::vector<std::unordered_map<int, int>>>> thread_count_info_;
  // the accumulated label sum information for category encoding per thread
  std::vector<std::unordered_map<int, std::vector<std::unordered_map<int, double>>>> thread_label_info_;
  // the accumulated label sum per fold
  std::vector<double> fold_label_sum_;
  // the accumulated label sum per thread per fold
  std::vector<std::vector<double>> thread_fold_label_sum_;
  // the accumulated number of data per fold per thread
  std::vector<std::vector<data_size_t>> thread_fold_num_data_;
  // number of data per fold
  std::vector<data_size_t> fold_num_data_;
  // categorical value converters
  std::vector<std::unique_ptr<CatConverter>> category_encoders_;
  // whether the old categorical handling method is used
  bool keep_raw_cat_method_;

  // temporary parser used when accumulating statistics from file
  std::unique_ptr<Parser> tmp_parser_;
  // temporary oneline_features used when accumulating statistics from file
  std::vector<std::pair<int, double>> tmp_oneline_features_;
  // temporary random generator used when accumulating statistics from file,
  // used to generate training data folds for category encoding calculations
  std::mt19937 tmp_mt_generator_;
  // temporary fold distribution probability when accumulating statistics from file
  std::vector<double> tmp_fold_probs_;
  // temporary fold distribution when accumulating statistics from file
  std::discrete_distribution<int> tmp_fold_distribution_;
  // temporary feature read mask when accumulating statistics from files
  std::vector<bool> tmp_is_feature_processed_;
  // mark whether the category encoding statistics is accumulated from file
  bool accumulated_from_file_;
};

class CategoryEncodingParser : public Parser {
 public:
    explicit CategoryEncodingParser(const Parser* inner_parser,
      const CategoryEncodingProvider* category_encoding_provider, const bool is_valid):
      inner_parser_(inner_parser), category_encoding_provider_(category_encoding_provider), is_valid_(is_valid) {}

    inline void ParseOneLine(const char* str,
      std::vector<std::pair<int, double>>* out_features,
      double* out_label, const int line_idx = -1) const override {
      inner_parser_->ParseOneLine(str, out_features, out_label);
      if (is_valid_) {
        category_encoding_provider_->ConvertCatToEncodingValues(out_features);
      } else {
        category_encoding_provider_->ConvertCatToEncodingValues(out_features, line_idx);
      }
    }

    inline int NumFeatures() const override {
      return category_encoding_provider_->GetNumTotalFeatures();
    }

 private:
    std::unique_ptr<const Parser> inner_parser_;
    const CategoryEncodingProvider* category_encoding_provider_;
    const bool is_valid_;
};

class Category_Encoding_CSC_RowIterator: public CSC_RowIterator {
 public:
  Category_Encoding_CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx,
                  const CategoryEncodingProvider::CatConverter* cat_converter,
                  const CategoryEncodingProvider* category_encoding_provider, bool is_valid, int64_t num_row):
    CSC_RowIterator(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, col_idx),
    col_idx_(col_idx),
    is_valid_(is_valid),
    num_row_(num_row) {
    cat_converter_ = cat_converter;
    category_encoding_provider_ = category_encoding_provider;
  }

  Category_Encoding_CSC_RowIterator(CSC_RowIterator* csc_iter, const int col_idx,
                  const CategoryEncodingProvider::CatConverter* cat_converter,
                  const CategoryEncodingProvider* category_encoding_provider, bool is_valid, int64_t num_row):
    CSC_RowIterator(*csc_iter),
    col_idx_(col_idx),
    is_valid_(is_valid),
    num_row_(num_row) {
    cat_converter_ = cat_converter;
    category_encoding_provider_ = category_encoding_provider;
  }

  double Get(int row_idx) override {
    const double value = CSC_RowIterator::Get(row_idx);
    if (is_valid_) {
      return category_encoding_provider_->ConvertCatToEncodingValues(value, cat_converter_, col_idx_);
    } else {
      return category_encoding_provider_->ConvertCatToEncodingValues(value, cat_converter_, col_idx_, row_idx);
    }
  }

  std::pair<int, double> NextNonZero() override {
    if (cur_row_idx_ + 1 < static_cast<int>(num_row_)) {
      auto pair = cached_pair_;
      if (cur_row_idx_ == cached_pair_.first) {
        cached_pair_ = CSC_RowIterator::NextNonZero();
      }
      if ((cur_row_idx_ + 1 < cached_pair_.first) || is_end_) {
        pair = std::make_pair(cur_row_idx_ + 1, 0.0f);
      } else {
        pair = cached_pair_;
      }
      ++cur_row_idx_;
      double value = 0.0f;
      if (is_valid_) {
        value = category_encoding_provider_->ConvertCatToEncodingValues(pair.second, cat_converter_, col_idx_);
      } else {
        value = category_encoding_provider_->ConvertCatToEncodingValues(pair.second, cat_converter_, col_idx_, pair.first);
      }
      pair.second = value;
      return pair;
    } else {
      return std::make_pair(-1, 0.0f);
    }
  }

  void Reset() override {
    CSC_RowIterator::Reset();
    cur_row_idx_ = -1;
    cached_pair_ = std::make_pair(-1, 0.0f);
  }

 private:
  const CategoryEncodingProvider::CatConverter* cat_converter_;
  const CategoryEncodingProvider* category_encoding_provider_;
  const int col_idx_;
  const bool is_valid_;
  const int64_t num_row_;

  int cur_row_idx_ = -1;
  std::pair<int, double> cached_pair_ = std::make_pair(-1, 0.0f);
};

}  // namespace LightGBM
#endif  // LightGBM_CATEGORY_ENCODING_PROVIDER_HPP_
