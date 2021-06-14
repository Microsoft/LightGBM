/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_PARSER_HPP_
#define LIGHTGBM_IO_PARSER_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace LightGBM {

class CSVParser: public Parser {
 public:
  explicit CSVParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features,
    double* out_label, const int /*line_idx*/ = -1) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    *out_label = 0.0f;
    while (*str != '\0') {
      str = atof_(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        offset = -1;
      } else if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        out_features->emplace_back(idx + offset, val);
      }
      ++idx;
      if (*str == ',') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as CSV");
      }
    }
  }

  inline int NumFeatures() const override {
    return total_columns_ - (label_idx_ >= 0);
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class TSVParser: public Parser {
 public:
  explicit TSVParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features,
    double* out_label, const int /*line_idx*/ = -1) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    while (*str != '\0') {
      str = atof_(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        offset = -1;
      } else if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        out_features->emplace_back(idx + offset, val);
      }
      ++idx;
      if (*str == '\t') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as TSV");
      }
    }
  }

  inline int NumFeatures() const override {
    return total_columns_ - (label_idx_ >= 0);
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class LibSVMParser: public Parser {
 public:
  explicit LibSVMParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
    if (label_idx > 0) {
      Log::Fatal("Label should be the first column in a LibSVM file");
    }
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features,
    double* out_label, const int /*line_idx*/ = -1) const override {
    int idx = 0;
    double val = 0.0f;
    if (label_idx_ == 0) {
      str = atof_(str, &val);
      *out_label = val;
      str = Common::SkipSpaceAndTab(str);
    }
    while (*str != '\0') {
      str = Common::Atoi(str, &idx);
      str = Common::SkipSpaceAndTab(str);
      if (*str == ':') {
        ++str;
        str = Common::Atof(str, &val);
        out_features->emplace_back(idx, val);
      } else {
        Log::Fatal("Input format error when parsing as LibSVM");
      }
      str = Common::SkipSpaceAndTab(str);
    }
  }

  inline int NumFeatures() const override {
    return total_columns_;
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

}  // namespace LightGBM
#endif   // LightGBM_IO_PARSER_HPP_
