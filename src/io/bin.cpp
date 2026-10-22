/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/bin.h>

#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/file_io.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "dense_bin.hpp"
#include "multi_val_dense_bin.hpp"
#include "multi_val_pairwise_lambdarank_bin.hpp"
#include "multi_val_sparse_bin.hpp"
#include "sparse_bin.hpp"
#include "pairwise_lambdarank_bin.hpp"

namespace LightGBM {

BinMapper::BinMapper(): num_bin_(1), is_trivial_(true), bin_type_(BinType::NumericalBin), pairwise_bin_num_(0) {
  bin_upper_bound_.clear();
  bin_upper_bound_.push_back(std::numeric_limits<double>::infinity());
}

// deep copy function for BinMapper
BinMapper::BinMapper(const BinMapper& other) {
  num_bin_ = other.num_bin_;
  missing_type_ = other.missing_type_;
  is_trivial_ = other.is_trivial_;
  sparse_rate_ = other.sparse_rate_;
  bin_type_ = other.bin_type_;
  if (bin_type_ == BinType::NumericalBin) {
    bin_upper_bound_ = other.bin_upper_bound_;
  } else {
    bin_2_categorical_ = other.bin_2_categorical_;
    categorical_2_bin_ = other.categorical_2_bin_;
  }
  min_val_ = other.min_val_;
  max_val_ = other.max_val_;
  default_bin_ = other.default_bin_;
  most_freq_bin_ = other.most_freq_bin_;
  pairwise_bin_num_ = other.pairwise_bin_num_;
  pairwise_bin_ranges_ = other.pairwise_bin_ranges_;
}

BinMapper::BinMapper(const void* memory) {
  CopyFrom(reinterpret_cast<const char*>(memory));
}

BinMapper::~BinMapper() {
}

void BinMapper::InitPairwiseBinRanges(uint32_t num_original_bin) {
  pairwise_bin_num_ = num_original_bin;
  pairwise_bin_ranges_.clear();
  if (pairwise_bin_num_ == 0 || num_bin_ <= 0) {
    return;
  }
  uint32_t max_bin = static_cast<uint32_t>(num_bin_ - 1);
  if (missing_type_ == MissingType::NaN && max_bin > 0) {
    --max_bin;
  }
  pairwise_bin_ranges_.resize(static_cast<size_t>(pairwise_bin_num_) * pairwise_bin_num_ * 2);
  for (size_t i = 0; i < pairwise_bin_ranges_.size(); i += 2) {
    pairwise_bin_ranges_[i] = 0;
    pairwise_bin_ranges_[i + 1] = max_bin;
  }
}

void BinMapper::SetPairwiseBinRange(uint32_t first_bin, uint32_t second_bin,
                                    uint32_t min_bin, uint32_t max_bin) {
  if (!HasPairwiseBinRanges()) {
    return;
  }
  CHECK_LT(first_bin, pairwise_bin_num_);
  CHECK_LT(second_bin, pairwise_bin_num_);
  CHECK_LE(min_bin, max_bin);
  const size_t offset = (static_cast<size_t>(first_bin) * pairwise_bin_num_ + second_bin) * 2;
  pairwise_bin_ranges_[offset] = min_bin;
  pairwise_bin_ranges_[offset + 1] = max_bin;

  Log::Warning("min_bin = %d, max_bin = %d, num_bin_ = %d", min_bin, max_bin, num_bin_);
}

bool NeedFilter(const std::vector<int>& cnt_in_bin, int total_cnt, int filter_cnt, BinType bin_type) {
  if (bin_type == BinType::NumericalBin) {
    int sum_left = 0;
    for (size_t i = 0; i < cnt_in_bin.size() - 1; ++i) {
      sum_left += cnt_in_bin[i];
      if (sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt) {
        return false;
      }
    }
  } else {
    if (cnt_in_bin.size() <= 2) {
      for (size_t i = 0; i < cnt_in_bin.size() - 1; ++i) {
        int sum_left = cnt_in_bin[i];
        if (sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt) {
          return false;
        }
      }
    } else {
      return false;
    }
  }
  return true;
}

std::vector<double> GreedyFindBin(const double* distinct_values, const int* counts,
                                  int num_distinct_values, int max_bin,
                                  size_t total_cnt, int min_data_in_bin) {
  std::vector<double> bin_upper_bound;
  CHECK_GT(max_bin, 0);
  if (num_distinct_values <= max_bin) {
    bin_upper_bound.clear();
    int cur_cnt_inbin = 0;
    for (int i = 0; i < num_distinct_values - 1; ++i) {
      cur_cnt_inbin += counts[i];
      if (cur_cnt_inbin >= min_data_in_bin) {
        auto val = Common::GetDoubleUpperBound((distinct_values[i] + distinct_values[i + 1]) / 2.0);
        if (bin_upper_bound.empty() || !Common::CheckDoubleEqualOrdered(bin_upper_bound.back(), val)) {
          bin_upper_bound.push_back(val);
          cur_cnt_inbin = 0;
        }
      }
    }
    cur_cnt_inbin += counts[num_distinct_values - 1];
    bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  } else {
    if (min_data_in_bin > 0) {
      max_bin = std::min(max_bin, static_cast<int>(total_cnt / min_data_in_bin));
      max_bin = std::max(max_bin, 1);
    }
    double mean_bin_size = static_cast<double>(total_cnt) / max_bin;

    // mean size for one bin
    int rest_bin_cnt = max_bin;
    int rest_sample_cnt = static_cast<int>(total_cnt);
    std::vector<bool> is_big_count_value(num_distinct_values, false);
    for (int i = 0; i < num_distinct_values; ++i) {
      if (counts[i] >= mean_bin_size) {
        is_big_count_value[i] = true;
        --rest_bin_cnt;
        rest_sample_cnt -= counts[i];
      }
    }
    mean_bin_size = static_cast<double>(rest_sample_cnt) / rest_bin_cnt;
    std::vector<double> upper_bounds(max_bin, std::numeric_limits<double>::infinity());
    std::vector<double> lower_bounds(max_bin, std::numeric_limits<double>::infinity());

    int bin_cnt = 0;
    lower_bounds[bin_cnt] = distinct_values[0];
    int cur_cnt_inbin = 0;
    for (int i = 0; i < num_distinct_values - 1; ++i) {
      if (!is_big_count_value[i]) {
        rest_sample_cnt -= counts[i];
      }
      cur_cnt_inbin += counts[i];
      // need a new bin
      if (is_big_count_value[i] || cur_cnt_inbin >= mean_bin_size ||
        (is_big_count_value[i + 1] && cur_cnt_inbin >= std::max(1.0, mean_bin_size * 0.5f))) {
        upper_bounds[bin_cnt] = distinct_values[i];
        ++bin_cnt;
        lower_bounds[bin_cnt] = distinct_values[i + 1];
        if (bin_cnt >= max_bin - 1) {
          break;
        }
        cur_cnt_inbin = 0;
        if (!is_big_count_value[i]) {
          --rest_bin_cnt;
          mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);
        }
      }
    }
    ++bin_cnt;
    // update bin upper bound
    bin_upper_bound.clear();
    for (int i = 0; i < bin_cnt - 1; ++i) {
      auto val = Common::GetDoubleUpperBound((upper_bounds[i] + lower_bounds[i + 1]) / 2.0);
      if (bin_upper_bound.empty() || !Common::CheckDoubleEqualOrdered(bin_upper_bound.back(), val)) {
        bin_upper_bound.push_back(val);
      }
    }
    // last bin upper bound
    bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  }
  return bin_upper_bound;
}

std::vector<double> FindBinWithPredefinedBin(const double* distinct_values, const int* counts,
                                              int num_distinct_values, int max_bin,
                                              size_t total_sample_cnt, int min_data_in_bin,
                                              const std::vector<double>& forced_upper_bounds) {
  std::vector<double> bin_upper_bound;

  // get number of positive and negative distinct values
  int left_cnt = -1;
  for (int i = 0; i < num_distinct_values; ++i) {
    if (distinct_values[i] > -kZeroThreshold) {
      left_cnt = i;
      break;
    }
  }
  if (left_cnt < 0) {
    left_cnt = num_distinct_values;
  }
  int right_start = -1;
  for (int i = left_cnt; i < num_distinct_values; ++i) {
    if (distinct_values[i] > kZeroThreshold) {
      right_start = i;
      break;
    }
  }

  // include zero bounds and infinity bound
  if (max_bin == 2) {
    if (left_cnt == 0) {
      bin_upper_bound.push_back(kZeroThreshold);
    } else {
      bin_upper_bound.push_back(-kZeroThreshold);
    }
  } else if (max_bin >= 3) {
    if (left_cnt > 0) {
      bin_upper_bound.push_back(-kZeroThreshold);
    }
    if (right_start >= 0) {
      bin_upper_bound.push_back(kZeroThreshold);
    }
  }
  bin_upper_bound.push_back(std::numeric_limits<double>::infinity());

  // add forced bounds, excluding zeros since we have already added zero bounds
  int max_to_insert = max_bin - static_cast<int>(bin_upper_bound.size());
  int num_inserted = 0;
  for (size_t i = 0; i < forced_upper_bounds.size(); ++i) {
    if (num_inserted >= max_to_insert) {
      break;
    }
    if (std::fabs(forced_upper_bounds[i]) > kZeroThreshold) {
      bin_upper_bound.push_back(forced_upper_bounds[i]);
      ++num_inserted;
    }
  }
  std::stable_sort(bin_upper_bound.begin(), bin_upper_bound.end());

  // find remaining bounds
  int free_bins = max_bin - static_cast<int>(bin_upper_bound.size());
  std::vector<double> bounds_to_add;
  int value_ind = 0;
  for (size_t i = 0; i < bin_upper_bound.size(); ++i) {
    int cnt_in_bin = 0;
    int distinct_cnt_in_bin = 0;
    int bin_start = value_ind;
    while ((value_ind < num_distinct_values) && (distinct_values[value_ind] < bin_upper_bound[i])) {
      cnt_in_bin += counts[value_ind];
      ++distinct_cnt_in_bin;
      ++value_ind;
    }
    int bins_remaining = max_bin - static_cast<int>(bin_upper_bound.size()) - static_cast<int>(bounds_to_add.size());
    int num_sub_bins = static_cast<int>(std::lround((static_cast<double>(cnt_in_bin) * free_bins / total_sample_cnt)));
    num_sub_bins = std::min(num_sub_bins, bins_remaining) + 1;
    if (i == bin_upper_bound.size() - 1) {
      num_sub_bins = bins_remaining + 1;
    }
    std::vector<double> new_upper_bounds = GreedyFindBin(distinct_values + bin_start, counts + bin_start, distinct_cnt_in_bin,
      num_sub_bins, cnt_in_bin, min_data_in_bin);
    bounds_to_add.insert(bounds_to_add.end(), new_upper_bounds.begin(), new_upper_bounds.end() - 1);  // last bound is infinity
  }
  bin_upper_bound.insert(bin_upper_bound.end(), bounds_to_add.begin(), bounds_to_add.end());
  std::stable_sort(bin_upper_bound.begin(), bin_upper_bound.end());
  CHECK_LE(bin_upper_bound.size(), static_cast<size_t>(max_bin));
  return bin_upper_bound;
}

std::vector<double> FindBinWithZeroAsOneBin(const double* distinct_values, const int* counts, int num_distinct_values,
                                            int max_bin, size_t total_sample_cnt, int min_data_in_bin) {
  std::vector<double> bin_upper_bound;
  int left_cnt_data = 0;
  int cnt_zero = 0;
  int right_cnt_data = 0;
  for (int i = 0; i < num_distinct_values; ++i) {
    if (distinct_values[i] <= -kZeroThreshold) {
      left_cnt_data += counts[i];
    } else if (distinct_values[i] > kZeroThreshold) {
      right_cnt_data += counts[i];
    } else {
      cnt_zero += counts[i];
    }
  }

  int left_cnt = -1;
  for (int i = 0; i < num_distinct_values; ++i) {
    if (distinct_values[i] > -kZeroThreshold) {
      left_cnt = i;
      break;
    }
  }

  if (left_cnt < 0) {
    left_cnt = num_distinct_values;
  }

  if ((left_cnt > 0) && (max_bin > 1)) {
    int left_max_bin = static_cast<int>(static_cast<double>(left_cnt_data) / (total_sample_cnt - cnt_zero) * (max_bin - 1));
    left_max_bin = std::max(1, left_max_bin);
    bin_upper_bound = GreedyFindBin(distinct_values, counts, left_cnt, left_max_bin, left_cnt_data, min_data_in_bin);
    if (bin_upper_bound.size() > 0) {
      bin_upper_bound.back() = -kZeroThreshold;
    }
  }

  int right_start = -1;
  for (int i = left_cnt; i < num_distinct_values; ++i) {
    if (distinct_values[i] > kZeroThreshold) {
      right_start = i;
      break;
    }
  }

  int right_max_bin = max_bin - 1 - static_cast<int>(bin_upper_bound.size());
  if (right_start >= 0 && right_max_bin > 0) {
    auto right_bounds = GreedyFindBin(distinct_values + right_start, counts + right_start,
      num_distinct_values - right_start, right_max_bin, right_cnt_data, min_data_in_bin);
    bin_upper_bound.push_back(kZeroThreshold);
    bin_upper_bound.insert(bin_upper_bound.end(), right_bounds.begin(), right_bounds.end());
  } else {
    bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
  }
  CHECK_LE(bin_upper_bound.size(), static_cast<size_t>(max_bin));
  return bin_upper_bound;
}

std::vector<double> FindBinWithZeroAsOneBin(const double* distinct_values, const int* counts, int num_distinct_values,
                                            int max_bin, size_t total_sample_cnt, int min_data_in_bin,
                                            const std::vector<double>& forced_upper_bounds) {
  if (forced_upper_bounds.empty()) {
    return FindBinWithZeroAsOneBin(distinct_values, counts, num_distinct_values, max_bin, total_sample_cnt, min_data_in_bin);
  } else {
    return FindBinWithPredefinedBin(distinct_values, counts, num_distinct_values, max_bin, total_sample_cnt, min_data_in_bin,
                                    forced_upper_bounds);
  }
}

void BinMapper::FindBin(double* values, int num_sample_values, size_t total_sample_cnt,
                        int max_bin, int min_data_in_bin, int min_split_data, bool pre_filter, BinType bin_type,
                        bool use_missing, bool zero_as_missing,
                        const std::vector<double>& forced_upper_bounds) {
  pairwise_bin_num_ = 0;
  pairwise_bin_ranges_.clear();
  int na_cnt = 0;
  int non_na_cnt = 0;
  for (int i = 0; i < num_sample_values; ++i) {
    if (!std::isnan(values[i])) {
      values[non_na_cnt++] = values[i];
    }
  }
  if (!use_missing) {
    missing_type_ = MissingType::None;
  } else if (zero_as_missing) {
    missing_type_ = MissingType::Zero;
  } else {
    if (non_na_cnt == num_sample_values) {
      missing_type_ = MissingType::None;
    } else {
      missing_type_ = MissingType::NaN;
      na_cnt = num_sample_values - non_na_cnt;
    }
  }
  num_sample_values = non_na_cnt;

  bin_type_ = bin_type;
  default_bin_ = 0;
  int zero_cnt = static_cast<int>(total_sample_cnt - num_sample_values - na_cnt);
  // find distinct_values first
  std::vector<double> distinct_values;
  std::vector<int> counts;  // count of data points for each distinct feature value.

  std::stable_sort(values, values + num_sample_values);

  // push zero in the front
  if (num_sample_values == 0 || (values[0] > 0.0f && zero_cnt > 0)) {
    distinct_values.push_back(0.0f);
    counts.push_back(zero_cnt);
  }

  if (num_sample_values > 0) {
    distinct_values.push_back(values[0]);
    counts.push_back(1);
  }

  for (int i = 1; i < num_sample_values; ++i) {
    if (!Common::CheckDoubleEqualOrdered(values[i - 1], values[i])) {
      if (values[i - 1] < 0.0f && values[i] > 0.0f) {
        distinct_values.push_back(0.0f);
        counts.push_back(zero_cnt);
      }
      distinct_values.push_back(values[i]);
      counts.push_back(1);
    } else {
      // use the large value
      distinct_values.back() = values[i];
      ++counts.back();
    }
  }

  // push zero in the back
  if (num_sample_values > 0 && values[num_sample_values - 1] < 0.0f && zero_cnt > 0) {
    distinct_values.push_back(0.0f);
    counts.push_back(zero_cnt);
  }
  min_val_ = distinct_values.front();
  max_val_ = distinct_values.back();
  std::vector<int> cnt_in_bin;  // count of data points in each bin.
  int num_distinct_values = static_cast<int>(distinct_values.size());
  if (bin_type_ == BinType::NumericalBin) {
    if (missing_type_ == MissingType::Zero) {
      bin_upper_bound_ = FindBinWithZeroAsOneBin(distinct_values.data(), counts.data(), num_distinct_values, max_bin, total_sample_cnt,
                                                  min_data_in_bin, forced_upper_bounds);
      if (bin_upper_bound_.size() == 2) {
        missing_type_ = MissingType::None;
      }
    } else if (missing_type_ == MissingType::None) {
      bin_upper_bound_ = FindBinWithZeroAsOneBin(distinct_values.data(), counts.data(), num_distinct_values, max_bin, total_sample_cnt,
                                                  min_data_in_bin, forced_upper_bounds);
    } else {
      bin_upper_bound_ = FindBinWithZeroAsOneBin(distinct_values.data(), counts.data(), num_distinct_values, max_bin - 1, total_sample_cnt - na_cnt,
                                                  min_data_in_bin, forced_upper_bounds);
      bin_upper_bound_.push_back(NaN);
    }
    num_bin_ = static_cast<int>(bin_upper_bound_.size());
    {
      cnt_in_bin.resize(num_bin_, 0);
      int i_bin = 0;
      for (int i = 0; i < num_distinct_values; ++i) {
        while (i_bin < num_bin_ - 1 && distinct_values[i] > bin_upper_bound_[i_bin]) {
          ++i_bin;
        }
        cnt_in_bin[i_bin] += counts[i];
      }
      if (missing_type_ == MissingType::NaN) {
        cnt_in_bin[num_bin_ - 1] = na_cnt;
      }
    }
    CHECK_LE(num_bin_, max_bin);
  } else {
    // convert to int type first
    std::vector<int> distinct_values_int;
    std::vector<int> counts_int;
    for (size_t i = 0; i < distinct_values.size(); ++i) {
      int val = static_cast<int>(distinct_values[i]);
      if (val < 0) {
        na_cnt += counts[i];
        Log::Warning("Met negative value in categorical features, will convert it to NaN");
      } else {
        if (distinct_values_int.empty() || val != distinct_values_int.back()) {
          distinct_values_int.push_back(val);
          counts_int.push_back(counts[i]);
        } else {
          counts_int.back() += counts[i];
        }
      }
    }
    int rest_cnt = static_cast<int>(total_sample_cnt - na_cnt);
    if (rest_cnt > 0) {
      const int SPARSE_RATIO = 100;
      if (distinct_values_int.back() / SPARSE_RATIO > static_cast<int>(distinct_values_int.size())) {
        Log::Warning("Met categorical feature which contains sparse values. "
                      "Consider renumbering to consecutive integers started from zero");
      }
      // sort by counts in descending order
      Common::SortForPair<int, int>(&counts_int, &distinct_values_int, 0, true);
      // will ignore the categorical of small counts
      int cut_cnt = static_cast<int>(
          Common::RoundInt((total_sample_cnt - na_cnt) * 0.99f));
      size_t cur_cat_idx = 0;  // index of current category.
      categorical_2_bin_.clear();
      bin_2_categorical_.clear();
      int used_cnt = 0;
      int distinct_cnt = static_cast<int>(distinct_values_int.size());
      if (na_cnt > 0) {
        ++distinct_cnt;
      }
      max_bin = std::min(distinct_cnt, max_bin);
      cnt_in_bin.clear();

      // Push the dummy bin for NaN
      bin_2_categorical_.push_back(-1);
      categorical_2_bin_[-1] = 0;
      cnt_in_bin.push_back(0);
      num_bin_ = 1;
      while (cur_cat_idx < distinct_values_int.size()
              && (used_cnt < cut_cnt || num_bin_ < max_bin)) {
        if (counts_int[cur_cat_idx] < min_data_in_bin && cur_cat_idx > 1) {
          break;
        }
        bin_2_categorical_.push_back(distinct_values_int[cur_cat_idx]);
        categorical_2_bin_[distinct_values_int[cur_cat_idx]] = static_cast<unsigned int>(num_bin_);
        used_cnt += counts_int[cur_cat_idx];
        cnt_in_bin.push_back(counts_int[cur_cat_idx]);
        ++num_bin_;
        ++cur_cat_idx;
      }
      // Use MissingType::None to represent this bin contains all categoricals
      if (cur_cat_idx == distinct_values_int.size() && na_cnt == 0) {
        missing_type_ = MissingType::None;
      } else {
        missing_type_ = MissingType::NaN;
      }
      // fix count of NaN bin
      cnt_in_bin[0] = static_cast<int>(total_sample_cnt - used_cnt);
    }
  }

  // check trivial(num_bin_ == 1) feature
  if (num_bin_ <= 1) {
    is_trivial_ = true;
  } else {
    is_trivial_ = false;
  }
  // check useless bin
  if (!is_trivial_ && pre_filter && NeedFilter(cnt_in_bin, static_cast<int>(total_sample_cnt), min_split_data, bin_type_)) {
    is_trivial_ = true;
  }

  if (!is_trivial_) {
    default_bin_ = ValueToBin(0);
    most_freq_bin_ =
        static_cast<uint32_t>(ArrayArgs<int>::ArgMax(cnt_in_bin));
    const double max_sparse_rate =
        static_cast<double>(cnt_in_bin[most_freq_bin_]) / total_sample_cnt;
    // When most_freq_bin_ != default_bin_, there are some additional data loading costs.
    // so use most_freq_bin_ = default_bin_ when there is not so sparse
    if (most_freq_bin_ != default_bin_ && max_sparse_rate < kSparseThreshold) {
      most_freq_bin_ = default_bin_;
    }
    sparse_rate_ =
        static_cast<double>(cnt_in_bin[most_freq_bin_]) / total_sample_cnt;
  } else {
    sparse_rate_ = 1.0f;
  }
}

void BinMapper::CopyTo(char * buffer) const {
  std::memcpy(buffer, &num_bin_, sizeof(num_bin_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(num_bin_));
  std::memcpy(buffer, &missing_type_, sizeof(missing_type_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(missing_type_));
  std::memcpy(buffer, &is_trivial_, sizeof(is_trivial_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(is_trivial_));
  std::memcpy(buffer, &sparse_rate_, sizeof(sparse_rate_));
  buffer += sizeof(sparse_rate_);
  std::memcpy(buffer, &bin_type_, sizeof(bin_type_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(bin_type_));
  std::memcpy(buffer, &min_val_, sizeof(min_val_));
  buffer += sizeof(min_val_);
  std::memcpy(buffer, &max_val_, sizeof(max_val_));
  buffer += sizeof(max_val_);
  std::memcpy(buffer, &default_bin_, sizeof(default_bin_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(default_bin_));
  std::memcpy(buffer, &most_freq_bin_, sizeof(most_freq_bin_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(most_freq_bin_));
  if (bin_type_ == BinType::NumericalBin) {
    std::memcpy(buffer, bin_upper_bound_.data(), num_bin_ * sizeof(double));
  } else {
    std::memcpy(buffer, bin_2_categorical_.data(), num_bin_ * sizeof(int));
  }
}

void BinMapper::CopyFrom(const char * buffer) {
  pairwise_bin_num_ = 0;
  pairwise_bin_ranges_.clear();
  std::memcpy(&num_bin_, buffer, sizeof(num_bin_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(num_bin_));
  std::memcpy(&missing_type_, buffer, sizeof(missing_type_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(missing_type_));
  std::memcpy(&is_trivial_, buffer, sizeof(is_trivial_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(is_trivial_));
  std::memcpy(&sparse_rate_, buffer, sizeof(sparse_rate_));
  buffer += sizeof(sparse_rate_);
  std::memcpy(&bin_type_, buffer, sizeof(bin_type_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(bin_type_));
  std::memcpy(&min_val_, buffer, sizeof(min_val_));
  buffer += sizeof(min_val_);
  std::memcpy(&max_val_, buffer, sizeof(max_val_));
  buffer += sizeof(max_val_);
  std::memcpy(&default_bin_, buffer, sizeof(default_bin_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(default_bin_));
  std::memcpy(&most_freq_bin_, buffer, sizeof(most_freq_bin_));
  buffer += VirtualFileWriter::AlignedSize(sizeof(most_freq_bin_));
  if (bin_type_ == BinType::NumericalBin) {
    bin_upper_bound_ = std::vector<double>(num_bin_);
    std::memcpy(bin_upper_bound_.data(), buffer, num_bin_ * sizeof(double));
  } else {
    bin_2_categorical_ = std::vector<int>(num_bin_);
    std::memcpy(bin_2_categorical_.data(), buffer, num_bin_ * sizeof(int));
    categorical_2_bin_.clear();
    for (int i = 0; i < num_bin_; ++i) {
      categorical_2_bin_[bin_2_categorical_[i]] = static_cast<unsigned int>(i);
    }
  }
}

void BinMapper::SaveBinaryToFile(BinaryWriter* writer) const {
  writer->AlignedWrite(&num_bin_, sizeof(num_bin_));
  writer->AlignedWrite(&missing_type_, sizeof(missing_type_));
  writer->AlignedWrite(&is_trivial_, sizeof(is_trivial_));
  writer->Write(&sparse_rate_, sizeof(sparse_rate_));
  writer->AlignedWrite(&bin_type_, sizeof(bin_type_));
  writer->Write(&min_val_, sizeof(min_val_));
  writer->Write(&max_val_, sizeof(max_val_));
  writer->AlignedWrite(&default_bin_, sizeof(default_bin_));
  writer->AlignedWrite(&most_freq_bin_, sizeof(most_freq_bin_));
  if (bin_type_ == BinType::NumericalBin) {
    writer->Write(bin_upper_bound_.data(), sizeof(double) * num_bin_);
  } else {
    writer->Write(bin_2_categorical_.data(), sizeof(int) * num_bin_);
  }
}

size_t BinMapper::SizesInByte() const {
  size_t ret = VirtualFileWriter::AlignedSize(sizeof(num_bin_)) +
                VirtualFileWriter::AlignedSize(sizeof(missing_type_)) +
                VirtualFileWriter::AlignedSize(sizeof(is_trivial_)) +
                sizeof(sparse_rate_) +
                VirtualFileWriter::AlignedSize(sizeof(bin_type_)) +
                sizeof(min_val_) + sizeof(max_val_) +
                VirtualFileWriter::AlignedSize(sizeof(default_bin_)) +
                VirtualFileWriter::AlignedSize(sizeof(most_freq_bin_));
  if (bin_type_ == BinType::NumericalBin) {
    ret += sizeof(double) *  num_bin_;
  } else {
    ret += sizeof(int) * num_bin_;
  }
  return ret;
}

template class DenseBin<uint8_t, true>;
template class DenseBin<uint8_t, false>;
template class DenseBin<uint16_t, false>;
template class DenseBin<uint32_t, false>;

template class SparseBin<uint8_t>;
template class SparseBin<uint16_t>;
template class SparseBin<uint32_t>;

template class MultiValDenseBin<uint8_t>;
template class MultiValDenseBin<uint16_t>;
template class MultiValDenseBin<uint32_t>;

Bin* Bin::CreateDenseBin(data_size_t num_data, int num_bin) {
  if (num_bin <= 16) {
    return new DenseBin<uint8_t, true>(num_data);
  } else if (num_bin <= 256) {
    return new DenseBin<uint8_t, false>(num_data);
  } else if (num_bin <= 65536) {
    return new DenseBin<uint16_t, false>(num_data);
  } else {
    return new DenseBin<uint32_t, false>(num_data);
  }
}

Bin* Bin::CreateSparseBin(data_size_t num_data, int num_bin) {
  if (num_bin <= 256) {
    return new SparseBin<uint8_t>(num_data);
  } else if (num_bin <= 65536) {
    return new SparseBin<uint16_t>(num_data);
  } else {
    return new SparseBin<uint32_t>(num_data);
  }
}

MultiValBin* MultiValBin::CreateMultiValSparseBin(data_size_t num_data,
                                                  int num_bin,
                                                  double estimate_element_per_row,
                                                  const bool /*use_pairwise_ranking*/,
                                                  const std::pair<data_size_t, data_size_t>* /*paired_ranking_item_global_index_map*/) {
  size_t estimate_total_entries =
      static_cast<size_t>(estimate_element_per_row * 1.1 * num_data);
  if (estimate_total_entries <= std::numeric_limits<uint16_t>::max()) {
    if (num_bin <= 256) {
      return new MultiValSparseBin<uint16_t, uint8_t>(
          num_data, num_bin, estimate_element_per_row);
    } else if (num_bin <= 65536) {
      return new MultiValSparseBin<uint16_t, uint16_t>(
          num_data, num_bin, estimate_element_per_row);
    } else {
      return new MultiValSparseBin<uint16_t, uint32_t>(
          num_data, num_bin, estimate_element_per_row);
    }
  } else if (estimate_total_entries <= std::numeric_limits<uint32_t>::max()) {
    if (num_bin <= 256) {
      return new MultiValSparseBin<uint32_t, uint8_t>(
          num_data, num_bin, estimate_element_per_row);
    } else if (num_bin <= 65536) {
      return new MultiValSparseBin<uint32_t, uint16_t>(
          num_data, num_bin, estimate_element_per_row);
    } else {
      return new MultiValSparseBin<uint32_t, uint32_t>(
          num_data, num_bin, estimate_element_per_row);
    }
  } else {
    if (num_bin <= 256) {
      return new MultiValSparseBin<size_t, uint8_t>(
          num_data, num_bin, estimate_element_per_row);
    } else if (num_bin <= 65536) {
      return new MultiValSparseBin<size_t, uint16_t>(
          num_data, num_bin, estimate_element_per_row);
    } else {
      return new MultiValSparseBin<size_t, uint32_t>(
          num_data, num_bin, estimate_element_per_row);
    }
  }
}

  Bin* Bin::CreateDensePairwiseRankingFirstBin(data_size_t num_original_data, int num_bin, data_size_t num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map) {
    if (num_bin <= 16) {
      return new DensePairwiseRankingFirstBin<uint8_t, true>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint8_t, true>(num_original_data));
    } else if (num_bin <= 256) {
      return new DensePairwiseRankingFirstBin<uint8_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint8_t, false>(num_original_data));
    } else if (num_bin <= 65536) {
      return new DensePairwiseRankingFirstBin<uint16_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint16_t, false>(num_original_data));
    } else {
      return new DensePairwiseRankingFirstBin<uint32_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint32_t, false>(num_original_data));
    }
  }

  Bin* Bin::CreateDensePairwiseRankingSecondBin(data_size_t num_original_data, int num_bin, data_size_t num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map) {
    if (num_bin <= 16) {
      return new DensePairwiseRankingSecondBin<uint8_t, true>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint8_t, true>(num_original_data));
    } else if (num_bin <= 256) {
      return new DensePairwiseRankingSecondBin<uint8_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint8_t, false>(num_original_data));
    } else if (num_bin <= 65536) {
      return new DensePairwiseRankingSecondBin<uint16_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint16_t, false>(num_original_data));
    } else {
      return new DensePairwiseRankingSecondBin<uint32_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint32_t, false>(num_original_data));
    }
  }

  Bin* Bin::CreateSparsePairwiseRankingFirstBin(data_size_t num_original_data, int num_bin, data_size_t num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map) {
    if (num_bin <= 256) {
      return new SparsePairwiseRankingFirstBin<uint8_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint8_t>(num_original_data));
    } else if (num_bin <= 65536) {
      return new SparsePairwiseRankingFirstBin<uint16_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint16_t>(num_original_data));
    } else {
      return new SparsePairwiseRankingFirstBin<uint32_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint32_t>(num_original_data));
    }
  }

  Bin* Bin::CreateSparsePairwiseRankingSecondBin(data_size_t num_original_data, int num_bin, data_size_t num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map) {
    if (num_bin <= 256) {
      return new SparsePairwiseRankingSecondBin<uint8_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint8_t>(num_original_data));
    } else if (num_bin <= 65536) {
      return new SparsePairwiseRankingSecondBin<uint16_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint16_t>(num_original_data));
    } else {
      return new SparsePairwiseRankingSecondBin<uint32_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint32_t>(num_original_data));
    }
  }

  Bin* Bin::CreateDensePairwiseRankingDiffBin(data_size_t num_original_data, int num_bin, data_size_t num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const std::vector<std::unique_ptr<const BinMapper>>* diff_bin_mappers, const std::vector<std::unique_ptr<const BinMapper>>* ori_bin_mappers, const std::vector<uint32_t>* bin_offsets, const std::vector<uint32_t>* diff_bin_offsets, const std::vector<float>* raw_values) {
    if (num_bin <= 16) {
      return new DensePairwiseRankingDiffBin<uint8_t, true>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint8_t, true>(num_original_data), diff_bin_mappers, ori_bin_mappers, bin_offsets, diff_bin_offsets, raw_values);
    } else if (num_bin <= 256) {
      return new DensePairwiseRankingDiffBin<uint8_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint8_t, false>(num_original_data), diff_bin_mappers, ori_bin_mappers, bin_offsets, diff_bin_offsets, raw_values);
    } else if (num_bin <= 65536) {
      return new DensePairwiseRankingDiffBin<uint16_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint16_t, false>(num_original_data), diff_bin_mappers, ori_bin_mappers, bin_offsets, diff_bin_offsets, raw_values);
    } else {
      return new DensePairwiseRankingDiffBin<uint32_t, false>(num_pairs, paired_ranking_item_index_map, new DenseBin<uint32_t, false>(num_original_data), diff_bin_mappers, ori_bin_mappers, bin_offsets, diff_bin_offsets, raw_values);
    }
  }

  Bin* Bin::CreateSparsePairwiseRankingDiffBin(data_size_t num_original_data, int num_bin, data_size_t num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const std::vector<std::unique_ptr<const BinMapper>>* diff_bin_mappers, const std::vector<std::unique_ptr<const BinMapper>>* ori_bin_mappers, const std::vector<uint32_t>* bin_offsets, const std::vector<uint32_t>* diff_bin_offsets, const std::vector<float>* raw_values) {
    if (num_bin <= 256) {
      return new SparsePairwiseRankingDiffBin<uint8_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint8_t>(num_original_data), diff_bin_mappers, ori_bin_mappers, bin_offsets, diff_bin_offsets, raw_values);
    } else if (num_bin <= 65536) {
      return new SparsePairwiseRankingDiffBin<uint16_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint16_t>(num_original_data), diff_bin_mappers, ori_bin_mappers, bin_offsets, diff_bin_offsets, raw_values);
    } else {
      return new SparsePairwiseRankingDiffBin<uint32_t>(num_pairs, paired_ranking_item_index_map, new SparseBin<uint32_t>(num_original_data), diff_bin_mappers, ori_bin_mappers, bin_offsets, diff_bin_offsets, raw_values);
    }
  }

  MultiValBin* MultiValBin::CreateMultiValBin(data_size_t num_data, int num_bin, int num_feature,
    double sparse_rate, const std::vector<uint32_t>& offsets, const bool use_pairwise_ranking, const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map, const std::vector<const BinMapper*> diff_feature_bin_mappers,
    const std::vector<const BinMapper*> original_feature_bin_mappers, const bool use_pairwise_bin_lookup,
    const std::vector<std::vector<float>>* raw_data,
    const std::vector<uint32_t>& all_offsets, const std::vector<int>& diff_feature_to_original_feature_slot,
    const std::vector<int>& diff_feature_to_raw_feature_index) {
    if (sparse_rate >= multi_val_bin_sparse_threshold) {
      const double average_element_per_row = (1.0 - sparse_rate) * num_feature;
      // if (use_pairwise_ranking) {
        Log::Warning("Pairwise ranking with sparse row-wse bins is not supported yet.");
        return CreateMultiValDenseBin(num_data, num_bin, num_feature, offsets, use_pairwise_ranking, paired_ranking_item_global_index_map, diff_feature_bin_mappers, original_feature_bin_mappers, use_pairwise_bin_lookup, raw_data, all_offsets, diff_feature_to_original_feature_slot, diff_feature_to_raw_feature_index);
      // } else {
      //   return CreateMultiValSparseBin(num_data, num_bin,
      //                                  average_element_per_row, use_pairwise_ranking, paired_ranking_item_global_index_map);
      // }
    } else {
      return CreateMultiValDenseBin(num_data, num_bin, num_feature, offsets, use_pairwise_ranking, paired_ranking_item_global_index_map, diff_feature_bin_mappers, original_feature_bin_mappers, use_pairwise_bin_lookup, raw_data, all_offsets, diff_feature_to_original_feature_slot, diff_feature_to_raw_feature_index);
    }
  }

  MultiValBin* MultiValBin::CreateMultiValDenseBin(data_size_t num_data,
                                                   int num_bin,
                                                   int num_feature,
                                                   const std::vector<uint32_t>& offsets,
                                                   const bool use_pairwise_ranking,
                                                   const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map, const std::vector<const BinMapper*> diff_feature_bin_mappers,
                                                   const std::vector<const BinMapper*> original_feature_bin_mappers, const bool use_pairwise_bin_lookup,
                                                   const std::vector<std::vector<float>>* raw_data,
                                                   const std::vector<uint32_t>& all_offsets, const std::vector<int>& diff_feature_to_original_feature_slot,
                                                   const std::vector<int>& diff_feature_to_raw_feature_index) {
    // calculate max bin of all features to select the int type in MultiValDenseBin
    int max_bin = 0;
    for (int i = 0; i < static_cast<int>(offsets.size()) - 1; ++i) {
      int feature_bin = offsets[i + 1] - offsets[i];
      if (feature_bin > max_bin) {
        max_bin = feature_bin;
      }
    }
    if (max_bin <= 256) {
      if (use_pairwise_ranking) {
        return new MultiValDensePairwiseLambdarankBin<uint8_t>(num_data, num_bin, num_feature, offsets, paired_ranking_item_global_index_map, diff_feature_bin_mappers, original_feature_bin_mappers, use_pairwise_bin_lookup, raw_data, all_offsets, diff_feature_to_original_feature_slot, diff_feature_to_raw_feature_index);
      } else {
        return new MultiValDenseBin<uint8_t>(num_data, num_bin, num_feature, offsets);
      }
    } else if (max_bin <= 65536) {
      if (use_pairwise_ranking) {
        return new MultiValDensePairwiseLambdarankBin<uint16_t>(num_data, num_bin, num_feature, offsets, paired_ranking_item_global_index_map, diff_feature_bin_mappers, original_feature_bin_mappers, use_pairwise_bin_lookup, raw_data, all_offsets, diff_feature_to_original_feature_slot, diff_feature_to_raw_feature_index);
      } else {
        return new MultiValDenseBin<uint16_t>(num_data, num_bin, num_feature, offsets);
      }
    } else {
      if (use_pairwise_ranking) {
        return new MultiValDensePairwiseLambdarankBin<uint32_t>(num_data, num_bin, num_feature, offsets, paired_ranking_item_global_index_map, diff_feature_bin_mappers, original_feature_bin_mappers, use_pairwise_bin_lookup, raw_data, all_offsets, diff_feature_to_original_feature_slot, diff_feature_to_raw_feature_index);
      } else {
        return new MultiValDenseBin<uint32_t>(num_data, num_bin, num_feature, offsets);
      }
  }
}

template <>
const void* DenseBin<uint8_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int /*num_threads*/) const {
  *is_sparse = false;
  *bit_type = 8;
  bin_iterator->clear();
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint16_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int /*num_threads*/) const {
  *is_sparse = false;
  *bit_type = 16;
  bin_iterator->clear();
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint32_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int /*num_threads*/) const {
  *is_sparse = false;
  *bit_type = 32;
  bin_iterator->clear();
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint8_t, true>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int /*num_threads*/) const {
  *is_sparse = false;
  *bit_type = 4;
  bin_iterator->clear();
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint8_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = false;
  *bit_type = 8;
  *bin_iterator = nullptr;
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint16_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = false;
  *bit_type = 16;
  *bin_iterator = nullptr;
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint32_t, false>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = false;
  *bit_type = 32;
  *bin_iterator = nullptr;
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* DenseBin<uint8_t, true>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = false;
  *bit_type = 4;
  *bin_iterator = nullptr;
  return reinterpret_cast<const void*>(data_.data());
}

template <>
const void* SparseBin<uint8_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int num_threads) const {
  *is_sparse = true;
  *bit_type = 8;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    bin_iterator->emplace_back(new SparseBinIterator<uint8_t>(this, 0));
  }
  return nullptr;
}

template <>
const void* SparseBin<uint16_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int num_threads) const {
  *is_sparse = true;
  *bit_type = 16;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    bin_iterator->emplace_back(new SparseBinIterator<uint16_t>(this, 0));
  }
  return nullptr;
}

template <>
const void* SparseBin<uint32_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  std::vector<BinIterator*>* bin_iterator,
  const int num_threads) const {
  *is_sparse = true;
  *bit_type = 32;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    bin_iterator->emplace_back(new SparseBinIterator<uint32_t>(this, 0));
  }
  return nullptr;
}

template <>
const void* SparseBin<uint8_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = true;
  *bit_type = 8;
  *bin_iterator = new SparseBinIterator<uint8_t>(this, 0);
  return nullptr;
}

template <>
const void* SparseBin<uint16_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = true;
  *bit_type = 16;
  *bin_iterator = new SparseBinIterator<uint16_t>(this, 0);
  return nullptr;
}

template <>
const void* SparseBin<uint32_t>::GetColWiseData(
  uint8_t* bit_type,
  bool* is_sparse,
  BinIterator** bin_iterator) const {
  *is_sparse = true;
  *bit_type = 32;
  *bin_iterator = new SparseBinIterator<uint32_t>(this, 0);
  return nullptr;
}

#ifdef USE_CUDA
template <>
const void* MultiValDenseBin<uint8_t>::GetRowWiseData(uint8_t* bit_type,
    size_t* total_size,
    bool* is_sparse,
    const void** out_data_ptr,
    uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = data_.data();
  *bit_type = 8;
  *total_size = static_cast<size_t>(num_data_) * static_cast<size_t>(num_feature_);
  CHECK_EQ(*total_size, data_.size());
  *is_sparse = false;
  *out_data_ptr = nullptr;
  *data_ptr_bit_type = 0;
  return to_return;
}

template <>
const void* MultiValDenseBin<uint16_t>::GetRowWiseData(uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint16_t* data_ptr = data_.data();
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_ptr);
  *bit_type = 16;
  *total_size = static_cast<size_t>(num_data_) * static_cast<size_t>(num_feature_);
  CHECK_EQ(*total_size, data_.size());
  *is_sparse = false;
  *out_data_ptr = nullptr;
  *data_ptr_bit_type = 0;
  return to_return;
}

template <>
const void* MultiValDenseBin<uint32_t>::GetRowWiseData(uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint32_t* data_ptr = data_.data();
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_ptr);
  *bit_type = 32;
  *total_size = static_cast<size_t>(num_data_) * static_cast<size_t>(num_feature_);
  CHECK_EQ(*total_size, data_.size());
  *is_sparse = false;
  *out_data_ptr = nullptr;
  *data_ptr_bit_type = 0;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint16_t, uint8_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = data_.data();
  *bit_type = 8;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 16;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint16_t, uint16_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 16;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 16;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint16_t, uint32_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 32;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 16;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint32_t, uint8_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = data_.data();
  *bit_type = 8;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 32;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint32_t, uint16_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 16;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 32;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint32_t, uint32_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 32;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 32;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint64_t, uint8_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = data_.data();
  *bit_type = 8;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 64;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint64_t, uint16_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 16;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 64;
  return to_return;
}

template <>
const void* MultiValSparseBin<uint64_t, uint32_t>::GetRowWiseData(
  uint8_t* bit_type,
  size_t* total_size,
  bool* is_sparse,
  const void** out_data_ptr,
  uint8_t* data_ptr_bit_type) const {
  const uint8_t* to_return = reinterpret_cast<const uint8_t*>(data_.data());
  *bit_type = 32;
  *total_size = data_.size();
  *is_sparse = true;
  *out_data_ptr = reinterpret_cast<const uint8_t*>(row_ptr_.data());
  *data_ptr_bit_type = 64;
  return to_return;
}

#endif  // USE_CUDA

}  // namespace LightGBM
