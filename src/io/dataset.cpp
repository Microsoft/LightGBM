/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <LightGBM/dataset.h>

#include <LightGBM/feature_group.h>
#include <LightGBM/cuda/vector_cudahost.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/threading.h>

#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace LightGBM {

Dataset::Dataset() {
  data_filename_ = "noname";
  num_data_ = 0;
  is_finish_load_ = false;
  has_raw_ = false;
}

Dataset::Dataset(data_size_t num_data) {
  CHECK_GT(num_data, 0);
  data_filename_ = "noname";
  num_data_ = num_data;
  metadata_.Init(num_data_, NO_SPECIFIC, NO_SPECIFIC);
  is_finish_load_ = false;
  group_bin_boundaries_.push_back(0);
  has_raw_ = false;
}

Dataset::~Dataset() {}

std::vector<std::vector<int>> NoGroup(const std::vector<int>& used_features) {
  std::vector<std::vector<int>> features_in_group;
  features_in_group.resize(used_features.size());
  for (size_t i = 0; i < used_features.size(); ++i) {
    features_in_group[i].emplace_back(used_features[i]);
  }
  return features_in_group;
}

int GetConflictCount(const std::vector<bool>& mark, const int* indices,
                     int num_indices, data_size_t max_cnt) {
  int ret = 0;
  for (int i = 0; i < num_indices; ++i) {
    if (mark[indices[i]]) {
      ++ret;
    }
    if (ret > max_cnt) {
      return -1;
    }
  }
  return ret;
}

void MarkUsed(std::vector<bool>* mark, const int* indices,
              data_size_t num_indices) {
  auto& ref_mark = *mark;
  for (int i = 0; i < num_indices; ++i) {
    ref_mark[indices[i]] = true;
  }
}

std::vector<int> FixSampleIndices(const BinMapper* bin_mapper,
                                  int num_total_samples, int num_indices,
                                  const int* sample_indices,
                                  const double* sample_values) {
  std::vector<int> ret;
  if (bin_mapper->GetDefaultBin() == bin_mapper->GetMostFreqBin()) {
    return ret;
  }
  int i = 0, j = 0;
  while (i < num_total_samples) {
    if (j < num_indices && sample_indices[j] < i) {
      ++j;
    } else if (j < num_indices && sample_indices[j] == i) {
      if (bin_mapper->ValueToBin(sample_values[j]) !=
          bin_mapper->GetMostFreqBin()) {
        ret.push_back(i);
      }
      ++i;
    } else {
      ret.push_back(i++);
    }
  }
  return ret;
}

std::vector<std::vector<int>> FindGroups(
    const std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
    const std::vector<int>& find_order, int** sample_indices,
    const int* num_per_col, int num_sample_col, data_size_t total_sample_cnt,
    data_size_t num_data, bool is_use_gpu, bool is_sparse,
    std::vector<int8_t>* multi_val_group) {
  const int max_search_group = 100;
  const int max_bin_per_group = 256;
  const data_size_t single_val_max_conflict_cnt =
      static_cast<data_size_t>(total_sample_cnt / 10000);
  multi_val_group->clear();

  Random rand(num_data);
  std::vector<std::vector<int>> features_in_group;
  std::vector<std::vector<bool>> conflict_marks;
  std::vector<data_size_t> group_used_row_cnt;
  std::vector<data_size_t> group_total_data_cnt;
  std::vector<int> group_num_bin;

  // first round: fill the single val group
  for (auto fidx : find_order) {
    bool is_filtered_feature = fidx >= num_sample_col;
    const data_size_t cur_non_zero_cnt =
        is_filtered_feature ? 0 : num_per_col[fidx];
    std::vector<int> available_groups;
    for (int gid = 0; gid < static_cast<int>(features_in_group.size()); ++gid) {
      auto cur_num_bin = group_num_bin[gid] + bin_mappers[fidx]->num_bin() +
                         (bin_mappers[fidx]->GetDefaultBin() == 0 ? -1 : 0);
      if (group_total_data_cnt[gid] + cur_non_zero_cnt <=
          total_sample_cnt + single_val_max_conflict_cnt) {
        if (!is_use_gpu || cur_num_bin <= max_bin_per_group) {
          available_groups.push_back(gid);
        }
      }
    }
    std::vector<int> search_groups;
    if (!available_groups.empty()) {
      int last = static_cast<int>(available_groups.size()) - 1;
      auto indices = rand.Sample(last, std::min(last, max_search_group - 1));
      // always push the last group
      search_groups.push_back(available_groups.back());
      for (auto idx : indices) {
        search_groups.push_back(available_groups[idx]);
      }
    }
    int best_gid = -1;
    int best_conflict_cnt = -1;
    for (auto gid : search_groups) {
      const data_size_t rest_max_cnt = single_val_max_conflict_cnt -
                                       group_total_data_cnt[gid] +
                                       group_used_row_cnt[gid];
      const data_size_t cnt =
          is_filtered_feature
              ? 0
              : GetConflictCount(conflict_marks[gid], sample_indices[fidx],
                                 num_per_col[fidx], rest_max_cnt);
      if (cnt >= 0 && cnt <= rest_max_cnt && cnt <= cur_non_zero_cnt / 2) {
        best_gid = gid;
        best_conflict_cnt = cnt;
        break;
      }
    }
    if (best_gid >= 0) {
      features_in_group[best_gid].push_back(fidx);
      group_total_data_cnt[best_gid] += cur_non_zero_cnt;
      group_used_row_cnt[best_gid] += cur_non_zero_cnt - best_conflict_cnt;
      if (!is_filtered_feature) {
        MarkUsed(&conflict_marks[best_gid], sample_indices[fidx],
                 num_per_col[fidx]);
      }
      group_num_bin[best_gid] +=
          bin_mappers[fidx]->num_bin() +
          (bin_mappers[fidx]->GetDefaultBin() == 0 ? -1 : 0);
    } else {
      features_in_group.emplace_back();
      features_in_group.back().push_back(fidx);
      conflict_marks.emplace_back(total_sample_cnt, false);
      if (!is_filtered_feature) {
        MarkUsed(&(conflict_marks.back()), sample_indices[fidx],
                 num_per_col[fidx]);
      }
      group_total_data_cnt.emplace_back(cur_non_zero_cnt);
      group_used_row_cnt.emplace_back(cur_non_zero_cnt);
      group_num_bin.push_back(
          1 + bin_mappers[fidx]->num_bin() +
          (bin_mappers[fidx]->GetDefaultBin() == 0 ? -1 : 0));
    }
  }
  if (!is_sparse) {
    multi_val_group->resize(features_in_group.size(), false);
    return features_in_group;
  }
  std::vector<int> second_round_features;
  std::vector<std::vector<int>> features_in_group2;
  std::vector<std::vector<bool>> conflict_marks2;

  const double dense_threshold = 0.4;
  for (int gid = 0; gid < static_cast<int>(features_in_group.size()); ++gid) {
    const double dense_rate =
        static_cast<double>(group_used_row_cnt[gid]) / total_sample_cnt;
    if (dense_rate >= dense_threshold) {
      features_in_group2.push_back(std::move(features_in_group[gid]));
      conflict_marks2.push_back(std::move(conflict_marks[gid]));
    } else {
      for (auto fidx : features_in_group[gid]) {
        second_round_features.push_back(fidx);
      }
    }
  }

  features_in_group = features_in_group2;
  conflict_marks = conflict_marks2;
  multi_val_group->resize(features_in_group.size(), false);
  if (!second_round_features.empty()) {
    features_in_group.emplace_back();
    conflict_marks.emplace_back(total_sample_cnt, false);
    bool is_multi_val = is_use_gpu ? true : false;
    int conflict_cnt = 0;
    for (auto fidx : second_round_features) {
      features_in_group.back().push_back(fidx);
      if (!is_multi_val) {
        const int rest_max_cnt = single_val_max_conflict_cnt - conflict_cnt;
        const auto cnt =
            GetConflictCount(conflict_marks.back(), sample_indices[fidx],
                             num_per_col[fidx], rest_max_cnt);
        conflict_cnt += cnt;
        if (cnt < 0 || conflict_cnt > single_val_max_conflict_cnt) {
          is_multi_val = true;
          continue;
        }
        MarkUsed(&(conflict_marks.back()), sample_indices[fidx],
                 num_per_col[fidx]);
      }
    }
    multi_val_group->push_back(is_multi_val);
  }
  return features_in_group;
}

std::vector<std::vector<int>> FastFeatureBundling(
    const std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
    int** sample_indices, double** sample_values, const int* num_per_col,
    int num_sample_col, data_size_t total_sample_cnt,
    const std::vector<int>& used_features, data_size_t num_data,
    bool is_use_gpu, bool is_sparse, std::vector<int8_t>* multi_val_group) {
  Common::FunctionTimer fun_timer("Dataset::FastFeatureBundling", global_timer);
  std::vector<size_t> feature_non_zero_cnt;
  feature_non_zero_cnt.reserve(used_features.size());
  // put dense feature first
  for (auto fidx : used_features) {
    if (fidx < num_sample_col) {
      feature_non_zero_cnt.emplace_back(num_per_col[fidx]);
    } else {
      feature_non_zero_cnt.emplace_back(0);
    }
  }
  // sort by non zero cnt
  std::vector<int> sorted_idx;
  sorted_idx.reserve(used_features.size());
  for (int i = 0; i < static_cast<int>(used_features.size()); ++i) {
    sorted_idx.emplace_back(i);
  }
  // sort by non zero cnt, bigger first
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&feature_non_zero_cnt](int a, int b) {
                     return feature_non_zero_cnt[a] > feature_non_zero_cnt[b];
                   });

  std::vector<int> feature_order_by_cnt;
  feature_order_by_cnt.reserve(sorted_idx.size());
  for (auto sidx : sorted_idx) {
    feature_order_by_cnt.push_back(used_features[sidx]);
  }

  std::vector<std::vector<int>> tmp_indices;
  std::vector<int> tmp_num_per_col(num_sample_col, 0);
  for (auto fidx : used_features) {
    if (fidx >= num_sample_col) {
      continue;
    }
    auto ret = FixSampleIndices(
        bin_mappers[fidx].get(), static_cast<int>(total_sample_cnt),
        num_per_col[fidx], sample_indices[fidx], sample_values[fidx]);
    if (!ret.empty()) {
      tmp_indices.push_back(ret);
      tmp_num_per_col[fidx] = static_cast<int>(ret.size());
      sample_indices[fidx] = tmp_indices.back().data();
    } else {
      tmp_num_per_col[fidx] = num_per_col[fidx];
    }
  }
  std::vector<int8_t> group_is_multi_val, group_is_multi_val2;
  auto features_in_group =
      FindGroups(bin_mappers, used_features, sample_indices,
                 tmp_num_per_col.data(), num_sample_col, total_sample_cnt,
                 num_data, is_use_gpu, is_sparse, &group_is_multi_val);
  auto group2 =
      FindGroups(bin_mappers, feature_order_by_cnt, sample_indices,
                 tmp_num_per_col.data(), num_sample_col, total_sample_cnt,
                 num_data, is_use_gpu, is_sparse, &group_is_multi_val2);

  if (features_in_group.size() > group2.size()) {
    features_in_group = group2;
    group_is_multi_val = group_is_multi_val2;
  }
  // shuffle groups
  int num_group = static_cast<int>(features_in_group.size());
  Random tmp_rand(num_data);
  for (int i = 0; i < num_group - 1; ++i) {
    int j = tmp_rand.NextShort(i + 1, num_group);
    std::swap(features_in_group[i], features_in_group[j]);
    // Using std::swap for vector<bool> will cause the wrong result.
    std::swap(group_is_multi_val[i], group_is_multi_val[j]);
  }
  *multi_val_group = group_is_multi_val;
  return features_in_group;
}

void Dataset::Construct(std::vector<std::unique_ptr<BinMapper>>* bin_mappers,
                        int num_total_features,
                        const std::vector<std::vector<double>>& forced_bins,
                        int** sample_non_zero_indices, double** sample_values,
                        const int* num_per_col, int num_sample_col,
                        size_t total_sample_cnt, const Config& io_config) {
  num_total_features_ = num_total_features;
  CHECK_EQ(num_total_features_, static_cast<int>(bin_mappers->size()));
  // get num_features
  std::vector<int> used_features;
  auto& ref_bin_mappers = *bin_mappers;
  for (int i = 0; i < static_cast<int>(bin_mappers->size()); ++i) {
    if (ref_bin_mappers[i] != nullptr && !ref_bin_mappers[i]->is_trivial()) {
      used_features.emplace_back(i);
    }
  }
  if (used_features.empty()) {
    Log::Warning(
        "There are no meaningful features, as all feature values are "
        "constant.");
  }
  auto features_in_group = NoGroup(used_features);

  auto is_sparse = io_config.is_enable_sparse;
  if (io_config.device_type == std::string("cuda")) {
      LGBM_config_::current_device = lgbm_device_cuda;
      if (is_sparse) {
        Log::Warning("Using sparse features with CUDA is currently not supported.");
      }
      is_sparse = false;
  }

  std::vector<int8_t> group_is_multi_val(used_features.size(), 0);
  if (io_config.enable_bundle && !used_features.empty()) {
    bool lgbm_is_gpu_used = io_config.device_type == std::string("gpu") || io_config.device_type == std::string("cuda");
    features_in_group = FastFeatureBundling(
        *bin_mappers, sample_non_zero_indices, sample_values, num_per_col,
        num_sample_col, static_cast<data_size_t>(total_sample_cnt),
        used_features, num_data_, lgbm_is_gpu_used,
        is_sparse, &group_is_multi_val);
  }

  num_features_ = 0;
  for (const auto& fs : features_in_group) {
    num_features_ += static_cast<int>(fs.size());
  }
  int cur_fidx = 0;
  used_feature_map_ = std::vector<int>(num_total_features_, -1);
  num_groups_ = static_cast<int>(features_in_group.size());
  real_feature_idx_.resize(num_features_);
  feature2group_.resize(num_features_);
  feature2subfeature_.resize(num_features_);
  feature_need_push_zeros_.clear();
  group_bin_boundaries_.clear();
  uint64_t num_total_bin = 0;
  group_bin_boundaries_.push_back(num_total_bin);
  group_feature_start_.resize(num_groups_);
  group_feature_cnt_.resize(num_groups_);
  for (int i = 0; i < num_groups_; ++i) {
    auto cur_features = features_in_group[i];
    int cur_cnt_features = static_cast<int>(cur_features.size());
    group_feature_start_[i] = cur_fidx;
    group_feature_cnt_[i] = cur_cnt_features;
    // get bin_mappers
    std::vector<std::unique_ptr<BinMapper>> cur_bin_mappers;
    for (int j = 0; j < cur_cnt_features; ++j) {
      int real_fidx = cur_features[j];
      used_feature_map_[real_fidx] = cur_fidx;
      real_feature_idx_[cur_fidx] = real_fidx;
      feature2group_[cur_fidx] = i;
      feature2subfeature_[cur_fidx] = j;
      cur_bin_mappers.emplace_back(ref_bin_mappers[real_fidx].release());
      if (cur_bin_mappers.back()->GetDefaultBin() !=
          cur_bin_mappers.back()->GetMostFreqBin()) {
        feature_need_push_zeros_.push_back(cur_fidx);
      }
      ++cur_fidx;
    }
    feature_groups_.emplace_back(std::unique_ptr<FeatureGroup>(
      new FeatureGroup(cur_cnt_features, group_is_multi_val[i], &cur_bin_mappers, num_data_, i)));
    num_total_bin += feature_groups_[i]->num_total_bin_;
    group_bin_boundaries_.push_back(num_total_bin);
  }
  if (!io_config.max_bin_by_feature.empty()) {
    CHECK_EQ(static_cast<size_t>(num_total_features_),
             io_config.max_bin_by_feature.size());
    CHECK_GT(*(std::min_element(io_config.max_bin_by_feature.begin(),
                                io_config.max_bin_by_feature.end())), 1);
    max_bin_by_feature_.resize(num_total_features_);
    max_bin_by_feature_.assign(io_config.max_bin_by_feature.begin(),
                               io_config.max_bin_by_feature.end());
  }
  forced_bin_bounds_ = forced_bins;
  max_bin_ = io_config.max_bin;
  min_data_in_bin_ = io_config.min_data_in_bin;
  bin_construct_sample_cnt_ = io_config.bin_construct_sample_cnt;
  use_missing_ = io_config.use_missing;
  zero_as_missing_ = io_config.zero_as_missing;
  has_raw_ = false;
  if (io_config.linear_tree) {
    has_raw_ = true;
  }
  numeric_feature_map_ = std::vector<int>(num_features_, -1);
  num_numeric_features_ = 0;
  for (int i = 0; i < num_features_; ++i) {
    if (FeatureBinMapper(i)->bin_type() == BinType::NumericalBin) {
      numeric_feature_map_[i] = num_numeric_features_;
      ++num_numeric_features_;
    }
  }
}

void Dataset::FinishLoad() {
  if (is_finish_load_) {
    return;
  }
  if (num_groups_ > 0) {
    for (int i = 0; i < num_groups_; ++i) {
      feature_groups_[i]->FinishLoad();
    }
  }
  is_finish_load_ = true;
}

void PushDataToMultiValBin(
    data_size_t num_data, const std::vector<uint32_t> most_freq_bins,
    const std::vector<uint32_t> offsets,
    std::vector<std::vector<std::unique_ptr<BinIterator>>>* iters,
    MultiValBin* ret) {
  Common::FunctionTimer fun_time("Dataset::PushDataToMultiValBin",
                                 global_timer);
  if (ret->IsSparse()) {
    Threading::For<data_size_t>(
        0, num_data, 1024, [&](int tid, data_size_t start, data_size_t end) {
          std::vector<uint32_t> cur_data;
          cur_data.reserve(most_freq_bins.size());
          for (size_t j = 0; j < most_freq_bins.size(); ++j) {
            (*iters)[tid][j]->Reset(start);
          }
          for (data_size_t i = start; i < end; ++i) {
            cur_data.clear();
            for (size_t j = 0; j < most_freq_bins.size(); ++j) {
              // for sparse multi value bin, we store the feature bin values with offset added
              auto cur_bin = (*iters)[tid][j]->Get(i);
              if (cur_bin == most_freq_bins[j]) {
                continue;
              }
              cur_bin += offsets[j];
              if (most_freq_bins[j] == 0) {
                cur_bin -= 1;
              }
              cur_data.push_back(cur_bin);
            }
            ret->PushOneRow(tid, i, cur_data);
          }
        });
  } else {
    Threading::For<data_size_t>(
        0, num_data, 1024, [&](int tid, data_size_t start, data_size_t end) {
          std::vector<uint32_t> cur_data(most_freq_bins.size(), 0);
          for (size_t j = 0; j < most_freq_bins.size(); ++j) {
            (*iters)[tid][j]->Reset(start);
          }
          for (data_size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < most_freq_bins.size(); ++j) {
              // for dense multi value bin, the feature bin values without offsets are used
              auto cur_bin = (*iters)[tid][j]->Get(i);
              cur_data[j] = cur_bin;
            }
            ret->PushOneRow(tid, i, cur_data);
          }
        });
  }
}

MultiValBin* Dataset::GetMultiBinFromSparseFeatures(const std::vector<uint32_t>& offsets) const {
  Common::FunctionTimer fun_time("Dataset::GetMultiBinFromSparseFeatures",
                                 global_timer);
  int multi_group_id = -1;
  for (int i = 0; i < num_groups_; ++i) {
    if (feature_groups_[i]->is_multi_val_) {
      if (multi_group_id < 0) {
        multi_group_id = i;
      } else {
        Log::Fatal("Bug. There should be only one multi-val group.");
      }
    }
  }
  if (multi_group_id < 0) {
    return nullptr;
  }
  const int num_feature = feature_groups_[multi_group_id]->num_feature_;
  int num_threads = OMP_NUM_THREADS();

  std::vector<std::vector<std::unique_ptr<BinIterator>>> iters(num_threads);
  std::vector<uint32_t> most_freq_bins;
  double sum_sparse_rate = 0;
  for (int i = 0; i < num_feature; ++i) {
#pragma omp parallel for schedule(static, 1)
    for (int tid = 0; tid < num_threads; ++tid) {
      iters[tid].emplace_back(
          feature_groups_[multi_group_id]->SubFeatureIterator(i));
    }
    most_freq_bins.push_back(
        feature_groups_[multi_group_id]->bin_mappers_[i]->GetMostFreqBin());
    sum_sparse_rate +=
        feature_groups_[multi_group_id]->bin_mappers_[i]->sparse_rate();
  }
  sum_sparse_rate /= num_feature;
  Log::Debug("Dataset::GetMultiBinFromSparseFeatures: sparse rate %f",
             sum_sparse_rate);
  std::unique_ptr<MultiValBin> ret;
  ret.reset(MultiValBin::CreateMultiValBin(num_data_, offsets.back(),
                                           num_feature, sum_sparse_rate, offsets));
  PushDataToMultiValBin(num_data_, most_freq_bins, offsets, &iters, ret.get());
  ret->FinishLoad();
  return ret.release();
}

MultiValBin* Dataset::GetMultiBinFromAllFeatures(const std::vector<uint32_t>& offsets) const {
  Common::FunctionTimer fun_time("Dataset::GetMultiBinFromAllFeatures",
                                 global_timer);
  int num_threads = OMP_NUM_THREADS();
  double sum_dense_ratio = 0;

  std::unique_ptr<MultiValBin> ret;
  std::vector<std::vector<std::unique_ptr<BinIterator>>> iters(num_threads);
  std::vector<uint32_t> most_freq_bins;
  int ncol = 0;
  for (int gid = 0; gid < num_groups_; ++gid) {
    if (feature_groups_[gid]->is_multi_val_) {
      ncol += feature_groups_[gid]->num_feature_;
    } else {
      ++ncol;
    }
    for (int fid = 0; fid < feature_groups_[gid]->num_feature_; ++fid) {
      const auto& bin_mapper = feature_groups_[gid]->bin_mappers_[fid];
      sum_dense_ratio += 1.0f - bin_mapper->sparse_rate();
    }
  }
  sum_dense_ratio /= ncol;
  for (int gid = 0; gid < num_groups_; ++gid) {
    if (feature_groups_[gid]->is_multi_val_) {
      for (int fid = 0; fid < feature_groups_[gid]->num_feature_; ++fid) {
        const auto& bin_mapper = feature_groups_[gid]->bin_mappers_[fid];
        most_freq_bins.push_back(bin_mapper->GetMostFreqBin());
#pragma omp parallel for schedule(static, 1)
        for (int tid = 0; tid < num_threads; ++tid) {
          iters[tid].emplace_back(
              feature_groups_[gid]->SubFeatureIterator(fid));
        }
      }
    } else {
      most_freq_bins.push_back(0);
      for (int tid = 0; tid < num_threads; ++tid) {
        iters[tid].emplace_back(feature_groups_[gid]->FeatureGroupIterator());
      }
    }
  }
  CHECK(static_cast<int>(most_freq_bins.size()) == ncol);
  Log::Debug("Dataset::GetMultiBinFromAllFeatures: sparse rate %f",
             1.0 - sum_dense_ratio);
  ret.reset(MultiValBin::CreateMultiValBin(
      num_data_, offsets.back(), static_cast<int>(most_freq_bins.size()),
      1.0 - sum_dense_ratio, offsets));
  PushDataToMultiValBin(num_data_, most_freq_bins, offsets, &iters, ret.get());
  ret->FinishLoad();
  return ret.release();
}

TrainingShareStates* Dataset::GetShareStates(
    score_t* gradients, score_t* hessians,
    const std::vector<int8_t>& is_feature_used, bool is_constant_hessian,
    bool force_col_wise, bool force_row_wise) const {
  Common::FunctionTimer fun_timer("Dataset::TestMultiThreadingMethod",
                                  global_timer);
  if (force_col_wise && force_row_wise) {
    Log::Fatal(
        "Cannot set both of `force_col_wise` and `force_row_wise` to `true` at "
        "the same time");
  }
  if (num_groups_ <= 0) {
    TrainingShareStates* share_state = new TrainingShareStates();
    share_state->is_col_wise = true;
    share_state->is_constant_hessian = is_constant_hessian;
    return share_state;
  }
  if (force_col_wise) {
    TrainingShareStates* share_state = new TrainingShareStates();
    std::vector<uint32_t> offsets;
    share_state->CalcBinOffsets(
      feature_groups_, &offsets, true);
    share_state->SetMultiValBin(GetMultiBinFromSparseFeatures(offsets),
      num_data_, feature_groups_, false, true);
    share_state->is_col_wise = true;
    share_state->is_constant_hessian = is_constant_hessian;
    return share_state;
  } else if (force_row_wise) {
    TrainingShareStates* share_state = new TrainingShareStates();
    std::vector<uint32_t> offsets;
    share_state->CalcBinOffsets(
      feature_groups_, &offsets, false);
    share_state->SetMultiValBin(GetMultiBinFromAllFeatures(offsets), num_data_,
      feature_groups_, false, false);
    share_state->is_col_wise = false;
    share_state->is_constant_hessian = is_constant_hessian;
    return share_state;
  } else {
    std::unique_ptr<MultiValBin> sparse_bin;
    std::unique_ptr<MultiValBin> all_bin;
    std::unique_ptr<TrainingShareStates> col_wise_state;
    std::unique_ptr<TrainingShareStates> row_wise_state;
    col_wise_state.reset(new TrainingShareStates());
    row_wise_state.reset(new TrainingShareStates());

    std::chrono::duration<double, std::milli> col_wise_init_time, row_wise_init_time;
    auto start_time = std::chrono::steady_clock::now();
    std::vector<uint32_t> col_wise_offsets;
    col_wise_state->CalcBinOffsets(feature_groups_, &col_wise_offsets, true);
    col_wise_state->SetMultiValBin(GetMultiBinFromSparseFeatures(col_wise_offsets), num_data_,
      feature_groups_, false, true);
    col_wise_init_time = std::chrono::steady_clock::now() - start_time;

    start_time = std::chrono::steady_clock::now();
    std::vector<uint32_t> row_wise_offsets;
    row_wise_state->CalcBinOffsets(feature_groups_, &row_wise_offsets, false);
    row_wise_state->SetMultiValBin(GetMultiBinFromAllFeatures(row_wise_offsets), num_data_,
      feature_groups_, false, false);
    row_wise_init_time = std::chrono::steady_clock::now() - start_time;

    uint64_t max_total_bin = std::max<uint64_t>(row_wise_state->num_hist_total_bin(),
      col_wise_state->num_hist_total_bin());
    std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>
        hist_data(max_total_bin * 2);

    Log::Debug(
      "init for col-wise cost %f seconds, init for row-wise cost %f seconds",
      col_wise_init_time * 1e-3, row_wise_init_time * 1e-3);

    col_wise_state->is_col_wise = true;
    col_wise_state->is_constant_hessian = is_constant_hessian;
    InitTrain(is_feature_used, col_wise_state.get());
    row_wise_state->is_col_wise = false;
    row_wise_state->is_constant_hessian = is_constant_hessian;
    InitTrain(is_feature_used, row_wise_state.get());
    std::chrono::duration<double, std::milli> col_wise_time, row_wise_time;
    start_time = std::chrono::steady_clock::now();
    ConstructHistograms(is_feature_used, nullptr, num_data_, gradients,
                        hessians, gradients, hessians, col_wise_state.get(),
                        hist_data.data());
    col_wise_time = std::chrono::steady_clock::now() - start_time;
    start_time = std::chrono::steady_clock::now();
    ConstructHistograms(is_feature_used, nullptr, num_data_, gradients,
                        hessians, gradients, hessians, row_wise_state.get(),
                        hist_data.data());
    row_wise_time = std::chrono::steady_clock::now() - start_time;

    if (col_wise_time < row_wise_time) {
      auto overhead_cost = row_wise_init_time + row_wise_time + col_wise_time;
      Log::Warning(
          "Auto-choosing col-wise multi-threading, the overhead of testing was "
          "%f seconds.\n"
          "You can set `force_col_wise=true` to remove the overhead.",
          overhead_cost * 1e-3);
      return col_wise_state.release();
    } else {
      auto overhead_cost = col_wise_init_time + row_wise_time + col_wise_time;
      Log::Warning(
          "Auto-choosing row-wise multi-threading, the overhead of testing was "
          "%f seconds.\n"
          "You can set `force_row_wise=true` to remove the overhead.\n"
          "And if memory is not enough, you can set `force_col_wise=true`.",
          overhead_cost * 1e-3);
      if (row_wise_state->IsSparseRowwise()) {
        Log::Debug("Using Sparse Multi-Val Bin");
      } else {
        Log::Debug("Using Dense Multi-Val Bin");
      }
      return row_wise_state.release();
    }
  }
}

void Dataset::CopyFeatureMapperFrom(const Dataset* dataset) {
  feature_groups_.clear();
  num_features_ = dataset->num_features_;
  num_groups_ = dataset->num_groups_;
  has_raw_ = dataset->has_raw();
  // copy feature bin mapper data
  for (int i = 0; i < num_groups_; ++i) {
    feature_groups_.emplace_back(
        new FeatureGroup(*dataset->feature_groups_[i], num_data_));
  }
  feature_groups_.shrink_to_fit();
  used_feature_map_ = dataset->used_feature_map_;
  num_total_features_ = dataset->num_total_features_;
  feature_names_ = dataset->feature_names_;
  label_idx_ = dataset->label_idx_;
  real_feature_idx_ = dataset->real_feature_idx_;
  feature2group_ = dataset->feature2group_;
  feature2subfeature_ = dataset->feature2subfeature_;
  group_bin_boundaries_ = dataset->group_bin_boundaries_;
  group_feature_start_ = dataset->group_feature_start_;
  group_feature_cnt_ = dataset->group_feature_cnt_;
  forced_bin_bounds_ = dataset->forced_bin_bounds_;
  feature_need_push_zeros_ = dataset->feature_need_push_zeros_;
}

void Dataset::CreateValid(const Dataset* dataset) {
  feature_groups_.clear();
  num_features_ = dataset->num_features_;
  num_groups_ = num_features_;
  max_bin_ = dataset->max_bin_;
  min_data_in_bin_ = dataset->min_data_in_bin_;
  bin_construct_sample_cnt_ = dataset->bin_construct_sample_cnt_;
  use_missing_ = dataset->use_missing_;
  zero_as_missing_ = dataset->zero_as_missing_;
  feature2group_.clear();
  feature2subfeature_.clear();
  feature_need_push_zeros_ = dataset->feature_need_push_zeros_;
  has_raw_ = dataset->has_raw();
  numeric_feature_map_ = dataset->numeric_feature_map_;
  num_numeric_features_ = dataset->num_numeric_features_;
  // copy feature bin mapper data
  group_bin_boundaries_.clear();
  uint64_t num_total_bin = 0;
  group_bin_boundaries_.push_back(num_total_bin);
  group_feature_start_.resize(num_groups_);
  group_feature_cnt_.resize(num_groups_);
  // copy feature bin mapper data
  for (int i = 0; i < num_features_; ++i) {
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    bin_mappers.emplace_back(new BinMapper(*(dataset->FeatureBinMapper(i))));
    feature_groups_.emplace_back(new FeatureGroup(&bin_mappers, num_data_));
    feature2group_.push_back(i);
    feature2subfeature_.push_back(0);
    num_total_bin += feature_groups_[i]->num_total_bin_;
    group_bin_boundaries_.push_back(num_total_bin);
    group_feature_start_[i] = i;
    group_feature_cnt_[i] = 1;
  }

  feature_groups_.shrink_to_fit();
  used_feature_map_ = dataset->used_feature_map_;
  num_total_features_ = dataset->num_total_features_;
  feature_names_ = dataset->feature_names_;
  label_idx_ = dataset->label_idx_;
  real_feature_idx_ = dataset->real_feature_idx_;
  forced_bin_bounds_ = dataset->forced_bin_bounds_;
  if (dataset->category_encoding_provider() != nullptr) {
    category_encoding_provider_.reset(CategoryEncodingProvider::RecoverFromModelString(
      dataset->category_encoding_provider()->DumpToString()));
  }
}

void Dataset::ReSize(data_size_t num_data) {
  if (num_data_ != num_data) {
    num_data_ = num_data;
    OMP_INIT_EX();
#pragma omp parallel for schedule(static)
    for (int group = 0; group < num_groups_; ++group) {
      OMP_LOOP_EX_BEGIN();
      feature_groups_[group]->ReSize(num_data_);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
}

void Dataset::CopySubrow(const Dataset* fullset,
                         const data_size_t* used_indices,
                         data_size_t num_used_indices, bool need_meta_data) {
  CHECK_EQ(num_used_indices, num_data_);

  std::vector<int> group_ids, subfeature_ids;
  group_ids.reserve(num_features_);
  subfeature_ids.reserve(num_features_);
  for (int group = 0; group < num_groups_; ++group) {
    if (fullset->IsMultiGroup(group)) {
      for (int sub_feature = 0; sub_feature <
          fullset->feature_groups_[group]->num_feature_; ++sub_feature) {
        group_ids.emplace_back(group);
        subfeature_ids.emplace_back(sub_feature);
      }
    } else {
      group_ids.emplace_back(group);
      subfeature_ids.emplace_back(-1);
    }
  }
  int num_copy_tasks = static_cast<int>(group_ids.size());

  OMP_INIT_EX();
  #pragma omp parallel for schedule(dynamic)
  for (int task_id = 0; task_id < num_copy_tasks; ++task_id) {
    OMP_LOOP_EX_BEGIN();
    int group = group_ids[task_id];
    int subfeature = subfeature_ids[task_id];
    feature_groups_[group]->CopySubrowByCol(fullset->feature_groups_[group].get(),
                                            used_indices, num_used_indices, subfeature);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  if (need_meta_data) {
    metadata_.Init(fullset->metadata_, used_indices, num_used_indices);
  }
  is_finish_load_ = true;
  numeric_feature_map_ = fullset->numeric_feature_map_;
  num_numeric_features_ = fullset->num_numeric_features_;
  if (has_raw_) {
    ResizeRaw(num_used_indices);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_used_indices; ++i) {
      for (int j = 0; j < num_numeric_features_; ++j) {
        raw_data_[j][i] = fullset->raw_data_[j][used_indices[i]];
      }
    }
  }
}

bool Dataset::SetFloatField(const char* field_name, const float* field_data,
                            data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("label") || name == std::string("target")) {
#ifdef LABEL_T_USE_DOUBLE
    Log::Fatal("Don't support LABEL_T_USE_DOUBLE");
#else
    metadata_.SetLabel(field_data, num_element);
#endif
  } else if (name == std::string("weight") || name == std::string("weights")) {
#ifdef LABEL_T_USE_DOUBLE
    Log::Fatal("Don't support LABEL_T_USE_DOUBLE");
#else
    metadata_.SetWeights(field_data, num_element);
#endif
  } else {
    return false;
  }
  return true;
}

bool Dataset::SetDoubleField(const char* field_name, const double* field_data,
                             data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("init_score")) {
    metadata_.SetInitScore(field_data, num_element);
  } else {
    return false;
  }
  return true;
}

bool Dataset::SetIntField(const char* field_name, const int* field_data,
                          data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("query") || name == std::string("group")) {
    metadata_.SetQuery(field_data, num_element);
  } else {
    return false;
  }
  return true;
}

bool Dataset::GetFloatField(const char* field_name, data_size_t* out_len,
                            const float** out_ptr) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("label") || name == std::string("target")) {
#ifdef LABEL_T_USE_DOUBLE
    Log::Fatal("Don't support LABEL_T_USE_DOUBLE");
#else
    *out_ptr = metadata_.label();
    *out_len = num_data_;
#endif
  } else if (name == std::string("weight") || name == std::string("weights")) {
#ifdef LABEL_T_USE_DOUBLE
    Log::Fatal("Don't support LABEL_T_USE_DOUBLE");
#else
    *out_ptr = metadata_.weights();
    *out_len = num_data_;
#endif
  } else {
    return false;
  }
  return true;
}

bool Dataset::GetDoubleField(const char* field_name, data_size_t* out_len,
                             const double** out_ptr) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("init_score")) {
    *out_ptr = metadata_.init_score();
    *out_len = static_cast<data_size_t>(metadata_.num_init_score());
  } else {
    return false;
  }
  return true;
}

bool Dataset::GetIntField(const char* field_name, data_size_t* out_len,
                          const int** out_ptr) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("query") || name == std::string("group")) {
    *out_ptr = metadata_.query_boundaries();
    *out_len = metadata_.num_queries() + 1;
  } else {
    return false;
  }
  return true;
}

void Dataset::SaveBinaryFile(const char* bin_filename) {
  if (bin_filename != nullptr && std::string(bin_filename) == data_filename_) {
    Log::Warning("Binary file %s already exists", bin_filename);
    return;
  }
  // if not pass a filename, just append ".bin" of original file
  std::string bin_filename_str(data_filename_);
  if (bin_filename == nullptr || bin_filename[0] == '\0') {
    bin_filename_str.append(".bin");
    bin_filename = bin_filename_str.c_str();
  }
  bool is_file_existed = false;

  if (VirtualFileWriter::Exists(bin_filename)) {
    is_file_existed = true;
    Log::Warning("File %s exists, cannot save binary to it", bin_filename);
  }

  if (!is_file_existed) {
    auto writer = VirtualFileWriter::Make(bin_filename);
    if (!writer->Init()) {
      Log::Fatal("Cannot write binary data to %s ", bin_filename);
    }
    Log::Info("Saving data to binary file %s", bin_filename);
    size_t size_of_token = std::strlen(Parser::binary_file_token);
    writer->AlignedWrite(Parser::binary_file_token, size_of_token);
    // get size of header
    size_t size_of_header =
        VirtualFileWriter::AlignedSize(sizeof(num_data_)) +
        VirtualFileWriter::AlignedSize(sizeof(num_features_)) +
        VirtualFileWriter::AlignedSize(sizeof(num_total_features_)) +
        VirtualFileWriter::AlignedSize(sizeof(int) * num_total_features_) +
        VirtualFileWriter::AlignedSize(sizeof(label_idx_)) +
        VirtualFileWriter::AlignedSize(sizeof(num_groups_)) +
        3 * VirtualFileWriter::AlignedSize(sizeof(int) * num_features_) +
        sizeof(uint64_t) * (num_groups_ + 1) +
        2 * VirtualFileWriter::AlignedSize(sizeof(int) * num_groups_) +
        VirtualFileWriter::AlignedSize(sizeof(int32_t) * num_total_features_) +
        VirtualFileWriter::AlignedSize(sizeof(int)) * 3 +
        VirtualFileWriter::AlignedSize(sizeof(bool)) * 3;
    // size of feature names
    for (int i = 0; i < num_total_features_; ++i) {
      size_of_header +=
          VirtualFileWriter::AlignedSize(feature_names_[i].size()) +
          VirtualFileWriter::AlignedSize(sizeof(int));
    }
    // size of forced bins
    for (int i = 0; i < num_total_features_; ++i) {
      size_of_header += forced_bin_bounds_[i].size() * sizeof(double) +
                        VirtualFileWriter::AlignedSize(sizeof(int));
    }
    writer->Write(&size_of_header, sizeof(size_of_header));
    // write header
    writer->AlignedWrite(&num_data_, sizeof(num_data_));
    writer->AlignedWrite(&num_features_, sizeof(num_features_));
    writer->AlignedWrite(&num_total_features_, sizeof(num_total_features_));
    writer->AlignedWrite(&label_idx_, sizeof(label_idx_));
    writer->AlignedWrite(&max_bin_, sizeof(max_bin_));
    writer->AlignedWrite(&bin_construct_sample_cnt_,
                         sizeof(bin_construct_sample_cnt_));
    writer->AlignedWrite(&min_data_in_bin_, sizeof(min_data_in_bin_));
    writer->AlignedWrite(&use_missing_, sizeof(use_missing_));
    writer->AlignedWrite(&zero_as_missing_, sizeof(zero_as_missing_));
    writer->AlignedWrite(&has_raw_, sizeof(has_raw_));
    writer->AlignedWrite(used_feature_map_.data(),
                         sizeof(int) * num_total_features_);
    writer->AlignedWrite(&num_groups_, sizeof(num_groups_));
    writer->AlignedWrite(real_feature_idx_.data(), sizeof(int) * num_features_);
    writer->AlignedWrite(feature2group_.data(), sizeof(int) * num_features_);
    writer->AlignedWrite(feature2subfeature_.data(),
                         sizeof(int) * num_features_);
    writer->Write(group_bin_boundaries_.data(),
                  sizeof(uint64_t) * (num_groups_ + 1));
    writer->AlignedWrite(group_feature_start_.data(),
                         sizeof(int) * num_groups_);
    writer->AlignedWrite(group_feature_cnt_.data(), sizeof(int) * num_groups_);
    if (max_bin_by_feature_.empty()) {
      ArrayArgs<int32_t>::Assign(&max_bin_by_feature_, -1, num_total_features_);
    }
    writer->AlignedWrite(max_bin_by_feature_.data(),
                  sizeof(int32_t) * num_total_features_);
    if (ArrayArgs<int32_t>::CheckAll(max_bin_by_feature_, -1)) {
      max_bin_by_feature_.clear();
    }
    // write feature names
    for (int i = 0; i < num_total_features_; ++i) {
      int str_len = static_cast<int>(feature_names_[i].size());
      writer->AlignedWrite(&str_len, sizeof(int));
      const char* c_str = feature_names_[i].c_str();
      writer->AlignedWrite(c_str, sizeof(char) * str_len);
    }
    // write forced bins
    for (int i = 0; i < num_total_features_; ++i) {
      int num_bounds = static_cast<int>(forced_bin_bounds_[i].size());
      writer->AlignedWrite(&num_bounds, sizeof(int));

      for (size_t j = 0; j < forced_bin_bounds_[i].size(); ++j) {
        writer->Write(&forced_bin_bounds_[i][j], sizeof(double));
      }
    }

    // get size of meta data
    size_t size_of_metadata = metadata_.SizesInByte();
    writer->Write(&size_of_metadata, sizeof(size_of_metadata));
    // write meta data
    metadata_.SaveBinaryToFile(writer.get());

    // write feature data
    for (int i = 0; i < num_groups_; ++i) {
      // get size of feature
      size_t size_of_feature = feature_groups_[i]->SizesInByte();
      writer->Write(&size_of_feature, sizeof(size_of_feature));
      // write feature
      feature_groups_[i]->SaveBinaryToFile(writer.get());
    }
    if (category_encoding_provider_ != nullptr) {
      const std::string category_encoding_provider_str = category_encoding_provider_->DumpToString();
      const size_t category_encoding_provider_str_size_in_bytes = category_encoding_provider_str.size() * sizeof(char);
      writer->Write(&category_encoding_provider_str_size_in_bytes, sizeof(category_encoding_provider_str_size_in_bytes));
      writer->Write(category_encoding_provider_str.c_str(), category_encoding_provider_str_size_in_bytes);
    } else {
      const size_t category_encoding_provider_str_size_in_bytes = 0;
      writer->Write(&category_encoding_provider_str_size_in_bytes, sizeof(category_encoding_provider_str_size_in_bytes));
    }
    // write raw data; use row-major order so we can read row-by-row
    if (has_raw_) {
      for (int i = 0; i < num_data_; ++i) {
        for (int j = 0; j < num_features_; ++j) {
          int feat_ind = numeric_feature_map_[j];
          if (feat_ind > -1) {
            writer->Write(&raw_data_[feat_ind][i], sizeof(float));
          }
        }
      }
    }
  }
}

void Dataset::DumpTextFile(const char* text_filename) {
  FILE* file = NULL;
#if _MSC_VER
  fopen_s(&file, text_filename, "wt");
#else
  file = fopen(text_filename, "wt");
#endif
  fprintf(file, "num_features: %d\n", num_features_);
  fprintf(file, "num_total_features: %d\n", num_total_features_);
  fprintf(file, "num_groups: %d\n", num_groups_);
  fprintf(file, "num_data: %d\n", num_data_);
  fprintf(file, "feature_names: ");
  for (auto n : feature_names_) {
    fprintf(file, "%s, ", n.c_str());
  }
  fprintf(file, "\nmax_bin_by_feature: ");
  for (auto i : max_bin_by_feature_) {
    fprintf(file, "%d, ", i);
  }
  fprintf(file, "\n");
  for (auto n : feature_names_) {
    fprintf(file, "%s, ", n.c_str());
  }
  fprintf(file, "\nforced_bins: ");
  for (int i = 0; i < num_total_features_; ++i) {
    fprintf(file, "\nfeature %d: ", i);
    for (size_t j = 0; j < forced_bin_bounds_[i].size(); ++j) {
      fprintf(file, "%lf, ", forced_bin_bounds_[i][j]);
    }
  }
  std::vector<std::unique_ptr<BinIterator>> iterators;
  iterators.reserve(num_features_);
  for (int j = 0; j < num_features_; ++j) {
    auto group_idx = feature2group_[j];
    auto sub_idx = feature2subfeature_[j];
    iterators.emplace_back(
        feature_groups_[group_idx]->SubFeatureIterator(sub_idx));
  }
  for (data_size_t i = 0; i < num_data_; ++i) {
    fprintf(file, "\n");
    for (int j = 0; j < num_total_features_; ++j) {
      auto inner_feature_idx = used_feature_map_[j];
      if (inner_feature_idx < 0) {
        fprintf(file, "NA, ");
      } else {
        fprintf(file, "%d, ", iterators[inner_feature_idx]->Get(i));
      }
    }
  }
  fclose(file);
}

void Dataset::InitTrain(const std::vector<int8_t>& is_feature_used,
                        TrainingShareStates* share_state) const {
  Common::FunctionTimer fun_time("Dataset::InitTrain", global_timer);
  share_state->InitTrain(group_feature_start_,
        feature_groups_,
        is_feature_used);
}

template <bool USE_INDICES, bool ORDERED>
void Dataset::ConstructHistogramsMultiVal(const data_size_t* data_indices,
                                          data_size_t num_data,
                                          const score_t* gradients,
                                          const score_t* hessians,
                                          TrainingShareStates* share_state,
                                          hist_t* hist_data) const {
  Common::FunctionTimer fun_time("Dataset::ConstructHistogramsMultiVal",
                                 global_timer);
  share_state->ConstructHistograms<USE_INDICES, ORDERED>(
      data_indices, num_data, gradients, hessians, hist_data);
}

template <bool USE_INDICES, bool USE_HESSIAN>
void Dataset::ConstructHistogramsInner(
    const std::vector<int8_t>& is_feature_used, const data_size_t* data_indices,
    data_size_t num_data, const score_t* gradients, const score_t* hessians,
    score_t* ordered_gradients, score_t* ordered_hessians,
    TrainingShareStates* share_state, hist_t* hist_data) const {
  if (!share_state->is_col_wise) {
    return ConstructHistogramsMultiVal<USE_INDICES, false>(
        data_indices, num_data, gradients, hessians, share_state, hist_data);
  }
  std::vector<int> used_dense_group;
  int multi_val_groud_id = -1;
  used_dense_group.reserve(num_groups_);
  for (int group = 0; group < num_groups_; ++group) {
    const int f_start = group_feature_start_[group];
    const int f_cnt = group_feature_cnt_[group];
    bool is_group_used = false;
    for (int j = 0; j < f_cnt; ++j) {
      const int fidx = f_start + j;
      if (is_feature_used[fidx]) {
        is_group_used = true;
        break;
      }
    }
    if (is_group_used) {
      if (feature_groups_[group]->is_multi_val_) {
        multi_val_groud_id = group;
      } else {
        used_dense_group.push_back(group);
      }
    }
  }
  int num_used_dense_group = static_cast<int>(used_dense_group.size());
  global_timer.Start("Dataset::dense_bin_histogram");
  auto ptr_ordered_grad = gradients;
  auto ptr_ordered_hess = hessians;
  if (num_used_dense_group > 0) {
    if (USE_INDICES) {
      if (USE_HESSIAN) {
#pragma omp parallel for schedule(static, 512) if (num_data >= 1024)
        for (data_size_t i = 0; i < num_data; ++i) {
          ordered_gradients[i] = gradients[data_indices[i]];
          ordered_hessians[i] = hessians[data_indices[i]];
        }
        ptr_ordered_grad = ordered_gradients;
        ptr_ordered_hess = ordered_hessians;
      } else {
#pragma omp parallel for schedule(static, 512) if (num_data >= 1024)
        for (data_size_t i = 0; i < num_data; ++i) {
          ordered_gradients[i] = gradients[data_indices[i]];
        }
        ptr_ordered_grad = ordered_gradients;
      }
    }
    OMP_INIT_EX();
#pragma omp parallel for schedule(static) num_threads(share_state->num_threads)
    for (int gi = 0; gi < num_used_dense_group; ++gi) {
      OMP_LOOP_EX_BEGIN();
      int group = used_dense_group[gi];
      auto data_ptr = hist_data + group_bin_boundaries_[group] * 2;
      const int num_bin = feature_groups_[group]->num_total_bin_;
      std::memset(reinterpret_cast<void*>(data_ptr), 0,
                  num_bin * kHistEntrySize);
      if (USE_HESSIAN) {
        if (USE_INDICES) {
          feature_groups_[group]->bin_data_->ConstructHistogram(
              data_indices, 0, num_data, ptr_ordered_grad, ptr_ordered_hess,
              data_ptr);
        } else {
          feature_groups_[group]->bin_data_->ConstructHistogram(
              0, num_data, ptr_ordered_grad, ptr_ordered_hess, data_ptr);
        }
      } else {
        if (USE_INDICES) {
          feature_groups_[group]->bin_data_->ConstructHistogram(
              data_indices, 0, num_data, ptr_ordered_grad, data_ptr);
        } else {
          feature_groups_[group]->bin_data_->ConstructHistogram(
              0, num_data, ptr_ordered_grad, data_ptr);
        }
        auto cnt_dst = reinterpret_cast<hist_cnt_t*>(data_ptr + 1);
        for (int i = 0; i < num_bin * 2; i += 2) {
          data_ptr[i + 1] = static_cast<double>(cnt_dst[i]) * hessians[0];
        }
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
  global_timer.Stop("Dataset::dense_bin_histogram");
  if (multi_val_groud_id >= 0) {
    if (num_used_dense_group > 0) {
      ConstructHistogramsMultiVal<USE_INDICES, true>(
          data_indices, num_data, ptr_ordered_grad, ptr_ordered_hess,
          share_state,
          hist_data + group_bin_boundaries_[multi_val_groud_id] * 2);
    } else {
      ConstructHistogramsMultiVal<USE_INDICES, false>(
          data_indices, num_data, gradients, hessians, share_state,
          hist_data + group_bin_boundaries_[multi_val_groud_id] * 2);
    }
  }
}

// explicitly initialize template methods, for cross module call
template void Dataset::ConstructHistogramsInner<true, true>(
    const std::vector<int8_t>& is_feature_used, const data_size_t* data_indices,
    data_size_t num_data, const score_t* gradients, const score_t* hessians,
    score_t* ordered_gradients, score_t* ordered_hessians,
    TrainingShareStates* share_state, hist_t* hist_data) const;

template void Dataset::ConstructHistogramsInner<true, false>(
    const std::vector<int8_t>& is_feature_used, const data_size_t* data_indices,
    data_size_t num_data, const score_t* gradients, const score_t* hessians,
    score_t* ordered_gradients, score_t* ordered_hessians,
    TrainingShareStates* share_state, hist_t* hist_data) const;

template void Dataset::ConstructHistogramsInner<false, true>(
    const std::vector<int8_t>& is_feature_used, const data_size_t* data_indices,
    data_size_t num_data, const score_t* gradients, const score_t* hessians,
    score_t* ordered_gradients, score_t* ordered_hessians,
    TrainingShareStates* share_state, hist_t* hist_data) const;

template void Dataset::ConstructHistogramsInner<false, false>(
    const std::vector<int8_t>& is_feature_used, const data_size_t* data_indices,
    data_size_t num_data, const score_t* gradients, const score_t* hessians,
    score_t* ordered_gradients, score_t* ordered_hessians,
    TrainingShareStates* share_state, hist_t* hist_data) const;

void Dataset::FixHistogram(int feature_idx, double sum_gradient,
                           double sum_hessian, hist_t* data) const {
  const int group = feature2group_[feature_idx];
  const int sub_feature = feature2subfeature_[feature_idx];
  const BinMapper* bin_mapper =
      feature_groups_[group]->bin_mappers_[sub_feature].get();
  const int most_freq_bin = bin_mapper->GetMostFreqBin();
  if (most_freq_bin > 0) {
    const int num_bin = bin_mapper->num_bin();
    GET_GRAD(data, most_freq_bin) = sum_gradient;
    GET_HESS(data, most_freq_bin) = sum_hessian;
    for (int i = 0; i < num_bin; ++i) {
      if (i != most_freq_bin) {
        GET_GRAD(data, most_freq_bin) -= GET_GRAD(data, i);
        GET_HESS(data, most_freq_bin) -= GET_HESS(data, i);
      }
    }
  }
}

template <typename T>
void PushVector(std::vector<T>* dest, const std::vector<T>& src) {
  dest->reserve(dest->size() + src.size());
  for (auto i : src) {
    dest->push_back(i);
  }
}

template <typename T>
void PushOffset(std::vector<T>* dest, const std::vector<T>& src,
                const T& offset) {
  dest->reserve(dest->size() + src.size());
  for (auto i : src) {
    dest->push_back(i + offset);
  }
}

template <typename T>
void PushClearIfEmpty(std::vector<T>* dest, const size_t dest_len,
                      const std::vector<T>& src, const size_t src_len,
                      const T& deflt) {
  if (!dest->empty() && !src.empty()) {
    PushVector(dest, src);
  } else if (!dest->empty() && src.empty()) {
    for (size_t i = 0; i < src_len; ++i) {
      dest->push_back(deflt);
    }
  } else if (dest->empty() && !src.empty()) {
    for (size_t i = 0; i < dest_len; ++i) {
      dest->push_back(deflt);
    }
    PushVector(dest, src);
  }
}

void Dataset::AddFeaturesFrom(Dataset* other) {
  if (other->num_data_ != num_data_) {
    Log::Fatal(
        "Cannot add features from other Dataset with a different number of "
        "rows");
  }
  if (other->has_raw_ != has_raw_) {
    Log::Fatal("Can only add features from other Dataset if both or neither have raw data.");
  }
  int mv_gid = -1;
  int other_mv_gid = -1;
  for (int i = 0; i < num_groups_; ++i) {
    if (IsMultiGroup(i)) {
      mv_gid = i;
    }
  }
  for (int i = 0; i < other->num_groups_; ++i) {
    if (other->IsMultiGroup(i)) {
      other_mv_gid = i;
    }
  }
  // Only one multi-val group, just simply merge
  if (mv_gid < 0 || other_mv_gid < 0) {
    PushVector(&feature2subfeature_, other->feature2subfeature_);
    PushVector(&group_feature_cnt_, other->group_feature_cnt_);
    feature_groups_.reserve(other->feature_groups_.size());
    for (auto& fg : other->feature_groups_) {
      const int cur_group_id = static_cast<int>(feature_groups_.size());
      feature_groups_.emplace_back(new FeatureGroup(*fg, true, cur_group_id));
    }
    for (auto feature_idx : other->used_feature_map_) {
      if (feature_idx >= 0) {
        used_feature_map_.push_back(feature_idx + num_features_);
      } else {
        used_feature_map_.push_back(-1);  // Unused feature.
      }
    }
    PushOffset(&real_feature_idx_, other->real_feature_idx_,
               num_total_features_);
    PushOffset(&feature2group_, other->feature2group_, num_groups_);
    auto bin_offset = group_bin_boundaries_.back();
    // Skip the leading 0 when copying group_bin_boundaries.
    for (auto i = other->group_bin_boundaries_.begin() + 1;
         i < other->group_bin_boundaries_.end(); ++i) {
      group_bin_boundaries_.push_back(*i + bin_offset);
    }
    PushOffset(&group_feature_start_, other->group_feature_start_,
               num_features_);
    num_groups_ += other->num_groups_;
    num_features_ += other->num_features_;
  } else {
    std::vector<std::vector<int>> features_in_group;
    for (int i = 0; i < num_groups_; ++i) {
      int f_start = group_feature_start_[i];
      int f_cnt = group_feature_cnt_[i];
      features_in_group.emplace_back();
      for (int j = 0; j < f_cnt; ++j) {
        const int real_fidx = real_feature_idx_[f_start + j];
        features_in_group.back().push_back(real_fidx);
      }
    }
    feature_groups_[mv_gid]->AddFeaturesFrom(
        other->feature_groups_[other_mv_gid].get(), mv_gid);
    for (int i = 0; i < other->num_groups_; ++i) {
      int f_start = other->group_feature_start_[i];
      int f_cnt = other->group_feature_cnt_[i];
      if (i == other_mv_gid) {
        for (int j = 0; j < f_cnt; ++j) {
          const int real_fidx = other->real_feature_idx_[f_start + j] + num_total_features_;
          features_in_group[mv_gid].push_back(real_fidx);
        }
      } else {
        features_in_group.emplace_back();
        for (int j = 0; j < f_cnt; ++j) {
          const int real_fidx = other->real_feature_idx_[f_start + j] + num_total_features_;
          features_in_group.back().push_back(real_fidx);
        }
        feature_groups_.emplace_back(
            new FeatureGroup(*other->feature_groups_[i], false, -1));
      }
    }
    // regenerate other fields
    num_groups_ += other->num_groups_ - 1;
    CHECK(num_groups_ == static_cast<int>(features_in_group.size()));
    num_features_ += other->num_features_;
    int cur_fidx = 0;
    used_feature_map_ =
      std::vector<int>(num_total_features_ + other->num_total_features_, -1);
    real_feature_idx_.resize(num_features_);
    feature2group_.resize(num_features_);
    feature2subfeature_.resize(num_features_);
    group_feature_start_.resize(num_groups_);
    group_feature_cnt_.resize(num_groups_);

    group_bin_boundaries_.clear();
    uint64_t num_total_bin = 0;
    group_bin_boundaries_.push_back(num_total_bin);
    for (int i = 0; i < num_groups_; ++i) {
      auto cur_features = features_in_group[i];
      int cur_cnt_features = static_cast<int>(cur_features.size());
      group_feature_start_[i] = cur_fidx;
      group_feature_cnt_[i] = cur_cnt_features;
      for (int j = 0; j < cur_cnt_features; ++j) {
        int real_fidx = cur_features[j];
        used_feature_map_[real_fidx] = cur_fidx;
        real_feature_idx_[cur_fidx] = real_fidx;
        feature2group_[cur_fidx] = i;
        feature2subfeature_[cur_fidx] = j;
        ++cur_fidx;
      }
      num_total_bin += feature_groups_[i]->num_total_bin_;
      group_bin_boundaries_.push_back(num_total_bin);
    }
  }
  std::unordered_set<std::string> feature_names_set;
  for (const auto& val : feature_names_) {
    feature_names_set.emplace(val);
  }
  for (const auto& val : other->feature_names_) {
    std::string new_name = val;
    int cnt = 2;
    while (feature_names_set.count(new_name)) {
      new_name = "D" + std::to_string(cnt) + "_" + val;
      ++cnt;
    }
    if (new_name != val) {
      Log::Warning(
        "Find the same feature name (%s) in Dataset::AddFeaturesFrom, change "
        "its name to (%s)",
        val.c_str(), new_name.c_str());
    }
    feature_names_set.emplace(new_name);
    feature_names_.push_back(new_name);
  }
  PushVector(&forced_bin_bounds_, other->forced_bin_bounds_);
  PushClearIfEmpty(&max_bin_by_feature_, num_total_features_,
                   other->max_bin_by_feature_, other->num_total_features_, -1);
  num_total_features_ += other->num_total_features_;
  for (size_t i = 0; i < (other->numeric_feature_map_).size(); ++i) {
    int feat_ind = numeric_feature_map_[i];
    if (feat_ind > -1) {
      numeric_feature_map_.push_back(feat_ind + num_numeric_features_);
    } else {
      numeric_feature_map_.push_back(-1);
    }
  }
  num_numeric_features_ += other->num_numeric_features_;
  if (has_raw_) {
    for (int i = 0; i < other->num_numeric_features_; ++i) {
      raw_data_.push_back(other->raw_data_[i]);
    }
  }
}

}  // namespace LightGBM
