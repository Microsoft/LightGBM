/*!
 * Copyright (c) 2024 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_

#include "multi_val_dense_bin.hpp"

namespace LightGBM {

template <typename BIN_TYPE, template<typename> class MULTI_VAL_BIN_TYPE>
class MultiValPairwiseLambdarankBin : public MULTI_VAL_BIN_TYPE<BIN_TYPE> {
 public:
  MultiValPairwiseLambdarankBin(data_size_t num_data, int num_bin, int num_feature,
                                const std::vector<uint32_t>& offsets)
      : MULTI_VAL_BIN_TYPE<BIN_TYPE>(num_data, num_bin, num_feature, offsets) {
    this->num_bin_ = num_bin;
    Log::Warning("num_bin = %d", num_bin);
  }

 protected:
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map_;
};

template <typename BIN_TYPE>
class MultiValDensePairwiseLambdarankBin
    : public MultiValPairwiseLambdarankBin<BIN_TYPE, MultiValDenseBin> {
 public:
  MultiValDensePairwiseLambdarankBin(
      data_size_t num_data, int num_bin, int num_feature,
      const std::vector<uint32_t>& offsets,
      const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map,
      const std::vector<const BinMapper*> diff_feature_bin_mappers,
      const std::vector<const BinMapper*> original_feature_bin_mappers,
      const bool use_pairwise_bin_lookup,
      const std::vector<std::vector<float>>* raw_data,
      const std::vector<uint32_t>& all_offsets,
      const std::vector<int>& diff_feature_to_original_feature_slot,
      const std::vector<int>& diff_feature_to_raw_feature_index)
      : MultiValPairwiseLambdarankBin<BIN_TYPE, LightGBM::MultiValDenseBin>(
            num_data, num_bin, num_feature, offsets),
        use_pairwise_bin_lookup_(use_pairwise_bin_lookup),
        diff_feature_to_original_feature_slot_(diff_feature_to_original_feature_slot) {
    this->paired_ranking_item_global_index_map_ =
        paired_ranking_item_global_index_map;
    CHECK_EQ(diff_feature_bin_mappers.size(), diff_feature_to_original_feature_slot.size());
    CHECK_EQ(diff_feature_bin_mappers.size(), diff_feature_to_raw_feature_index.size());
    num_diff_features_ = static_cast<int>(diff_feature_bin_mappers.size());
    diff_feature_bin_mappers_.reserve(num_diff_features_);
    diff_feature_hist_offsets_.reserve(num_diff_features_);
    diff_feature_bin_value_offsets_.reserve(num_diff_features_);
    diff_feature_raw_data_ptrs_.reserve(num_diff_features_);
    for (int j = 0; j < num_diff_features_; ++j) {
      diff_feature_bin_mappers_.emplace_back(diff_feature_bin_mappers[j]);
      diff_feature_hist_offsets_.push_back(all_offsets[2 * this->num_feature_ + j]);
      diff_feature_bin_value_offsets_.push_back(
          1u - static_cast<uint32_t>(diff_feature_bin_mappers_[j]->GetMostFreqBin() == 0));
      if (raw_data != nullptr && !raw_data->empty()) {
        const int raw_feature_index = diff_feature_to_raw_feature_index[j];
        CHECK_GE(raw_feature_index, 0);
        CHECK_LT(raw_feature_index, static_cast<int>(raw_data->size()));
        diff_feature_raw_data_ptrs_.push_back((*raw_data)[raw_feature_index].data());
      } else {
        diff_feature_raw_data_ptrs_.push_back(nullptr);
      }
    }

    original_feature_to_diff_feature_slot_.resize(this->num_feature_, -1);
    for (size_t i = 0; i < diff_feature_to_original_feature_slot_.size(); ++i) {
      original_feature_to_diff_feature_slot_[diff_feature_to_original_feature_slot_[i]] = static_cast<int>(i);
      Log::Warning("i = %ld, diff_feature_to_original_feature_slot_[%ld] = %d", i, i, diff_feature_to_original_feature_slot_[i]);
    }

    (void)original_feature_bin_mappers;
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* gradients,
                          const score_t* hessians, hist_t* out) const override {
    ConstructHistogramInner<true, true, false>(data_indices, start, end,
                                               gradients, hessians, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* gradients, const score_t* hessians,
                          hist_t* out) const override {
    ConstructHistogramInner<false, false, false>(nullptr, start, end, gradients,
                                                 hessians, out);
  }

  void ConstructHistogramOrdered(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* hessians,
                                 hist_t* out) const override {
    ConstructHistogramInner<true, true, true>(data_indices, start, end,
                                              gradients, hessians, out);
  }

  template <bool USE_INDICES, bool USE_PREFETCH, bool ORDERED>
  void ConstructHistogramInner(const data_size_t* data_indices,
                               data_size_t start, data_size_t end,
                               const score_t* gradients,
                               const score_t* hessians, hist_t* out) const {
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;

    if (!diff_feature_raw_data_ptrs_.empty() && use_pairwise_bin_lookup_) {
      for (; i < end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        const data_size_t first_idx =
            this->paired_ranking_item_global_index_map_[idx].first;
        const data_size_t second_idx =
            this->paired_ranking_item_global_index_map_[idx].second;
        const auto first_j_start = this->RowPtr(first_idx);
        const BIN_TYPE* first_data_ptr = this->data_.data() + first_j_start;
        const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[i] : hessians[idx];

        const auto second_j_start = this->RowPtr(second_idx);
        const BIN_TYPE* second_data_ptr = this->data_.data() + second_j_start;
        const auto base_offset = this->offsets_.back();

        for (int j = 0; j < this->num_feature_; ++j) {
          uint32_t first_bin = static_cast<uint32_t>(first_data_ptr[j]);
          const auto first_ti = (first_bin + this->offsets_[j]) << 1;
          grad[first_ti] += gradient;
          hess[first_ti] += hessian;

          uint32_t second_bin = static_cast<uint32_t>(second_data_ptr[j]);
          const auto second_ti = (second_bin + this->offsets_[j] + base_offset) << 1;
          grad[second_ti] += gradient;
          hess[second_ti] += hessian;

          const float* feature_values = diff_feature_raw_data_ptrs_[j];
          if (feature_values == nullptr) {
            continue;
          }
          const double diff_value =
              static_cast<double>(feature_values[first_idx]) -
              static_cast<double>(feature_values[second_idx]);
          const int diff_feature_slot =
              original_feature_to_diff_feature_slot_[j];
          if (diff_feature_slot == -1) {
            continue;
          }

          const uint32_t diff_bin =
              diff_feature_bin_mappers_[diff_feature_slot]->ValueToBinWithPairwiseRange(
                  diff_value, first_bin, second_bin);
          // The original row-wise bins already exclude feature-group offsets.
          // Differential features still need the single-feature-group packing fix below.
          const uint32_t bin = diff_bin + diff_feature_bin_value_offsets_[diff_feature_slot];
          const auto ti = (bin + diff_feature_hist_offsets_[diff_feature_slot]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    } else {
      for (; i < end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        const data_size_t first_idx =
            this->paired_ranking_item_global_index_map_[idx].first;
        const data_size_t second_idx =
            this->paired_ranking_item_global_index_map_[idx].second;
        const auto first_j_start = this->RowPtr(first_idx);
        const BIN_TYPE* first_data_ptr = this->data_.data() + first_j_start;
        const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[i] : hessians[idx];
        for (int j = 0; j < this->num_feature_; ++j) {
          const uint32_t bin = static_cast<uint32_t>(first_data_ptr[j]);
          const auto ti = (bin + this->offsets_[j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }

        const auto second_j_start = this->RowPtr(second_idx);
        const BIN_TYPE* second_data_ptr = this->data_.data() + second_j_start;
        const auto base_offset = this->offsets_.back();
        for (int j = 0; j < this->num_feature_; ++j) {
          const uint32_t bin = static_cast<uint32_t>(second_data_ptr[j]);
          const auto ti = (bin + this->offsets_[j] + base_offset) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }

        if (diff_feature_raw_data_ptrs_.empty()) {
          continue;
        }
        for (int j = 0; j < num_diff_features_; ++j) {
          const float* feature_values = diff_feature_raw_data_ptrs_[j];
          if (feature_values == nullptr) {
            continue;
          }
          const double diff_value =
              static_cast<double>(feature_values[first_idx]) -
              static_cast<double>(feature_values[second_idx]);
          const uint32_t diff_bin = diff_feature_bin_mappers_[j]->ValueToBin(diff_value);
          // imitates push the bin into a single feature group (+ min_bin_ (which is always 1 for single feature groups) - offset)
          // and then extracted in a FeatureGroupIteartor (+ min_bin_ (which is 1 for FeatureGroupIterator) - offset (which is 1 for FeatureGroupIterator))
          // thus effectively should do only (+ 1 - offset)
          const uint32_t bin = diff_bin + diff_feature_bin_value_offsets_[j];
          const auto ti = (bin + diff_feature_hist_offsets_[j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
  }

 private:
  std::vector<std::unique_ptr<const BinMapper>> diff_feature_bin_mappers_;
  std::vector<int> diff_feature_to_original_feature_slot_;
  std::vector<int> original_feature_to_diff_feature_slot_;
  std::vector<uint32_t> diff_feature_hist_offsets_;
  std::vector<uint32_t> diff_feature_bin_value_offsets_;
  std::vector<const float*> diff_feature_raw_data_ptrs_;
  int num_diff_features_;
  const bool use_pairwise_bin_lookup_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
