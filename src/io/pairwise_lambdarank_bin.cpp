/*!
 * Copyright (c) 2024 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "pairwise_lambdarank_bin.hpp"

#include <algorithm>

namespace LightGBM {

template <typename VAL_T, bool IS_4BIT>
uint32_t DensePairwiseRankingDiffBin<VAL_T, IS_4BIT>::GetBinAt(const data_size_t paired_data_index) const {
  const data_size_t first_data_index = this->paired_ranking_item_index_map_[paired_data_index].first;
  const data_size_t second_data_index = this->paired_ranking_item_index_map_[paired_data_index].second;

  // Log::Warning("raw_values_ == nullptr = %d", static_cast<int>(raw_values_ == nullptr));

  double diff_value = 0.0;

  if (raw_values_ == nullptr) {
    const uint32_t first_bin = static_cast<uint32_t>(this->unpaired_bin_->data(first_data_index));
    const uint32_t second_bin = static_cast<uint32_t>(this->unpaired_bin_->data(second_data_index));
    int first_feature_index = static_cast<int>(std::upper_bound(bin_offsets_->begin(), bin_offsets_->end(), first_bin) - bin_offsets_->begin()) - 1;
    int second_feature_index = static_cast<int>(std::upper_bound(bin_offsets_->begin(), bin_offsets_->end(), second_bin) - bin_offsets_->begin()) - 1;

    // TODO(shiyu1994): better original value, handle nan as missing
    const double first_value = first_feature_index >= 0 ? ori_bin_mappers_->at(first_feature_index)->BinToValue(first_bin) : 0.0;
    const double second_value = second_feature_index >= 0 ? ori_bin_mappers_->at(second_feature_index)->BinToValue(second_bin) : 0.0;
    diff_value = first_value - second_value;

    CHECK(first_feature_index >= 0 || first_bin == 0);
    if (first_feature_index >= 0 && first_feature_index == second_feature_index) {
      const uint32_t min_bin = diff_bin_offsets_->at(first_feature_index);
      const uint32_t most_freq_bin = diff_bin_mappers_->at(first_feature_index)->GetMostFreqBin();
      const uint32_t diff_bin = diff_bin_mappers_->at(first_feature_index)->ValueToBin(diff_value);
      return diff_bin + min_bin - static_cast<uint32_t>(most_freq_bin == 0);
    } else {
      return 0;
    }
  } else {
    const double first_value = raw_values_->at(first_data_index);
    const double second_value = raw_values_->at(second_data_index);
    diff_value = first_value - second_value;
    // Log::Warning("first_value = %f, second_value = %f", first_value, second_value);
    const uint32_t min_bin = diff_bin_offsets_->at(0);
    const uint32_t most_freq_bin = diff_bin_mappers_->at(0)->GetMostFreqBin();
    const uint32_t diff_bin = diff_bin_mappers_->at(0)->ValueToBin(diff_value);
    return diff_bin + min_bin - static_cast<uint32_t>(most_freq_bin == 0);
  }
}

template uint32_t DensePairwiseRankingDiffBin<uint8_t, true>::GetBinAt(const data_size_t paired_data_index) const;
template uint32_t DensePairwiseRankingDiffBin<uint8_t, false>::GetBinAt(const data_size_t paired_data_index) const;
template uint32_t DensePairwiseRankingDiffBin<uint16_t, false>::GetBinAt(const data_size_t paired_data_index) const;
template uint32_t DensePairwiseRankingDiffBin<uint32_t, false>::GetBinAt(const data_size_t paired_data_index) const;

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads) {
  unpaired_bin_->InitStreaming(num_thread, omp_max_threads);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::Push(int tid, data_size_t idx, uint32_t value) {
  unpaired_bin_->Push(tid, idx, value);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) {
  Log::Warning("copy subrow in pairwie ranking bin");
  unpaired_bin_->CopySubrow(full_bin, used_indices, num_used_indices);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::SaveBinaryToFile(BinaryWriter* writer) const {
  unpaired_bin_->SaveBinaryToFile(writer);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices) {
  unpaired_bin_->LoadFromMemory(memory, local_used_indices);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
size_t PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::SizesInByte() const {
  return unpaired_bin_->SizesInByte();
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
data_size_t PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::num_data() const {
  return unpaired_bin_->num_data();
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::ReSize(data_size_t num_data) {
  return unpaired_bin_->ReSize(num_data);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInner(
  const data_size_t* data_indices,
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const {
  data_size_t i = start;
  hist_t* grad = out;
  hist_t* hess = out + 1;
  hist_cnt_t* cnt = reinterpret_cast<hist_cnt_t*>(hess);
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset;
    for (; i < pf_end; ++i) {
      const auto paired_idx = USE_INDICES ? data_indices[i] : i;
      const auto ti = GetBinAt(paired_idx) << 1;
      // if (this->group_index_ == 0) {
      //   Log::Warning("group index = %d bin = %d gradient = %f hessian = %f", this->group_index_, ti / 2, ordered_gradients[i], ordered_hessians[i]);
      // }
      if (USE_HESSIAN) {
        grad[ti] += ordered_gradients[i];
        hess[ti] += ordered_hessians[i];
      } else {
        grad[ti] += ordered_gradients[i];
        ++cnt[ti];
      }
    }
  }
  for (; i < end; ++i) {
    const auto paired_idx = USE_INDICES ? data_indices[i] : i;
    const auto ti = GetBinAt(paired_idx) << 1;
    // if (this->group_index_ == 0) {
    //   Log::Warning("group index = %d bin = %d gradient = %f hessian = %f", this->group_index_, ti / 2, ordered_gradients[i], ordered_hessians[i]);
    // }
    if (USE_HESSIAN) {
      grad[ti] += ordered_gradients[i];
      hess[ti] += ordered_hessians[i];
    } else {
      grad[ti] += ordered_gradients[i];
      ++cnt[ti];
    }
  }
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN, typename PACKED_HIST_T, int HIST_BITS>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramIntInner(
  const data_size_t* data_indices,
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  data_size_t i = start;
  PACKED_HIST_T* out_ptr = reinterpret_cast<PACKED_HIST_T*>(out);
  const int16_t* gradients_ptr = reinterpret_cast<const int16_t*>(ordered_gradients);
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset;
    for (; i < pf_end; ++i) {
      const auto paired_idx = USE_INDICES ? data_indices[i] : i;
      const auto ti = GetBinAt(paired_idx) << 1;
      const int16_t gradient_16 = gradients_ptr[i];
      if (USE_HESSIAN) {
        const PACKED_HIST_T gradient_packed = HIST_BITS == 8 ? gradient_16 :
          (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (gradient_16 & 0xff);
        out_ptr[ti] += gradient_packed;
      } else {
        const PACKED_HIST_T gradient_packed = HIST_BITS == 8 ? gradient_16 :
          (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (1);
        out_ptr[ti] += gradient_packed;
      }
    }
  }
  for (; i < end; ++i) {
    const auto paired_idx = USE_INDICES ? data_indices[i] : i;
    const auto ti = GetBinAt(paired_idx) << 1;
    const int16_t gradient_16 = gradients_ptr[i];
    if (USE_HESSIAN) {
      const PACKED_HIST_T gradient_packed = HIST_BITS == 8 ? gradient_16 :
          (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (gradient_16 & 0xff);
      out_ptr[ti] += gradient_packed;
    } else {
      const PACKED_HIST_T gradient_packed = HIST_BITS == 8 ? gradient_16 :
          (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (1);
      out_ptr[ti] += gradient_packed;
    }
  }
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const {
  ConstructHistogramInner<true, true, true>(
        data_indices, start, end, ordered_gradients, ordered_hessians, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const {
  ConstructHistogramInner<false, false, true>(
      nullptr, start, end, ordered_gradients, ordered_hessians, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramInner<true, true, false>(data_indices, start, end,
                                              ordered_gradients, nullptr, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramInner<false, false, false>(
      nullptr, start, end, ordered_gradients, nullptr, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, true, int16_t, 8>(
      data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, true, int16_t, 8>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, false, int16_t, 8>(
    data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, false, int16_t, 8>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, true, int32_t, 16>(
      data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, true, int32_t, 16>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, false, int32_t, 16>(
    data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, false, int32_t, 16>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, true, int64_t, 32>(
      data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, true, int64_t, 32>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, false, int64_t, 32>(
    data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, false, int64_t, 32>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
template <bool MISS_IS_ZERO, bool MISS_IS_NA, bool MFB_IS_ZERO,
            bool MFB_IS_NA, bool USE_MIN_BIN>
data_size_t DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::SplitInner(uint32_t min_bin, uint32_t max_bin,
                        uint32_t default_bin, uint32_t most_freq_bin,
                        bool default_left, uint32_t threshold,
                        const data_size_t* data_indices, data_size_t cnt,
                        data_size_t* lte_indices,
                        data_size_t* gt_indices) const {
  auto th = static_cast<VAL_T>(threshold + min_bin);
  auto t_zero_bin = static_cast<VAL_T>(min_bin + default_bin);
  if (most_freq_bin == 0) {
    --th;
    --t_zero_bin;
  }
  const auto minb = static_cast<VAL_T>(min_bin);
  const auto maxb = static_cast<VAL_T>(max_bin);
  data_size_t lte_count = 0;
  data_size_t gt_count = 0;
  data_size_t* default_indices = gt_indices;
  data_size_t* default_count = &gt_count;
  data_size_t* missing_default_indices = gt_indices;
  data_size_t* missing_default_count = &gt_count;
  if (most_freq_bin <= threshold) {
    default_indices = lte_indices;
    default_count = &lte_count;
  }
  if (MISS_IS_ZERO || MISS_IS_NA) {
    if (default_left) {
      missing_default_indices = lte_indices;
      missing_default_count = &lte_count;
    }
  }
  if (min_bin < max_bin) {
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t paired_idx = data_indices[i];
      const auto bin = GetBinAt(paired_idx);
      if ((MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
          (MISS_IS_NA && !MFB_IS_NA && bin == maxb)) {
        missing_default_indices[(*missing_default_count)++] = paired_idx;
      } else if ((USE_MIN_BIN && (bin < minb || bin > maxb)) ||
                  (!USE_MIN_BIN && bin == 0)) {
        if ((MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO)) {
          missing_default_indices[(*missing_default_count)++] = paired_idx;
        } else {
          default_indices[(*default_count)++] = paired_idx;
        }
      } else if (bin > th) {
        gt_indices[gt_count++] = paired_idx;
      } else {
        lte_indices[lte_count++] = paired_idx;
      }
    }
  } else {
    data_size_t* max_bin_indices = gt_indices;
    data_size_t* max_bin_count = &gt_count;
    if (maxb <= th) {
      max_bin_indices = lte_indices;
      max_bin_count = &lte_count;
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t paired_idx = data_indices[i];
      const auto bin = GetBinAt(paired_idx);
      if (MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        missing_default_indices[(*missing_default_count)++] = paired_idx;
      } else if (bin != maxb) {
        if ((MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO)) {
          missing_default_indices[(*missing_default_count)++] = paired_idx;
        } else {
          default_indices[(*default_count)++] = paired_idx;
        }
      } else {
        if (MISS_IS_NA && !MFB_IS_NA) {
          missing_default_indices[(*missing_default_count)++] = paired_idx;
        } else {
          max_bin_indices[(*max_bin_count)++] = paired_idx;
        }
      }
    }
  }
  return lte_count;
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
data_size_t DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const {
  #define ARGUMENTS                                                        \
  min_bin, max_bin, default_bin, most_freq_bin, default_left, threshold, \
      data_indices, cnt, lte_indices, gt_indices
    if (missing_type == MissingType::None) {
      return SplitInner<false, false, false, false, true>(ARGUMENTS);
    } else if (missing_type == MissingType::Zero) {
      if (default_bin == most_freq_bin) {
        return SplitInner<true, false, true, false, true>(ARGUMENTS);
      } else {
        return SplitInner<true, false, false, false, true>(ARGUMENTS);
      }
    } else {
      if (max_bin == most_freq_bin + min_bin && most_freq_bin > 0) {
        return SplitInner<false, true, false, true, true>(ARGUMENTS);
      } else {
        return SplitInner<false, true, false, false, true>(ARGUMENTS);
      }
    }
#undef ARGUMENTS
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
data_size_t DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const {
#define ARGUMENTS                                                  \
  1, max_bin, default_bin, most_freq_bin, default_left, threshold, \
      data_indices, cnt, lte_indices, gt_indices
    if (missing_type == MissingType::None) {
      return SplitInner<false, false, false, false, false>(ARGUMENTS);
    } else if (missing_type == MissingType::Zero) {
      if (default_bin == most_freq_bin) {
        return SplitInner<true, false, true, false, false>(ARGUMENTS);
      } else {
        return SplitInner<true, false, false, false, false>(ARGUMENTS);
      }
    } else {
      if (max_bin == most_freq_bin + 1 && most_freq_bin > 0) {
        return SplitInner<false, true, false, true, false>(ARGUMENTS);
      } else {
        return SplitInner<false, true, false, false, false>(ARGUMENTS);
      }
    }
#undef ARGUMENTS
}


#define REGISTER_BIN_TYPE(BIN_TYPE, ITERATOR_TYPE) \
  template void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads); \
  template void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::Push(int tid, data_size_t idx, uint32_t value); \
  template void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices); \
  template void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::SaveBinaryToFile(BinaryWriter* writer) const; \
  template void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices); \
  template size_t PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::SizesInByte() const; \
  template data_size_t PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::num_data() const; \
  template void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::ReSize(data_size_t num_data);

#define COMMA ,

#define REGISTER_DENSE_HISTOGRAM_CONSTRUCTION_WITH_ITERATOR_TYPE_AND_FUNC_NAME(DATA_TYPE, USE_4BIT, ITERATOR_TYPE, FUNC_NAME) \
  template void DensePairwiseRankingBin<DATA_TYPE, USE_4BIT, ITERATOR_TYPE>::FUNC_NAME( \
  const data_size_t* data_indices, data_size_t start, data_size_t end, \
  const score_t* ordered_gradients, const score_t* ordered_hessians, \
  hist_t* out) const; \
  template void DensePairwiseRankingBin<DATA_TYPE, USE_4BIT, ITERATOR_TYPE>::FUNC_NAME( \
  data_size_t start, data_size_t end, \
  const score_t* ordered_gradients, \
  const score_t* ordered_hessians, \
  hist_t* out) const; \
  template void DensePairwiseRankingBin<DATA_TYPE, USE_4BIT, ITERATOR_TYPE>::FUNC_NAME( \
  const data_size_t* data_indices, data_size_t start, \
  data_size_t end, const score_t* ordered_gradients, \
  hist_t* out) const; \
  template void DensePairwiseRankingBin<DATA_TYPE, USE_4BIT, ITERATOR_TYPE>::FUNC_NAME( \
  data_size_t start, data_size_t end, \
  const score_t* ordered_gradients, \
  hist_t* out) const;


#define REGISTER_DENSE_TREE_LEARNING_FUNC_WITH_ITERATOR_TYPE(DATA_TYPE, USE_4BIT, ITERATOR_TYPE) \
  REGISTER_DENSE_HISTOGRAM_CONSTRUCTION_WITH_ITERATOR_TYPE_AND_FUNC_NAME(DATA_TYPE, USE_4BIT, ITERATOR_TYPE, ConstructHistogram) \
  REGISTER_DENSE_HISTOGRAM_CONSTRUCTION_WITH_ITERATOR_TYPE_AND_FUNC_NAME(DATA_TYPE, USE_4BIT, ITERATOR_TYPE, ConstructHistogramInt8) \
  REGISTER_DENSE_HISTOGRAM_CONSTRUCTION_WITH_ITERATOR_TYPE_AND_FUNC_NAME(DATA_TYPE, USE_4BIT, ITERATOR_TYPE, ConstructHistogramInt16) \
  REGISTER_DENSE_HISTOGRAM_CONSTRUCTION_WITH_ITERATOR_TYPE_AND_FUNC_NAME(DATA_TYPE, USE_4BIT, ITERATOR_TYPE, ConstructHistogramInt32) \
  template data_size_t DensePairwiseRankingBin<DATA_TYPE, USE_4BIT, ITERATOR_TYPE>::Split(uint32_t min_bin, uint32_t max_bin, \
                    uint32_t default_bin, uint32_t most_freq_bin, \
                    MissingType missing_type, bool default_left, \
                    uint32_t threshold, const data_size_t* data_indices, \
                    data_size_t cnt, \
                    data_size_t* lte_indices, \
                    data_size_t* gt_indices) const; \
  template data_size_t DensePairwiseRankingBin<DATA_TYPE, USE_4BIT, ITERATOR_TYPE>::Split(uint32_t max_bin, uint32_t default_bin, \
                            uint32_t most_freq_bin, MissingType missing_type, \
                            bool default_left, uint32_t threshold, \
                            const data_size_t* data_indices, data_size_t cnt, \
                            data_size_t* lte_indices, \
                            data_size_t* gt_indices) const;


#define REGISTER_DENSE_TREE_LEARNING_FUNC_FOR_ITERATOR_TYPE(ITERATOR_TYPE) \
  REGISTER_DENSE_TREE_LEARNING_FUNC_WITH_ITERATOR_TYPE(uint8_t, true, ITERATOR_TYPE) \
  REGISTER_DENSE_TREE_LEARNING_FUNC_WITH_ITERATOR_TYPE(uint8_t, false, ITERATOR_TYPE) \
  REGISTER_DENSE_TREE_LEARNING_FUNC_WITH_ITERATOR_TYPE(uint16_t, false, ITERATOR_TYPE) \
  REGISTER_DENSE_TREE_LEARNING_FUNC_WITH_ITERATOR_TYPE(uint32_t, false, ITERATOR_TYPE)


#define REGISTER_ITERATOR(ITERATOR_TYPE) \
  REGISTER_BIN_TYPE(DenseBin<uint8_t COMMA true>, ITERATOR_TYPE) \
  REGISTER_BIN_TYPE(DenseBin<uint8_t COMMA false>, ITERATOR_TYPE) \
  REGISTER_BIN_TYPE(DenseBin<uint16_t COMMA false>, ITERATOR_TYPE) \
  REGISTER_BIN_TYPE(DenseBin<uint32_t COMMA false>, ITERATOR_TYPE) \
  REGISTER_BIN_TYPE(SparseBin<uint8_t>, ITERATOR_TYPE) \
  REGISTER_BIN_TYPE(SparseBin<uint16_t>, ITERATOR_TYPE) \
  REGISTER_BIN_TYPE(SparseBin<uint32_t>, ITERATOR_TYPE) \
  REGISTER_DENSE_TREE_LEARNING_FUNC_FOR_ITERATOR_TYPE(ITERATOR_TYPE)


REGISTER_ITERATOR(PairwiseRankingFirstIterator)
REGISTER_ITERATOR(PairwiseRankingSecondIterator)
REGISTER_ITERATOR(PairwiseRankingDiffIterator)


#undef COMMA
#undef REGISTER_TYPE
#undef REGISTER_BIN_TYPE
#undef REGISTER_DENSE_TREE_LEARNING_FUNC_FOR_ITERATOR_TYPE
#undef REGISTER_DENSE_TREE_LEARNING_FUNC_WITH_ITERATOR_TYPE
#undef REGISTER_DENSE_HISTOGRAM_CONSTRUCTION_WITH_ITERATOR_TYPE_AND_FUNC_NAME


}  // namespace LightGBM
