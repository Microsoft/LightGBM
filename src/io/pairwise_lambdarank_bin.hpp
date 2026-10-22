/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_

#include <memory>
#include <utility>

#include <LightGBM/bin.h>

#include "dense_bin.hpp"
#include "sparse_bin.hpp"

namespace LightGBM {

template <typename BIN_TYPE>
class PairwiseRankingFirstBin;

template <typename BIN_TYPE>
class PairwiseRankingSecondBin;

template <typename BIN_TYPE>
class PairwiseRankingDiffBin;

template <typename BIN_TYPE>
class PairwiseRankingFirstIterator: public BinIterator {
 public:
  friend PairwiseRankingFirstBin<BIN_TYPE>;

  PairwiseRankingFirstIterator(const BIN_TYPE* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const uint32_t min_bin, const uint32_t max_bin, const uint32_t most_freq_bin) {
    unpaired_bin_ = unpaired_bin;
    unpaired_bin_iterator_.reset(unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin));
    unpaired_bin_iterator_->Reset(0);
    paired_ranking_item_index_map_ = paired_ranking_item_index_map;
    prev_index_ = -1;
    prev_val_ = 0;
  }

  ~PairwiseRankingFirstIterator() {}

  uint32_t Get(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map_[idx].first;
    if (data_index != prev_index_) {
      CHECK_GE(data_index, prev_index_);
      prev_val_ = unpaired_bin_iterator_->Get(data_index);
    }
    prev_index_ = data_index;
    return prev_val_;
  }

  uint32_t RawGet(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map_[idx].first;
    if (data_index != prev_index_) {
      CHECK_GE(data_index, prev_index_);
      prev_val_ = unpaired_bin_iterator_->RawGet(data_index);
    }
    prev_index_ = data_index;
    return prev_val_;
  }

  void Reset(data_size_t idx) {
    const data_size_t first_idx = paired_ranking_item_index_map_[idx].first;
    unpaired_bin_iterator_->Reset(first_idx);
    prev_index_ = -1;
    prev_val_ = 0;
  }

 private:
  const BIN_TYPE* unpaired_bin_;
  std::unique_ptr<BinIterator> unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  data_size_t prev_index_;
  uint32_t prev_val_;
};

template <typename BIN_TYPE>
class PairwiseRankingSecondIterator: public BinIterator {
 public:
  friend PairwiseRankingSecondBin<BIN_TYPE>;

  PairwiseRankingSecondIterator(const BIN_TYPE* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const uint32_t min_bin, const uint32_t max_bin, const uint32_t most_freq_bin) {
    unpaired_bin_ = unpaired_bin;
    unpaired_bin_iterator_.reset(unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin));
    unpaired_bin_iterator_->Reset(0);
    paired_ranking_item_index_map_ = paired_ranking_item_index_map;
    prev_index_ = 0;
  }

  ~PairwiseRankingSecondIterator() {}

  uint32_t Get(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map_[idx].second;
    if (data_index < prev_index_) {
      unpaired_bin_iterator_->Reset(0);
    }
    prev_index_ = data_index;
    return unpaired_bin_iterator_->Get(data_index);
  }

  uint32_t RawGet(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map_[idx].second;
    if (data_index < prev_index_) {
      unpaired_bin_iterator_->Reset(0);
    }
    prev_index_ = data_index;
    return unpaired_bin_iterator_->RawGet(data_index);
  }

  void Reset(data_size_t idx) {
    const data_size_t second_idx = paired_ranking_item_index_map_[idx].second;
    unpaired_bin_iterator_->Reset(second_idx);
    prev_index_ = 0;
  }

 private:
  const BIN_TYPE* unpaired_bin_;
  std::unique_ptr<BinIterator> unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  data_size_t prev_index_;
};


template <typename BIN_TYPE>
class PairwiseRankingDiffIterator: public BinIterator {
 public:
  friend PairwiseRankingDiffBin<BIN_TYPE>;

  PairwiseRankingDiffIterator(const BIN_TYPE* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const uint32_t min_bin, const uint32_t max_bin, const uint32_t most_freq_bin, const BinMapper* original_feature_bin_mapper, const BinMapper* diff_feature_bin_mapper, const std::vector<float>* raw_values): min_bin_(min_bin), max_bin_(max_bin), offset_(diff_feature_bin_mapper->GetMostFreqBin() == 0) {
    unpaired_bin_ = unpaired_bin;
    first_unpaired_bin_iterator_.reset(unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin));
    first_unpaired_bin_iterator_->Reset(0);
    second_unpaired_bin_iterator_.reset(unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin));
    second_unpaired_bin_iterator_->Reset(0);
    paired_ranking_item_index_map_ = paired_ranking_item_index_map;
    first_prev_index_ = 0;
    second_prev_index_ = 0;
    original_feature_bin_mapper_ = original_feature_bin_mapper;
    diff_feature_bin_mapper_ = diff_feature_bin_mapper;
    raw_values_ = raw_values;
  }

  ~PairwiseRankingDiffIterator() {}

  uint32_t Get(data_size_t idx) {
    const data_size_t first_data_index = paired_ranking_item_index_map_[idx].first;
    const data_size_t second_data_index = paired_ranking_item_index_map_[idx].second;
    if (second_data_index < second_prev_index_) {
      second_unpaired_bin_iterator_->Reset(0);
    }
    first_prev_index_ = first_data_index;
    second_prev_index_ = second_data_index;
    double diff_value = 0.0;
    if (raw_values_ == nullptr) {
      const uint32_t first_bin = first_unpaired_bin_iterator_->Get(first_data_index);
      const uint32_t second_bin = second_unpaired_bin_iterator_->Get(second_data_index);
      // TODO(shiyu1994): better original value
      const double first_value = original_feature_bin_mapper_->BinToValue(first_bin);
      const double second_value = original_feature_bin_mapper_->BinToValue(second_bin);
      diff_value = first_value - second_value;
    } else {
      const float first_value = raw_values_->at(first_data_index);
      const float second_value = raw_values_->at(second_data_index);
      // Log::Warning("first_value = %f, second_value = %f", first_value, second_value);
      diff_value = first_value - second_value;
    }
    const uint32_t diff_bin = diff_feature_bin_mapper_->ValueToBin(diff_value);
    return diff_bin;
  }

  uint32_t RawGet(data_size_t idx) {
    const uint32_t bin = Get(idx);
    return bin + min_bin_ - offset_;
  }

  void Reset(data_size_t idx) {
    const data_size_t first_idx = paired_ranking_item_index_map_[idx].first;
    const data_size_t second_idx = paired_ranking_item_index_map_[idx].second;
    first_unpaired_bin_iterator_->Reset(first_idx);
    second_unpaired_bin_iterator_->Reset(second_idx);
    first_prev_index_ = -1;
    second_prev_index_ = 0;
  }

 private:
  const BIN_TYPE* unpaired_bin_;
  std::unique_ptr<BinIterator> first_unpaired_bin_iterator_;
  std::unique_ptr<BinIterator> second_unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  const BinMapper* original_feature_bin_mapper_;
  const BinMapper* diff_feature_bin_mapper_;
  data_size_t first_prev_index_;
  data_size_t second_prev_index_;
  const uint32_t min_bin_;
  const uint32_t max_bin_;
  const uint32_t offset_;
  const std::vector<float>* raw_values_;
};


template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
class PairwiseRankingBin: public BIN_TYPE {
 public:
  PairwiseRankingBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, BIN_TYPE* unpaired_bin): BIN_TYPE(0), paired_ranking_item_index_map_(paired_ranking_item_index_map), unpaired_bin_(unpaired_bin) {
    num_data_ = num_data;
  }

  virtual ~PairwiseRankingBin() {}

  void InitStreaming(uint32_t num_thread, int32_t omp_max_threads) override;

  void Push(int tid, data_size_t idx, uint32_t value) override;

  void FinishLoad() override {
    unpaired_bin_->FinishLoad();
  }

  void CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override;

  void SaveBinaryToFile(BinaryWriter* writer) const override;

  void LoadFromMemory(const void* memory,
    const std::vector<data_size_t>& local_used_indices) override;

  size_t SizesInByte() const override;

  data_size_t num_data() const override;

  void* get_data() override {
    return unpaired_bin_->get_data();
  }

  BinIterator* GetUnpairedIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    return unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin);
  }

  void ReSize(data_size_t num_data) override;

  data_size_t Split(uint32_t /*min_bin*/, uint32_t /*max_bin*/,
                    uint32_t /*default_bin*/, uint32_t /*most_freq_bin*/,
                    MissingType /*missing_type*/, bool /*default_left*/,
                    uint32_t /*threshold*/, const data_size_t* /*data_indices*/,
                    data_size_t /*cnt*/,
                    data_size_t* /*lte_indices*/,
                    data_size_t* /*gt_indices*/) const override {
    Log::Fatal("Not implemented.");
    return 0;
  }

  data_size_t SplitCategorical(
      uint32_t /*min_bin*/, uint32_t /*max_bin*/, uint32_t /*most_freq_bin*/,
      const uint32_t* /*threshold*/, int /*num_threshold*/,
      const data_size_t* /*data_indices*/, data_size_t /*cnt*/,
      data_size_t* /*lte_indices*/, data_size_t* /*gt_indices*/) const override {
    Log::Fatal("Not implemented.");
    return 0;
  }

  data_size_t Split(uint32_t /*max_bin*/, uint32_t /*default_bin*/,
                            uint32_t /*most_freq_bin*/, MissingType /*missing_type*/,
                            bool /*default_left*/, uint32_t /*threshold*/,
                            const data_size_t* /*data_indices*/, data_size_t /*cnt*/,
                            data_size_t* /*lte_indices*/,
                            data_size_t* /*gt_indices*/) const override {
    Log::Fatal("Not implemented.");
    return 0;
  }

  data_size_t SplitCategorical(
      uint32_t /*max_bin*/, uint32_t /*most_freq_bin*/, const uint32_t* /*threshold*/,
      int /*num_threshold*/, const data_size_t* /*data_indices*/, data_size_t /*cnt*/,
      data_size_t* /*lte_indices*/, data_size_t* /*gt_indices*/) const override {
    Log::Fatal("Not implemented.");
    return 0;
  }

  void ConstructHistogram(
    const data_size_t* /*data_indices*/, data_size_t /*start*/, data_size_t /*end*/,
    const score_t* /*ordered_gradients*/, const score_t* /*ordered_hessians*/,
    hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogram(data_size_t /*start*/, data_size_t /*end*/,
    const score_t* /*ordered_gradients*/, const score_t* /*ordered_hessians*/,
    hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt8(
    const data_size_t* /*data_indices*/, data_size_t /*start*/, data_size_t /*end*/,
    const score_t* /*ordered_gradients*/, const score_t* /*ordered_hessians*/,
    hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt8(data_size_t /*start*/, data_size_t /*end*/,
    const score_t* /*ordered_gradients*/, const score_t* /*ordered_hessians*/,
    hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt16(
    const data_size_t* /*data_indices*/, data_size_t /*start*/, data_size_t /*end*/,
    const score_t* /*ordered_gradients*/, const score_t* /*ordered_hessians*/,
    hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt16(data_size_t /*start*/, data_size_t /*end*/,
    const score_t* /*ordered_gradients*/, const score_t* /*ordered_hessians*/,
    hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt32(
    const data_size_t* /*data_indices*/, data_size_t /*start*/, data_size_t /*end*/,
    const score_t* /*ordered_gradients*/, const score_t* /*ordered_hessians*/,
    hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt32(data_size_t /*start*/, data_size_t /*end*/,
    const score_t* /*ordered_gradients*/, const score_t* /*ordered_hessians*/,
    hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogram(const data_size_t* /*data_indices*/, data_size_t /*start*/, data_size_t /*end*/,
                                  const score_t* /*ordered_gradients*/, hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogram(data_size_t /*start*/, data_size_t /*end*/,
                                  const score_t* /*ordered_gradients*/, hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt8(const data_size_t* /*data_indices*/, data_size_t /*start*/, data_size_t /*end*/,
                                       const score_t* /*ordered_gradients*/, hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt8(data_size_t /*start*/, data_size_t /*end*/,
                                       const score_t* /*ordered_gradients*/, hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt16(const data_size_t* /*data_indices*/, data_size_t /*start*/, data_size_t /*end*/,
                                       const score_t* /*ordered_gradients*/, hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt16(data_size_t /*start*/, data_size_t /*end*/,
                                       const score_t* /*ordered_gradients*/, hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt32(const data_size_t* /*data_indices*/, data_size_t /*start*/, data_size_t /*end*/,
                                       const score_t* /*ordered_gradients*/, hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  void ConstructHistogramInt32(data_size_t /*start*/, data_size_t /*end*/,
                                       const score_t* /*ordered_gradients*/, hist_t* /*out*/) const override {
    Log::Fatal("Not implemented.");
  }

  const void* GetColWiseData(uint8_t* /*bit_type*/, bool* /*is_sparse*/, std::vector<BinIterator*>* /*bin_iterator*/, const int /*num_threads*/) const override {
    Log::Fatal("Not implemented.");
    return nullptr;
  }

  const void* GetColWiseData(uint8_t* /*bit_type*/, bool* /*is_sparse*/, BinIterator** /*bin_iterator*/) const override {
    Log::Fatal("Not implemented.");
    return nullptr;
  }

 protected:

  virtual data_size_t get_unpaired_index(const data_size_t paired_index) const = 0;

  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  const std::unique_ptr<BIN_TYPE> unpaired_bin_;
  data_size_t num_data_;
};

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
class DensePairwiseRankingBin: public PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE> {
 public:
  DensePairwiseRankingBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, DenseBin<VAL_T, IS_4BIT>* unpaired_bin): PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>(num_data, paired_ranking_item_index_map, unpaired_bin) {}

  void ConstructHistogram(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt8(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt16(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt32(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                  const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogram(data_size_t start, data_size_t end,
                                  const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  data_size_t Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const override;

  data_size_t Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const override;

 protected:
  template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN>
  void ConstructHistogramInner(const data_size_t* data_indices,
                               data_size_t start, data_size_t end,
                               const score_t* ordered_gradients,
                               const score_t* ordered_hessians,
                               hist_t* out) const;

  template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN, typename PACKED_HIST_T, int HIST_BITS>
  void ConstructHistogramIntInner(const data_size_t* data_indices,
                                  data_size_t start, data_size_t end,
                                  const score_t* ordered_gradients,
                                  hist_t* out) const;

  template <bool MISS_IS_ZERO, bool MISS_IS_NA, bool MFB_IS_ZERO,
            bool MFB_IS_NA, bool USE_MIN_BIN>
  data_size_t SplitInner(uint32_t min_bin, uint32_t max_bin,
                         uint32_t default_bin, uint32_t most_freq_bin,
                         bool default_left, uint32_t threshold,
                         const data_size_t* data_indices, data_size_t cnt,
                         data_size_t* lte_indices,
                         data_size_t* gt_indices) const;

  virtual inline uint32_t GetBinAt(const data_size_t paired_data_index) const {
    const data_size_t idx = this->get_unpaired_index(paired_data_index);
    return this->unpaired_bin_->data(idx);
  }
};

template <typename VAL_T, template<typename> class ITERATOR_TYPE>
class SparsePairwiseRankingBin: public PairwiseRankingBin<SparseBin<VAL_T>, ITERATOR_TYPE> {
 public:
  SparsePairwiseRankingBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, SparseBin<VAL_T>* unpaired_bin): PairwiseRankingBin<SparseBin<VAL_T>, ITERATOR_TYPE>(num_data, paired_ranking_item_index_map, unpaired_bin) {}
};

template <typename VAL_T, bool IS_4BIT>
class DensePairwiseRankingFirstBin: public DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingFirstIterator> {
 public:
  DensePairwiseRankingFirstBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, DenseBin<VAL_T, IS_4BIT>* unpaired_bin): DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingFirstIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    return new PairwiseRankingFirstIterator<DenseBin<VAL_T, IS_4BIT>>(this->unpaired_bin_.get(), this->paired_ranking_item_index_map_, min_bin, max_bin, most_freq_bin);
  }

 private:
  data_size_t get_unpaired_index(const data_size_t paired_index) const {
    return this->paired_ranking_item_index_map_[paired_index].first;
  }
};

template <typename VAL_T, bool IS_4BIT>
class DensePairwiseRankingSecondBin: public DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingSecondIterator> {
 public:
  DensePairwiseRankingSecondBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, DenseBin<VAL_T, IS_4BIT>* unpaired_bin): DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingSecondIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    return new PairwiseRankingSecondIterator<DenseBin<VAL_T, IS_4BIT>>(this->unpaired_bin_.get(), this->paired_ranking_item_index_map_, min_bin, max_bin, most_freq_bin);
  }

 private:
  data_size_t get_unpaired_index(const data_size_t paired_index) const {
    return this->paired_ranking_item_index_map_[paired_index].second;
  }
};

template <typename VAL_T, bool IS_4BIT>
class DensePairwiseRankingDiffBin: public DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingDiffIterator> {
 public:
  DensePairwiseRankingDiffBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, DenseBin<VAL_T, IS_4BIT>* unpaired_bin, const std::vector<std::unique_ptr<const BinMapper>>* diff_bin_mappers, const std::vector<std::unique_ptr<const BinMapper>>* ori_bin_mappers, const std::vector<uint32_t>* bin_offsets, const std::vector<uint32_t>* diff_bin_offsets, const std::vector<float>* raw_values): DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingDiffIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {
    diff_bin_mappers_ = diff_bin_mappers;
    ori_bin_mappers_ = ori_bin_mappers;
    bin_offsets_ = bin_offsets;
    diff_bin_offsets_ = diff_bin_offsets;
    raw_values_ = raw_values;
  }

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    int sub_feature_index = -1;
    for (int i = 0; i < static_cast<int>(bin_offsets_->size()); ++i) {
      if (bin_offsets_->at(i) == min_bin) {
        sub_feature_index = i;
        break;
      }
    }
    CHECK_GE(sub_feature_index, 0);

    // Log::Warning("dense raw_values_ = nullptr = %d", static_cast<int>(raw_values_ == nullptr));

    return new PairwiseRankingDiffIterator<DenseBin<VAL_T, IS_4BIT>>(this->unpaired_bin_.get(), this->paired_ranking_item_index_map_, min_bin, max_bin, most_freq_bin, ori_bin_mappers_->at(sub_feature_index).get(), diff_bin_mappers_->at(sub_feature_index).get(), raw_values_);
  }

 private:
  data_size_t get_unpaired_index(const data_size_t /*paired_index*/) const {
    Log::Fatal("get_unpaired_index of DensePairwiseRankingDiffBin should not be called.");
  }

  inline uint32_t GetBinAt(const data_size_t paired_data_index) const override;

  const std::vector<uint32_t>* bin_offsets_;
  const std::vector<uint32_t>* diff_bin_offsets_;
  const std::vector<std::unique_ptr<const BinMapper>>* diff_bin_mappers_;
  const std::vector<std::unique_ptr<const BinMapper>>* ori_bin_mappers_;
  const std::vector<float>* raw_values_;
};

template <typename VAL_T>
class SparsePairwiseRankingFirstBin: public SparsePairwiseRankingBin<VAL_T, PairwiseRankingFirstIterator> {
 public:
  SparsePairwiseRankingFirstBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, SparseBin<VAL_T>* unpaired_bin): SparsePairwiseRankingBin<VAL_T, PairwiseRankingFirstIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    return new PairwiseRankingFirstIterator<SparseBin<VAL_T>>(this->unpaired_bin_.get(), this->paired_ranking_item_index_map_, min_bin, max_bin, most_freq_bin);
  }

 private:
  data_size_t get_unpaired_index(const data_size_t paired_index) const {
    return this->paired_ranking_item_index_map_[paired_index].first;
  }
};

template <typename VAL_T>
class SparsePairwiseRankingSecondBin: public SparsePairwiseRankingBin<VAL_T, PairwiseRankingSecondIterator> {
 public:
  SparsePairwiseRankingSecondBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, SparseBin<VAL_T>* unpaired_bin): SparsePairwiseRankingBin<VAL_T, PairwiseRankingSecondIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    return new PairwiseRankingSecondIterator<SparseBin<VAL_T>>(this->unpaired_bin_.get(), this->paired_ranking_item_index_map_, min_bin, max_bin, most_freq_bin);
  }

 private:
  data_size_t get_unpaired_index(const data_size_t paired_index) const {
    return this->paired_ranking_item_index_map_[paired_index].second;
  }
};

template <typename VAL_T>
class SparsePairwiseRankingDiffBin: public SparsePairwiseRankingBin<VAL_T, PairwiseRankingDiffIterator> {
 public:
  SparsePairwiseRankingDiffBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, SparseBin<VAL_T>* unpaired_bin, const std::vector<std::unique_ptr<const BinMapper>>* diff_bin_mappers, const std::vector<std::unique_ptr<const BinMapper>>* ori_bin_mappers, const std::vector<uint32_t>* bin_offsets, const std::vector<uint32_t>* diff_bin_offsets, const std::vector<float>* raw_values): SparsePairwiseRankingBin<VAL_T, PairwiseRankingDiffIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {
    bin_offsets_ = bin_offsets;
    diff_bin_offsets_ = diff_bin_offsets;
    diff_bin_mappers_ = diff_bin_mappers;
    ori_bin_mappers_ = ori_bin_mappers;
    raw_values_ = raw_values;
  }

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    int sub_feature_index = -1;
    for (int i = 0; i < static_cast<int>(bin_offsets_->size()); ++i) {
      if (bin_offsets_->at(i) == min_bin) {
        CHECK_GT(i, 0);
        sub_feature_index = i;
        break;
      }
    }
    CHECK_GE(sub_feature_index, 0);
    // Log::Warning("sparse raw_values_ = nullptr = %d", static_cast<int>(raw_values_ == nullptr));
    return new PairwiseRankingDiffIterator<SparseBin<VAL_T>>(this->unpaired_bin_.get(), this->paired_ranking_item_index_map_, min_bin, max_bin, most_freq_bin, ori_bin_mappers_->at(sub_feature_index).get(), diff_bin_mappers_->at(sub_feature_index).get(), raw_values_);
  }

 private:
  data_size_t get_unpaired_index(const data_size_t /*paired_index*/) const {
    Log::Fatal("get_unpaired_index of SparsePairwiseRankingDiffBin should not be called.");
  }

  const std::vector<uint32_t>* bin_offsets_;
  const std::vector<uint32_t>* diff_bin_offsets_;
  const std::vector<std::unique_ptr<const BinMapper>>* diff_bin_mappers_;
  const std::vector<std::unique_ptr<const BinMapper>>* ori_bin_mappers_;
  const std::vector<float>* raw_values_;
};

template <typename MULTI_VAL_BIN_TYPE>
class MultiValPairwiseBin : public MULTI_VAL_BIN_TYPE {
 public:
  
};


}  // namespace LightGBM

#endif  // LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_
