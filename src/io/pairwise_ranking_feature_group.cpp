/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include <LightGBM/pairwise_ranking_feature_group.h>
#include <LightGBM/utils/threading.h>

#include "pairwise_lambdarank_bin.hpp"

namespace LightGBM {

PairwiseRankingFeatureGroup::PairwiseRankingFeatureGroup(const FeatureGroup& other, int num_original_data, const int is_first_or_second_in_pairing, int num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map):
  FeatureGroup(other, num_original_data), paired_ranking_item_index_map_(paired_ranking_item_index_map), num_data_(num_pairs), is_first_or_second_in_pairing_(is_first_or_second_in_pairing) {

  CreateBinData(num_original_data, is_multi_val_, !is_sparse_, is_sparse_);

  Threading::For<data_size_t>(0, num_original_data, 512, [this, &other] (int block_index, data_size_t block_start, data_size_t block_end) {
    for (int feature_index = 0; feature_index < num_feature_; ++feature_index) {
      std::unique_ptr<BinIterator> bin_iterator(other.SubFeatureIterator(feature_index));
      bin_iterator->Reset(block_start);
      for (data_size_t index = block_start; index < block_end; ++index) {
        PushBinData(block_index, feature_index, index, bin_iterator->Get(index));
      }
    }
  });

  FinishLoad();
}

void PairwiseRankingFeatureGroup::CreateBinData(int num_data, bool is_multi_val, bool force_dense, bool force_sparse) {
  CHECK(!is_multi_val);  // do not support multi-value bin for now
  if (is_multi_val) {
    multi_bin_data_.clear();
    for (int i = 0; i < num_feature_; ++i) {
      int addi = bin_mappers_[i]->GetMostFreqBin() == 0 ? 0 : 1;
      // if (bin_mappers_[i]->sparse_rate() >= kSparseThreshold) {
      //   if (is_first_or_second_in_pairing_ == 0) {
      //     multi_bin_data_.emplace_back(Bin::CreateSparsePairwiseRankingFirstBin(
      //         num_data, bin_mappers_[i]->num_bin() + addi, num_data_, paired_ranking_item_index_map_));
      //   } else {
      //     multi_bin_data_.emplace_back(Bin::CreateSparsePairwiseRankingSecondBin(
      //         num_data, bin_mappers_[i]->num_bin() + addi, num_data_, paired_ranking_item_index_map_));
      //   }
      // } else {
      if (is_first_or_second_in_pairing_ == 0) {
        multi_bin_data_.emplace_back(
            Bin::CreateDensePairwiseRankingFirstBin(num_data, bin_mappers_[i]->num_bin() + addi, num_data_, paired_ranking_item_index_map_));
      } else {
        multi_bin_data_.emplace_back(
            Bin::CreateDensePairwiseRankingSecondBin(num_data, bin_mappers_[i]->num_bin() + addi, num_data_, paired_ranking_item_index_map_));
      }
      // }
    }
    is_multi_val_ = true;
  } else {
    // if (force_sparse ||
    //     (!force_dense && num_feature_ == 1 &&
    //       bin_mappers_[0]->sparse_rate() >= kSparseThreshold)) {
    //   is_sparse_ = true;
    //   if (is_first_or_second_in_pairing_ == 0) {
    //     bin_data_.reset(Bin::CreateSparsePairwiseRankingFirstBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_));
    //   } else {
    //     bin_data_.reset(Bin::CreateSparsePairwiseRankingSecondBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_));
    //   }
    // } else {
    is_sparse_ = false;
    if (is_first_or_second_in_pairing_ == 0) {
      bin_data_.reset(Bin::CreateDensePairwiseRankingFirstBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_));
    } else {
      bin_data_.reset(Bin::CreateDensePairwiseRankingSecondBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_));
    }
    // }
    is_multi_val_ = false;
  }
}

PairwiseRankingDifferentialFeatureGroup::PairwiseRankingDifferentialFeatureGroup(const FeatureGroup& other, int num_original_data, const int is_first_or_second_in_pairing, int num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, std::vector<std::unique_ptr<BinMapper>>& diff_feature_bin_mappers, std::vector<std::unique_ptr<BinMapper>>& ori_feature_bin_mappers, const std::vector<float>* raw_values): PairwiseRankingFeatureGroup(other, num_original_data, is_first_or_second_in_pairing, num_pairs, paired_ranking_item_index_map), raw_values_(raw_values) {
  for (auto& bin_mapper_ref : diff_feature_bin_mappers) {
    diff_feature_bin_mappers_.emplace_back(bin_mapper_ref.release());
  }
  for (auto& bin_mapper_ref : ori_feature_bin_mappers) {
    ori_feature_bin_mappers_.emplace_back(bin_mapper_ref.release());
  }

  for (size_t i = 0; i < diff_feature_bin_mappers_.size(); ++i) {
    Log::Warning("diff_feature_bin_mappers_[%ld]->num_bin() = %d", i, diff_feature_bin_mappers_[i]->num_bin());
  }

  CreateBinData(num_original_data, is_multi_val_, !is_sparse_, is_sparse_);

  Threading::For<data_size_t>(0, num_original_data, 512, [this, &other] (int block_index, data_size_t block_start, data_size_t block_end) {
    for (int feature_index = 0; feature_index < num_feature_; ++feature_index) {
      std::unique_ptr<BinIterator> bin_iterator(other.SubFeatureIterator(feature_index));
      bin_iterator->Reset(block_start);
      for (data_size_t index = block_start; index < block_end; ++index) {
        PushBinData(block_index, feature_index, index, bin_iterator->Get(index));
      }
    }
  });

  FinishLoad();

  // calculate diff bin offsets
  const int offset = 1;
  original_bin_offsets_ = bin_offsets_;
  bin_offsets_.clear();
  num_total_bin_ = offset;
  bin_offsets_.emplace_back(num_total_bin_);
  for (int i = 0; i < num_feature_; ++i) {
    auto num_bin = diff_feature_bin_mappers_[i]->num_bin();
    if (diff_feature_bin_mappers_[i]->GetMostFreqBin() == 0) {
      num_bin -= offset;
    }
    num_total_bin_ += num_bin;
    bin_offsets_.emplace_back(num_total_bin_);
  }

  bin_mappers_.clear();
  for (const auto& bin_mapper : diff_feature_bin_mappers_) {
    bin_mappers_.emplace_back(new BinMapper(*bin_mapper.get()));
  }

  if (raw_values_ != nullptr) {
    Log::Warning("raw_values_->size() = %ld", raw_values_->size());
  }

}

void PairwiseRankingDifferentialFeatureGroup::CreateBinData(int num_data, bool is_multi_val, bool force_dense, bool force_sparse) {
  CHECK(!is_multi_val);  // do not support multi-value bin for now
  if (force_sparse ||
      (!force_dense && num_feature_ == 1 &&
        bin_mappers_[0]->sparse_rate() >= kSparseThreshold)) {
    is_sparse_ = true;
    bin_data_.reset(Bin::CreateDensePairwiseRankingDiffBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_, &diff_feature_bin_mappers_, &ori_feature_bin_mappers_, &original_bin_offsets_, &bin_offsets_, raw_values_));
  } else {
    is_sparse_ = false;
    bin_data_.reset(Bin::CreateDensePairwiseRankingDiffBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_, &diff_feature_bin_mappers_, &ori_feature_bin_mappers_, &original_bin_offsets_, &bin_offsets_, raw_values_));
  }
  is_multi_val_ = false;
}

inline BinIterator* PairwiseRankingDifferentialFeatureGroup::SubFeatureIterator(int sub_feature) const {
  uint32_t most_freq_bin = ori_feature_bin_mappers_[sub_feature]->GetMostFreqBin();
  if (!is_multi_val_) {
    uint32_t min_bin = original_bin_offsets_[sub_feature];
    uint32_t max_bin = original_bin_offsets_[sub_feature + 1] - 1;
    return bin_data_->GetIterator(min_bin, max_bin, most_freq_bin);
  } else {
    int addi = most_freq_bin == 0 ? 0 : 1;
    uint32_t min_bin = 1;
    uint32_t max_bin = ori_feature_bin_mappers_[sub_feature]->num_bin() - 1 + addi;
    return multi_bin_data_[sub_feature]->GetIterator(min_bin, max_bin,
                                                      most_freq_bin);
  }
}

inline BinIterator* PairwiseRankingDifferentialFeatureGroup::FeatureGroupIterator() {
  if (is_multi_val_) {
    return nullptr;
  }
  uint32_t min_bin = original_bin_offsets_[0];
  uint32_t max_bin = original_bin_offsets_.back() - 1;
  uint32_t most_freq_bin = 0;
  return bin_data_->GetIterator(min_bin, max_bin, most_freq_bin);
}

}  // namespace LightGBM
