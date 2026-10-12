/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_FEATURE_GROUP_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_FEATURE_GROUP_H_

#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/random.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace LightGBM {

class Dataset;
class DatasetLoader;
struct TrainingShareStates;
class MultiValBinWrapper;
/*! \brief Using to store data and providing some operations on one feature
 * group*/
class FeatureGroup {
 public:
  friend Dataset;
  friend DatasetLoader;
  friend TrainingShareStates;
  friend MultiValBinWrapper;
  /*!
  * \brief Constructor
  * \param num_feature number of features of this group
  * \param bin_mappers Bin mapper for features
  * \param num_data Total number of data
  * \param is_enable_sparse True if enable sparse feature
  */
  FeatureGroup(int num_feature, int8_t is_multi_val,
    std::vector<std::unique_ptr<BinMapper>>* bin_mappers,
    data_size_t num_data, int group_id) :
    num_feature_(num_feature), is_multi_val_(is_multi_val > 0), is_sparse_(false) {
    CHECK_EQ(static_cast<int>(bin_mappers->size()), num_feature);
    auto& ref_bin_mappers = *bin_mappers;
    double sum_sparse_rate = 0.0f;
    for (int i = 0; i < num_feature_; ++i) {
      bin_mappers_.emplace_back(ref_bin_mappers[i].release());
      sum_sparse_rate += bin_mappers_.back()->sparse_rate();
    }
    sum_sparse_rate /= num_feature_;
    int offset = 1;
    is_dense_multi_val_ = false;
    if (sum_sparse_rate < MultiValBin::multi_val_bin_sparse_threshold && is_multi_val_) {
      // use dense multi val bin
      offset = 0;
      is_dense_multi_val_ = true;
    }
    // use bin at zero to store most_freq_bin only when not using dense multi val bin
    num_total_bin_ = offset;
    // however, we should force to leave one bin, if dense multi val bin is the first bin
    // and its first feature has most freq bin > 0
    if (group_id == 0 && num_feature_ > 0 && is_dense_multi_val_ &&
      bin_mappers_[0]->GetMostFreqBin() > 0) {
      num_total_bin_ = 1;
    }
    bin_offsets_.emplace_back(num_total_bin_);
    for (int i = 0; i < num_feature_; ++i) {
      auto num_bin = bin_mappers_[i]->num_bin();
      if (bin_mappers_[i]->GetMostFreqBin() == 0) {
        num_bin -= offset;
      }
      num_total_bin_ += num_bin;
      bin_offsets_.emplace_back(num_total_bin_);
    }
    CreateBinData(num_data, is_multi_val_, true, false);
  }

  FeatureGroup(const FeatureGroup& other, int num_data) {
    num_feature_ = other.num_feature_;
    is_multi_val_ = other.is_multi_val_;
    is_dense_multi_val_ = other.is_dense_multi_val_;
    is_sparse_ = other.is_sparse_;
    num_total_bin_ = other.num_total_bin_;
    bin_offsets_ = other.bin_offsets_;

    bin_mappers_.reserve(other.bin_mappers_.size());
    for (auto& bin_mapper : other.bin_mappers_) {
      bin_mappers_.emplace_back(new BinMapper(*bin_mapper));
    }
    CreateBinData(num_data, is_multi_val_, !is_sparse_, is_sparse_);
  }

  FeatureGroup(std::vector<std::unique_ptr<BinMapper>>* bin_mappers,
    data_size_t num_data) : num_feature_(1), is_multi_val_(false) {
    CHECK_EQ(static_cast<int>(bin_mappers->size()), 1);
    // use bin at zero to store default_bin
    num_total_bin_ = 1;
    is_dense_multi_val_ = false;
    bin_offsets_.emplace_back(num_total_bin_);
    auto& ref_bin_mappers = *bin_mappers;
    for (int i = 0; i < num_feature_; ++i) {
      bin_mappers_.emplace_back(ref_bin_mappers[i].release());
      auto num_bin = bin_mappers_[i]->num_bin();
      if (bin_mappers_[i]->GetMostFreqBin() == 0) {
        num_bin -= 1;
      }
      num_total_bin_ += num_bin;
      bin_offsets_.emplace_back(num_total_bin_);
    }
    CreateBinData(num_data, false, false, false);
  }

  /*!
   * \brief Constructor from memory when data is present
   * \param memory Pointer of memory
   * \param num_all_data Number of global data
   * \param local_used_indices Local used indices, empty means using all data
   * \param group_id Id of group
   */
  FeatureGroup(const void* memory,
               data_size_t num_all_data,
               const std::vector<data_size_t>& local_used_indices,
               int group_id) {
    // Load the definition schema first
    const char* memory_ptr = LoadDefinitionFromMemory(memory, group_id);

    // Allocate memory for the data
    data_size_t num_data = num_all_data;
    if (!local_used_indices.empty()) {
      num_data = static_cast<data_size_t>(local_used_indices.size());
    }
    AllocateBins(num_data);

    // Now load the actual data
    if (is_multi_val_) {
      for (int i = 0; i < num_feature_; ++i) {
        multi_bin_data_[i]->LoadFromMemory(memory_ptr, local_used_indices);
        memory_ptr += multi_bin_data_[i]->SizesInByte();
      }
    } else {
      bin_data_->LoadFromMemory(memory_ptr, local_used_indices);
    }
  }

  /*!
   * \brief Constructor from definition in memory (without data)
   * \param memory Pointer of memory
   * \param local_used_indices Local used indices, empty means using all data
   */
  FeatureGroup(const void* memory, data_size_t num_data, int group_id) {
    LoadDefinitionFromMemory(memory, group_id);
    AllocateBins(num_data);
  }

  /*! \brief Destructor */
  ~FeatureGroup() {}

  /*!
   * \brief Load the overall definition of the feature group from binary serialized data
   * \param memory Pointer of memory
   * \param group_id Id of group
   */
  const char* LoadDefinitionFromMemory(const void* memory, int group_id) {
    const char* memory_ptr = reinterpret_cast<const char*>(memory);
    // get is_sparse
    is_multi_val_ = *(reinterpret_cast<const bool*>(memory_ptr));
    memory_ptr += VirtualFileWriter::AlignedSize(sizeof(is_multi_val_));
    is_dense_multi_val_ = *(reinterpret_cast<const bool*>(memory_ptr));
    memory_ptr += VirtualFileWriter::AlignedSize(sizeof(is_dense_multi_val_));
    is_sparse_ = *(reinterpret_cast<const bool*>(memory_ptr));
    memory_ptr += VirtualFileWriter::AlignedSize(sizeof(is_sparse_));
    num_feature_ = *(reinterpret_cast<const int*>(memory_ptr));
    memory_ptr += VirtualFileWriter::AlignedSize(sizeof(num_feature_));

    // get bin mapper(s)
    bin_mappers_.clear();
    for (int i = 0; i < num_feature_; ++i) {
      bin_mappers_.emplace_back(new BinMapper(memory_ptr));
      memory_ptr += bin_mappers_[i]->SizesInByte();
    }

    bin_offsets_.clear();
    int offset = 1;
    if (is_dense_multi_val_) {
      offset = 0;
    }
    // use bin at zero to store most_freq_bin only when not using dense multi val bin
    num_total_bin_ = offset;
    // however, we should force to leave one bin, if dense multi val bin is the first bin
    // and its first feature has most freq bin > 0
    if (group_id == 0 && num_feature_ > 0 && is_dense_multi_val_ &&
      bin_mappers_[0]->GetMostFreqBin() > 0) {
      num_total_bin_ = 1;
    }
    bin_offsets_.emplace_back(num_total_bin_);
    for (int i = 0; i < num_feature_; ++i) {
      auto num_bin = bin_mappers_[i]->num_bin();
      if (bin_mappers_[i]->GetMostFreqBin() == 0) {
        num_bin -= offset;
      }
      num_total_bin_ += num_bin;
      bin_offsets_.emplace_back(num_total_bin_);
    }

    return memory_ptr;
  }

  /*!
   * \brief Allocate the bins
   * \param num_all_data Number of global data
   */
  inline void AllocateBins(data_size_t num_data) {
    if (is_multi_val_) {
      for (int i = 0; i < num_feature_; ++i) {
        int addi = bin_mappers_[i]->GetMostFreqBin() == 0 ? 0 : 1;
        if (bin_mappers_[i]->sparse_rate() >= kSparseThreshold) {
          multi_bin_data_.emplace_back(Bin::CreateSparseBin(num_data, bin_mappers_[i]->num_bin() + addi));
        } else {
          multi_bin_data_.emplace_back(Bin::CreateDenseBin(num_data, bin_mappers_[i]->num_bin() + addi));
        }
      }
    } else {
      if (is_sparse_) {
        bin_data_.reset(Bin::CreateSparseBin(num_data, num_total_bin_));
      } else {
        bin_data_.reset(Bin::CreateDenseBin(num_data, num_total_bin_));
      }
    }
  }

  /*!
  * \brief Initialize for pushing in a streaming fashion.  By default, no action needed.
  * \param num_thread The number of external threads that will be calling the push APIs
  * \param omp_max_threads The maximum number of OpenMP threads to allocate for
  */
  void InitStreaming(int32_t num_thread, int32_t omp_max_threads) {
    if (is_multi_val_) {
      for (int i = 0; i < num_feature_; ++i) {
        multi_bin_data_[i]->InitStreaming(num_thread, omp_max_threads);
      }
    } else {
      bin_data_->InitStreaming(num_thread, omp_max_threads);
    }
  }

  /*!
   * \brief Push one record, will auto convert to bin and push to bin data
   * \param tid Thread id
   * \param sub_feature_idx Index of the subfeature
   * \param line_idx Index of record
   * \param value feature value of record
   */
  inline void PushData(int tid, int sub_feature_idx, data_size_t line_idx, double value) {
    uint32_t bin = bin_mappers_[sub_feature_idx]->ValueToBin(value);
    if (bin == bin_mappers_[sub_feature_idx]->GetMostFreqBin()) {
      return;
    }
    if (bin_mappers_[sub_feature_idx]->GetMostFreqBin() == 0) {
      bin -= 1;
    }
    if (is_multi_val_) {
      multi_bin_data_[sub_feature_idx]->Push(tid, line_idx, bin + 1);
    } else {
      bin += bin_offsets_[sub_feature_idx];
      bin_data_->Push(tid, line_idx, bin);
    }
  }

  void ReSize(int num_data) {
    if (!is_multi_val_) {
      bin_data_->ReSize(num_data);
    } else {
      for (int i = 0; i < num_feature_; ++i) {
        multi_bin_data_[i]->ReSize(num_data);
      }
    }
  }

  inline void CopySubrow(const FeatureGroup* full_feature, const data_size_t* used_indices, data_size_t num_used_indices) {
    if (!is_multi_val_) {
      bin_data_->CopySubrow(full_feature->bin_data_.get(), used_indices, num_used_indices);
    } else {
      for (int i = 0; i < num_feature_; ++i) {
        multi_bin_data_[i]->CopySubrow(full_feature->multi_bin_data_[i].get(), used_indices, num_used_indices);
      }
    }
  }

  inline void CopySubrowByCol(const FeatureGroup* full_feature, const data_size_t* used_indices, data_size_t num_used_indices, int fidx) {
    if (!is_multi_val_) {
      bin_data_->CopySubrow(full_feature->bin_data_.get(), used_indices, num_used_indices);
    } else {
      multi_bin_data_[fidx]->CopySubrow(full_feature->multi_bin_data_[fidx].get(), used_indices, num_used_indices);
    }
  }

  void AddFeaturesFrom(const FeatureGroup* other, int group_id) {
    CHECK(is_multi_val_);
    CHECK(other->is_multi_val_);
    // every time when new features are added, we need to reconsider sparse or dense
    double sum_sparse_rate = 0.0f;
    for (int i = 0; i < num_feature_; ++i) {
      sum_sparse_rate += bin_mappers_[i]->sparse_rate();
    }
    for (int i = 0; i < other->num_feature_; ++i) {
      sum_sparse_rate += other->bin_mappers_[i]->sparse_rate();
    }
    sum_sparse_rate /= (num_feature_ + other->num_feature_);
    int offset = 1;
    is_dense_multi_val_ = false;
    if (sum_sparse_rate < MultiValBin::multi_val_bin_sparse_threshold && is_multi_val_) {
      // use dense multi val bin
      offset = 0;
      is_dense_multi_val_ = true;
    }
    bin_offsets_.clear();
    num_total_bin_ = offset;
    // however, we should force to leave one bin, if dense multi val bin is the first bin
    // and its first feature has most freq bin > 0
    if (group_id == 0 && num_feature_ > 0 && is_dense_multi_val_ &&
      bin_mappers_[0]->GetMostFreqBin() > 0) {
      num_total_bin_ = 1;
    }
    bin_offsets_.emplace_back(num_total_bin_);
    for (int i = 0; i < num_feature_; ++i) {
      auto num_bin = bin_mappers_[i]->num_bin();
      if (bin_mappers_[i]->GetMostFreqBin() == 0) {
        num_bin -= offset;
      }
      num_total_bin_ += num_bin;
      bin_offsets_.emplace_back(num_total_bin_);
    }
    for (int i = 0; i < other->num_feature_; ++i) {
      const auto& other_bin_mapper = other->bin_mappers_[i];
      bin_mappers_.emplace_back(new BinMapper(*other_bin_mapper));
      auto num_bin = other_bin_mapper->num_bin();
      if (other_bin_mapper->GetMostFreqBin() == 0) {
        num_bin -= offset;
      }
      num_total_bin_ += num_bin;
      bin_offsets_.emplace_back(num_total_bin_);
      multi_bin_data_.emplace_back(other->multi_bin_data_[i]->Clone());
    }
    num_feature_ += other->num_feature_;
  }

  inline BinIterator* SubFeatureIterator(int sub_feature) {
    uint32_t most_freq_bin = bin_mappers_[sub_feature]->GetMostFreqBin();
    if (!is_multi_val_) {
      uint32_t min_bin = bin_offsets_[sub_feature];
      uint32_t max_bin = bin_offsets_[sub_feature + 1] - 1;
      return bin_data_->GetIterator(min_bin, max_bin, most_freq_bin);
    } else {
      int addi = bin_mappers_[sub_feature]->GetMostFreqBin() == 0 ? 0 : 1;
      uint32_t min_bin = 1;
      uint32_t max_bin = bin_mappers_[sub_feature]->num_bin() - 1 + addi;
      return multi_bin_data_[sub_feature]->GetIterator(min_bin, max_bin,
                                                       most_freq_bin);
    }
  }

  inline void FinishLoad() {
    if (is_multi_val_) {
      OMP_INIT_EX();
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
      for (int i = 0; i < num_feature_; ++i) {
        OMP_LOOP_EX_BEGIN();
        multi_bin_data_[i]->FinishLoad();
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    } else {
      bin_data_->FinishLoad();
    }
  }

  inline BinIterator* FeatureGroupIterator() {
    if (is_multi_val_) {
      return nullptr;
    }
    uint32_t min_bin = bin_offsets_[0];
    uint32_t max_bin = bin_offsets_.back() - 1;
    uint32_t most_freq_bin = 0;
    return bin_data_->GetIterator(min_bin, max_bin, most_freq_bin);
  }

  inline size_t FeatureGroupSizesInByte() {
    return bin_data_->SizesInByte();
  }

  inline void* FeatureGroupData() {
    if (is_multi_val_) {
      return nullptr;
    }
    return bin_data_->get_data();
  }

  inline data_size_t Split(int sub_feature, const uint32_t* threshold,
                           int num_threshold, bool default_left,
                           const data_size_t* data_indices, data_size_t cnt,
                           data_size_t* lte_indices,
                           data_size_t* gt_indices) const {
    uint32_t default_bin = bin_mappers_[sub_feature]->GetDefaultBin();
    uint32_t most_freq_bin = bin_mappers_[sub_feature]->GetMostFreqBin();
    if (!is_multi_val_) {
      uint32_t min_bin = bin_offsets_[sub_feature];
      uint32_t max_bin = bin_offsets_[sub_feature + 1] - 1;
      if (bin_mappers_[sub_feature]->bin_type() == BinType::NumericalBin) {
        auto missing_type = bin_mappers_[sub_feature]->missing_type();
        if (num_feature_ == 1) {
          return bin_data_->Split(max_bin, default_bin, most_freq_bin,
                                  missing_type, default_left, *threshold,
                                  data_indices, cnt, lte_indices, gt_indices);
        } else {
          return bin_data_->Split(min_bin, max_bin, default_bin, most_freq_bin,
                                  missing_type, default_left, *threshold,
                                  data_indices, cnt, lte_indices, gt_indices);
        }
      } else {
        if (num_feature_ == 1) {
          return bin_data_->SplitCategorical(max_bin, most_freq_bin, threshold,
                                             num_threshold, data_indices, cnt,
                                             lte_indices, gt_indices);
        } else {
          return bin_data_->SplitCategorical(
              min_bin, max_bin, most_freq_bin, threshold, num_threshold,
              data_indices, cnt, lte_indices, gt_indices);
        }
      }
    } else {
      int addi = bin_mappers_[sub_feature]->GetMostFreqBin() == 0 ? 0 : 1;
      uint32_t max_bin = bin_mappers_[sub_feature]->num_bin() - 1 + addi;
      if (bin_mappers_[sub_feature]->bin_type() == BinType::NumericalBin) {
        auto missing_type = bin_mappers_[sub_feature]->missing_type();
        return multi_bin_data_[sub_feature]->Split(
            max_bin, default_bin, most_freq_bin, missing_type, default_left,
            *threshold, data_indices, cnt, lte_indices, gt_indices);
      } else {
        return multi_bin_data_[sub_feature]->SplitCategorical(
            max_bin, most_freq_bin, threshold, num_threshold, data_indices, cnt,
            lte_indices, gt_indices);
      }
    }
  }

  /*!
   * \brief From bin to feature value
   * \param bin
   * \return FeatureGroup value of this bin
   */
  inline double BinToValue(int sub_feature_idx, uint32_t bin) const {
    return bin_mappers_[sub_feature_idx]->BinToValue(bin);
  }

  /*!
   * \brief Write to binary stream
   * \param writer Writer
   * \param include_data Whether to write data (true) or just header information (false)
   */
  void SerializeToBinary(BinaryWriter* writer, bool include_data = true) const {
    writer->AlignedWrite(&is_multi_val_, sizeof(is_multi_val_));
    writer->AlignedWrite(&is_dense_multi_val_, sizeof(is_dense_multi_val_));
    writer->AlignedWrite(&is_sparse_, sizeof(is_sparse_));
    writer->AlignedWrite(&num_feature_, sizeof(num_feature_));
    for (int i = 0; i < num_feature_; ++i) {
      bin_mappers_[i]->SaveBinaryToFile(writer);
    }

    if (include_data) {
      if (is_multi_val_) {
        for (int i = 0; i < num_feature_; ++i) {
          multi_bin_data_[i]->SaveBinaryToFile(writer);
        }
      } else {
        bin_data_->SaveBinaryToFile(writer);
      }
    }
  }

  /*!
   * \brief Get sizes in byte of this object
   */
  size_t SizesInByte(bool include_data = true) const {
    size_t ret = VirtualFileWriter::AlignedSize(sizeof(is_multi_val_)) +
                 VirtualFileWriter::AlignedSize(sizeof(is_dense_multi_val_)) +
                 VirtualFileWriter::AlignedSize(sizeof(is_sparse_)) +
                 VirtualFileWriter::AlignedSize(sizeof(num_feature_));
    for (int i = 0; i < num_feature_; ++i) {
      ret += bin_mappers_[i]->SizesInByte();
    }
    if (include_data) {
      if (!is_multi_val_) {
        ret += bin_data_->SizesInByte();
      } else {
        for (int i = 0; i < num_feature_; ++i) {
          ret += multi_bin_data_[i]->SizesInByte();
        }
      }
    }
    return ret;
  }

  /*! \brief Disable copy */
  FeatureGroup& operator=(const FeatureGroup&) = delete;

  /*! \brief Deep copy */
  FeatureGroup(const FeatureGroup& other, bool should_handle_dense_mv,
    int group_id) {
    num_feature_ = other.num_feature_;
    is_multi_val_ = other.is_multi_val_;
    is_dense_multi_val_ = other.is_dense_multi_val_;
    is_sparse_ = other.is_sparse_;
    num_total_bin_ = other.num_total_bin_;
    bin_offsets_ = other.bin_offsets_;

    bin_mappers_.reserve(other.bin_mappers_.size());
    for (auto& bin_mapper : other.bin_mappers_) {
      bin_mappers_.emplace_back(new BinMapper(*bin_mapper));
    }
    if (!is_multi_val_) {
      bin_data_.reset(other.bin_data_->Clone());
    } else {
      multi_bin_data_.clear();
      for (int i = 0; i < num_feature_; ++i) {
        multi_bin_data_.emplace_back(other.multi_bin_data_[i]->Clone());
      }
    }

    if (should_handle_dense_mv && is_dense_multi_val_ && group_id > 0) {
      // this feature group was the first feature group, but now no longer is,
      // so we need to eliminate its special empty bin for multi val dense bin
      if (bin_mappers_[0]->GetMostFreqBin() > 0 && bin_offsets_[0] == 1) {
        for (size_t i = 0; i < bin_offsets_.size(); ++i) {
          bin_offsets_[i] -= 1;
        }
        num_total_bin_ -= 1;
      }
    }
  }

  const void* GetColWiseData(const int sub_feature_index,
    uint8_t* bit_type,
    bool* is_sparse,
    std::vector<BinIterator*>* bin_iterator,
    const int num_threads) const {
    if (sub_feature_index >= 0) {
      CHECK(is_multi_val_);
      return multi_bin_data_[sub_feature_index]->GetColWiseData(bit_type, is_sparse, bin_iterator, num_threads);
    } else {
      CHECK(!is_multi_val_);
      return bin_data_->GetColWiseData(bit_type, is_sparse, bin_iterator, num_threads);
    }
  }

  const void* GetColWiseData(const int sub_feature_index,
    uint8_t* bit_type,
    bool* is_sparse,
    BinIterator** bin_iterator) const {
    if (sub_feature_index >= 0) {
      CHECK(is_multi_val_);
      return multi_bin_data_[sub_feature_index]->GetColWiseData(bit_type, is_sparse, bin_iterator);
    } else {
      CHECK(!is_multi_val_);
      return bin_data_->GetColWiseData(bit_type, is_sparse, bin_iterator);
    }
  }

  uint32_t feature_max_bin(const int sub_feature_index) const {
    if (!is_multi_val_) {
      return bin_offsets_[sub_feature_index + 1] - 1;
    } else {
      int addi = bin_mappers_[sub_feature_index]->GetMostFreqBin() == 0 ? 0 : 1;
      return bin_mappers_[sub_feature_index]->num_bin() - 1 + addi;
    }
  }

  uint32_t feature_min_bin(const int sub_feature_index) const {
    if (!is_multi_val_) {
      return bin_offsets_[sub_feature_index];
    } else {
      return 1;
    }
  }

 private:
  void CreateBinData(int num_data, bool is_multi_val, bool force_dense, bool force_sparse) {
    if (is_multi_val) {
      multi_bin_data_.clear();
      for (int i = 0; i < num_feature_; ++i) {
        int addi = bin_mappers_[i]->GetMostFreqBin() == 0 ? 0 : 1;
        if (bin_mappers_[i]->sparse_rate() >= kSparseThreshold) {
          multi_bin_data_.emplace_back(Bin::CreateSparseBin(
              num_data, bin_mappers_[i]->num_bin() + addi));
        } else {
          multi_bin_data_.emplace_back(
              Bin::CreateDenseBin(num_data, bin_mappers_[i]->num_bin() + addi));
        }
      }
      is_multi_val_ = true;
    } else {
      if (force_sparse ||
          (!force_dense && num_feature_ == 1 &&
           bin_mappers_[0]->sparse_rate() >= kSparseThreshold)) {
        is_sparse_ = true;
        bin_data_.reset(Bin::CreateSparseBin(num_data, num_total_bin_));
      } else {
        is_sparse_ = false;
        bin_data_.reset(Bin::CreateDenseBin(num_data, num_total_bin_));
      }
      is_multi_val_ = false;
    }
  }

  /*! \brief Number of features */
  int num_feature_;
  /*! \brief Bin mapper for sub features */
  std::vector<std::unique_ptr<BinMapper>> bin_mappers_;
  /*! \brief Bin offsets for sub features */
  std::vector<uint32_t> bin_offsets_;
  /*! \brief Bin data of this feature */
  std::unique_ptr<Bin> bin_data_;
  std::vector<std::unique_ptr<Bin>> multi_bin_data_;
  /*! \brief True if this feature is sparse */
  bool is_multi_val_;
  bool is_dense_multi_val_;
  bool is_sparse_;
  int num_total_bin_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_FEATURE_GROUP_H_
