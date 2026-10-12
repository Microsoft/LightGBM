/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_TRAIN_SHARE_STATES_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_TRAIN_SHARE_STATES_H_

#include <LightGBM/bin.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace LightGBM {

class MultiValBinWrapper {
 public:
  MultiValBinWrapper(MultiValBin* bin, data_size_t num_data,
    const std::vector<int>& feature_groups_contained, const int num_grad_quant_bins);

  bool IsSparse() const {
    if (multi_val_bin_ != nullptr) {
      return multi_val_bin_->IsSparse();
    }
    return false;
  }

  void InitTrain(const std::vector<int>& group_feature_start,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* bagging_use_indices,
    data_size_t bagging_indices_cnt);

  template <bool USE_QUANT_GRAD, int HIST_BITS, int INNER_HIST_BITS>
  void HistMove(const std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>& hist_buf);

  template <bool USE_QUANT_GRAD, int HIST_BITS, int INNER_HIST_BITS>
  void HistMerge(std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf);

  void ResizeHistBuf(std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
    MultiValBin* sub_multi_val_bin,
    hist_t* origin_hist_data);

  template <bool USE_INDICES, bool ORDERED, bool USE_QUANT_GRAD, int HIST_BITS>
  void ConstructHistograms(const data_size_t* data_indices,
      data_size_t num_data,
      const score_t* gradients,
      const score_t* hessians,
      std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
      hist_t* origin_hist_data) {
    const auto cur_multi_val_bin = (is_use_subcol_ || is_use_subrow_)
          ? multi_val_bin_subset_.get()
          : multi_val_bin_.get();
    if (cur_multi_val_bin != nullptr) {
      global_timer.Start("Dataset::sparse_bin_histogram");
      n_data_block_ = 1;
      data_block_size_ = num_data;
      Threading::BlockInfo<data_size_t>(num_threads_, num_data, min_block_size_,
                                        &n_data_block_, &data_block_size_);
      ResizeHistBuf(hist_buf, cur_multi_val_bin, origin_hist_data);
      const int inner_hist_bits = (data_block_size_ * num_grad_quant_bins_ < 256 && HIST_BITS == 16) ? 8 : HIST_BITS;
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) num_threads(num_threads_)
      for (int block_id = 0; block_id < n_data_block_; ++block_id) {
        OMP_LOOP_EX_BEGIN();
        data_size_t start = block_id * data_block_size_;
        data_size_t end = std::min<data_size_t>(start + data_block_size_, num_data);
        if (inner_hist_bits == 8) {
          ConstructHistogramsForBlock<USE_INDICES, ORDERED, USE_QUANT_GRAD, 8>(
            cur_multi_val_bin, start, end, data_indices, gradients, hessians,
            block_id, hist_buf);
        } else {
          ConstructHistogramsForBlock<USE_INDICES, ORDERED, USE_QUANT_GRAD, HIST_BITS>(
            cur_multi_val_bin, start, end, data_indices, gradients, hessians,
            block_id, hist_buf);
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      global_timer.Stop("Dataset::sparse_bin_histogram");

      global_timer.Start("Dataset::sparse_bin_histogram_merge");
      if (inner_hist_bits == 8) {
        HistMerge<USE_QUANT_GRAD, HIST_BITS, 8>(hist_buf);
      } else {
        HistMerge<USE_QUANT_GRAD, HIST_BITS, HIST_BITS>(hist_buf);
      }
      global_timer.Stop("Dataset::sparse_bin_histogram_merge");
      global_timer.Start("Dataset::sparse_bin_histogram_move");
      if (inner_hist_bits == 8) {
        HistMove<USE_QUANT_GRAD, HIST_BITS, 8>(*hist_buf);
      } else {
        HistMove<USE_QUANT_GRAD, HIST_BITS, HIST_BITS>(*hist_buf);
      }
      global_timer.Stop("Dataset::sparse_bin_histogram_move");
    }
  }

  template <bool USE_INDICES, bool ORDERED, bool USE_QUANT_GRAD, int HIST_BITS>
  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf) {
    if (USE_QUANT_GRAD) {
      if (HIST_BITS == 8) {
        int8_t* hist_buf_ptr = reinterpret_cast<int8_t*>(hist_buf->data());
        int8_t* data_ptr = hist_buf_ptr +
          static_cast<size_t>(num_bin_aligned_) * block_id * 2;
        std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kInt8HistBufferEntrySize);
        if (USE_INDICES) {
          if (ORDERED) {
            sub_multi_val_bin->ConstructHistogramOrderedInt8(data_indices, start, end,
                                                              gradients, hessians,
                                                              reinterpret_cast<hist_t*>(data_ptr));
          } else {
            sub_multi_val_bin->ConstructHistogramInt8(data_indices, start, end, gradients,
                                                       hessians,
                                                       reinterpret_cast<hist_t*>(data_ptr));
          }
        } else {
          sub_multi_val_bin->ConstructHistogramInt8(start, end, gradients, hessians,
                                                     reinterpret_cast<hist_t*>(data_ptr));
        }
      } else if (HIST_BITS == 16) {
        int16_t* data_ptr = reinterpret_cast<int16_t*>(origin_hist_data_);
        int16_t* hist_buf_ptr = reinterpret_cast<int16_t*>(hist_buf->data());
        if (block_id == 0) {
          if (is_use_subcol_) {
            data_ptr = hist_buf_ptr + hist_buf->size() - 2 * static_cast<size_t>(num_bin_aligned_);
          }
        } else {
          data_ptr = hist_buf_ptr +
            static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
        }
        std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kInt16HistBufferEntrySize);
        if (USE_INDICES) {
          if (ORDERED) {
            sub_multi_val_bin->ConstructHistogramOrderedInt16(data_indices, start, end,
                                                              gradients, hessians,
                                                              reinterpret_cast<hist_t*>(data_ptr));
          } else {
            sub_multi_val_bin->ConstructHistogramInt16(data_indices, start, end, gradients,
                                                       hessians,
                                                       reinterpret_cast<hist_t*>(data_ptr));
          }
        } else {
          sub_multi_val_bin->ConstructHistogramInt16(start, end, gradients, hessians,
                                                     reinterpret_cast<hist_t*>(data_ptr));
        }
      } else {
        int32_t* data_ptr = reinterpret_cast<int32_t*>(origin_hist_data_);
        int32_t* hist_buf_ptr = reinterpret_cast<int32_t*>(hist_buf->data());
        if (block_id == 0) {
          if (is_use_subcol_) {
            data_ptr = hist_buf_ptr + hist_buf->size() - 2 * static_cast<size_t>(num_bin_aligned_);
          }
        } else {
          data_ptr = hist_buf_ptr +
            static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
        }
        std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kInt32HistBufferEntrySize);
        if (USE_INDICES) {
          if (ORDERED) {
            sub_multi_val_bin->ConstructHistogramOrderedInt32(data_indices, start, end,
                                                              gradients, hessians,
                                                              reinterpret_cast<hist_t*>(data_ptr));
          } else {
            sub_multi_val_bin->ConstructHistogramInt32(data_indices, start, end, gradients,
                                                       hessians,
                                                       reinterpret_cast<hist_t*>(data_ptr));
          }
        } else {
          sub_multi_val_bin->ConstructHistogramInt32(start, end, gradients, hessians,
                                                     reinterpret_cast<hist_t*>(data_ptr));
        }
      }
    } else {
      hist_t* data_ptr = origin_hist_data_;
      if (block_id == 0) {
        if (is_use_subcol_) {
          data_ptr = hist_buf->data() + hist_buf->size() - 2 * static_cast<size_t>(num_bin_aligned_);
        }
      } else {
        data_ptr = hist_buf->data() +
          static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
      }
      std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHistBufferEntrySize);
      if (USE_INDICES) {
        if (ORDERED) {
          sub_multi_val_bin->ConstructHistogramOrdered(data_indices, start, end,
                                                  gradients, hessians, data_ptr);
        } else {
          sub_multi_val_bin->ConstructHistogram(data_indices, start, end, gradients,
                                            hessians, data_ptr);
        }
      } else {
        sub_multi_val_bin->ConstructHistogram(start, end, gradients, hessians,
                                          data_ptr);
      }
    }
  }

  void CopyMultiValBinSubset(const std::vector<int>& group_feature_start,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* bagging_use_indices,
    data_size_t bagging_indices_cnt);

  void SetUseSubrow(bool is_use_subrow) {
    is_use_subrow_ = is_use_subrow;
  }

  void SetSubrowCopied(bool is_subrow_copied) {
    is_subrow_copied_ = is_subrow_copied;
  }


  #ifdef USE_CUDA
  const void* GetRowWiseData(
    uint8_t* bit_type,
    size_t* total_size,
    bool* is_sparse,
    const void** out_data_ptr,
    uint8_t* data_ptr_bit_type) const {
    if (multi_val_bin_ == nullptr) {
      *bit_type = 0;
      *total_size = 0;
      *is_sparse = false;
      return nullptr;
    } else {
      return multi_val_bin_->GetRowWiseData(bit_type, total_size, is_sparse, out_data_ptr, data_ptr_bit_type);
    }
  }
  #endif  // USE_CUDA

 private:
  bool is_use_subcol_ = false;
  bool is_use_subrow_ = false;
  bool is_subrow_copied_ = false;
  std::unique_ptr<MultiValBin> multi_val_bin_;
  std::unique_ptr<MultiValBin> multi_val_bin_subset_;
  std::vector<uint32_t> hist_move_src_;
  std::vector<uint32_t> hist_move_dest_;
  std::vector<uint32_t> hist_move_size_;
  const std::vector<int> feature_groups_contained_;

  int num_threads_;
  int num_bin_;
  int num_bin_aligned_;
  int n_data_block_;
  int data_block_size_;
  int min_block_size_;
  int num_data_;
  int num_grad_quant_bins_;

  hist_t* origin_hist_data_;

  const size_t kHistBufferEntrySize = 2 * sizeof(hist_t);
  const size_t kInt32HistBufferEntrySize = 2 * sizeof(int32_t);
  const size_t kInt16HistBufferEntrySize = 2 * sizeof(int16_t);
  const size_t kInt8HistBufferEntrySize = 2 * sizeof(int8_t);
};

struct TrainingShareStates {
  int num_threads = 0;
  bool is_col_wise = true;
  bool is_constant_hessian = true;
  const data_size_t* bagging_use_indices;
  data_size_t bagging_indices_cnt;

  TrainingShareStates() {
    multi_val_bin_wrapper_.reset(nullptr);
    bagging_use_indices = nullptr;
    bagging_indices_cnt = 0;
  }

  int num_hist_total_bin() const { return num_hist_total_bin_; }

  const std::vector<uint32_t>& feature_hist_offsets() const { return feature_hist_offsets_; }

  #ifdef USE_CUDA
  const std::vector<uint32_t>& column_hist_offsets() const { return column_hist_offsets_; }
  #endif  // USE_CUDA

  bool IsSparseRowwise() const {
    return (multi_val_bin_wrapper_ != nullptr && multi_val_bin_wrapper_->IsSparse());
  }

  void SetMultiValBin(MultiValBin* bin, data_size_t num_data,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    bool dense_only, bool sparse_only, const int num_grad_quant_bins);

  void CalcBinOffsets(const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    std::vector<uint32_t>* offsets, bool is_col_wise);

  void InitTrain(const std::vector<int>& group_feature_start,
        const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
        const std::vector<int8_t>& is_feature_used) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->InitTrain(group_feature_start,
        feature_groups,
        is_feature_used,
        bagging_use_indices,
        bagging_indices_cnt);
    }
  }

  template <bool USE_INDICES, bool ORDERED, bool USE_QUANT_GRAD, int HIST_BITS>
  void ConstructHistograms(const data_size_t* data_indices,
                          data_size_t num_data,
                          const score_t* gradients,
                          const score_t* hessians,
                          hist_t* hist_data) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->ConstructHistograms<USE_INDICES, ORDERED, USE_QUANT_GRAD, HIST_BITS>(
        data_indices, num_data, gradients, hessians, &hist_buf_, hist_data);
    }
  }

  void SetUseSubrow(bool is_use_subrow) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->SetUseSubrow(is_use_subrow);
    }
  }

  void SetSubrowCopied(bool is_subrow_copied) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->SetSubrowCopied(is_subrow_copied);
    }
  }


  #ifdef USE_CUDA
  const void* GetRowWiseData(uint8_t* bit_type,
    size_t* total_size,
    bool* is_sparse,
    const void** out_data_ptr,
    uint8_t* data_ptr_bit_type) {
    if (multi_val_bin_wrapper_ != nullptr) {
      return multi_val_bin_wrapper_->GetRowWiseData(bit_type, total_size, is_sparse, out_data_ptr, data_ptr_bit_type);
    } else {
      *bit_type = 0;
      *total_size = 0;
      *is_sparse = false;
      return nullptr;
    }
  }
  #endif  // USE_CUDA

 private:
  std::vector<uint32_t> feature_hist_offsets_;
  #ifdef USE_CUDA
  std::vector<uint32_t> column_hist_offsets_;
  #endif  // USE_CUDA
  int num_hist_total_bin_ = 0;
  std::unique_ptr<MultiValBinWrapper> multi_val_bin_wrapper_;
  std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>> hist_buf_;
  int num_total_bin_ = 0;
  double num_elements_per_row_ = 0.0f;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_TRAIN_SHARE_STATES_H_
