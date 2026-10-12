/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_IO_MULTI_VAL_SPARSE_BIN_HPP_
#define LIGHTGBM_SRC_IO_MULTI_VAL_SPARSE_BIN_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/threading.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace LightGBM {

template <typename INDEX_T, typename VAL_T>
class MultiValSparseBin : public MultiValBin {
 public:
  explicit MultiValSparseBin(data_size_t num_data, int num_bin,
                             double estimate_element_per_row)
      : num_data_(num_data),
        num_bin_(num_bin),
        estimate_element_per_row_(estimate_element_per_row) {
    row_ptr_.resize(num_data_ + 1, 0);
    INDEX_T estimate_num_data = static_cast<INDEX_T>(estimate_element_per_row_ * 1.1 * num_data_);
    int num_threads = OMP_NUM_THREADS();
    if (num_threads > 1) {
      t_data_.resize(num_threads - 1);
      for (size_t i = 0; i < t_data_.size(); ++i) {
        t_data_[i].resize(estimate_num_data / num_threads);
      }
    }
    t_size_.resize(num_threads, 0);
    data_.resize(estimate_num_data / num_threads);
  }

  ~MultiValSparseBin() {}

  data_size_t num_data() const override { return num_data_; }

  int num_bin() const override { return num_bin_; }

  double num_element_per_row() const override {
    return estimate_element_per_row_;
  }

  const std::vector<uint32_t>& offsets() const override { return offsets_; }

  void PushOneRow(int tid, data_size_t idx,
                  const std::vector<uint32_t>& values) override {
    const int pre_alloc_size = 50;
    row_ptr_[idx + 1] = static_cast<INDEX_T>(values.size());
    if (tid == 0) {
      if (t_size_[tid] + row_ptr_[idx + 1] >
          static_cast<INDEX_T>(data_.size())) {
        data_.resize(t_size_[tid] + row_ptr_[idx + 1] * pre_alloc_size);
      }
      for (auto val : values) {
        data_[t_size_[tid]++] = static_cast<VAL_T>(val);
      }
    } else {
      if (t_size_[tid] + row_ptr_[idx + 1] >
          static_cast<INDEX_T>(t_data_[tid - 1].size())) {
        t_data_[tid - 1].resize(t_size_[tid] +
                                row_ptr_[idx + 1] * pre_alloc_size);
      }
      for (auto val : values) {
        t_data_[tid - 1][t_size_[tid]++] = static_cast<VAL_T>(val);
      }
    }
  }

  void MergeData(const INDEX_T* sizes) {
    Common::FunctionTimer fun_time("MultiValSparseBin::MergeData", global_timer);
    for (data_size_t i = 0; i < num_data_; ++i) {
      row_ptr_[i + 1] += row_ptr_[i];
    }
    if (t_data_.size() > 0) {
      std::vector<INDEX_T> offsets(1 + t_data_.size());
      offsets[0] = sizes[0];
      for (size_t tid = 0; tid < t_data_.size() - 1; ++tid) {
        offsets[tid + 1] = offsets[tid] + sizes[tid + 1];
      }
      data_.resize(row_ptr_[num_data_]);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1)
      for (int tid = 0; tid < static_cast<int>(t_data_.size()); ++tid) {
        std::copy_n(t_data_[tid].data(), sizes[tid + 1],
                    data_.data() + offsets[tid]);
      }
    } else {
      data_.resize(row_ptr_[num_data_]);
    }
  }

  void FinishLoad() override {
    MergeData(t_size_.data());
    t_size_.clear();
    row_ptr_.shrink_to_fit();
    data_.shrink_to_fit();
    t_data_.clear();
    t_data_.shrink_to_fit();
    // update estimate_element_per_row_ by all data
    estimate_element_per_row_ =
        static_cast<double>(row_ptr_[num_data_]) / num_data_;
  }

  bool IsSparse() override { return true; }

  template <bool USE_INDICES, bool USE_PREFETCH, bool ORDERED>
  void ConstructHistogramInner(const data_size_t* data_indices,
                               data_size_t start, data_size_t end,
                               const score_t* gradients,
                               const score_t* hessians, hist_t* out) const {
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;
    const VAL_T* data_ptr = data_.data();
    if (USE_PREFETCH) {
      const data_size_t pf_offset = 32 / sizeof(VAL_T);
      const data_size_t pf_end = end - pf_offset;

      for (; i < pf_end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        const auto pf_idx =
            USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
        if (!ORDERED) {
          PREFETCH_T0(gradients + pf_idx);
          PREFETCH_T0(hessians + pf_idx);
        }
        PREFETCH_T0(row_ptr_.data() + pf_idx);
        PREFETCH_T0(data_ptr + row_ptr_[pf_idx]);
        const auto j_start = RowPtr(idx);
        const auto j_end = RowPtr(idx + 1);
        const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[i] : hessians[idx];
        for (auto j = j_start; j < j_end; ++j) {
          const auto ti = static_cast<uint32_t>(data_ptr[j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
    for (; i < end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto j_start = RowPtr(idx);
      const auto j_end = RowPtr(idx + 1);
      const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
      const score_t hessian = ORDERED ? hessians[i] : hessians[idx];
      for (auto j = j_start; j < j_end; ++j) {
        const auto ti = static_cast<uint32_t>(data_ptr[j]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
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
    ConstructHistogramInner<false, false, false>(
        nullptr, start, end, gradients, hessians, out);
  }

  void ConstructHistogramOrdered(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* hessians,
                                 hist_t* out) const override {
    ConstructHistogramInner<true, true, true>(data_indices, start, end,
                                              gradients, hessians, out);
  }

  template <bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename PACKED_HIST_T, int HIST_BITS>
  void ConstructHistogramIntInner(const data_size_t* data_indices,
                               data_size_t start, data_size_t end,
                               const score_t* gradients_and_hessians, hist_t* out) const {
    data_size_t i = start;
    PACKED_HIST_T* out_ptr = reinterpret_cast<PACKED_HIST_T*>(out);
    const int16_t* gradients_and_hessians_ptr = reinterpret_cast<const int16_t*>(gradients_and_hessians);
    const VAL_T* data_ptr = data_.data();
    const INDEX_T* row_ptr_base = row_ptr_.data();
    if (USE_PREFETCH) {
      const data_size_t pf_offset = 32 / sizeof(VAL_T);
      const data_size_t pf_end = end - pf_offset;

      for (; i < pf_end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        const auto pf_idx =
            USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
        if (!ORDERED) {
          PREFETCH_T0(gradients_and_hessians_ptr + pf_idx);
        }
        PREFETCH_T0(row_ptr_base + pf_idx);
        PREFETCH_T0(data_ptr + row_ptr_[pf_idx]);
        const auto j_start = RowPtr(idx);
        const auto j_end = RowPtr(idx + 1);
        const int16_t gradient_16 = ORDERED ? gradients_and_hessians_ptr[i] : gradients_and_hessians_ptr[idx];
        const PACKED_HIST_T gradient_packed = (HIST_BITS == 8) ? gradient_16 :
          ((static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) |
          static_cast<PACKED_HIST_T>(gradient_16 & 0xff));
        for (auto j = j_start; j < j_end; ++j) {
          const auto ti = static_cast<uint32_t>(data_ptr[j]);
          out_ptr[ti] += gradient_packed;
        }
      }
    }
    for (; i < end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto j_start = RowPtr(idx);
      const auto j_end = RowPtr(idx + 1);
      const int16_t gradient_16 = ORDERED ? gradients_and_hessians_ptr[i] : gradients_and_hessians_ptr[idx];
      const PACKED_HIST_T gradient_packed = (HIST_BITS == 8) ? gradient_16 :
          ((static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) |
          static_cast<PACKED_HIST_T>(gradient_16 & 0xff));
      for (auto j = j_start; j < j_end; ++j) {
        const auto ti = static_cast<uint32_t>(data_ptr[j]);
        out_ptr[ti] += gradient_packed;
      }
    }
  }

  void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* gradients,
                          const score_t* /*hessians*/, hist_t* out) const override {
    ConstructHistogramIntInner<true, true, false, int64_t, 32>(data_indices, start, end,
                                                               gradients, out);
  }

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
                          const score_t* gradients, const score_t* /*hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, false, int64_t, 32>(
        nullptr, start, end, gradients, out);
  }

  void ConstructHistogramOrderedInt32(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* /*hessians*/,
                                 hist_t* out) const override {
    ConstructHistogramIntInner<true, true, true, int64_t, 32>(data_indices, start, end,
                                                              gradients, out);
  }

  void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* gradients,
                          const score_t* /*hessians*/, hist_t* out) const override {
    ConstructHistogramIntInner<true, true, false, int32_t, 16>(data_indices, start, end,
                                                               gradients, out);
  }

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
                          const score_t* gradients, const score_t* /*hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, false, int32_t, 16>(
        nullptr, start, end, gradients, out);
  }

  void ConstructHistogramOrderedInt16(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* /*hessians*/,
                                 hist_t* out) const override {
    ConstructHistogramIntInner<true, true, true, int32_t, 16>(data_indices, start, end,
                                                              gradients, out);
  }

  void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* gradients,
                          const score_t* /*hessians*/, hist_t* out) const override {
    ConstructHistogramIntInner<true, true, false, int16_t, 8>(data_indices, start, end,
                                                               gradients, out);
  }

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
                          const score_t* gradients, const score_t* /*hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, false, int16_t, 8>(
        nullptr, start, end, gradients, out);
  }

  void ConstructHistogramOrderedInt8(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* /*hessians*/,
                                 hist_t* out) const override {
    ConstructHistogramIntInner<true, true, true, int16_t, 8>(data_indices, start, end,
                                                              gradients, out);
  }

  MultiValBin* CreateLike(data_size_t num_data, int num_bin, int,
                          double estimate_element_per_row,
                          const std::vector<uint32_t>& /*offsets*/) const override {
    return new MultiValSparseBin<INDEX_T, VAL_T>(num_data, num_bin,
                                                 estimate_element_per_row);
  }

  void ReSize(data_size_t num_data, int num_bin, int,
              double estimate_element_per_row, const std::vector<uint32_t>& /*offsets*/) override {
    num_data_ = num_data;
    num_bin_ = num_bin;
    estimate_element_per_row_ = estimate_element_per_row;
    INDEX_T estimate_num_data =
        static_cast<INDEX_T>(estimate_element_per_row_ * 1.1 * num_data_);
    size_t npart = 1 + t_data_.size();
    INDEX_T avg_num_data = static_cast<INDEX_T>(estimate_num_data / npart);
    if (static_cast<INDEX_T>(data_.size()) < avg_num_data) {
      data_.resize(avg_num_data, 0);
    }
    for (size_t i = 0; i < t_data_.size(); ++i) {
      if (static_cast<INDEX_T>(t_data_[i].size()) < avg_num_data) {
        t_data_[i].resize(avg_num_data, 0);
      }
    }
    if (num_data_ + 1 > static_cast<data_size_t>(row_ptr_.size())) {
      row_ptr_.resize(num_data_ + 1);
    }
  }

  template <bool SUBROW, bool SUBCOL>
  void CopyInner(const MultiValBin* full_bin, const data_size_t* used_indices,
                 data_size_t num_used_indices,
                 const std::vector<uint32_t>& lower,
                 const std::vector<uint32_t>& upper,
                 const std::vector<uint32_t>& delta) {
    const auto other =
        reinterpret_cast<const MultiValSparseBin<INDEX_T, VAL_T>*>(full_bin);
    if (SUBROW) {
      CHECK_EQ(num_data_, num_used_indices);
    }
    int n_block = 1;
    data_size_t block_size = num_data_;
    Threading::BlockInfo<data_size_t>(static_cast<int>(t_data_.size() + 1),
                                      num_data_, 1024, &n_block, &block_size);
    std::vector<INDEX_T> sizes(t_data_.size() + 1, 0);
    const int pre_alloc_size = 50;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1)
    for (int tid = 0; tid < n_block; ++tid) {
      data_size_t start = tid * block_size;
      data_size_t end = std::min(num_data_, start + block_size);
      auto& buf = (tid == 0) ? data_ : t_data_[tid - 1];
      INDEX_T size = 0;
      for (data_size_t i = start; i < end; ++i) {
        const auto j_start =
            SUBROW ? other->RowPtr(used_indices[i]) : other->RowPtr(i);
        const auto j_end =
            SUBROW ? other->RowPtr(used_indices[i] + 1) : other->RowPtr(i + 1);
        if (size + (j_end - j_start) > static_cast<INDEX_T>(buf.size())) {
          buf.resize(size + (j_end - j_start) * pre_alloc_size);
        }
        int k = 0;
        const auto pre_size = size;
        for (auto j = j_start; j < j_end; ++j) {
          const auto val = other->data_[j];
          if (SUBCOL) {
            while (val >= upper[k]) {
              ++k;
            }
            if (val >= lower[k]) {
              buf[size++] = static_cast<VAL_T>(val - delta[k]);
            }
          } else {
            buf[size++] = val;
          }
        }
        row_ptr_[i + 1] = size - pre_size;
      }
      sizes[tid] = size;
    }
    MergeData(sizes.data());
  }

  void CopySubrow(const MultiValBin* full_bin, const data_size_t* used_indices,
                  data_size_t num_used_indices) override {
    CopyInner<true, false>(full_bin, used_indices, num_used_indices,
                           std::vector<uint32_t>(), std::vector<uint32_t>(),
                           std::vector<uint32_t>());
  }

  void CopySubcol(const MultiValBin* full_bin, const std::vector<int>&,
                  const std::vector<uint32_t>& lower,
                  const std::vector<uint32_t>& upper,
                  const std::vector<uint32_t>& delta) override {
    CopyInner<false, true>(full_bin, nullptr, num_data_, lower, upper, delta);
  }

  void CopySubrowAndSubcol(const MultiValBin* full_bin,
                           const data_size_t* used_indices,
                           data_size_t num_used_indices,
                           const std::vector<int>&,
                           const std::vector<uint32_t>& lower,
                           const std::vector<uint32_t>& upper,
                           const std::vector<uint32_t>& delta) override {
    CopyInner<true, true>(full_bin, used_indices, num_used_indices, lower,
                          upper, delta);
  }

  inline INDEX_T RowPtr(data_size_t idx) const { return row_ptr_[idx]; }

  MultiValSparseBin<INDEX_T, VAL_T>* Clone() override;


  #ifdef USE_CUDA
  const void* GetRowWiseData(uint8_t* bit_type,
    size_t* total_size,
    bool* is_sparse,
    const void** out_data_ptr,
    uint8_t* data_ptr_bit_type) const override;
  #endif  // USE_CUDA

 private:
  data_size_t num_data_;
  int num_bin_;
  double estimate_element_per_row_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, 32>> data_;
  std::vector<INDEX_T, Common::AlignmentAllocator<INDEX_T, 32>>
      row_ptr_;
  std::vector<std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, 32>>>
      t_data_;
  std::vector<INDEX_T> t_size_;
  std::vector<uint32_t> offsets_;

  MultiValSparseBin(const MultiValSparseBin<INDEX_T, VAL_T>& other)
      : num_data_(other.num_data_),
        num_bin_(other.num_bin_),
        estimate_element_per_row_(other.estimate_element_per_row_),
        data_(other.data_),
        row_ptr_(other.row_ptr_) {}
};

template <typename INDEX_T, typename VAL_T>
MultiValSparseBin<INDEX_T, VAL_T>* MultiValSparseBin<INDEX_T, VAL_T>::Clone() {
  return new MultiValSparseBin<INDEX_T, VAL_T>(*this);
}

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_IO_MULTI_VAL_SPARSE_BIN_HPP_
