/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_IO_DENSE_BIN_HPP_
#define LIGHTGBM_SRC_IO_DENSE_BIN_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/cuda/vector_cudahost.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace LightGBM {

template <typename VAL_T, bool IS_4BIT>
class DenseBin;

template <typename VAL_T, bool IS_4BIT>
class DenseBinIterator : public BinIterator {
 public:
  explicit DenseBinIterator(const DenseBin<VAL_T, IS_4BIT>* bin_data,
                            uint32_t min_bin, uint32_t max_bin,
                            uint32_t most_freq_bin)
      : bin_data_(bin_data),
        min_bin_(static_cast<VAL_T>(min_bin)),
        max_bin_(static_cast<VAL_T>(max_bin)),
        most_freq_bin_(static_cast<VAL_T>(most_freq_bin)) {
    if (most_freq_bin_ == 0) {
      offset_ = 1;
    } else {
      offset_ = 0;
    }
  }
  inline uint32_t RawGet(data_size_t idx) override;
  inline uint32_t Get(data_size_t idx) override;
  inline void Reset(data_size_t) override {}

 private:
  const DenseBin<VAL_T, IS_4BIT>* bin_data_;
  VAL_T min_bin_;
  VAL_T max_bin_;
  VAL_T most_freq_bin_;
  uint8_t offset_;
};
/*!
 * \brief Used to store bins for dense feature
 * Use template to reduce memory cost
 */
template <typename VAL_T, bool IS_4BIT>
class DenseBin : public Bin {
 public:
  friend DenseBinIterator<VAL_T, IS_4BIT>;
  explicit DenseBin(data_size_t num_data)
      : num_data_(num_data) {
    if (IS_4BIT) {
      CHECK_EQ(sizeof(VAL_T), 1);
      data_.resize((num_data_ + 1) / 2, static_cast<uint8_t>(0));
      buf_.resize((num_data_ + 1) / 2, static_cast<uint8_t>(0));
    } else {
      data_.resize(num_data_, static_cast<VAL_T>(0));
    }
  }

  ~DenseBin() {}

  void Push(int, data_size_t idx, uint32_t value) override {
    if (IS_4BIT) {
      const int i1 = idx >> 1;
      const int i2 = (idx & 1) << 2;
      const uint8_t val = static_cast<uint8_t>(value) << i2;
      if (i2 == 0) {
        data_[i1] = val;
      } else {
        buf_[i1] = val;
      }
    } else {
      data_[idx] = static_cast<VAL_T>(value);
    }
  }

  void ReSize(data_size_t num_data) override {
    if (num_data_ != num_data) {
      num_data_ = num_data;
      if (IS_4BIT) {
        data_.resize((num_data_ + 1) / 2, static_cast<VAL_T>(0));
      } else {
        data_.resize(num_data_);
      }
    }
  }

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin,
                           uint32_t most_freq_bin) const override;

  template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN>
  void ConstructHistogramInner(const data_size_t* data_indices,
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
        const auto idx = USE_INDICES ? data_indices[i] : i;
        const auto pf_idx =
            USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
        if (IS_4BIT) {
          PREFETCH_T0(data_.data() + (pf_idx >> 1));
        } else {
          PREFETCH_T0(data_.data() + pf_idx);
        }
        const auto ti = static_cast<uint32_t>(data(idx)) << 1;
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
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto ti = static_cast<uint32_t>(data(idx)) << 1;
      if (USE_HESSIAN) {
        grad[ti] += ordered_gradients[i];
        hess[ti] += ordered_hessians[i];
      } else {
        grad[ti] += ordered_gradients[i];
        ++cnt[ti];
      }
    }
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          const score_t* ordered_hessians,
                          hist_t* out) const override {
    ConstructHistogramInner<true, true, true>(
        data_indices, start, end, ordered_gradients, ordered_hessians, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* ordered_hessians,
                          hist_t* out) const override {
    ConstructHistogramInner<false, false, true>(
        nullptr, start, end, ordered_gradients, ordered_hessians, out);
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructHistogramInner<true, true, false>(data_indices, start, end,
                                               ordered_gradients, nullptr, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructHistogramInner<false, false, false>(
        nullptr, start, end, ordered_gradients, nullptr, out);
  }


  template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN, typename PACKED_HIST_T, int HIST_BITS>
  void ConstructHistogramIntInner(const data_size_t* data_indices,
                               data_size_t start, data_size_t end,
                               const score_t* ordered_gradients,
                               hist_t* out) const {
    data_size_t i = start;
    PACKED_HIST_T* out_ptr = reinterpret_cast<PACKED_HIST_T*>(out);
    const int16_t* gradients_ptr = reinterpret_cast<const int16_t*>(ordered_gradients);
    const VAL_T* data_ptr_base = data_.data();
    if (USE_PREFETCH) {
      const data_size_t pf_offset = 64 / sizeof(VAL_T);
      const data_size_t pf_end = end - pf_offset;
      for (; i < pf_end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        const auto pf_idx =
            USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
        if (IS_4BIT) {
          PREFETCH_T0(data_ptr_base + (pf_idx >> 1));
        } else {
          PREFETCH_T0(data_ptr_base + pf_idx);
        }
        const auto ti = static_cast<uint32_t>(data(idx));
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
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto ti = static_cast<uint32_t>(data(idx));
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

  void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<true, true, true, int16_t, 8>(
        data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, true, int16_t, 8>(
        nullptr, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructHistogramIntInner<true, true, false, int16_t, 8>(
      data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, false, int16_t, 8>(
        nullptr, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<true, true, true, int32_t, 16>(
        data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, true, int32_t, 16>(
        nullptr, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructHistogramIntInner<true, true, false, int32_t, 16>(
      data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, false, int32_t, 16>(
        nullptr, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<true, true, true, int64_t, 32>(
        data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, true, int64_t, 32>(
        nullptr, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructHistogramIntInner<true, true, false, int64_t, 32>(
      data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructHistogramIntInner<false, false, false, int64_t, 32>(
        nullptr, start, end, ordered_gradients, out);
  }

  template <bool MISS_IS_ZERO, bool MISS_IS_NA, bool MFB_IS_ZERO,
            bool MFB_IS_NA, bool USE_MIN_BIN>
  data_size_t SplitInner(uint32_t min_bin, uint32_t max_bin,
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
        const data_size_t idx = data_indices[i];
        const auto bin = data(idx);
        if ((MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
            (MISS_IS_NA && !MFB_IS_NA && bin == maxb)) {
          missing_default_indices[(*missing_default_count)++] = idx;
        } else if ((USE_MIN_BIN && (bin < minb || bin > maxb)) ||
                   (!USE_MIN_BIN && bin == 0)) {
          if ((MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO)) {
            missing_default_indices[(*missing_default_count)++] = idx;
          } else {
            default_indices[(*default_count)++] = idx;
          }
        } else if (bin > th) {
          gt_indices[gt_count++] = idx;
        } else {
          lte_indices[lte_count++] = idx;
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
        const data_size_t idx = data_indices[i];
        const auto bin = data(idx);
        if (MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
          missing_default_indices[(*missing_default_count)++] = idx;
        } else if (bin != maxb) {
          if ((MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO)) {
            missing_default_indices[(*missing_default_count)++] = idx;
          } else {
            default_indices[(*default_count)++] = idx;
          }
        } else {
          if (MISS_IS_NA && !MFB_IS_NA) {
            missing_default_indices[(*missing_default_count)++] = idx;
          } else {
            max_bin_indices[(*max_bin_count)++] = idx;
          }
        }
      }
    }
    return lte_count;
  }

  data_size_t Split(uint32_t min_bin, uint32_t max_bin, uint32_t default_bin,
                    uint32_t most_freq_bin, MissingType missing_type,
                    bool default_left, uint32_t threshold,
                    const data_size_t* data_indices, data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const override {
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

  data_size_t Split(uint32_t max_bin, uint32_t default_bin,
                    uint32_t most_freq_bin, MissingType missing_type,
                    bool default_left, uint32_t threshold,
                    const data_size_t* data_indices, data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const override {
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

  template <bool USE_MIN_BIN>
  data_size_t SplitCategoricalInner(uint32_t min_bin, uint32_t max_bin,
                                    uint32_t most_freq_bin,
                                    const uint32_t* threshold,
                                    int num_threshold,
                                    const data_size_t* data_indices,
                                    data_size_t cnt, data_size_t* lte_indices,
                                    data_size_t* gt_indices) const {
    data_size_t lte_count = 0;
    data_size_t gt_count = 0;
    data_size_t* default_indices = gt_indices;
    data_size_t* default_count = &gt_count;
    int8_t offset = most_freq_bin == 0 ? 1 : 0;
    if (most_freq_bin > 0 &&
        Common::FindInBitset(threshold, num_threshold, most_freq_bin)) {
      default_indices = lte_indices;
      default_count = &lte_count;
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t idx = data_indices[i];
      const uint32_t bin = data(idx);
      if (USE_MIN_BIN && (bin < min_bin || bin > max_bin)) {
        default_indices[(*default_count)++] = idx;
      } else if (!USE_MIN_BIN && bin == 0) {
        default_indices[(*default_count)++] = idx;
      } else if (Common::FindInBitset(threshold, num_threshold,
                                      bin - min_bin + offset)) {
        lte_indices[lte_count++] = idx;
      } else {
        gt_indices[gt_count++] = idx;
      }
    }
    return lte_count;
  }

  data_size_t SplitCategorical(uint32_t min_bin, uint32_t max_bin,
                               uint32_t most_freq_bin,
                               const uint32_t* threshold, int num_threshold,
                               const data_size_t* data_indices, data_size_t cnt,
                               data_size_t* lte_indices,
                               data_size_t* gt_indices) const override {
    return SplitCategoricalInner<true>(min_bin, max_bin, most_freq_bin,
                                       threshold, num_threshold, data_indices,
                                       cnt, lte_indices, gt_indices);
  }

  data_size_t SplitCategorical(uint32_t max_bin, uint32_t most_freq_bin,
                               const uint32_t* threshold, int num_threshold,
                               const data_size_t* data_indices, data_size_t cnt,
                               data_size_t* lte_indices,
                               data_size_t* gt_indices) const override {
    return SplitCategoricalInner<false>(1, max_bin, most_freq_bin, threshold,
                                        num_threshold, data_indices, cnt,
                                        lte_indices, gt_indices);
  }

  data_size_t num_data() const override { return num_data_; }

  void* get_data() override { return data_.data(); }

  void FinishLoad() override {
    if (IS_4BIT) {
      if (buf_.empty()) {
        return;
      }
      int len = (num_data_ + 1) / 2;
      for (int i = 0; i < len; ++i) {
        data_[i] |= buf_[i];
      }
      buf_.clear();
    }
  }

  void LoadFromMemory(
      const void* memory,
      const std::vector<data_size_t>& local_used_indices) override {
    const VAL_T* mem_data = reinterpret_cast<const VAL_T*>(memory);
    if (!local_used_indices.empty()) {
      if (IS_4BIT) {
        const data_size_t rest = num_data_ & 1;
        for (int i = 0; i < num_data_ - rest; i += 2) {
          // get old bins
          data_size_t idx = local_used_indices[i];
          const auto bin1 = static_cast<uint8_t>(
              (mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
          idx = local_used_indices[i + 1];
          const auto bin2 = static_cast<uint8_t>(
              (mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
          // add
          const int i1 = i >> 1;
          data_[i1] = (bin1 | (bin2 << 4));
        }
        if (rest) {
          data_size_t idx = local_used_indices[num_data_ - 1];
          data_[num_data_ >> 1] =
              (mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
        }
      } else {
        for (int i = 0; i < num_data_; ++i) {
          data_[i] = mem_data[local_used_indices[i]];
        }
      }
    } else {
      for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = mem_data[i];
      }
    }
  }

  inline VAL_T data(data_size_t idx) const {
    if (IS_4BIT) {
      return (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
    } else {
      return data_[idx];
    }
  }

  void CopySubrow(const Bin* full_bin, const data_size_t* used_indices,
                  data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const DenseBin<VAL_T, IS_4BIT>*>(full_bin);
    if (IS_4BIT) {
      const data_size_t rest = num_used_indices & 1;
      for (int i = 0; i < num_used_indices - rest; i += 2) {
        data_size_t idx = used_indices[i];
        const auto bin1 = static_cast<uint8_t>(
            (other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
        idx = used_indices[i + 1];
        const auto bin2 = static_cast<uint8_t>(
            (other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
        const int i1 = i >> 1;
        data_[i1] = (bin1 | (bin2 << 4));
      }
      if (rest) {
        data_size_t idx = used_indices[num_used_indices - 1];
        data_[num_used_indices >> 1] =
            (other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
      }
    } else {
      for (int i = 0; i < num_used_indices; ++i) {
        data_[i] = other_bin->data_[used_indices[i]];
      }
    }
  }

  void SaveBinaryToFile(BinaryWriter* writer) const override {
    writer->AlignedWrite(data_.data(), sizeof(VAL_T) * data_.size());
  }

  size_t SizesInByte() const override {
    return VirtualFileWriter::AlignedSize(sizeof(VAL_T) * data_.size());
  }

  DenseBin<VAL_T, IS_4BIT>* Clone() override;

  const void* GetColWiseData(uint8_t* bit_type, bool* is_sparse, std::vector<BinIterator*>* bin_iterator, const int num_threads) const override;

  const void* GetColWiseData(uint8_t* bit_type, bool* is_sparse, BinIterator** bin_iterator) const override;

 private:
  data_size_t num_data_;
#ifdef USE_CUDA
  std::vector<VAL_T, CHAllocator<VAL_T>> data_;
#else
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, kAlignedSize>> data_;
#endif
  std::vector<uint8_t> buf_;

  DenseBin(const DenseBin<VAL_T, IS_4BIT>& other)
      : num_data_(other.num_data_), data_(other.data_) {}
};

template <typename VAL_T, bool IS_4BIT>
DenseBin<VAL_T, IS_4BIT>* DenseBin<VAL_T, IS_4BIT>::Clone() {
  return new DenseBin<VAL_T, IS_4BIT>(*this);
}

template <typename VAL_T, bool IS_4BIT>
uint32_t DenseBinIterator<VAL_T, IS_4BIT>::Get(data_size_t idx) {
  auto ret = bin_data_->data(idx);
  if (ret >= min_bin_ && ret <= max_bin_) {
    return ret - min_bin_ + offset_;
  } else {
    return most_freq_bin_;
  }
}

template <typename VAL_T, bool IS_4BIT>
inline uint32_t DenseBinIterator<VAL_T, IS_4BIT>::RawGet(data_size_t idx) {
  return bin_data_->data(idx);
}

template <typename VAL_T, bool IS_4BIT>
BinIterator* DenseBin<VAL_T, IS_4BIT>::GetIterator(
    uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const {
  return new DenseBinIterator<VAL_T, IS_4BIT>(this, min_bin, max_bin,
                                              most_freq_bin);
}

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_IO_DENSE_BIN_HPP_
