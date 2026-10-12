/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_IO_SPARSE_BIN_HPP_
#define LIGHTGBM_SRC_IO_SPARSE_BIN_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace LightGBM {

template <typename VAL_T>
class SparseBin;

const size_t kNumFastIndex = 64;

template <typename VAL_T>
class SparseBinIterator : public BinIterator {
 public:
  SparseBinIterator(const SparseBin<VAL_T>* bin_data, uint32_t min_bin,
                    uint32_t max_bin, uint32_t most_freq_bin)
      : bin_data_(bin_data),
        min_bin_(static_cast<VAL_T>(min_bin)),
        max_bin_(static_cast<VAL_T>(max_bin)),
        most_freq_bin_(static_cast<VAL_T>(most_freq_bin)) {
    if (most_freq_bin_ == 0) {
      offset_ = 1;
    } else {
      offset_ = 0;
    }
    Reset(0);
  }
  SparseBinIterator(const SparseBin<VAL_T>* bin_data, data_size_t start_idx)
      : bin_data_(bin_data) {
    Reset(start_idx);
  }

  inline uint32_t RawGet(data_size_t idx) override;
  inline VAL_T InnerRawGet(data_size_t idx);

  inline uint32_t Get(data_size_t idx) override {
    VAL_T ret = InnerRawGet(idx);
    if (ret >= min_bin_ && ret <= max_bin_) {
      return ret - min_bin_ + offset_;
    } else {
      return most_freq_bin_;
    }
  }

  inline void Reset(data_size_t idx) override;

 private:
  const SparseBin<VAL_T>* bin_data_;
  data_size_t cur_pos_;
  data_size_t i_delta_;
  VAL_T min_bin_;
  VAL_T max_bin_;
  VAL_T most_freq_bin_;
  uint8_t offset_;
};

template <typename VAL_T>
class SparseBin : public Bin {
 public:
  friend class SparseBinIterator<VAL_T>;

  explicit SparseBin(data_size_t num_data) : num_data_(num_data) {
    int num_threads = OMP_NUM_THREADS();
    push_buffers_.resize(num_threads);
  }

  ~SparseBin() {}

  void InitStreaming(uint32_t num_thread, int32_t omp_max_threads) override {
    // Each external thread needs its own set of OpenMP push buffers,
    // so allocate num_thread times the maximum number of OMP threads per external thread
    push_buffers_.resize(omp_max_threads * num_thread);
  };

  void ReSize(data_size_t num_data) override { num_data_ = num_data; }

  void Push(int tid, data_size_t idx, uint32_t value) override {
    auto cur_bin = static_cast<VAL_T>(value);
    if (cur_bin != 0) {
      push_buffers_[tid].emplace_back(idx, cur_bin);
    }
  }

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin,
                           uint32_t most_freq_bin) const override;

#define ACC_GH(hist, i, g, h)               \
  const auto ti = static_cast<int>(i) << 1; \
  hist[ti] += g;                            \
  hist[ti + 1] += h;

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          const score_t* ordered_hessians,
                          hist_t* out) const override {
    data_size_t i_delta, cur_pos;
    InitIndex(data_indices[start], &i_delta, &cur_pos);
    data_size_t i = start;
    for (;;) {
      if (cur_pos < data_indices[i]) {
        cur_pos += deltas_[++i_delta];
        if (i_delta >= num_vals_) {
          break;
        }
      } else if (cur_pos > data_indices[i]) {
        if (++i >= end) {
          break;
        }
      } else {
        const VAL_T bin = vals_[i_delta];
        ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
        if (++i >= end) {
          break;
        }
        cur_pos += deltas_[++i_delta];
        if (i_delta >= num_vals_) {
          break;
        }
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* ordered_hessians,
                          hist_t* out) const override {
    data_size_t i_delta, cur_pos;
    InitIndex(start, &i_delta, &cur_pos);
    while (cur_pos < start && i_delta < num_vals_) {
      cur_pos += deltas_[++i_delta];
    }
    while (cur_pos < end && i_delta < num_vals_) {
      const VAL_T bin = vals_[i_delta];
      ACC_GH(out, bin, ordered_gradients[cur_pos], ordered_hessians[cur_pos]);
      cur_pos += deltas_[++i_delta];
    }
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          hist_t* out) const override {
    data_size_t i_delta, cur_pos;
    InitIndex(data_indices[start], &i_delta, &cur_pos);
    data_size_t i = start;
    hist_t* grad = out;
    hist_cnt_t* cnt = reinterpret_cast<hist_cnt_t*>(out + 1);
    for (;;) {
      if (cur_pos < data_indices[i]) {
        cur_pos += deltas_[++i_delta];
        if (i_delta >= num_vals_) {
          break;
        }
      } else if (cur_pos > data_indices[i]) {
        if (++i >= end) {
          break;
        }
      } else {
        const uint32_t ti = static_cast<uint32_t>(vals_[i_delta]) << 1;
        grad[ti] += ordered_gradients[i];
        ++cnt[ti];
        if (++i >= end) {
          break;
        }
        cur_pos += deltas_[++i_delta];
        if (i_delta >= num_vals_) {
          break;
        }
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          hist_t* out) const override {
    data_size_t i_delta, cur_pos;
    InitIndex(start, &i_delta, &cur_pos);
    hist_t* grad = out;
    hist_cnt_t* cnt = reinterpret_cast<hist_cnt_t*>(out + 1);
    while (cur_pos < start && i_delta < num_vals_) {
      cur_pos += deltas_[++i_delta];
    }
    while (cur_pos < end && i_delta < num_vals_) {
      const uint32_t ti = static_cast<uint32_t>(vals_[i_delta]) << 1;
      grad[ti] += ordered_gradients[cur_pos];
      ++cnt[ti];
      cur_pos += deltas_[++i_delta];
    }
  }
#undef ACC_GH

  template <bool USE_HESSIAN, typename PACKED_HIST_T, typename GRAD_HIST_T, typename HESS_HIST_T, int HIST_BITS>
  void ConstructIntHistogramInner(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients_and_hessians,
                          hist_t* out) const {
    data_size_t i_delta, cur_pos;
    InitIndex(start, &i_delta, &cur_pos);
    if (USE_HESSIAN) {
      PACKED_HIST_T* out_ptr = reinterpret_cast<PACKED_HIST_T*>(out);
      const int16_t* gradients_and_hessians_ptr = reinterpret_cast<const int16_t*>(ordered_gradients_and_hessians);
      while (cur_pos < start && i_delta < num_vals_) {
        cur_pos += deltas_[++i_delta];
      }
      while (cur_pos < end && i_delta < num_vals_) {
        const VAL_T bin = vals_[i_delta];
        const int16_t gradient_16 = gradients_and_hessians_ptr[cur_pos];
        const PACKED_HIST_T gradient_64 = (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (gradient_16 & 0xff);
        out_ptr[bin] += gradient_64;
        cur_pos += deltas_[++i_delta];
      }
    } else {
      GRAD_HIST_T* grad = reinterpret_cast<GRAD_HIST_T*>(out);
      HESS_HIST_T* cnt = reinterpret_cast<HESS_HIST_T*>(out) + 1;
      const int8_t* gradients_and_hessians_ptr = reinterpret_cast<const int8_t*>(ordered_gradients_and_hessians);
      while (cur_pos < start && i_delta < num_vals_) {
        cur_pos += deltas_[++i_delta];
      }
      while (cur_pos < end && i_delta < num_vals_) {
        const uint32_t ti = static_cast<uint32_t>(vals_[i_delta]) << 1;
        grad[ti] += gradients_and_hessians_ptr[cur_pos];
        ++cnt[ti];
        cur_pos += deltas_[++i_delta];
      }
    }
  }

  template <bool USE_HESSIAN, typename PACKED_HIST_T, typename GRAD_HIST_T, typename HESS_HIST_T, int HIST_BITS>
  void ConstructIntHistogramInner(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients_and_hessians,
                          hist_t* out) const {
    data_size_t i_delta, cur_pos;
    InitIndex(data_indices[start], &i_delta, &cur_pos);
    data_size_t i = start;
    if (USE_HESSIAN) {
      PACKED_HIST_T* out_ptr = reinterpret_cast<PACKED_HIST_T*>(out);
      const int16_t* gradients_and_hessians_ptr = reinterpret_cast<const int16_t*>(ordered_gradients_and_hessians);
      for (;;) {
        if (cur_pos < data_indices[i]) {
          cur_pos += deltas_[++i_delta];
          if (i_delta >= num_vals_) {
            break;
          }
        } else if (cur_pos > data_indices[i]) {
          if (++i >= end) {
            break;
          }
        } else {
          const VAL_T bin = vals_[i_delta];
          const int16_t gradient_16 = gradients_and_hessians_ptr[i];
          const PACKED_HIST_T gradient_packed = (HIST_BITS == 8) ? gradient_16 :
            (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (gradient_16 & 0xff);
          out_ptr[bin] += gradient_packed;
          if (++i >= end) {
            break;
          }
          cur_pos += deltas_[++i_delta];
          if (i_delta >= num_vals_) {
            break;
          }
        }
      }
    } else {
      GRAD_HIST_T* grad = reinterpret_cast<GRAD_HIST_T*>(out);
      HESS_HIST_T* cnt = reinterpret_cast<HESS_HIST_T*>(out) + 1;
      const int8_t* gradients_and_hessians_ptr = reinterpret_cast<const int8_t*>(ordered_gradients_and_hessians);
      for (;;) {
        if (cur_pos < data_indices[i]) {
          cur_pos += deltas_[++i_delta];
          if (i_delta >= num_vals_) {
            break;
          }
        } else if (cur_pos > data_indices[i]) {
          if (++i >= end) {
            break;
          }
        } else {
          const uint32_t ti = static_cast<uint32_t>(vals_[i_delta]) << 1;
          grad[ti] += gradients_and_hessians_ptr[i << 1];
          ++cnt[ti];
          if (++i >= end) {
            break;
          }
          cur_pos += deltas_[++i_delta];
          if (i_delta >= num_vals_) {
            break;
          }
        }
      }
    }
  }

  void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructIntHistogramInner<true, int64_t, int32_t, uint32_t, 32>(data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructIntHistogramInner<true, int64_t, int32_t, uint32_t, 32>(start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructIntHistogramInner<false, int64_t, int32_t, uint32_t, 32>(data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructIntHistogramInner<false, int64_t, int32_t, uint32_t, 32>(start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructIntHistogramInner<true, int32_t, int16_t, uint16_t, 16>(data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructIntHistogramInner<true, int32_t, int16_t, uint16_t, 16>(start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructIntHistogramInner<false, int32_t, int16_t, uint16_t, 16>(data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructIntHistogramInner<false, int32_t, int16_t, uint16_t, 16>(start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructIntHistogramInner<true, int16_t, uint8_t, uint8_t, 8>(data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          const score_t* /*ordered_hessians*/,
                          hist_t* out) const override {
    ConstructIntHistogramInner<true, int16_t, uint8_t, uint8_t, 8>(start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructIntHistogramInner<false, int16_t, uint8_t, uint8_t, 8>(data_indices, start, end, ordered_gradients, out);
  }

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
                          const score_t* ordered_gradients,
                          hist_t* out) const override {
    ConstructIntHistogramInner<false, int16_t, uint8_t, uint8_t, 8>(start, end, ordered_gradients, out);
  }

  inline void NextNonzeroFast(data_size_t* i_delta,
                              data_size_t* cur_pos) const {
    *cur_pos += deltas_[++(*i_delta)];
    if (*i_delta >= num_vals_) {
      *cur_pos = num_data_;
    }
  }

  inline bool NextNonzero(data_size_t* i_delta, data_size_t* cur_pos) const {
    *cur_pos += deltas_[++(*i_delta)];
    if (*i_delta < num_vals_) {
      return true;
    } else {
      *cur_pos = num_data_;
      return false;
    }
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
    SparseBinIterator<VAL_T> iterator(this, data_indices[0]);
    if (min_bin < max_bin) {
      for (data_size_t i = 0; i < cnt; ++i) {
        const data_size_t idx = data_indices[i];
        const auto bin = iterator.InnerRawGet(idx);
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
        const auto bin = iterator.InnerRawGet(idx);
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
    SparseBinIterator<VAL_T> iterator(this, data_indices[0]);
    int8_t offset = most_freq_bin == 0 ? 1 : 0;
    if (most_freq_bin > 0 && Common::FindInBitset(threshold, num_threshold, most_freq_bin)) {
      default_indices = lte_indices;
      default_count = &lte_count;
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t idx = data_indices[i];
      const uint32_t bin = iterator.RawGet(idx);
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

  void* get_data() override { return nullptr; }

  void FinishLoad() override {
    // get total non zero size
    size_t pair_cnt = 0;
    for (size_t i = 0; i < push_buffers_.size(); ++i) {
      pair_cnt += push_buffers_[i].size();
    }
    std::vector<std::pair<data_size_t, VAL_T>>& idx_val_pairs =
        push_buffers_[0];
    idx_val_pairs.reserve(pair_cnt);

    for (size_t i = 1; i < push_buffers_.size(); ++i) {
      idx_val_pairs.insert(idx_val_pairs.end(), push_buffers_[i].begin(),
                           push_buffers_[i].end());
      push_buffers_[i].clear();
      push_buffers_[i].shrink_to_fit();
    }
    // sort by data index
    std::sort(idx_val_pairs.begin(), idx_val_pairs.end(),
              [](const std::pair<data_size_t, VAL_T>& a,
                 const std::pair<data_size_t, VAL_T>& b) {
                return a.first < b.first;
              });
    // load delta array
    LoadFromPair(idx_val_pairs);
  }

  void LoadFromPair(
      const std::vector<std::pair<data_size_t, VAL_T>>& idx_val_pairs) {
    deltas_.clear();
    vals_.clear();
    deltas_.reserve(idx_val_pairs.size());
    vals_.reserve(idx_val_pairs.size());
    // transform to delta array
    data_size_t last_idx = 0;
    for (size_t i = 0; i < idx_val_pairs.size(); ++i) {
      const data_size_t cur_idx = idx_val_pairs[i].first;
      const VAL_T bin = idx_val_pairs[i].second;
      data_size_t cur_delta = cur_idx - last_idx;
      // disallow the multi-val in one row
      if (i > 0 && cur_delta == 0) {
        continue;
      }
      while (cur_delta >= 256) {
        deltas_.push_back(255);
        vals_.push_back(0);
        cur_delta -= 255;
      }
      deltas_.push_back(static_cast<uint8_t>(cur_delta));
      vals_.push_back(bin);
      last_idx = cur_idx;
    }
    // avoid out of range
    deltas_.push_back(0);
    num_vals_ = static_cast<data_size_t>(vals_.size());

    // reduce memory cost
    deltas_.shrink_to_fit();
    vals_.shrink_to_fit();

    // generate fast index
    GetFastIndex();
  }

  void GetFastIndex() {
    fast_index_.clear();
    // get shift cnt
    data_size_t mod_size = (num_data_ + kNumFastIndex - 1) / kNumFastIndex;
    data_size_t pow2_mod_size = 1;
    fast_index_shift_ = 0;
    while (pow2_mod_size < mod_size) {
      pow2_mod_size <<= 1;
      ++fast_index_shift_;
    }
    // build fast index
    data_size_t i_delta = -1;
    data_size_t cur_pos = 0;
    data_size_t next_threshold = 0;
    while (NextNonzero(&i_delta, &cur_pos)) {
      while (next_threshold <= cur_pos) {
        fast_index_.emplace_back(i_delta, cur_pos);
        next_threshold += pow2_mod_size;
      }
    }
    // avoid out of range
    while (next_threshold < num_data_) {
      fast_index_.emplace_back(num_vals_ - 1, cur_pos);
      next_threshold += pow2_mod_size;
    }
    fast_index_.shrink_to_fit();
  }

  void SaveBinaryToFile(BinaryWriter* writer) const override {
    writer->AlignedWrite(&num_vals_, sizeof(num_vals_));
    writer->AlignedWrite(deltas_.data(), sizeof(uint8_t) * (num_vals_ + 1));
    writer->AlignedWrite(vals_.data(), sizeof(VAL_T) * num_vals_);
  }

  size_t SizesInByte() const override {
    return VirtualFileWriter::AlignedSize(sizeof(num_vals_)) +
           VirtualFileWriter::AlignedSize(sizeof(uint8_t) * (num_vals_ + 1)) +
           VirtualFileWriter::AlignedSize(sizeof(VAL_T) * num_vals_);
  }

  void LoadFromMemory(
      const void* memory,
      const std::vector<data_size_t>& local_used_indices) override {
    const char* mem_ptr = reinterpret_cast<const char*>(memory);
    data_size_t tmp_num_vals = *(reinterpret_cast<const data_size_t*>(mem_ptr));
    mem_ptr += VirtualFileWriter::AlignedSize(sizeof(tmp_num_vals));
    const uint8_t* tmp_delta = reinterpret_cast<const uint8_t*>(mem_ptr);
    mem_ptr += VirtualFileWriter::AlignedSize(sizeof(uint8_t) * (tmp_num_vals + 1));
    const VAL_T* tmp_vals = reinterpret_cast<const VAL_T*>(mem_ptr);

    deltas_.clear();
    vals_.clear();
    num_vals_ = tmp_num_vals;
    for (data_size_t i = 0; i < num_vals_; ++i) {
      deltas_.push_back(tmp_delta[i]);
      vals_.push_back(tmp_vals[i]);
    }
    deltas_.push_back(0);
    // reduce memory cost
    deltas_.shrink_to_fit();
    vals_.shrink_to_fit();

    if (local_used_indices.empty()) {
      // generate fast index
      GetFastIndex();
    } else {
      std::vector<std::pair<data_size_t, VAL_T>> tmp_pair;
      data_size_t cur_pos = 0;
      data_size_t j = -1;
      for (data_size_t i = 0;
           i < static_cast<data_size_t>(local_used_indices.size()); ++i) {
        const data_size_t idx = local_used_indices[i];
        while (cur_pos < idx && j < num_vals_) {
          NextNonzero(&j, &cur_pos);
        }
        if (cur_pos == idx && j < num_vals_ && vals_[j] > 0) {
          // new row index is i
          tmp_pair.emplace_back(i, vals_[j]);
        }
      }
      LoadFromPair(tmp_pair);
    }
  }

  void CopySubrow(const Bin* full_bin, const data_size_t* used_indices,
                  data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const SparseBin<VAL_T>*>(full_bin);
    deltas_.clear();
    vals_.clear();
    data_size_t start = 0;
    if (num_used_indices > 0) {
      start = used_indices[0];
    }
    SparseBinIterator<VAL_T> iterator(other_bin, start);
    // transform to delta array
    data_size_t last_idx = 0;
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      auto bin = iterator.InnerRawGet(used_indices[i]);
      if (bin > 0) {
        data_size_t cur_delta = i - last_idx;
        while (cur_delta >= 256) {
          deltas_.push_back(255);
          vals_.push_back(0);
          cur_delta -= 255;
        }
        deltas_.push_back(static_cast<uint8_t>(cur_delta));
        vals_.push_back(bin);
        last_idx = i;
      }
    }
    // avoid out of range
    deltas_.push_back(0);
    num_vals_ = static_cast<data_size_t>(vals_.size());

    // reduce memory cost
    deltas_.shrink_to_fit();
    vals_.shrink_to_fit();

    // generate fast index
    GetFastIndex();
  }

  SparseBin<VAL_T>* Clone() override;

  SparseBin(const SparseBin<VAL_T>& other)
      : num_data_(other.num_data_),
        deltas_(other.deltas_),
        vals_(other.vals_),
        num_vals_(other.num_vals_),
        push_buffers_(other.push_buffers_),
        fast_index_(other.fast_index_),
        fast_index_shift_(other.fast_index_shift_) {}

  void InitIndex(data_size_t start_idx, data_size_t* i_delta,
                 data_size_t* cur_pos) const {
    auto idx = start_idx >> fast_index_shift_;
    if (static_cast<size_t>(idx) < fast_index_.size()) {
      const auto fast_pair = fast_index_[start_idx >> fast_index_shift_];
      *i_delta = fast_pair.first;
      *cur_pos = fast_pair.second;
    } else {
      *i_delta = -1;
      *cur_pos = 0;
    }
  }

  const void* GetColWiseData(uint8_t* bit_type, bool* is_sparse, std::vector<BinIterator*>* bin_iterator, const int num_threads) const override;

  const void* GetColWiseData(uint8_t* bit_type, bool* is_sparse, BinIterator** bin_iterator) const override;

 private:
  data_size_t num_data_;
  std::vector<uint8_t, Common::AlignmentAllocator<uint8_t, kAlignedSize>>
      deltas_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, kAlignedSize>> vals_;
  data_size_t num_vals_;
  std::vector<std::vector<std::pair<data_size_t, VAL_T>>> push_buffers_;
  std::vector<std::pair<data_size_t, data_size_t>> fast_index_;
  data_size_t fast_index_shift_;
};

template <typename VAL_T>
SparseBin<VAL_T>* SparseBin<VAL_T>::Clone() {
  return new SparseBin(*this);
}

template <typename VAL_T>
inline uint32_t SparseBinIterator<VAL_T>::RawGet(data_size_t idx) {
  return InnerRawGet(idx);
}

template <typename VAL_T>
inline VAL_T SparseBinIterator<VAL_T>::InnerRawGet(data_size_t idx) {
  while (cur_pos_ < idx) {
    bin_data_->NextNonzeroFast(&i_delta_, &cur_pos_);
  }
  if (cur_pos_ == idx) {
    return bin_data_->vals_[i_delta_];
  } else {
    return 0;
  }
}

template <typename VAL_T>
inline void SparseBinIterator<VAL_T>::Reset(data_size_t start_idx) {
  bin_data_->InitIndex(start_idx, &i_delta_, &cur_pos_);
}

template <typename VAL_T>
BinIterator* SparseBin<VAL_T>::GetIterator(uint32_t min_bin, uint32_t max_bin,
                                           uint32_t most_freq_bin) const {
  return new SparseBinIterator<VAL_T>(this, min_bin, max_bin, most_freq_bin);
}

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_IO_SPARSE_BIN_HPP_
