/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_THREADING_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_THREADING_H_

#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace LightGBM {

class Threading {
 public:
  template <typename INDEX_T>
  static inline void BlockInfo(INDEX_T cnt, INDEX_T min_cnt_per_block,
                               int* out_nblock, INDEX_T* block_size) {
    int num_threads = OMP_NUM_THREADS();
    BlockInfo<INDEX_T>(num_threads, cnt, min_cnt_per_block, out_nblock,
                       block_size);
  }

  template <typename INDEX_T>
  static inline void BlockInfo(int num_threads, INDEX_T cnt,
                               INDEX_T min_cnt_per_block, int* out_nblock,
                               INDEX_T* block_size) {
    *out_nblock = std::min<int>(
        num_threads,
        static_cast<int>((cnt + min_cnt_per_block - 1) / min_cnt_per_block));
    if (*out_nblock > 1) {
      *block_size = SIZE_ALIGNED((cnt + (*out_nblock) - 1) / (*out_nblock));
    } else {
      *block_size = cnt;
    }
  }

  template <typename INDEX_T>
  static inline void BlockInfoForceSize(int num_threads, INDEX_T cnt,
                                        INDEX_T min_cnt_per_block,
                                        int* out_nblock, INDEX_T* block_size) {
    *out_nblock = std::min<int>(
        num_threads,
        static_cast<int>((cnt + min_cnt_per_block - 1) / min_cnt_per_block));
    if (*out_nblock > 1) {
      *block_size = (cnt + (*out_nblock) - 1) / (*out_nblock);
      // force the block size to the times of min_cnt_per_block
      *block_size = (*block_size + min_cnt_per_block - 1) / min_cnt_per_block *
                    min_cnt_per_block;
    } else {
      *block_size = cnt;
    }
  }

  template <typename INDEX_T>
  static inline void BlockInfoForceSize(INDEX_T cnt, INDEX_T min_cnt_per_block,
                                        int* out_nblock, INDEX_T* block_size) {
    int num_threads = OMP_NUM_THREADS();
    BlockInfoForceSize<INDEX_T>(num_threads, cnt, min_cnt_per_block, out_nblock,
                                block_size);
  }

  template <typename INDEX_T>
  static inline int For(
      INDEX_T start, INDEX_T end, INDEX_T min_block_size,
      const std::function<void(int, INDEX_T, INDEX_T)>& inner_fun) {
    int n_block = 1;
    INDEX_T num_inner = end - start;
    BlockInfo<INDEX_T>(num_inner, min_block_size, &n_block, &num_inner);
    OMP_INIT_EX();
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1)
    for (int i = 0; i < n_block; ++i) {
      OMP_LOOP_EX_BEGIN();
      INDEX_T inner_start = start + num_inner * i;
      INDEX_T inner_end = std::min(end, inner_start + num_inner);
      if (inner_start < inner_end) {
          inner_fun(i, inner_start, inner_end);
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    return n_block;
  }
};

template <typename INDEX_T, bool TWO_BUFFER>
class ParallelPartitionRunner {
 public:
  ParallelPartitionRunner(INDEX_T num_data, INDEX_T min_block_size)
      : min_block_size_(min_block_size) {
    num_threads_ = OMP_NUM_THREADS();
    left_.resize(num_data);
    if (TWO_BUFFER) {
      right_.resize(num_data);
    }
    offsets_.resize(num_threads_);
    left_cnts_.resize(num_threads_);
    right_cnts_.resize(num_threads_);
    left_write_pos_.resize(num_threads_);
    right_write_pos_.resize(num_threads_);
  }

  ~ParallelPartitionRunner() {}

  void ReSize(INDEX_T num_data) {
    left_.resize(num_data);
    if (TWO_BUFFER) {
      right_.resize(num_data);
    }
  }

  template<bool FORCE_SIZE>
  INDEX_T Run(
      INDEX_T cnt,
      const std::function<INDEX_T(int, INDEX_T, INDEX_T, INDEX_T*, INDEX_T*)>& func,
      INDEX_T* out) {
    if (cnt == 0) {
      return 0;
    }
    int nblock = 1;
    INDEX_T inner_size = cnt;
    if (FORCE_SIZE) {
      Threading::BlockInfoForceSize<INDEX_T>(num_threads_, cnt, min_block_size_,
                                             &nblock, &inner_size);
    } else {
      Threading::BlockInfo<INDEX_T>(num_threads_, cnt, min_block_size_, &nblock,
                                    &inner_size);
    }

    OMP_INIT_EX();
#pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
    for (int i = 0; i < nblock; ++i) {
      OMP_LOOP_EX_BEGIN();
      INDEX_T cur_start = i * inner_size;
      INDEX_T cur_cnt = std::min(inner_size, cnt - cur_start);
      offsets_[i] = cur_start;
      if (cur_cnt <= 0) {
        left_cnts_[i] = 0;
        right_cnts_[i] = 0;
        continue;
      }
      auto left_ptr = left_.data() + cur_start;
      INDEX_T* right_ptr = nullptr;
      if (TWO_BUFFER) {
        right_ptr = right_.data() + cur_start;
      }
      // split data inner, reduce the times of function called
      INDEX_T cur_left_count =
          func(i, cur_start, cur_cnt, left_ptr, right_ptr);
      if (!TWO_BUFFER) {
        // reverse for one buffer
        std::reverse(left_ptr + cur_left_count, left_ptr + cur_cnt);
      }
      left_cnts_[i] = cur_left_count;
      right_cnts_[i] = cur_cnt - cur_left_count;
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();

    left_write_pos_[0] = 0;
    right_write_pos_[0] = 0;
    for (int i = 1; i < nblock; ++i) {
      left_write_pos_[i] = left_write_pos_[i - 1] + left_cnts_[i - 1];
      right_write_pos_[i] = right_write_pos_[i - 1] + right_cnts_[i - 1];
    }
    data_size_t left_cnt = left_write_pos_[nblock - 1] + left_cnts_[nblock - 1];

    auto right_start = out + left_cnt;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
    for (int i = 0; i < nblock; ++i) {
      std::copy_n(left_.data() + offsets_[i], left_cnts_[i],
                  out + left_write_pos_[i]);
      if (TWO_BUFFER) {
        std::copy_n(right_.data() + offsets_[i], right_cnts_[i],
                    right_start + right_write_pos_[i]);
      } else {
        std::copy_n(left_.data() + offsets_[i] + left_cnts_[i], right_cnts_[i],
                    right_start + right_write_pos_[i]);
      }
    }
    return left_cnt;
  }

 private:
  int num_threads_;
  INDEX_T min_block_size_;
  std::vector<INDEX_T> left_;
  std::vector<INDEX_T> right_;
  std::vector<INDEX_T> offsets_;
  std::vector<INDEX_T> left_cnts_;
  std::vector<INDEX_T> right_cnts_;
  std::vector<INDEX_T> left_write_pos_;
  std::vector<INDEX_T> right_write_pos_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_THREADING_H_
