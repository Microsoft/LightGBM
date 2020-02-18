/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_THREADING_H_
#define LIGHTGBM_UTILS_THREADING_H_

#include <LightGBM/utils/openmp_wrapper.h>

#include <functional>
#include <vector>

namespace LightGBM {

class Threading {
 public:
  template <typename INDEX_T>
  static inline void BlockInfo(INDEX_T cnt, INDEX_T min_cnt_per_block,
                               int* out_nblock, INDEX_T* block_size) {
    int num_threads = 1;
#pragma omp parallel
#pragma omp master
    { num_threads = omp_get_num_threads(); }
    *out_nblock = std::min<int>(
        num_threads,
        static_cast<int>((cnt + min_cnt_per_block - 1) / min_cnt_per_block));
    if (*out_nblock > 1) {
      *block_size =
          SIZE_ALIGNED((cnt + (*out_nblock) - 1) / (*out_nblock));
    } else {
      *block_size = cnt;
    }
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
  static inline int For(
      INDEX_T start, INDEX_T end,
      INDEX_T min_block_size, const std::function<void(int, INDEX_T, INDEX_T)>& inner_fun) {
    int n_block = 1;
    INDEX_T num_inner = end - start;
    BlockInfo<INDEX_T>(end - start, min_block_size, &n_block, &num_inner);
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < n_block; ++i) {
      OMP_LOOP_EX_BEGIN();
      INDEX_T inner_start = start + num_inner * i;
      INDEX_T inner_end = std::min(end, inner_start + num_inner);
      inner_fun(i, inner_start, inner_end);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    return n_block;
  }
  template <typename INDEX_T>
  static inline void BalancedFor(int n, const std::vector<INDEX_T>& size,
                                 const std::function<void(int)>& inner_fun) {
    int num_threads = 1;
#pragma omp parallel
#pragma omp master
    { num_threads = omp_get_num_threads(); }
    std::vector<std::vector<int>> groups(num_threads, std::vector<int>());
    std::vector<INDEX_T> group_sizes(num_threads, 0);
    for (int i = 0; i < n; ++i) {
      int cur_group =
          static_cast<INDEX_T>(ArrayArgs<INDEX_T>::ArgMin(group_sizes));
      group_sizes[cur_group] += size[i];
      groups[cur_group].push_back(i);
    }
    OMP_INIT_EX();
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < num_threads; ++i) {
      OMP_LOOP_EX_BEGIN();
      for (auto j : groups[i]) {
        inner_fun(j);
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
};

}   // namespace LightGBM

#endif   // LightGBM_UTILS_THREADING_H_
