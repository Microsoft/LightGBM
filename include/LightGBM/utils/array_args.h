/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_ARRAY_AGRS_H_
#define LIGHTGBM_UTILS_ARRAY_AGRS_H_

#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/threading.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace LightGBM {

/*!
* \brief Contains some operation for an array, e.g. ArgMax, TopK.
*/
template<typename VAL_T>
class ArrayArgs {
 public:
  inline static size_t ArgMaxMT(const std::vector<VAL_T>& array) {
    int num_threads = OMP_NUM_THREADS();
    std::vector<size_t> arg_maxs(num_threads, 0);
    int n_blocks = Threading::For<size_t>(
        0, array.size(), 1024,
        [&array, &arg_maxs](int i, size_t start, size_t end) {
          size_t arg_max = start;
          for (size_t j = start + 1; j < end; ++j) {
            if (array[j] > array[arg_max]) {
              arg_max = j;
            }
          }
          arg_maxs[i] = arg_max;
        });
    size_t ret = arg_maxs[0];
    for (int i = 1; i < n_blocks; ++i) {
      if (array[arg_maxs[i]] > array[ret]) {
        ret = arg_maxs[i];
      }
    }
    return ret;
  }
  inline static size_t ArgMax(const std::vector<VAL_T>& array) {
    if (array.empty()) {
      return 0;
    }
    if (array.size() > 1024) {
      return ArgMaxMT(array);
    } else {
      size_t arg_max = 0;
      for (size_t i = 1; i < array.size(); ++i) {
        if (array[i] > array[arg_max]) {
          arg_max = i;
        }
      }
      return arg_max;
    }
  }

  inline static size_t ArgMin(const std::vector<VAL_T>& array) {
    if (array.empty()) {
      return 0;
    }
    size_t arg_min = 0;
    for (size_t i = 1; i < array.size(); ++i) {
      if (array[i] < array[arg_min]) {
        arg_min = i;
      }
    }
    return arg_min;
  }

  inline static size_t ArgMax(const VAL_T* array, size_t n) {
    if (n <= 0) {
      return 0;
    }
    size_t arg_max = 0;
    for (size_t i = 1; i < n; ++i) {
      if (array[i] > array[arg_max]) {
        arg_max = i;
      }
    }
    return arg_max;
  }

  inline static size_t ArgMin(const VAL_T* array, size_t n) {
    if (n <= 0) {
      return 0;
    }
    size_t arg_min = 0;
    for (size_t i = 1; i < n; ++i) {
      if (array[i] < array[arg_min]) {
        arg_min = i;
      }
    }
    return arg_min;
  }

  inline static void Partition(std::vector<VAL_T>* arr, int start, int end, int* l, int* r) {
    int i = start - 1;
    int j = end - 1;
    int p = i;
    int q = j;
    if (start >= end - 1) {
      *l = start - 1;
      *r = end;
      return;
    }
    std::vector<VAL_T>& ref = *arr;
    VAL_T v = ref[end - 1];
    for (;;) {
      while (ref[++i] > v) {}
      while (v > ref[--j]) { if (j == start) { break; } }
      if (i >= j) { break; }
      std::swap(ref[i], ref[j]);
      if (ref[i] == v) { p++; std::swap(ref[p], ref[i]); }
      if (v == ref[j]) { q--; std::swap(ref[j], ref[q]); }
    }
    std::swap(ref[i], ref[end - 1]);
    j = i - 1;
    i = i + 1;
    for (int k = start; k <= p; k++, j--) { std::swap(ref[k], ref[j]); }
    for (int k = end - 2; k >= q; k--, i++) { std::swap(ref[i], ref[k]); }
    *l = j;
    *r = i;
  }

  // Note: k refer to index here. e.g. k=0 means get the max number.
  inline static int ArgMaxAtK(std::vector<VAL_T>* arr, int start, int end, int k) {
    if (start >= end - 1) {
      return start;
    }
    int l = start;
    int r = end - 1;
    Partition(arr, start, end, &l, &r);
    // if find or all elements are the same.
    if ((k > l && k < r) || (l == start - 1 && r == end - 1)) {
      return k;
    } else if (k <= l) {
      return ArgMaxAtK(arr, start, l + 1, k);
    } else {
      return ArgMaxAtK(arr, r, end, k);
    }
  }

  // Note: k is 1-based here. e.g. k=3 means get the top-3 numbers.
  inline static void MaxK(const std::vector<VAL_T>& array, int k, std::vector<VAL_T>* out) {
    out->clear();
    if (k <= 0) {
      return;
    }
    for (auto val : array) {
      out->push_back(val);
    }
    if (static_cast<size_t>(k) >= array.size()) {
      return;
    }
    ArgMaxAtK(out, 0, static_cast<int>(out->size()), k - 1);
    out->erase(out->begin() + k, out->end());
  }

  inline static void Assign(std::vector<VAL_T>* array, VAL_T t, size_t n) {
    array->resize(n);
    for (size_t i = 0; i < array->size(); ++i) {
      (*array)[i] = t;
    }
  }

  inline static bool CheckAllZero(const std::vector<VAL_T>& array) {
    for (size_t i = 0; i < array.size(); ++i) {
      if (array[i] != VAL_T(0)) {
        return false;
      }
    }
    return true;
  }

  inline static bool CheckAll(const std::vector<VAL_T>& array, VAL_T t) {
    for (size_t i = 0; i < array.size(); ++i) {
      if (array[i] != t) {
        return false;
      }
    }
    return true;
  }

  static double CalculateThresholdMVS(std::vector<VAL_T>* gradients, data_size_t begin, data_size_t end,
                                             const double sample_size) {
    double current_sum_small = 0.0;
    data_size_t big_grad_size = 0;

    while (begin != end) {
      data_size_t middle_begin = begin - 1, middle_end = end;
      ArrayArgs<score_t>::Partition(gradients, begin, end, &middle_begin, &middle_end);
      ++middle_begin;  // for half intervals
      const data_size_t n_middle = middle_end - middle_begin;
      const data_size_t large_size = middle_begin - begin;

      const double sum_small = std::accumulate(gradients->begin() + middle_end, gradients->begin() + end, 0.0);
      const double sum_middle = (*gradients)[middle_begin] * n_middle;

      const double
          current_sampling_rate = (current_sum_small + sum_small) / (*gradients)[middle_begin] + big_grad_size + n_middle + large_size;

      if (current_sampling_rate > sample_size) {
        current_sum_small += sum_small + sum_middle;
        end = middle_begin;
      } else {
        big_grad_size += n_middle + large_size;
        begin = middle_end;
      }
    }

    return current_sum_small / (sample_size - big_grad_size + kEpsilon);
  }
};

}  // namespace LightGBM

#endif   // LightGBM_UTILS_ARRAY_AGRS_H_
