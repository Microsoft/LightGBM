/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_SPLIT_INFO_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_SPLIT_INFO_HPP_

#ifdef USE_CUDA

#include <LightGBM/meta.h>

namespace LightGBM {

class CUDASplitInfo {
 public:
  bool is_valid;
  int leaf_index;
  double gain;
  int inner_feature_index;
  uint32_t threshold;
  bool default_left;

  double left_sum_gradients;
  double left_sum_hessians;
  int64_t left_sum_of_gradients_hessians;
  data_size_t left_count;
  double left_gain;
  double left_value;

  double right_sum_gradients;
  double right_sum_hessians;
  int64_t right_sum_of_gradients_hessians;
  data_size_t right_count;
  double right_gain;
  double right_value;

  int num_cat_threshold = 0;
  uint32_t* cat_threshold = nullptr;
  int* cat_threshold_real = nullptr;

  __host__ __device__ CUDASplitInfo() {
    num_cat_threshold = 0;
    cat_threshold = nullptr;
    cat_threshold_real = nullptr;
  }

  __host__ __device__ ~CUDASplitInfo() {
    if (num_cat_threshold > 0) {
      if (cat_threshold != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaFree(cat_threshold));
      }
      if (cat_threshold_real != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaFree(cat_threshold_real));
      }
    }
  }

  __host__ __device__ CUDASplitInfo& operator=(const CUDASplitInfo& other) {
    is_valid = other.is_valid;
    leaf_index = other.leaf_index;
    gain = other.gain;
    inner_feature_index = other.inner_feature_index;
    threshold = other.threshold;
    default_left = other.default_left;

    left_sum_gradients = other.left_sum_gradients;
    left_sum_hessians = other.left_sum_hessians;
    left_count = other.left_count;
    left_gain = other.left_gain;
    left_value = other.left_value;

    right_sum_gradients = other.right_sum_gradients;
    right_sum_hessians = other.right_sum_hessians;
    right_count = other.right_count;
    right_gain = other.right_gain;
    right_value = other.right_value;

    num_cat_threshold = other.num_cat_threshold;
    if (num_cat_threshold > 0 && cat_threshold == nullptr) {
      cat_threshold = new uint32_t[num_cat_threshold];
    }
    if (num_cat_threshold > 0 && cat_threshold_real == nullptr) {
      cat_threshold_real = new int[num_cat_threshold];
    }
    if (num_cat_threshold > 0) {
      if (other.cat_threshold != nullptr) {
        for (int i = 0; i < num_cat_threshold; ++i) {
          cat_threshold[i] = other.cat_threshold[i];
        }
      }
      if (other.cat_threshold_real != nullptr) {
        for (int i = 0; i < num_cat_threshold; ++i) {
          cat_threshold_real[i] = other.cat_threshold_real[i];
        }
      }
    }
    return *this;
  }
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_SPLIT_INFO_HPP_
