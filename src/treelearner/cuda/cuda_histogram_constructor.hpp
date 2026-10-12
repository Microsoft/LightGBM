/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
#define LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_row_data.hpp>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/feature_group.h>
#include <LightGBM/tree.h>

#include <memory>
#include <vector>

#include "cuda_leaf_splits.hpp"

#define NUM_DATA_PER_THREAD (400)
#define NUM_THREADS_PER_BLOCK (504)
#define NUM_FEATURE_PER_THREAD_GROUP (28)
#define SUBTRACT_BLOCK_SIZE (1024)
#define FIX_HISTOGRAM_SHARED_MEM_SIZE (1024)
#define FIX_HISTOGRAM_BLOCK_SIZE (512)
#define USED_HISTOGRAM_BUFFER_NUM (8)

namespace LightGBM {

class CUDAHistogramConstructor {
 public:
  CUDAHistogramConstructor(
    const Dataset* train_data,
    const int num_leaves,
    const int num_threads,
    const std::vector<uint32_t>& feature_hist_offsets,
    const int min_data_in_leaf,
    const double min_sum_hessian_in_leaf,
    const int gpu_device_id,
    const bool gpu_use_dp,
    const bool use_discretized_grad,
    const int grad_discretized_bins);

  ~CUDAHistogramConstructor();

  void Init(const Dataset* train_data, TrainingShareStates* share_state);

  void ConstructHistogramForLeaf(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
    const data_size_t global_num_data_in_smaller_leaf,
    const data_size_t global_num_data_in_larger_leaf,
    const data_size_t num_data_in_smaller_leaf,
    const data_size_t num_data_in_larger_leaf,
    const double sum_hessians_in_smaller_leaf,
    const double sum_hessians_in_larger_leaf,
    const uint8_t num_bits_in_histogram_bins);

  void SubtractHistogramForLeaf(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
    const bool use_discretized_grad,
    const uint8_t parent_num_bits_in_histogram_bins,
    const uint8_t smaller_num_bits_in_histogram_bins,
    const uint8_t larger_num_bits_in_histogram_bins);

  void ResetTrainingData(const Dataset* train_data, TrainingShareStates* share_states);

  void ResetConfig(const Config* config);

  void BeforeTrain(const score_t* gradients, const score_t* hessians);

  const hist_t* cuda_hist() const { return cuda_hist_.RawData(); }

  hist_t* cuda_hist_pointer() { return cuda_hist_.RawData(); }

 private:
  void InitFeatureMetaInfo(const Dataset* train_data, const std::vector<uint32_t>& feature_hist_offsets);

  void CalcConstructHistogramKernelDim(
    int* grid_dim_x,
    int* grid_dim_y,
    int* block_dim_x,
    int* block_dim_y,
    const data_size_t num_data_in_smaller_leaf);

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE>
  void LaunchConstructHistogramKernelInner(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE>
  void LaunchConstructHistogramKernelInner0(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE>
  void LaunchConstructHistogramKernelInner1(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE, bool USE_GLOBAL_MEM_BUFFER>
  void LaunchConstructHistogramKernelInner2(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  void LaunchConstructHistogramKernel(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const data_size_t num_data_in_smaller_leaf,
    const uint8_t num_bits_in_histogram_bins);

  void LaunchSubtractHistogramKernel(
    const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
    const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
    const bool use_discretized_grad,
    const uint8_t parent_num_bits_in_histogram_bins,
    const uint8_t smaller_num_bits_in_histogram_bins,
    const uint8_t larger_num_bits_in_histogram_bins);

  // Host memory

  /*! \brief size of training data */
  data_size_t num_data_;
  /*! \brief number of features in training data */
  int num_features_;
  /*! \brief maximum number of leaves */
  int num_leaves_;
  /*! \brief number of threads */
  int num_threads_;
  /*! \brief total number of bins in histogram */
  int num_total_bin_;
  /*! \brief number of bins per feature */
  std::vector<uint32_t> feature_num_bins_;
  /*! \brief offsets in histogram of all features */
  std::vector<uint32_t> feature_hist_offsets_;
  /*! \brief most frequent bins in each feature */
  std::vector<uint32_t> feature_most_freq_bins_;
  /*! \brief minimum number of data allowed per leaf */
  int min_data_in_leaf_;
  /*! \brief minimum sum value of hessians allowed per leaf */
  double min_sum_hessian_in_leaf_;
  /*! \brief cuda stream for histogram construction */
  cudaStream_t cuda_stream_;
  /*! \brief indices of feature whose histograms need to be fixed */
  std::vector<int> need_fix_histogram_features_;
  /*! \brief aligned number of bins of the features whose histograms need to be fixed */
  std::vector<uint32_t> need_fix_histogram_features_num_bin_aligend_;
  /*! \brief minimum number of blocks allowed in the y dimension */
  const int min_grid_dim_y_ = 160;


  // CUDA memory, held by this object

  /*! \brief CUDA row wise data */
  std::unique_ptr<CUDARowData> cuda_row_data_;
  /*! \brief number of bins per feature */
  CUDAVector<uint32_t> cuda_feature_num_bins_;
  /*! \brief offsets in histogram of all features */
  CUDAVector<uint32_t> cuda_feature_hist_offsets_;
  /*! \brief most frequent bins in each feature */
  CUDAVector<uint32_t> cuda_feature_most_freq_bins_;
  /*! \brief CUDA histograms */
  CUDAVector<hist_t> cuda_hist_;
  /*! \brief CUDA histograms buffer for each block */
  CUDAVector<float> cuda_hist_buffer_;
  /*! \brief indices of feature whose histograms need to be fixed */
  CUDAVector<int> cuda_need_fix_histogram_features_;
  /*! \brief aligned number of bins of the features whose histograms need to be fixed */
  CUDAVector<uint32_t> cuda_need_fix_histogram_features_num_bin_aligned_;
  /*! \brief histogram buffer used in histogram subtraction with different number of bits for histogram bins */
  CUDAVector<hist_t> hist_buffer_for_num_bit_change_;

  // CUDA memory, held by other object

  /*! \brief gradients on CUDA */
  const score_t* cuda_gradients_;
  /*! \brief hessians on CUDA */
  const score_t* cuda_hessians_;

  /*! \brief GPU device index */
  const int gpu_device_id_;
  /*! \brief use double precision histogram per block */
  const bool gpu_use_dp_;
  /*! \brief whether to use quantized gradients */
  const bool use_quantized_grad_;
  /*! \brief the number of bins to quantized gradients */
  const int num_grad_quant_bins_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
