/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_DATA_PARTITION_HPP_
#define LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_DATA_PARTITION_HPP_

#ifdef USE_CUDA

#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/tree.h>

#include <vector>

#include <LightGBM/cuda/cuda_column_data.hpp>
#include <LightGBM/cuda/cuda_split_info.hpp>
#include <LightGBM/cuda/cuda_tree.hpp>

#include "cuda_leaf_splits.hpp"

#define FILL_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)
#define SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)
#define AGGREGATE_BLOCK_SIZE_DATA_PARTITION (1024)

namespace LightGBM {

class CUDADataPartition: public NCCLInfo {
 public:
  CUDADataPartition(
    const Dataset* train_data,
    const int num_total_bin,
    const int num_leaves,
    const int num_threads,
    const bool use_quantized_grad,
    hist_t* cuda_hist);

  ~CUDADataPartition();

  void Init();

  void BeforeTrain();

  void Split(
    // input best split info
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    const int leaf_best_split_feature,
    const uint32_t leaf_best_split_threshold,
    const uint32_t* categorical_bitset,
    const int categorical_bitset_len,
    const uint8_t leaf_best_split_default_left,
    const data_size_t num_data_in_leaf,
    const data_size_t leaf_data_start,
    // for leaf information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians,
    double* left_leaf_sum_of_gradients,
    double* right_leaf_sum_of_gradients,
    data_size_t* global_left_leaf_num_data,
    data_size_t* global_right_leaf_num_data);

  void UpdateTrainScore(const Tree* tree, double* cuda_scores);

  void SetUsedDataIndices(const data_size_t* used_indices, const data_size_t num_used_indices);

  void SetBaggingSubset(const Dataset* subset);

  void ResetTrainingData(const Dataset* train_data, const int num_total_bin, hist_t* cuda_hist);

  void ResetConfig(const Config* config, hist_t* cuda_hist);

  void ResetByLeafPred(const std::vector<int>& leaf_pred, int num_leaves);

  void ReduceLeafGradStat(
    const score_t* gradients, const score_t* hessians,
    CUDATree* tree, double* leaf_grad_stat_buffer, double* leaf_hess_state_buffer) const;

  data_size_t root_num_data() const {
    if (use_bagging_) {
      return num_used_indices_;
    } else {
      return num_data_;
    }
  }

  const data_size_t* cuda_data_indices() const { return cuda_data_indices_.RawData(); }

  const data_size_t* cuda_leaf_num_data() const { return cuda_leaf_num_data_.RawData(); }

  const data_size_t* cuda_leaf_data_start() const { return cuda_leaf_data_start_.RawData(); }

  const int* cuda_data_index_to_leaf_index() const { return cuda_data_index_to_leaf_index_.RawData(); }

  bool use_bagging() const { return use_bagging_; }

 private:
  void CalcBlockDim(const data_size_t num_data_in_leaf);

  void GenDataToLeftBitVector(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint32_t* categorical_bitset,
    const int categorical_bitset_len,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index);

  void SplitInner(
    // input best split info
    const data_size_t num_data_in_leaf,
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    // for leaf splits information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians,
    double* left_leaf_sum_of_gradients,
    double* right_leaf_sum_of_gradients,
    data_size_t* global_left_leaf_num_data,
    data_size_t* global_right_leaf_num_data);

  // kernel launch functions
  void LaunchFillDataIndicesBeforeTrain();

  void LaunchSplitInnerKernel(
    // input best split info
    const data_size_t num_data_in_leaf,
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    // for leaf splits information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians,
    double* left_leaf_sum_of_gradients,
    double* right_leaf_sum_of_gradients,
    data_size_t* global_left_leaf_num_data,
    data_size_t* global_right_leaf_num_data);

  void LaunchGenDataToLeftBitVectorKernel(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index);

  void LaunchGenDataToLeftBitVectorCategoricalKernel(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t* bitset,
    const int bitset_len,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index);

#define GenDataToLeftBitVectorKernel_PARAMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin, \
  const uint32_t min_bin, \
  const uint8_t split_default_to_left, \
  const uint8_t split_missing_default_to_left

  template <typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool missing_is_zero,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner0(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner1(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner2(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner3(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool max_bin_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelInner4(
    GenDataToLeftBitVectorKernel_PARAMS,
    const bool is_single_feature_in_column);

#undef GenDataToLeftBitVectorKernel_PARAMS

#define UpdateDataIndexToLeafIndexKernel_PARAMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin_ref, \
  const uint32_t min_bin_ref, \
  const int left_leaf_index, \
  const int right_leaf_index, \
  const int default_leaf_index, \
  const int missing_default_leaf_index

  template <typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool missing_is_zero,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner0(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner1(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner2(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool mfb_is_na,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner3(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool max_to_left,
    const bool is_single_feature_in_column);

  template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel_Inner4(
    UpdateDataIndexToLeafIndexKernel_PARAMS,
    const bool is_single_feature_in_column);

#undef UpdateDataIndexToLeafIndexKernel_PARAMS

  void LaunchAddPredictionToScoreKernel(const double* leaf_value, double* cuda_scores);

  void LaunchFillDataIndexToLeafIndex();

  void LaunchReduceLeafGradStat(
    const score_t* gradients, const score_t* hessians,
    CUDATree* tree, double* leaf_grad_stat_buffer, double* leaf_hess_state_buffer) const;

  // Host memory

  // dataset information
  /*! \brief number of training data */
  data_size_t num_data_;
  /*! \brief number of features in training data */
  int num_features_;
  /*! \brief number of total bins in training data */
  int num_total_bin_;
  /*! \brief bin data stored by column */
  const CUDAColumnData* cuda_column_data_;
  /*! \brief grid dimension when splitting one leaf */
  int grid_dim_;
  /*! \brief block dimension when splitting one leaf */
  int block_dim_;
  /*! \brief data indices used in this iteration */
  const data_size_t* used_indices_;
  /*! \brief marks whether a feature is a categorical feature */
  std::vector<bool> is_categorical_feature_;
  /*! \brief marks whether a feature is the only feature in its group */
  std::vector<bool> is_single_feature_in_column_;

  // config information
  /*! \brief maximum number of leaves in a tree */
  int num_leaves_;
  /*! \brief number of threads */
  int num_threads_;
  /*! \brief whether to use quantized gradients */
  bool use_quantized_grad_;

  // per iteration information
  /*! \brief whether bagging is used in this iteration */
  bool use_bagging_;
  /*! \brief number of used data indices in this iteration */
  data_size_t num_used_indices_;

  // tree structure information
  /*! \brief current number of leaves in tree */
  int cur_num_leaves_;

  // split algorithm related
  /*! \brief maximum number of blocks to aggregate after finding bit vector by blocks */
  int max_num_split_indices_blocks_;

  // CUDA streams
  /*! \brief cuda streams used for asynchronizing kernel computing and memory copy */
  std::vector<cudaStream_t> cuda_streams_;


  // CUDA memory, held by this object

  // tree structure information
  /*! \brief data indices by leaf */
  CUDAVector<data_size_t> cuda_data_indices_;
  /*! \brief start position of each leaf in cuda_data_indices_ */
  CUDAVector<data_size_t> cuda_leaf_data_start_;
  /*! \brief end position of each leaf in cuda_data_indices_ */
  CUDAVector<data_size_t> cuda_leaf_data_end_;
  /*! \brief number of data in each leaf */
  CUDAVector<data_size_t> cuda_leaf_num_data_;
  /*! \brief records the histogram of each leaf */
  CUDAVector<hist_t*> cuda_hist_pool_;
  /*! \brief records the value of each leaf */
  CUDAVector<double> cuda_leaf_output_;

  // split data algorithm related
  CUDAVector<uint16_t> cuda_block_to_left_offset_;
  /*! \brief maps data index to leaf index, for adding scores to training data set */
  CUDAVector<int> cuda_data_index_to_leaf_index_;
  /*! \brief prefix sum of number of data going to left in all blocks */
  CUDAVector<data_size_t> cuda_block_data_to_left_offset_;
  /*! \brief prefix sum of number of data going to right in all blocks */
  CUDAVector<data_size_t> cuda_block_data_to_right_offset_;
  /*! \brief buffer for splitting data indices, will be copied back to cuda_data_indices_ after split */
  CUDAVector<data_size_t> cuda_out_data_indices_in_leaf_;

  // split tree structure algorithm related
  /*! \brief buffer to store split information, prepared to be copied to cpu */
  CUDAVector<int> cuda_split_info_buffer_;

  // dataset information
  /*! \brief number of data in training set, for initialization of cuda_leaf_num_data_ and cuda_leaf_data_end_ */
  CUDAVector<data_size_t> cuda_num_data_;


  // CUDA memory, held by other object

  // dataset information
  /*! \brief beginning of histograms, for initialization of cuda_hist_pool_ */
  hist_t* cuda_hist_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_TREELEARNER_CUDA_CUDA_DATA_PARTITION_HPP_
