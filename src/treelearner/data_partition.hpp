/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_DATA_PARTITION_HPP_
#define LIGHTGBM_SRC_TREELEARNER_DATA_PARTITION_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/threading.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace LightGBM {
/*!
* \brief DataPartition is used to store the the partition of data on tree.
*/
class DataPartition {
 public:
  DataPartition(data_size_t num_data, int num_leaves)
      : num_data_(num_data), num_leaves_(num_leaves), runner_(num_data, 512) {
    leaf_begin_.resize(num_leaves_);
    leaf_count_.resize(num_leaves_);
    indices_.resize(num_data_);
    used_data_indices_ = nullptr;
  }

  void ResetLeaves(int num_leaves) {
    num_leaves_ = num_leaves;
    leaf_begin_.resize(num_leaves_);
    leaf_count_.resize(num_leaves_);
  }

  void ResetNumData(int num_data) {
    num_data_ = num_data;
    indices_.resize(num_data_);
    runner_.ReSize(num_data_);
  }

  ~DataPartition() {
  }

  /*!
  * \brief Init, will put all data on the root(leaf_idx = 0)
  */
  void Init() {
    std::fill(leaf_begin_.begin(), leaf_begin_.end(), 0);
    std::fill(leaf_count_.begin(), leaf_count_.end(), 0);
    if (used_data_indices_ == nullptr) {
      // if using all data
      leaf_count_[0] = num_data_;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_data_ >= 1024)
      for (data_size_t i = 0; i < num_data_; ++i) {
        indices_[i] = i;
      }
    } else {
      // if bagging
      leaf_count_[0] = used_data_count_;
      std::memcpy(indices_.data(), used_data_indices_, used_data_count_ * sizeof(data_size_t));
    }
  }

  void ResetByLeafPred(const std::vector<int>& leaf_pred, int num_leaves) {
    ResetLeaves(num_leaves);
    std::vector<std::vector<data_size_t>> indices_per_leaf(num_leaves_);
    for (data_size_t i = 0; i < static_cast<data_size_t>(leaf_pred.size()); ++i) {
      indices_per_leaf[leaf_pred[i]].push_back(i);
    }
    data_size_t offset = 0;
    for (int i = 0; i < num_leaves_; ++i) {
      leaf_begin_[i] = offset;
      leaf_count_[i] = static_cast<data_size_t>(indices_per_leaf[i].size());
      std::copy(indices_per_leaf[i].begin(), indices_per_leaf[i].end(), indices_.begin() + leaf_begin_[i]);
      offset += leaf_count_[i];
    }
  }

  /*!
  * \brief Get the data indices of one leaf
  * \param leaf index of leaf
  * \param out_len number of data on this leaf
  * \return the data indices of this leaf
  */
  const data_size_t* GetIndexOnLeaf(int leaf, data_size_t* out_len) const {
    // copy reference, maybe unsafe, but faster
    data_size_t begin = leaf_begin_[leaf];
    *out_len = leaf_count_[leaf];
    return indices_.data() + begin;
  }

  /*!
  * \brief Split the data
  * \param leaf index of leaf
  * \param dataset dataset that holds the bin data
  * \param feature index of the feature to split on
  * \param threshold threshold that want to split
  * \param num_threshold number of "words" in the bitset for categorical features, 1 for continuous features
  * \param default_left whether missing values go to the left leaf
  * \param right_leaf index of right leaf
  */
  void Split(int leaf, const Dataset* dataset, int feature,
             const uint32_t* threshold, int num_threshold, bool default_left,
             int right_leaf) {
    Common::FunctionTimer fun_timer("DataPartition::Split", global_timer);
    // get leaf boundary
    const data_size_t begin = leaf_begin_[leaf];
    const data_size_t cnt = leaf_count_[leaf];
    auto left_start = indices_.data() + begin;
    const auto left_cnt = runner_.Run<false>(
        cnt,
        [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t* left,
            data_size_t* right) {
          return dataset->Split(feature, threshold, num_threshold, default_left,
                                left_start + cur_start, cur_cnt, left, right);
        },
        left_start);
    leaf_count_[leaf] = left_cnt;
    leaf_begin_[right_leaf] = left_cnt + begin;
    leaf_count_[right_leaf] = cnt - left_cnt;
  }

  /*!
  * \brief SetLabelAt used data indices before training, used for bagging
  * \param used_data_indices indices of used data
  * \param num_used_data number of used data
  */
  void SetUsedDataIndices(const data_size_t* used_data_indices, data_size_t num_used_data) {
    used_data_indices_ = used_data_indices;
    used_data_count_ = num_used_data;
  }

  /*!
  * \brief Get number of data on one leaf
  * \param leaf index of leaf
  * \return number of data of this leaf
  */
  data_size_t leaf_count(int leaf) const { return leaf_count_[leaf]; }

  /*!
  * \brief Get leaf begin
  * \param leaf index of leaf
  * \return begin index of this leaf
  */
  data_size_t leaf_begin(int leaf) const { return leaf_begin_[leaf]; }

  const data_size_t* indices() const { return indices_.data(); }

  /*! \brief Get number of leaves */
  int num_leaves() const { return num_leaves_; }

 private:
  /*! \brief Number of all data */
  data_size_t num_data_;
  /*! \brief Number of all leaves */
  int num_leaves_;
  /*! \brief start index of data on one leaf */
  std::vector<data_size_t> leaf_begin_;
  /*! \brief number of data on one leaf */
  std::vector<data_size_t> leaf_count_;
  /*! \brief Store all data's indices, order by leaf[data_in_leaf0,..,data_leaf1,..] */
  std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>> indices_;
  /*! \brief used data indices, used for bagging */
  const data_size_t* used_data_indices_;
  /*! \brief used data count, used for bagging */
  data_size_t used_data_count_;
  ParallelPartitionRunner<data_size_t, true> runner_;
};

}  // namespace LightGBM
#endif   // LIGHTGBM_SRC_TREELEARNER_DATA_PARTITION_HPP_
