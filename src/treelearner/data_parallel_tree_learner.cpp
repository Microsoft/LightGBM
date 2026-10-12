/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <algorithm>
#include <cstring>
#include <tuple>
#include <vector>

#include "parallel_tree_learner.h"

namespace LightGBM {

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::DataParallelTreeLearner(const Config* config):TREELEARNER_T(config) {
}

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::~DataParallelTreeLearner() {
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  // initialize SerialTreeLearner
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  // Get local rank and global machine size
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  auto max_cat_threshold = this->config_->max_cat_threshold;
  // need to be able to hold smaller and larger best splits in SyncUpGlobalBestSplit
  size_t split_info_size = static_cast<size_t>(SplitInfo::Size(max_cat_threshold) * 2);
  size_t histogram_size = this->config_->use_quantized_grad ?
    static_cast<size_t>(this->share_state_->num_hist_total_bin() * kInt32HistEntrySize) :
    static_cast<size_t>(this->share_state_->num_hist_total_bin() * kHistEntrySize);

  // allocate buffer for communication
  size_t buffer_size = std::max(histogram_size, split_info_size);

  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  is_feature_aggregated_.resize(this->num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  if (this->config_->use_quantized_grad) {
    block_start_int16_.resize(num_machines_);
    block_len_int16_.resize(num_machines_);
  }

  buffer_write_start_pos_.resize(this->num_features_);
  buffer_read_start_pos_.resize(this->num_features_);

  if (this->config_->use_quantized_grad) {
    buffer_write_start_pos_int16_.resize(this->num_features_);
    buffer_read_start_pos_int16_.resize(this->num_features_);
  }

  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::ResetConfig(const Config* config) {
  TREELEARNER_T::ResetConfig(config);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::PrepareBufferPos(
  const std::vector<std::vector<int>>& feature_distribution,
  std::vector<comm_size_t>* block_start,
  std::vector<comm_size_t>* block_len,
  std::vector<comm_size_t>* buffer_write_start_pos,
  std::vector<comm_size_t>* buffer_read_start_pos,
  comm_size_t* reduce_scatter_size,
  size_t hist_entry_size) {
  // get block start and block len for reduce scatter
  *reduce_scatter_size = 0;
  for (int i = 0; i < num_machines_; ++i) {
    (*block_len)[i] = 0;
    for (auto fid : feature_distribution[i]) {
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
        num_bin -= 1;
      }
      (*block_len)[i] += num_bin * hist_entry_size;
    }
    *reduce_scatter_size += (*block_len)[i];
  }

  (*block_start)[0] = 0;
  for (int i = 1; i < num_machines_; ++i) {
    (*block_start)[i] = (*block_start)[i - 1] + (*block_len)[i - 1];
  }

  // get buffer_write_start_pos
  int bin_size = 0;
  for (int i = 0; i < num_machines_; ++i) {
    for (auto fid : feature_distribution[i]) {
      (*buffer_write_start_pos)[fid] = bin_size;
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
        num_bin -= 1;
      }
      bin_size += num_bin * hist_entry_size;
    }
  }

  // get buffer_read_start_pos
  bin_size = 0;
  for (auto fid : feature_distribution[rank_]) {
    (*buffer_read_start_pos)[fid] = bin_size;
    auto num_bin = this->train_data_->FeatureNumBin(fid);
    if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
      num_bin -= 1;
    }
    bin_size += num_bin * hist_entry_size;
  }
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::BeforeTrain() {
  TREELEARNER_T::BeforeTrain();
  // generate feature partition for current tree
  std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  std::vector<int> num_bins_distributed(num_machines_, 0);
  for (int i = 0; i < this->train_data_->num_total_features(); ++i) {
    int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
    if (inner_feature_index == -1) {
      continue;
    }
    if (this->col_sampler_.is_feature_used_bytree()[inner_feature_index]) {
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
      feature_distribution[cur_min_machine].push_back(inner_feature_index);
      auto num_bin = this->train_data_->FeatureNumBin(inner_feature_index);
      if (this->train_data_->FeatureBinMapper(inner_feature_index)->GetMostFreqBin() == 0) {
        num_bin -= 1;
      }
      num_bins_distributed[cur_min_machine] += num_bin;
    }
    is_feature_aggregated_[inner_feature_index] = false;
  }
  // get local used feature
  for (auto fid : feature_distribution[rank_]) {
    is_feature_aggregated_[fid] = true;
  }

  // get block start and block len for reduce scatter
  if (this->config_->use_quantized_grad) {
    PrepareBufferPos(feature_distribution, &block_start_, &block_len_, &buffer_write_start_pos_,
      &buffer_read_start_pos_, &reduce_scatter_size_, kInt32HistEntrySize);
    PrepareBufferPos(feature_distribution, &block_start_int16_, &block_len_int16_, &buffer_write_start_pos_int16_,
      &buffer_read_start_pos_int16_, &reduce_scatter_size_int16_, kInt16HistEntrySize);
  } else {
    PrepareBufferPos(feature_distribution, &block_start_, &block_len_, &buffer_write_start_pos_,
      &buffer_read_start_pos_, &reduce_scatter_size_, kHistEntrySize);
  }

  if (this->config_->use_quantized_grad) {
    // sync global data sumup info
    std::tuple<data_size_t, double, double, int64_t> data(this->smaller_leaf_splits_->num_data_in_leaf(),
                                                          this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
                                                          this->smaller_leaf_splits_->int_sum_gradients_and_hessians());
    int size = sizeof(data);
    std::memcpy(input_buffer_.data(), &data, size);
    // global sumup reduce
    Network::Allreduce(input_buffer_.data(), size, sizeof(std::tuple<data_size_t, double, double, int64_t>), output_buffer_.data(), [](const char *src, char *dst, int type_size, comm_size_t len) {
      comm_size_t used_size = 0;
      const std::tuple<data_size_t, double, double, int64_t> *p1;
      std::tuple<data_size_t, double, double, int64_t> *p2;
      while (used_size < len) {
        p1 = reinterpret_cast<const std::tuple<data_size_t, double, double, int64_t> *>(src);
        p2 = reinterpret_cast<std::tuple<data_size_t, double, double, int64_t> *>(dst);
        std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
        std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
        std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
        std::get<3>(*p2) = std::get<3>(*p2) + std::get<3>(*p1);
        src += type_size;
        dst += type_size;
        used_size += type_size;
      }
    });
    // copy back
    std::memcpy(reinterpret_cast<void*>(&data), output_buffer_.data(), size);
    // set global sumup info
    this->smaller_leaf_splits_->Init(std::get<1>(data), std::get<2>(data), std::get<3>(data));
    // init global data count in leaf
    global_data_count_in_leaf_[0] = std::get<0>(data);
    // reset hist num bits according to global num data
    this->gradient_discretizer_->template SetNumBitsInHistogramBin<true>(0, -1, GetGlobalDataCountInLeaf(0), 0);
  } else {
    // sync global data sumup info
    std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(),
                                                this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
    int size = sizeof(data);
    std::memcpy(input_buffer_.data(), &data, size);
    // global sumup reduce
    Network::Allreduce(input_buffer_.data(), size, sizeof(std::tuple<data_size_t, double, double>), output_buffer_.data(), [](const char *src, char *dst, int type_size, comm_size_t len) {
      comm_size_t used_size = 0;
      const std::tuple<data_size_t, double, double> *p1;
      std::tuple<data_size_t, double, double> *p2;
      while (used_size < len) {
        p1 = reinterpret_cast<const std::tuple<data_size_t, double, double> *>(src);
        p2 = reinterpret_cast<std::tuple<data_size_t, double, double> *>(dst);
        std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
        std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
        std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
        src += type_size;
        dst += type_size;
        used_size += type_size;
      }
    });
    // copy back
    std::memcpy(reinterpret_cast<void*>(&data), output_buffer_.data(), size);
    // set global sumup info
    this->smaller_leaf_splits_->Init(std::get<1>(data), std::get<2>(data));
    // init global data count in leaf
    global_data_count_in_leaf_[0] = std::get<0>(data);
  }
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplits(const Tree* tree) {
  TREELEARNER_T::ConstructHistograms(
      this->col_sampler_.is_feature_used_bytree(), true);
  const int smaller_leaf_index = this->smaller_leaf_splits_->leaf_index();
  const data_size_t local_data_on_smaller_leaf = this->data_partition_->leaf_count(smaller_leaf_index);
  if (local_data_on_smaller_leaf <= 0) {
    // clear histogram buffer before synchronizing
    // otherwise histogram contents from the previous iteration will be sent
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
      if (this->col_sampler_.is_feature_used_bytree()[feature_index] == false)
        continue;
      const BinMapper* feature_bin_mapper = this->train_data_->FeatureBinMapper(feature_index);
      const int offset = static_cast<int>(feature_bin_mapper->GetMostFreqBin() == 0);
      const int num_bin = feature_bin_mapper->num_bin();
      if (this->config_->use_quantized_grad) {
        int32_t* hist_ptr = this->smaller_leaf_histogram_array_[feature_index].RawDataInt32();
        std::memset(reinterpret_cast<void*>(hist_ptr), 0, (num_bin - offset) * kInt32HistEntrySize);
        int16_t* hist_ptr_int16 = this->smaller_leaf_histogram_array_[feature_index].RawDataInt16();
        std::memset(reinterpret_cast<void*>(hist_ptr_int16), 0, (num_bin - offset) * kInt16HistEntrySize);
      } else {
        hist_t* hist_ptr = this->smaller_leaf_histogram_array_[feature_index].RawData();
        std::memset(reinterpret_cast<void*>(hist_ptr), 0, (num_bin - offset) * kHistEntrySize);
      }
    }
  }
  // construct local histograms
  global_timer.Start("DataParallelTreeLearner::ReduceHistogram");
  global_timer.Start("DataParallelTreeLearner::ReduceHistogram::Copy");
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    if (this->col_sampler_.is_feature_used_bytree()[feature_index] == false)
      continue;
    // copy to buffer
    if (this->config_->use_quantized_grad) {
      const uint8_t local_smaller_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<false>(this->smaller_leaf_splits_->leaf_index());
      const uint8_t smaller_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<true>(this->smaller_leaf_splits_->leaf_index());
      if (smaller_leaf_num_bits <= 16) {
        std::memcpy(input_buffer_.data() + buffer_write_start_pos_int16_[feature_index],
                    this->smaller_leaf_histogram_array_[feature_index].RawDataInt16(),
                    this->smaller_leaf_histogram_array_[feature_index].SizeOfInt16Histogram());
      } else {
        if (local_smaller_leaf_num_bits == 32) {
          std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
                      this->smaller_leaf_histogram_array_[feature_index].RawDataInt32(),
                      this->smaller_leaf_histogram_array_[feature_index].SizeOfInt32Histogram());
        } else {
          this->smaller_leaf_histogram_array_[feature_index].CopyFromInt16ToInt32(
            input_buffer_.data() + buffer_write_start_pos_[feature_index]);
        }
      }
    } else {
      std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
                this->smaller_leaf_histogram_array_[feature_index].RawData(),
                this->smaller_leaf_histogram_array_[feature_index].SizeOfHistogram());
    }
  }
  global_timer.Stop("DataParallelTreeLearner::ReduceHistogram::Copy");
  // Reduce scatter for histogram
  global_timer.Start("DataParallelTreeLearner::ReduceHistogram::ReduceScatter");
  if (!this->config_->use_quantized_grad) {
    Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(hist_t), block_start_.data(),
                           block_len_.data(), output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &HistogramSumReducer);
  } else {
    const uint8_t smaller_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<true>(this->smaller_leaf_splits_->leaf_index());
    if (smaller_leaf_num_bits <= 16) {
      Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_int16_, sizeof(int16_t), block_start_int16_.data(),
                            block_len_int16_.data(), output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &Int16HistogramSumReducer);
    } else {
      Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(int_hist_t), block_start_.data(),
                            block_len_.data(), output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &Int32HistogramSumReducer);
    }
  }
  global_timer.Stop("DataParallelTreeLearner::ReduceHistogram::ReduceScatter");
  global_timer.Stop("DataParallelTreeLearner::ReduceHistogram");
  this->FindBestSplitsFromHistograms(
      this->col_sampler_.is_feature_used_bytree(), true, tree);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t>&, bool, const Tree* tree) {
  std::vector<SplitInfo> smaller_bests_per_thread(this->share_state_->num_threads);
  std::vector<SplitInfo> larger_bests_per_thread(this->share_state_->num_threads);
  std::vector<int8_t> smaller_node_used_features =
      this->col_sampler_.GetByNode(tree, this->smaller_leaf_splits_->leaf_index());
  std::vector<int8_t> larger_node_used_features =
      this->col_sampler_.GetByNode(tree, this->larger_leaf_splits_->leaf_index());
  double smaller_leaf_parent_output = this->GetParentOutput(tree, this->smaller_leaf_splits_.get());
  double larger_leaf_parent_output = this->GetParentOutput(tree, this->larger_leaf_splits_.get());

  if (this->config_->use_quantized_grad && this->larger_leaf_splits_ != nullptr && this->larger_leaf_splits_->leaf_index() >= 0) {
    const int parent_index = std::min(this->smaller_leaf_splits_->leaf_index(), this->larger_leaf_splits_->leaf_index());
    const uint8_t parent_num_bits = this->gradient_discretizer_->template GetHistBitsInNode<true>(parent_index);
    const uint8_t larger_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<true>(this->larger_leaf_splits_->leaf_index());
    const uint8_t smaller_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<true>(this->smaller_leaf_splits_->leaf_index());
    if (parent_num_bits > 16 && larger_leaf_num_bits <= 16) {
      CHECK_LE(smaller_leaf_num_bits, 16);
      OMP_INIT_EX();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
        OMP_LOOP_EX_BEGIN();
        if (!is_feature_aggregated_[feature_index]) continue;
        this->larger_leaf_histogram_array_[feature_index].CopyToBuffer(this->gradient_discretizer_->GetChangeHistBitsBuffer(feature_index));
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    }
  }

  OMP_INIT_EX();
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_aggregated_[feature_index]) continue;
    const int tid = omp_get_thread_num();
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    // restore global histograms from buffer
    if (this->config_->use_quantized_grad) {
      const uint8_t smaller_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<true>(this->smaller_leaf_splits_->leaf_index());
      if (smaller_leaf_num_bits <= 16) {
        this->smaller_leaf_histogram_array_[feature_index].FromMemoryInt16(
          output_buffer_.data() + buffer_read_start_pos_int16_[feature_index]);
      } else {
        this->smaller_leaf_histogram_array_[feature_index].FromMemoryInt32(
          output_buffer_.data() + buffer_read_start_pos_[feature_index]);
      }
    } else {
      this->smaller_leaf_histogram_array_[feature_index].FromMemory(
        output_buffer_.data() + buffer_read_start_pos_[feature_index]);
    }

    if (this->config_->use_quantized_grad) {
      const uint8_t smaller_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<true>(this->smaller_leaf_splits_->leaf_index());
      const int64_t int_sum_gradient_and_hessian = this->smaller_leaf_splits_->int_sum_gradients_and_hessians();
      if (smaller_leaf_num_bits <= 16) {
        this->train_data_->template FixHistogramInt<int32_t, int32_t, 16, 16>(
          feature_index,
          int_sum_gradient_and_hessian,
          reinterpret_cast<hist_t*>(this->smaller_leaf_histogram_array_[feature_index].RawDataInt16()));
      } else {
        this->train_data_->template FixHistogramInt<int64_t, int64_t, 32, 32>(
          feature_index,
          int_sum_gradient_and_hessian,
          reinterpret_cast<hist_t*>(this->smaller_leaf_histogram_array_[feature_index].RawDataInt32()));
      }
    } else {
      this->train_data_->FixHistogram(feature_index,
                                      this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
                                      this->smaller_leaf_histogram_array_[feature_index].RawData());
    }

    this->ComputeBestSplitForFeature(
        this->smaller_leaf_histogram_array_, feature_index, real_feature_index,
        smaller_node_used_features[feature_index],
        GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->leaf_index()),
        this->smaller_leaf_splits_.get(),
        &smaller_bests_per_thread[tid],
        smaller_leaf_parent_output);

    // only root leaf
    if (this->larger_leaf_splits_ == nullptr || this->larger_leaf_splits_->leaf_index() < 0) continue;

    // construct histgroms for large leaf, we init larger leaf as the parent, so we can just subtract the smaller leaf's histograms
    if (this->config_->use_quantized_grad) {
      const int parent_index = std::min(this->smaller_leaf_splits_->leaf_index(), this->larger_leaf_splits_->leaf_index());
      const uint8_t parent_num_bits = this->gradient_discretizer_->template GetHistBitsInNode<true>(parent_index);
      const uint8_t larger_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<true>(this->larger_leaf_splits_->leaf_index());
      const uint8_t smaller_leaf_num_bits = this->gradient_discretizer_->template GetHistBitsInLeaf<true>(this->smaller_leaf_splits_->leaf_index());
      if (parent_num_bits <= 16) {
        CHECK_LE(smaller_leaf_num_bits, 16);
        CHECK_LE(larger_leaf_num_bits, 16);
        this->larger_leaf_histogram_array_[feature_index].template Subtract<true, int32_t, int32_t, int32_t, 16, 16, 16>(
              this->smaller_leaf_histogram_array_[feature_index]);
      } else if (larger_leaf_num_bits <= 16) {
        CHECK_LE(smaller_leaf_num_bits, 16);
        this->larger_leaf_histogram_array_[feature_index].template Subtract<true, int64_t, int32_t, int32_t, 32, 16, 16>(
            this->smaller_leaf_histogram_array_[feature_index], this->gradient_discretizer_->GetChangeHistBitsBuffer(feature_index));
      } else if (smaller_leaf_num_bits <= 16) {
        this->larger_leaf_histogram_array_[feature_index].template Subtract<true, int64_t, int32_t, int64_t, 32, 16, 32>(
              this->smaller_leaf_histogram_array_[feature_index]);
      } else {
        this->larger_leaf_histogram_array_[feature_index].template Subtract<true, int64_t, int64_t, int64_t, 32, 32, 32>(
              this->smaller_leaf_histogram_array_[feature_index]);
      }
    } else {
      this->larger_leaf_histogram_array_[feature_index].Subtract(
        this->smaller_leaf_histogram_array_[feature_index]);
    }

    this->ComputeBestSplitForFeature(
        this->larger_leaf_histogram_array_, feature_index, real_feature_index,
        larger_node_used_features[feature_index],
        GetGlobalDataCountInLeaf(this->larger_leaf_splits_->leaf_index()),
        this->larger_leaf_splits_.get(),
        &larger_bests_per_thread[tid],
        larger_leaf_parent_output);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_bests_per_thread);
  int leaf = this->smaller_leaf_splits_->leaf_index();
  this->best_split_per_leaf_[leaf] = smaller_bests_per_thread[smaller_best_idx];

  if (this->larger_leaf_splits_ != nullptr &&  this->larger_leaf_splits_->leaf_index() >= 0) {
    leaf = this->larger_leaf_splits_->leaf_index();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_bests_per_thread);
    this->best_split_per_leaf_[leaf] = larger_bests_per_thread[larger_best_idx];
  }

  SplitInfo smaller_best_split, larger_best_split;
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->leaf_index() >= 0) {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()];
  }

  // sync global best info
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);

  // set best split
  this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()] = smaller_best_split;
  if (this->larger_leaf_splits_->leaf_index() >= 0) {
    this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()] = larger_best_split;
  }
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  TREELEARNER_T::SplitInner(tree, best_Leaf, left_leaf, right_leaf, false);
  const SplitInfo& best_split_info = this->best_split_per_leaf_[best_Leaf];
  // need update global number of data in leaf
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
  // reset hist num bits according to global num data
  if (this->config_->use_quantized_grad) {
    this->gradient_discretizer_->template SetNumBitsInHistogramBin<true>(*left_leaf, *right_leaf, GetGlobalDataCountInLeaf(*left_leaf), GetGlobalDataCountInLeaf(*right_leaf));
  }
}

// instantiate template classes, otherwise linker cannot find the code
template class DataParallelTreeLearner<GPUTreeLearner>;
template class DataParallelTreeLearner<SerialTreeLearner>;

}  // namespace LightGBM
