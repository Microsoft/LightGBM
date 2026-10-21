/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/utils/common.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <tuple>
#include <vector>

#include "parallel_tree_learner.h"

namespace LightGBM {

template <typename TREELEARNER_T>
VotingParallelTreeLearner<TREELEARNER_T>::VotingParallelTreeLearner(const Config* config):TREELEARNER_T(config) {
  top_k_ = this->config_->top_k;
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  // limit top k
  if (top_k_ > this->num_features_) {
    top_k_ = this->num_features_;
  }
  // get max bin
  int max_bin = 0;
  for (int i = 0; i < this->num_features_; ++i) {
    if (max_bin < this->train_data_->FeatureNumBin(i)) {
      max_bin = this->train_data_->FeatureNumBin(i);
    }
  }
  // calculate buffer size
  size_t buffer_size = 2 * top_k_ * std::max(max_bin * kHistEntrySize, sizeof(LightSplitInfo) * num_machines_);
  auto max_cat_threshold = this->config_->max_cat_threshold;
  // need to be able to hold smaller and larger best splits in SyncUpGlobalBestSplit
  size_t split_info_size = static_cast<size_t>(SplitInfo::Size(max_cat_threshold) * 2);
  buffer_size = std::max(buffer_size, split_info_size);
  // left and right on same time, so need double size
  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  smaller_is_feature_aggregated_.resize(this->num_features_);
  larger_is_feature_aggregated_.resize(this->num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  smaller_buffer_read_start_pos_.resize(this->num_features_);
  larger_buffer_read_start_pos_.resize(this->num_features_);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);

  smaller_leaf_splits_global_.reset(new LeafSplits(train_data->num_data(), this->config_));
  larger_leaf_splits_global_.reset(new LeafSplits(train_data->num_data(), this->config_));

  local_config_ = *this->config_;
  local_config_.min_data_in_leaf /= num_machines_;
  local_config_.min_sum_hessian_in_leaf /= num_machines_;

  this->histogram_pool_.ResetConfig(train_data, &local_config_);

  // initialize histograms for global
  smaller_leaf_histogram_array_global_.reset(new FeatureHistogram[this->num_features_]);
  larger_leaf_histogram_array_global_.reset(new FeatureHistogram[this->num_features_]);
  std::vector<uint32_t> offsets = this->share_state_->feature_hist_offsets();
  int num_total_bin = this->share_state_->num_hist_total_bin();
  smaller_leaf_histogram_data_.resize(num_total_bin * 2);
  larger_leaf_histogram_data_.resize(num_total_bin * 2);
  HistogramPool::SetFeatureInfo<true, true>(train_data, this->config_, &feature_metas_);
  for (int j = 0; j < train_data->num_features(); ++j) {
    smaller_leaf_histogram_array_global_[j].Init(smaller_leaf_histogram_data_.data() + offsets[j] * 2, &feature_metas_[j]);
    larger_leaf_histogram_array_global_[j].Init(larger_leaf_histogram_data_.data() + offsets[j] * 2, &feature_metas_[j]);
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::ResetConfig(const Config* config) {
  TREELEARNER_T::ResetConfig(config);

  local_config_ = *this->config_;
  local_config_.min_data_in_leaf /= num_machines_;
  local_config_.min_sum_hessian_in_leaf /= num_machines_;

  this->histogram_pool_.ResetConfig(this->train_data_, &local_config_);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);

  HistogramPool::SetFeatureInfo<false, true>(this->train_data_, config, &feature_metas_);
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::SyncAfsGainsAcrossMachines() {
  if (!this->config_->afs_enable || num_machines_ <= 1) {
    return;
  }
  // SUM all-reduce reconstructs the full global per-feature gain vector on every
  // worker, so the gain EMA (and thus the feature selection) is identical across
  // workers. See DataParallelTreeLearner for the detailed rationale.
  this->afs_feature_gain_current_tree_ =
      Network::GlobalSum(&this->afs_feature_gain_current_tree_);
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::SyncAfsSelectedFeaturesAcrossMachines() {
  if (!this->config_->afs_enable || num_machines_ <= 1) {
    return;
  }
  // Force every worker to adopt rank 0's selected-feature mask.
  const int n = static_cast<int>(this->afs_selected_features_.size());
  std::vector<int> mask(n);
  for (int i = 0; i < n; ++i) {
    mask[i] = (rank_ == 0) ? static_cast<int>(this->afs_selected_features_[i]) : 0;
  }
  std::vector<int> global_mask = Network::GlobalSum(&mask);
  for (int i = 0; i < n; ++i) {
    this->afs_selected_features_[i] = static_cast<int8_t>(global_mask[i] != 0 ? 1 : 0);
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::BeforeTrain() {
  TREELEARNER_T::BeforeTrain();
  // sync global data sumup info
  std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(), this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
  int size = sizeof(std::tuple<data_size_t, double, double>);
  std::memcpy(input_buffer_.data(), &data, size);

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

  std::memcpy(reinterpret_cast<void*>(&data), output_buffer_.data(), size);

  // set global sumup info
  smaller_leaf_splits_global_->Init(std::get<1>(data), std::get<2>(data));
  larger_leaf_splits_global_->Init();
  // init global data count in leaf
  global_data_count_in_leaf_[0] = std::get<0>(data);
}

template <typename TREELEARNER_T>
bool VotingParallelTreeLearner<TREELEARNER_T>::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  if (TREELEARNER_T::BeforeFindBestSplit(tree, left_leaf, right_leaf)) {
    data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
    data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
    if (right_leaf < 0) {
      return true;
    } else if (num_data_in_left_child < num_data_in_right_child) {
      // get local sumup
      this->smaller_leaf_splits_->Init(left_leaf, this->data_partition_.get(), this->gradients_, this->hessians_);
      this->larger_leaf_splits_->Init(right_leaf, this->data_partition_.get(), this->gradients_, this->hessians_);
    } else {
      // get local sumup
      this->smaller_leaf_splits_->Init(right_leaf, this->data_partition_.get(), this->gradients_, this->hessians_);
      this->larger_leaf_splits_->Init(left_leaf, this->data_partition_.get(), this->gradients_, this->hessians_);
    }
    return true;
  } else {
    return false;
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::GlobalVoting(int leaf_idx, const std::vector<LightSplitInfo>& splits, std::vector<int>* out) {
  out->clear();
  if (leaf_idx < 0) {
    return;
  }
  // get mean number on machines
  score_t mean_num_data = GetGlobalDataCountInLeaf(leaf_idx) / static_cast<score_t>(num_machines_);
  std::vector<LightSplitInfo> feature_best_split(this->train_data_->num_total_features() , LightSplitInfo());
  for (auto & split : splits) {
    int fid = split.feature;
    if (fid < 0) {
      continue;
    }
    // weighted gain
    double gain = split.gain * (split.left_count + split.right_count) / mean_num_data;
    if (gain > feature_best_split[fid].gain) {
      feature_best_split[fid] = split;
      feature_best_split[fid].gain = gain;
    }
  }
  // get top k
  std::vector<LightSplitInfo> top_k_splits;
  ArrayArgs<LightSplitInfo>::MaxK(feature_best_split, top_k_, &top_k_splits);
  std::stable_sort(top_k_splits.begin(), top_k_splits.end(), std::greater<LightSplitInfo>());
  for (auto& split : top_k_splits) {
    if (split.gain == kMinScore || split.feature == -1) {
      continue;
    }
    out->push_back(split.feature);
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::CopyLocalHistogram(const std::vector<int>& smaller_top_features, const std::vector<int>& larger_top_features) {
  for (int i = 0; i < this->num_features_; ++i) {
    smaller_is_feature_aggregated_[i] = false;
    larger_is_feature_aggregated_[i] = false;
  }
  size_t total_num_features = smaller_top_features.size() + larger_top_features.size();
  size_t average_feature = (total_num_features + num_machines_ - 1) / num_machines_;
  size_t used_num_features = 0, smaller_idx = 0, larger_idx = 0;
  block_start_[0] = 0;
  reduce_scatter_size_ = 0;
  // Copy histogram to buffer, and Get local aggregate features
  for (int i = 0; i < num_machines_; ++i) {
    size_t cur_size = 0, cur_used_features = 0;
    size_t cur_total_feature = std::min(average_feature, total_num_features - used_num_features);
    // copy histograms.
    while (cur_used_features < cur_total_feature) {
      // copy smaller leaf histograms first
      if (smaller_idx < smaller_top_features.size()) {
        int inner_feature_index = this->train_data_->InnerFeatureIndex(smaller_top_features[smaller_idx]);
        ++cur_used_features;
        // mark local aggregated feature
        if (i == rank_) {
          smaller_is_feature_aggregated_[inner_feature_index] = true;
          smaller_buffer_read_start_pos_[inner_feature_index] = static_cast<int>(cur_size);
        }
        // copy
        std::memcpy(input_buffer_.data() + reduce_scatter_size_, this->smaller_leaf_histogram_array_[inner_feature_index].RawData(), this->smaller_leaf_histogram_array_[inner_feature_index].SizeOfHistogram());
        cur_size += this->smaller_leaf_histogram_array_[inner_feature_index].SizeOfHistogram();
        reduce_scatter_size_ += this->smaller_leaf_histogram_array_[inner_feature_index].SizeOfHistogram();
        ++smaller_idx;
      }
      if (cur_used_features >= cur_total_feature) {
        break;
      }
      // then copy larger leaf histograms
      if (larger_idx < larger_top_features.size()) {
        int inner_feature_index = this->train_data_->InnerFeatureIndex(larger_top_features[larger_idx]);
        ++cur_used_features;
        // mark local aggregated feature
        if (i == rank_) {
          larger_is_feature_aggregated_[inner_feature_index] = true;
          larger_buffer_read_start_pos_[inner_feature_index] = static_cast<int>(cur_size);
        }
        // copy
        std::memcpy(input_buffer_.data() + reduce_scatter_size_, this->larger_leaf_histogram_array_[inner_feature_index].RawData(), this->larger_leaf_histogram_array_[inner_feature_index].SizeOfHistogram());
        cur_size += this->larger_leaf_histogram_array_[inner_feature_index].SizeOfHistogram();
        reduce_scatter_size_ += this->larger_leaf_histogram_array_[inner_feature_index].SizeOfHistogram();
        ++larger_idx;
      }
    }
    used_num_features += cur_used_features;
    block_len_[i] = static_cast<int>(cur_size);
    if (i < num_machines_ - 1) {
      block_start_[i + 1] = block_start_[i] + block_len_[i];
    }
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::FindBestSplits(const Tree* tree) {
  // use local data to find local best splits
  std::vector<int8_t> is_feature_used(this->num_features_, 0);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    if (!this->col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    // AFS: skip features not selected by adaptive screening
    if (this->config_->afs_enable && !this->afs_selected_features_[feature_index]) continue;
    if (this->parent_leaf_histogram_array_ != nullptr
      && !this->parent_leaf_histogram_array_[feature_index].is_splittable()) {
      this->smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
  }
  bool use_subtract = true;
  if (this->parent_leaf_histogram_array_ == nullptr) {
    use_subtract = false;
  }
  TREELEARNER_T::ConstructHistograms(is_feature_used, use_subtract);

  const int smaller_leaf_index = this->smaller_leaf_splits_->leaf_index();
  const data_size_t local_data_on_smaller_leaf = this->data_partition_->leaf_count(smaller_leaf_index);
  if (local_data_on_smaller_leaf <= 0) {
    // clear histogram buffer before synchronizing
    // otherwise histogram contents from the previous iteration will be sent
    OMP_INIT_EX();
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
      OMP_LOOP_EX_BEGIN();
      if (!is_feature_used[feature_index]) {
        continue;
      }
      const BinMapper* feature_bin_mapper = this->train_data_->FeatureBinMapper(feature_index);
      const int num_bin = feature_bin_mapper->num_bin();
      const int offset = static_cast<int>(feature_bin_mapper->GetMostFreqBin() == 0);
      hist_t* hist_ptr = this->smaller_leaf_histogram_array_[feature_index].RawData();
      std::memset(reinterpret_cast<void*>(hist_ptr), 0, (num_bin - offset) * kHistEntrySize);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }

  if (this->larger_leaf_splits_ != nullptr) {
    const int larger_leaf_index = this->larger_leaf_splits_->leaf_index();
    if (larger_leaf_index >= 0) {
      const data_size_t local_data_on_larger_leaf = this->data_partition_->leaf_count(larger_leaf_index);
      if (local_data_on_larger_leaf <= 0) {
        OMP_INIT_EX();
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
          OMP_LOOP_EX_BEGIN();
          if (!is_feature_used[feature_index]) {
            continue;
          }
          const BinMapper* feature_bin_mapper = this->train_data_->FeatureBinMapper(feature_index);
          const int num_bin = feature_bin_mapper->num_bin();
          const int offset = static_cast<int>(feature_bin_mapper->GetMostFreqBin() == 0);
          hist_t* hist_ptr = this->larger_leaf_histogram_array_[feature_index].RawData();
          std::memset(reinterpret_cast<void*>(hist_ptr), 0, (num_bin - offset) * kHistEntrySize);
          OMP_LOOP_EX_END();
        }
        OMP_THROW_EX();
      }
    }
  }

  std::vector<SplitInfo> smaller_bestsplit_per_features(this->num_features_);
  std::vector<SplitInfo> larger_bestsplit_per_features(this->num_features_);
  double smaller_leaf_parent_output = this->GetParentOutput(tree, this->smaller_leaf_splits_.get());
  double larger_leaf_parent_output = this->GetParentOutput(tree, this->larger_leaf_splits_.get());
  OMP_INIT_EX();
  // find splits
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_used[feature_index]) {
      continue;
    }
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    this->train_data_->FixHistogram(feature_index,
      this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
      this->smaller_leaf_histogram_array_[feature_index].RawData());

    this->ComputeBestSplitForFeature(
        this->smaller_leaf_histogram_array_, feature_index, real_feature_index,
        true, this->smaller_leaf_splits_->num_data_in_leaf(),
        this->smaller_leaf_splits_.get(),
        &smaller_bestsplit_per_features[feature_index],
        smaller_leaf_parent_output);
    // only has root leaf
    if (this->larger_leaf_splits_ == nullptr || this->larger_leaf_splits_->leaf_index() < 0) {
      continue;
    }

    if (use_subtract) {
      this->larger_leaf_histogram_array_[feature_index].Subtract(this->smaller_leaf_histogram_array_[feature_index]);
    } else {
      this->train_data_->FixHistogram(feature_index, this->larger_leaf_splits_->sum_gradients(), this->larger_leaf_splits_->sum_hessians(),
        this->larger_leaf_histogram_array_[feature_index].RawData());
    }
    this->ComputeBestSplitForFeature(
        this->larger_leaf_histogram_array_, feature_index, real_feature_index,
        true, this->larger_leaf_splits_->num_data_in_leaf(),
        this->larger_leaf_splits_.get(),
        &larger_bestsplit_per_features[feature_index],
        larger_leaf_parent_output);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  std::vector<SplitInfo> smaller_top_k_splits, larger_top_k_splits;
  // local voting
  ArrayArgs<SplitInfo>::MaxK(smaller_bestsplit_per_features, top_k_, &smaller_top_k_splits);
  ArrayArgs<SplitInfo>::MaxK(larger_bestsplit_per_features, top_k_, &larger_top_k_splits);

  std::vector<LightSplitInfo> smaller_top_k_light_splits(top_k_);
  std::vector<LightSplitInfo> larger_top_k_light_splits(top_k_);
  for (int i = 0; i < top_k_; ++i) {
    smaller_top_k_light_splits[i].CopyFrom(smaller_top_k_splits[i]);
    larger_top_k_light_splits[i].CopyFrom(larger_top_k_splits[i]);
  }

  // gather
  int offset = 0;
  for (int i = 0; i < top_k_; ++i) {
    std::memcpy(input_buffer_.data() + offset, &smaller_top_k_light_splits[i], sizeof(LightSplitInfo));
    offset += sizeof(LightSplitInfo);
    std::memcpy(input_buffer_.data() + offset, &larger_top_k_light_splits[i], sizeof(LightSplitInfo));
    offset += sizeof(LightSplitInfo);
  }
  Network::Allgather(input_buffer_.data(), offset, output_buffer_.data());
  // get all top-k from all machines
  std::vector<LightSplitInfo> smaller_top_k_splits_global;
  std::vector<LightSplitInfo> larger_top_k_splits_global;
  offset = 0;
  for (int i = 0; i < num_machines_; ++i) {
    for (int j = 0; j < top_k_; ++j) {
      smaller_top_k_splits_global.push_back(LightSplitInfo());
      std::memcpy(&smaller_top_k_splits_global.back(), output_buffer_.data() + offset, sizeof(LightSplitInfo));
      offset += sizeof(LightSplitInfo);
      larger_top_k_splits_global.push_back(LightSplitInfo());
      std::memcpy(&larger_top_k_splits_global.back(), output_buffer_.data() + offset, sizeof(LightSplitInfo));
      offset += sizeof(LightSplitInfo);
    }
  }
  // global voting
  std::vector<int> smaller_top_features, larger_top_features;
  GlobalVoting(this->smaller_leaf_splits_->leaf_index(), smaller_top_k_splits_global, &smaller_top_features);
  GlobalVoting(this->larger_leaf_splits_->leaf_index(), larger_top_k_splits_global, &larger_top_features);
  // copy local histograms to buffer
  CopyLocalHistogram(smaller_top_features, larger_top_features);

  // Reduce scatter for histogram
  Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(hist_t), block_start_.data(), block_len_.data(),
                         output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &HistogramSumReducer);

  this->FindBestSplitsFromHistograms(is_feature_used, false, tree);
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t>&, bool, const Tree* tree) {
  std::vector<SplitInfo> smaller_bests_per_thread(this->share_state_->num_threads);
  std::vector<SplitInfo> larger_bests_per_thread(this->share_state_->num_threads);
  std::vector<int8_t> smaller_node_used_features =
      this->col_sampler_.GetByNode(tree, this->smaller_leaf_splits_->leaf_index());
  std::vector<int8_t> larger_node_used_features =
      this->col_sampler_.GetByNode(tree, this->larger_leaf_splits_->leaf_index());
  double smaller_leaf_parent_output = this->GetParentOutput(tree, this->smaller_leaf_splits_global_.get());
  double larger_leaf_parent_output = this->GetParentOutput(tree, this->larger_leaf_splits_global_.get());
  // find best split from local aggregated histograms
  OMP_INIT_EX();
#pragma omp parallel for schedule(static) num_threads(this->share_state_->num_threads)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    if (smaller_is_feature_aggregated_[feature_index]) {
      // restore from buffer
      smaller_leaf_histogram_array_global_[feature_index].FromMemory(
        output_buffer_.data() + smaller_buffer_read_start_pos_[feature_index]);

      this->train_data_->FixHistogram(feature_index,
                                      smaller_leaf_splits_global_->sum_gradients(), smaller_leaf_splits_global_->sum_hessians(),
                                      smaller_leaf_histogram_array_global_[feature_index].RawData());

      this->ComputeBestSplitForFeature(
          smaller_leaf_histogram_array_global_.get(), feature_index,
          real_feature_index, smaller_node_used_features[feature_index],
          GetGlobalDataCountInLeaf(smaller_leaf_splits_global_->leaf_index()),
          smaller_leaf_splits_global_.get(), &smaller_bests_per_thread[tid],
          smaller_leaf_parent_output);
    }

    if (larger_is_feature_aggregated_[feature_index]) {
      // restore from buffer
      larger_leaf_histogram_array_global_[feature_index].FromMemory(output_buffer_.data() + larger_buffer_read_start_pos_[feature_index]);

      this->train_data_->FixHistogram(feature_index,
                                      larger_leaf_splits_global_->sum_gradients(), larger_leaf_splits_global_->sum_hessians(),
                                      larger_leaf_histogram_array_global_[feature_index].RawData());

      this->ComputeBestSplitForFeature(
          larger_leaf_histogram_array_global_.get(), feature_index,
          real_feature_index,
          larger_node_used_features[feature_index],
          GetGlobalDataCountInLeaf(larger_leaf_splits_global_->leaf_index()),
          larger_leaf_splits_global_.get(), &larger_bests_per_thread[tid],
          larger_leaf_parent_output);
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_bests_per_thread);
  int leaf = this->smaller_leaf_splits_->leaf_index();
  this->best_split_per_leaf_[leaf] = smaller_bests_per_thread[smaller_best_idx];

  if (this->larger_leaf_splits_ != nullptr && this->larger_leaf_splits_->leaf_index() >= 0) {
    leaf = this->larger_leaf_splits_->leaf_index();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_bests_per_thread);
    this->best_split_per_leaf_[leaf] = larger_bests_per_thread[larger_best_idx];
  }

  // find local best
  SplitInfo smaller_best_split, larger_best_split;
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->leaf_index() >= 0) {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()];
  }
  // sync global best info
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);

  // copy back
  this->best_split_per_leaf_[smaller_leaf_splits_global_->leaf_index()] = smaller_best_split;
  if (larger_best_split.feature >= 0 && larger_leaf_splits_global_->leaf_index() >= 0) {
    this->best_split_per_leaf_[larger_leaf_splits_global_->leaf_index()] = larger_best_split;
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  TREELEARNER_T::SplitInner(tree, best_Leaf, left_leaf, right_leaf, false);
  const SplitInfo& best_split_info = this->best_split_per_leaf_[best_Leaf];
  // set the global number of data for leaves
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
  // init the global sumup info
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_global_->Init(*left_leaf, this->data_partition_.get(),
      best_split_info.left_sum_gradient,
      best_split_info.left_sum_hessian,
      best_split_info.left_output);
    larger_leaf_splits_global_->Init(*right_leaf, this->data_partition_.get(),
      best_split_info.right_sum_gradient,
      best_split_info.right_sum_hessian,
      best_split_info.right_output);
  } else {
    smaller_leaf_splits_global_->Init(*right_leaf, this->data_partition_.get(),
      best_split_info.right_sum_gradient,
      best_split_info.right_sum_hessian,
      best_split_info.right_output);
    larger_leaf_splits_global_->Init(*left_leaf, this->data_partition_.get(),
      best_split_info.left_sum_gradient,
      best_split_info.left_sum_hessian,
      best_split_info.left_output);
  }
}

// instantiate template classes, otherwise linker cannot find the code
template class VotingParallelTreeLearner<GPUTreeLearner>;
template class VotingParallelTreeLearner<SerialTreeLearner>;
}  // namespace LightGBM
