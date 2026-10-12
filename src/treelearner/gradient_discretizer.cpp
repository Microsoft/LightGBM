/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "gradient_discretizer.hpp"
#include <LightGBM/network.h>

#include <algorithm>
#include <string>
#include <vector>

namespace LightGBM {

void GradientDiscretizer::Init(
  const data_size_t num_data, const int num_leaves,
  const int num_features, const Dataset* train_data) {
  discretized_gradients_and_hessians_vector_.resize(num_data * 2);
  gradient_random_values_.resize(num_data);
  hessian_random_values_.resize(num_data);
  random_values_use_start_eng_ = std::mt19937(random_seed_);
  random_values_use_start_dist_ = std::uniform_int_distribution<data_size_t>(0, num_data);

  const int num_threads = OMP_NUM_THREADS();
  int num_blocks = 0;
  data_size_t block_size = 0;
  Threading::BlockInfo<data_size_t>(num_data, 512, &num_blocks, &block_size);
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
  for (int thread_id = 0; thread_id < num_blocks; ++thread_id) {
    const data_size_t start = thread_id * block_size;
    const data_size_t end = std::min(start + block_size, num_data);
    std::mt19937 gradient_random_values_eng(random_seed_ + thread_id);
    std::uniform_real_distribution<double> gradient_random_values_dist(0.0f, 1.0f);
    std::mt19937 hessian_random_values_eng(random_seed_ + thread_id + num_threads);
    std::uniform_real_distribution<double> hessian_random_values_dist(0.0f, 1.0f);
    for (data_size_t i = start; i < end; ++i) {
      gradient_random_values_[i] = gradient_random_values_dist(gradient_random_values_eng);
      hessian_random_values_[i] = hessian_random_values_dist(hessian_random_values_eng);
    }
  }

  max_gradient_abs_ = 0.0f;
  max_hessian_abs_ = 0.0f;

  gradient_scale_ = 0.0f;
  hessian_scale_ = 0.0f;
  inverse_gradient_scale_ = 0.0f;
  inverse_hessian_scale_ = 0.0f;

  num_leaves_ = num_leaves;
  leaf_num_bits_in_histogram_bin_.resize(num_leaves_, 0);
  node_num_bits_in_histogram_bin_.resize(num_leaves_, 0);
  global_leaf_num_bits_in_histogram_bin_.resize(num_leaves_, 0);
  global_node_num_bits_in_histogram_bin_.resize(num_leaves_, 0);

  leaf_grad_hess_stats_.resize(num_leaves_ * 2, 0.0);
  change_hist_bits_buffer_.resize(num_features);
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int feature_index = 0; feature_index < num_features; ++feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    change_hist_bits_buffer_[feature_index].resize((bin_mapper->num_bin() - static_cast<int>(bin_mapper->GetMostFreqBin() == 0)) * 2);
  }

  ordered_int_gradients_and_hessians_.resize(2 * num_data);
}

void GradientDiscretizer::DiscretizeGradients(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians) {
  double max_gradient = std::fabs(input_gradients[0]);
  double max_hessian = std::fabs(input_hessians[0]);
  const int num_threads = OMP_NUM_THREADS();
  std::vector<double> thread_max_gradient(num_threads, max_gradient);
  std::vector<double> thread_max_hessian(num_threads, max_hessian);
  Threading::For<data_size_t>(0, num_data, 1024,
    [input_gradients, input_hessians, &thread_max_gradient, &thread_max_hessian]
    (int, data_size_t start, data_size_t end) {
      int thread_id = omp_get_thread_num();
      for (data_size_t i = start; i < end; ++i) {
        double fabs_grad = std::fabs(input_gradients[i]);
        double fabs_hess = std::fabs(input_hessians[i]);
        if (fabs_grad > thread_max_gradient[thread_id]) {
          thread_max_gradient[thread_id] = fabs_grad;
        }
        if (fabs_hess > thread_max_hessian[thread_id]) {
          thread_max_hessian[thread_id] = fabs_hess;
        }
      }});
  max_gradient = thread_max_gradient[0];
  max_hessian = thread_max_hessian[0];
  for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
    if (max_gradient < thread_max_gradient[thread_id]) {
      max_gradient = thread_max_gradient[thread_id];
    }
    if (max_hessian < thread_max_hessian[thread_id]) {
      max_hessian = thread_max_hessian[thread_id];
    }
  }
  if (Network::num_machines() > 1) {
    max_gradient = Network::GlobalSyncUpByMax(max_gradient);
    max_hessian = Network::GlobalSyncUpByMax(max_hessian);
  }
  max_gradient_abs_ = max_gradient;
  max_hessian_abs_ = max_hessian;
  gradient_scale_ = max_gradient_abs_ / static_cast<double>(num_grad_quant_bins_ / 2);
  if (is_constant_hessian_) {
    hessian_scale_ = max_hessian_abs_;
  } else {
    hessian_scale_ = max_hessian_abs_ / static_cast<double>(num_grad_quant_bins_);
  }
  inverse_gradient_scale_ = 1.0f / gradient_scale_;
  inverse_hessian_scale_ = 1.0f / hessian_scale_;

  const int random_values_use_start = random_values_use_start_dist_(random_values_use_start_eng_);
  int8_t* discretized_int8 = discretized_gradients_and_hessians_vector_.data();
  if (stochastic_rounding_) {
    if (is_constant_hessian_) {
      #pragma omp parallel for schedule(static) num_threads(num_threads)
      for (data_size_t i = 0; i < num_data; ++i) {
        const double gradient = input_gradients[i];
        const data_size_t random_value_pos = (i + random_values_use_start) % num_data;
        discretized_int8[2 * i + 1] = gradient >= 0.0f ?
          static_cast<int8_t>(gradient * inverse_gradient_scale_ + gradient_random_values_[random_value_pos]) :
          static_cast<int8_t>(gradient * inverse_gradient_scale_ - gradient_random_values_[random_value_pos]);
        discretized_int8[2 * i] = static_cast<int8_t>(1);
      }
    } else {
      #pragma omp parallel for schedule(static) num_threads(num_threads)
      for (data_size_t i = 0; i < num_data; ++i) {
        const double gradient = input_gradients[i];
        const data_size_t random_value_pos = (i + random_values_use_start) % num_data;
        discretized_int8[2 * i + 1] = gradient >= 0.0f ?
          static_cast<int8_t>(gradient * inverse_gradient_scale_ + gradient_random_values_[random_value_pos]) :
          static_cast<int8_t>(gradient * inverse_gradient_scale_ - gradient_random_values_[random_value_pos]);
        discretized_int8[2 * i] = static_cast<int8_t>(input_hessians[i] * inverse_hessian_scale_ + hessian_random_values_[random_value_pos]);
      }
    }
  } else {
    if (is_constant_hessian_) {
      #pragma omp parallel for schedule(static) num_threads(num_threads)
      for (data_size_t i = 0; i < num_data; ++i) {
        const double gradient = input_gradients[i];
        discretized_int8[2 * i + 1] = gradient >= 0.0f ?
          static_cast<int8_t>(gradient * inverse_gradient_scale_ + 0.5) :
          static_cast<int8_t>(gradient * inverse_gradient_scale_ - 0.5);
        discretized_int8[2 * i] = static_cast<int8_t>(1);
      }
    } else {
      #pragma omp parallel for schedule(static) num_threads(num_threads)
      for (data_size_t i = 0; i < num_data; ++i) {
        const double gradient = input_gradients[i];
        discretized_int8[2 * i + 1] = gradient >= 0.0f ?
          static_cast<int8_t>(gradient * inverse_gradient_scale_ + 0.5) :
          static_cast<int8_t>(gradient * inverse_gradient_scale_ - 0.5);
        discretized_int8[2 * i] = static_cast<int8_t>(input_hessians[i] * inverse_hessian_scale_ + 0.5);
      }
    }
  }
}

template <bool IS_GLOBAL>
void GradientDiscretizer::SetNumBitsInHistogramBin(
  const int left_leaf_index, const int right_leaf_index,
  const data_size_t num_data_in_left_leaf, const data_size_t num_data_in_right_leaf) {
  std::vector<int8_t>& leaf_num_bits_in_histogram_bin = IS_GLOBAL ?
    global_leaf_num_bits_in_histogram_bin_ : leaf_num_bits_in_histogram_bin_;
  std::vector<int8_t>& node_num_bits_in_histogram_bin = IS_GLOBAL ?
    global_node_num_bits_in_histogram_bin_ : node_num_bits_in_histogram_bin_;
  if (right_leaf_index == -1) {
    const uint64_t max_stat_per_bin = static_cast<uint64_t>(num_data_in_left_leaf) * static_cast<uint64_t>(num_grad_quant_bins_);
    if (max_stat_per_bin < 256) {
      leaf_num_bits_in_histogram_bin[left_leaf_index] = 8;
    } else if (max_stat_per_bin < 65536) {
      leaf_num_bits_in_histogram_bin[left_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_bin[left_leaf_index] = 32;
    }
  } else {
    const uint64_t max_stat_left_per_bin = static_cast<uint64_t>(num_data_in_left_leaf) * static_cast<uint64_t>(num_grad_quant_bins_);
    const uint64_t max_stat_right_per_bin = static_cast<uint64_t>(num_data_in_right_leaf) * static_cast<uint64_t>(num_grad_quant_bins_);
    node_num_bits_in_histogram_bin[left_leaf_index] = leaf_num_bits_in_histogram_bin[left_leaf_index];
    if (max_stat_left_per_bin < 256) {
      leaf_num_bits_in_histogram_bin[left_leaf_index] = 8;
    } else if (max_stat_left_per_bin < 65536) {
      leaf_num_bits_in_histogram_bin[left_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_bin[left_leaf_index] = 32;
    }
    if (max_stat_right_per_bin < 256) {
      leaf_num_bits_in_histogram_bin[right_leaf_index] = 8;
    } else if (max_stat_right_per_bin < 65536) {
      leaf_num_bits_in_histogram_bin[right_leaf_index] = 16;
    } else {
      leaf_num_bits_in_histogram_bin[right_leaf_index] = 32;
    }
  }
}

template void GradientDiscretizer::SetNumBitsInHistogramBin<false>(
  const int left_leaf_index, const int right_leaf_index,
  const data_size_t num_data_in_left_leaf, const data_size_t num_data_in_right_leaf);

template void GradientDiscretizer::SetNumBitsInHistogramBin<true>(
  const int left_leaf_index, const int right_leaf_index,
  const data_size_t num_data_in_left_leaf, const data_size_t num_data_in_right_leaf);

void GradientDiscretizer::RenewIntGradTreeOutput(
  Tree* tree, const Config* config, const DataPartition* data_partition,
  const score_t* gradients, const score_t* hessians,
  const std::function<data_size_t(int)>& leaf_index_to_global_num_data) {
  global_timer.Start("GradientDiscretizer::RenewIntGradTreeOutput");
  if (config->tree_learner == std::string("data")) {
    for (int leaf_id = 0; leaf_id < tree->num_leaves(); ++leaf_id) {
      data_size_t leaf_cnt = 0;
      const data_size_t* data_indices = data_partition->GetIndexOnLeaf(leaf_id, &leaf_cnt);
      double sum_gradient = 0.0f, sum_hessian = 0.0f;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_gradient, sum_hessian)
      for (data_size_t i = 0; i < leaf_cnt; ++i) {
        const data_size_t index = data_indices[i];
        const score_t grad = gradients[index];
        const score_t hess = hessians[index];
        sum_gradient += grad;
        sum_hessian += hess;
      }
      leaf_grad_hess_stats_[2 * leaf_id] = sum_gradient;
      leaf_grad_hess_stats_[2 * leaf_id + 1] = sum_hessian;
    }
    std::vector<double> global_leaf_grad_hess_stats = Network::GlobalSum<double>(&leaf_grad_hess_stats_);
    for (int leaf_id = 0; leaf_id < tree->num_leaves(); ++leaf_id) {
      const double sum_gradient = global_leaf_grad_hess_stats[2 * leaf_id];
      const double sum_hessian = global_leaf_grad_hess_stats[2 * leaf_id + 1];
      const double leaf_output = FeatureHistogram::CalculateSplittedLeafOutput<true, true, false>(
        sum_gradient, sum_hessian,
        config->lambda_l1, config->lambda_l2, config->max_delta_step, config->path_smooth,
        leaf_index_to_global_num_data(leaf_id), 0.0f);
      tree->SetLeafOutput(leaf_id, leaf_output);
    }
  } else {
    for (int leaf_id = 0; leaf_id < tree->num_leaves(); ++leaf_id) {
      data_size_t leaf_cnt = 0;
      const data_size_t* data_indices = data_partition->GetIndexOnLeaf(leaf_id, &leaf_cnt);
      double sum_gradient = 0.0f, sum_hessian = 0.0f;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_gradient, sum_hessian)
      for (data_size_t i = 0; i < leaf_cnt; ++i) {
        const data_size_t index = data_indices[i];
        const score_t grad = gradients[index];
        const score_t hess = hessians[index];
        sum_gradient += grad;
        sum_hessian += hess;
      }
      const double leaf_output = FeatureHistogram::CalculateSplittedLeafOutput<true, true, false>(sum_gradient, sum_hessian,
        config->lambda_l1, config->lambda_l2, config->max_delta_step, config->path_smooth,
        leaf_cnt, 0.0f);
      tree->SetLeafOutput(leaf_id, leaf_output);
    }
  }
  global_timer.Stop("GradientDiscretizer::RenewIntGradTreeOutput");
}

}  // namespace LightGBM
