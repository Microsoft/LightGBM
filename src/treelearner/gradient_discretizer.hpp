/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_GRADIENT_DISCRETIZER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_GRADIENT_DISCRETIZER_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/tree.h>
#include <LightGBM/utils/threading.h>

#include <random>
#include <vector>

#include "data_partition.hpp"
#include "feature_histogram.hpp"

namespace LightGBM {

class GradientDiscretizer {
 public:
  GradientDiscretizer(int num_grad_quant_bins, int num_trees, int random_seed, bool is_constant_hessian, const bool stochastic_rounding) {
    num_grad_quant_bins_ = num_grad_quant_bins;
    iter_ = 0;
    num_trees_ = num_trees;
    random_seed_ = random_seed;
    is_constant_hessian_ = is_constant_hessian;
    stochastic_rounding_ = stochastic_rounding;
  }

  virtual ~GradientDiscretizer() {}

  virtual void DiscretizeGradients(
    const data_size_t num_data,
    const score_t* input_gradients,
    const score_t* input_hessians);

  virtual const int8_t* discretized_gradients_and_hessians() const {
    return discretized_gradients_and_hessians_vector_.data();
  }

  virtual double grad_scale() const {
    return gradient_scale_;
  }

  virtual double hess_scale() const {
    return hessian_scale_;
  }

  virtual void Init(
    const data_size_t num_data, const int num_leaves,
    const int num_features, const Dataset* train_data);

  template <bool IS_GLOBAL>
  void SetNumBitsInHistogramBin(
    const int left_leaf_index, const int right_leaf_index,
    const data_size_t num_data_in_left_leaf, const data_size_t num_data_in_right_leaf);

  template <bool IS_GLOBAL>
  int8_t GetHistBitsInLeaf(const int leaf_index) {
    if (IS_GLOBAL) {
      return global_leaf_num_bits_in_histogram_bin_[leaf_index];
    } else {
      return leaf_num_bits_in_histogram_bin_[leaf_index];
    }
  }

  template <bool IS_GLOBAL>
  int8_t GetHistBitsInNode(const int node_index) {
    if (IS_GLOBAL) {
      return global_node_num_bits_in_histogram_bin_[node_index];
    } else {
      return node_num_bits_in_histogram_bin_[node_index];
    }
  }

  int8_t* ordered_int_gradients_and_hessians() {
    return ordered_int_gradients_and_hessians_.data();
  }

  void RenewIntGradTreeOutput(
    Tree* tree, const Config* config, const DataPartition* data_partition,
    const score_t* gradients, const score_t* hessians,
    const std::function<data_size_t(int)>& leaf_index_to_global_num_data);

  int32_t* GetChangeHistBitsBuffer(const int feature_index) {
    return change_hist_bits_buffer_[feature_index].data();
  }

 protected:
  int num_grad_quant_bins_;
  int iter_;
  int num_trees_;
  int random_seed_;
  bool stochastic_rounding_;

  std::vector<double> gradient_random_values_;
  std::vector<double> hessian_random_values_;
  std::mt19937 random_values_use_start_eng_;
  std::uniform_int_distribution<data_size_t> random_values_use_start_dist_;
  std::vector<int8_t> discretized_gradients_and_hessians_vector_;
  std::vector<int8_t> ordered_int_gradients_and_hessians_;

  double max_gradient_abs_;
  double max_hessian_abs_;

  double gradient_scale_;
  double hessian_scale_;
  double inverse_gradient_scale_;
  double inverse_hessian_scale_;

  bool is_constant_hessian_;
  int num_leaves_;

  std::vector<int8_t> leaf_num_bits_in_histogram_bin_;
  std::vector<int8_t> node_num_bits_in_histogram_bin_;
  std::vector<int8_t> global_leaf_num_bits_in_histogram_bin_;
  std::vector<int8_t> global_node_num_bits_in_histogram_bin_;

  std::vector<double> leaf_grad_hess_stats_;
  std::vector<std::vector<int32_t>> change_hist_bits_buffer_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_GRADIENT_DISCRETIZER_HPP_
