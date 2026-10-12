/*!
 * Copyright (c) 2024-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2024-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "feature_histogram.hpp"

#include <algorithm>
#include <vector>

namespace LightGBM {

void FeatureHistogram::FuncForCategorical() {
  if (meta_->config->extra_trees) {
    if (meta_->config->monotone_constraints.empty()) {
      FuncForCategoricalL1<true, false>();
    } else {
      FuncForCategoricalL1<true, true>();
    }
  } else {
    if (meta_->config->monotone_constraints.empty()) {
      FuncForCategoricalL1<false, false>();
    } else {
      FuncForCategoricalL1<false, true>();
    }
  }
}

template <bool USE_RAND, bool USE_MC>
void FeatureHistogram::FuncForCategoricalL1() {
  if (meta_->config->path_smooth > kEpsilon) {
    FuncForCategoricalL2<USE_RAND, USE_MC, true>();
  } else {
    FuncForCategoricalL2<USE_RAND, USE_MC, false>();
  }
}

template <bool USE_RAND, bool USE_MC, bool USE_SMOOTHING>
void FeatureHistogram::FuncForCategoricalL2() {
  if (meta_->config->use_quantized_grad) {
#define LAMBDA_PARAMS_INT \
    int64_t int_sum_gradient_and_hessian, \
    const double grad_scale, const double hess_scale, \
    const uint8_t hist_bits_bin, const uint8_t hist_bits_acc, \
    data_size_t num_data, \
    const FeatureConstraint* constraints, \
    double parent_output, \
    SplitInfo* output

#define ARGUMENTS_INT \
    int_sum_gradient_and_hessian, grad_scale, hess_scale, num_data, constraints, parent_output, output

    if (meta_->config->lambda_l1 > 0) {
      if (meta_->config->max_delta_step > 0) {
        int_find_best_threshold_fun_ = [=] (LAMBDA_PARAMS_INT) {
          if (hist_bits_acc <= 16) {
            CHECK_LE(hist_bits_bin, 16);
            FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, true, true, USE_SMOOTHING, int32_t, int32_t, int16_t, int16_t, 16, 16>(ARGUMENTS_INT);
          } else {
            if (hist_bits_bin <= 16) {
              FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, true, true, USE_SMOOTHING, int32_t, int64_t, int16_t, int32_t, 16, 32>(ARGUMENTS_INT);
            } else {
              FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, true, true, USE_SMOOTHING, int64_t, int64_t, int32_t, int32_t, 32, 32>(ARGUMENTS_INT);
            }
          }
        };
      } else {
        int_find_best_threshold_fun_ = [=] (LAMBDA_PARAMS_INT) {
          if (hist_bits_acc <= 16) {
            CHECK_LE(hist_bits_bin, 16);
            FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, true, false, USE_SMOOTHING, int32_t, int32_t, int16_t, int16_t, 16, 16>(ARGUMENTS_INT);
          } else {
            if (hist_bits_bin <= 16) {
              FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, true, false, USE_SMOOTHING, int32_t, int64_t, int16_t, int32_t, 16, 32>(ARGUMENTS_INT);
            } else {
              FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, true, false, USE_SMOOTHING, int64_t, int64_t, int32_t, int32_t, 32, 32>(ARGUMENTS_INT);
            }
          }
        };
      }
    } else {
      if (meta_->config->max_delta_step > 0) {
        int_find_best_threshold_fun_ = [=] (LAMBDA_PARAMS_INT) {
          if (hist_bits_acc <= 16) {
            CHECK_LE(hist_bits_bin, 16);
            FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, false, true, USE_SMOOTHING, int32_t, int32_t, int16_t, int16_t, 16, 16>(ARGUMENTS_INT);
          } else {
            if (hist_bits_bin <= 16) {
              FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, false, true, USE_SMOOTHING, int32_t, int64_t, int16_t, int32_t, 16, 32>(ARGUMENTS_INT);
            } else {
              FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, false, true, USE_SMOOTHING, int64_t, int64_t, int32_t, int32_t, 32, 32>(ARGUMENTS_INT);
            }
          }
        };
      } else {
        int_find_best_threshold_fun_ = [=] (LAMBDA_PARAMS_INT) {
          if (hist_bits_acc <= 16) {
            CHECK_LE(hist_bits_bin, 16);
            FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, false, false, USE_SMOOTHING, int32_t, int32_t, int16_t, int16_t, 16, 16>(ARGUMENTS_INT);
          } else {
            if (hist_bits_bin <= 16) {
              FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, false, false, USE_SMOOTHING, int32_t, int64_t, int16_t, int32_t, 16, 32>(ARGUMENTS_INT);
            } else {
              FindBestThresholdCategoricalIntInner<USE_RAND, USE_MC, false, false, USE_SMOOTHING, int64_t, int64_t, int32_t, int32_t, 32, 32>(ARGUMENTS_INT);
            }
          }
        };
      }
    }
#undef LAMBDA_ARGUMENTS_INT
#undef ARGUMENTS_INT
  } else {
#define ARGUMENTS                                                      \
  std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, \
      std::placeholders::_4, std::placeholders::_5, std::placeholders::_6
    if (meta_->config->lambda_l1 > 0) {
      if (meta_->config->max_delta_step > 0) {
        find_best_threshold_fun_ =
            std::bind(&FeatureHistogram::FindBestThresholdCategoricalInner<
                          USE_RAND, USE_MC, true, true, USE_SMOOTHING>,
                      this, ARGUMENTS);
      } else {
        find_best_threshold_fun_ =
            std::bind(&FeatureHistogram::FindBestThresholdCategoricalInner<
                          USE_RAND, USE_MC, true, false, USE_SMOOTHING>,
                      this, ARGUMENTS);
      }
    } else {
      if (meta_->config->max_delta_step > 0) {
        find_best_threshold_fun_ =
            std::bind(&FeatureHistogram::FindBestThresholdCategoricalInner<
                          USE_RAND, USE_MC, false, true, USE_SMOOTHING>,
                      this, ARGUMENTS);
      } else {
        find_best_threshold_fun_ =
            std::bind(&FeatureHistogram::FindBestThresholdCategoricalInner<
                          USE_RAND, USE_MC, false, false, USE_SMOOTHING>,
                      this, ARGUMENTS);
      }
    }
#undef ARGUMENTS
  }
}

template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
void FeatureHistogram::FindBestThresholdCategoricalInner(double sum_gradient,
                                        double sum_hessian,
                                        data_size_t num_data,
                                        const FeatureConstraint* constraints,
                                        double parent_output,
                                        SplitInfo* output) {
  is_splittable_ = false;
  output->default_left = false;
  double best_gain = kMinScore;
  data_size_t best_left_count = 0;
  double best_sum_left_gradient = 0;
  double best_sum_left_hessian = 0;
  double gain_shift;
  if (USE_MC) {
    constraints->InitCumulativeConstraints(true);
  }
  if (USE_SMOOTHING) {
    gain_shift = GetLeafGainGivenOutput<USE_L1>(
        sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2, parent_output);
  } else {
    // Need special case for no smoothing to preserve existing behaviour. If no smoothing, the parent output is calculated
    // with the larger categorical l2, whereas min_split_gain uses the original l2.
    gain_shift = GetLeafGain<USE_L1, USE_MAX_OUTPUT, false>(sum_gradient, sum_hessian,
        meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step, 0,
        num_data, 0);
  }

  double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
  const int8_t offset = meta_->offset;
  const int bin_start = 1 - offset;
  const int bin_end = meta_->num_bin - offset;
  int used_bin = -1;

  std::vector<int> sorted_idx;
  double l2 = meta_->config->lambda_l2;
  bool use_onehot = meta_->num_bin <= meta_->config->max_cat_to_onehot;
  int best_threshold = -1;
  int best_dir = 1;
  const double cnt_factor = num_data / sum_hessian;
  int rand_threshold = 0;
  if (use_onehot) {
    if (USE_RAND) {
      if (bin_end - bin_start > 0) {
        rand_threshold = meta_->rand.NextInt(bin_start, bin_end);
      }
    }
    for (int t = bin_start; t < bin_end; ++t) {
      const auto grad = GET_GRAD(data_, t);
      const auto hess = GET_HESS(data_, t);
      data_size_t cnt =
          static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
      // if data not enough, or sum hessian too small
      if (cnt < meta_->config->min_data_in_leaf ||
          hess < meta_->config->min_sum_hessian_in_leaf) {
        continue;
      }
      data_size_t other_count = num_data - cnt;
      // if data not enough
      if (other_count < meta_->config->min_data_in_leaf) {
        continue;
      }

      double sum_other_hessian = sum_hessian - hess - kEpsilon;
      // if sum hessian too small
      if (sum_other_hessian < meta_->config->min_sum_hessian_in_leaf) {
        continue;
      }

      double sum_other_gradient = sum_gradient - grad;
      if (USE_RAND) {
        if (t != rand_threshold) {
          continue;
        }
      }
      // current split gain
      double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
          sum_other_gradient, sum_other_hessian, grad, hess + kEpsilon,
          meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
          constraints, 0, meta_->config->path_smooth, other_count, cnt, parent_output);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) {
        continue;
      }

      // mark as able to be split
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
        best_threshold = t;
        best_sum_left_gradient = grad;
        best_sum_left_hessian = hess + kEpsilon;
        best_left_count = cnt;
        best_gain = current_gain;
      }
    }
  } else {
    for (int i = bin_start; i < bin_end; ++i) {
      if (Common::RoundInt(GET_HESS(data_, i) * cnt_factor) >=
          meta_->config->cat_smooth) {
        sorted_idx.push_back(i);
      }
    }
    used_bin = static_cast<int>(sorted_idx.size());

    l2 += meta_->config->cat_l2;

    auto ctr_fun = [this](double sum_grad, double sum_hess) {
      return (sum_grad) / (sum_hess + meta_->config->cat_smooth);
    };
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(), [this, &ctr_fun](int i, int j) {
          return ctr_fun(GET_GRAD(data_, i), GET_HESS(data_, i)) <
                  ctr_fun(GET_GRAD(data_, j), GET_HESS(data_, j));
        });

    std::vector<int> find_direction(1, 1);
    std::vector<int> start_position(1, 0);
    find_direction.push_back(-1);
    start_position.push_back(used_bin - 1);
    const int max_num_cat =
        std::min(meta_->config->max_cat_threshold, (used_bin + 1) / 2);
    int max_threshold = std::max(std::min(max_num_cat, used_bin) - 1, 0);
    if (USE_RAND) {
      if (max_threshold > 0) {
        rand_threshold = meta_->rand.NextInt(0, max_threshold);
      }
    }

    is_splittable_ = false;
    for (size_t out_i = 0; out_i < find_direction.size(); ++out_i) {
      auto dir = find_direction[out_i];
      auto start_pos = start_position[out_i];
      data_size_t min_data_per_group = meta_->config->min_data_per_group;
      data_size_t cnt_cur_group = 0;
      double sum_left_gradient = 0.0f;
      double sum_left_hessian = kEpsilon;
      data_size_t left_count = 0;
      for (int i = 0; i < used_bin && i < max_num_cat; ++i) {
        auto t = sorted_idx[start_pos];
        start_pos += dir;
        const auto grad = GET_GRAD(data_, t);
        const auto hess = GET_HESS(data_, t);
        data_size_t cnt =
            static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));

        sum_left_gradient += grad;
        sum_left_hessian += hess;
        left_count += cnt;
        cnt_cur_group += cnt;

        if (left_count < meta_->config->min_data_in_leaf ||
            sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) {
          continue;
        }
        data_size_t right_count = num_data - left_count;
        if (right_count < meta_->config->min_data_in_leaf ||
            right_count < min_data_per_group) {
          break;
        }

        double sum_right_hessian = sum_hessian - sum_left_hessian;
        if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) {
          break;
        }

        if (cnt_cur_group < min_data_per_group) {
          continue;
        }

        cnt_cur_group = 0;

        double sum_right_gradient = sum_gradient - sum_left_gradient;
        if (USE_RAND) {
          if (i != rand_threshold) {
            continue;
          }
        }
        double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, meta_->config->lambda_l1, l2,
            meta_->config->max_delta_step, constraints, 0, meta_->config->path_smooth,
            left_count, right_count, parent_output);
        if (current_gain <= min_gain_shift) {
          continue;
        }
        is_splittable_ = true;
        if (current_gain > best_gain) {
          best_left_count = left_count;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_threshold = i;
          best_gain = current_gain;
          best_dir = dir;
        }
      }
    }
  }

  if (is_splittable_) {
    output->left_output = CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        best_sum_left_gradient, best_sum_left_hessian,
        meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
        constraints->LeftToBasicConstraint(), meta_->config->path_smooth, best_left_count, parent_output);
    output->left_count = best_left_count;
    output->left_sum_gradient = best_sum_left_gradient;
    output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
    output->right_output = CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        sum_gradient - best_sum_left_gradient,
        sum_hessian - best_sum_left_hessian, meta_->config->lambda_l1, l2,
        meta_->config->max_delta_step, constraints->RightToBasicConstraint(), meta_->config->path_smooth,
        num_data - best_left_count, parent_output);
    output->right_count = num_data - best_left_count;
    output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
    output->right_sum_hessian =
        sum_hessian - best_sum_left_hessian - kEpsilon;
    output->gain = best_gain - min_gain_shift;
    if (use_onehot) {
      output->num_cat_threshold = 1;
      output->cat_threshold =
          std::vector<uint32_t>(1, static_cast<uint32_t>(best_threshold + offset));
    } else {
      output->num_cat_threshold = best_threshold + 1;
      output->cat_threshold =
          std::vector<uint32_t>(output->num_cat_threshold);
      if (best_dir == 1) {
        for (int i = 0; i < output->num_cat_threshold; ++i) {
          auto t = sorted_idx[i] + offset;
          output->cat_threshold[i] = t;
        }
      } else {
        for (int i = 0; i < output->num_cat_threshold; ++i) {
          auto t = sorted_idx[used_bin - 1 - i] + offset;
          output->cat_threshold[i] = t;
        }
      }
    }
    output->monotone_type = 0;
  }
}

template <
bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING,
typename PACKED_HIST_BIN_T, typename PACKED_HIST_ACC_T, typename HIST_BIN_T, typename HIST_ACC_T,
int HIST_BITS_BIN, int HIST_BITS_ACC
>
void FeatureHistogram::FindBestThresholdCategoricalIntInner(int64_t int_sum_gradient_and_hessian,
                                          const double grad_scale, const double hess_scale,
                                          data_size_t num_data,
                                          const FeatureConstraint* constraints,
                                          double parent_output,
                                          SplitInfo* output) {
  is_splittable_ = false;
  output->default_left = false;
  double best_gain = kMinScore;
  PACKED_HIST_ACC_T best_sum_left_gradient_and_hessian = 0;
  double gain_shift;
  if (USE_MC) {
    constraints->InitCumulativeConstraints(true);
  }

  PACKED_HIST_ACC_T local_int_sum_gradient_and_hessian =
    HIST_BITS_ACC == 16 ?
    ((static_cast<int32_t>(int_sum_gradient_and_hessian >> 32) << 16) | static_cast<int32_t>(int_sum_gradient_and_hessian & 0x0000ffff)) :
    static_cast<PACKED_HIST_ACC_T>(int_sum_gradient_and_hessian);

  // recover sum of gradient and hessian from the sum of quantized gradient and hessian
  double sum_gradient = static_cast<double>(static_cast<int32_t>(int_sum_gradient_and_hessian >> 32)) * grad_scale;
  double sum_hessian = static_cast<double>(static_cast<uint32_t>(int_sum_gradient_and_hessian & 0x00000000ffffffff)) * hess_scale;
  if (USE_SMOOTHING) {
    gain_shift = GetLeafGainGivenOutput<USE_L1>(
        sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2, parent_output);
  } else {
    // Need special case for no smoothing to preserve existing behaviour. If no smoothing, the parent output is calculated
    // with the larger categorical l2, whereas min_split_gain uses the original l2.
    gain_shift = GetLeafGain<USE_L1, USE_MAX_OUTPUT, false>(sum_gradient, sum_hessian,
        meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step, 0,
        num_data, 0);
  }

  double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
  const int8_t offset = meta_->offset;
  const int bin_start = 1 - offset;
  const int bin_end = meta_->num_bin - offset;
  int used_bin = -1;

  std::vector<int> sorted_idx;
  double l2 = meta_->config->lambda_l2;
  bool use_onehot = meta_->num_bin <= meta_->config->max_cat_to_onehot;
  int best_threshold = -1;
  int best_dir = 1;
  const double cnt_factor = static_cast<double>(num_data) /
    static_cast<double>(static_cast<uint32_t>(int_sum_gradient_and_hessian & 0x00000000ffffffff));
  int rand_threshold = 0;

  const PACKED_HIST_BIN_T* data_ptr = nullptr;
  if (HIST_BITS_BIN == 16) {
    data_ptr = reinterpret_cast<const PACKED_HIST_BIN_T*>(data_int16_);
  } else {
    data_ptr = reinterpret_cast<const PACKED_HIST_BIN_T*>(data_);
  }

  if (use_onehot) {
    if (USE_RAND) {
      if (bin_end - bin_start > 0) {
        rand_threshold = meta_->rand.NextInt(bin_start, bin_end);
      }
    }
    for (int t = bin_start; t < bin_end; ++t) {
      const PACKED_HIST_BIN_T grad_and_hess = data_ptr[t];
      const uint32_t int_hess = HIST_BITS_BIN == 16 ?
        static_cast<uint32_t>(grad_and_hess & 0x0000ffff) :
        static_cast<uint32_t>(grad_and_hess & 0x00000000ffffffff);
      data_size_t cnt =
          static_cast<data_size_t>(Common::RoundInt(int_hess * cnt_factor));
      const double hess = int_hess * hess_scale;
      // if data not enough, or sum hessian too small
      if (cnt < meta_->config->min_data_in_leaf ||
          hess < meta_->config->min_sum_hessian_in_leaf) {
        continue;
      }
      data_size_t other_count = num_data - cnt;
      // if data not enough
      if (other_count < meta_->config->min_data_in_leaf) {
        continue;
      }

      const PACKED_HIST_ACC_T grad_and_hess_acc = HIST_BITS_ACC != HIST_BITS_BIN ?
        ((static_cast<PACKED_HIST_ACC_T>(static_cast<HIST_BIN_T>(grad_and_hess >> HIST_BITS_BIN)) << HIST_BITS_ACC) |
        (static_cast<PACKED_HIST_ACC_T>(grad_and_hess & 0x0000ffff))) :
        grad_and_hess;
      const PACKED_HIST_ACC_T sum_other_grad_and_hess = local_int_sum_gradient_and_hessian - grad_and_hess_acc;
      const uint32_t sum_other_hess_int = HIST_BITS_ACC == 16 ?
        static_cast<uint32_t>(sum_other_grad_and_hess & 0x0000ffff) :
        static_cast<uint32_t>(sum_other_grad_and_hess & 0x00000000ffffffff);
      double sum_other_hessian = sum_other_hess_int * hess_scale;
      // if sum hessian too small
      if (sum_other_hessian < meta_->config->min_sum_hessian_in_leaf) {
        continue;
      }

      const int32_t int_grad = HIST_BITS_ACC == 16 ?
        static_cast<int32_t>(static_cast<int16_t>(grad_and_hess_acc >> 16)) :
        static_cast<int32_t>(static_cast<int64_t>(grad_and_hess_acc) >> 32);
      const double grad = int_grad * grad_scale;

      const int32_t sum_other_grad_int = HIST_BITS_ACC == 16 ?
        static_cast<int32_t>(static_cast<int16_t>(sum_other_grad_and_hess >> 16)) :
        static_cast<int32_t>(static_cast<int64_t>(sum_other_grad_and_hess) >> 32);
      const double sum_other_gradient = sum_other_grad_int * grad_scale;

      if (USE_RAND) {
        if (t != rand_threshold) {
          continue;
        }
      }
      // current split gain
      double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
          sum_other_gradient, sum_other_hessian, grad, hess,
          meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
          constraints, 0, meta_->config->path_smooth, other_count, cnt, parent_output);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) {
        continue;
      }

      // mark as able to be split
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
        best_threshold = t;
        best_sum_left_gradient_and_hessian = grad_and_hess_acc;
        best_gain = current_gain;
      }
    }
  } else {
    for (int i = bin_start; i < bin_end; ++i) {
      const PACKED_HIST_BIN_T int_grad_and_hess = data_ptr[i];
      const uint32_t int_hess = HIST_BITS_BIN == 16 ?
        static_cast<uint32_t>(int_grad_and_hess & 0x0000ffff) :
        static_cast<uint32_t>(int_grad_and_hess & 0x00000000ffffffff);
      const int cnt = Common::RoundInt(int_hess * cnt_factor);
      if (cnt >= meta_->config->cat_smooth) {
        sorted_idx.push_back(i);
      }
    }
    used_bin = static_cast<int>(sorted_idx.size());

    l2 += meta_->config->cat_l2;

    auto ctr_fun = [this](double sum_grad, double sum_hess) {
      return (sum_grad) / (sum_hess + meta_->config->cat_smooth);
    };
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(), [data_ptr, &ctr_fun, grad_scale, hess_scale](int i, int j) {
          const PACKED_HIST_BIN_T int_grad_and_hess_i = data_ptr[i];
          const PACKED_HIST_BIN_T int_grad_and_hess_j = data_ptr[j];
          const int32_t int_grad_i = HIST_BITS_BIN == 16 ?
            static_cast<int32_t>(static_cast<int16_t>(int_grad_and_hess_i >> 16)) :
            static_cast<int32_t>(static_cast<int64_t>(int_grad_and_hess_i) >> 32);
          const uint32_t int_hess_i = HIST_BITS_BIN == 16 ?
            static_cast<int32_t>(int_grad_and_hess_i & 0x0000ffff) :
            static_cast<int32_t>(int_grad_and_hess_i & 0x00000000ffffffff);
          const int32_t int_grad_j = HIST_BITS_BIN == 16 ?
            static_cast<int32_t>(static_cast<int16_t>(int_grad_and_hess_j >> 16)) :
            static_cast<int32_t>(static_cast<int64_t>(int_grad_and_hess_j) >> 32);
          const uint32_t int_hess_j = HIST_BITS_BIN == 16 ?
            static_cast<int32_t>(int_grad_and_hess_j & 0x0000ffff) :
            static_cast<int32_t>(int_grad_and_hess_j & 0x00000000ffffffff);

          const double grad_i = int_grad_i * grad_scale;
          const double hess_i = int_hess_i * hess_scale;
          const double grad_j = int_grad_j * grad_scale;
          const double hess_j = int_hess_j * hess_scale;

          return ctr_fun(grad_i, hess_i) < ctr_fun(grad_j, hess_j);
        });

    std::vector<int> find_direction(1, 1);
    std::vector<int> start_position(1, 0);
    find_direction.push_back(-1);
    start_position.push_back(used_bin - 1);
    const int max_num_cat =
        std::min(meta_->config->max_cat_threshold, (used_bin + 1) / 2);
    int max_threshold = std::max(std::min(max_num_cat, used_bin) - 1, 0);
    if (USE_RAND) {
      if (max_threshold > 0) {
        rand_threshold = meta_->rand.NextInt(0, max_threshold);
      }
    }

    is_splittable_ = false;
    for (size_t out_i = 0; out_i < find_direction.size(); ++out_i) {
      auto dir = find_direction[out_i];
      auto start_pos = start_position[out_i];
      data_size_t min_data_per_group = meta_->config->min_data_per_group;
      data_size_t cnt_cur_group = 0;
      PACKED_HIST_ACC_T int_sum_left_gradient_and_hessian = 0;
      data_size_t left_count = 0;
      for (int i = 0; i < used_bin && i < max_num_cat; ++i) {
        auto t = sorted_idx[start_pos];
        start_pos += dir;
        PACKED_HIST_BIN_T int_grad_and_hess = data_ptr[t];

        uint32_t int_hess = HIST_BITS_BIN == 16 ?
          static_cast<uint32_t>(int_grad_and_hess & 0x0000ffff) :
          static_cast<uint32_t>(int_grad_and_hess & 0x00000000ffffffff);
        data_size_t cnt =
            static_cast<data_size_t>(Common::RoundInt(int_hess * cnt_factor));

        if (HIST_BITS_ACC != HIST_BITS_BIN) {
          PACKED_HIST_ACC_T int_grad_and_hess_acc =
            (static_cast<PACKED_HIST_ACC_T>(static_cast<int64_t>(int_grad_and_hess & 0xffff0000)) << 32) |
            (static_cast<PACKED_HIST_ACC_T>(int_grad_and_hess & 0x0000ffff));
          int_sum_left_gradient_and_hessian += int_grad_and_hess_acc;
        } else {
          int_sum_left_gradient_and_hessian += int_grad_and_hess;
        }

        left_count += cnt;
        cnt_cur_group += cnt;

        const uint32_t int_left_sum_hessian = HIST_BITS_ACC == 16 ?
          static_cast<uint32_t>(int_sum_left_gradient_and_hessian & 0x0000ffff) :
          static_cast<uint32_t>(int_sum_left_gradient_and_hessian & 0x00000000ffffffff);
        const double sum_left_hessian = int_left_sum_hessian * hess_scale;

        if (left_count < meta_->config->min_data_in_leaf ||
            sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) {
          continue;
        }
        data_size_t right_count = num_data - left_count;
        if (right_count < meta_->config->min_data_in_leaf ||
            right_count < min_data_per_group) {
          break;
        }

        const PACKED_HIST_ACC_T int_sum_right_gradient_and_hessian = local_int_sum_gradient_and_hessian - int_sum_left_gradient_and_hessian;
        const uint32_t int_right_sum_hessian = HIST_BITS_ACC == 16 ?
          static_cast<uint32_t>(int_sum_right_gradient_and_hessian & 0x0000ffff) :
          static_cast<uint32_t>(int_sum_right_gradient_and_hessian & 0x00000000ffffffff);
        const double sum_right_hessian = int_right_sum_hessian * hess_scale;

        if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) {
          break;
        }

        if (cnt_cur_group < min_data_per_group) {
          continue;
        }

        cnt_cur_group = 0;

        const int32_t int_sum_left_gradient = HIST_BITS_ACC == 16 ?
          static_cast<int32_t>(static_cast<int16_t>(int_sum_left_gradient_and_hessian >> 16)) :
          static_cast<int32_t>(static_cast<int64_t>(int_sum_left_gradient_and_hessian) >> 32);
        const double sum_left_gradient = int_sum_left_gradient * grad_scale;

        const int32_t int_sum_right_gradient = HIST_BITS_ACC == 16 ?
          static_cast<int32_t>(static_cast<int16_t>(int_sum_right_gradient_and_hessian >> 16)) :
          static_cast<int32_t>(static_cast<int64_t>(int_sum_right_gradient_and_hessian) >> 32);
        const double sum_right_gradient = int_sum_right_gradient * grad_scale;

        if (USE_RAND) {
          if (i != rand_threshold) {
            continue;
          }
        }
        double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, meta_->config->lambda_l1, l2,
            meta_->config->max_delta_step, constraints, 0, meta_->config->path_smooth,
            left_count, right_count, parent_output);
        if (current_gain <= min_gain_shift) {
          continue;
        }
        is_splittable_ = true;
        if (current_gain > best_gain) {
          best_sum_left_gradient_and_hessian = int_sum_left_gradient_and_hessian;
          best_threshold = i;
          best_gain = current_gain;
          best_dir = dir;
        }
      }
    }
  }

  if (is_splittable_) {
    const int32_t int_best_sum_left_gradient = HIST_BITS_ACC == 16 ?
      static_cast<int32_t>(static_cast<int16_t>(best_sum_left_gradient_and_hessian >> 16)) :
      static_cast<int32_t>(static_cast<int64_t>(best_sum_left_gradient_and_hessian) >> 32);
    const uint32_t int_best_sum_left_hessian = HIST_BITS_ACC == 16 ?
      static_cast<uint32_t>(best_sum_left_gradient_and_hessian & 0x0000ffff) :
      static_cast<uint32_t>(best_sum_left_gradient_and_hessian & 0x00000000ffffffff);
    const double best_sum_left_gradient = int_best_sum_left_gradient * grad_scale;
    const double best_sum_left_hessian = int_best_sum_left_hessian * hess_scale;

    const PACKED_HIST_ACC_T best_sum_right_gradient_and_hessian = local_int_sum_gradient_and_hessian - best_sum_left_gradient_and_hessian;
    const int32_t int_best_sum_right_gradient = HIST_BITS_ACC == 16 ?
      static_cast<int32_t>(static_cast<int16_t>(best_sum_right_gradient_and_hessian >> 16)) :
      static_cast<int32_t>(static_cast<int64_t>(best_sum_right_gradient_and_hessian) >> 32);
    const uint32_t int_best_sum_right_hessian = HIST_BITS_ACC == 16 ?
      static_cast<uint32_t>(best_sum_right_gradient_and_hessian & 0x0000ffff) :
      static_cast<uint32_t>(best_sum_right_gradient_and_hessian & 0x00000000ffffffff);
    const double best_sum_right_gradient = int_best_sum_right_gradient * grad_scale;
    const double best_sum_right_hessian = int_best_sum_right_hessian * hess_scale;

    const data_size_t best_left_count = Common::RoundInt(static_cast<double>(int_best_sum_left_hessian) * cnt_factor);
    const data_size_t best_right_count = Common::RoundInt(static_cast<double>(int_best_sum_right_hessian) * cnt_factor);

    const int64_t best_sum_left_gradient_and_hessian_int64 = HIST_BITS_ACC == 16 ?
        ((static_cast<int64_t>(static_cast<int16_t>(best_sum_left_gradient_and_hessian >> 16)) << 32) |
        static_cast<int64_t>(best_sum_left_gradient_and_hessian & 0x0000ffff)) :
        best_sum_left_gradient_and_hessian;
    const int64_t best_sum_right_gradient_and_hessian_int64 = int_sum_gradient_and_hessian - best_sum_left_gradient_and_hessian_int64;

    output->left_output = CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        best_sum_left_gradient, best_sum_left_hessian,
        meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
        constraints->LeftToBasicConstraint(), meta_->config->path_smooth, best_left_count, parent_output);
    output->left_count = best_left_count;
    output->left_sum_gradient = best_sum_left_gradient;
    output->left_sum_hessian = best_sum_left_hessian;
    output->right_output = CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        best_sum_right_gradient,
        best_sum_right_hessian, meta_->config->lambda_l1, l2,
        meta_->config->max_delta_step, constraints->RightToBasicConstraint(), meta_->config->path_smooth,
        best_right_count, parent_output);
    output->right_count = best_right_count;
    output->right_sum_gradient = best_sum_right_gradient;
    output->right_sum_hessian = best_sum_right_hessian;
    output->gain = best_gain - min_gain_shift;

    output->left_sum_gradient_and_hessian = best_sum_left_gradient_and_hessian_int64;
    output->right_sum_gradient_and_hessian = best_sum_right_gradient_and_hessian_int64;
    if (use_onehot) {
      output->num_cat_threshold = 1;
      output->cat_threshold =
          std::vector<uint32_t>(1, static_cast<uint32_t>(best_threshold + offset));
    } else {
      output->num_cat_threshold = best_threshold + 1;
      output->cat_threshold =
          std::vector<uint32_t>(output->num_cat_threshold);
      if (best_dir == 1) {
        for (int i = 0; i < output->num_cat_threshold; ++i) {
          auto t = sorted_idx[i] + offset;
          output->cat_threshold[i] = t;
        }
      } else {
        for (int i = 0; i < output->num_cat_threshold; ++i) {
          auto t = sorted_idx[used_bin - 1 - i] + offset;
          output->cat_threshold[i] = t;
        }
      }
    }
    output->monotone_type = 0;
  }
}

}  // namespace LightGBM
