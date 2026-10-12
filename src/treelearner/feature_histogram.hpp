/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_FEATURE_HISTOGRAM_HPP_
#define LIGHTGBM_SRC_TREELEARNER_FEATURE_HISTOGRAM_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/array_args.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "monotone_constraints.hpp"
#include "split_info.hpp"

namespace LightGBM {

class FeatureMetainfo {
 public:
  int num_bin;
  MissingType missing_type;
  int8_t offset = 0;
  uint32_t default_bin;
  int8_t monotone_type = 0;
  double penalty = 1.0;
  /*! \brief pointer of tree config */
  const Config* config;
  BinType bin_type;
  /*! \brief random number generator for extremely randomized trees */
  mutable Random rand;
};
/*!
 * \brief FeatureHistogram is used to construct and store a histogram for a
 * feature.
 */
class FeatureHistogram {
 public:
  FeatureHistogram() { data_ = nullptr; }

  ~FeatureHistogram() {}

  /*! \brief Disable copy */
  FeatureHistogram& operator=(const FeatureHistogram&) = delete;
  /*! \brief Disable copy */
  FeatureHistogram(const FeatureHistogram&) = delete;

  /*!
   * \brief Init the feature histogram
   * \param data the histogram data for this feature
   * \param data_int16 the 16-bit histogram data for this feature (used for quantized training via 'use_quantized_grad')
   * \param meta pointer to a FeatureMetaInfo object for this feature, with details like how missing values are handled and a pointer to the Config object with parameters
   */
  void Init(hist_t* data, int16_t* data_int16, const FeatureMetainfo* meta) {
    meta_ = meta;
    data_ = data;
    data_int16_ = data_int16;
    ResetFunc();
  }

  /*!
   * \brief Init the feature histogram
   * \param data the histogram data for this feature
   * \param meta pointer to a FeatureMetaInfo object for this feature, with details like how missing values are handled and a pointer to the Config object with parameters
   */
  void Init(hist_t* data, const FeatureMetainfo* meta) {
    meta_ = meta;
    data_ = data;
    data_int16_ = nullptr;
    ResetFunc();
  }

  void ResetFunc() {
    if (meta_->bin_type == BinType::NumericalBin) {
      FuncForNumrical();
    } else {
      FuncForCategorical();
    }
  }

  hist_t* RawData() { return data_; }

  int32_t* RawDataInt32() { return reinterpret_cast<int32_t*>(data_); }

  int16_t* RawDataInt16() { return data_int16_; }

  /*!
   * \brief Subtract current histograms with other
   * \param other The histogram that want to subtract
   */
  template <bool USE_DIST_GRAD = false,
    typename THIS_HIST_T = hist_t, typename OTHER_HIST_T = hist_t, typename RESULT_HIST_T = hist_t,
    int THIS_HIST_BITS = 0, int OTHER_HIST_BITS = 0, int RESULT_HIST_BITS = 0>
  void Subtract(const FeatureHistogram& other, const int32_t* buffer = nullptr) {
    if (USE_DIST_GRAD) {
      const THIS_HIST_T* this_int_data = THIS_HIST_BITS == 16 ?
        reinterpret_cast<const THIS_HIST_T*>(data_int16_) :
        (RESULT_HIST_BITS == 16 ?
          reinterpret_cast<const THIS_HIST_T*>(buffer) :
          reinterpret_cast<const THIS_HIST_T*>(data_));
      const OTHER_HIST_T* other_int_data = OTHER_HIST_BITS == 16 ?
        reinterpret_cast<OTHER_HIST_T*>(other.data_int16_) :
        reinterpret_cast<OTHER_HIST_T*>(other.data_);
      RESULT_HIST_T* result_int_data = RESULT_HIST_BITS == 16 ?
        reinterpret_cast<RESULT_HIST_T*>(data_int16_) :
        reinterpret_cast<RESULT_HIST_T*>(data_);
      if (THIS_HIST_BITS == 32 && OTHER_HIST_BITS == 16 && RESULT_HIST_BITS == 32) {
        for (int i = 0; i < meta_->num_bin - meta_->offset; ++i) {
          const int32_t other_grad_hess = static_cast<int32_t>(other_int_data[i]);
          const int64_t this_grad_hess = this_int_data[i];
          const int64_t other_grad_hess_int64 =
            (static_cast<int64_t>(static_cast<int16_t>(other_grad_hess >> 16)) << 32) |
            (static_cast<int64_t>(other_grad_hess & 0x0000ffff));
          const int64_t result_grad_hess = this_grad_hess - other_grad_hess_int64;
          result_int_data[i] = static_cast<RESULT_HIST_T>(result_grad_hess);
        }
      } else if (THIS_HIST_BITS == 32 && OTHER_HIST_BITS == 16 && RESULT_HIST_BITS == 16) {
        for (int i = 0; i < meta_->num_bin - meta_->offset; ++i) {
          const int32_t other_grad_hess = static_cast<int32_t>(other_int_data[i]);
          const int64_t this_grad_hess = this_int_data[i];
          const int64_t other_grad_hess_int64 =
            (static_cast<int64_t>(static_cast<int16_t>(other_grad_hess >> 16)) << 32) |
            (static_cast<int64_t>(other_grad_hess & 0x0000ffff));
          const int64_t result_grad_hess = this_grad_hess - other_grad_hess_int64;
          const int32_t result_grad_hess_int32 =
            (static_cast<int32_t>(result_grad_hess >> 32) << 16) |
            static_cast<int32_t>(result_grad_hess & 0x00000000ffffffff);
          result_int_data[i] = result_grad_hess_int32;
        }
      } else {
        for (int i = 0; i < meta_->num_bin - meta_->offset; ++i) {
          result_int_data[i] = this_int_data[i] - other_int_data[i];
        }
      }
    } else {
      for (int i = 0; i < (meta_->num_bin - meta_->offset) * 2; ++i) {
        data_[i] -= other.data_[i];
      }
    }
  }

  void CopyToBuffer(int32_t* buffer) {
    const int64_t* data_ptr = reinterpret_cast<const int64_t*>(data_);
    int64_t* buffer_ptr = reinterpret_cast<int64_t*>(buffer);
    for (int i = 0; i < meta_->num_bin - meta_->offset; ++i) {
      buffer_ptr[i] = data_ptr[i];
    }
  }

  void CopyFromInt16ToInt32(char* buffer) {
    const int32_t* int16_data = reinterpret_cast<const int32_t*>(RawDataInt16());
    int64_t* int32_data = reinterpret_cast<int64_t*>(buffer);
    for (int i = 0; i < meta_->num_bin - meta_->offset; ++i) {
      const int32_t int16_val = int16_data[i];
      int32_data[i] = (static_cast<int64_t>(static_cast<int16_t>(int16_val >> 16)) << 32) |
        static_cast<int64_t>(int16_val & 0x0000ffff);
    }
  }

  void FindBestThreshold(double sum_gradient, double sum_hessian,
                         data_size_t num_data,
                         const FeatureConstraint* constraints,
                         double parent_output,
                         SplitInfo* output) {
    output->default_left = true;
    output->gain = kMinScore;
    find_best_threshold_fun_(sum_gradient, sum_hessian + 2 * kEpsilon, num_data,
                             constraints, parent_output, output);
    output->gain *= meta_->penalty;
  }

  void FindBestThresholdInt(int64_t sum_gradient_and_hessian,
                            double grad_scale, double hess_scale,
                            const uint8_t num_bits_bin,
                            const uint8_t num_bits_acc,
                            data_size_t num_data,
                            const FeatureConstraint* constraints,
                            double parent_output,
                            SplitInfo* output) {
    output->default_left = true;
    output->gain = kMinScore;
    int_find_best_threshold_fun_(sum_gradient_and_hessian, grad_scale, hess_scale, num_bits_bin, num_bits_acc, num_data,
                             constraints, parent_output, output);
    output->gain *= meta_->penalty;
  }

  template <bool USE_RAND, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  double BeforeNumerical(double sum_gradient, double sum_hessian, double parent_output, data_size_t num_data,
                        SplitInfo* output, int* rand_threshold) {
    is_splittable_ = false;
    output->monotone_type = meta_->monotone_type;

    double gain_shift = GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2,
        meta_->config->max_delta_step, meta_->config->path_smooth, num_data, parent_output);
    *rand_threshold = 0;
    if (USE_RAND) {
      if (meta_->num_bin - 2 > 0) {
        *rand_threshold = meta_->rand.NextInt(0, meta_->num_bin - 2);
      }
    }
    return gain_shift + meta_->config->min_gain_to_split;
  }

  template <bool USE_RAND, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  double BeforeNumericalInt(int64_t sum_gradient_and_hessian, double grad_scale, double hess_scale, double parent_output, data_size_t num_data,
                        SplitInfo* output, int* rand_threshold) {
    is_splittable_ = false;
    output->monotone_type = meta_->monotone_type;
    const int32_t int_sum_gradient = static_cast<int32_t>(sum_gradient_and_hessian >> 32);
    const uint32_t int_sum_hessian = static_cast<uint32_t>(sum_gradient_and_hessian & 0x00000000ffffffff);
    const double sum_gradient = static_cast<double>(int_sum_gradient) * grad_scale;
    const double sum_hessian = static_cast<double>(int_sum_hessian) * hess_scale;
    double gain_shift = GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2,
        meta_->config->max_delta_step, meta_->config->path_smooth, num_data, parent_output);
    *rand_threshold = 0;
    if (USE_RAND) {
      if (meta_->num_bin - 2 > 0) {
        *rand_threshold = meta_->rand.NextInt(0, meta_->num_bin - 2);
      }
    }
    return gain_shift + meta_->config->min_gain_to_split;
  }

  void FuncForNumrical() {
    if (meta_->config->extra_trees) {
      if (meta_->config->monotone_constraints.empty()) {
        FuncForNumricalL1<true, false>();
      } else {
        FuncForNumricalL1<true, true>();
      }
    } else {
      if (meta_->config->monotone_constraints.empty()) {
        FuncForNumricalL1<false, false>();
      } else {
        FuncForNumricalL1<false, true>();
      }
    }
  }
  template <bool USE_RAND, bool USE_MC>
  void FuncForNumricalL1() {
    if (meta_->config->lambda_l1 > 0) {
      if (meta_->config->max_delta_step > 0) {
        FuncForNumricalL2<USE_RAND, USE_MC, true, true>();
      } else {
        FuncForNumricalL2<USE_RAND, USE_MC, true, false>();
      }
    } else {
      if (meta_->config->max_delta_step > 0) {
        FuncForNumricalL2<USE_RAND, USE_MC, false, true>();
      } else {
        FuncForNumricalL2<USE_RAND, USE_MC, false, false>();
      }
    }
  }

  template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT>
  void FuncForNumricalL2() {
    if (meta_->config->path_smooth > kEpsilon) {
      FuncForNumricalL3<USE_RAND, USE_MC, USE_L1, USE_MAX_OUTPUT, true>();
    } else {
      FuncForNumricalL3<USE_RAND, USE_MC, USE_L1, USE_MAX_OUTPUT, false>();
    }
  }

  template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  void FuncForNumricalL3() {
  if (meta_->config->use_quantized_grad) {
#define TEMPLATE_PREFIX_INT USE_RAND, USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING
#define LAMBDA_ARGUMENTS_INT                                         \
  int64_t sum_gradient_and_hessian, double grad_scale, double hess_scale, const uint8_t hist_bits_bin, const uint8_t hist_bits_acc, data_size_t num_data, \
      const FeatureConstraint* constraints, double parent_output, SplitInfo *output
#define BEFORE_ARGUMENTS_INT sum_gradient_and_hessian, grad_scale, hess_scale, parent_output, num_data, output, &rand_threshold
#define FUNC_ARGUMENTS_INT                                                      \
  sum_gradient_and_hessian, grad_scale, hess_scale, num_data, constraints, min_gain_shift, \
      output, rand_threshold, parent_output

      if (meta_->num_bin > 2 && meta_->missing_type != MissingType::None) {
        if (meta_->missing_type == MissingType::Zero) {
          int_find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS_INT) {
            int rand_threshold = 0;
            double min_gain_shift =
                BeforeNumericalInt<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                    BEFORE_ARGUMENTS_INT);
            if (hist_bits_acc <= 16) {
              CHECK_LE(hist_bits_bin, 16);
              FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, true, false, int32_t, int32_t, int16_t, int16_t, 16, 16>(
                  FUNC_ARGUMENTS_INT);
              FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, false, true, false, int32_t, int32_t, int16_t, int16_t, 16, 16>(
                  FUNC_ARGUMENTS_INT);
            } else {
              if (hist_bits_bin == 32) {
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, true, false, int64_t, int64_t, int32_t, int32_t, 32, 32>(
                    FUNC_ARGUMENTS_INT);
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, false, true, false, int64_t, int64_t, int32_t, int32_t, 32, 32>(
                    FUNC_ARGUMENTS_INT);
              } else {
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, true, false, int32_t, int64_t, int16_t, int32_t, 16, 32>(
                    FUNC_ARGUMENTS_INT);
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, false, true, false, int32_t, int64_t, int16_t, int32_t, 16, 32>(
                    FUNC_ARGUMENTS_INT);
              }
            }
          };
        } else {
          int_find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS_INT) {
            int rand_threshold = 0;
            double min_gain_shift =
                BeforeNumericalInt<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                    BEFORE_ARGUMENTS_INT);
            if (hist_bits_acc <= 16) {
              CHECK_LE(hist_bits_bin, 16);
              FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, true, int32_t, int32_t, int16_t, int16_t, 16, 16>(
                  FUNC_ARGUMENTS_INT);
              FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, false, false, true, int32_t, int32_t, int16_t, int16_t, 16, 16>(
                  FUNC_ARGUMENTS_INT);
            } else {
              if (hist_bits_bin == 32) {
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, true, int64_t, int64_t, int32_t, int32_t, 32, 32>(
                    FUNC_ARGUMENTS_INT);
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, false, false, true, int64_t, int64_t, int32_t, int32_t, 32, 32>(
                    FUNC_ARGUMENTS_INT);
              } else {
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, true, int32_t, int64_t, int16_t, int32_t, 16, 32>(
                    FUNC_ARGUMENTS_INT);
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, false, false, true, int32_t, int64_t, int16_t, int32_t, 16, 32>(
                    FUNC_ARGUMENTS_INT);
              }
            }
          };
        }
      } else {
        if (meta_->missing_type != MissingType::NaN) {
          int_find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS_INT) {
            int rand_threshold = 0;
            double min_gain_shift =
                BeforeNumericalInt<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                    BEFORE_ARGUMENTS_INT);
            if (hist_bits_acc <= 16) {
              CHECK_LE(hist_bits_bin, 16);
              FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, false, int32_t, int32_t, int16_t, int16_t, 16, 16>(
                  FUNC_ARGUMENTS_INT);
            } else {
              if (hist_bits_bin == 32) {
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, false, int64_t, int64_t, int32_t, int32_t, 32, 32>(
                    FUNC_ARGUMENTS_INT);
              } else {
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, false, int32_t, int64_t, int16_t, int32_t, 16, 32>(
                    FUNC_ARGUMENTS_INT);
              }
            }
          };
        } else {
          int_find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS_INT) {
            int rand_threshold = 0;
            double min_gain_shift =
                BeforeNumericalInt<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                    BEFORE_ARGUMENTS_INT);
            if (hist_bits_acc <= 16) {
              CHECK_LE(hist_bits_bin, 16);
              FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, false, int32_t, int32_t, int16_t, int16_t, 16, 16>(
                  FUNC_ARGUMENTS_INT);
            } else {
              if (hist_bits_bin == 32) {
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, false, int64_t, int64_t, int32_t, int32_t, 32, 32>(
                    FUNC_ARGUMENTS_INT);
              } else {
                FindBestThresholdSequentiallyInt<TEMPLATE_PREFIX_INT, true, false, false, int32_t, int64_t, int16_t, int32_t, 16, 32>(
                    FUNC_ARGUMENTS_INT);
              }
            }
            output->default_left = false;
          };
        }
      }
#undef TEMPLATE_PREFIX_INT
#undef LAMBDA_ARGUMENTS_INT
#undef BEFORE_ARGUMENTS_INT
#undef FUNC_ARGURMENTS_INT
  } else {
#define TEMPLATE_PREFIX USE_RAND, USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING
#define LAMBDA_ARGUMENTS                                         \
  double sum_gradient, double sum_hessian, data_size_t num_data, \
      const FeatureConstraint* constraints, double parent_output, SplitInfo *output
#define BEFORE_ARGUMENTS sum_gradient, sum_hessian, parent_output, num_data, output, &rand_threshold
#define FUNC_ARGUMENTS                                                      \
  sum_gradient, sum_hessian, num_data, constraints, min_gain_shift, \
      output, rand_threshold, parent_output

      if (meta_->num_bin > 2 && meta_->missing_type != MissingType::None) {
        if (meta_->missing_type == MissingType::Zero) {
          find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS) {
            int rand_threshold = 0;
            double min_gain_shift =
                BeforeNumerical<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                    BEFORE_ARGUMENTS);
            FindBestThresholdSequentially<TEMPLATE_PREFIX, true, true, false>(
                FUNC_ARGUMENTS);
            FindBestThresholdSequentially<TEMPLATE_PREFIX, false, true, false>(
                FUNC_ARGUMENTS);
          };
        } else {
          find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS) {
            int rand_threshold = 0;
            double min_gain_shift =
                BeforeNumerical<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                    BEFORE_ARGUMENTS);
            FindBestThresholdSequentially<TEMPLATE_PREFIX, true, false, true>(
                FUNC_ARGUMENTS);
            FindBestThresholdSequentially<TEMPLATE_PREFIX, false, false, true>(
                FUNC_ARGUMENTS);
          };
        }
      } else {
        if (meta_->missing_type != MissingType::NaN) {
          find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS) {
            int rand_threshold = 0;
            double min_gain_shift =
                BeforeNumerical<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                    BEFORE_ARGUMENTS);
            FindBestThresholdSequentially<TEMPLATE_PREFIX, true, false, false>(
                FUNC_ARGUMENTS);
          };
        } else {
          find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS) {
            int rand_threshold = 0;
            double min_gain_shift =
                BeforeNumerical<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                    BEFORE_ARGUMENTS);
            FindBestThresholdSequentially<TEMPLATE_PREFIX, true, false, false>(
                FUNC_ARGUMENTS);
            output->default_left = false;
          };
        }
      }
#undef TEMPLATE_PREFIX
#undef LAMBDA_ARGUMENTS
#undef BEFORE_ARGUMENTS
#undef FUNC_ARGURMENTS
    }
  }

  void FuncForCategorical();

  template <bool USE_RAND, bool USE_MC>
  void FuncForCategoricalL1();

  template <bool USE_RAND, bool USE_MC, bool USE_SMOOTHING>
  void FuncForCategoricalL2();

  template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  void FindBestThresholdCategoricalInner(double sum_gradient,
                                         double sum_hessian,
                                         data_size_t num_data,
                                         const FeatureConstraint* constraints,
                                         double parent_output,
                                         SplitInfo* output);

  template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING, typename PACKED_HIST_BIN_T, typename PACKED_HIST_ACC_T,
          typename HIST_BIN_T, typename HIST_ACC_T, int HIST_BITS_BIN, int HIST_BITS_ACC>
  void FindBestThresholdCategoricalIntInner(int64_t int_sum_gradient_and_hessian,
                                            const double grad_scale, const double hess_scale,
                                            data_size_t num_data,
                                            const FeatureConstraint* constraints,
                                            double parent_output,
                                            SplitInfo* output);

  void GatherInfoForThreshold(double sum_gradient, double sum_hessian,
                              uint32_t threshold, data_size_t num_data,
                              double parent_output, SplitInfo* output) {
    if (meta_->bin_type == BinType::NumericalBin) {
      GatherInfoForThresholdNumerical(sum_gradient, sum_hessian, threshold,
                                      num_data, parent_output, output);
    } else {
      GatherInfoForThresholdCategorical(sum_gradient, sum_hessian, threshold,
                                        num_data, parent_output, output);
    }
  }

  void GatherInfoForThresholdNumerical(double sum_gradient, double sum_hessian,
                                       uint32_t threshold, data_size_t num_data,
                                       double parent_output, SplitInfo* output) {
    bool use_smoothing = meta_->config->path_smooth > kEpsilon;
    if (use_smoothing) {
      GatherInfoForThresholdNumericalInner<true>(sum_gradient, sum_hessian,
                                                 threshold, num_data,
                                                 parent_output, output);
    } else {
      GatherInfoForThresholdNumericalInner<false>(sum_gradient, sum_hessian,
                                                  threshold, num_data,
                                                  parent_output, output);
    }
  }

  template<bool USE_SMOOTHING>
  void GatherInfoForThresholdNumericalInner(double sum_gradient, double sum_hessian,
                                            uint32_t threshold, data_size_t num_data,
                                            double parent_output, SplitInfo* output) {
    double gain_shift = GetLeafGainGivenOutput<true>(
        sum_gradient, sum_hessian, meta_->config->lambda_l1,
        meta_->config->lambda_l2, parent_output);
    double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;

    // do stuff here
    const int8_t offset = meta_->offset;

    double sum_right_gradient = 0.0f;
    double sum_right_hessian = kEpsilon;
    data_size_t right_count = 0;

    // set values
    bool use_na_as_missing = false;
    bool skip_default_bin = false;
    if (meta_->missing_type == MissingType::Zero) {
      skip_default_bin = true;
    } else if (meta_->missing_type == MissingType::NaN) {
      use_na_as_missing = true;
    }

    int t = meta_->num_bin - 1 - offset - use_na_as_missing;
    const int t_end = 1 - offset;
    const double cnt_factor = num_data / sum_hessian;
    // from right to left, and we don't need data in bin0
    for (; t >= t_end; --t) {
      if (static_cast<uint32_t>(t + offset) <= threshold) {
        break;
      }

      // need to skip default bin
      if (skip_default_bin &&
          (t + offset) == static_cast<int>(meta_->default_bin)) {
        continue;
      }
      const auto grad = GET_GRAD(data_, t);
      const auto hess = GET_HESS(data_, t);
      data_size_t cnt =
          static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
      sum_right_gradient += grad;
      sum_right_hessian += hess;
      right_count += cnt;
    }
    double sum_left_gradient = sum_gradient - sum_right_gradient;
    double sum_left_hessian = sum_hessian - sum_right_hessian;
    data_size_t left_count = num_data - right_count;
    double current_gain =
        GetLeafGain<true, true, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, meta_->config->lambda_l1,
            meta_->config->lambda_l2, meta_->config->max_delta_step,
            meta_->config->path_smooth, left_count, parent_output) +
        GetLeafGain<true, true, USE_SMOOTHING>(
            sum_right_gradient, sum_right_hessian, meta_->config->lambda_l1,
            meta_->config->lambda_l2, meta_->config->max_delta_step,
            meta_->config->path_smooth, right_count, parent_output);

    // gain with split is worse than without split
    if (std::isnan(current_gain) || current_gain <= min_gain_shift) {
      output->gain = kMinScore;
      Log::Warning(
          "'Forced Split' will be ignored since the gain getting worse.");
      return;
    }

    // update split information
    output->threshold = threshold;
    output->left_output = CalculateSplittedLeafOutput<true, true, USE_SMOOTHING>(
        sum_left_gradient, sum_left_hessian, meta_->config->lambda_l1,
        meta_->config->lambda_l2, meta_->config->max_delta_step,
        meta_->config->path_smooth, left_count, parent_output);
    output->left_count = left_count;
    output->left_sum_gradient = sum_left_gradient;
    output->left_sum_hessian = sum_left_hessian - kEpsilon;
    output->right_output = CalculateSplittedLeafOutput<true, true, USE_SMOOTHING>(
        sum_gradient - sum_left_gradient, sum_hessian - sum_left_hessian,
        meta_->config->lambda_l1, meta_->config->lambda_l2,
        meta_->config->max_delta_step, meta_->config->path_smooth,
        right_count, parent_output);
    output->right_count = num_data - left_count;
    output->right_sum_gradient = sum_gradient - sum_left_gradient;
    output->right_sum_hessian = sum_hessian - sum_left_hessian - kEpsilon;
    output->gain = current_gain - min_gain_shift;
    output->default_left = true;
  }

  void GatherInfoForThresholdCategorical(double sum_gradient,  double sum_hessian,
                                         uint32_t threshold, data_size_t num_data,
                                         double parent_output, SplitInfo* output) {
    bool use_smoothing = meta_->config->path_smooth > kEpsilon;
    if (use_smoothing) {
      GatherInfoForThresholdCategoricalInner<true>(sum_gradient, sum_hessian, threshold,
                                                   num_data, parent_output, output);
    } else {
      GatherInfoForThresholdCategoricalInner<false>(sum_gradient, sum_hessian, threshold,
                                                    num_data, parent_output, output);
    }
  }

  template<bool USE_SMOOTHING>
  void GatherInfoForThresholdCategoricalInner(double sum_gradient,
                                              double sum_hessian, uint32_t threshold,
                                              data_size_t num_data, double parent_output,
                                              SplitInfo* output) {
    // get SplitInfo for a given one-hot categorical split.
    output->default_left = false;
    double gain_shift = GetLeafGainGivenOutput<true>(
        sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2, parent_output);
    double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
    if (threshold >= static_cast<uint32_t>(meta_->num_bin) || threshold == 0) {
      output->gain = kMinScore;
      Log::Warning("Invalid categorical threshold split");
      return;
    }
    const double cnt_factor = num_data / sum_hessian;
    const auto grad = GET_GRAD(data_, threshold - meta_->offset);
    const auto hess = GET_HESS(data_, threshold - meta_->offset);
    data_size_t cnt =
        static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));

    double l2 = meta_->config->lambda_l2;
    data_size_t left_count = cnt;
    data_size_t right_count = num_data - left_count;
    double sum_left_hessian = hess + kEpsilon;
    double sum_right_hessian = sum_hessian - sum_left_hessian;
    double sum_left_gradient = grad;
    double sum_right_gradient = sum_gradient - sum_left_gradient;
    // current split gain
    double current_gain =
        GetLeafGain<true, true, USE_SMOOTHING>(sum_right_gradient, sum_right_hessian,
                                      meta_->config->lambda_l1, l2,
                                      meta_->config->max_delta_step,
                                      meta_->config->path_smooth, right_count,
                                      parent_output) +
        GetLeafGain<true, true, USE_SMOOTHING>(sum_left_gradient, sum_left_hessian,
                                      meta_->config->lambda_l1, l2,
                                      meta_->config->max_delta_step,
                                      meta_->config->path_smooth, left_count,
                                      parent_output);
    if (std::isnan(current_gain) || current_gain <= min_gain_shift) {
      output->gain = kMinScore;
      Log::Warning(
          "'Forced Split' will be ignored since the gain getting worse.");
      return;
    }
    output->left_output = CalculateSplittedLeafOutput<true, true, USE_SMOOTHING>(
        sum_left_gradient, sum_left_hessian, meta_->config->lambda_l1, l2,
        meta_->config->max_delta_step, meta_->config->path_smooth, left_count,
        parent_output);
    output->left_count = left_count;
    output->left_sum_gradient = sum_left_gradient;
    output->left_sum_hessian = sum_left_hessian - kEpsilon;
    output->right_output = CalculateSplittedLeafOutput<true, true, USE_SMOOTHING>(
        sum_right_gradient, sum_right_hessian, meta_->config->lambda_l1, l2,
        meta_->config->max_delta_step, meta_->config->path_smooth, right_count,
        parent_output);
    output->right_count = right_count;
    output->right_sum_gradient = sum_gradient - sum_left_gradient;
    output->right_sum_hessian = sum_right_hessian - kEpsilon;
    output->gain = current_gain - min_gain_shift;
    output->num_cat_threshold = 1;
    output->cat_threshold = std::vector<uint32_t>(1, threshold);
  }

  /*!
   * \brief Binary size of this histogram
   */
  int SizeOfHistogram() const {
    return (meta_->num_bin - meta_->offset) * kHistEntrySize;
  }

  int SizeOfInt32Histogram() const {
    return (meta_->num_bin - meta_->offset) * kInt32HistEntrySize;
  }

  int SizeOfInt16Histogram() const {
    return (meta_->num_bin - meta_->offset) * kInt16HistEntrySize;
  }

  /*!
   * \brief Restore histogram from memory
   */
  void FromMemory(char* memory_data) {
    std::memcpy(data_, memory_data,
                (meta_->num_bin - meta_->offset) * kHistEntrySize);
  }

  void FromMemoryInt32(char* memory_data) {
    std::memcpy(data_, memory_data,
                (meta_->num_bin - meta_->offset) * kInt32HistEntrySize);
  }

  void FromMemoryInt16(char* memory_data) {
    std::memcpy(data_int16_, memory_data,
                (meta_->num_bin - meta_->offset) * kInt16HistEntrySize);
  }

  /*!
   * \brief True if this histogram can be splitted
   */
  bool is_splittable() const { return is_splittable_; }

  /*!
   * \brief Set splittable to this histogram
   */
  void set_is_splittable(bool val) { is_splittable_ = val; }

  static double ThresholdL1(double s, double l1) {
    const double reg_s = std::max(0.0, std::fabs(s) - l1);
    return Common::Sign(s) * reg_s;
  }

  template <bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  static double CalculateSplittedLeafOutput(double sum_gradients,
                                            double sum_hessians, double l1,
                                            double l2, double max_delta_step,
                                            double smoothing, data_size_t num_data,
                                            double parent_output) {
    double ret;
    if (USE_L1) {
      ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
    } else {
      ret = -sum_gradients / (sum_hessians + l2);
    }
    if (USE_MAX_OUTPUT) {
      if (max_delta_step > 0 && std::fabs(ret) > max_delta_step) {
        ret = Common::Sign(ret) * max_delta_step;
      }
    }
    if (USE_SMOOTHING) {
      ret = ret * (num_data / smoothing) / (num_data / smoothing + 1) \
          + parent_output / (num_data / smoothing + 1);
    }
    return ret;
  }

  template <bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  static double CalculateSplittedLeafOutput(
      double sum_gradients, double sum_hessians, double l1, double l2,
      double max_delta_step, const BasicConstraint& constraints,
      double smoothing, data_size_t num_data, double parent_output) {
    double ret = CalculateSplittedLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        sum_gradients, sum_hessians, l1, l2, max_delta_step, smoothing, num_data, parent_output);
    if (USE_MC) {
      if (ret < constraints.min) {
        ret = constraints.min;
      } else if (ret > constraints.max) {
        ret = constraints.max;
      }
    }
    return ret;
  }

 private:
  template <bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  static double GetSplitGains(double sum_left_gradients,
                              double sum_left_hessians,
                              double sum_right_gradients,
                              double sum_right_hessians, double l1, double l2,
                              double max_delta_step,
                              const FeatureConstraint* constraints,
                              int8_t monotone_constraint,
                              double smoothing,
                              data_size_t left_count,
                              data_size_t right_count,
                              double parent_output) {
    if (!USE_MC) {
      return GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(sum_left_gradients,
                                                                sum_left_hessians, l1, l2,
                                                                max_delta_step, smoothing,
                                                                left_count, parent_output) +
             GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(sum_right_gradients,
                                                                sum_right_hessians, l1, l2,
                                                                max_delta_step, smoothing,
                                                                right_count, parent_output);
    } else {
      double left_output =
          CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
              sum_left_gradients, sum_left_hessians, l1, l2, max_delta_step,
              constraints->LeftToBasicConstraint(), smoothing, left_count, parent_output);
      double right_output =
          CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
              sum_right_gradients, sum_right_hessians, l1, l2, max_delta_step,
              constraints->RightToBasicConstraint(), smoothing, right_count, parent_output);
      if (((monotone_constraint > 0) && (left_output > right_output)) ||
          ((monotone_constraint < 0) && (left_output < right_output))) {
        return 0;
      }
      return GetLeafGainGivenOutput<USE_L1>(
                 sum_left_gradients, sum_left_hessians, l1, l2, left_output) +
             GetLeafGainGivenOutput<USE_L1>(
                 sum_right_gradients, sum_right_hessians, l1, l2, right_output);
    }
  }

  template <bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
  static double GetLeafGain(double sum_gradients, double sum_hessians,
                            double l1, double l2, double max_delta_step,
                            double smoothing, data_size_t num_data, double parent_output) {
    if (!USE_MAX_OUTPUT && !USE_SMOOTHING) {
      if (USE_L1) {
        const double sg_l1 = ThresholdL1(sum_gradients, l1);
        return (sg_l1 * sg_l1) / (sum_hessians + l2);
      } else {
        return (sum_gradients * sum_gradients) / (sum_hessians + l2);
      }
    } else {
      double output = CalculateSplittedLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
          sum_gradients, sum_hessians, l1, l2, max_delta_step, smoothing, num_data, parent_output);
      return GetLeafGainGivenOutput<USE_L1>(sum_gradients, sum_hessians, l1, l2, output);
    }
  }

  template <bool USE_L1>
  static double GetLeafGainGivenOutput(double sum_gradients,
                                       double sum_hessians, double l1,
                                       double l2, double output) {
    if (USE_L1) {
      const double sg_l1 = ThresholdL1(sum_gradients, l1);
      return -(2.0 * sg_l1 * output + (sum_hessians + l2) * output * output);
    } else {
      return -(2.0 * sum_gradients * output +
               (sum_hessians + l2) * output * output);
    }
  }

  template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING,
            bool REVERSE, bool SKIP_DEFAULT_BIN, bool NA_AS_MISSING>
  void FindBestThresholdSequentially(double sum_gradient, double sum_hessian,
                                     data_size_t num_data,
                                     const FeatureConstraint* constraints,
                                     double min_gain_shift, SplitInfo* output,
                                     int rand_threshold, double parent_output) {
    const int8_t offset = meta_->offset;
    double best_sum_left_gradient = NAN;
    double best_sum_left_hessian = NAN;
    double best_gain = kMinScore;
    data_size_t best_left_count = 0;
    uint32_t best_threshold = static_cast<uint32_t>(meta_->num_bin);
    const double cnt_factor = num_data / sum_hessian;

    BasicConstraint best_right_constraints;
    BasicConstraint best_left_constraints;
    bool constraint_update_necessary =
        USE_MC && constraints->ConstraintDifferentDependingOnThreshold();

    if (USE_MC) {
      constraints->InitCumulativeConstraints(REVERSE);
    }

    if (REVERSE) {
      double sum_right_gradient = 0.0f;
      double sum_right_hessian = kEpsilon;
      data_size_t right_count = 0;

      int t = meta_->num_bin - 1 - offset - NA_AS_MISSING;
      const int t_end = 1 - offset;

      // from right to left, and we don't need data in bin0
      for (; t >= t_end; --t) {
        // need to skip default bin
        if (SKIP_DEFAULT_BIN) {
          if ((t + offset) == static_cast<int>(meta_->default_bin)) {
            continue;
          }
        }
        const auto grad = GET_GRAD(data_, t);
        const auto hess = GET_HESS(data_, t);
        data_size_t cnt =
            static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
        sum_right_gradient += grad;
        sum_right_hessian += hess;
        right_count += cnt;
        // if data not enough, or sum hessian too small
        if (right_count < meta_->config->min_data_in_leaf ||
            sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) {
          continue;
        }
        data_size_t left_count = num_data - right_count;
        // if data not enough
        if (left_count < meta_->config->min_data_in_leaf) {
          break;
        }

        double sum_left_hessian = sum_hessian - sum_right_hessian;
        // if sum hessian too small
        if (sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) {
          break;
        }

        double sum_left_gradient = sum_gradient - sum_right_gradient;
        if (USE_RAND) {
          if (t - 1 + offset != rand_threshold) {
            continue;
          }
        }

        if (USE_MC && constraint_update_necessary) {
          constraints->Update(t + offset);
        }

        // current split gain
        double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, meta_->config->lambda_l1,
            meta_->config->lambda_l2, meta_->config->max_delta_step,
            constraints, meta_->monotone_type, meta_->config->path_smooth,
            left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain <= min_gain_shift) {
          continue;
        }

        // mark as able to be split
        is_splittable_ = true;
        // better split point
        if (current_gain > best_gain) {
          if (USE_MC) {
            best_right_constraints = constraints->RightToBasicConstraint();
            best_left_constraints = constraints->LeftToBasicConstraint();
            if (best_right_constraints.min > best_right_constraints.max ||
                best_left_constraints.min > best_left_constraints.max) {
              continue;
            }
          }
          best_left_count = left_count;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          // left is <= threshold, right is > threshold.  so this is t-1
          best_threshold = static_cast<uint32_t>(t - 1 + offset);
          best_gain = current_gain;
        }
      }
    } else {
      double sum_left_gradient = 0.0f;
      double sum_left_hessian = kEpsilon;
      data_size_t left_count = 0;

      int t = 0;
      const int t_end = meta_->num_bin - 2 - offset;

      if (NA_AS_MISSING) {
        if (offset == 1) {
          sum_left_gradient = sum_gradient;
          sum_left_hessian = sum_hessian - kEpsilon;
          left_count = num_data;
          for (int i = 0; i < meta_->num_bin - offset; ++i) {
            const auto grad = GET_GRAD(data_, i);
            const auto hess = GET_HESS(data_, i);
            data_size_t cnt =
                static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
            sum_left_gradient -= grad;
            sum_left_hessian -= hess;
            left_count -= cnt;
          }
          t = -1;
        }
      }

      for (; t <= t_end; ++t) {
        if (SKIP_DEFAULT_BIN) {
          if ((t + offset) == static_cast<int>(meta_->default_bin)) {
            continue;
          }
        }
        if (t >= 0) {
          sum_left_gradient += GET_GRAD(data_, t);
          sum_left_hessian += GET_HESS(data_, t);
          left_count += static_cast<data_size_t>(
              Common::RoundInt(GET_HESS(data_, t) * cnt_factor));
        }
        // if data not enough, or sum hessian too small
        if (left_count < meta_->config->min_data_in_leaf ||
            sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) {
          continue;
        }
        data_size_t right_count = num_data - left_count;
        // if data not enough
        if (right_count < meta_->config->min_data_in_leaf) {
          break;
        }

        double sum_right_hessian = sum_hessian - sum_left_hessian;
        // if sum Hessian too small
        if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) {
          break;
        }

        double sum_right_gradient = sum_gradient - sum_left_gradient;
        if (USE_RAND) {
          if (t + offset != rand_threshold) {
            continue;
          }
        }
        // current split gain
        double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, meta_->config->lambda_l1,
            meta_->config->lambda_l2, meta_->config->max_delta_step,
            constraints, meta_->monotone_type, meta_->config->path_smooth, left_count,
            right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain <= min_gain_shift) {
          continue;
        }

        // mark as able to be split
        is_splittable_ = true;
        // better split point
        if (current_gain > best_gain) {
          if (USE_MC) {
            best_right_constraints = constraints->RightToBasicConstraint();
            best_left_constraints = constraints->LeftToBasicConstraint();
            if (best_right_constraints.min > best_right_constraints.max ||
                best_left_constraints.min > best_left_constraints.max) {
              continue;
            }
          }
          best_left_count = left_count;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_threshold = static_cast<uint32_t>(t + offset);
          best_gain = current_gain;
        }
      }
    }

    if (is_splittable_ && best_gain > output->gain + min_gain_shift) {
      // update split information
      output->threshold = best_threshold;
      output->left_output =
          CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
              best_sum_left_gradient, best_sum_left_hessian,
              meta_->config->lambda_l1, meta_->config->lambda_l2,
              meta_->config->max_delta_step, best_left_constraints, meta_->config->path_smooth,
              best_left_count, parent_output);
      output->left_count = best_left_count;
      output->left_sum_gradient = best_sum_left_gradient;
      output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
      output->right_output =
          CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
              sum_gradient - best_sum_left_gradient,
              sum_hessian - best_sum_left_hessian, meta_->config->lambda_l1,
              meta_->config->lambda_l2, meta_->config->max_delta_step,
              best_right_constraints, meta_->config->path_smooth, num_data - best_left_count,
              parent_output);
      output->right_count = num_data - best_left_count;
      output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
      output->right_sum_hessian =
          sum_hessian - best_sum_left_hessian - kEpsilon;
      output->gain = best_gain - min_gain_shift;
      output->default_left = REVERSE;
    }
  }

  template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING,
          bool REVERSE, bool SKIP_DEFAULT_BIN, bool NA_AS_MISSING, typename PACKED_HIST_BIN_T, typename PACKED_HIST_ACC_T,
          typename HIST_BIN_T, typename HIST_ACC_T, int HIST_BITS_BIN, int HIST_BITS_ACC>
  void FindBestThresholdSequentiallyInt(int64_t int_sum_gradient_and_hessian,
                                        const double grad_scale, const double hess_scale,
                                        data_size_t num_data,
                                        const FeatureConstraint* constraints,
                                        double min_gain_shift, SplitInfo* output,
                                        int rand_threshold, double parent_output) {
    const int8_t offset = meta_->offset;
    PACKED_HIST_ACC_T best_sum_left_gradient_and_hessian = 0;
    PACKED_HIST_ACC_T local_int_sum_gradient_and_hessian =
      HIST_BITS_ACC == 16 ?
      ((static_cast<int32_t>(int_sum_gradient_and_hessian >> 32) << 16) | static_cast<int32_t>(int_sum_gradient_and_hessian & 0x0000ffff)) :
      static_cast<PACKED_HIST_ACC_T>(int_sum_gradient_and_hessian);
    double best_gain = kMinScore;
    uint32_t best_threshold = static_cast<uint32_t>(meta_->num_bin);
    const double cnt_factor = static_cast<double>(num_data) /
      static_cast<double>(static_cast<uint32_t>(int_sum_gradient_and_hessian & 0x00000000ffffffff));

    BasicConstraint best_right_constraints;
    BasicConstraint best_left_constraints;
    bool constraint_update_necessary =
        USE_MC && constraints->ConstraintDifferentDependingOnThreshold();

    if (USE_MC) {
      constraints->InitCumulativeConstraints(REVERSE);
    }

    const PACKED_HIST_BIN_T* data_ptr = nullptr;
    if (HIST_BITS_BIN == 16) {
      data_ptr = reinterpret_cast<const PACKED_HIST_BIN_T*>(data_int16_);
    } else {
      data_ptr = reinterpret_cast<const PACKED_HIST_BIN_T*>(data_);
    }
    if (REVERSE) {
      PACKED_HIST_ACC_T sum_right_gradient_and_hessian = 0;

      int t = meta_->num_bin - 1 - offset - NA_AS_MISSING;
      const int t_end = 1 - offset;

      // from right to left, and we don't need data in bin0
      for (; t >= t_end; --t) {
        // need to skip default bin
        if (SKIP_DEFAULT_BIN) {
          if ((t + offset) == static_cast<int>(meta_->default_bin)) {
            continue;
          }
        }
        const PACKED_HIST_BIN_T grad_and_hess = data_ptr[t];
        if (HIST_BITS_ACC != HIST_BITS_BIN) {
          const PACKED_HIST_ACC_T grad_and_hess_acc = HIST_BITS_BIN == 16 ?
            ((static_cast<PACKED_HIST_ACC_T>(static_cast<HIST_BIN_T>(grad_and_hess >> HIST_BITS_BIN)) << HIST_BITS_ACC) |
            (static_cast<PACKED_HIST_ACC_T>(grad_and_hess & 0x0000ffff))) :
            ((static_cast<PACKED_HIST_ACC_T>(static_cast<HIST_BIN_T>(grad_and_hess >> HIST_BITS_BIN)) << HIST_BITS_ACC) |
            (static_cast<PACKED_HIST_ACC_T>(grad_and_hess & 0x00000000ffffffff)));
          sum_right_gradient_and_hessian += grad_and_hess_acc;
        } else {
          sum_right_gradient_and_hessian += grad_and_hess;
        }
        const uint32_t int_sum_right_hessian = HIST_BITS_ACC == 16 ?
          static_cast<uint32_t>(sum_right_gradient_and_hessian & 0x0000ffff) :
          static_cast<uint32_t>(sum_right_gradient_and_hessian & 0x00000000ffffffff);
        data_size_t right_count = Common::RoundInt(int_sum_right_hessian * cnt_factor);
        double sum_right_hessian = int_sum_right_hessian * hess_scale;
        // if data not enough, or sum hessian too small
        if (right_count < meta_->config->min_data_in_leaf ||
            sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) {
          continue;
        }
        data_size_t left_count = num_data - right_count;
        // if data not enough
        if (left_count < meta_->config->min_data_in_leaf) {
          break;
        }

        const PACKED_HIST_ACC_T sum_left_gradient_and_hessian = local_int_sum_gradient_and_hessian - sum_right_gradient_and_hessian;
        const uint32_t int_sum_left_hessian = HIST_BITS_ACC == 16 ?
          static_cast<uint32_t>(sum_left_gradient_and_hessian & 0x0000ffff) :
          static_cast<uint32_t>(sum_left_gradient_and_hessian & 0x00000000ffffffff);
        double sum_left_hessian = int_sum_left_hessian * hess_scale;
        // if sum hessian too small
        if (sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) {
          break;
        }

        double sum_right_gradient = HIST_BITS_ACC == 16 ?
          static_cast<double>(static_cast<int16_t>(sum_right_gradient_and_hessian >> 16)) * grad_scale :
          static_cast<double>(static_cast<int32_t>(static_cast<int64_t>(sum_right_gradient_and_hessian) >> 32)) * grad_scale;
        double sum_left_gradient = HIST_BITS_ACC == 16 ?
          static_cast<double>(static_cast<int16_t>(sum_left_gradient_and_hessian >> 16)) * grad_scale :
          static_cast<double>(static_cast<int32_t>(static_cast<int64_t>(sum_left_gradient_and_hessian) >> 32)) * grad_scale;
        if (USE_RAND) {
          if (t - 1 + offset != rand_threshold) {
            continue;
          }
        }

        if (USE_MC && constraint_update_necessary) {
          constraints->Update(t + offset);
        }

        // current split gain
        double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian + kEpsilon, sum_right_gradient,
            sum_right_hessian + kEpsilon, meta_->config->lambda_l1,
            meta_->config->lambda_l2, meta_->config->max_delta_step,
            constraints, meta_->monotone_type, meta_->config->path_smooth,
            left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain <= min_gain_shift) {
          continue;
        }

        // mark as able to be split
        is_splittable_ = true;
        // better split point
        if (current_gain > best_gain) {
          if (USE_MC) {
            best_right_constraints = constraints->RightToBasicConstraint();
            best_left_constraints = constraints->LeftToBasicConstraint();
            if (best_right_constraints.min > best_right_constraints.max ||
                best_left_constraints.min > best_left_constraints.max) {
              continue;
            }
          }
          best_sum_left_gradient_and_hessian = sum_left_gradient_and_hessian;
          // left is <= threshold, right is > threshold.  so this is t-1
          best_threshold = static_cast<uint32_t>(t - 1 + offset);
          best_gain = current_gain;
        }
      }
    } else {
      PACKED_HIST_ACC_T sum_left_gradient_and_hessian = 0;

      int t = 0;
      const int t_end = meta_->num_bin - 2 - offset;

      if (NA_AS_MISSING) {
        if (offset == 1) {
          sum_left_gradient_and_hessian = local_int_sum_gradient_and_hessian;
          for (int i = 0; i < meta_->num_bin - offset; ++i) {
            const PACKED_HIST_BIN_T grad_and_hess = data_ptr[i];
            if (HIST_BITS_ACC != HIST_BITS_BIN) {
              const PACKED_HIST_ACC_T grad_and_hess_acc = HIST_BITS_BIN == 16 ?
                ((static_cast<PACKED_HIST_ACC_T>(static_cast<HIST_BIN_T>(grad_and_hess >> HIST_BITS_BIN)) << HIST_BITS_ACC) |
                (static_cast<PACKED_HIST_ACC_T>(grad_and_hess & 0x0000ffff))) :
                ((static_cast<PACKED_HIST_ACC_T>(static_cast<HIST_BIN_T>(grad_and_hess >> HIST_BITS_BIN)) << HIST_BITS_ACC) |
                (static_cast<PACKED_HIST_ACC_T>(grad_and_hess & 0x00000000ffffffff)));
              sum_left_gradient_and_hessian -= grad_and_hess_acc;
            } else {
              sum_left_gradient_and_hessian -= grad_and_hess;
            }
          }
          t = -1;
        }
      }

      for (; t <= t_end; ++t) {
        if (SKIP_DEFAULT_BIN) {
          if ((t + offset) == static_cast<int>(meta_->default_bin)) {
            continue;
          }
        }
        if (t >= 0) {
          const PACKED_HIST_BIN_T grad_and_hess = data_ptr[t];
          if (HIST_BITS_ACC != HIST_BITS_BIN) {
            const PACKED_HIST_ACC_T grad_and_hess_acc = HIST_BITS_BIN == 16 ?
              ((static_cast<PACKED_HIST_ACC_T>(static_cast<HIST_BIN_T>(grad_and_hess >> HIST_BITS_BIN)) << HIST_BITS_ACC) |
              (static_cast<PACKED_HIST_ACC_T>(grad_and_hess & 0x0000ffff))) :
              ((static_cast<PACKED_HIST_ACC_T>(static_cast<HIST_BIN_T>(grad_and_hess >> HIST_BITS_BIN)) << HIST_BITS_ACC) |
              (static_cast<PACKED_HIST_ACC_T>(grad_and_hess & 0x00000000ffffffff)));
            sum_left_gradient_and_hessian += grad_and_hess_acc;
          } else {
            sum_left_gradient_and_hessian += grad_and_hess;
          }
        }
        // if data not enough, or sum hessian too small
        const uint32_t int_sum_left_hessian = HIST_BITS_ACC == 16 ?
          static_cast<uint32_t>(sum_left_gradient_and_hessian & 0x0000ffff) :
          static_cast<uint32_t>(sum_left_gradient_and_hessian & 0x00000000ffffffff);
        const data_size_t left_count = Common::RoundInt(static_cast<double>(int_sum_left_hessian) * cnt_factor);
        const double sum_left_hessian = static_cast<double>(int_sum_left_hessian) * hess_scale;
        if (left_count < meta_->config->min_data_in_leaf ||
            sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) {
          continue;
        }
        data_size_t right_count = num_data - left_count;
        // if data not enough
        if (right_count < meta_->config->min_data_in_leaf) {
          break;
        }

        const PACKED_HIST_ACC_T sum_right_gradient_and_hessian = local_int_sum_gradient_and_hessian - sum_left_gradient_and_hessian;
        const uint32_t int_sum_right_hessian = HIST_BITS_ACC == 16 ?
          static_cast<uint32_t>(sum_right_gradient_and_hessian & 0x0000ffff) :
          static_cast<uint32_t>(sum_right_gradient_and_hessian & 0x00000000ffffffff);
        const double sum_right_hessian = static_cast<double>(int_sum_right_hessian) * hess_scale;
        // if sum Hessian too small
        if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) {
          break;
        }

        double sum_right_gradient = HIST_BITS_ACC == 16 ?
          static_cast<double>(static_cast<int16_t>(sum_right_gradient_and_hessian >> 16)) * grad_scale :
          static_cast<double>(static_cast<int32_t>(static_cast<int64_t>(sum_right_gradient_and_hessian) >> 32)) * grad_scale;
        double sum_left_gradient = HIST_BITS_ACC == 16 ?
          static_cast<double>(static_cast<int16_t>(sum_left_gradient_and_hessian >> 16)) * grad_scale :
          static_cast<double>(static_cast<int32_t>(static_cast<int64_t>(sum_left_gradient_and_hessian) >> 32)) * grad_scale;
        if (USE_RAND) {
          if (t + offset != rand_threshold) {
            continue;
          }
        }
        // current split gain
        double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian + kEpsilon, sum_right_gradient,
            sum_right_hessian + kEpsilon, meta_->config->lambda_l1,
            meta_->config->lambda_l2, meta_->config->max_delta_step,
            constraints, meta_->monotone_type, meta_->config->path_smooth, left_count,
            right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain <= min_gain_shift) {
          continue;
        }

        // mark as able to be split
        is_splittable_ = true;
        // better split point
        if (current_gain > best_gain) {
          if (USE_MC) {
            best_right_constraints = constraints->RightToBasicConstraint();
            best_left_constraints = constraints->LeftToBasicConstraint();
            if (best_right_constraints.min > best_right_constraints.max ||
                best_left_constraints.min > best_left_constraints.max) {
              continue;
            }
          }
          best_sum_left_gradient_and_hessian = sum_left_gradient_and_hessian;
          best_threshold = static_cast<uint32_t>(t + offset);
          best_gain = current_gain;
        }
      }
    }

    if (is_splittable_ && best_gain > output->gain + min_gain_shift) {
      const int32_t int_best_sum_left_gradient = HIST_BITS_ACC == 16 ?
        static_cast<int32_t>(static_cast<int16_t>(best_sum_left_gradient_and_hessian >> 16)) :
        static_cast<int32_t>(static_cast<int64_t>(best_sum_left_gradient_and_hessian) >> 32);
      const uint32_t int_best_sum_left_hessian = HIST_BITS_ACC == 16 ?
        static_cast<uint32_t>(best_sum_left_gradient_and_hessian & 0x0000ffff) :
        static_cast<uint32_t>(best_sum_left_gradient_and_hessian & 0x00000000ffffffff);
      const double best_sum_left_gradient = static_cast<double>(int_best_sum_left_gradient) * grad_scale;
      const double best_sum_left_hessian = static_cast<double>(int_best_sum_left_hessian) * hess_scale;
      const int64_t best_sum_left_gradient_and_hessian_int64 = HIST_BITS_ACC == 16 ?
          ((static_cast<int64_t>(static_cast<int16_t>(best_sum_left_gradient_and_hessian >> 16)) << 32) |
          static_cast<int64_t>(best_sum_left_gradient_and_hessian & 0x0000ffff)) :
          best_sum_left_gradient_and_hessian;
      const int64_t best_sum_right_gradient_and_hessian = int_sum_gradient_and_hessian - best_sum_left_gradient_and_hessian_int64;
      const int32_t int_best_sum_right_gradient = static_cast<int32_t>(best_sum_right_gradient_and_hessian >> 32);
      const uint32_t int_best_sum_right_hessian = static_cast<uint32_t>(best_sum_right_gradient_and_hessian & 0x00000000ffffffff);
      const double best_sum_right_gradient = static_cast<double>(int_best_sum_right_gradient) * grad_scale;
      const double best_sum_right_hessian = static_cast<double>(int_best_sum_right_hessian) * hess_scale;
      const data_size_t best_left_count = Common::RoundInt(static_cast<double>(int_best_sum_left_hessian) * cnt_factor);
      const data_size_t best_right_count = Common::RoundInt(static_cast<double>(int_best_sum_right_hessian) * cnt_factor);
      // update split information
      output->threshold = best_threshold;
      output->left_output =
          CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
              best_sum_left_gradient, best_sum_left_hessian,
              meta_->config->lambda_l1, meta_->config->lambda_l2,
              meta_->config->max_delta_step, best_left_constraints, meta_->config->path_smooth,
              best_left_count, parent_output);
      output->left_count = best_left_count;
      output->left_sum_gradient = best_sum_left_gradient;
      output->left_sum_hessian = best_sum_left_hessian;
      output->left_sum_gradient_and_hessian = best_sum_left_gradient_and_hessian_int64;
      output->right_output =
          CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
              best_sum_right_gradient,
              best_sum_right_hessian, meta_->config->lambda_l1,
              meta_->config->lambda_l2, meta_->config->max_delta_step,
              best_right_constraints, meta_->config->path_smooth, best_right_count,
              parent_output);
      output->right_count = best_right_count;
      output->right_sum_gradient = best_sum_right_gradient;
      output->right_sum_hessian = best_sum_right_hessian;
      output->right_sum_gradient_and_hessian = best_sum_right_gradient_and_hessian;
      output->gain = best_gain - min_gain_shift;
      output->default_left = REVERSE;
    }
  }

  const FeatureMetainfo* meta_;
  /*! \brief sum of gradient of each bin */
  hist_t* data_;
  int16_t* data_int16_;
  bool is_splittable_ = true;

  std::function<void(double, double, data_size_t, const FeatureConstraint*,
                     double, SplitInfo*)>
      find_best_threshold_fun_;

  std::function<void(int64_t, double, double, const uint8_t, const uint8_t, data_size_t, const FeatureConstraint*,
                     double, SplitInfo*)>
      int_find_best_threshold_fun_;
};

class HistogramPool {
 public:
  /*!
   * \brief Constructor
   */
  HistogramPool() {
    cache_size_ = 0;
    total_size_ = 0;
  }

  /*!
   * \brief Destructor
   */
  ~HistogramPool() {}

  /*!
   * \brief Reset pool size
   * \param cache_size Max cache size
   * \param total_size Total size will be used
   */
  void Reset(int cache_size, int total_size) {
    cache_size_ = cache_size;
    // at least need 2 bucket to store smaller leaf and larger leaf
    CHECK_GE(cache_size_, 2);
    total_size_ = total_size;
    if (cache_size_ > total_size_) {
      cache_size_ = total_size_;
    }
    is_enough_ = (cache_size_ == total_size_);
    if (!is_enough_) {
      mapper_.resize(total_size_);
      inverse_mapper_.resize(cache_size_);
      last_used_time_.resize(cache_size_);
      ResetMap();
    }
  }

  /*!
   * \brief Reset mapper
   */
  void ResetMap() {
    if (!is_enough_) {
      cur_time_ = 0;
      std::fill(mapper_.begin(), mapper_.end(), -1);
      std::fill(inverse_mapper_.begin(), inverse_mapper_.end(), -1);
      std::fill(last_used_time_.begin(), last_used_time_.end(), 0);
    }
  }
  template <bool USE_DATA, bool USE_CONFIG>
  static void SetFeatureInfo(const Dataset* train_data, const Config* config,
                             std::vector<FeatureMetainfo>* feature_meta) {
    auto& ref_feature_meta = *feature_meta;
    const int num_feature = train_data->num_features();
    ref_feature_meta.resize(num_feature);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_feature >= 1024)
    for (int i = 0; i < num_feature; ++i) {
      if (USE_DATA) {
        ref_feature_meta[i].num_bin = train_data->FeatureNumBin(i);
        ref_feature_meta[i].default_bin =
            train_data->FeatureBinMapper(i)->GetDefaultBin();
        ref_feature_meta[i].missing_type =
            train_data->FeatureBinMapper(i)->missing_type();
        if (train_data->FeatureBinMapper(i)->GetMostFreqBin() == 0) {
          ref_feature_meta[i].offset = 1;
        } else {
          ref_feature_meta[i].offset = 0;
        }
        ref_feature_meta[i].bin_type =
            train_data->FeatureBinMapper(i)->bin_type();
      }
      if (USE_CONFIG) {
        const int real_fidx = train_data->RealFeatureIndex(i);
        if (!config->monotone_constraints.empty()) {
          ref_feature_meta[i].monotone_type =
              config->monotone_constraints[real_fidx];
        } else {
          ref_feature_meta[i].monotone_type = 0;
        }
        if (!config->feature_contri.empty()) {
          ref_feature_meta[i].penalty = config->feature_contri[real_fidx];
        } else {
          ref_feature_meta[i].penalty = 1.0;
        }
        ref_feature_meta[i].rand = Random(config->extra_seed + i);
      }
      ref_feature_meta[i].config = config;
    }
  }

  void DynamicChangeSize(const Dataset* train_data, int num_total_bin,
                        const std::vector<uint32_t>& offsets, const Config* config,
                        int cache_size, int total_size) {
    if (feature_metas_.empty()) {
      SetFeatureInfo<true, true>(train_data, config, &feature_metas_);
      uint64_t bin_cnt_over_features = 0;
      for (int i = 0; i < train_data->num_features(); ++i) {
        bin_cnt_over_features +=
            static_cast<uint64_t>(feature_metas_[i].num_bin);
      }
      Log::Info("Total Bins %d", bin_cnt_over_features);
    }
    int old_cache_size = static_cast<int>(pool_.size());
    Reset(cache_size, total_size);

    if (cache_size > old_cache_size) {
      pool_.resize(cache_size);
      data_.resize(cache_size);
    }

    if (config->use_quantized_grad) {
      OMP_INIT_EX();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int i = old_cache_size; i < cache_size; ++i) {
        OMP_LOOP_EX_BEGIN();
        pool_[i].reset(new FeatureHistogram[train_data->num_features()]);
        data_[i].resize(num_total_bin);
        for (int j = 0; j < train_data->num_features(); ++j) {
          int16_t* data_ptr = reinterpret_cast<int16_t*>(data_[i].data());
          pool_[i][j].Init(data_[i].data() + offsets[j], data_ptr + 2 * offsets[j], &feature_metas_[j]);
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    } else {
      OMP_INIT_EX();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int i = old_cache_size; i < cache_size; ++i) {
        OMP_LOOP_EX_BEGIN();
        pool_[i].reset(new FeatureHistogram[train_data->num_features()]);
        data_[i].resize(num_total_bin * 2);
        for (int j = 0; j < train_data->num_features(); ++j) {
          pool_[i][j].Init(data_[i].data() + offsets[j] * 2, &feature_metas_[j]);
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    }
  }

  void ResetConfig(const Dataset* train_data, const Config* config) {
    CHECK_GT(train_data->num_features(), 0);
    const Config* old_config = feature_metas_[0].config;
    SetFeatureInfo<false, true>(train_data, config, &feature_metas_);
    // if need to reset the function pointers
    if (old_config->lambda_l1 != config->lambda_l1 ||
        old_config->monotone_constraints != config->monotone_constraints ||
        old_config->extra_trees != config->extra_trees ||
        old_config->max_delta_step != config->max_delta_step ||
        old_config->path_smooth != config->path_smooth) {
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int i = 0; i < cache_size_; ++i) {
        for (int j = 0; j < train_data->num_features(); ++j) {
          pool_[i][j].ResetFunc();
        }
      }
    }
  }

  /*!
   * \brief Get data for the specific index
   * \param idx which index want to get
   * \param out output data will store into this
   * \return True if this index is in the pool, False if this index is not in
   * the pool
   */
  bool Get(int idx, FeatureHistogram** out) {
    if (is_enough_) {
      *out = pool_[idx].get();
      return true;
    } else if (mapper_[idx] >= 0) {
      int slot = mapper_[idx];
      *out = pool_[slot].get();
      last_used_time_[slot] = ++cur_time_;
      return true;
    } else {
      // choose the least used slot
      int slot = static_cast<int>(ArrayArgs<int>::ArgMin(last_used_time_));
      *out = pool_[slot].get();
      last_used_time_[slot] = ++cur_time_;

      // reset previous mapper
      if (inverse_mapper_[slot] >= 0) mapper_[inverse_mapper_[slot]] = -1;

      // update current mapper
      mapper_[idx] = slot;
      inverse_mapper_[slot] = idx;
      return false;
    }
  }

  /*!
   * \brief Move data from one index to another index
   * \param src_idx
   * \param dst_idx
   */
  void Move(int src_idx, int dst_idx) {
    if (is_enough_) {
      std::swap(pool_[src_idx], pool_[dst_idx]);
      return;
    }
    if (mapper_[src_idx] < 0) {
      return;
    }
    // get slot of src idx
    int slot = mapper_[src_idx];
    // reset src_idx
    mapper_[src_idx] = -1;

    // move to dst idx
    mapper_[dst_idx] = slot;
    last_used_time_[slot] = ++cur_time_;
    inverse_mapper_[slot] = dst_idx;
  }

 private:
  std::vector<std::unique_ptr<FeatureHistogram[]>> pool_;
  std::vector<
      std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>>
      data_;
  std::vector<FeatureMetainfo> feature_metas_;
  int cache_size_;
  int total_size_;
  bool is_enough_ = false;
  std::vector<int> mapper_;
  std::vector<int> inverse_mapper_;
  std::vector<int> last_used_time_;
  int cur_time_ = 0;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_TREELEARNER_FEATURE_HISTOGRAM_HPP_
