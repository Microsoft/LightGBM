/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_OBJECTIVE_FUNCTION_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_OBJECTIVE_FUNCTION_H_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>

#include <string>
#include <functional>

namespace LightGBM {
/*!
* \brief The interface of Objective Function.
*/
class ObjectiveFunction {
 public:
  /*! \brief virtual destructor */
  virtual ~ObjectiveFunction() {}

  /*!
  * \brief Initialize
  * \param metadata Label data
  * \param num_data Number of data
  */
  virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;

  /*!
  * \brief calculating first order derivative of loss function
  * \param score prediction score in this round
  * \gradients Output gradients
  * \hessians Output hessians
  */
  virtual void GetGradients(const double* score,
    score_t* gradients, score_t* hessians) const = 0;

    /*!
  * \brief calculating first order derivative of loss function, used only for bagging by query in lambdarank
  * \param score prediction score in this round
  * \param num_sampled_queries number of in-bag queries
  * \param sampled_query_indices indices of in-bag queries
  * \gradients Output gradients
  * \hessians Output hessians
  */
  virtual void GetGradientsWithSampledQueries(const double* score, const data_size_t /*num_sampled_queries*/, const data_size_t* /*sampled_query_indices*/,
    score_t* gradients, score_t* hessians) const { GetGradients(score, gradients, hessians); }

  virtual const char* GetName() const = 0;

  virtual bool IsConstantHessian() const { return false; }

  virtual bool IsRenewTreeOutput() const { return false; }

  virtual double RenewTreeOutput(double ori_output, std::function<double(const label_t*, int)>,
                                 const data_size_t*,
                                 const data_size_t*,
                                 data_size_t) const { return ori_output; }

  virtual void RenewTreeOutputCUDA(const double* /*score*/, const data_size_t* /*data_indices_in_leaf*/, const data_size_t* /*num_data_in_leaf*/,
    const data_size_t* /*data_start_in_leaf*/, const int /*num_leaves*/, double* /*leaf_value*/) const {}

  virtual double BoostFromScore(int /*class_id*/) const { return 0.0; }

  virtual bool ClassNeedTrain(int /*class_id*/) const { return true; }

  virtual bool SkipEmptyClass() const { return false; }

  virtual int NumModelPerIteration() const { return 1; }

  virtual int NumPredictOneRow() const { return 1; }

  /*! \brief The prediction should be accurate or not. True will disable early stopping for prediction. */
  virtual bool NeedAccuratePrediction() const { return true; }

  /*! \brief Return the number of positive samples. Return 0 if no binary classification tasks.*/
  virtual data_size_t NumPositiveData() const { return 0; }

  virtual void ConvertOutput(const double* input, double* output) const {
    output[0] = input[0];
  }

  virtual std::string ToString() const = 0;

  ObjectiveFunction() = default;
  /*! \brief Disable copy */
  ObjectiveFunction& operator=(const ObjectiveFunction&) = delete;
  /*! \brief Disable copy */
  ObjectiveFunction(const ObjectiveFunction&) = delete;

  /*!
  * \brief Create object of objective function
  * \param type Specific type of objective function
  * \param config Config for objective function
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& type,
    const Config& config);

  /*!
  * \brief Load objective function from string object
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& str);

  /*!
  * \brief Whether boosting is done on CUDA
  */
  virtual bool IsCUDAObjective() const { return false; }

  #ifdef USE_CUDA
  /*!
  * \brief Convert output for CUDA version
  */
  virtual const double* ConvertOutputCUDA(data_size_t /*num_data*/, const double* input, double* /*output*/) const {
    return input;
  }

  virtual bool NeedConvertOutputCUDA () const { return false; }

  virtual void SetNCCLInfo(
    ncclComm_t /*nccl_communicator*/,
    int /*nccl_gpu_rank*/,
    int /*local_gpu_rank*/,
    int /*gpu_device_id*/,
    data_size_t /*global_num_data*/) {}

  /*!
  * \brief Create object of objective function on CUDA
  * \param type Specific type of objective function
  * \param config Config for objective function
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunctionCUDA(const std::string& type,
    const Config& config);

  #endif  // USE_CUDA

  virtual void SetDataIndices(const data_size_t* used_data_indices) const { used_data_indices_ = used_data_indices; }

 private:
  mutable const data_size_t* used_data_indices_ = nullptr;
};

void UpdatePointwiseScoresForOneQuery(double* score_pointwise, const double* score_pairwise, data_size_t cnt_pointwise,
  const std::pair<data_size_t, data_size_t>* paired_index_map,  
  const std::vector<std::vector<std::pair<short, data_size_t>>>& right2left2pair_map, const std::vector<std::vector<std::pair<short, data_size_t>>>& left2right2pair_map,
  int truncation_level, double sigma, const CommonC::SigmoidCache& sigmoid_cache, bool model_indirect_comparison, bool model_conditional_rel,
  bool indirect_comparison_above_only, bool logarithmic_discounts, bool hard_pairwise_preference, int indirect_comparison_max_rank, double indirect_comparison_weight, double l2_pairwise_diff_weight);

inline data_size_t get_pair_index(const std::vector<std::pair<short, data_size_t>>& vec, short key) noexcept
{
  constexpr size_t LINEAR_THRESHOLD = 20; // tune experimentally

  if (vec.size() < LINEAR_THRESHOLD) {
    // Linear scan with branchless equality
    for (const auto& p : vec) {
      // (p.first == key) evaluates to 1 or 0 → mask result
      short match = (p.first == key);
      if (match) return p.second;
    }
    return static_cast<short>(-1);
  }

  // Binary search for larger vectors
  auto it = std::lower_bound(
    vec.begin(), vec.end(),
    key,
    [](const auto& p, short k) noexcept { return p.first < k; }
  );

  // Branchless equality for final check
  return (it != vec.end() && it->first == key)
    ? it->second
    : static_cast<data_size_t>(-1);
}

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_OBJECTIVE_FUNCTION_H_
