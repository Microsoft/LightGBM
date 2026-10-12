/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_regression_objective.hpp"

#include <string>
#include <vector>

namespace LightGBM {

CUDARegressionL2loss::CUDARegressionL2loss(const Config& config):
CUDARegressionObjectiveInterface<RegressionL2loss>(config) {}

CUDARegressionL2loss::CUDARegressionL2loss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionL2loss>(strs) {}

CUDARegressionL2loss::~CUDARegressionL2loss() {}

void CUDARegressionL2loss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionObjectiveInterface<RegressionL2loss>::Init(metadata, num_data);
}

CUDARegressionL1loss::CUDARegressionL1loss(const Config& config):
CUDARegressionObjectiveInterface<RegressionL1loss>(config) {}

CUDARegressionL1loss::CUDARegressionL1loss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionL1loss>(strs) {}

CUDARegressionL1loss::~CUDARegressionL1loss() {}

void CUDARegressionL1loss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionObjectiveInterface<RegressionL1loss>::Init(metadata, num_data);
  cuda_data_indices_buffer_.Resize(static_cast<size_t>(num_data));
  cuda_percentile_result_.Resize(1);
  if (cuda_weights_ != nullptr) {
    const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION + 1;
    cuda_weights_prefix_sum_.Resize(static_cast<size_t>(num_data));
    cuda_weights_prefix_sum_buffer_.Resize(static_cast<size_t>(num_blocks));
    cuda_weight_by_leaf_buffer_.Resize(static_cast<size_t>(num_data));
  }
  cuda_residual_buffer_.Resize(static_cast<size_t>(num_data));
}


CUDARegressionHuberLoss::CUDARegressionHuberLoss(const Config& config):
CUDARegressionObjectiveInterface<RegressionHuberLoss>(config) {}

CUDARegressionHuberLoss::CUDARegressionHuberLoss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionHuberLoss>(strs) {}

CUDARegressionHuberLoss::~CUDARegressionHuberLoss() {}


CUDARegressionFairLoss::CUDARegressionFairLoss(const Config& config):
CUDARegressionObjectiveInterface<RegressionFairLoss>(config) {}

CUDARegressionFairLoss::CUDARegressionFairLoss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionFairLoss>(strs) {}

CUDARegressionFairLoss::~CUDARegressionFairLoss() {}


CUDARegressionPoissonLoss::CUDARegressionPoissonLoss(const Config& config):
CUDARegressionObjectiveInterface<RegressionPoissonLoss>(config) {}

CUDARegressionPoissonLoss::CUDARegressionPoissonLoss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionPoissonLoss>(strs) {}

CUDARegressionPoissonLoss::~CUDARegressionPoissonLoss() {}

void CUDARegressionPoissonLoss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionObjectiveInterface<RegressionPoissonLoss>::Init(metadata, num_data);
  LaunchCheckLabelKernel();
}

double CUDARegressionPoissonLoss::LaunchCalcInitScoreKernel(const int class_id) const {
  return Common::SafeLog(CUDARegressionObjectiveInterface<RegressionPoissonLoss>::LaunchCalcInitScoreKernel(class_id));
}


CUDARegressionQuantileloss::CUDARegressionQuantileloss(const Config& config):
CUDARegressionObjectiveInterface<RegressionQuantileloss>(config) {}

CUDARegressionQuantileloss::CUDARegressionQuantileloss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionQuantileloss>(strs) {}

CUDARegressionQuantileloss::~CUDARegressionQuantileloss() {}

void CUDARegressionQuantileloss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionObjectiveInterface<RegressionQuantileloss>::Init(metadata, num_data);
  cuda_data_indices_buffer_.Resize(static_cast<size_t>(num_data));
  cuda_percentile_result_.Resize(1);
  if (cuda_weights_ != nullptr) {
    const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION + 1;
    cuda_weights_prefix_sum_.Resize(static_cast<size_t>(num_data));
    cuda_weights_prefix_sum_buffer_.Resize(static_cast<size_t>(num_blocks));
    cuda_weight_by_leaf_buffer_.Resize(static_cast<size_t>(num_data));
  }
  cuda_residual_buffer_.Resize(static_cast<size_t>(num_data));
}

}  // namespace LightGBM

#endif  // USE_CUDA
