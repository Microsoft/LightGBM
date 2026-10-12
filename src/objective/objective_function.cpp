/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/objective_function.h>

#include <string>

#include "binary_objective.hpp"
#include "multiclass_objective.hpp"
#include "rank_objective.hpp"
#include "regression_objective.hpp"
#include "xentropy_objective.hpp"

#include <LightGBM/cuda/cuda_objective_function.hpp>

#include "cuda/cuda_binary_objective.hpp"
#include "cuda/cuda_multiclass_objective.hpp"
#include "cuda/cuda_rank_objective.hpp"
#include "cuda/cuda_regression_objective.hpp"

namespace LightGBM {

#ifdef USE_CUDA
ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunctionCUDA(const std::string& type, const Config& config) {
  if (type == std::string("regression")) {
    return new CUDARegressionL2loss(config);
  } else if (type == std::string("regression_l1")) {
    return new CUDARegressionL1loss(config);
  } else if (type == std::string("quantile")) {
    return new CUDARegressionQuantileloss(config);
  } else if (type == std::string("huber")) {
    return new CUDARegressionHuberLoss(config);
  } else if (type == std::string("fair")) {
    return new CUDARegressionFairLoss(config);
  } else if (type == std::string("poisson")) {
    return new CUDARegressionPoissonLoss(config);
  } else if (type == std::string("binary")) {
    return new CUDABinaryLogloss(config);
  } else if (type == std::string("lambdarank")) {
    return new CUDALambdarankNDCG(config);
  } else if (type == std::string("rank_xendcg")) {
    return new CUDARankXENDCG(config);
  } else if (type == std::string("multiclass")) {
    return new CUDAMulticlassSoftmax(config);
  } else if (type == std::string("multiclassova")) {
    return new CUDAMulticlassOVA(config);
  } else if (type == std::string("cross_entropy")) {
    Log::Warning("Objective cross_entropy is not implemented in cuda version. Fall back to boosting on CPU.");
    return new CrossEntropy(config);
  } else if (type == std::string("cross_entropy_lambda")) {
    Log::Warning("Objective cross_entropy_lambda is not implemented in cuda version. Fall back to boosting on CPU.");
    return new CrossEntropyLambda(config);
  } else if (type == std::string("mape")) {
    Log::Warning("Objective mape is not implemented in cuda version. Fall back to boosting on CPU.");
    return new RegressionMAPELOSS(config);
  } else if (type == std::string("gamma")) {
    Log::Warning("Objective gamma is not implemented in cuda version. Fall back to boosting on CPU.");
    return new RegressionGammaLoss(config);
  } else if (type == std::string("tweedie")) {
    Log::Warning("Objective tweedie is not implemented in cuda version. Fall back to boosting on CPU.");
    return new RegressionTweedieLoss(config);
  } else if (type == std::string("custom")) {
    Log::Warning("Using customized objective with cuda. This requires copying gradients from CPU to GPU, which can be slow.");
    return nullptr;
  }
}
#endif  // USE_CUDA

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const Config& config) {
  #ifdef USE_CUDA
  if (config.device_type == std::string("cuda") &&
      config.data_sample_strategy != std::string("goss") &&
      config.boosting != std::string("rf")) {
    return CreateObjectiveFunctionCUDA(type, config);
  } else {
  #endif  // USE_CUDA
    if (type == std::string("regression")) {
      return new RegressionL2loss(config);
    } else if (type == std::string("regression_l1")) {
      return new RegressionL1loss(config);
    } else if (type == std::string("quantile")) {
      return new RegressionQuantileloss(config);
    } else if (type == std::string("huber")) {
      return new RegressionHuberLoss(config);
    } else if (type == std::string("fair")) {
      return new RegressionFairLoss(config);
    } else if (type == std::string("poisson")) {
      return new RegressionPoissonLoss(config);
    } else if (type == std::string("binary")) {
      return new BinaryLogloss(config);
    } else if (type == std::string("lambdarank")) {
      return new LambdarankNDCG(config);
    } else if (type == std::string("rank_xendcg")) {
      return new RankXENDCG(config);
    } else if (type == std::string("multiclass")) {
      return new MulticlassSoftmax(config);
    } else if (type == std::string("multiclassova")) {
      return new MulticlassOVA(config);
    } else if (type == std::string("cross_entropy")) {
      return new CrossEntropy(config);
    } else if (type == std::string("cross_entropy_lambda")) {
      return new CrossEntropyLambda(config);
    } else if (type == std::string("mape")) {
      return new RegressionMAPELOSS(config);
    } else if (type == std::string("gamma")) {
      return new RegressionGammaLoss(config);
    } else if (type == std::string("tweedie")) {
      return new RegressionTweedieLoss(config);
    } else if (type == std::string("custom")) {
      return nullptr;
    }
  #ifdef USE_CUDA
  }
  #endif  // USE_CUDA
  Log::Fatal("Unknown objective type name: %s", type.c_str());
  return nullptr;
}

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& str) {
  auto strs = Common::Split(str.c_str(), ' ');
  auto type = strs[0];
  if (type == std::string("regression")) {
    return new RegressionL2loss(strs);
  } else if (type == std::string("regression_l1")) {
    return new RegressionL1loss(strs);
  } else if (type == std::string("quantile")) {
    return new RegressionQuantileloss(strs);
  } else if (type == std::string("huber")) {
    return new RegressionHuberLoss(strs);
  } else if (type == std::string("fair")) {
    return new RegressionFairLoss(strs);
  } else if (type == std::string("poisson")) {
    return new RegressionPoissonLoss(strs);
  } else if (type == std::string("binary")) {
    return new BinaryLogloss(strs);
  } else if (type == std::string("lambdarank")) {
    return new LambdarankNDCG(strs);
  } else if (type == std::string("rank_xendcg")) {
    return new RankXENDCG(strs);
  } else if (type == std::string("multiclass")) {
    return new MulticlassSoftmax(strs);
  } else if (type == std::string("multiclassova")) {
    return new MulticlassOVA(strs);
  } else if (type == std::string("cross_entropy")) {
    return new CrossEntropy(strs);
  } else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambda(strs);
  } else if (type == std::string("mape")) {
    return new RegressionMAPELOSS(strs);
  } else if (type == std::string("gamma")) {
    return new RegressionGammaLoss(strs);
  } else if (type == std::string("tweedie")) {
    return new RegressionTweedieLoss(strs);
  } else if (type == std::string("custom")) {
    return nullptr;
  }
  Log::Fatal("Unknown objective type name: %s", type.c_str());
  return nullptr;
}

}  // namespace LightGBM
