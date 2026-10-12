/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/sample_strategy.h>

#include <string>

#include "goss.hpp"
#include "bagging.hpp"

namespace LightGBM {

SampleStrategy* SampleStrategy::CreateSampleStrategy(
  const Config* config,
  const Dataset* train_data,
  const ObjectiveFunction* objective_function,
  int num_tree_per_iteration) {
  if (config->data_sample_strategy == std::string("goss")) {
    return new GOSSStrategy(config, train_data, num_tree_per_iteration);
  } else {
    return new BaggingSampleStrategy(config, train_data, objective_function, num_tree_per_iteration);
  }
}

}  // namespace LightGBM
