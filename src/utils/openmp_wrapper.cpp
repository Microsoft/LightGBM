/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/utils/openmp_wrapper.h>

int LGBM_MAX_NUM_THREADS = -1;

int LGBM_DEFAULT_NUM_THREADS = -1;

#ifdef _OPENMP

#include <omp.h>

int OMP_NUM_THREADS() {
  int default_num_threads = 1;

  if (LGBM_DEFAULT_NUM_THREADS > 0) {
    // if LightGBM-specific default has been set, ignore OpenMP-global config
    default_num_threads = LGBM_DEFAULT_NUM_THREADS;
  } else {
    // otherwise, default to OpenMP-global config
    default_num_threads = omp_get_max_threads();
  }

  // ensure that if LGBM_SetMaxThreads() was ever called, LightGBM doesn't
  // use more than that many threads
  if (LGBM_MAX_NUM_THREADS > 0 && default_num_threads > LGBM_MAX_NUM_THREADS) {
    return LGBM_MAX_NUM_THREADS;
  }

  return default_num_threads;
}

void OMP_SET_NUM_THREADS(int num_threads) {
  if (num_threads <= 0) {
    LGBM_DEFAULT_NUM_THREADS = -1;
  } else {
    LGBM_DEFAULT_NUM_THREADS = num_threads;
  }
}

#endif  // _OPENMP
