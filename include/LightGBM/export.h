/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_EXPORT_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_EXPORT_H_

/**
 * Macros for exporting symbols.
 *
 * On Windows (MSVC and MinGW), only symbols marked __declspec(dllexport) are exported.
 * MinGW otherwise exports every symbol when nothing is marked.
 *
 * On other platforms, everything else is hidden by building with -fvisibility=hidden
 * and a linker allowlist (see CMakeLists.txt).
 *
 * The R package registers its routines with R, so it exports nothing but its
 * initialization function, R_init_lightgbm().
 **/

#ifdef __cplusplus
#define LIGHTGBM_EXTERN_C extern "C"
#else
#define LIGHTGBM_EXTERN_C
#endif


#if defined(LGB_R_BUILD)
  // the R package only exports R_init_lightgbm() (see R-package/src/lightgbm_R.cpp)
  #define LIGHTGBM_C_EXPORT  LIGHTGBM_EXTERN_C
#elif defined(_WIN32)
  #define LIGHTGBM_C_EXPORT  LIGHTGBM_EXTERN_C __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
  #define LIGHTGBM_C_EXPORT  LIGHTGBM_EXTERN_C __attribute__((visibility("default")))
#else
  #define LIGHTGBM_C_EXPORT  LIGHTGBM_EXTERN_C
#endif

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_EXPORT_H_
