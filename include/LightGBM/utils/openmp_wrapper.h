/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_OPENMP_WRAPPER_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_OPENMP_WRAPPER_H_

#include <LightGBM/export.h>

// this can only be changed by LGBM_SetMaxThreads()
LIGHTGBM_EXTERN_C int LGBM_MAX_NUM_THREADS;

// this is modified by OMP_SET_NUM_THREADS(), for example
// by passing num_thread through params
LIGHTGBM_EXTERN_C int LGBM_DEFAULT_NUM_THREADS;

#ifdef _OPENMP

#include <LightGBM/utils/log.h>

#include <omp.h>

#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

/*
    Get number of threads to use in OpenMP parallel regions.

    By default, this will return the result of omp_get_max_threads(),
    which is OpenMP-implementation dependent but generally can be controlled
    by environment variable OMP_NUM_THREADS.

    ref:
      - https://www.openmp.org/spec-html/5.0/openmpsu112.html
      - https://gcc.gnu.org/onlinedocs/libgomp/omp_005fget_005fmax_005fthreads.html
*/
LIGHTGBM_EXTERN_C int OMP_NUM_THREADS();

/*
    Update the default number of threads that'll be used in OpenMP parallel
    regions for LightGBM routines where the number of threads aren't directly
    supplied.
*/
LIGHTGBM_EXTERN_C void OMP_SET_NUM_THREADS(int num_threads);

class ThreadExceptionHelper {
 public:
  ThreadExceptionHelper() {
    ex_ptr_ = nullptr;
  }

  ~ThreadExceptionHelper() {
    ReThrow();
  }
  void ReThrow() {
    if (ex_ptr_ != nullptr) {
      std::rethrow_exception(ex_ptr_);
    }
  }
  void CaptureException() {
    // only catch first exception.
    if (ex_ptr_ != nullptr) {
      return;
    }
    std::unique_lock<std::mutex> guard(lock_);
    if (ex_ptr_ != nullptr) {
      return;
    }
    ex_ptr_ = std::current_exception();
  }

 private:
  std::exception_ptr ex_ptr_;
  std::mutex lock_;
};

#define OMP_INIT_EX() ThreadExceptionHelper omp_except_helper
#define OMP_LOOP_EX_BEGIN() try {
#define OMP_LOOP_EX_END()                 \
  }                                       \
  catch (std::exception & ex) {           \
    Log::Warning(ex.what());              \
    omp_except_helper.CaptureException(); \
  }                                       \
  catch (...) {                           \
    omp_except_helper.CaptureException(); \
  }
#define OMP_THROW_EX() omp_except_helper.ReThrow()

#else

/*
 * To be compatible with OpenMP, define a nothrow macro which is used by gcc
 * openmp, but not by clang.
 * See also https://github.com/dmlc/dmlc-core/blob/3106c1cbdcc9fc9ef3a2c1d2196a7a6f6616c13d/include/dmlc/omp.h#L14
 */
#if defined(__clang__)
#undef __GOMP_NOTHROW
#define __GOMP_NOTHROW
#elif defined(__cplusplus)
#undef __GOMP_NOTHROW
#define __GOMP_NOTHROW throw()
#else
#undef __GOMP_NOTHROW
#define __GOMP_NOTHROW __attribute__((__nothrow__))
#endif

#ifdef _MSC_VER
  #pragma warning(disable : 4068)  // disable unknown pragma warning
#endif

#ifdef __cplusplus
  extern "C" {
#endif
  /** Fall here if no OPENMP support, so just
      simulate a single thread running.
      All #pragma omp should be ignored by the compiler **/
  inline void OMP_SET_NUM_THREADS(int) __GOMP_NOTHROW {}
  inline int omp_get_thread_num() __GOMP_NOTHROW {return 0;}
  inline int OMP_NUM_THREADS() __GOMP_NOTHROW { return 1; }
#ifdef __cplusplus
}  // extern "C"
#endif

#define OMP_INIT_EX()
#define OMP_LOOP_EX_BEGIN()
#define OMP_LOOP_EX_END()
#define OMP_THROW_EX()

#endif

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_OPENMP_WRAPPER_H_
