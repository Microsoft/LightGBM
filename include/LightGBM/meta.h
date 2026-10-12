/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_META_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_META_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_AMD64))) || defined(__INTEL_COMPILER) || MM_PREFETCH
  #include <xmmintrin.h>
  #define PREFETCH_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(__GNUC__)
  #define PREFETCH_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else
  #define PREFETCH_T0(addr) do {} while (0)
#endif

namespace LightGBM {

/*! \brief Type of data size, it is better to use signed type*/
typedef int32_t data_size_t;

// Enable following macro to use double for score_t
// #define SCORE_T_USE_DOUBLE

// Enable following macro to use double for label_t
// #define LABEL_T_USE_DOUBLE

/*! \brief Type of score, and gradients */
#ifdef SCORE_T_USE_DOUBLE
typedef double score_t;
#else
typedef float score_t;
#endif

/*! \brief Type of metadata, include weight and label */
#ifdef LABEL_T_USE_DOUBLE
typedef double label_t;
#else
typedef float label_t;
#endif

const score_t kMinScore = -std::numeric_limits<score_t>::infinity();

const score_t kMaxScore = std::numeric_limits<score_t>::infinity();

const score_t kEpsilon = 1e-15f;

const double kZeroThreshold = 1e-35f;


typedef int32_t comm_size_t;

using PredictFunction =
std::function<void(const std::vector<std::pair<int, double>>&, double* output)>;

using PredictSparseFunction =
std::function<void(const std::vector<std::pair<int, double>>&, std::vector<std::unordered_map<int, double>>* output)>;

typedef void(*ReduceFunction)(const char* input, char* output, int type_size, comm_size_t array_size);


typedef void(*ReduceScatterFunction)(char* input, comm_size_t input_size, int type_size,
                                     const comm_size_t* block_start, const comm_size_t* block_len, int num_block, char* output, comm_size_t output_size,
                                     const ReduceFunction& reducer);

typedef void(*AllgatherFunction)(char* input, comm_size_t input_size, const comm_size_t* block_start,
                                 const comm_size_t* block_len, int num_block, char* output, comm_size_t output_size);


#define NO_SPECIFIC (-1)

const int kAlignedSize = 32;

#define SIZE_ALIGNED(t) ((t) + kAlignedSize - 1) / kAlignedSize * kAlignedSize

// Refer to https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-4-c4127?view=vs-2019
#ifdef _MSC_VER
#pragma warning(disable : 4127)
#endif

}  // namespace LightGBM

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_META_H_
