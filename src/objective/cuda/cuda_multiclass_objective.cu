/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <algorithm>

#include "cuda_multiclass_objective.hpp"

namespace LightGBM {

__device__ void SoftmaxCUDA(double* softmax_buffer, int len) {
  double wmax = softmax_buffer[0];
  for (int i = 1; i < len; ++i) {
    wmax = max(softmax_buffer[i], wmax);
  }
  double wsum = 0.0f;
  for (int i = 0; i < len; ++i) {
    softmax_buffer[i] = exp(softmax_buffer[i] - wmax);
    wsum += softmax_buffer[i];
  }
  for (int i = 0; i < len; ++i) {
    softmax_buffer[i] /= static_cast<double>(wsum);
  }
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_MulticlassSoftmax(
  const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights,
  const double factor, const int num_class, const data_size_t num_data,
  double* cuda_softmax_buffer, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const data_size_t offset = data_index * num_class;
    double* softmax_result = cuda_softmax_buffer + offset;
    for (int k = 0; k < num_class; ++k) {
      softmax_result[k] = cuda_scores[k * num_data + data_index];
    }
    SoftmaxCUDA(softmax_result, num_class);
    if (!USE_WEIGHT) {
      for (int k = 0; k < num_class; ++k) {
        const double p = softmax_result[k];
        size_t idx = static_cast<size_t>(num_data) * k + data_index;
        if (static_cast<int>(cuda_labels[data_index]) == k) {
          cuda_out_gradients[idx] = static_cast<score_t>(p - 1.0f);
        } else {
          cuda_out_gradients[idx] = static_cast<score_t>(p);
        }
        cuda_out_hessians[idx] = static_cast<score_t>(factor * p * (1.0f - p));
      }
    } else {
      for (int k = 0; k < num_class; ++k) {
        const double p = softmax_result[k];
        const double weight = cuda_weights[data_index];
        size_t idx = static_cast<size_t>(num_data) * k + data_index;
        if (static_cast<int>(cuda_labels[data_index]) == k) {
          cuda_out_gradients[idx] = static_cast<score_t>((p - 1.0f) * weight);
        } else {
          cuda_out_gradients[idx] = static_cast<score_t>(p * weight);
        }
        cuda_out_hessians[idx] = static_cast<score_t>((factor * p * (1.0f - p)) * weight);
      }
    }
  }
}

void CUDAMulticlassSoftmax::LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_MULTICLASS - 1) / GET_GRADIENTS_BLOCK_SIZE_MULTICLASS;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_MulticlassSoftmax<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
      scores, cuda_labels_, cuda_weights_, factor_, num_class_, num_data_,
      cuda_softmax_buffer_.RawData(), gradients, hessians);
  } else {
    GetGradientsKernel_MulticlassSoftmax<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
      scores, cuda_labels_, cuda_weights_, factor_, num_class_, num_data_,
      cuda_softmax_buffer_.RawData(), gradients, hessians);
  }
}

__global__ void ConvertOutputCUDAKernel_MulticlassSoftmax(
  const int num_class, const data_size_t num_data, const double* input, double* cuda_softmax_buffer, double* output) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const data_size_t offset = data_index * num_class;
    double* cuda_softmax_buffer_ptr = cuda_softmax_buffer + offset;
    for (int class_index = 0; class_index < num_class; ++class_index) {
      cuda_softmax_buffer_ptr[class_index] = input[class_index * num_data + data_index];
    }
    SoftmaxCUDA(cuda_softmax_buffer_ptr, num_class);
    for (int class_index = 0; class_index < num_class; ++class_index) {
      output[class_index * num_data + data_index] = cuda_softmax_buffer_ptr[class_index];
    }
  }
}

const double* CUDAMulticlassSoftmax::LaunchConvertOutputCUDAKernel(
  const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_MULTICLASS - 1) / GET_GRADIENTS_BLOCK_SIZE_MULTICLASS;
  ConvertOutputCUDAKernel_MulticlassSoftmax<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
    num_class_, num_data, input, cuda_softmax_buffer_.RawData(), output);
  return output;
}

}  // namespace LightGBM

#endif  // USE_CUDA
