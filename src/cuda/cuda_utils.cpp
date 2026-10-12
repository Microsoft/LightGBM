/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_rocm_interop.h>
#include <LightGBM/cuda/cuda_utils.hu>

namespace LightGBM {

void SynchronizeCUDADevice(const char* file, const int line) {
  gpuAssert(cudaDeviceSynchronize(), file, line);
}

void SynchronizeCUDAStream(cudaStream_t cuda_stream, const char* file, const int line) {
  gpuAssert(cudaStreamSynchronize(cuda_stream), file, line);
}

void PrintLastCUDAError() {
  const char* error_name = cudaGetErrorName(cudaGetLastError());
  Log::Fatal(error_name);
}

void SetCUDADevice(int gpu_device_id, const char* file, int line) {
  int cur_gpu_device_id = 0;
  CUDASUCCESS_OR_FATAL_OUTER(cudaGetDevice(&cur_gpu_device_id));
  if (cur_gpu_device_id != gpu_device_id) {
    CUDASUCCESS_OR_FATAL_OUTER(cudaSetDevice(gpu_device_id));
  }
}

int GetCUDADevice(const char* file, int line) {
  int cur_gpu_device_id = 0;
  CUDASUCCESS_OR_FATAL_OUTER(cudaGetDevice(&cur_gpu_device_id));
  return cur_gpu_device_id;
}

cudaStream_t CUDAStreamCreate() {
  cudaStream_t cuda_stream;
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream));
  return cuda_stream;
}

void CUDAStreamDestroy(cudaStream_t cuda_stream) {
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_stream));
}

void NCCLGroupStart() {
  NCCLCHECK(ncclGroupStart());
}

void NCCLGroupEnd() {
  NCCLCHECK(ncclGroupEnd());
}

}  // namespace LightGBM

#endif  // USE_CUDA
