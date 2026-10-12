/*!
 * Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROCM_INTEROP_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROCM_INTEROP_H_

#ifdef USE_CUDA

#if defined(__HIP_PLATFORM_AMD__)

// ROCm doesn't have atomicAdd_block, but it should be semantically the same as atomicAdd
#define atomicAdd_block atomicAdd

// hipify
#include <hip/hip_runtime.h>
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostAlloc
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemset hipMemset
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

// ROCm 7.0 did add __shfl_down_sync et al, but the following hack still works.
// Since mask is full 0xffffffff, we can use __shfl_down instead.
#define __shfl_down_sync(mask, val, offset) __shfl_down(val, offset)
#define __shfl_up_sync(mask, val, offset) __shfl_up(val, offset)

// warpSize is only allowed for device code.
// HIP header used to define warpSize as a constexpr that was either 32 or 64
// depending on the target device, and then always set it to 64 for host code.
static inline constexpr int WARP_SIZE_INTERNAL() {
#if defined(__GFX9__)
  return 64;
#else  // __GFX9__
  return 32;
#endif  // __GFX9__
}
#define WARPSIZE (WARP_SIZE_INTERNAL())

#else  // __HIP_PLATFORM_AMD__
// CUDA warpSize is not a constexpr, but always 32
#define WARPSIZE 32
#endif  // defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROCM_INTEROP_H_
