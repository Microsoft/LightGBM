/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */


#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_column_data.hpp>

#define COPY_SUBROW_BLOCK_SIZE_COLUMN_DATA (1024)

namespace LightGBM {

__global__ void CopySubrowKernel_ColumnData(
  uint8_t* const* in_cuda_data_by_column,
  const uint8_t* cuda_column_bit_type,
  const data_size_t* cuda_used_indices,
  const data_size_t num_used_indices,
  const int num_column,
  uint8_t** out_cuda_data_by_column) {
  const data_size_t local_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (local_data_index < num_used_indices) {
    for (int column_index = 0; column_index < num_column; ++column_index) {
      const uint8_t* in_column_data = in_cuda_data_by_column[column_index];
      uint8_t* out_column_data = out_cuda_data_by_column[column_index];
      const uint8_t bit_type = cuda_column_bit_type[column_index];
      if (bit_type == 8) {
        const uint8_t* true_in_column_data = reinterpret_cast<const uint8_t*>(in_column_data);
        uint8_t* true_out_column_data = reinterpret_cast<uint8_t*>(out_column_data);
        const data_size_t global_data_index = cuda_used_indices[local_data_index];
        true_out_column_data[local_data_index] = true_in_column_data[global_data_index];
      } else if (bit_type == 16) {
        const uint16_t* true_in_column_data = reinterpret_cast<const uint16_t*>(in_column_data);
        uint16_t* true_out_column_data = reinterpret_cast<uint16_t*>(out_column_data);
        const data_size_t global_data_index = cuda_used_indices[local_data_index];
        true_out_column_data[local_data_index] = true_in_column_data[global_data_index];
      } else if (bit_type == 32) {
        const uint32_t* true_in_column_data = reinterpret_cast<const uint32_t*>(in_column_data);
        uint32_t* true_out_column_data = reinterpret_cast<uint32_t*>(out_column_data);
        const data_size_t global_data_index = cuda_used_indices[local_data_index];
        true_out_column_data[local_data_index] = true_in_column_data[global_data_index];
      }
    }
  }
}

void CUDAColumnData::LaunchCopySubrowKernel(uint8_t* const* in_cuda_data_by_column) {
  const int num_blocks = (num_used_indices_ + COPY_SUBROW_BLOCK_SIZE_COLUMN_DATA - 1) / COPY_SUBROW_BLOCK_SIZE_COLUMN_DATA;
  CopySubrowKernel_ColumnData<<<num_blocks, COPY_SUBROW_BLOCK_SIZE_COLUMN_DATA>>>(
    in_cuda_data_by_column,
    cuda_column_bit_type_.RawData(),
    cuda_used_indices_.RawData(),
    num_used_indices_,
    num_columns_,
    cuda_data_by_column_.RawData());
}

}  // namespace LightGBM

#endif  // USE_CUDA
