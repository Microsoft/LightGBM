/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_column_data.hpp>

#include <cstdint>
#include <vector>

namespace LightGBM {

CUDAColumnData::CUDAColumnData(const data_size_t num_data, const int gpu_device_id) {
  num_threads_ = OMP_NUM_THREADS();
  num_data_ = num_data;
  gpu_device_id_ = gpu_device_id >= 0 ? gpu_device_id : 0;
  SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
  data_by_column_.clear();
}

CUDAColumnData::~CUDAColumnData() {}

template <bool IS_SPARSE, bool IS_4BIT, typename BIN_TYPE>
void CUDAColumnData::InitOneColumnData(const void* in_column_data, BinIterator* bin_iterator, CUDAVector<uint8_t>* out_column_data_pointer) {
  CUDAVector<BIN_TYPE> cuda_column_data;
  if (!IS_SPARSE) {
    if (IS_4BIT) {
      std::vector<BIN_TYPE> expanded_column_data(num_data_, 0);
      const BIN_TYPE* in_column_data_reintrepreted = reinterpret_cast<const BIN_TYPE*>(in_column_data);
      for (data_size_t i = 0; i < num_data_; ++i) {
        expanded_column_data[i] = static_cast<BIN_TYPE>((in_column_data_reintrepreted[i >> 1] >> ((i & 1) << 2)) & 0xf);
      }
      cuda_column_data.InitFromHostVector(expanded_column_data);
    } else {
      cuda_column_data.InitFromHostMemory(reinterpret_cast<const BIN_TYPE*>(in_column_data), static_cast<size_t>(num_data_));
    }
  } else {
    // need to iterate bin iterator
    std::vector<BIN_TYPE> expanded_column_data(num_data_, 0);
    for (data_size_t i = 0; i < num_data_; ++i) {
      expanded_column_data[i] = static_cast<BIN_TYPE>(bin_iterator->RawGet(i));
    }
    cuda_column_data.InitFromHostVector(expanded_column_data);
  }
  out_column_data_pointer->MoveFrom(cuda_column_data, sizeof(BIN_TYPE) * cuda_column_data.Size());
}

void CUDAColumnData::Init(const int num_columns,
                          const std::vector<const void*>& column_data,
                          const std::vector<BinIterator*>& column_bin_iterator,
                          const std::vector<uint8_t>& column_bit_type,
                          const std::vector<uint32_t>& feature_max_bin,
                          const std::vector<uint32_t>& feature_min_bin,
                          const std::vector<uint32_t>& feature_offset,
                          const std::vector<uint32_t>& feature_most_freq_bin,
                          const std::vector<uint32_t>& feature_default_bin,
                          const std::vector<uint8_t>& feature_missing_is_zero,
                          const std::vector<uint8_t>& feature_missing_is_na,
                          const std::vector<uint8_t>& feature_mfb_is_zero,
                          const std::vector<uint8_t>& feature_mfb_is_na,
                          const std::vector<int>& feature_to_column) {
  num_columns_ = num_columns;
  column_bit_type_ = column_bit_type;
  feature_max_bin_ = feature_max_bin;
  feature_min_bin_ = feature_min_bin;
  feature_offset_ = feature_offset;
  feature_most_freq_bin_ = feature_most_freq_bin;
  feature_default_bin_ = feature_default_bin;
  feature_missing_is_zero_ = feature_missing_is_zero;
  feature_missing_is_na_ = feature_missing_is_na;
  feature_mfb_is_zero_ = feature_mfb_is_zero;
  feature_mfb_is_na_ = feature_mfb_is_na;
  for (int column_index = 0; column_index < num_columns_; ++column_index) {
    data_by_column_.emplace_back(new CUDAVector<uint8_t>());
  }
  OMP_INIT_EX();
  #pragma omp parallel num_threads(num_threads_)
  {
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
    #pragma omp for schedule(static)
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      OMP_LOOP_EX_BEGIN();
      const int8_t bit_type = column_bit_type[column_index];
      if (column_data[column_index] != nullptr) {
        // is dense column
        if (bit_type == 4) {
          column_bit_type_[column_index] = 8;
          InitOneColumnData<false, true, uint8_t>(column_data[column_index], nullptr, data_by_column_[column_index].get());
        } else if (bit_type == 8) {
          InitOneColumnData<false, false, uint8_t>(column_data[column_index], nullptr, data_by_column_[column_index].get());
        } else if (bit_type == 16) {
          InitOneColumnData<false, false, uint16_t>(column_data[column_index], nullptr, data_by_column_[column_index].get());
        } else if (bit_type == 32) {
          InitOneColumnData<false, false, uint32_t>(column_data[column_index], nullptr, data_by_column_[column_index].get());
        } else {
          Log::Fatal("Unknown column bit type %d", bit_type);
        }
      } else {
        // is sparse column
        if (bit_type == 8) {
          InitOneColumnData<true, false, uint8_t>(nullptr, column_bin_iterator[column_index], data_by_column_[column_index].get());
        } else if (bit_type == 16) {
          InitOneColumnData<true, false, uint16_t>(nullptr, column_bin_iterator[column_index], data_by_column_[column_index].get());
        } else if (bit_type == 32) {
          InitOneColumnData<true, false, uint32_t>(nullptr, column_bin_iterator[column_index], data_by_column_[column_index].get());
        } else {
          Log::Fatal("Unknown column bit type %d", bit_type);
        }
      }
      OMP_LOOP_EX_END();
    }
  }
  OMP_THROW_EX();
  feature_to_column_ = feature_to_column;
  cuda_data_by_column_.InitFromHostVector(GetDataByColumnPointers(data_by_column_));
  InitColumnMetaInfo();
}

void CUDAColumnData::CopySubrow(
  const CUDAColumnData* full_set,
  const data_size_t* used_indices,
  const data_size_t num_used_indices) {
  num_threads_ = full_set->num_threads_;
  num_columns_ = full_set->num_columns_;
  column_bit_type_ = full_set->column_bit_type_;
  feature_min_bin_ = full_set->feature_min_bin_;
  feature_max_bin_ = full_set->feature_max_bin_;
  feature_offset_ = full_set->feature_offset_;
  feature_most_freq_bin_ = full_set->feature_most_freq_bin_;
  feature_default_bin_ = full_set->feature_default_bin_;
  feature_missing_is_zero_ = full_set->feature_missing_is_zero_;
  feature_missing_is_na_ = full_set->feature_missing_is_na_;
  feature_mfb_is_zero_ = full_set->feature_mfb_is_zero_;
  feature_mfb_is_na_ = full_set->feature_mfb_is_na_;
  feature_to_column_ = full_set->feature_to_column_;
  if (cuda_used_indices_.Size() == 0) {
    // initialize the subset cuda column data
    const size_t num_used_indices_size = static_cast<size_t>(num_used_indices);
    cuda_used_indices_.Resize(num_used_indices_size);
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      data_by_column_.emplace_back(new CUDAVector<uint8_t>());
    }
    OMP_INIT_EX();
    #pragma omp parallel num_threads(num_threads_)
    {
      SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
      #pragma omp for schedule(static)
      for (int column_index = 0; column_index < num_columns_; ++column_index) {
        OMP_LOOP_EX_BEGIN();
        const uint8_t bit_type = column_bit_type_[column_index];
        if (bit_type == 8) {
          CUDAVector<uint8_t> column_data;
          column_data.Resize(num_used_indices_size);
          data_by_column_[column_index]->MoveFrom(column_data, sizeof(uint8_t) * column_data.Size());
        } else if (bit_type == 16) {
          CUDAVector<uint16_t> column_data;
          column_data.Resize(num_used_indices_size);
          data_by_column_[column_index]->MoveFrom(column_data, sizeof(uint16_t) * column_data.Size());
        } else if (bit_type == 32) {
          CUDAVector<uint32_t> column_data;
          column_data.Resize(num_used_indices_size);
          data_by_column_[column_index]->MoveFrom(column_data, sizeof(uint32_t) * column_data.Size());
        }
        OMP_LOOP_EX_END();
      }
    }
    OMP_THROW_EX();
    cuda_data_by_column_.InitFromHostVector(GetDataByColumnPointers(data_by_column_));
    InitColumnMetaInfo();
    cur_subset_buffer_size_ = num_used_indices;
  } else {
    if (num_used_indices > cur_subset_buffer_size_) {
      ResizeWhenCopySubrow(num_used_indices);
      cur_subset_buffer_size_ = num_used_indices;
    }
  }
  cuda_used_indices_.InitFromHostMemory(used_indices, static_cast<size_t>(num_used_indices));
  num_used_indices_ = num_used_indices;
  LaunchCopySubrowKernel(full_set->cuda_data_by_column());
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDAColumnData::ResizeWhenCopySubrow(const data_size_t num_used_indices) {
  const size_t num_used_indices_size = static_cast<size_t>(num_used_indices);
  cuda_used_indices_.Resize(num_used_indices_size);
  OMP_INIT_EX();
  #pragma omp parallel num_threads(num_threads_)
  {
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
    #pragma omp for schedule(static)
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      OMP_LOOP_EX_BEGIN();
      const uint8_t bit_type = column_bit_type_[column_index];
      if (bit_type == 8) {
        data_by_column_[column_index]->Resize(sizeof(uint8_t) * num_used_indices_size);
      } else if (bit_type == 16) {
        data_by_column_[column_index]->Resize(sizeof(uint16_t) * num_used_indices_size);
      } else if (bit_type == 32) {
        data_by_column_[column_index]->Resize(sizeof(uint32_t) * num_used_indices_size);
      }
      OMP_LOOP_EX_END();
    }
  }
  OMP_THROW_EX();
  cuda_data_by_column_.InitFromHostVector(GetDataByColumnPointers(data_by_column_));
}

void CUDAColumnData::InitColumnMetaInfo() {
  cuda_column_bit_type_.InitFromHostVector(column_bit_type_);
  cuda_feature_max_bin_.InitFromHostVector(feature_max_bin_);
  cuda_feature_min_bin_.InitFromHostVector(feature_min_bin_);
  cuda_feature_offset_.InitFromHostVector(feature_offset_);
  cuda_feature_most_freq_bin_.InitFromHostVector(feature_most_freq_bin_);
  cuda_feature_default_bin_.InitFromHostVector(feature_default_bin_);
  cuda_feature_missing_is_zero_.InitFromHostVector(feature_missing_is_zero_);
  cuda_feature_missing_is_na_.InitFromHostVector(feature_missing_is_na_);
  cuda_feature_mfb_is_zero_.InitFromHostVector(feature_mfb_is_zero_);
  cuda_feature_mfb_is_na_.InitFromHostVector(feature_mfb_is_na_);
  cuda_feature_to_column_.InitFromHostVector(feature_to_column_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
