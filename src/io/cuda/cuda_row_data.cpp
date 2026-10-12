/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_row_data.hpp>

#include <vector>

namespace LightGBM {

CUDARowData::CUDARowData(const Dataset* train_data,
                         const TrainingShareStates* train_share_state,
                         const int gpu_device_id,
                         const bool gpu_use_dp):
gpu_device_id_(gpu_device_id),
gpu_use_dp_(gpu_use_dp) {
  num_threads_ = OMP_NUM_THREADS();
  num_data_ = train_data->num_data();
  const auto& feature_hist_offsets = train_share_state->feature_hist_offsets();
  if (gpu_use_dp_) {
    shared_hist_size_ = DP_SHARED_HIST_SIZE;
  } else {
    shared_hist_size_ = SP_SHARED_HIST_SIZE;
  }
  if (feature_hist_offsets.empty()) {
    num_total_bin_ = 0;
  } else {
    num_total_bin_ = static_cast<int>(feature_hist_offsets.back());
  }
  num_feature_group_ = train_data->num_feature_groups();
  num_feature_ = train_data->num_features();
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
}

CUDARowData::~CUDARowData() {}

void CUDARowData::Init(const Dataset* train_data, TrainingShareStates* train_share_state) {
  if (num_feature_ == 0) {
    return;
  }
  DivideCUDAFeatureGroups(train_data, train_share_state);
  bit_type_ = 0;
  size_t total_size = 0;
  const void* host_row_ptr = nullptr;
  row_ptr_bit_type_ = 0;
  const void* host_data = train_share_state->GetRowWiseData(&bit_type_, &total_size, &is_sparse_, &host_row_ptr, &row_ptr_bit_type_);
  if (bit_type_ == 8) {
    if (!is_sparse_) {
      std::vector<uint8_t> partitioned_data;
      GetDenseDataPartitioned<uint8_t>(reinterpret_cast<const uint8_t*>(host_data), &partitioned_data);
      cuda_data_uint8_t_.InitFromHostVector(partitioned_data);
    } else {
      if (row_ptr_bit_type_ == 16) {
        InitSparseData<uint8_t, uint16_t>(
          reinterpret_cast<const uint8_t*>(host_data),
          reinterpret_cast<const uint16_t*>(host_row_ptr),
          &cuda_data_uint8_t_,
          &cuda_row_ptr_uint16_t_,
          &cuda_partition_ptr_uint16_t_);
      } else if (row_ptr_bit_type_ == 32) {
        InitSparseData<uint8_t, uint32_t>(
          reinterpret_cast<const uint8_t*>(host_data),
          reinterpret_cast<const uint32_t*>(host_row_ptr),
          &cuda_data_uint8_t_,
          &cuda_row_ptr_uint32_t_,
          &cuda_partition_ptr_uint32_t_);
      } else if (row_ptr_bit_type_ == 64) {
        InitSparseData<uint8_t, uint64_t>(
          reinterpret_cast<const uint8_t*>(host_data),
          reinterpret_cast<const uint64_t*>(host_row_ptr),
          &cuda_data_uint8_t_,
          &cuda_row_ptr_uint64_t_,
          &cuda_partition_ptr_uint64_t_);
      } else {
        Log::Fatal("Unknown data ptr bit type %d", row_ptr_bit_type_);
      }
    }
  } else if (bit_type_ == 16) {
    if (!is_sparse_) {
      std::vector<uint16_t> partitioned_data;
      GetDenseDataPartitioned<uint16_t>(reinterpret_cast<const uint16_t*>(host_data), &partitioned_data);
      cuda_data_uint16_t_.InitFromHostVector(partitioned_data);
    } else {
      if (row_ptr_bit_type_ == 16) {
        InitSparseData<uint16_t, uint16_t>(
          reinterpret_cast<const uint16_t*>(host_data),
          reinterpret_cast<const uint16_t*>(host_row_ptr),
          &cuda_data_uint16_t_,
          &cuda_row_ptr_uint16_t_,
          &cuda_partition_ptr_uint16_t_);
      } else if (row_ptr_bit_type_ == 32) {
        InitSparseData<uint16_t, uint32_t>(
          reinterpret_cast<const uint16_t*>(host_data),
          reinterpret_cast<const uint32_t*>(host_row_ptr),
          &cuda_data_uint16_t_,
          &cuda_row_ptr_uint32_t_,
          &cuda_partition_ptr_uint32_t_);
      } else if (row_ptr_bit_type_ == 64) {
        InitSparseData<uint16_t, uint64_t>(
          reinterpret_cast<const uint16_t*>(host_data),
          reinterpret_cast<const uint64_t*>(host_row_ptr),
          &cuda_data_uint16_t_,
          &cuda_row_ptr_uint64_t_,
          &cuda_partition_ptr_uint64_t_);
      } else {
        Log::Fatal("Unknown data ptr bit type %d", row_ptr_bit_type_);
      }
    }
  } else if (bit_type_ == 32) {
    if (!is_sparse_) {
      std::vector<uint32_t> partitioned_data;
      GetDenseDataPartitioned<uint32_t>(reinterpret_cast<const uint32_t*>(host_data), &partitioned_data);
      cuda_data_uint32_t_.InitFromHostVector(partitioned_data);
    } else {
      if (row_ptr_bit_type_ == 16) {
        InitSparseData<uint32_t, uint16_t>(
          reinterpret_cast<const uint32_t*>(host_data),
          reinterpret_cast<const uint16_t*>(host_row_ptr),
          &cuda_data_uint32_t_,
          &cuda_row_ptr_uint16_t_,
          &cuda_partition_ptr_uint16_t_);
      } else if (row_ptr_bit_type_ == 32) {
        InitSparseData<uint32_t, uint32_t>(
          reinterpret_cast<const uint32_t*>(host_data),
          reinterpret_cast<const uint32_t*>(host_row_ptr),
          &cuda_data_uint32_t_,
          &cuda_row_ptr_uint32_t_,
          &cuda_partition_ptr_uint32_t_);
      } else if (row_ptr_bit_type_ == 64) {
        InitSparseData<uint32_t, uint64_t>(
          reinterpret_cast<const uint32_t*>(host_data),
          reinterpret_cast<const uint64_t*>(host_row_ptr),
          &cuda_data_uint32_t_,
          &cuda_row_ptr_uint64_t_,
          &cuda_partition_ptr_uint64_t_);
      } else {
        Log::Fatal("Unknown data ptr bit type %d", row_ptr_bit_type_);
      }
    }
  } else {
    Log::Fatal("Unknown bit type = %d", bit_type_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDARowData::DivideCUDAFeatureGroups(const Dataset* train_data, TrainingShareStates* share_state) {
  const uint32_t max_num_bin_per_partition = shared_hist_size_ / 2;
  const std::vector<uint32_t>& column_hist_offsets = share_state->column_hist_offsets();
  std::vector<int> feature_group_num_feature_offsets;
  int offsets = 0;
  int prev_group_index = -1;
  for (int feature_index = 0; feature_index < num_feature_; ++feature_index) {
    const int feature_group_index = train_data->Feature2Group(feature_index);
    if (prev_group_index == -1 || feature_group_index != prev_group_index) {
      feature_group_num_feature_offsets.emplace_back(offsets);
      prev_group_index = feature_group_index;
    }
    ++offsets;
  }
  CHECK_EQ(offsets, num_feature_);
  feature_group_num_feature_offsets.emplace_back(offsets);

  uint32_t start_hist_offset = 0;
  feature_partition_column_index_offsets_.clear();
  column_hist_offsets_.clear();
  partition_hist_offsets_.clear();
  feature_partition_column_index_offsets_.emplace_back(0);
  partition_hist_offsets_.emplace_back(0);
  const int num_feature_groups = train_data->num_feature_groups();
  int column_index = 0;
  num_feature_partitions_ = 0;
  large_bin_partitions_.clear();
  small_bin_partitions_.clear();
  for (int feature_group_index = 0; feature_group_index < num_feature_groups; ++feature_group_index) {
    if (!train_data->IsMultiGroup(feature_group_index)) {
      const uint32_t column_feature_hist_start = column_hist_offsets[column_index];
      const uint32_t column_feature_hist_end = column_hist_offsets[column_index + 1];
      const uint32_t num_bin_in_dense_group = column_feature_hist_end - column_feature_hist_start;

      // if one column has too many bins, use a separate partition for that column
      if (num_bin_in_dense_group > max_num_bin_per_partition) {
        feature_partition_column_index_offsets_.emplace_back(column_index + 1);
        start_hist_offset = column_feature_hist_end;
        partition_hist_offsets_.emplace_back(start_hist_offset);
        large_bin_partitions_.emplace_back(num_feature_partitions_);
        ++num_feature_partitions_;
        column_hist_offsets_.emplace_back(0);
        ++column_index;
        continue;
      }

      // try if adding this column exceed the maximum number per partition
      const uint32_t cur_hist_num_bin = column_feature_hist_end - start_hist_offset;
      if (cur_hist_num_bin > max_num_bin_per_partition) {
        feature_partition_column_index_offsets_.emplace_back(column_index);
        start_hist_offset = column_feature_hist_start;
        partition_hist_offsets_.emplace_back(start_hist_offset);
        small_bin_partitions_.emplace_back(num_feature_partitions_);
        ++num_feature_partitions_;
      }
      column_hist_offsets_.emplace_back(column_hist_offsets[column_index] - start_hist_offset);
      if (feature_group_index == num_feature_groups - 1) {
        feature_partition_column_index_offsets_.emplace_back(column_index + 1);
        partition_hist_offsets_.emplace_back(column_hist_offsets.back());
        small_bin_partitions_.emplace_back(num_feature_partitions_);
        ++num_feature_partitions_;
      }
      ++column_index;
    } else {
      const int group_feature_index_start = feature_group_num_feature_offsets[feature_group_index];
      const int num_feature_in_group = feature_group_num_feature_offsets[feature_group_index + 1] - group_feature_index_start;
      for (int sub_feature_index = 0; sub_feature_index < num_feature_in_group; ++sub_feature_index) {
        const int feature_index = group_feature_index_start + sub_feature_index;
        const uint32_t column_feature_hist_start = column_hist_offsets[column_index];
        const uint32_t column_feature_hist_end = column_hist_offsets[column_index + 1];
        const uint32_t num_bin_in_dense_group = column_feature_hist_end - column_feature_hist_start;

        // if one column has too many bins, use a separate partition for that column
        if (num_bin_in_dense_group > max_num_bin_per_partition) {
          feature_partition_column_index_offsets_.emplace_back(column_index + 1);
          start_hist_offset = column_feature_hist_end;
          partition_hist_offsets_.emplace_back(start_hist_offset);
          large_bin_partitions_.emplace_back(num_feature_partitions_);
          ++num_feature_partitions_;
          column_hist_offsets_.emplace_back(0);
          ++column_index;
          continue;
        }

        // try if adding this column exceed the maximum number per partition
        const uint32_t cur_hist_num_bin = column_feature_hist_end - start_hist_offset;
        if (cur_hist_num_bin > max_num_bin_per_partition) {
          feature_partition_column_index_offsets_.emplace_back(column_index);
          start_hist_offset = column_feature_hist_start;
          partition_hist_offsets_.emplace_back(start_hist_offset);
          small_bin_partitions_.emplace_back(num_feature_partitions_);
          ++num_feature_partitions_;
        }
        column_hist_offsets_.emplace_back(column_hist_offsets[column_index] - start_hist_offset);
        if (feature_group_index == num_feature_groups - 1 && sub_feature_index == num_feature_in_group - 1) {
          CHECK_EQ(feature_index, num_feature_ - 1);
          feature_partition_column_index_offsets_.emplace_back(column_index + 1);
          partition_hist_offsets_.emplace_back(column_hist_offsets.back());
          small_bin_partitions_.emplace_back(num_feature_partitions_);
          ++num_feature_partitions_;
        }
        ++column_index;
      }
    }
  }
  column_hist_offsets_.emplace_back(column_hist_offsets.back() - start_hist_offset);
  max_num_column_per_partition_ = 0;
  for (size_t i = 0; i < feature_partition_column_index_offsets_.size() - 1; ++i) {
    const int num_column = feature_partition_column_index_offsets_[i + 1] - feature_partition_column_index_offsets_[i];
    if (num_column > max_num_column_per_partition_) {
      max_num_column_per_partition_ = num_column;
    }
  }

  cuda_feature_partition_column_index_offsets_.InitFromHostVector(feature_partition_column_index_offsets_);
  cuda_column_hist_offsets_.InitFromHostVector(column_hist_offsets_);
  cuda_partition_hist_offsets_.InitFromHostVector(partition_hist_offsets_);
}

template <typename BIN_TYPE>
void CUDARowData::GetDenseDataPartitioned(const BIN_TYPE* row_wise_data, std::vector<BIN_TYPE>* partitioned_data) {
  const int num_total_columns = feature_partition_column_index_offsets_.back();
  partitioned_data->resize(static_cast<size_t>(num_total_columns) * static_cast<size_t>(num_data_), 0);
  BIN_TYPE* out_data = partitioned_data->data();
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, num_total_columns, row_wise_data, out_data] (int /*thread_index*/, data_size_t start, data_size_t end) {
      for (size_t i = 0; i < feature_partition_column_index_offsets_.size() - 1; ++i) {
        const int num_prev_columns = static_cast<int>(feature_partition_column_index_offsets_[i]);
        const size_t offset = static_cast<size_t>(num_data_) * static_cast<size_t>(num_prev_columns);
        const int partition_column_start = feature_partition_column_index_offsets_[i];
        const int partition_column_end = feature_partition_column_index_offsets_[i + 1];
        const int num_columns_in_cur_partition = partition_column_end - partition_column_start;
        for (data_size_t data_index = start; data_index < end; ++data_index) {
          const size_t data_offset = offset + static_cast<size_t>(data_index) * num_columns_in_cur_partition;
          const size_t read_data_offset = static_cast<size_t>(data_index) * num_total_columns;
          for (int column_index = 0; column_index < num_columns_in_cur_partition; ++column_index) {
            const size_t true_column_index = read_data_offset + column_index + partition_column_start;
            const BIN_TYPE bin = row_wise_data[true_column_index];
            out_data[data_offset + column_index] = bin;
          }
        }
      }
    });
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE>
void CUDARowData::GetSparseDataPartitioned(
  const BIN_TYPE* row_wise_data,
  const DATA_PTR_TYPE* row_ptr,
  std::vector<std::vector<BIN_TYPE>>* partitioned_data,
  std::vector<std::vector<DATA_PTR_TYPE>>* partitioned_row_ptr,
  std::vector<DATA_PTR_TYPE>* partition_ptr) {
  const int num_partitions = static_cast<int>(feature_partition_column_index_offsets_.size()) - 1;
  partitioned_data->resize(num_partitions);
  partitioned_row_ptr->resize(num_partitions);
  std::vector<int> thread_max_elements_per_row(num_threads_, 0);
  Threading::For<int>(0, num_partitions, 1,
    [partitioned_data, partitioned_row_ptr, row_ptr, row_wise_data, &thread_max_elements_per_row, this] (int thread_index, int start, int end) {
      for (int partition_index = start; partition_index < end; ++partition_index) {
        std::vector<BIN_TYPE>& data_for_this_partition = partitioned_data->at(partition_index);
        std::vector<DATA_PTR_TYPE>& row_ptr_for_this_partition = partitioned_row_ptr->at(partition_index);
        const int partition_hist_start = partition_hist_offsets_[partition_index];
        const int partition_hist_end = partition_hist_offsets_[partition_index + 1];
        DATA_PTR_TYPE offset = 0;
        row_ptr_for_this_partition.clear();
        data_for_this_partition.clear();
        row_ptr_for_this_partition.emplace_back(offset);
        for (data_size_t data_index = 0; data_index < num_data_; ++data_index) {
          const DATA_PTR_TYPE row_start = row_ptr[data_index];
          const DATA_PTR_TYPE row_end = row_ptr[data_index + 1];
          const BIN_TYPE* row_data_start = row_wise_data + row_start;
          const BIN_TYPE* row_data_end = row_wise_data + row_end;
          const size_t partition_start_in_row = std::lower_bound(row_data_start, row_data_end, partition_hist_start) - row_data_start;
          const size_t partition_end_in_row = std::lower_bound(row_data_start, row_data_end, partition_hist_end) - row_data_start;
          for (size_t pos = partition_start_in_row; pos < partition_end_in_row; ++pos) {
            const BIN_TYPE bin = row_data_start[pos];
            CHECK_GE(bin, static_cast<BIN_TYPE>(partition_hist_start));
            data_for_this_partition.emplace_back(bin - partition_hist_start);
          }
          CHECK_GE(partition_end_in_row, partition_start_in_row);
          const data_size_t num_elements_in_row = partition_end_in_row - partition_start_in_row;
          offset += static_cast<DATA_PTR_TYPE>(num_elements_in_row);
          row_ptr_for_this_partition.emplace_back(offset);
          if (num_elements_in_row > thread_max_elements_per_row[thread_index]) {
            thread_max_elements_per_row[thread_index] = num_elements_in_row;
          }
        }
      }
    });
  partition_ptr->clear();
  DATA_PTR_TYPE offset = 0;
  partition_ptr->emplace_back(offset);
  for (size_t i = 0; i < partitioned_row_ptr->size(); ++i) {
    offset += partitioned_row_ptr->at(i).back();
    partition_ptr->emplace_back(offset);
  }
  max_num_column_per_partition_ = 0;
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    if (thread_max_elements_per_row[thread_index] > max_num_column_per_partition_) {
      max_num_column_per_partition_ = thread_max_elements_per_row[thread_index];
    }
  }
}

template <typename BIN_TYPE, typename ROW_PTR_TYPE>
void CUDARowData::InitSparseData(const BIN_TYPE* host_data,
                                 const ROW_PTR_TYPE* host_row_ptr,
                                 CUDAVector<BIN_TYPE>* cuda_data,
                                 CUDAVector<ROW_PTR_TYPE>* cuda_row_ptr,
                                 CUDAVector<ROW_PTR_TYPE>* cuda_partition_ptr) {
  std::vector<std::vector<BIN_TYPE>> partitioned_data;
  std::vector<std::vector<ROW_PTR_TYPE>> partitioned_data_ptr;
  std::vector<ROW_PTR_TYPE> partition_ptr;
  GetSparseDataPartitioned<BIN_TYPE, ROW_PTR_TYPE>(host_data, host_row_ptr, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
  cuda_partition_ptr->InitFromHostVector(partition_ptr);
  cuda_data->Resize(partition_ptr.back());
  cuda_row_ptr->Resize((num_data_ + 1) * partitioned_data_ptr.size());
  for (size_t i = 0; i < partitioned_data.size(); ++i) {
    const std::vector<ROW_PTR_TYPE>& data_ptr_for_this_partition = partitioned_data_ptr[i];
    const std::vector<BIN_TYPE>& data_for_this_partition = partitioned_data[i];
    CopyFromHostToCUDADevice<BIN_TYPE>(cuda_data->RawData() + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<ROW_PTR_TYPE>(cuda_row_ptr->RawData() + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size(), __FILE__, __LINE__);
  }
}

template <typename BIN_TYPE>
const BIN_TYPE* CUDARowData::GetBin() const {
  if (bit_type_ == 8) {
    return reinterpret_cast<const BIN_TYPE*>(cuda_data_uint8_t_.RawData());
  } else if (bit_type_ == 16) {
    return reinterpret_cast<const BIN_TYPE*>(cuda_data_uint16_t_.RawData());
  } else if (bit_type_ == 32) {
    return reinterpret_cast<const BIN_TYPE*>(cuda_data_uint32_t_.RawData());
  } else {
    Log::Fatal("Unknown bit_type %d for GetBin.", bit_type_);
  }
}

template const uint8_t* CUDARowData::GetBin<uint8_t>() const;

template const uint16_t* CUDARowData::GetBin<uint16_t>() const;

template const uint32_t* CUDARowData::GetBin<uint32_t>() const;

template <typename PTR_TYPE>
const PTR_TYPE* CUDARowData::GetRowPtr() const {
  if (row_ptr_bit_type_ == 16) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_row_ptr_uint16_t_.RawData());
  } else if (row_ptr_bit_type_ == 32) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_row_ptr_uint32_t_.RawData());
  } else if (row_ptr_bit_type_ == 64) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_row_ptr_uint64_t_.RawData());
  } else {
    Log::Fatal("Unknown row_ptr_bit_type = %d for GetRowPtr.", row_ptr_bit_type_);
  }
}

template const uint16_t* CUDARowData::GetRowPtr<uint16_t>() const;

template const uint32_t* CUDARowData::GetRowPtr<uint32_t>() const;

template const uint64_t* CUDARowData::GetRowPtr<uint64_t>() const;

template <typename PTR_TYPE>
const PTR_TYPE* CUDARowData::GetPartitionPtr() const {
  if (row_ptr_bit_type_ == 16) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_partition_ptr_uint16_t_.RawData());
  } else if (row_ptr_bit_type_ == 32) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_partition_ptr_uint32_t_.RawData());
  } else if (row_ptr_bit_type_ == 64) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_partition_ptr_uint64_t_.RawData());
  } else {
    Log::Fatal("Unknown row_ptr_bit_type = %d for GetPartitionPtr.", row_ptr_bit_type_);
  }
}

template const uint16_t* CUDARowData::GetPartitionPtr<uint16_t>() const;

template const uint32_t* CUDARowData::GetPartitionPtr<uint32_t>() const;

template const uint64_t* CUDARowData::GetPartitionPtr<uint64_t>() const;

}  // namespace LightGBM

#endif  // USE_CUDA
