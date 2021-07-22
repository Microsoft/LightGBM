/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_CUDA
#include "cuda_tree_learner.h"

#include <LightGBM/bin.h>
#include <LightGBM/network.h>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/common.h>

#include <pthread.h>

#include <algorithm>
#include <cinttypes>
#include <vector>

#include "../io/dense_bin.hpp"

namespace LightGBM {

#define cudaMemcpy_DEBUG 0  // 1: DEBUG cudaMemcpy
#define ResetTrainingData_DEBUG 0  // 1: Debug ResetTrainingData

#define CUDA_DEBUG 0

static void *launch_cuda_histogram(void *thread_data) {
  ThreadData td = *(reinterpret_cast<ThreadData*>(thread_data));
  int device_id = td.device_id;
  CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));

  // launch cuda kernel
  cuda_histogram(td.histogram_size,
                 td.leaf_num_data, td.num_data, td.use_all_features,
                 td.is_constant_hessian, td.num_workgroups, td.stream,
                 td.device_features,
                 td.device_feature_masks,
                 td.num_data,
                 td.device_data_indices,
                 td.leaf_num_data,
                 td.device_gradients,
                 td.device_hessians, td.hessians_const,
                 td.device_subhistograms, td.sync_counters,
                 td.device_histogram_outputs,
                 td.exp_workgroups_per_feature);

  CUDASUCCESS_OR_FATAL(cudaGetLastError());

  return NULL;
}

CUDATreeLearner::CUDATreeLearner(const Config* config)
  :SerialTreeLearner(config) {
  use_bagging_ = false;
  nthreads_ = 0;
  if (config->gpu_use_dp && USE_DP_FLOAT) {
    Log::Info("LightGBM using CUDA trainer with DP float!!");
  } else {
    Log::Info("LightGBM using CUDA trainer with SP float!!");
  }
}

CUDATreeLearner::~CUDATreeLearner() {
}


void CUDATreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  // initialize SerialTreeLearner
  SerialTreeLearner::Init(train_data, is_constant_hessian);

  // some additional variables needed for GPU trainer
  num_feature_groups_ = train_data_->num_feature_groups();

  // Initialize GPU buffers and kernels: get device info
  InitGPU(config_->num_gpu);
}

// some functions used for debugging the GPU histogram construction
#if CUDA_DEBUG > 0

void PrintHistograms(hist_t* h, size_t size) {
  double total_hess = 0;
  for (size_t i = 0; i < size; ++i) {
    printf("%03lu=%9.3g,%9.3g\t", i, GET_GRAD(h, i), GET_HESS(h, i));
    if ((i & 3) == 3)
        printf("\n");
    total_hess += GET_HESS(h, i);
  }
  printf("\nSum hessians: %9.3g\n", total_hess);
}

union Float_t {
    int64_t i;
    double f;
    static int64_t ulp_diff(Float_t a, Float_t b) {
      return abs(a.i - b.i);
    }
};

int CompareHistograms(hist_t* h1, hist_t* h2, size_t size, int feature_id, int dp_flag, int const_flag) {
  int i;
  int retval = 0;
  printf("Comparing Histograms, feature_id = %d, size = %d\n", feature_id, static_cast<int>(size));
  if (dp_flag) {  // double precision
    double af, bf;
    int64_t ai, bi;
    for (i = 0; i < static_cast<int>(size); ++i) {
      af = GET_GRAD(h1, i);
      bf = GET_GRAD(h2, i);
      if ((((std::fabs(af - bf))/af) >= 1e-6) && ((std::fabs(af - bf)) >= 1e-6)) {
        printf("i = %5d, h1.grad %13.6lf, h2.grad %13.6lf\n", i, af, bf);
        ++retval;
      }
      if (const_flag) {
        ai = GET_HESS((reinterpret_cast<int64_t *>(h1)), i);
        bi = GET_HESS((reinterpret_cast<int64_t *>(h2)), i);
        if (ai != bi) {
          printf("i = %5d, h1.hess %" PRId64 ", h2.hess %" PRId64 "\n", i, ai, bi);
          ++retval;
        }
      } else {
        af = GET_HESS(h1, i);
        bf = GET_HESS(h2, i);
        if (((std::fabs(af - bf))/af) >= 1e-6) {
          printf("i = %5d, h1.hess %13.6lf, h2.hess %13.6lf\n", i, af, bf);
          ++retval;
        }
      }
    }
  } else {  // single precision
    float af, bf;
    int ai, bi;
    for (i = 0; i < static_cast<int>(size); ++i) {
      af = GET_GRAD(h1, i);
      bf = GET_GRAD(h2, i);
      if ((((std::fabs(af - bf))/af) >= 1e-6) && ((std::fabs(af - bf)) >= 1e-6)) {
        printf("i = %5d, h1.grad %13.6f, h2.grad %13.6f\n", i, af, bf);
        ++retval;
      }
      if (const_flag) {
        ai = GET_HESS(h1, i);
        bi = GET_HESS(h2, i);
        if (ai != bi) {
          printf("i = %5d, h1.hess %d, h2.hess %d\n", i, ai, bi);
          ++retval;
        }
      } else {
        af = GET_HESS(h1, i);
        bf = GET_HESS(h2, i);
        if (((std::fabs(af - bf))/af) >= 1e-5) {
          printf("i = %5d, h1.hess %13.6f, h2.hess %13.6f\n", i, af, bf);
          ++retval;
        }
      }
    }
  }
  printf("DONE Comparing Histograms...\n");
  return retval;
}
#endif

int CUDATreeLearner::GetNumWorkgroupsPerFeature(data_size_t leaf_num_data) {
  // we roughly want 256 workgroups per device, and we have num_dense_feature4_ feature tuples.
  // also guarantee that there are at least 2K examples per workgroup
  double x = 256.0 / num_dense_feature_groups_;

  int exp_workgroups_per_feature = static_cast<int>(ceil(log2(x)));
  double t = leaf_num_data / 1024.0;

  Log::Debug("We can have at most %d workgroups per feature4 for efficiency reasons\n"
         "Best workgroup size per feature for full utilization is %d\n", static_cast<int>(ceil(t)), (1 << exp_workgroups_per_feature));

  exp_workgroups_per_feature = std::min(exp_workgroups_per_feature, static_cast<int>(ceil(log(static_cast<double>(t))/log(2.0))));
  if (exp_workgroups_per_feature < 0)
      exp_workgroups_per_feature = 0;
  if (exp_workgroups_per_feature > kMaxLogWorkgroupsPerFeature)
      exp_workgroups_per_feature = kMaxLogWorkgroupsPerFeature;

  return exp_workgroups_per_feature;
}

void CUDATreeLearner::GPUHistogram(data_size_t leaf_num_data, bool use_all_features) {
  // we have already copied ordered gradients, ordered hessians and indices to GPU
  // decide the best number of workgroups working on one feature4 tuple
  // set work group size based on feature size
  // each 2^exp_workgroups_per_feature workgroups work on a feature4 tuple
  int exp_workgroups_per_feature = GetNumWorkgroupsPerFeature(leaf_num_data);
  std::vector<int> num_gpu_workgroups;
  ThreadData *thread_data = reinterpret_cast<ThreadData*>(_mm_malloc(sizeof(ThreadData) * num_gpu_, 16));

  for (int device_id = 0; device_id < num_gpu_; ++device_id) {
    int num_gpu_feature_groups = num_gpu_feature_groups_[device_id];
    int num_workgroups = (1 << exp_workgroups_per_feature) * num_gpu_feature_groups;
    num_gpu_workgroups.push_back(num_workgroups);
    if (num_workgroups > preallocd_max_num_wg_[device_id]) {
      preallocd_max_num_wg_.at(device_id) = num_workgroups;
      CUDASUCCESS_OR_FATAL(cudaFree(device_subhistograms_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_subhistograms_[device_id]), static_cast<size_t>(num_workgroups * dword_features_ * device_bin_size_ * (3 * hist_bin_entry_sz_ / 2))));
    }
    // set thread_data
    SetThreadData(thread_data, device_id, histogram_size_, leaf_num_data, use_all_features,
                  num_workgroups, exp_workgroups_per_feature);
  }

  for (int device_id = 0; device_id < num_gpu_; ++device_id) {
    if (pthread_create(cpu_threads_[device_id], NULL, launch_cuda_histogram, reinterpret_cast<void *>(&thread_data[device_id]))) {
      Log::Fatal("Error in creating threads.");
    }
  }

  /* Wait for the threads to finish */
  for (int device_id = 0; device_id < num_gpu_; ++device_id) {
    if (pthread_join(*(cpu_threads_[device_id]), NULL)) {
      Log::Fatal("Error in joining threads.");
    }
  }

  for (int device_id = 0; device_id < num_gpu_; ++device_id) {
    // copy the results asynchronously. Size depends on if double precision is used

    size_t output_size = num_gpu_feature_groups_[device_id] * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
    size_t host_output_offset = offset_gpu_feature_groups_[device_id] * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;

    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(reinterpret_cast<char*>(host_histogram_outputs_) + host_output_offset, device_histogram_outputs_[device_id], output_size, cudaMemcpyDeviceToHost, stream_[device_id]));
    CUDASUCCESS_OR_FATAL(cudaEventRecord(histograms_wait_obj_[device_id], stream_[device_id]));
  }
}


template <typename HistType>
void CUDATreeLearner::WaitAndGetHistograms(FeatureHistogram* leaf_histogram_array) {
  HistType* hist_outputs = reinterpret_cast<HistType*>(host_histogram_outputs_);

  #pragma omp parallel for schedule(static, num_gpu_)
  for (int device_id = 0; device_id < num_gpu_; ++device_id) {
    // when the output is ready, the computation is done
    CUDASUCCESS_OR_FATAL(cudaEventSynchronize(histograms_wait_obj_[device_id]));
  }

  HistType* histograms = reinterpret_cast<HistType*>(leaf_histogram_array[0].RawData() - kHistOffset);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (!feature_masks_[i]) {
      continue;
    }
    int dense_group_index = dense_feature_group_map_[i];
    auto old_histogram_array = histograms + train_data_->GroupBinBoundary(dense_group_index) * 2;
    int bin_size = train_data_->FeatureGroupNumBin(dense_group_index);

    for (int j = 0; j < bin_size; ++j) {
      GET_GRAD(old_histogram_array, j) = GET_GRAD(hist_outputs, i * device_bin_size_+ j);
      GET_HESS(old_histogram_array, j) = GET_HESS(hist_outputs, i * device_bin_size_+ j);
    }
  }
}

void CUDATreeLearner::CountDenseFeatureGroups() {
  num_dense_feature_groups_ = 0;

  for (int i = 0; i < num_feature_groups_; ++i) {
    if (!train_data_->IsMultiGroup(i)) {
      num_dense_feature_groups_++;
    }
  }
  if (!num_dense_feature_groups_) {
    Log::Warning("GPU acceleration is disabled because no non-trivial dense features can be found");
  }
}

void CUDATreeLearner::prevAllocateGPUMemory() {
  // how many feature-group tuples we have
  // leave some safe margin for prefetching
  // 256 work-items per workgroup. Each work-item prefetches one tuple for that feature

  allocated_num_data_ = std::max(num_data_ + 256 * (1 << kMaxLogWorkgroupsPerFeature), allocated_num_data_);

  // clear sparse/dense maps
  dense_feature_group_map_.clear();
  sparse_feature_group_map_.clear();

  // do nothing it there is no dense feature
  if (!num_dense_feature_groups_) {
    return;
  }

  // calculate number of feature groups per gpu
  num_gpu_feature_groups_.resize(num_gpu_);
  offset_gpu_feature_groups_.resize(num_gpu_);
  int num_features_per_gpu = num_dense_feature_groups_ / num_gpu_;
  int remain_features = num_dense_feature_groups_ - num_features_per_gpu * num_gpu_;

  int offset = 0;

  for (int i = 0; i < num_gpu_; ++i) {
    offset_gpu_feature_groups_.at(i) = offset;
    num_gpu_feature_groups_.at(i) = (i < remain_features) ? num_features_per_gpu + 1 : num_features_per_gpu;
    offset += num_gpu_feature_groups_.at(i);
  }

  feature_masks_.resize(num_dense_feature_groups_);
  Log::Debug("Resized feature masks");

  ptr_pinned_feature_masks_ = feature_masks_.data();
  Log::Debug("Memset pinned_feature_masks_");
  memset(ptr_pinned_feature_masks_, 0, num_dense_feature_groups_);

  // histogram bin entry size depends on the precision (single/double)
  hist_bin_entry_sz_ = 2 * (config_->gpu_use_dp ? sizeof(hist_t) : sizeof(gpu_hist_t));  // two elements in this "size"

  CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void **>(&host_histogram_outputs_), static_cast<size_t>(num_dense_feature_groups_ * device_bin_size_ * hist_bin_entry_sz_), cudaHostAllocPortable));

  nthreads_ = std::min(omp_get_max_threads(), num_dense_feature_groups_ / dword_features_);
  nthreads_ = std::max(nthreads_, 1);
}

// allocate GPU memory for each GPU
void CUDATreeLearner::AllocateGPUMemory() {
  #pragma omp parallel for schedule(static, num_gpu_)

  for (int device_id = 0; device_id < num_gpu_; ++device_id) {
    // do nothing it there is no gpu feature
    int num_gpu_feature_groups = num_gpu_feature_groups_[device_id];
    if (num_gpu_feature_groups) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));

      // allocate memory for all features
      if (device_features_[device_id] != NULL) {
        CUDASUCCESS_OR_FATAL(cudaFree(device_features_[device_id]));
      }

      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_features_[device_id]),  static_cast<size_t>(num_gpu_feature_groups * num_data_ * sizeof(uint8_t))));
      Log::Debug("Allocated device_features_ addr=%p sz=%lu", device_features_[device_id], num_gpu_feature_groups * num_data_);

      // allocate space for gradients and hessians on device
      // we will copy gradients and hessians in after ordered_gradients_ and ordered_hessians_ are constructed
      if (device_gradients_[device_id] != NULL) {
        CUDASUCCESS_OR_FATAL(cudaFree(device_gradients_[device_id]));
      }

      if (device_hessians_[device_id] != NULL) {
        CUDASUCCESS_OR_FATAL(cudaFree(device_hessians_[device_id]));
      }

      if (device_feature_masks_[device_id] != NULL) {
         CUDASUCCESS_OR_FATAL(cudaFree(device_feature_masks_[device_id]));
      }

      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_gradients_[device_id]), static_cast<size_t>(allocated_num_data_ * sizeof(score_t))));
      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_hessians_[device_id]),  static_cast<size_t>(allocated_num_data_ * sizeof(score_t))));

      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_feature_masks_[device_id]), static_cast<size_t>(num_gpu_feature_groups)));

      // copy indices to the device
      if (device_data_indices_[device_id] != NULL) {
        CUDASUCCESS_OR_FATAL(cudaFree(device_data_indices_[device_id]));
      }

      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_data_indices_[device_id]), static_cast<size_t>(allocated_num_data_ * sizeof(data_size_t))));
      CUDASUCCESS_OR_FATAL(cudaMemsetAsync(device_data_indices_[device_id], 0, allocated_num_data_ * sizeof(data_size_t), stream_[device_id]));

      Log::Debug("Memset device_data_indices_");

      // create output buffer, each feature has a histogram with device_bin_size_ bins,
      // each work group generates a sub-histogram of dword_features_ features.
      if (!device_subhistograms_[device_id]) {
        // only initialize once here, as this will not need to change when ResetTrainingData() is called
        CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_subhistograms_[device_id]), static_cast<size_t>(preallocd_max_num_wg_[device_id] * dword_features_ * device_bin_size_ * (3 * hist_bin_entry_sz_ / 2))));

        Log::Debug("created device_subhistograms_: %p", device_subhistograms_[device_id]);
      }

      // create atomic counters for inter-group coordination
      CUDASUCCESS_OR_FATAL(cudaFree(sync_counters_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaMalloc(&(sync_counters_[device_id]), static_cast<size_t>(num_gpu_feature_groups * sizeof(int))));
      CUDASUCCESS_OR_FATAL(cudaMemsetAsync(sync_counters_[device_id], 0, num_gpu_feature_groups * sizeof(int), stream_[device_id]));

      // The output buffer is allocated to host directly, to overlap compute and data transfer
      CUDASUCCESS_OR_FATAL(cudaFree(device_histogram_outputs_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaMalloc(&(device_histogram_outputs_[device_id]), static_cast<size_t>(num_gpu_feature_groups * device_bin_size_ * hist_bin_entry_sz_)));
    }
  }
}

void CUDATreeLearner::ResetGPUMemory() {
  // clear sparse/dense maps
  dense_feature_group_map_.clear();
  sparse_feature_group_map_.clear();
}

void CUDATreeLearner::copyDenseFeature() {
  if (num_feature_groups_ == 0) {
    LGBM_config_::current_learner = use_cpu_learner;
    return;
  }

  Log::Debug("Started copying dense features from CPU to GPU");
  // find the dense feature-groups and group then into Feature4 data structure (several feature-groups packed into 4 bytes)
  size_t copied_feature = 0;
  // set device info
  int device_id = 0;
  uint8_t* device_features = device_features_[device_id];
  CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
  Log::Debug("Started copying dense features from CPU to GPU - 1");

  for (int i = 0; i < num_feature_groups_; ++i) {
    // looking for dword_features_ non-sparse feature-groups
    if (!train_data_->IsMultiGroup(i)) {
      dense_feature_group_map_.push_back(i);
      auto sizes_in_byte = std::min(train_data_->FeatureGroupSizesInByte(i), static_cast<size_t>(num_data_));
      void* tmp_data = train_data_->FeatureGroupData(i);
      Log::Debug("Started copying dense features from CPU to GPU - 2");
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(&device_features[copied_feature * num_data_], tmp_data, sizes_in_byte, cudaMemcpyHostToDevice, stream_[device_id]));
      Log::Debug("Started copying dense features from CPU to GPU - 3");
      copied_feature++;
      // reset device info
      if (copied_feature == static_cast<size_t>(num_gpu_feature_groups_[device_id])) {
         CUDASUCCESS_OR_FATAL(cudaEventRecord(features_future_[device_id], stream_[device_id]));
         device_id += 1;
         copied_feature = 0;
         if (device_id < num_gpu_) {
           device_features = device_features_[device_id];
           CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
         }
      }
    } else {
      sparse_feature_group_map_.push_back(i);
    }
  }
}



// InitGPU w/ num_gpu
void CUDATreeLearner::InitGPU(int num_gpu) {
  // Get the max bin size, used for selecting best GPU kernel
  max_num_bin_ = 0;

  #if CUDA_DEBUG >= 1
  printf("bin_size: ");
  #endif
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (train_data_->IsMultiGroup(i)) {
      continue;
    }
    #if CUDA_DEBUG >= 1
    printf("%d, ", train_data_->FeatureGroupNumBin(i));
    #endif
    max_num_bin_ = std::max(max_num_bin_, train_data_->FeatureGroupNumBin(i));
  }
  #if CUDA_DEBUG >= 1
  printf("\n");
  #endif

  if (max_num_bin_ <= 16) {
    device_bin_size_ = 16;
    histogram_size_ = 16;
    dword_features_ = 1;
  } else if (max_num_bin_ <= 64) {
    device_bin_size_ = 64;
    histogram_size_ = 64;
    dword_features_ = 1;
  } else if (max_num_bin_ <= 256) {
    Log::Debug("device_bin_size_ = 256");
    device_bin_size_ = 256;
    histogram_size_ = 256;
    dword_features_ = 1;
  } else {
    Log::Fatal("bin size %d cannot run on GPU", max_num_bin_);
  }

  // ignore the feature groups that contain categorical features when producing warnings about max_bin.
  // these groups may contain larger number of bins due to categorical features, but not due to the setting of max_bin.
  int max_num_bin_no_categorical = 0;
  int cur_feature_group = 0;
  bool categorical_feature_found = false;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const int feature_group = train_data_->Feature2Group(inner_feature_index);
    const BinMapper* feature_bin_mapper = train_data_->FeatureBinMapper(inner_feature_index);
    if (feature_bin_mapper->bin_type() == BinType::CategoricalBin) {
      categorical_feature_found = true;
    }
    if (feature_group != cur_feature_group || inner_feature_index == num_features_ - 1) {
      if (!categorical_feature_found) {
        max_num_bin_no_categorical = std::max(max_num_bin_no_categorical, train_data_->FeatureGroupNumBin(cur_feature_group));
      }
      categorical_feature_found = false;
      cur_feature_group = feature_group;
    }
  }
  if (max_num_bin_no_categorical == 65) {
    Log::Warning("Setting max_bin to 63 is suggested for best performance");
  }
  if (max_num_bin_no_categorical == 17) {
    Log::Warning("Setting max_bin to 15 is suggested for best performance");
  }

  // get num_dense_feature_groups_
  CountDenseFeatureGroups();

  if (num_gpu > num_dense_feature_groups_) num_gpu = num_dense_feature_groups_;

  // initialize GPU
  int gpu_count;

  CUDASUCCESS_OR_FATAL(cudaGetDeviceCount(&gpu_count));
  num_gpu_ = (gpu_count < num_gpu) ? gpu_count : num_gpu;

  // set cpu threads
  cpu_threads_ = reinterpret_cast<pthread_t **>(_mm_malloc(sizeof(pthread_t *)*num_gpu_, 16));
  for (int device_id = 0; device_id < num_gpu_; ++device_id) {
    cpu_threads_[device_id] = reinterpret_cast<pthread_t *>(_mm_malloc(sizeof(pthread_t), 16));
  }

  // resize device memory pointers
  device_features_.resize(num_gpu_);
  device_gradients_.resize(num_gpu_);
  device_hessians_.resize(num_gpu_);
  device_feature_masks_.resize(num_gpu_);
  device_data_indices_.resize(num_gpu_);
  sync_counters_.resize(num_gpu_);
  device_subhistograms_.resize(num_gpu_);
  device_histogram_outputs_.resize(num_gpu_);

  // create stream & events to handle multiple GPUs
  preallocd_max_num_wg_.resize(num_gpu_, 1024);
  stream_.resize(num_gpu_);
  hessians_future_.resize(num_gpu_);
  gradients_future_.resize(num_gpu_);
  indices_future_.resize(num_gpu_);
  features_future_.resize(num_gpu_);
  kernel_start_.resize(num_gpu_);
  kernel_wait_obj_.resize(num_gpu_);
  histograms_wait_obj_.resize(num_gpu_);

  for (int i = 0; i < num_gpu_; ++i) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(i));
    CUDASUCCESS_OR_FATAL(cudaStreamCreate(&(stream_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(hessians_future_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(gradients_future_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(indices_future_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(features_future_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(kernel_start_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(kernel_wait_obj_[i])));
    CUDASUCCESS_OR_FATAL(cudaEventCreate(&(histograms_wait_obj_[i])));
  }

  allocated_num_data_ = 0;
  prevAllocateGPUMemory();

  AllocateGPUMemory();

  copyDenseFeature();
}

Tree* CUDATreeLearner::Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) {
  Tree *ret = SerialTreeLearner::Train(gradients, hessians, is_first_tree);
  return ret;
}

void CUDATreeLearner::ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) {
  // check data size
  data_size_t old_allocated_num_data = allocated_num_data_;

  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);

  #if ResetTrainingData_DEBUG == 1
  serial_time = std::chrono::steady_clock::now() - start_serial_time;
  #endif

  num_feature_groups_ = train_data_->num_feature_groups();

  // GPU memory has to been reallocated because data may have been changed
  #if ResetTrainingData_DEBUG == 1
  auto start_alloc_gpu_time = std::chrono::steady_clock::now();
  #endif

  // AllocateGPUMemory only when the number of data increased
  int old_num_feature_groups = num_dense_feature_groups_;
  CountDenseFeatureGroups();
  if ((old_allocated_num_data < (num_data_ + 256 * (1 << kMaxLogWorkgroupsPerFeature))) || (old_num_feature_groups < num_dense_feature_groups_)) {
    prevAllocateGPUMemory();
    AllocateGPUMemory();
  } else {
    ResetGPUMemory();
  }

  copyDenseFeature();

  #if ResetTrainingData_DEBUG == 1
  alloc_gpu_time = std::chrono::steady_clock::now() - start_alloc_gpu_time;
  #endif

  // setup GPU kernel arguments after we allocating all the buffers
  #if ResetTrainingData_DEBUG == 1
  auto start_set_arg_time = std::chrono::steady_clock::now();
  #endif

  #if ResetTrainingData_DEBUG == 1
  set_arg_time = std::chrono::steady_clock::now() - start_set_arg_time;
  reset_training_data_time = std::chrono::steady_clock::now() - start_reset_training_data_time;
  Log::Info("reset_training_data_time: %f secs.", reset_training_data_time.count() * 1e-3);
  Log::Info("serial_time: %f secs.", serial_time.count() * 1e-3);
  Log::Info("alloc_gpu_time: %f secs.", alloc_gpu_time.count() * 1e-3);
  Log::Info("set_arg_time: %f secs.", set_arg_time.count() * 1e-3);
  #endif
}

void CUDATreeLearner::BeforeTrain() {
  #if cudaMemcpy_DEBUG == 1
  std::chrono::duration<double, std::milli> device_hessians_time = std::chrono::milliseconds(0);
  std::chrono::duration<double, std::milli> device_gradients_time = std::chrono::milliseconds(0);
  #endif

  SerialTreeLearner::BeforeTrain();

  #if CUDA_DEBUG >= 2
  printf("CUDATreeLearner::BeforeTrain() Copying initial full gradients and hessians to device\n");
  #endif

  // Copy initial full hessians and gradients to GPU.
  // We start copying as early as possible, instead of at ConstructHistogram().
  if ((hessians_ != NULL) && (gradients_ != NULL)) {
    if (!use_bagging_ && num_dense_feature_groups_) {
      Log::Debug("CudaTreeLearner::BeforeTrain() No baggings, dense_feature_groups_=%d", num_dense_feature_groups_);

      for (int device_id = 0; device_id < num_gpu_; ++device_id) {
        if (!(share_state_->is_constant_hessian)) {
          Log::Debug("CUDATreeLearner::BeforeTrain(): Starting hessians_ -> device_hessians_");

          #if cudaMemcpy_DEBUG == 1
          auto start_device_hessians_time = std::chrono::steady_clock::now();
          #endif

          CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_hessians_[device_id], hessians_, num_data_*sizeof(score_t), cudaMemcpyHostToDevice, stream_[device_id]));

          CUDASUCCESS_OR_FATAL(cudaEventRecord(hessians_future_[device_id], stream_[device_id]));

          #if cudaMemcpy_DEBUG == 1
          device_hessians_time = std::chrono::steady_clock::now() - start_device_hessians_time;
          #endif

          Log::Debug("queued copy of device_hessians_");
        }

        #if cudaMemcpy_DEBUG == 1
        auto start_device_gradients_time = std::chrono::steady_clock::now();
        #endif

        CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_gradients_[device_id], gradients_, num_data_ * sizeof(score_t), cudaMemcpyHostToDevice, stream_[device_id]));
        CUDASUCCESS_OR_FATAL(cudaEventRecord(gradients_future_[device_id], stream_[device_id]));

        #if cudaMemcpy_DEBUG == 1
        device_gradients_time = std::chrono::steady_clock::now() - start_device_gradients_time;
        #endif

        Log::Debug("CUDATreeLearner::BeforeTrain: issued gradients_ -> device_gradients_");
      }
    }
  }

  // use bagging
  if ((hessians_ != NULL) && (gradients_ != NULL)) {
    if (data_partition_->leaf_count(0) != num_data_ && num_dense_feature_groups_) {
      // On GPU, we start copying indices, gradients and hessians now, instead at ConstructHistogram()
      // copy used gradients and hessians to ordered buffer
      const data_size_t* indices = data_partition_->indices();
      data_size_t cnt = data_partition_->leaf_count(0);

      // transfer the indices to GPU
      for (int device_id = 0; device_id < num_gpu_; ++device_id) {
        CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_data_indices_[device_id], indices, cnt * sizeof(*indices), cudaMemcpyHostToDevice, stream_[device_id]));
        CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_future_[device_id], stream_[device_id]));

        if (!(share_state_->is_constant_hessian)) {
          CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_hessians_[device_id], const_cast<void*>(reinterpret_cast<const void*>(&(hessians_[0]))), num_data_ * sizeof(score_t), cudaMemcpyHostToDevice, stream_[device_id]));
          CUDASUCCESS_OR_FATAL(cudaEventRecord(hessians_future_[device_id], stream_[device_id]));
        }

        CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_gradients_[device_id], const_cast<void*>(reinterpret_cast<const void*>(&(gradients_[0]))), num_data_ * sizeof(score_t), cudaMemcpyHostToDevice, stream_[device_id]));
        CUDASUCCESS_OR_FATAL(cudaEventRecord(gradients_future_[device_id], stream_[device_id]));
      }
    }
  }
}

bool CUDATreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  int smaller_leaf;

  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);

  // only have root
  if (right_leaf < 0) {
    smaller_leaf = -1;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    smaller_leaf = left_leaf;
  } else {
    smaller_leaf = right_leaf;
  }

  // Copy indices, gradients and hessians as early as possible
  if (smaller_leaf >= 0 && num_dense_feature_groups_) {
    // only need to initialize for smaller leaf
    // Get leaf boundary
    const data_size_t* indices = data_partition_->indices();
    data_size_t begin = data_partition_->leaf_begin(smaller_leaf);
    data_size_t end = begin + data_partition_->leaf_count(smaller_leaf);

    for (int device_id = 0; device_id < num_gpu_; ++device_id) {
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_data_indices_[device_id], &indices[begin], (end-begin) * sizeof(data_size_t), cudaMemcpyHostToDevice, stream_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_future_[device_id], stream_[device_id]));
    }
  }

  const bool ret = SerialTreeLearner::BeforeFindBestSplit(tree, left_leaf, right_leaf);

  return ret;
}

bool CUDATreeLearner::ConstructGPUHistogramsAsync(
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* data_indices, data_size_t num_data) {
  if (num_data <= 0) {
    return false;
  }

  // do nothing if no features can be processed on GPU
  if (!num_dense_feature_groups_) {
    Log::Debug("no dense feature groups, returning");
    return false;
  }

  // copy data indices if it is not null
  if (data_indices != nullptr && num_data != num_data_) {
    for (int device_id = 0; device_id < num_gpu_; ++device_id) {
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_data_indices_[device_id], data_indices, num_data * sizeof(data_size_t), cudaMemcpyHostToDevice, stream_[device_id]));
      CUDASUCCESS_OR_FATAL(cudaEventRecord(indices_future_[device_id], stream_[device_id]));
    }
  }

  // converted indices in is_feature_used to feature-group indices
  std::vector<int8_t> is_feature_group_used(num_feature_groups_, 0);

  #pragma omp parallel for schedule(static, 1024) if (num_features_ >= 2048)
  for (int i = 0; i < num_features_; ++i) {
    if (is_feature_used[i]) {
      int feature_group = train_data_->Feature2Group(i);
      is_feature_group_used[feature_group] = (train_data_->FeatureGroupNumBin(feature_group) <= 16) ? 2 : 1;
    }
  }

  // construct the feature masks for dense feature-groups
  int used_dense_feature_groups = 0;
  #pragma omp parallel for schedule(static, 1024) reduction(+:used_dense_feature_groups) if (num_dense_feature_groups_ >= 2048)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (is_feature_group_used[dense_feature_group_map_[i]]) {
      feature_masks_[i] = is_feature_group_used[dense_feature_group_map_[i]];
      ++used_dense_feature_groups;
    } else {
      feature_masks_[i] = 0;
    }
  }
  bool use_all_features = ((used_dense_feature_groups == num_dense_feature_groups_) && (data_indices != nullptr));
  // if no feature group is used, just return and do not use GPU
  if (used_dense_feature_groups == 0) {
    return false;
  }

  // if not all feature groups are used, we need to transfer the feature mask to GPU
  // otherwise, we will use a specialized GPU kernel with all feature groups enabled

  // We now copy even if all features are used.
  #pragma omp parallel for schedule(static, num_gpu_)
  for (int device_id = 0; device_id < num_gpu_; ++device_id) {
    int offset = offset_gpu_feature_groups_[device_id];
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_feature_masks_[device_id], ptr_pinned_feature_masks_ + offset, num_gpu_feature_groups_[device_id] , cudaMemcpyHostToDevice, stream_[device_id]));
  }

  // All data have been prepared, now run the GPU kernel
  GPUHistogram(num_data, use_all_features);

  return true;
}

void CUDATreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  std::vector<int8_t> is_sparse_feature_used(num_features_, 0);
  std::vector<int8_t> is_dense_feature_used(num_features_, 0);
  int num_dense_features = 0, num_sparse_features = 0;

  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    if (!is_feature_used[feature_index]) continue;
    if (train_data_->IsMultiGroup(train_data_->Feature2Group(feature_index))) {
      is_sparse_feature_used[feature_index] = 1;
      num_sparse_features++;
    } else {
      is_dense_feature_used[feature_index] = 1;
      num_dense_features++;
    }
  }

  // construct smaller leaf
  hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;

  // Check workgroups per feature4 tuple..
  int exp_workgroups_per_feature = GetNumWorkgroupsPerFeature(smaller_leaf_splits_->num_data_in_leaf());

  // if the workgroup per feature is 1 (2^0), return as the work is too small for a GPU
  if (exp_workgroups_per_feature == 0) {
    return SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
  }

  // ConstructGPUHistogramsAsync will return true if there are availabe feature groups dispatched to GPU
  bool is_gpu_used = ConstructGPUHistogramsAsync(is_feature_used,
    nullptr, smaller_leaf_splits_->num_data_in_leaf());

  // then construct sparse features on CPU
  // We set data_indices to null to avoid rebuilding ordered gradients/hessians
  if (num_sparse_features > 0) {
    train_data_->ConstructHistograms(is_sparse_feature_used,
    smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
    gradients_, hessians_,
    ordered_gradients_.data(), ordered_hessians_.data(),
    share_state_.get(),
    ptr_smaller_leaf_hist_data);
  }

  // wait for GPU to finish, only if GPU is actually used
  if (is_gpu_used) {
    if (config_->gpu_use_dp) {
      // use double precision
      WaitAndGetHistograms<hist_t>(smaller_leaf_histogram_array_);
    } else {
      // use single precision
      WaitAndGetHistograms<gpu_hist_t>(smaller_leaf_histogram_array_);
    }
  }

  // Compare GPU histogram with CPU histogram, useful for debuggin GPU code problem
  // #define CUDA_DEBUG_COMPARE
#ifdef CUDA_DEBUG_COMPARE
  printf("Start Comparing_Histogram between GPU and CPU, num_dense_feature_groups_ = %d\n", num_dense_feature_groups_);
  bool compare = true;
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (!feature_masks_[i])
      continue;
    int dense_feature_group_index = dense_feature_group_map_[i];
    size_t size = train_data_->FeatureGroupNumBin(dense_feature_group_index);
    hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
    hist_t* current_histogram = ptr_smaller_leaf_hist_data + train_data_->GroupBinBoundary(dense_feature_group_index) * 2;
    hist_t* gpu_histogram = new hist_t[size * 2];
    data_size_t num_data = smaller_leaf_splits_->num_data_in_leaf();
    printf("Comparing histogram for feature %d, num_data %d, num_data_ = %d, %lu bins\n", dense_feature_group_index, num_data, num_data_, size);
    std::copy(current_histogram, current_histogram + size * 2, gpu_histogram);
    std::memset(current_histogram, 0, size * sizeof(hist_t) * 2);
    if (train_data_->FeatureGroupBin(dense_feature_group_index) == nullptr) {
      continue;
    }
    if (num_data == num_data_) {
      if (share_state_->is_constant_hessian) {
        printf("ConstructHistogram(): num_data == num_data_ is_constant_hessian\n");
        train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
            0,
            num_data,
            gradients_,
            current_histogram);
      } else {
        printf("ConstructHistogram(): num_data == num_data_\n");
        train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
            0,
            num_data,
            gradients_, hessians_,
            current_histogram);
      }
    } else {
      if (share_state_->is_constant_hessian) {
        printf("ConstructHistogram(): is_constant_hessian\n");
        train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
            smaller_leaf_splits_->data_indices(),
            0,
            num_data,
            gradients_,
            current_histogram);
      } else {
        printf("ConstructHistogram(): 4, num_data = %d, num_data_ = %d\n", num_data, num_data_);
        train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
            smaller_leaf_splits_->data_indices(),
            0,
            num_data,
            gradients_, hessians_,
            current_histogram);
      }
    }
    int retval;
    if ((num_data != num_data_) && compare) {
        retval = CompareHistograms(gpu_histogram, current_histogram, size, dense_feature_group_index, config_->gpu_use_dp, share_state_->is_constant_hessian);
        printf("CompareHistograms reports %d errors\n", retval);
        compare = false;
    }
    retval = CompareHistograms(gpu_histogram, current_histogram, size, dense_feature_group_index, config_->gpu_use_dp, share_state_->is_constant_hessian);
    if (num_data == num_data_) {
        printf("CompareHistograms reports %d errors\n", retval);
    } else {
        printf("CompareHistograms reports %d errors\n", retval);
    }
    std::copy(gpu_histogram, gpu_histogram + size * 2, current_histogram);
    delete [] gpu_histogram;
  }
  printf("End Comparing Histogram between GPU and CPU\n");
  fflush(stderr);
  fflush(stdout);
#endif

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    // construct larger leaf
    hist_t* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - kHistOffset;

    is_gpu_used = ConstructGPUHistogramsAsync(is_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf());

    // then construct sparse features on CPU
    // We set data_indices to null to avoid rebuilding ordered gradients/hessians
    if (num_sparse_features > 0) {
    train_data_->ConstructHistograms(is_sparse_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(),
      share_state_.get(),
      ptr_larger_leaf_hist_data);
    }

    // wait for GPU to finish, only if GPU is actually used
    if (is_gpu_used) {
      if (config_->gpu_use_dp) {
        // use double precision
        WaitAndGetHistograms<hist_t>(larger_leaf_histogram_array_);
      } else {
        // use single precision
        WaitAndGetHistograms<gpu_hist_t>(larger_leaf_histogram_array_);
      }
    }
  }
}

void CUDATreeLearner::FindBestSplits(const Tree* tree) {
  SerialTreeLearner::FindBestSplits(tree);

#if CUDA_DEBUG >= 3
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    if (parent_leaf_histogram_array_ != nullptr
        && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    size_t bin_size = train_data_->FeatureNumBin(feature_index) + 1;
    printf("CUDATreeLearner::FindBestSplits() Feature %d bin_size=%zd smaller leaf:\n", feature_index, bin_size);
    PrintHistograms(smaller_leaf_histogram_array_[feature_index].RawData() - kHistOffset, bin_size);
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->leaf_index() < 0) { continue; }
    printf("CUDATreeLearner::FindBestSplits() Feature %d bin_size=%zd larger leaf:\n", feature_index, bin_size);

    PrintHistograms(larger_leaf_histogram_array_[feature_index].RawData() - kHistOffset, bin_size);
  }
#endif
}

void CUDATreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];
#if CUDA_DEBUG >= 2
  printf("Splitting leaf %d with feature %d thresh %d gain %f stat %f %f %f %f\n", best_Leaf, best_split_info.feature, best_split_info.threshold, best_split_info.gain, best_split_info.left_sum_gradient, best_split_info.right_sum_gradient, best_split_info.left_sum_hessian, best_split_info.right_sum_hessian);
#endif
  SerialTreeLearner::Split(tree, best_Leaf, left_leaf, right_leaf);
  if (Network::num_machines() == 1) {
    // do some sanity check for the GPU algorithm
    if (best_split_info.left_count < best_split_info.right_count) {
      if ((best_split_info.left_count != smaller_leaf_splits_->num_data_in_leaf()) ||
          (best_split_info.right_count!= larger_leaf_splits_->num_data_in_leaf())) {
        Log::Fatal("Bug in GPU histogram! split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      }
    } else {
      if ((best_split_info.left_count != larger_leaf_splits_->num_data_in_leaf()) ||
          (best_split_info.right_count!= smaller_leaf_splits_->num_data_in_leaf())) {
        Log::Fatal("Bug in GPU histogram! split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      }
    }
  }
}

}   // namespace LightGBM
#undef cudaMemcpy_DEBUG
#endif  // USE_CUDA
