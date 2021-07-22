/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_GPU

#include "gpu_tree_learner.h"

#include <LightGBM/bin.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/array_args.h>

#include <algorithm>

#include "../io/dense_bin.hpp"

#define GPU_DEBUG 0

namespace LightGBM {

GPUTreeLearner::GPUTreeLearner(const Config* config)
  :SerialTreeLearner(config) {
  use_bagging_ = false;
  Log::Info("This is the GPU trainer!!");
}

GPUTreeLearner::~GPUTreeLearner() {
  if (ptr_pinned_gradients_) {
    queue_.enqueue_unmap_buffer(pinned_gradients_, ptr_pinned_gradients_);
  }
  if (ptr_pinned_hessians_) {
    queue_.enqueue_unmap_buffer(pinned_hessians_, ptr_pinned_hessians_);
  }
  if (ptr_pinned_feature_masks_) {
    queue_.enqueue_unmap_buffer(pinned_feature_masks_, ptr_pinned_feature_masks_);
  }
}

void GPUTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  // initialize SerialTreeLearner
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  // some additional variables needed for GPU trainer
  num_feature_groups_ = train_data_->num_feature_groups();
  // Initialize GPU buffers and kernels
  InitGPU(config_->gpu_platform_id, config_->gpu_device_id);
}

// some functions used for debugging the GPU histogram construction
#if GPU_DEBUG > 0

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


void CompareHistograms(hist_t* h1, hist_t* h2, size_t size, int feature_id) {
  size_t i;
  Float_t a, b;
  for (i = 0; i < size; ++i) {
    a.f = GET_GRAD(h1, i);
    b.f = GET_GRAD(h2, i);
    int32_t ulps = Float_t::ulp_diff(a, b);
    if (ulps > 0) {
      // printf("grad %g != %g (%d ULPs)\n", GET_GRAD(h1, i), GET_GRAD(h2, i), ulps);
      // goto err;
    }
    a.f = GET_HESS(h1, i);
    b.f = GET_HESS(h2, i);
    ulps = Float_t::ulp_diff(a, b);
    if (std::fabs(a.f - b.f) >= 1e-20) {
      printf("hessian %g != %g (%d ULPs)\n", GET_HESS(h1, i), GET_HESS(h2, i), ulps);
      goto err;
    }
  }
  return;
err:
  Log::Warning("Mismatched histograms found for feature %d at location %lu.", feature_id, i);
  std::cin.get();
  PrintHistograms(h1, size);
  printf("\n");
  PrintHistograms(h2, size);
  std::cin.get();
}
#endif

int GPUTreeLearner::GetNumWorkgroupsPerFeature(data_size_t leaf_num_data) {
  // we roughly want 256 workgroups per device, and we have num_dense_feature4_ feature tuples.
  // also guarantee that there are at least 2K examples per workgroup
  double x = 256.0 / num_dense_feature4_;
  int exp_workgroups_per_feature = static_cast<int>(ceil(log2(x)));
  double t = leaf_num_data / 1024.0;
  #if GPU_DEBUG >= 4
  printf("Computing histogram for %d examples and (%d * %d) feature groups\n", leaf_num_data, dword_features_, num_dense_feature4_);
  printf("We can have at most %d workgroups per feature4 for efficiency reasons.\n"
         "Best workgroup size per feature for full utilization is %d\n", static_cast<int>(ceil(t)), (1 << exp_workgroups_per_feature));
  #endif
  exp_workgroups_per_feature = std::min(exp_workgroups_per_feature, static_cast<int>(ceil(log(static_cast<double>(t))/log(2.0))));
  if (exp_workgroups_per_feature < 0)
      exp_workgroups_per_feature = 0;
  if (exp_workgroups_per_feature > kMaxLogWorkgroupsPerFeature)
      exp_workgroups_per_feature = kMaxLogWorkgroupsPerFeature;
  // return 0;
  return exp_workgroups_per_feature;
}

void GPUTreeLearner::GPUHistogram(data_size_t leaf_num_data, bool use_all_features) {
  // we have already copied ordered gradients, ordered Hessians and indices to GPU
  // decide the best number of workgroups working on one feature4 tuple
  // set work group size based on feature size
  // each 2^exp_workgroups_per_feature workgroups work on a feature4 tuple
  int exp_workgroups_per_feature = GetNumWorkgroupsPerFeature(leaf_num_data);
  int num_workgroups = (1 << exp_workgroups_per_feature) * num_dense_feature4_;
  if (num_workgroups > preallocd_max_num_wg_) {
    preallocd_max_num_wg_ = num_workgroups;
    Log::Info("Increasing preallocd_max_num_wg_ to %d for launching more workgroups", preallocd_max_num_wg_);
    device_subhistograms_.reset(new boost::compute::vector<char>(
                              preallocd_max_num_wg_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_, ctx_));
    // we need to refresh the kernel arguments after reallocating
    for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
      // The only argument that needs to be changed later is num_data_
      histogram_kernels_[i].set_arg(7, *device_subhistograms_);
      histogram_allfeats_kernels_[i].set_arg(7, *device_subhistograms_);
      histogram_fulldata_kernels_[i].set_arg(7, *device_subhistograms_);
    }
  }
  #if GPU_DEBUG >= 4
  printf("Setting exp_workgroups_per_feature to %d, using %u work groups\n", exp_workgroups_per_feature, num_workgroups);
  printf("Constructing histogram with %d examples\n", leaf_num_data);
  #endif

  // the GPU kernel will process all features in one call, and each
  // 2^exp_workgroups_per_feature (compile time constant) workgroup will
  // process one feature4 tuple

  if (use_all_features) {
    histogram_allfeats_kernels_[exp_workgroups_per_feature].set_arg(4, leaf_num_data);
  } else {
    histogram_kernels_[exp_workgroups_per_feature].set_arg(4, leaf_num_data);
  }
  // for the root node, indices are not copied
  if (leaf_num_data != num_data_) {
    indices_future_.wait();
  }
  // for constant hessian, hessians are not copied except for the root node
  if (!share_state_->is_constant_hessian) {
    hessians_future_.wait();
  }
  gradients_future_.wait();
  // there will be 2^exp_workgroups_per_feature = num_workgroups / num_dense_feature4 sub-histogram per feature4
  // and we will launch num_feature workgroups for this kernel
  // will launch threads for all features
  // the queue should be asynchronous, and we will can WaitAndGetHistograms() before we start processing dense feature groups
  if (leaf_num_data == num_data_) {
    kernel_wait_obj_ = boost::compute::wait_list(
      queue_.enqueue_1d_range_kernel(histogram_fulldata_kernels_[exp_workgroups_per_feature], 0, num_workgroups * 256, 256));
  } else {
    if (use_all_features) {
      kernel_wait_obj_ = boost::compute::wait_list(
        queue_.enqueue_1d_range_kernel(histogram_allfeats_kernels_[exp_workgroups_per_feature], 0, num_workgroups * 256, 256));
    } else {
      kernel_wait_obj_ = boost::compute::wait_list(
        queue_.enqueue_1d_range_kernel(histogram_kernels_[exp_workgroups_per_feature], 0, num_workgroups * 256, 256));
    }
  }
  // copy the results asynchronously. Size depends on if double precision is used
  size_t output_size = num_dense_feature4_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
  boost::compute::event histogram_wait_event;
  host_histogram_outputs_ = reinterpret_cast<void*>(queue_.enqueue_map_buffer_async(
    device_histogram_outputs_, boost::compute::command_queue::map_read, 0, output_size, histogram_wait_event, kernel_wait_obj_));
  // we will wait for this object in WaitAndGetHistograms
  histograms_wait_obj_ = boost::compute::wait_list(histogram_wait_event);
}

template <typename HistType>
void GPUTreeLearner::WaitAndGetHistograms(hist_t* histograms) {
  HistType* hist_outputs = reinterpret_cast<HistType*>(host_histogram_outputs_);
  // when the output is ready, the computation is done
  histograms_wait_obj_.wait();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (!feature_masks_[i]) {
      continue;
    }
    int dense_group_index = dense_feature_group_map_[i];
    auto old_histogram_array = histograms + train_data_->GroupBinBoundary(dense_group_index) * 2;
    int bin_size = train_data_->FeatureGroupNumBin(dense_group_index);
    if (device_bin_mults_[i] == 1) {
      for (int j = 0; j < bin_size; ++j) {
        GET_GRAD(old_histogram_array, j) = GET_GRAD(hist_outputs, i * device_bin_size_+ j);
        GET_HESS(old_histogram_array, j) = GET_HESS(hist_outputs, i * device_bin_size_+ j);
      }
    } else {
      // values of this feature has been redistributed to multiple bins; need a reduction here
      int ind = 0;
      for (int j = 0; j < bin_size; ++j) {
        double sum_g = 0.0, sum_h = 0.0;
        for (int k = 0; k < device_bin_mults_[i]; ++k) {
          sum_g += GET_GRAD(hist_outputs, i * device_bin_size_+ ind);
          sum_h += GET_HESS(hist_outputs, i * device_bin_size_+ ind);
          ind++;
        }
        GET_GRAD(old_histogram_array, j) = sum_g;
        GET_HESS(old_histogram_array, j) = sum_h;
      }
    }
  }
  queue_.enqueue_unmap_buffer(device_histogram_outputs_, host_histogram_outputs_);
}

void GPUTreeLearner::AllocateGPUMemory() {
  num_dense_feature_groups_ = 0;
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (!train_data_->IsMultiGroup(i)) {
      num_dense_feature_groups_++;
    }
  }
  // how many feature-group tuples we have
  num_dense_feature4_ = (num_dense_feature_groups_ + (dword_features_ - 1)) / dword_features_;
  // leave some safe margin for prefetching
  // 256 work-items per workgroup. Each work-item prefetches one tuple for that feature
  int allocated_num_data_ = num_data_ + 256 * (1 << kMaxLogWorkgroupsPerFeature);
  // clear sparse/dense maps
  dense_feature_group_map_.clear();
  device_bin_mults_.clear();
  sparse_feature_group_map_.clear();
  // do nothing if no features can be processed on GPU
  if (!num_dense_feature_groups_) {
    Log::Warning("GPU acceleration is disabled because no non-trivial dense features can be found");
    return;
  }
  // allocate memory for all features (FIXME: 4 GB barrier on some devices, need to split to multiple buffers)
  device_features_.reset();
  device_features_ = std::unique_ptr<boost::compute::vector<Feature4>>(new boost::compute::vector<Feature4>(num_dense_feature4_ * num_data_, ctx_));
  // unpin old buffer if necessary before destructing them
  if (ptr_pinned_gradients_) {
    queue_.enqueue_unmap_buffer(pinned_gradients_, ptr_pinned_gradients_);
  }
  if (ptr_pinned_hessians_) {
    queue_.enqueue_unmap_buffer(pinned_hessians_, ptr_pinned_hessians_);
  }
  if (ptr_pinned_feature_masks_) {
    queue_.enqueue_unmap_buffer(pinned_feature_masks_, ptr_pinned_feature_masks_);
  }
  // make ordered_gradients and Hessians larger (including extra room for prefetching), and pin them
  ordered_gradients_.reserve(allocated_num_data_);
  ordered_hessians_.reserve(allocated_num_data_);
  pinned_gradients_ = boost::compute::buffer();  // deallocate
  pinned_gradients_ = boost::compute::buffer(ctx_, allocated_num_data_ * sizeof(score_t),
                                             boost::compute::memory_object::read_write | boost::compute::memory_object::use_host_ptr,
                                             ordered_gradients_.data());
  ptr_pinned_gradients_ = queue_.enqueue_map_buffer(pinned_gradients_, boost::compute::command_queue::map_write_invalidate_region,
                                                    0, allocated_num_data_ * sizeof(score_t));
  pinned_hessians_ = boost::compute::buffer();  // deallocate
  pinned_hessians_  = boost::compute::buffer(ctx_, allocated_num_data_ * sizeof(score_t),
                                             boost::compute::memory_object::read_write | boost::compute::memory_object::use_host_ptr,
                                             ordered_hessians_.data());
  ptr_pinned_hessians_ = queue_.enqueue_map_buffer(pinned_hessians_, boost::compute::command_queue::map_write_invalidate_region,
                                                   0, allocated_num_data_ * sizeof(score_t));
  // allocate space for gradients and Hessians on device
  // we will copy gradients and Hessians in after ordered_gradients_ and ordered_hessians_ are constructed
  device_gradients_ = boost::compute::buffer();  // deallocate
  device_gradients_ = boost::compute::buffer(ctx_, allocated_num_data_ * sizeof(score_t),
                      boost::compute::memory_object::read_only, nullptr);
  device_hessians_ = boost::compute::buffer();  // deallocate
  device_hessians_  = boost::compute::buffer(ctx_, allocated_num_data_ * sizeof(score_t),
                      boost::compute::memory_object::read_only, nullptr);
  // allocate feature mask, for disabling some feature-groups' histogram calculation
  feature_masks_.resize(num_dense_feature4_ * dword_features_);
  device_feature_masks_ = boost::compute::buffer();  // deallocate
  device_feature_masks_ = boost::compute::buffer(ctx_, num_dense_feature4_ * dword_features_,
                          boost::compute::memory_object::read_only, nullptr);
  pinned_feature_masks_ = boost::compute::buffer(ctx_, num_dense_feature4_ * dword_features_,
                                             boost::compute::memory_object::read_write | boost::compute::memory_object::use_host_ptr,
                                             feature_masks_.data());
  ptr_pinned_feature_masks_ = queue_.enqueue_map_buffer(pinned_feature_masks_, boost::compute::command_queue::map_write_invalidate_region,
                                                        0, num_dense_feature4_ * dword_features_);
  memset(ptr_pinned_feature_masks_, 0, num_dense_feature4_ * dword_features_);
  // copy indices to the device
  device_data_indices_.reset();
  device_data_indices_ = std::unique_ptr<boost::compute::vector<data_size_t>>(new boost::compute::vector<data_size_t>(allocated_num_data_, ctx_));
  boost::compute::fill(device_data_indices_->begin(), device_data_indices_->end(), 0, queue_);
  // histogram bin entry size depends on the precision (single/double)
  hist_bin_entry_sz_ = config_->gpu_use_dp ? sizeof(hist_t) * 2 : sizeof(gpu_hist_t) * 2;
  Log::Info("Size of histogram bin entry: %d", hist_bin_entry_sz_);
  // create output buffer, each feature has a histogram with device_bin_size_ bins,
  // each work group generates a sub-histogram of dword_features_ features.
  if (!device_subhistograms_) {
    // only initialize once here, as this will not need to change when ResetTrainingData() is called
    device_subhistograms_ = std::unique_ptr<boost::compute::vector<char>>(new boost::compute::vector<char>(
                              preallocd_max_num_wg_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_, ctx_));
  }
  // create atomic counters for inter-group coordination
  sync_counters_.reset();
  sync_counters_ = std::unique_ptr<boost::compute::vector<int>>(new boost::compute::vector<int>(
                    num_dense_feature4_, ctx_));
  boost::compute::fill(sync_counters_->begin(), sync_counters_->end(), 0, queue_);
  // The output buffer is allocated to host directly, to overlap compute and data transfer
  device_histogram_outputs_ = boost::compute::buffer();  // deallocate
  device_histogram_outputs_ = boost::compute::buffer(ctx_, num_dense_feature4_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_,
                           boost::compute::memory_object::write_only | boost::compute::memory_object::alloc_host_ptr, nullptr);
  // find the dense feature-groups and group then into Feature4 data structure (several feature-groups packed into 4 bytes)
  int k = 0, copied_feature4 = 0;
  std::vector<int> dense_dword_ind(dword_features_);
  for (int i = 0; i < num_feature_groups_; ++i) {
    // looking for dword_features_ non-sparse feature-groups
    if (!train_data_->IsMultiGroup(i)) {
      dense_dword_ind[k] = i;
      // decide if we need to redistribute the bin
      double t = device_bin_size_ / static_cast<double>(train_data_->FeatureGroupNumBin(i));
      // multiplier must be a power of 2
      device_bin_mults_.push_back(static_cast<int>(round(pow(2, floor(log2(t))))));
      // device_bin_mults_.push_back(1);
      #if GPU_DEBUG >= 1
      printf("feature-group %d using multiplier %d\n", i, device_bin_mults_.back());
      #endif
      k++;
    } else {
      sparse_feature_group_map_.push_back(i);
    }
    // found
    if (k == dword_features_) {
      k = 0;
      for (int j = 0; j < dword_features_; ++j) {
        dense_feature_group_map_.push_back(dense_dword_ind[j]);
      }
      copied_feature4++;
    }
  }
  // for data transfer time
  auto start_time = std::chrono::steady_clock::now();
  // Now generate new data structure feature4, and copy data to the device
  int nthreads = std::min(omp_get_max_threads(), static_cast<int>(dense_feature_group_map_.size()) / dword_features_);
  nthreads = std::max(nthreads, 1);
  std::vector<Feature4*> host4_vecs(nthreads);
  std::vector<boost::compute::buffer> host4_bufs(nthreads);
  std::vector<Feature4*> host4_ptrs(nthreads);
  // preallocate arrays for all threads, and pin them
  for (int i = 0; i < nthreads; ++i) {
    host4_vecs[i] = reinterpret_cast<Feature4*>(boost::alignment::aligned_alloc(4096, num_data_ * sizeof(Feature4)));
    host4_bufs[i] = boost::compute::buffer(ctx_, num_data_ * sizeof(Feature4),
                    boost::compute::memory_object::read_write | boost::compute::memory_object::use_host_ptr,
                    host4_vecs[i]);
    host4_ptrs[i] = reinterpret_cast<Feature4*>(queue_.enqueue_map_buffer(host4_bufs[i], boost::compute::command_queue::map_write_invalidate_region,
                    0, num_data_ * sizeof(Feature4)));
  }
  // building Feature4 bundles; each thread handles dword_features_ features
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(dense_feature_group_map_.size() / dword_features_); ++i) {
    int tid = omp_get_thread_num();
    Feature4* host4 = host4_ptrs[tid];
    auto dense_ind = dense_feature_group_map_.begin() + i * dword_features_;
    auto dev_bin_mult = device_bin_mults_.begin() + i * dword_features_;
    #if GPU_DEBUG >= 1
    printf("Copying feature group ");
    for (int l = 0; l < dword_features_; ++l) {
      printf("%d ", dense_ind[l]);
    }
    printf("to devices\n");
    #endif
    if (dword_features_ == 8) {
      // one feature datapoint is 4 bits
      BinIterator* bin_iters[8];
      for (int s_idx = 0; s_idx < 8; ++s_idx) {
        bin_iters[s_idx] = train_data_->FeatureGroupIterator(dense_ind[s_idx]);
        if (dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[s_idx]) == 0) {
          Log::Fatal("GPU tree learner assumes that all bins are Dense4bitsBin when num_bin <= 16, but feature %d is not", dense_ind[s_idx]);
        }
      }
      // this guarantees that the RawGet() function is inlined, rather than using virtual function dispatching
      DenseBinIterator<uint8_t, true> iters[8] = {
        *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[0]),
        *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[1]),
        *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[2]),
        *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[3]),
        *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[4]),
        *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[5]),
        *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[6]),
        *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[7])};
      for (int j = 0; j < num_data_; ++j) {
        host4[j].s[0] = (uint8_t)((iters[0].RawGet(j) * dev_bin_mult[0] + ((j+0) & (dev_bin_mult[0] - 1)))
                      |((iters[1].RawGet(j) * dev_bin_mult[1] + ((j+1) & (dev_bin_mult[1] - 1))) << 4));
        host4[j].s[1] = (uint8_t)((iters[2].RawGet(j) * dev_bin_mult[2] + ((j+2) & (dev_bin_mult[2] - 1)))
                      |((iters[3].RawGet(j) * dev_bin_mult[3] + ((j+3) & (dev_bin_mult[3] - 1))) << 4));
        host4[j].s[2] = (uint8_t)((iters[4].RawGet(j) * dev_bin_mult[4] + ((j+4) & (dev_bin_mult[4] - 1)))
                      |((iters[5].RawGet(j) * dev_bin_mult[5] + ((j+5) & (dev_bin_mult[5] - 1))) << 4));
        host4[j].s[3] = (uint8_t)((iters[6].RawGet(j) * dev_bin_mult[6] + ((j+6) & (dev_bin_mult[6] - 1)))
                      |((iters[7].RawGet(j) * dev_bin_mult[7] + ((j+7) & (dev_bin_mult[7] - 1))) << 4));
      }
    } else if (dword_features_ == 4) {
      // one feature datapoint is one byte
      for (int s_idx = 0; s_idx < 4; ++s_idx) {
        BinIterator* bin_iter = train_data_->FeatureGroupIterator(dense_ind[s_idx]);
        // this guarantees that the RawGet() function is inlined, rather than using virtual function dispatching
        if (dynamic_cast<DenseBinIterator<uint8_t, false>*>(bin_iter) != 0) {
          // Dense bin
          DenseBinIterator<uint8_t, false> iter = *static_cast<DenseBinIterator<uint8_t, false>*>(bin_iter);
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[s_idx] = (uint8_t)(iter.RawGet(j) * dev_bin_mult[s_idx] + ((j+s_idx) & (dev_bin_mult[s_idx] - 1)));
          }
        } else if (dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iter) != 0) {
          // Dense 4-bit bin
          DenseBinIterator<uint8_t, true> iter = *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iter);
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[s_idx] = (uint8_t)(iter.RawGet(j) * dev_bin_mult[s_idx] + ((j+s_idx) & (dev_bin_mult[s_idx] - 1)));
          }
        } else {
          Log::Fatal("Bug in GPU tree builder: only DenseBin and Dense4bitsBin are supported");
        }
      }
    } else {
      Log::Fatal("Bug in GPU tree builder: dword_features_ can only be 4 or 8");
    }
    #pragma omp critical
    queue_.enqueue_write_buffer(device_features_->get_buffer(),
                        i * num_data_ * sizeof(Feature4), num_data_ * sizeof(Feature4), host4);
    #if GPU_DEBUG >= 1
    printf("first example of feature-group tuple is: %d %d %d %d\n", host4[0].s[0], host4[0].s[1], host4[0].s[2], host4[0].s[3]);
    printf("Feature-groups copied to device with multipliers ");
    for (int l = 0; l < dword_features_; ++l) {
      printf("%d ", dev_bin_mult[l]);
    }
    printf("\n");
    #endif
  }
  // working on the remaining (less than dword_features_) feature groups
  if (k != 0) {
    Feature4* host4 = host4_ptrs[0];
    if (dword_features_ == 8) {
      memset(host4, 0, num_data_ * sizeof(Feature4));
    }
    #if GPU_DEBUG >= 1
    printf("%d features left\n", k);
    #endif
    for (int i = 0; i < k; ++i) {
      if (dword_features_ == 8) {
        BinIterator* bin_iter = train_data_->FeatureGroupIterator(dense_dword_ind[i]);
        if (dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iter) != 0) {
          DenseBinIterator<uint8_t, true> iter = *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iter);
          #pragma omp parallel for schedule(static)
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[i >> 1] |= (uint8_t)((iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i]
                                + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1)))
                               << ((i & 1) << 2));
          }
        } else {
          Log::Fatal("GPU tree learner assumes that all bins are Dense4bitsBin when num_bin <= 16, but feature %d is not", dense_dword_ind[i]);
        }
      } else if (dword_features_ == 4) {
        BinIterator* bin_iter = train_data_->FeatureGroupIterator(dense_dword_ind[i]);
        if (dynamic_cast<DenseBinIterator<uint8_t, false>*>(bin_iter) != 0) {
          DenseBinIterator<uint8_t, false> iter = *static_cast<DenseBinIterator<uint8_t, false>*>(bin_iter);
          #pragma omp parallel for schedule(static)
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[i] = (uint8_t)(iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i]
                          + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1)));
          }
        } else if (dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iter) != 0) {
          DenseBinIterator<uint8_t, true> iter = *static_cast<DenseBinIterator<uint8_t, true>*>(bin_iter);
          #pragma omp parallel for schedule(static)
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[i] = (uint8_t)(iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i]
                          + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1)));
          }
        } else {
          Log::Fatal("BUG in GPU tree builder: only DenseBin and Dense4bitsBin are supported");
        }
      } else {
        Log::Fatal("Bug in GPU tree builder: dword_features_ can only be 4 or 8");
      }
    }
    // fill the leftover features
    if (dword_features_ == 8) {
      #pragma omp parallel for schedule(static)
      for (int j = 0; j < num_data_; ++j) {
        for (int i = k; i < dword_features_; ++i) {
          // fill this empty feature with some "random" value
          host4[j].s[i >> 1] |= (uint8_t)((j & 0xf) << ((i & 1) << 2));
        }
      }
    } else if (dword_features_ == 4) {
      #pragma omp parallel for schedule(static)
      for (int j = 0; j < num_data_; ++j) {
        for (int i = k; i < dword_features_; ++i) {
          // fill this empty feature with some "random" value
          host4[j].s[i] = (uint8_t)j;
        }
      }
    }
    // copying the last 1 to (dword_features - 1) feature-groups in the last tuple
    queue_.enqueue_write_buffer(device_features_->get_buffer(),
                        (num_dense_feature4_ - 1) * num_data_ * sizeof(Feature4), num_data_ * sizeof(Feature4), host4);
    #if GPU_DEBUG >= 1
    printf("Last features copied to device\n");
    #endif
    for (int i = 0; i < k; ++i) {
      dense_feature_group_map_.push_back(dense_dword_ind[i]);
    }
  }
  // deallocate pinned space for feature copying
  for (int i = 0; i < nthreads; ++i) {
      queue_.enqueue_unmap_buffer(host4_bufs[i], host4_ptrs[i]);
      host4_bufs[i] = boost::compute::buffer();
      boost::alignment::aligned_free(host4_vecs[i]);
  }
  // data transfer time
  std::chrono::duration<double, std::milli> end_time = std::chrono::steady_clock::now() - start_time;
  Log::Info("%d dense feature groups (%.2f MB) transferred to GPU in %f secs. %d sparse feature groups",
            dense_feature_group_map_.size(), ((dense_feature_group_map_.size() + (dword_features_ - 1)) / dword_features_) * num_data_ * sizeof(Feature4) / (1024.0 * 1024.0),
            end_time * 1e-3, sparse_feature_group_map_.size());
  #if GPU_DEBUG >= 1
  printf("Dense feature group list (size %lu): ", dense_feature_group_map_.size());
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    printf("%d ", dense_feature_group_map_[i]);
  }
  printf("\n");
  printf("Sparse feature group list (size %lu): ", sparse_feature_group_map_.size());
  for (int i = 0; i < num_feature_groups_ - num_dense_feature_groups_; ++i) {
    printf("%d ", sparse_feature_group_map_[i]);
  }
  printf("\n");
  #endif
}

std::string GPUTreeLearner::GetBuildLog(const std::string &opts) {
  boost::compute::program program = boost::compute::program::create_with_source(kernel_source_, ctx_);
  try {
    program.build(opts);
  }
  catch (boost::compute::opencl_error &e) {
    auto error_code = e.error_code();
    std::string log("No log available.\n");
    // for other types of failure, build log might not be available; program.build_log() can crash
    if (error_code == CL_INVALID_PROGRAM || error_code == CL_BUILD_PROGRAM_FAILURE) {
      try {
        log = program.build_log();
      }
      catch(...) {
        // Something bad happened. Just return "No log available."
      }
    }
    return log;
  }
  // build is okay, log may contain warnings
  return program.build_log();
}

void GPUTreeLearner::BuildGPUKernels() {
  Log::Info("Compiling OpenCL Kernel with %d bins...", device_bin_size_);
  // destroy any old kernels
  histogram_kernels_.clear();
  histogram_allfeats_kernels_.clear();
  histogram_fulldata_kernels_.clear();
  // create OpenCL kernels for different number of workgroups per feature
  histogram_kernels_.resize(kMaxLogWorkgroupsPerFeature+1);
  histogram_allfeats_kernels_.resize(kMaxLogWorkgroupsPerFeature+1);
  histogram_fulldata_kernels_.resize(kMaxLogWorkgroupsPerFeature+1);
  // currently we don't use constant memory
  int use_constants = 0;
  OMP_INIT_EX();
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
    OMP_LOOP_EX_BEGIN();
    boost::compute::program program;
    std::ostringstream opts;
    // compile the GPU kernel depending if double precision is used, constant hessian is used, etc.
    opts << " -D POWER_FEATURE_WORKGROUPS=" << i
         << " -D USE_CONSTANT_BUF=" << use_constants << " -D USE_DP_FLOAT=" << int(config_->gpu_use_dp)
         << " -D CONST_HESSIAN=" << int(share_state_->is_constant_hessian)
         << " -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math";
    #if GPU_DEBUG >= 1
    std::cout << "Building GPU kernels with options: " << opts.str() << std::endl;
    #endif
    // kernel with indices in an array
    try {
      program = boost::compute::program::build_with_source(kernel_source_, ctx_, opts.str());
    }
    catch (boost::compute::opencl_error &e) {
      #pragma omp critical
      {
        std::cerr << "Build Options:" << opts.str() << std::endl;
        std::cerr << "Build Log:" << std::endl << GetBuildLog(opts.str()) << std::endl;
        Log::Fatal("Cannot build GPU program: %s", e.what());
      }
    }
    histogram_kernels_[i] = program.create_kernel(kernel_name_);

    // kernel with all features enabled, with eliminated branches
    opts << " -D ENABLE_ALL_FEATURES=1";
    try {
      program = boost::compute::program::build_with_source(kernel_source_, ctx_, opts.str());
    }
    catch (boost::compute::opencl_error &e) {
      #pragma omp critical
      {
        std::cerr << "Build Options:" << opts.str() << std::endl;
        std::cerr << "Build Log:" << std::endl << GetBuildLog(opts.str()) << std::endl;
        Log::Fatal("Cannot build GPU program: %s", e.what());
      }
    }
    histogram_allfeats_kernels_[i] = program.create_kernel(kernel_name_);

    // kernel with all data indices (for root node, and assumes that root node always uses all features)
    opts << " -D IGNORE_INDICES=1";
    try {
      program = boost::compute::program::build_with_source(kernel_source_, ctx_, opts.str());
    }
    catch (boost::compute::opencl_error &e) {
      #pragma omp critical
      {
        std::cerr << "Build Options:" << opts.str() << std::endl;
        std::cerr << "Build Log:" << std::endl << GetBuildLog(opts.str()) << std::endl;
        Log::Fatal("Cannot build GPU program: %s", e.what());
      }
    }
    histogram_fulldata_kernels_[i] = program.create_kernel(kernel_name_);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  Log::Info("GPU programs have been built");
}

void GPUTreeLearner::SetupKernelArguments() {
  // do nothing if no features can be processed on GPU
  if (!num_dense_feature_groups_) {
    return;
  }
  for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
    // The only argument that needs to be changed later is num_data_
    if (share_state_->is_constant_hessian) {
      // hessian is passed as a parameter, but it is not available now.
      // hessian will be set in BeforeTrain()
      histogram_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                         *device_data_indices_, num_data_, device_gradients_, 0.0f,
                                         *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
      histogram_allfeats_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                         *device_data_indices_, num_data_, device_gradients_, 0.0f,
                                         *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
      histogram_fulldata_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                          *device_data_indices_, num_data_, device_gradients_, 0.0f,
                                          *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
    } else {
      histogram_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                         *device_data_indices_, num_data_, device_gradients_, device_hessians_,
                                         *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
      histogram_allfeats_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                         *device_data_indices_, num_data_, device_gradients_, device_hessians_,
                                         *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
      histogram_fulldata_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                          *device_data_indices_, num_data_, device_gradients_, device_hessians_,
                                          *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
    }
  }
}

void GPUTreeLearner::InitGPU(int platform_id, int device_id) {
  // Get the max bin size, used for selecting best GPU kernel
  max_num_bin_ = 0;
  #if GPU_DEBUG >= 1
  printf("bin size: ");
  #endif
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (train_data_->IsMultiGroup(i)) {
      continue;
    }
    #if GPU_DEBUG >= 1
    printf("%d, ", train_data_->FeatureGroupNumBin(i));
    #endif
    max_num_bin_ = std::max(max_num_bin_, train_data_->FeatureGroupNumBin(i));
  }
  #if GPU_DEBUG >= 1
  printf("\n");
  #endif
  // initialize GPU
  dev_ = boost::compute::system::default_device();
  if (platform_id >= 0 && device_id >= 0) {
    const std::vector<boost::compute::platform> platforms = boost::compute::system::platforms();
    if (static_cast<int>(platforms.size()) > platform_id) {
      const std::vector<boost::compute::device> platform_devices = platforms[platform_id].devices();
      if (static_cast<int>(platform_devices.size()) > device_id) {
        Log::Info("Using requested OpenCL platform %d device %d", platform_id, device_id);
        dev_ = platform_devices[device_id];
      }
    }
  }
  // determine which kernel to use based on the max number of bins
  if (max_num_bin_ <= 16) {
    // the +9 skips extra characters ")", newline, "#endif" and newline at the beginning
    kernel_source_ = kernel16_src_ + 9;
    kernel_name_ = "histogram16";
    device_bin_size_ = 16;
    dword_features_ = 8;
  } else if (max_num_bin_ <= 64) {
    // the +9 skips extra characters ")", newline, "#endif" and newline at the beginning
    kernel_source_ = kernel64_src_ + 9;
    kernel_name_ = "histogram64";
    device_bin_size_ = 64;
    dword_features_ = 4;
  } else if (max_num_bin_ <= 256) {
    // the +9 skips extra characters ")", newline, "#endif" and newline at the beginning
    kernel_source_ = kernel256_src_ + 9;
    kernel_name_ = "histogram256";
    device_bin_size_ = 256;
    dword_features_ = 4;
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
  ctx_ = boost::compute::context(dev_);
  queue_ = boost::compute::command_queue(ctx_, dev_);
  Log::Info("Using GPU Device: %s, Vendor: %s", dev_.name().c_str(), dev_.vendor().c_str());
  BuildGPUKernels();
  AllocateGPUMemory();
  // setup GPU kernel arguments after we allocating all the buffers
  SetupKernelArguments();
}

Tree* GPUTreeLearner::Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) {
  return SerialTreeLearner::Train(gradients, hessians, is_first_tree);
}

void GPUTreeLearner::ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) {
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
  num_feature_groups_ = train_data_->num_feature_groups();
  // GPU memory has to been reallocated because data may have been changed
  AllocateGPUMemory();
  // setup GPU kernel arguments after we allocating all the buffers
  SetupKernelArguments();
}

void GPUTreeLearner::ResetIsConstantHessian(bool is_constant_hessian) {
  if (is_constant_hessian != share_state_->is_constant_hessian) {
    SerialTreeLearner::ResetIsConstantHessian(is_constant_hessian);
    BuildGPUKernels();
    SetupKernelArguments();
  }
}

void GPUTreeLearner::BeforeTrain() {
  #if GPU_DEBUG >= 2
  printf("Copying intial full gradients and hessians to device\n");
  #endif
  // Copy initial full hessians and gradients to GPU.
  // We start copying as early as possible, instead of at ConstructHistogram().
  if (!use_bagging_ && num_dense_feature_groups_) {
    if (!share_state_->is_constant_hessian) {
      hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, num_data_ * sizeof(score_t), hessians_);
    } else {
      // setup hessian parameters only
      score_t const_hessian = hessians_[0];
      for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
        // hessian is passed as a parameter
        histogram_kernels_[i].set_arg(6, const_hessian);
        histogram_allfeats_kernels_[i].set_arg(6, const_hessian);
        histogram_fulldata_kernels_[i].set_arg(6, const_hessian);
      }
    }
    gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, num_data_ * sizeof(score_t), gradients_);
  }

  SerialTreeLearner::BeforeTrain();

  // use bagging
  if (data_partition_->leaf_count(0) != num_data_ && num_dense_feature_groups_) {
    // On GPU, we start copying indices, gradients and Hessians now, instead at ConstructHistogram()
    // copy used gradients and Hessians to ordered buffer
    const data_size_t* indices = data_partition_->indices();
    data_size_t cnt = data_partition_->leaf_count(0);
    #if GPU_DEBUG > 0
    printf("Using bagging, examples count = %d\n", cnt);
    #endif
    // transfer the indices to GPU
    indices_future_ = boost::compute::copy_async(indices, indices + cnt, device_data_indices_->begin(), queue_);
    if (!share_state_->is_constant_hessian) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < cnt; ++i) {
        ordered_hessians_[i] = hessians_[indices[i]];
      }
      // transfer hessian to GPU
      hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, cnt * sizeof(score_t), ordered_hessians_.data());
    } else {
      // setup hessian parameters only
      score_t const_hessian = hessians_[indices[0]];
      for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
        // hessian is passed as a parameter
        histogram_kernels_[i].set_arg(6, const_hessian);
        histogram_allfeats_kernels_[i].set_arg(6, const_hessian);
        histogram_fulldata_kernels_[i].set_arg(6, const_hessian);
      }
    }
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < cnt; ++i) {
      ordered_gradients_[i] = gradients_[indices[i]];
    }
    // transfer gradients to GPU
    gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, cnt * sizeof(score_t), ordered_gradients_.data());
  }
}

bool GPUTreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
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

  // Copy indices, gradients and Hessians as early as possible
  if (smaller_leaf >= 0 && num_dense_feature_groups_) {
    // only need to initialize for smaller leaf
    // Get leaf boundary
    const data_size_t* indices = data_partition_->indices();
    data_size_t begin = data_partition_->leaf_begin(smaller_leaf);
    data_size_t end = begin + data_partition_->leaf_count(smaller_leaf);

    // copy indices to the GPU:
    #if GPU_DEBUG >= 2
    Log::Info("Copying indices, gradients and Hessians to GPU...");
    printf("Indices size %d being copied (left = %d, right = %d)\n", end - begin, num_data_in_left_child, num_data_in_right_child);
    #endif
    indices_future_ = boost::compute::copy_async(indices + begin, indices + end, device_data_indices_->begin(), queue_);

    if (!share_state_->is_constant_hessian) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = begin; i < end; ++i) {
        ordered_hessians_[i - begin] = hessians_[indices[i]];
      }
      // copy ordered Hessians to the GPU:
      hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, (end - begin) * sizeof(score_t), ptr_pinned_hessians_);
    }

    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      ordered_gradients_[i - begin] = gradients_[indices[i]];
    }
    // copy ordered gradients to the GPU:
    gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, (end - begin) * sizeof(score_t), ptr_pinned_gradients_);

    #if GPU_DEBUG >= 2
    Log::Info("Gradients/Hessians/indices copied to device with size %d", end - begin);
    #endif
  }
  return SerialTreeLearner::BeforeFindBestSplit(tree, left_leaf, right_leaf);
}

bool GPUTreeLearner::ConstructGPUHistogramsAsync(
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* data_indices, data_size_t num_data,
  const score_t* gradients, const score_t* hessians,
  score_t* ordered_gradients, score_t* ordered_hessians) {
  if (num_data <= 0) {
    return false;
  }
  // do nothing if no features can be processed on GPU
  if (!num_dense_feature_groups_) {
    return false;
  }

  // copy data indices if it is not null
  if (data_indices != nullptr && num_data != num_data_) {
    indices_future_ = boost::compute::copy_async(data_indices, data_indices + num_data, device_data_indices_->begin(), queue_);
  }
  // generate and copy ordered_gradients if gradients is not null
  if (gradients != nullptr) {
    if (num_data != num_data_) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_gradients[i] = gradients[data_indices[i]];
      }
      gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, num_data * sizeof(score_t), ptr_pinned_gradients_);
    } else {
      gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, num_data * sizeof(score_t), gradients);
    }
  }
  // generate and copy ordered_hessians if Hessians is not null
  if (hessians != nullptr && !share_state_->is_constant_hessian) {
    if (num_data != num_data_) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_hessians[i] = hessians[data_indices[i]];
      }
      hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, num_data * sizeof(score_t), ptr_pinned_hessians_);
    } else {
      hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, num_data * sizeof(score_t), hessians);
    }
  }
  // converted indices in is_feature_used to feature-group indices
  std::vector<int8_t> is_feature_group_used(num_feature_groups_, 0);
  #pragma omp parallel for schedule(static, 1024) if (num_features_ >= 2048)
  for (int i = 0; i < num_features_; ++i) {
    if (is_feature_used[i]) {
      is_feature_group_used[train_data_->Feature2Group(i)] = 1;
    }
  }
  // construct the feature masks for dense feature-groups
  int used_dense_feature_groups = 0;
  #pragma omp parallel for schedule(static, 1024) reduction(+:used_dense_feature_groups) if (num_dense_feature_groups_ >= 2048)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (is_feature_group_used[dense_feature_group_map_[i]]) {
      feature_masks_[i] = 1;
      ++used_dense_feature_groups;
    } else {
      feature_masks_[i] = 0;
    }
  }
  bool use_all_features = used_dense_feature_groups == num_dense_feature_groups_;
  // if no feature group is used, just return and do not use GPU
  if (used_dense_feature_groups == 0) {
    return false;
  }
#if GPU_DEBUG >= 1
  printf("Feature masks:\n");
  for (unsigned int i = 0; i < feature_masks_.size(); ++i) {
    printf("%d ", feature_masks_[i]);
  }
  printf("\n");
  printf("%d feature groups, %d used, %d\n", num_dense_feature_groups_, used_dense_feature_groups, use_all_features);
#endif
  // if not all feature groups are used, we need to transfer the feature mask to GPU
  // otherwise, we will use a specialized GPU kernel with all feature groups enabled
  if (!use_all_features) {
    queue_.enqueue_write_buffer(device_feature_masks_, 0, num_dense_feature4_ * dword_features_, ptr_pinned_feature_masks_);
  }
  // All data have been prepared, now run the GPU kernel
  GPUHistogram(num_data, use_all_features);
  return true;
}

void GPUTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  std::vector<int8_t> is_sparse_feature_used(num_features_, 0);
  std::vector<int8_t> is_dense_feature_used(num_features_, 0);
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    if (!is_feature_used[feature_index]) continue;
    if (train_data_->IsMultiGroup(train_data_->Feature2Group(feature_index))) {
      is_sparse_feature_used[feature_index] = 1;
    } else {
      is_dense_feature_used[feature_index] = 1;
    }
  }
  // construct smaller leaf
  hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
  // ConstructGPUHistogramsAsync will return true if there are available feature groups dispatched to GPU
  bool is_gpu_used = ConstructGPUHistogramsAsync(is_feature_used,
    nullptr, smaller_leaf_splits_->num_data_in_leaf(),
    nullptr, nullptr,
    nullptr, nullptr);
  // then construct sparse features on CPU
  train_data_->ConstructHistograms(is_sparse_feature_used,
    smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
    gradients_, hessians_,
    ordered_gradients_.data(), ordered_hessians_.data(),
    share_state_.get(),
    ptr_smaller_leaf_hist_data);
  // wait for GPU to finish, only if GPU is actually used
  if (is_gpu_used) {
    if (config_->gpu_use_dp) {
      // use double precision
      WaitAndGetHistograms<hist_t>(ptr_smaller_leaf_hist_data);
    } else {
      // use single precision
      WaitAndGetHistograms<gpu_hist_t>(ptr_smaller_leaf_hist_data);
    }
  }

  // Compare GPU histogram with CPU histogram, useful for debugging GPU code problem
  // #define GPU_DEBUG_COMPARE
  #ifdef GPU_DEBUG_COMPARE
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (!feature_masks_[i])
      continue;
    int dense_feature_group_index = dense_feature_group_map_[i];
    size_t size = train_data_->FeatureGroupNumBin(dense_feature_group_index);
    hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
    hist_t* current_histogram = ptr_smaller_leaf_hist_data + train_data_->GroupBinBoundary(dense_feature_group_index) * 2;
    hist_t* gpu_histogram = new hist_t[size * 2];
    data_size_t num_data = smaller_leaf_splits_->num_data_in_leaf();
    printf("Comparing histogram for feature %d size %d, %lu bins\n", dense_feature_group_index, num_data, size);
    std::copy(current_histogram, current_histogram + size * 2, gpu_histogram);
    std::memset(current_histogram, 0, size * sizeof(hist_t) * 2);
    if (train_data_->FeatureGroupBin(dense_feature_group_index) == nullptr) {
      continue;
    }
    if (num_data != num_data_) {
      train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
        smaller_leaf_splits_->data_indices(),
        0,
        num_data,
        ordered_gradients_.data(),
        ordered_hessians_.data(),
        current_histogram);
    } else {
      train_data_->FeatureGroupBin(dense_feature_group_index)->ConstructHistogram(
        0,
        num_data,
        gradients_,
        hessians_,
        current_histogram);
    }
    CompareHistograms(gpu_histogram, current_histogram, size, dense_feature_group_index);
    std::copy(gpu_histogram, gpu_histogram + size * 2, current_histogram);
    delete [] gpu_histogram;
  }
  #endif

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    // construct larger leaf
    hist_t* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - kHistOffset;
    is_gpu_used = ConstructGPUHistogramsAsync(is_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data());
    // then construct sparse features on CPU
    train_data_->ConstructHistograms(is_sparse_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(),
      share_state_.get(),
      ptr_larger_leaf_hist_data);
    // wait for GPU to finish, only if GPU is actually used
    if (is_gpu_used) {
      if (config_->gpu_use_dp) {
        // use double precision
        WaitAndGetHistograms<hist_t>(ptr_larger_leaf_hist_data);
      } else {
        // use single precision
        WaitAndGetHistograms<gpu_hist_t>(ptr_larger_leaf_hist_data);
      }
    }
  }
}

void GPUTreeLearner::FindBestSplits(const Tree* tree) {
  SerialTreeLearner::FindBestSplits(tree);

#if GPU_DEBUG >= 3
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    if (parent_leaf_histogram_array_ != nullptr
        && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    size_t bin_size = train_data_->FeatureNumBin(feature_index) + 1;
    printf("Feature %d smaller leaf:\n", feature_index);
    PrintHistograms(smaller_leaf_histogram_array_[feature_index].RawData() - kHistOffset, bin_size);
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) { continue; }
    printf("Feature %d larger leaf:\n", feature_index);
    PrintHistograms(larger_leaf_histogram_array_[feature_index].RawData() - kHistOffset, bin_size);
  }
#endif
}

void GPUTreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];
#if GPU_DEBUG >= 2
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
      smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian, best_split_info.right_output);
      larger_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian, best_split_info.left_output);
      if ((best_split_info.left_count != larger_leaf_splits_->num_data_in_leaf()) ||
          (best_split_info.right_count!= smaller_leaf_splits_->num_data_in_leaf())) {
        Log::Fatal("Bug in GPU histogram! split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      }
    }
  }
}

}   // namespace LightGBM
#endif  // USE_GPU
