/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_NCCL_TOPOLOGY_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_NCCL_TOPOLOGY_HPP_

#ifdef USE_CUDA

#ifdef USE_ROCM
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif

#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/network.h>
#include <LightGBM/utils/common.h>

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <functional>
#include <thread>

namespace LightGBM {

class NCCLTopology {
 public:
  NCCLTopology(const int master_gpu_device_id, const int num_gpu, const std::string& gpu_device_id_list, const data_size_t global_num_data) {
    num_gpu_ = num_gpu;
    master_gpu_device_id_ = master_gpu_device_id;
    global_num_data_ = global_num_data;
    int max_num_gpu = 0;
    CUDASUCCESS_OR_FATAL(cudaGetDeviceCount(&max_num_gpu));
    if (gpu_device_id_list != std::string("")) {
      std::set<int> gpu_id_set;
      std::vector<std::string> gpu_list_str = Common::Split(gpu_device_id_list.c_str(), ",");
      for (const auto& gpu_str : gpu_list_str) {
        int gpu_id = 0;
        Common::Atoi<int>(gpu_str.c_str(), &gpu_id);
        if (gpu_id < 0 || gpu_id >= max_num_gpu) {
          Log::Warning("Invalid GPU device ID %d in gpu_device_list is ignored.", gpu_id);
        } else {
          gpu_id_set.insert(gpu_id);
        }
      }
      for (const int gpu_id : gpu_id_set) {
        gpu_list_.push_back(gpu_id);
      }
    }
    if (!gpu_list_.empty() && num_gpu_ != static_cast<int>(gpu_list_.size())) {
      Log::Warning("num_gpu = %d is different from the number of valid device IDs in gpu_device_list (%d), using %d GPUs instead.", \
                  num_gpu_, static_cast<int>(gpu_list_.size()), static_cast<int>(gpu_list_.size()));
      num_gpu_ = static_cast<int>(gpu_list_.size());
    }

    if (!gpu_list_.empty()) {
      bool check_master_gpu = false;
      for (int i = 0; i < static_cast<int>(gpu_list_.size()); ++i) {
        const int gpu_id = gpu_list_[i];
        if (gpu_id == master_gpu_device_id_) {
          check_master_gpu = true;
          master_gpu_index_ = i;
          break;
        }
      }
      if (!check_master_gpu) {
        Log::Warning("Master GPU index not in gpu_device_list. Using %d as the master GPU instead.", gpu_list_[0]);
        master_gpu_device_id_ = gpu_list_[0];
        master_gpu_index_ = 0;
      }
    } else {
      if (num_gpu_ <= 0) {
        num_gpu_ = 1;
      } else if (num_gpu_ > max_num_gpu) {
        Log::Warning("Only %d GPUs available, using num_gpu = %d.", max_num_gpu, max_num_gpu);
        num_gpu_ = max_num_gpu;
      }
      if (master_gpu_device_id_ < 0 || master_gpu_device_id_ >= num_gpu_) {
        Log::Warning("Invalid gpu_device_id = %d for master GPU index, using gpu_device_id = 0 instead.", master_gpu_device_id_);
        master_gpu_device_id_ = 0;
        master_gpu_index_ = 0;
      }
      for (int i = 0; i < num_gpu_; ++i) {
        gpu_list_.push_back(i);
      }
    }

    Log::Info("Using GPU devices %s, and local master GPU device %d.", Common::Join<int>(gpu_list_, ",").c_str(), master_gpu_device_id_);

    const int num_threads = OMP_NUM_THREADS();
    if (num_gpu_ > num_threads) {
      Log::Fatal("Number of GPUs %d is greater than the number of threads %d. Please use more threads.", num_gpu_, num_threads);
    }

    host_threads_.resize(num_gpu_);
  }

  ~NCCLTopology() {}

  void InitNCCL() {
    nccl_gpu_rank_.resize(num_gpu_, -1);
    nccl_communicators_.resize(num_gpu_);
    ncclUniqueId nccl_unique_id;
    if (Network::num_machines() == 1 || Network::rank() == 0) {
      NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
    }
    if (Network::num_machines() > 1) {
      std::vector<ncclUniqueId> output_buffer(Network::num_machines());
      Network::Allgather(
        reinterpret_cast<char*>(&nccl_unique_id),
        sizeof(ncclUniqueId) / sizeof(char),
        reinterpret_cast<char*>(output_buffer.data()));
      if (Network::rank() > 0) {
        nccl_unique_id = output_buffer[0];
      }
    }

    if (Network::num_machines() > 1) {
      node_rank_offset_.resize(Network::num_machines() + 1, 0);
      Network::Allgather(
        reinterpret_cast<char*>(&num_gpu_),
        sizeof(int) / sizeof(char),
        reinterpret_cast<char*>(node_rank_offset_.data() + 1));
      for (int rank = 1; rank < Network::num_machines() + 1; ++rank) {
        node_rank_offset_[rank] += node_rank_offset_[rank - 1];
      }
      CHECK_EQ(node_rank_offset_[Network::rank() + 1] - node_rank_offset_[Network::rank()], num_gpu_);
      NCCLCHECK(ncclGroupStart());
      for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
        SetCUDADevice(gpu_list_[gpu_index], __FILE__, __LINE__);
        nccl_gpu_rank_[gpu_index] = gpu_index + node_rank_offset_[Network::rank()];
        NCCLCHECK(ncclCommInitRank(&nccl_communicators_[gpu_index], node_rank_offset_.back(), nccl_unique_id, nccl_gpu_rank_[gpu_index]));
      }
      NCCLCHECK(ncclGroupEnd());
    } else {
      NCCLCHECK(ncclGroupStart());
      for (int gpu_index = 0; gpu_index < num_gpu_; ++gpu_index) {
        SetCUDADevice(gpu_list_[gpu_index], __FILE__, __LINE__);
        nccl_gpu_rank_[gpu_index] = gpu_index;
        NCCLCHECK(ncclCommInitRank(&nccl_communicators_[gpu_index], num_gpu_, nccl_unique_id, gpu_index));
      }
      NCCLCHECK(ncclGroupEnd());
    }

    // return to master gpu device
    CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
  }

  template <typename ARG_T, typename RET_T>
  void RunPerDevice(const std::vector<std::unique_ptr<ARG_T>>& objs, const std::function<RET_T(ARG_T*)>& func) {
    #pragma omp parallel for schedule(static) num_threads(num_gpu_)
    for (int i = 0; i < num_gpu_; ++i) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_list_[i]));
      func(objs[i].get());
    }
    CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
  }

  template <typename RET_T>
  void InitPerDevice(std::vector<std::unique_ptr<RET_T>>* vec) {
    vec->resize(num_gpu_);
    #pragma omp parallel for schedule(static) num_threads(num_gpu_)
    for (int i = 0; i < num_gpu_; ++i) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_list_[i]));
      RET_T* nccl_info = new RET_T();
      nccl_info->SetNCCLInfo(nccl_communicators_[i], nccl_gpu_rank_[i], i, gpu_list_[i], global_num_data_);
      vec->operator[](i).reset(nccl_info);
    }
    CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
  }

  template <typename ARG_T>
  void DispatchPerDevice(std::vector<std::unique_ptr<ARG_T>>* objs, const std::function<void(ARG_T*)>& func) {
    for (int i = 0; i < num_gpu_; ++i) {
      host_threads_[i] = std::thread([this, i, &func, objs] () {
        CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_list_[i]))
        func(objs->operator[](i).get());
      });
    }
    for (int i = 0; i < num_gpu_; ++i) {
      host_threads_[i].join();
    }
    CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
  }

  template <typename ARG_T, typename RET_T>
  void RunOnMasterDevice(const std::vector<std::unique_ptr<ARG_T>>& objs, const std::function<RET_T(ARG_T*)>& func) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
    func(objs[master_gpu_index_].get());
  }

  template <typename ARG_T, typename RET_T>
  void RunOnNonMasterDevice(const std::vector<std::unique_ptr<ARG_T>>& objs, const std::function<RET_T(ARG_T*)>& func) {
    for (int i = 0; i < num_gpu_; ++i) {
      if (i != master_gpu_index_) {
        CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_list_[i]));
        func(objs[i].get());
      }
    }
    CUDASUCCESS_OR_FATAL(cudaSetDevice(master_gpu_device_id_));
  }

  int num_gpu() const { return num_gpu_; }

  int master_gpu_index() const { return master_gpu_index_; }

  int master_gpu_device_id() const { return master_gpu_device_id_; }

  const std::vector<int>& gpu_list() const { return gpu_list_; }

 private:
  int num_gpu_;
  int master_gpu_index_;
  int master_gpu_device_id_;
  std::vector<int> gpu_list_;
  data_size_t global_num_data_;

  ncclUniqueId nccl_unique_id_;
  std::vector<int> node_rank_offset_;
  std::vector<int> nccl_gpu_rank_;
  std::vector<ncclComm_t> nccl_communicators_;
  std::vector<std::thread> host_threads_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_NCCL_TOPOLOGY_HPP_
