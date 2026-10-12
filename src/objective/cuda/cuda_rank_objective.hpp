/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_RANK_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define NUM_QUERY_PER_BLOCK (10)

#include <LightGBM/cuda/cuda_objective_function.hpp>
#include <LightGBM/utils/threading.h>

#include <fstream>
#include <string>
#include <vector>

#include "../rank_objective.hpp"

namespace LightGBM {

template <typename HOST_OBJECTIVE>
class CUDALambdaRankObjectiveInterface : public CUDAObjectiveInterface<HOST_OBJECTIVE> {
 public:
  explicit CUDALambdaRankObjectiveInterface(const Config& config): CUDAObjectiveInterface<HOST_OBJECTIVE>(config) {}

  explicit CUDALambdaRankObjectiveInterface(const std::vector<std::string>& strs): CUDAObjectiveInterface<HOST_OBJECTIVE>(strs) {}

  ~CUDALambdaRankObjectiveInterface() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    CUDAObjectiveInterface<HOST_OBJECTIVE>::Init(metadata, num_data);

    const int num_threads = OMP_NUM_THREADS();
    std::vector<uint16_t> thread_max_num_items_in_query(num_threads);
    Threading::For<data_size_t>(0, this->num_queries_, 1,
      [this, &thread_max_num_items_in_query] (int thread_index, data_size_t start, data_size_t end) {
        for (data_size_t query_index = start; query_index < end; ++query_index) {
          const data_size_t query_item_count = this->query_boundaries_[query_index + 1] - this->query_boundaries_[query_index];
          if (query_item_count > thread_max_num_items_in_query[thread_index]) {
            thread_max_num_items_in_query[thread_index] = query_item_count;
          }
        }
      });
    data_size_t max_items_in_query = 0;
    for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
      if (thread_max_num_items_in_query[thread_index] > max_items_in_query) {
        max_items_in_query = thread_max_num_items_in_query[thread_index];
      }
    }
    max_items_in_query_aligned_ = 1;
    --max_items_in_query;
    while (max_items_in_query > 0) {
      max_items_in_query >>= 1;
      max_items_in_query_aligned_ <<= 1;
    }
    if (max_items_in_query_aligned_ > 2048) {
      cuda_item_indices_buffer_.Resize(static_cast<size_t>(metadata.query_boundaries()[metadata.num_queries()]));
    }
    this->cuda_labels_ = metadata.cuda_metadata()->cuda_label();
    cuda_query_boundaries_ = metadata.cuda_metadata()->cuda_query_boundaries();
  }

 protected:
  // CUDA memory, held by this object
  CUDAVector<int> cuda_item_indices_buffer_;

  // CUDA memory, held by other objects
  const data_size_t* cuda_query_boundaries_;

  // Host memory
  int max_items_in_query_aligned_;
};


class CUDALambdarankNDCG: public CUDALambdaRankObjectiveInterface<LambdarankNDCG> {
 public:
  explicit CUDALambdarankNDCG(const Config& config);

  explicit CUDALambdarankNDCG(const std::vector<std::string>& strs);

  void Init(const Metadata& mdtadata, data_size_t num_data) override;

  ~CUDALambdarankNDCG();

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  // CUDA memory, held by this object
  CUDAVector<double> cuda_inverse_max_dcgs_;
  CUDAVector<double> cuda_label_gain_;
};


class CUDARankXENDCG : public CUDALambdaRankObjectiveInterface<RankXENDCG> {
 public:
  explicit CUDARankXENDCG(const Config& config);

  explicit CUDARankXENDCG(const std::vector<std::string>& strs);

  ~CUDARankXENDCG();

  void Init(const Metadata& metadata, data_size_t num_data) override;

 protected:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const;

  void GenerateItemRands() const;

  mutable std::vector<double> item_rands_;
  CUDAVector<double> cuda_item_rands_;
  CUDAVector<double> cuda_params_buffer_;
};


}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_SRC_OBJECTIVE_CUDA_CUDA_RANK_OBJECTIVE_HPP_
