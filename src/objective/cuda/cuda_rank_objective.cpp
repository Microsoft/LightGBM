/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <string>
#include <vector>

#include "cuda_rank_objective.hpp"

namespace LightGBM {

CUDALambdarankNDCG::CUDALambdarankNDCG(const Config& config): CUDALambdaRankObjectiveInterface<LambdarankNDCG>(config) {}

CUDALambdarankNDCG::CUDALambdarankNDCG(const std::vector<std::string>& strs): CUDALambdaRankObjectiveInterface<LambdarankNDCG>(strs) {}

CUDALambdarankNDCG::~CUDALambdarankNDCG() {}

void CUDALambdarankNDCG::Init(const Metadata& metadata, data_size_t num_data) {
  CUDALambdaRankObjectiveInterface<LambdarankNDCG>::Init(metadata, num_data);
  cuda_inverse_max_dcgs_.Resize(this->inverse_max_dcgs_.size());
  CopyFromHostToCUDADevice(cuda_inverse_max_dcgs_.RawData(), this->inverse_max_dcgs_.data(), this->inverse_max_dcgs_.size(), __FILE__, __LINE__);
  cuda_label_gain_.Resize(this->label_gain_.size());
  CopyFromHostToCUDADevice(cuda_label_gain_.RawData(), this->label_gain_.data(), this->label_gain_.size(), __FILE__, __LINE__);
}


CUDARankXENDCG::CUDARankXENDCG(const Config& config): CUDALambdaRankObjectiveInterface<RankXENDCG>(config) {}

CUDARankXENDCG::CUDARankXENDCG(const std::vector<std::string>& strs): CUDALambdaRankObjectiveInterface<RankXENDCG>(strs) {}

CUDARankXENDCG::~CUDARankXENDCG() {}

void CUDARankXENDCG::Init(const Metadata& metadata, data_size_t num_data) {
  CUDALambdaRankObjectiveInterface<RankXENDCG>::Init(metadata, num_data);
  for (data_size_t i = 0; i < num_queries_; ++i) {
    rands_.emplace_back(seed_ + i);
  }
  item_rands_.resize(num_data, 0.0f);
  cuda_item_rands_.Resize(static_cast<size_t>(num_data));
  if (max_items_in_query_aligned_ >= 2048) {
    cuda_params_buffer_.Resize(static_cast<size_t>(num_data_));
  }
}

void CUDARankXENDCG::GenerateItemRands() const {
  const int num_threads = OMP_NUM_THREADS();
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (data_size_t i = 0; i < num_queries_; ++i) {
    OMP_LOOP_EX_BEGIN();
    const data_size_t start = query_boundaries_[i];
    const data_size_t end = query_boundaries_[i + 1];
    for (data_size_t j = start; j < end; ++j) {
      item_rands_[j] = rands_[i].NextFloat();
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
}

}  // namespace LightGBM

#endif  // USE_CUDA
