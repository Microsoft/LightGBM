/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/metric.h>

#include <set>
#include <vector>

#include "nccl_gbdt.hpp"
#include "nccl_gbdt_component.hpp"

#ifdef USE_CUDA

namespace LightGBM {

template <typename GBDT_T>
NCCLGBDT<GBDT_T>::NCCLGBDT(): GBDT_T() {}

template <typename GBDT_T>
NCCLGBDT<GBDT_T>::~NCCLGBDT() {}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::Init(
  const Config* gbdt_config, const Dataset* train_data,
  const ObjectiveFunction* objective_function,
  const std::vector<const Metric*>& training_metrics) {
  GBDT_T::Init(gbdt_config, train_data, objective_function, training_metrics);

  this->tree_learner_.reset();

  nccl_topology_.reset(new NCCLTopology(this->config_->gpu_device_id, this->config_->num_gpu, this->config_->gpu_device_id_list, train_data->num_data()));

  nccl_topology_->InitNCCL();

  nccl_topology_->InitPerDevice<NCCLGBDTComponent>(&nccl_gbdt_components_);
  nccl_topology_->RunPerDevice<NCCLGBDTComponent, void>(nccl_gbdt_components_, [this, gbdt_config, train_data]
    (NCCLGBDTComponent* nccl_gbdt_component) { nccl_gbdt_component->Init(
      gbdt_config, train_data, this->num_tree_per_iteration_, this->boosting_on_gpu_, this->is_constant_hessian_);
  });
}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::BoostingThread(NCCLGBDTComponent* thread_data) {
  const ObjectiveFunction* objective_function = thread_data->objective_function();
  score_t* gradients = thread_data->gradients();
  score_t* hessians = thread_data->hessians();
  const double* score = thread_data->train_score_updater()->score();
  objective_function->GetGradients(score, gradients, hessians);
}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::Boosting() {
  Common::FunctionTimer fun_timer("NCCLGBDT::Boosting", global_timer);
  if (this->objective_function_ == nullptr) {
    Log::Fatal("No object function provided");
  }
  nccl_topology_->DispatchPerDevice<NCCLGBDTComponent>(&nccl_gbdt_components_, BoostingThread);
}

template <typename GBDT_T>
double NCCLGBDT<GBDT_T>::BoostFromAverage(int class_id, bool update_scorer) {
  double init_score = GBDT_T::BoostFromAverage(class_id, update_scorer);

  if (init_score != 0.0) {
    nccl_topology_->RunPerDevice<NCCLGBDTComponent, void>(nccl_gbdt_components_, [init_score, class_id] (NCCLGBDTComponent* thread_data) {
      thread_data->train_score_updater()->AddScore(init_score, class_id);
    });
  }

  return init_score;
}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::TrainTreeLearnerThread(NCCLGBDTComponent* thread_data, const int class_id, const bool is_first_tree) {
  const data_size_t num_data_in_gpu = thread_data->num_data_in_gpu();
  const score_t* gradients = thread_data->gradients() + class_id * num_data_in_gpu;
  const score_t* hessians = thread_data->hessians() + class_id * num_data_in_gpu;
  thread_data->SetTree(thread_data->tree_learner()->Train(gradients, hessians, is_first_tree));
}

template <typename GBDT_T>
bool NCCLGBDT<GBDT_T>::TrainOneIter(const score_t* gradients, const score_t* hessians) {
  Common::FunctionTimer fun_timer("NCCLGBDT::TrainOneIter", global_timer);
  std::vector<double> init_scores(this->num_tree_per_iteration_, 0.0);
  // boosting first
  if (gradients == nullptr || hessians == nullptr) {
    for (int cur_tree_id = 0; cur_tree_id < this->num_tree_per_iteration_; ++cur_tree_id) {
      init_scores[cur_tree_id] = BoostFromAverage(cur_tree_id, true);
    }
    Boosting();
  } else {
    nccl_topology_->RunPerDevice<NCCLGBDTComponent, void>(nccl_gbdt_components_, [this, gradients, hessians] (NCCLGBDTComponent* thread_data) {
      const data_size_t data_start_index = thread_data->data_start_index();
      const data_size_t num_data_in_gpu = thread_data->num_data_in_gpu();

      for (int class_id = 0; class_id < this->num_class_; ++class_id) {
        CopyFromHostToCUDADevice<score_t>(
          thread_data->gradients() + class_id * num_data_in_gpu,
          gradients + class_id * this->num_data_ + data_start_index, num_data_in_gpu, __FILE__, __LINE__);
        CopyFromHostToCUDADevice<score_t>(
          thread_data->hessians() + class_id * num_data_in_gpu,
          hessians + class_id * this->num_data_ + data_start_index, num_data_in_gpu, __FILE__, __LINE__);
      }
    });
  }

  bool should_continue = false;
  for (int cur_tree_id = 0; cur_tree_id < this->num_tree_per_iteration_; ++cur_tree_id) {
    if (this->class_need_train_[cur_tree_id] && this->train_data_->num_features() > 0) {
      if (this->data_sample_strategy_->is_use_subset() && this->data_sample_strategy_->bag_data_cnt() < this->num_data_) {
        Log::Fatal("Bagging is not supported for NCCLGBDT");
      }
      bool is_first_tree = this->models_.size() < static_cast<size_t>(this->num_tree_per_iteration_);
      nccl_topology_->DispatchPerDevice<NCCLGBDTComponent>(&nccl_gbdt_components_,
        [is_first_tree, cur_tree_id] (NCCLGBDTComponent* thread_data) -> void {
          TrainTreeLearnerThread(thread_data, cur_tree_id, is_first_tree);
      });
    }

    nccl_topology_->DispatchPerDevice<NCCLGBDTComponent>(&nccl_gbdt_components_, [cur_tree_id, this, init_scores] (NCCLGBDTComponent* thread_data) -> void {
      this->UpdateScoreThread(thread_data, cur_tree_id, this->config_->learning_rate, init_scores[cur_tree_id]);
    });

    nccl_topology_->RunOnMasterDevice<NCCLGBDTComponent, void>(nccl_gbdt_components_, [&should_continue, this, cur_tree_id] (NCCLGBDTComponent* thread_data) -> void {
      if (thread_data->new_tree()->num_leaves() > 1) {
        should_continue = true;
      }
      for (auto& score_updater : this->valid_score_updater_) {
        score_updater->AddScore(thread_data->new_tree(), cur_tree_id);
      }
    });

    if (!should_continue) {
      if (this->models_.size() < static_cast<size_t>(this->num_tree_per_iteration_)) {
        Log::Warning("Training stopped with no splits.");
      }
    }

    // add model
    nccl_topology_->RunOnMasterDevice<NCCLGBDTComponent, void>(nccl_gbdt_components_, [this] (NCCLGBDTComponent* thread_data) -> void {
      this->models_.emplace_back(thread_data->release_new_tree());
    });

    nccl_topology_->RunOnNonMasterDevice<NCCLGBDTComponent, void>(nccl_gbdt_components_, [this] (NCCLGBDTComponent* thread_data) -> void {
      thread_data->clear_new_tree();
    });
  }

  if (!should_continue) {
    Log::Warning("Stopped training because there are no more leaves that meet the split requirements");
    if (this->models_.size() > static_cast<size_t>(this->num_tree_per_iteration_)) {
      for (int cur_tree_id = 0; cur_tree_id < this->num_tree_per_iteration_; ++cur_tree_id) {
        this->models_.pop_back();
      }
    }
    return true;
  }

  ++this->iter_;
  return false;
}

template <typename GBDT_T>
void NCCLGBDT<GBDT_T>::UpdateScoreThread(NCCLGBDTComponent* thread_data, const int cur_tree_id, const double shrinkage_rate, const double init_score) {
  if (thread_data->new_tree()->num_leaves() > 1) {
    // TODO(shiyu1994): implement bagging
    if (thread_data->objective_function() != nullptr && thread_data->objective_function()->IsRenewTreeOutput()) {
      // TODO(shiyu1994): implement renewing
    }
    thread_data->new_tree()->Shrinkage(shrinkage_rate);
    thread_data->train_score_updater()->AddScore(
      thread_data->tree_learner(),
      thread_data->new_tree(),
      cur_tree_id);
    if (std::fabs(init_score) > kEpsilon) {
      thread_data->new_tree()->AddBias(init_score);
    }
  }
}

template <typename GBDT_T>
std::vector<double> NCCLGBDT<GBDT_T>::EvalOneMetric(const Metric* metric, const double* score, const data_size_t num_data) const {
  if (score == this->train_score_updater_->score()) {
    // delegate to per gpu train score updater
    std::vector<double> tmp_score(num_data * this->num_class_, 0.0f);

    nccl_topology_->RunPerDevice<NCCLGBDTComponent, void>(nccl_gbdt_components_, [this, &tmp_score] (NCCLGBDTComponent* thread_data) {
      const data_size_t data_start = thread_data->data_start_index();
      const data_size_t num_data_in_gpu = thread_data->num_data_in_gpu();
      for (int class_id = 0; class_id < this->num_class_; ++class_id) {
        CopyFromCUDADeviceToHost<double>(tmp_score.data() + class_id * this->num_data_ + data_start,
          thread_data->train_score_updater()->score() + class_id * num_data_in_gpu,
          static_cast<size_t>(num_data_in_gpu), __FILE__, __LINE__);
      }
    });

    return metric->Eval(tmp_score.data(), this->objective_function_);
  } else {
    return GBDT_T::EvalOneMetric(metric, score, num_data);
  }
}

template class NCCLGBDT<GBDT>;

}  // namespace LightGBM

#endif  // USE_CUDA
