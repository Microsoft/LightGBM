/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_MULTICLASS_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_MULTICLASS_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <cmath>
#include <utility>
#include <vector>

#include "gather_eval_data.hpp"

namespace LightGBM {
/*!
* \brief Metric for multiclass task.
* Use static class "PointWiseLossCalculator" to calculate loss point-wise
*/
template<typename PointWiseLossCalculator>
class MulticlassMetric: public Metric {
 public:
  explicit MulticlassMetric(const Config& config) :config_(config) {
    num_class_ = config.num_class;
  }

  virtual ~MulticlassMetric() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back(PointWiseLossCalculator::Name(config_));
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0;
    int num_tree_per_iteration = num_class_;
    int num_pred_per_row = num_class_;
    if (objective != nullptr) {
      num_tree_per_iteration = objective->NumModelPerIteration();
      num_pred_per_row = objective->NumPredictOneRow();
    }
    if (objective != nullptr) {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> raw_score(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
          }
          std::vector<double> rec(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rec.data());
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], &rec, config_);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> raw_score(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            raw_score[k] = static_cast<double>(score[idx]);
          }
          std::vector<double> rec(num_pred_per_row);
          objective->ConvertOutput(raw_score.data(), rec.data());
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], &rec, config_) * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rec(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rec[k] = static_cast<double>(score[idx]);
          }
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], &rec, config_);
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          std::vector<double> rec(num_tree_per_iteration);
          for (int k = 0; k < num_tree_per_iteration; ++k) {
            size_t idx = static_cast<size_t>(num_data_) * k + i;
            rec[k] = static_cast<double>(score[idx]);
          }
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], &rec, config_) * weights_[i];
        }
      }
    }
    double sum_weights = sum_weights_;
    if (Network::num_machines() > 1) {
      sum_loss = Network::GlobalSyncUpBySum(sum_loss);
      sum_weights = Network::GlobalSyncUpBySum(sum_weights);
      if (sum_weights <= 0.0f) {
        Log::Fatal("Validation data has no positive total weight");
      }
    }
    double loss = sum_loss / sum_weights;
    return std::vector<double>(1, loss);
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of this test set */
  std::vector<std::string> name_;
  int num_class_;
  /*! \brief config parameters*/
  Config config_;
};

/*! \brief top-k error for multiclass task; if k=1 (default) this is the usual multi-error */
class MultiErrorMetric: public MulticlassMetric<MultiErrorMetric> {
 public:
  explicit MultiErrorMetric(const Config& config) :MulticlassMetric<MultiErrorMetric>(config) {}

  inline static double LossOnPoint(label_t label, std::vector<double>* score, const Config& config) {
    size_t k = static_cast<size_t>(label);
    auto& ref_score = *score;
    int num_larger = 0;
    for (size_t i = 0; i < score->size(); ++i) {
      if (ref_score[i] >= ref_score[k]) ++num_larger;
      if (num_larger > config.multi_error_top_k) return 1.0f;
    }
    return 0.0f;
  }

  inline static const std::string Name(const Config& config) {
    if (config.multi_error_top_k == 1) {
      return "multi_error";
    } else {
      return "multi_error@" + std::to_string(config.multi_error_top_k);
    }
  }
};

/*! \brief Logloss for multiclass task */
class MultiSoftmaxLoglossMetric: public MulticlassMetric<MultiSoftmaxLoglossMetric> {
 public:
  explicit MultiSoftmaxLoglossMetric(const Config& config) :MulticlassMetric<MultiSoftmaxLoglossMetric>(config) {}

  inline static double LossOnPoint(label_t label, std::vector<double>* score, const Config&) {
    size_t k = static_cast<size_t>(label);
    auto& ref_score = *score;
    if (ref_score[k] > kEpsilon) {
      return static_cast<double>(-std::log(ref_score[k]));
    } else {
      return -std::log(kEpsilon);
    }
  }

  inline static const std::string Name(const Config&) {
    return "multi_logloss";
  }
};

/*! \brief AUC mu for multiclass task*/
class AucMuMetric : public Metric {
 public:
  explicit AucMuMetric(const Config& config) : config_(config) {
    num_class_ = config.num_class;
    class_weights_ = config.auc_mu_weights_matrix;
  }

  virtual ~AucMuMetric() {}

  const std::vector<std::string>& GetName() const override { return name_; }

  double factor_to_bigger_better() const override { return 1.0f; }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("auc_mu");

    num_data_ = num_data;
    label_ = metadata.label();

    // get weights
    weights_ = metadata.weights();
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }

    // sort the data indices by true class
    // In distributed mode these are recomputed from the gathered data in Eval;
    // only compute locally when running single-machine.
    if (Network::num_machines() <= 1) {
      sorted_data_idx_ = std::vector<data_size_t>(num_data_, 0);
      for (data_size_t i = 0; i < num_data_; ++i) {
        sorted_data_idx_[i] = i;
      }
      Common::ParallelSort(sorted_data_idx_.begin(), sorted_data_idx_.end(),
        [this](data_size_t a, data_size_t b) { return label_[a] < label_[b]; });

      // get size of each class
      class_sizes_ = std::vector<data_size_t>(num_class_, 0);
      for (data_size_t i = 0; i < num_data_; ++i) {
        data_size_t curr_label = static_cast<data_size_t>(label_[i]);
        ++class_sizes_[curr_label];
      }

      // get total weight of data in each class
      class_data_weights_ = std::vector<double>(num_class_, 0);
      if (weights_ != nullptr) {
        for (data_size_t i = 0; i < num_data_; ++i) {
          data_size_t curr_label = static_cast<data_size_t>(label_[i]);
          class_data_weights_[curr_label] += weights_[i];
        }
      }
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    if (num_data_ == 0 && Network::num_machines() == 1) {
      return std::vector<double>(1, 1.0f);
    }
    if (Network::num_machines() > 1) {
      std::vector<label_t> all_labels;
      std::vector<double> all_scores;
      std::vector<label_t> all_weights;
      bool has_weights;
      GatherEvalData(label_, score, weights_, num_data_, num_class_,
                     all_labels, all_scores, all_weights, has_weights);
      data_size_t n = static_cast<data_size_t>(all_labels.size());
      if (n == 0) {
        return std::vector<double>(1, 1.0f);
      }
      const label_t* lbl = all_labels.data();
      const double* scr = all_scores.data();
      const label_t* w = has_weights ? all_weights.data() : nullptr;

      // Recompute sorted by label
      std::vector<data_size_t> sorted_idx(n);
      for (data_size_t i = 0; i < n; ++i) sorted_idx[i] = i;
      Common::ParallelSort(sorted_idx.begin(), sorted_idx.end(),
          [lbl](data_size_t a, data_size_t b) { return lbl[a] < lbl[b]; });

      // Recompute class sizes and weights
      std::vector<data_size_t> cls_sizes(num_class_, 0);
      std::vector<double> cls_data_w(num_class_, 0);
      for (data_size_t i = 0; i < n; ++i) {
        data_size_t c = static_cast<data_size_t>(lbl[i]);
        ++cls_sizes[c];
        if (w) cls_data_w[c] += w[i];
      }

      auto S = std::vector<std::vector<double>>(num_class_, std::vector<double>(num_class_, 0));
      int i_start = 0;
      for (int i = 0; i < num_class_; ++i) {
        int j_start = i_start + cls_sizes[i];
        for (int j = i + 1; j < num_class_; ++j) {
          std::vector<double> curr_v;
          for (int k = 0; k < num_class_; ++k) {
            curr_v.emplace_back(class_weights_[i][k] - class_weights_[j][k]);
          }
          double t1 = curr_v[i] - curr_v[j];
          std::vector<data_size_t> class_i_j_indices;
          class_i_j_indices.assign(sorted_idx.begin() + i_start, sorted_idx.begin() + i_start + cls_sizes[i]);
          class_i_j_indices.insert(class_i_j_indices.end(),
            sorted_idx.begin() + j_start, sorted_idx.begin() + j_start + cls_sizes[j]);
          std::vector<std::pair<data_size_t, double>> dist;
          for (data_size_t k = 0; static_cast<size_t>(k) < class_i_j_indices.size(); ++k) {
            data_size_t a = class_i_j_indices[k];
            double v_a = 0;
            for (int m = 0; m < num_class_; ++m) {
              v_a += curr_v[m] * scr[n * m + a];
            }
            dist.push_back(std::pair<data_size_t, double>(a, t1 * v_a));
          }
          Common::ParallelSort(dist.begin(), dist.end(),
            [lbl](std::pair<data_size_t, double> a, std::pair<data_size_t, double> b) {
            if (std::fabs(a.second - b.second) < kEpsilon) {
              return lbl[a.first] > lbl[b.first];
            } else if (a.second < b.second) {
              return true;
            } else {
              return false;
            }
          });
          double num_j = 0, last_j_dist = 0, num_current_j = 0;
          if (w == nullptr) {
            for (size_t k = 0; k < dist.size(); ++k) {
              data_size_t a = dist[k].first;
              double curr_dist = dist[k].second;
              if (lbl[a] == i) {
                if (std::fabs(curr_dist - last_j_dist) < kEpsilon) {
                  S[i][j] += num_j - 0.5 * num_current_j;
                } else {
                  S[i][j] += num_j;
                }
              } else {
                ++num_j;
                if (std::fabs(curr_dist - last_j_dist) < kEpsilon) {
                  ++num_current_j;
                } else {
                  last_j_dist = dist[k].second;
                  num_current_j = 1;
                }
              }
            }
          } else {
            for (size_t k = 0; k < dist.size(); ++k) {
              data_size_t a = dist[k].first;
              double curr_dist = dist[k].second;
              double curr_weight = w[a];
              if (lbl[a] == i) {
                if (std::fabs(curr_dist - last_j_dist) < kEpsilon) {
                  S[i][j] += curr_weight * (num_j - 0.5 * num_current_j);
                } else {
                  S[i][j] += curr_weight * num_j;
                }
              } else {
                num_j += curr_weight;
                if (std::fabs(curr_dist - last_j_dist) < kEpsilon) {
                  num_current_j += curr_weight;
                } else {
                  last_j_dist = dist[k].second;
                  num_current_j = curr_weight;
                }
              }
            }
          }
          j_start += cls_sizes[j];
        }
        i_start += cls_sizes[i];
      }
      double ans = 0;
      for (int i = 0; i < num_class_; ++i) {
        for (int j = i + 1; j < num_class_; ++j) {
          if (w == nullptr) {
            ans += (S[i][j] / cls_sizes[i]) / cls_sizes[j];
          } else {
            ans += (S[i][j] / cls_data_w[i]) / cls_data_w[j];
          }
        }
      }
      ans = (2.0 * ans / num_class_) / (num_class_ - 1);
      return std::vector<double>(1, ans);
    }
    // the notation follows that used in the paper introducing the auc-mu metric:
    // https://proceedings.mlr.press/v97/kleiman19a.html

    auto S = std::vector<std::vector<double>>(num_class_, std::vector<double>(num_class_, 0));
    int i_start = 0;
    for (int i = 0; i < num_class_; ++i) {
      int j_start = i_start + class_sizes_[i];
      for (int j = i + 1; j < num_class_; ++j) {
        std::vector<double> curr_v;
        for (int k = 0; k < num_class_; ++k) {
          curr_v.emplace_back(class_weights_[i][k] - class_weights_[j][k]);
        }
        double t1 = curr_v[i] - curr_v[j];
        // extract the data indices belonging to class i or j
        std::vector<data_size_t> class_i_j_indices;
        class_i_j_indices.assign(sorted_data_idx_.begin() + i_start, sorted_data_idx_.begin() + i_start + class_sizes_[i]);
        class_i_j_indices.insert(class_i_j_indices.end(),
          sorted_data_idx_.begin() + j_start, sorted_data_idx_.begin() + j_start + class_sizes_[j]);
        // sort according to distance from separating hyperplane
        std::vector<std::pair<data_size_t, double>> dist;
        for (data_size_t k = 0; static_cast<size_t>(k) < class_i_j_indices.size(); ++k) {
          data_size_t a = class_i_j_indices[k];
          double v_a = 0;
          for (int m = 0; m < num_class_; ++m) {
            v_a += curr_v[m] * score[num_data_ * m + a];
          }
          dist.push_back(std::pair<data_size_t, double>(a, t1 * v_a));
        }
        Common::ParallelSort(dist.begin(), dist.end(),
          [this](std::pair<data_size_t, double> a, std::pair<data_size_t, double> b) {
          // if scores are equal, put j class first
          if (std::fabs(a.second - b.second) < kEpsilon) {
            return label_[a.first] > label_[b.first];
          } else if (a.second < b.second) {
            return true;
          } else {
            return false;
          }
        });
        // calculate AUC
        double num_j = 0;
        double last_j_dist = 0;
        double num_current_j = 0;
        if (weights_ == nullptr) {
          for (size_t k = 0; k < dist.size(); ++k) {
            data_size_t a = dist[k].first;
            double curr_dist = dist[k].second;
            if (label_[a] == i) {
              if (std::fabs(curr_dist - last_j_dist) < kEpsilon) {
                S[i][j] += num_j - 0.5 * num_current_j;  // members of class j with same distance as a contribute 0.5
              } else {
                S[i][j] += num_j;
              }
            } else {
              ++num_j;
              if (std::fabs(curr_dist - last_j_dist) < kEpsilon) {
                ++num_current_j;
              } else {
                last_j_dist = dist[k].second;
                num_current_j = 1;
              }
            }
          }
        } else {
          for (size_t k = 0; k < dist.size(); ++k) {
            data_size_t a = dist[k].first;
            double curr_dist = dist[k].second;
            double curr_weight = weights_[a];
            if (label_[a] == i) {
              if (std::fabs(curr_dist - last_j_dist) < kEpsilon) {
                S[i][j] += curr_weight * (num_j - 0.5 * num_current_j);  // members of class j with same distance as a contribute 0.5
              } else {
                S[i][j] += curr_weight * num_j;
              }
            } else {
              num_j += curr_weight;
              if (std::fabs(curr_dist - last_j_dist) < kEpsilon) {
                num_current_j += curr_weight;
              } else {
                last_j_dist = dist[k].second;
                num_current_j = curr_weight;
              }
            }
          }
        }
        j_start += class_sizes_[j];
      }
      i_start += class_sizes_[i];
    }
    double ans = 0;
    for (int i = 0; i < num_class_; ++i) {
      for (int j = i + 1; j < num_class_; ++j) {
        if (weights_ == nullptr) {
          ans += (S[i][j] / class_sizes_[i]) / class_sizes_[j];
        } else {
          ans += (S[i][j] / class_data_weights_[i]) / class_data_weights_[j];
        }
      }
    }
    ans = (2.0 * ans / num_class_) / (num_class_ - 1);
    return std::vector<double>(1, ans);
  }

 private:
  /*! \brief Number of data*/
  data_size_t num_data_;
  /*! \brief Pointer to label*/
  const label_t* label_;
  /*! \brief Name of this metric*/
  std::vector<std::string> name_;
  /*! \brief Number of classes*/
  int num_class_;
  /*! \brief Class auc-mu weights*/
  std::vector<std::vector<double>> class_weights_;
  /*! \brief Data weights */
  const label_t* weights_;
  /*! \brief Sum of data weights */
  double sum_weights_;
  /*! \brief Sum of data weights in each class*/
  std::vector<double> class_data_weights_;
  /*! \brief Number of data in each class*/
  std::vector<data_size_t> class_sizes_;
  /*! \brief config parameters*/
  Config config_;
  /*! \brief index to data, sorted by true class*/
  std::vector<data_size_t> sorted_data_idx_;
};

}  // namespace LightGBM
#endif   // LIGHTGBM_SRC_METRIC_MULTICLASS_METRIC_HPP_
