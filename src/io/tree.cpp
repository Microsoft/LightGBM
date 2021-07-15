/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/tree.h>

#include <LightGBM/dataset.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/threading.h>

#include <functional>
#include <iomanip>
#include <sstream>

namespace LightGBM {

Tree::Tree(int max_leaves, bool track_branch_features, bool is_linear)
  :max_leaves_(max_leaves), track_branch_features_(track_branch_features) {
  left_child_.resize(max_leaves_ - 1);
  right_child_.resize(max_leaves_ - 1);
  split_feature_inner_.resize(max_leaves_ - 1);
  split_feature_.resize(max_leaves_ - 1);
  threshold_in_bin_.resize(max_leaves_ - 1);
  threshold_.resize(max_leaves_ - 1);
  decision_type_.resize(max_leaves_ - 1, 0);
  split_gain_.resize(max_leaves_ - 1);
  leaf_parent_.resize(max_leaves_);
  leaf_value_.resize(max_leaves_);
  leaf_weight_.resize(max_leaves_);
  leaf_count_.resize(max_leaves_);
  internal_value_.resize(max_leaves_ - 1);
  internal_weight_.resize(max_leaves_ - 1);
  internal_count_.resize(max_leaves_ - 1);
  leaf_depth_.resize(max_leaves_);
  if (track_branch_features_) {
    branch_features_ = std::vector<std::vector<int>>(max_leaves_);
  }
  // root is in the depth 0
  leaf_depth_[0] = 0;
  num_leaves_ = 1;
  leaf_value_[0] = 0.0f;
  leaf_weight_[0] = 0.0f;
  leaf_parent_[0] = -1;
  shrinkage_ = 1.0f;
  num_cat_ = 0;
  cat_boundaries_.push_back(0);
  cat_boundaries_inner_.push_back(0);
  max_depth_ = -1;
  is_linear_ = is_linear;
  if (is_linear_) {
    leaf_coeff_.resize(max_leaves_);
    leaf_const_ = std::vector<double>(max_leaves_, 0);
    leaf_features_.resize(max_leaves_);
    leaf_features_inner_.resize(max_leaves_);
  }
}

int Tree::Split(int leaf, int feature, int real_feature, uint32_t threshold_bin,
                double threshold_double, double left_value, double right_value,
                int left_cnt, int right_cnt, double left_weight, double right_weight, float gain,
                MissingType missing_type, bool default_left) {
  Split(leaf, feature, real_feature, left_value, right_value, left_cnt, right_cnt, left_weight, right_weight, gain);
  int new_node_idx = num_leaves_ - 1;
  decision_type_[new_node_idx] = 0;
  SetDecisionType(&decision_type_[new_node_idx], false, kCategoricalMask);
  SetDecisionType(&decision_type_[new_node_idx], default_left, kDefaultLeftMask);
  SetMissingType(&decision_type_[new_node_idx], static_cast<uint8_t>(missing_type));
  threshold_in_bin_[new_node_idx] = threshold_bin;
  threshold_[new_node_idx] = threshold_double;
  ++num_leaves_;
  return num_leaves_ - 1;
}

int Tree::SplitCategorical(int leaf, int feature, int real_feature, const uint32_t* threshold_bin, int num_threshold_bin,
                           const uint32_t* threshold, int num_threshold, double left_value, double right_value,
                           data_size_t left_cnt, data_size_t right_cnt, double left_weight, double right_weight, float gain, MissingType missing_type) {
  Split(leaf, feature, real_feature, left_value, right_value, left_cnt, right_cnt, left_weight, right_weight, gain);
  int new_node_idx = num_leaves_ - 1;
  decision_type_[new_node_idx] = 0;
  SetDecisionType(&decision_type_[new_node_idx], true, kCategoricalMask);
  SetMissingType(&decision_type_[new_node_idx], static_cast<uint8_t>(missing_type));
  threshold_in_bin_[new_node_idx] = num_cat_;
  threshold_[new_node_idx] = num_cat_;
  ++num_cat_;
  cat_boundaries_.push_back(cat_boundaries_.back() + num_threshold);
  for (int i = 0; i < num_threshold; ++i) {
    cat_threshold_.push_back(threshold[i]);
  }
  cat_boundaries_inner_.push_back(cat_boundaries_inner_.back() + num_threshold_bin);
  for (int i = 0; i < num_threshold_bin; ++i) {
    cat_threshold_inner_.push_back(threshold_bin[i]);
  }
  ++num_leaves_;
  return num_leaves_ - 1;
}

#define PredictionFun(niter, fidx_in_iter, start_pos, decision_fun, iter_idx, \
                      data_idx)                                               \
  std::vector<std::unique_ptr<BinIterator>> iter((niter));                    \
  for (int i = 0; i < (niter); ++i) {                                         \
    iter[i].reset(data->FeatureIterator((fidx_in_iter)));                     \
    iter[i]->Reset((start_pos));                                              \
  }                                                                           \
  for (data_size_t i = start; i < end; ++i) {                                 \
    int node = 0;                                                             \
    while (node >= 0) {                                                       \
      node = decision_fun(iter[(iter_idx)]->Get((data_idx)), node,            \
                          default_bins[node], max_bins[node]);                \
    }                                                                         \
    score[(data_idx)] += static_cast<double>(leaf_value_[~node]);             \
  }\


#define PredictionFunLinear(niter, fidx_in_iter, start_pos, decision_fun,     \
                            iter_idx, data_idx)                               \
  std::vector<std::unique_ptr<BinIterator>> iter((niter));                    \
  for (int i = 0; i < (niter); ++i) {                                         \
    iter[i].reset(data->FeatureIterator((fidx_in_iter)));                     \
    iter[i]->Reset((start_pos));                                              \
  }                                                                           \
  for (data_size_t i = start; i < end; ++i) {                                 \
    int node = 0;                                                             \
    if (num_leaves_ > 1) {                                                    \
      while (node >= 0) {                                                     \
        node = decision_fun(iter[(iter_idx)]->Get((data_idx)), node,          \
                            default_bins[node], max_bins[node]);              \
      }                                                                       \
      node = ~node;                                                           \
    }                                                                         \
    double add_score = leaf_const_[node];                                     \
    bool nan_found = false;                                                   \
    const double* coeff_ptr = leaf_coeff_[node].data();                       \
    const float** data_ptr = feat_ptr[node].data();                           \
    for (size_t j = 0; j < leaf_features_inner_[node].size(); ++j) {          \
       float feat_val = data_ptr[j][(data_idx)];                              \
       if (std::isnan(feat_val)) {                                            \
          nan_found = true;                                                   \
          break;                                                              \
       }                                                                      \
       add_score += coeff_ptr[j] * feat_val;                                  \
    }                                                                         \
    if (nan_found) {                                                          \
       score[(data_idx)] += leaf_value_[node];                                \
    } else {                                                                  \
      score[(data_idx)] += add_score;                                         \
    }                                                                         \
}\


void Tree::AddPredictionToScore(const Dataset* data, data_size_t num_data, double* score) const {
  if (!is_linear_ && num_leaves_ <= 1) {
    if (leaf_value_[0] != 0.0f) {
#pragma omp parallel for schedule(static, 512) if (num_data >= 1024)
      for (data_size_t i = 0; i < num_data; ++i) {
        score[i] += leaf_value_[0];
      }
    }
    return;
  }
  std::vector<uint32_t> default_bins(num_leaves_ - 1);
  std::vector<uint32_t> max_bins(num_leaves_ - 1);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    const int fidx = split_feature_inner_[i];
    auto bin_mapper = data->FeatureBinMapper(fidx);
    default_bins[i] = bin_mapper->GetDefaultBin();
    max_bins[i] = bin_mapper->num_bin() - 1;
  }
  if (is_linear_) {
    std::vector<std::vector<const float*>> feat_ptr(num_leaves_);
    for (int leaf_num = 0; leaf_num < num_leaves_; ++leaf_num) {
      for (int feat : leaf_features_inner_[leaf_num]) {
        feat_ptr[leaf_num].push_back(data->raw_index(feat));
      }
    }
    if (num_cat_ > 0) {
      if (data->num_features() > num_leaves_ - 1) {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &default_bins, &max_bins, &feat_ptr]
        (int, data_size_t start, data_size_t end) {
          PredictionFunLinear(num_leaves_ - 1, split_feature_inner_[i], start, DecisionInner, node, i);
        });
      } else {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &default_bins, &max_bins, &feat_ptr]
        (int, data_size_t start, data_size_t end) {
          PredictionFunLinear(data->num_features(), i, start, DecisionInner, split_feature_inner_[node], i);
        });
      }
    } else {
      if (data->num_features() > num_leaves_ - 1) {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &default_bins, &max_bins, &feat_ptr]
        (int, data_size_t start, data_size_t end) {
          PredictionFunLinear(num_leaves_ - 1, split_feature_inner_[i], start, NumericalDecisionInner, node, i);
        });
      } else {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &default_bins, &max_bins, &feat_ptr]
        (int, data_size_t start, data_size_t end) {
          PredictionFunLinear(data->num_features(), i, start, NumericalDecisionInner, split_feature_inner_[node], i);
        });
      }
    }
  } else {
    if (num_cat_ > 0) {
      if (data->num_features() > num_leaves_ - 1) {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &default_bins, &max_bins]
        (int, data_size_t start, data_size_t end) {
          PredictionFun(num_leaves_ - 1, split_feature_inner_[i], start, DecisionInner, node, i);
        });
      } else {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &default_bins, &max_bins]
        (int, data_size_t start, data_size_t end) {
          PredictionFun(data->num_features(), i, start, DecisionInner, split_feature_inner_[node], i);
        });
      }
    } else {
      if (data->num_features() > num_leaves_ - 1) {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &default_bins, &max_bins]
        (int, data_size_t start, data_size_t end) {
          PredictionFun(num_leaves_ - 1, split_feature_inner_[i], start, NumericalDecisionInner, node, i);
        });
      } else {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, &default_bins, &max_bins]
        (int, data_size_t start, data_size_t end) {
          PredictionFun(data->num_features(), i, start, NumericalDecisionInner, split_feature_inner_[node], i);
        });
      }
    }
  }
}

void Tree::AddPredictionToScore(const Dataset* data,
  const data_size_t* used_data_indices,
  data_size_t num_data, double* score) const {
  if (!is_linear_ && num_leaves_ <= 1) {
    if (leaf_value_[0] != 0.0f) {
#pragma omp parallel for schedule(static, 512) if (num_data >= 1024)
      for (data_size_t i = 0; i < num_data; ++i) {
        score[used_data_indices[i]] += leaf_value_[0];
      }
    }
    return;
  }
  std::vector<uint32_t> default_bins(num_leaves_ - 1);
  std::vector<uint32_t> max_bins(num_leaves_ - 1);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    const int fidx = split_feature_inner_[i];
    auto bin_mapper = data->FeatureBinMapper(fidx);
    default_bins[i] = bin_mapper->GetDefaultBin();
    max_bins[i] = bin_mapper->num_bin() - 1;
  }
  if (is_linear_) {
    std::vector<std::vector<const float*>> feat_ptr(num_leaves_);
    for (int leaf_num = 0; leaf_num < num_leaves_; ++leaf_num) {
      for (int feat : leaf_features_inner_[leaf_num]) {
        feat_ptr[leaf_num].push_back(data->raw_index(feat));
      }
    }
    if (num_cat_ > 0) {
      if (data->num_features() > num_leaves_ - 1) {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &default_bins, &max_bins, &feat_ptr]
        (int, data_size_t start, data_size_t end) {
          PredictionFunLinear(num_leaves_ - 1, split_feature_inner_[i], used_data_indices[start], DecisionInner,
                              node, used_data_indices[i]);
        });
      } else {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &default_bins, &max_bins, &feat_ptr]
        (int, data_size_t start, data_size_t end) {
          PredictionFunLinear(data->num_features(), i, used_data_indices[start], DecisionInner, split_feature_inner_[node], used_data_indices[i]);
        });
      }
    } else {
      if (data->num_features() > num_leaves_ - 1) {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &default_bins, &max_bins, &feat_ptr]
        (int, data_size_t start, data_size_t end) {
          PredictionFunLinear(num_leaves_ - 1, split_feature_inner_[i], used_data_indices[start], NumericalDecisionInner,
                              node, used_data_indices[i]);
        });
      } else {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &default_bins, &max_bins, &feat_ptr]
        (int, data_size_t start, data_size_t end) {
          PredictionFunLinear(data->num_features(), i, used_data_indices[start], NumericalDecisionInner,
                              split_feature_inner_[node], used_data_indices[i]);
        });
      }
    }
  } else {
    if (num_cat_ > 0) {
      if (data->num_features() > num_leaves_ - 1) {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &default_bins, &max_bins]
        (int, data_size_t start, data_size_t end) {
          PredictionFun(num_leaves_ - 1, split_feature_inner_[i], used_data_indices[start], DecisionInner, node, used_data_indices[i]);
        });
      } else {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &default_bins, &max_bins]
        (int, data_size_t start, data_size_t end) {
          PredictionFun(data->num_features(), i, used_data_indices[start], DecisionInner, split_feature_inner_[node], used_data_indices[i]);
        });
      }
    } else {
      if (data->num_features() > num_leaves_ - 1) {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &default_bins, &max_bins]
        (int, data_size_t start, data_size_t end) {
          PredictionFun(num_leaves_ - 1, split_feature_inner_[i], used_data_indices[start], NumericalDecisionInner, node, used_data_indices[i]);
        });
      } else {
        Threading::For<data_size_t>(0, num_data, 512, [this, &data, score, used_data_indices, &default_bins, &max_bins]
        (int, data_size_t start, data_size_t end) {
          PredictionFun(data->num_features(), i, used_data_indices[start], NumericalDecisionInner, split_feature_inner_[node], used_data_indices[i]);
        });
      }
    }
  }
}

#undef PredictionFun
#undef PredictionFunLinear

double Tree::GetUpperBoundValue() const {
  double upper_bound = leaf_value_[0];
  for (int i = 1; i < num_leaves_; ++i) {
    if (leaf_value_[i] > upper_bound) {
      upper_bound = leaf_value_[i];
    }
  }
  return upper_bound;
}

double Tree::GetLowerBoundValue() const {
  double lower_bound = leaf_value_[0];
  for (int i = 1; i < num_leaves_; ++i) {
    if (leaf_value_[i] < lower_bound) {
      lower_bound = leaf_value_[i];
    }
  }
  return lower_bound;
}

std::string Tree::ToString() const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);

  #if ((defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__)))
  using CommonLegacy::ArrayToString;  // Slower & unsafe regarding locale.
  #else
  using CommonC::ArrayToString;
  #endif

  str_buf << "num_leaves=" << num_leaves_ << '\n';
  str_buf << "num_cat=" << num_cat_ << '\n';
  str_buf << "split_feature="
    << ArrayToString(split_feature_, num_leaves_ - 1) << '\n';
  str_buf << "split_gain="
    << ArrayToString(split_gain_, num_leaves_ - 1) << '\n';
  str_buf << "threshold="
    << ArrayToString<true>(threshold_, num_leaves_ - 1) << '\n';
  str_buf << "decision_type="
    << ArrayToString(Common::ArrayCast<uint8_t, int>(decision_type_), num_leaves_ - 1) << '\n';
  str_buf << "left_child="
    << ArrayToString(left_child_, num_leaves_ - 1) << '\n';
  str_buf << "right_child="
    << ArrayToString(right_child_, num_leaves_ - 1) << '\n';
  str_buf << "leaf_value="
    << ArrayToString<true>(leaf_value_, num_leaves_) << '\n';
  str_buf << "leaf_weight="
    << ArrayToString<true>(leaf_weight_, num_leaves_) << '\n';
  str_buf << "leaf_count="
    << ArrayToString(leaf_count_, num_leaves_) << '\n';
  str_buf << "internal_value="
    << ArrayToString(internal_value_, num_leaves_ - 1) << '\n';
  str_buf << "internal_weight="
    << ArrayToString(internal_weight_, num_leaves_ - 1) << '\n';
  str_buf << "internal_count="
    << ArrayToString(internal_count_, num_leaves_ - 1) << '\n';
  if (num_cat_ > 0) {
    str_buf << "cat_boundaries="
      << ArrayToString(cat_boundaries_, num_cat_ + 1) << '\n';
    str_buf << "cat_threshold="
      << ArrayToString(cat_threshold_, cat_threshold_.size()) << '\n';
  }
  str_buf << "is_linear=" << is_linear_ << '\n';

  if (is_linear_) {
    str_buf << "leaf_const="
      << ArrayToString<true>(leaf_const_, num_leaves_) << '\n';
    std::vector<int> num_feat(num_leaves_);
    for (int i = 0; i < num_leaves_; ++i) {
      num_feat[i] = static_cast<int>(leaf_coeff_[i].size());
    }
    str_buf << "num_features="
      << ArrayToString(num_feat, num_leaves_) << '\n';
    str_buf << "leaf_features=";
    for (int i = 0; i < num_leaves_; ++i) {
      if (num_feat[i] > 0) {
        str_buf << ArrayToString(leaf_features_[i], leaf_features_[i].size()) << ' ';
      }
      str_buf << ' ';
    }
    str_buf << '\n';
    str_buf << "leaf_coeff=";
    for (int i = 0; i < num_leaves_; ++i) {
      if (num_feat[i] > 0) {
        str_buf << ArrayToString<true>(leaf_coeff_[i], leaf_coeff_[i].size()) << ' ';
      }
      str_buf << ' ';
    }
    str_buf << '\n';
  }
  str_buf << "shrinkage=" << shrinkage_ << '\n';
  str_buf << '\n';

  return str_buf.str();
}

std::string Tree::ToJSON() const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << "\"num_leaves\":" << num_leaves_ << "," << '\n';
  str_buf << "\"num_cat\":" << num_cat_ << "," << '\n';
  str_buf << "\"shrinkage\":" << shrinkage_ << "," << '\n';
  if (num_leaves_ == 1) {
    if (is_linear_) {
      str_buf << "\"tree_structure\":{" << "\"leaf_value\":" << leaf_value_[0] << ", " << "\n";
      str_buf << LinearModelToJSON(0) << "}" << "\n";
    } else {
      str_buf << "\"tree_structure\":{" << "\"leaf_value\":" << leaf_value_[0] << "}" << '\n';
    }
  } else {
    str_buf << "\"tree_structure\":" << NodeToJSON(0) << '\n';
  }
  return str_buf.str();
}

std::string Tree::LinearModelToJSON(int index) const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << "\"leaf_const\":" << leaf_const_[index] << "," << "\n";
  int num_features = static_cast<int>(leaf_features_[index].size());
  if (num_features > 0) {
    str_buf << "\"leaf_features\":[";
    for (int i = 0; i < num_features - 1; ++i) {
      str_buf << leaf_features_[index][i] << ", ";
    }
    str_buf << leaf_features_[index][num_features - 1] << "]" << ", " << "\n";
    str_buf << "\"leaf_coeff\":[";
    for (int i = 0; i < num_features - 1; ++i) {
      str_buf << leaf_coeff_[index][i] << ", ";
    }
    str_buf << leaf_coeff_[index][num_features - 1] << "]" << "\n";
  } else {
    str_buf << "\"leaf_features\":[],\n";
    str_buf << "\"leaf_coeff\":[]\n";
  }
  return str_buf.str();
}

std::string Tree::NodeToJSON(int index) const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  if (index >= 0) {
    // non-leaf
    str_buf << "{" << '\n';
    str_buf << "\"split_index\":" << index << "," << '\n';
    str_buf << "\"split_feature\":" << split_feature_[index] << "," << '\n';
    str_buf << "\"split_gain\":" << Common::AvoidInf(split_gain_[index]) << "," << '\n';
    if (GetDecisionType(decision_type_[index], kCategoricalMask)) {
      int cat_idx = static_cast<int>(threshold_[index]);
      std::vector<int> cats;
      for (int i = cat_boundaries_[cat_idx]; i < cat_boundaries_[cat_idx + 1]; ++i) {
        for (int j = 0; j < 32; ++j) {
          int cat = (i - cat_boundaries_[cat_idx]) * 32 + j;
          if (Common::FindInBitset(cat_threshold_.data() + cat_boundaries_[cat_idx],
                                   cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx], cat)) {
            cats.push_back(cat);
          }
        }
      }
      str_buf << "\"threshold\":\"" << CommonC::Join(cats, "||") << "\"," << '\n';
      str_buf << "\"decision_type\":\"==\"," << '\n';
    } else {
      str_buf << "\"threshold\":" << Common::AvoidInf(threshold_[index]) << "," << '\n';
      str_buf << "\"decision_type\":\"<=\"," << '\n';
    }
    if (GetDecisionType(decision_type_[index], kDefaultLeftMask)) {
      str_buf << "\"default_left\":true," << '\n';
    } else {
      str_buf << "\"default_left\":false," << '\n';
    }
    uint8_t missing_type = GetMissingType(decision_type_[index]);
    if (missing_type == MissingType::None) {
      str_buf << "\"missing_type\":\"None\"," << '\n';
    } else if (missing_type == MissingType::Zero) {
      str_buf << "\"missing_type\":\"Zero\"," << '\n';
    } else {
      str_buf << "\"missing_type\":\"NaN\"," << '\n';
    }
    str_buf << "\"internal_value\":" << internal_value_[index] << "," << '\n';
    str_buf << "\"internal_weight\":" << internal_weight_[index] << "," << '\n';
    str_buf << "\"internal_count\":" << internal_count_[index] << "," << '\n';
    str_buf << "\"left_child\":" << NodeToJSON(left_child_[index]) << "," << '\n';
    str_buf << "\"right_child\":" << NodeToJSON(right_child_[index]) << '\n';
    str_buf << "}";
  } else {
    // leaf
    index = ~index;
    str_buf << "{" << '\n';
    str_buf << "\"leaf_index\":" << index << "," << '\n';
    str_buf << "\"leaf_value\":" << leaf_value_[index] << "," << '\n';
    str_buf << "\"leaf_weight\":" << leaf_weight_[index] << "," << '\n';
    if (is_linear_) {
      str_buf << "\"leaf_count\":" << leaf_count_[index] << "," << '\n';
      str_buf << LinearModelToJSON(index);
    } else {
      str_buf << "\"leaf_count\":" << leaf_count_[index] << '\n';
    }
    str_buf << "}";
  }
  return str_buf.str();
}

std::string Tree::NumericalDecisionIfElse(int node) const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  uint8_t missing_type = GetMissingType(decision_type_[node]);
  bool default_left = GetDecisionType(decision_type_[node], kDefaultLeftMask);
  if (missing_type == MissingType::None
      || (missing_type == MissingType::Zero && default_left && kZeroThreshold < threshold_[node])) {
    str_buf << "if (fval <= " << threshold_[node] << ") {";
  } else if (missing_type == MissingType::Zero) {
    if (default_left) {
      str_buf << "if (fval <= " << threshold_[node] << " || Tree::IsZero(fval)" << " || std::isnan(fval)) {";
    } else {
      str_buf << "if (fval <= " << threshold_[node] << " && !Tree::IsZero(fval)" << " && !std::isnan(fval)) {";
    }
  } else {
    if (default_left) {
      str_buf << "if (fval <= " << threshold_[node] << " || std::isnan(fval)) {";
    } else {
      str_buf << "if (fval <= " << threshold_[node] << " && !std::isnan(fval)) {";
    }
  }
  return str_buf.str();
}

std::string Tree::CategoricalDecisionIfElse(int node) const {
  uint8_t missing_type = GetMissingType(decision_type_[node]);
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  if (missing_type == MissingType::NaN) {
    str_buf << "if (std::isnan(fval)) { int_fval = -1; } else { int_fval = static_cast<int>(fval); }";
  } else {
    str_buf << "if (std::isnan(fval)) { int_fval = 0; } else { int_fval = static_cast<int>(fval); }";
  }
  int cat_idx = static_cast<int>(threshold_[node]);
  str_buf << "if (int_fval >= 0 && int_fval < 32 * (";
  str_buf << cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx];
  str_buf << ") && (((cat_threshold[" << cat_boundaries_[cat_idx];
  str_buf << " + int_fval / 32] >> (int_fval & 31)) & 1))) {";
  return str_buf.str();
}

std::string Tree::ToIfElse(int index, bool predict_leaf_index) const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  str_buf << "double PredictTree" << index;
  if (predict_leaf_index) {
    str_buf << "Leaf";
  }
  str_buf << "(const double* arr) { ";
  if (num_leaves_ <= 1) {
    str_buf << "return " << leaf_value_[0] << ";";
  } else {
    str_buf << "const std::vector<uint32_t> cat_threshold = {";
    for (size_t i = 0; i < cat_threshold_.size(); ++i) {
      if (i != 0) {
        str_buf << ",";
      }
      str_buf << cat_threshold_[i];
    }
    str_buf << "};";
    // use this for the missing value conversion
    str_buf << "double fval = 0.0f; ";
    if (num_cat_ > 0) {
      str_buf << "int int_fval = 0; ";
    }
    str_buf << NodeToIfElse(0, predict_leaf_index);
  }
  str_buf << " }" << '\n';

  // Predict func by Map to ifelse
  str_buf << "double PredictTree" << index;
  if (predict_leaf_index) {
    str_buf << "LeafByMap";
  } else {
    str_buf << "ByMap";
  }
  str_buf << "(const std::unordered_map<int, double>& arr) { ";
  if (num_leaves_ <= 1) {
    str_buf << "return " << leaf_value_[0] << ";";
  } else {
    str_buf << "const std::vector<uint32_t> cat_threshold = {";
    for (size_t i = 0; i < cat_threshold_.size(); ++i) {
      if (i != 0) {
        str_buf << ",";
      }
      str_buf << cat_threshold_[i];
    }
    str_buf << "};";
    // use this for the missing value conversion
    str_buf << "double fval = 0.0f; ";
    if (num_cat_ > 0) {
      str_buf << "int int_fval = 0; ";
    }
    str_buf << NodeToIfElseByMap(0, predict_leaf_index);
  }
  str_buf << " }" << '\n';

  return str_buf.str();
}

std::string Tree::NodeToIfElse(int index, bool predict_leaf_index) const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  if (index >= 0) {
    // non-leaf
    str_buf << "fval = arr[" << split_feature_[index] << "];";
    if (GetDecisionType(decision_type_[index], kCategoricalMask) == 0) {
      str_buf << NumericalDecisionIfElse(index);
    } else {
      str_buf << CategoricalDecisionIfElse(index);
    }
    // left subtree
    str_buf << NodeToIfElse(left_child_[index], predict_leaf_index);
    str_buf << " } else { ";
    // right subtree
    str_buf << NodeToIfElse(right_child_[index], predict_leaf_index);
    str_buf << " }";
  } else {
    // leaf
    str_buf << "return ";
    if (predict_leaf_index) {
      str_buf << ~index;
    } else {
      str_buf << leaf_value_[~index];
    }
    str_buf << ";";
  }

  return str_buf.str();
}

std::string Tree::NodeToIfElseByMap(int index, bool predict_leaf_index) const {
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  if (index >= 0) {
    // non-leaf
    str_buf << "fval = arr.count(" << split_feature_[index] << ") > 0 ? arr.at(" << split_feature_[index] << ") : 0.0f;";
    if (GetDecisionType(decision_type_[index], kCategoricalMask) == 0) {
      str_buf << NumericalDecisionIfElse(index);
    } else {
      str_buf << CategoricalDecisionIfElse(index);
    }
    // left subtree
    str_buf << NodeToIfElseByMap(left_child_[index], predict_leaf_index);
    str_buf << " } else { ";
    // right subtree
    str_buf << NodeToIfElseByMap(right_child_[index], predict_leaf_index);
    str_buf << " }";
  } else {
    // leaf
    str_buf << "return ";
    if (predict_leaf_index) {
      str_buf << ~index;
    } else {
      str_buf << leaf_value_[~index];
    }
    str_buf << ";";
  }

  return str_buf.str();
}

Tree::Tree(const char* str, size_t* used_len) {
  auto p = str;
  std::unordered_map<std::string, std::string> key_vals;
  const int max_num_line = 22;
  int read_line = 0;
  while (read_line < max_num_line) {
    if (*p == '\r' || *p == '\n') break;
    auto start = p;
    while (*p != '=') ++p;
    std::string key(start, p - start);
    ++p;
    start = p;
    while (*p != '\r' && *p != '\n') ++p;
    key_vals[key] = std::string(start, p - start);
    ++read_line;
    if (*p == '\r') ++p;
    if (*p == '\n') ++p;
  }
  *used_len = p - str;

  if (key_vals.count("num_leaves") <= 0) {
    Log::Fatal("Tree model should contain num_leaves field");
  }

  Common::Atoi(key_vals["num_leaves"].c_str(), &num_leaves_);

  if (key_vals.count("num_cat") <= 0) {
    Log::Fatal("Tree model should contain num_cat field");
  }

  Common::Atoi(key_vals["num_cat"].c_str(), &num_cat_);

  if (key_vals.count("leaf_value")) {
    leaf_value_ = CommonC::StringToArray<double>(key_vals["leaf_value"], num_leaves_);
  } else {
    Log::Fatal("Tree model string format error, should contain leaf_value field");
  }

  if (key_vals.count("shrinkage")) {
    CommonC::Atof(key_vals["shrinkage"].c_str(), &shrinkage_);
  } else {
    shrinkage_ = 1.0f;
  }

  if (key_vals.count("is_linear")) {
    int is_linear_int;
    Common::Atoi(key_vals["is_linear"].c_str(), &is_linear_int);
    is_linear_ = static_cast<bool>(is_linear_int);
  } else {
    is_linear_ = false;
  }

  if ((num_leaves_ <= 1) && !is_linear_) {
    return;
  }

  if (key_vals.count("left_child")) {
    left_child_ = CommonC::StringToArrayFast<int>(key_vals["left_child"], num_leaves_ - 1);
  } else {
    Log::Fatal("Tree model string format error, should contain left_child field");
  }

  if (key_vals.count("right_child")) {
    right_child_ = CommonC::StringToArrayFast<int>(key_vals["right_child"], num_leaves_ - 1);
  } else {
    Log::Fatal("Tree model string format error, should contain right_child field");
  }

  if (key_vals.count("split_feature")) {
    split_feature_ = CommonC::StringToArrayFast<int>(key_vals["split_feature"], num_leaves_ - 1);
  } else {
    Log::Fatal("Tree model string format error, should contain split_feature field");
  }

  if (key_vals.count("threshold")) {
    threshold_ = CommonC::StringToArray<double>(key_vals["threshold"], num_leaves_ - 1);
  } else {
    Log::Fatal("Tree model string format error, should contain threshold field");
  }

  if (key_vals.count("split_gain")) {
    split_gain_ = CommonC::StringToArrayFast<float>(key_vals["split_gain"], num_leaves_ - 1);
  } else {
    split_gain_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("internal_count")) {
    internal_count_ = CommonC::StringToArrayFast<int>(key_vals["internal_count"], num_leaves_ - 1);
  } else {
    internal_count_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("internal_value")) {
    internal_value_ = CommonC::StringToArrayFast<double>(key_vals["internal_value"], num_leaves_ - 1);
  } else {
    internal_value_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("internal_weight")) {
    internal_weight_ = CommonC::StringToArrayFast<double>(key_vals["internal_weight"], num_leaves_ - 1);
  } else {
    internal_weight_.resize(num_leaves_ - 1);
  }

  if (key_vals.count("leaf_weight")) {
    leaf_weight_ = CommonC::StringToArray<double>(key_vals["leaf_weight"], num_leaves_);
  } else {
    leaf_weight_.resize(num_leaves_);
  }

  if (key_vals.count("leaf_count")) {
    leaf_count_ = CommonC::StringToArrayFast<int>(key_vals["leaf_count"], num_leaves_);
  } else {
    leaf_count_.resize(num_leaves_);
  }

  if (key_vals.count("decision_type")) {
    decision_type_ = CommonC::StringToArrayFast<uint8_t>(key_vals["decision_type"], num_leaves_ - 1);
  } else {
    decision_type_ = std::vector<uint8_t>(num_leaves_ - 1, 0);
  }

  if (is_linear_) {
    if (key_vals.count("leaf_const")) {
      leaf_const_ = Common::StringToArray<double>(key_vals["leaf_const"], num_leaves_);
    } else {
      leaf_const_.resize(num_leaves_);
    }
    std::vector<int> num_feat;
    if (key_vals.count("num_features")) {
      num_feat = Common::StringToArrayFast<int>(key_vals["num_features"], num_leaves_);
    }
    leaf_coeff_.resize(num_leaves_);
    leaf_features_.resize(num_leaves_);
    leaf_features_inner_.resize(num_leaves_);
    if (num_feat.size() > 0) {
      int total_num_feat = 0;
      for (size_t i = 0; i < num_feat.size(); ++i) {
        total_num_feat += num_feat[i];
      }
      std::vector<int> all_leaf_features;
      if (key_vals.count("leaf_features")) {
        all_leaf_features = Common::StringToArrayFast<int>(key_vals["leaf_features"], total_num_feat);
      }
      std::vector<double> all_leaf_coeff;
      if (key_vals.count("leaf_coeff")) {
        all_leaf_coeff = Common::StringToArray<double>(key_vals["leaf_coeff"], total_num_feat);
      }
      int sum_num_feat = 0;
      for (int i = 0; i < num_leaves_; ++i) {
        if (num_feat[i] > 0) {
          if (key_vals.count("leaf_features"))  {
            leaf_features_[i].assign(all_leaf_features.begin() + sum_num_feat, all_leaf_features.begin() + sum_num_feat + num_feat[i]);
          }
          if (key_vals.count("leaf_coeff")) {
            leaf_coeff_[i].assign(all_leaf_coeff.begin() + sum_num_feat, all_leaf_coeff.begin() + sum_num_feat + num_feat[i]);
          }
        }
        sum_num_feat += num_feat[i];
      }
    }
  }

  if (num_cat_ > 0) {
    if (key_vals.count("cat_boundaries")) {
      cat_boundaries_ = CommonC::StringToArrayFast<int>(key_vals["cat_boundaries"], num_cat_ + 1);
    } else {
      Log::Fatal("Tree model should contain cat_boundaries field.");
    }

    if (key_vals.count("cat_threshold")) {
      cat_threshold_ = CommonC::StringToArrayFast<uint32_t>(key_vals["cat_threshold"], cat_boundaries_.back());
    } else {
      Log::Fatal("Tree model should contain cat_threshold field");
    }
  }
  max_depth_ = -1;
}

void Tree::ExtendPath(PathElement *unique_path, int unique_depth,
                      double zero_fraction, double one_fraction, int feature_index) {
  unique_path[unique_depth].feature_index = feature_index;
  unique_path[unique_depth].zero_fraction = zero_fraction;
  unique_path[unique_depth].one_fraction = one_fraction;
  unique_path[unique_depth].pweight = (unique_depth == 0 ? 1 : 0);
  for (int i = unique_depth - 1; i >= 0; i--) {
    unique_path[i + 1].pweight += one_fraction*unique_path[i].pweight*(i + 1)
      / static_cast<double>(unique_depth + 1);
    unique_path[i].pweight = zero_fraction*unique_path[i].pweight*(unique_depth - i)
      / static_cast<double>(unique_depth + 1);
  }
}

void Tree::UnwindPath(PathElement *unique_path, int unique_depth, int path_index) {
  const double one_fraction = unique_path[path_index].one_fraction;
  const double zero_fraction = unique_path[path_index].zero_fraction;
  double next_one_portion = unique_path[unique_depth].pweight;

  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const double tmp = unique_path[i].pweight;
      unique_path[i].pweight = next_one_portion*(unique_depth + 1)
        / static_cast<double>((i + 1)*one_fraction);
      next_one_portion = tmp - unique_path[i].pweight*zero_fraction*(unique_depth - i)
        / static_cast<double>(unique_depth + 1);
    } else {
      unique_path[i].pweight = (unique_path[i].pweight*(unique_depth + 1))
        / static_cast<double>(zero_fraction*(unique_depth - i));
    }
  }

  for (int i = path_index; i < unique_depth; ++i) {
    unique_path[i].feature_index = unique_path[i + 1].feature_index;
    unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction;
    unique_path[i].one_fraction = unique_path[i + 1].one_fraction;
  }
}

double Tree::UnwoundPathSum(const PathElement *unique_path, int unique_depth, int path_index) {
  const double one_fraction = unique_path[path_index].one_fraction;
  const double zero_fraction = unique_path[path_index].zero_fraction;
  double next_one_portion = unique_path[unique_depth].pweight;
  double total = 0;
  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const double tmp = next_one_portion*(unique_depth + 1)
        / static_cast<double>((i + 1)*one_fraction);
      total += tmp;
      next_one_portion = unique_path[i].pweight - tmp*zero_fraction*((unique_depth - i)
                                                                     / static_cast<double>(unique_depth + 1));
    } else {
      total += (unique_path[i].pweight / zero_fraction) / ((unique_depth - i)
                                                           / static_cast<double>(unique_depth + 1));
    }
  }
  return total;
}

// recursive computation of SHAP values for a decision tree
void Tree::TreeSHAP(const double *feature_values, double *phi,
                    int node, int unique_depth,
                    PathElement *parent_unique_path, double parent_zero_fraction,
                    double parent_one_fraction, int parent_feature_index) const {
  // extend the unique path
  PathElement* unique_path = parent_unique_path + unique_depth;
  if (unique_depth > 0) {
    std::copy(parent_unique_path, parent_unique_path + unique_depth, unique_path);
  }
  ExtendPath(unique_path, unique_depth, parent_zero_fraction,
             parent_one_fraction, parent_feature_index);

  // leaf node
  if (node < 0) {
    for (int i = 1; i <= unique_depth; ++i) {
      const double w = UnwoundPathSum(unique_path, unique_depth, i);
      const PathElement &el = unique_path[i];
      phi[el.feature_index] += w*(el.one_fraction - el.zero_fraction)*leaf_value_[~node];
    }

    // internal node
  } else {
    const int hot_index = Decision(feature_values[split_feature_[node]], node);
    const int cold_index = (hot_index == left_child_[node] ? right_child_[node] : left_child_[node]);
    const double w = data_count(node);
    const double hot_zero_fraction = data_count(hot_index) / w;
    const double cold_zero_fraction = data_count(cold_index) / w;
    double incoming_zero_fraction = 1;
    double incoming_one_fraction = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    int path_index = 0;
    for (; path_index <= unique_depth; ++path_index) {
      if (unique_path[path_index].feature_index == split_feature_[node]) break;
    }
    if (path_index != unique_depth + 1) {
      incoming_zero_fraction = unique_path[path_index].zero_fraction;
      incoming_one_fraction = unique_path[path_index].one_fraction;
      UnwindPath(unique_path, unique_depth, path_index);
      unique_depth -= 1;
    }

    TreeSHAP(feature_values, phi, hot_index, unique_depth + 1, unique_path,
             hot_zero_fraction*incoming_zero_fraction, incoming_one_fraction, split_feature_[node]);

    TreeSHAP(feature_values, phi, cold_index, unique_depth + 1, unique_path,
             cold_zero_fraction*incoming_zero_fraction, 0, split_feature_[node]);
  }
}

// recursive sparse computation of SHAP values for a decision tree
void Tree::TreeSHAPByMap(const std::unordered_map<int, double>& feature_values, std::unordered_map<int, double>* phi,
                         int node, int unique_depth,
                         PathElement *parent_unique_path, double parent_zero_fraction,
                         double parent_one_fraction, int parent_feature_index) const {
  // extend the unique path
  PathElement* unique_path = parent_unique_path + unique_depth;
  if (unique_depth > 0) {
    std::copy(parent_unique_path, parent_unique_path + unique_depth, unique_path);
  }
  ExtendPath(unique_path, unique_depth, parent_zero_fraction,
             parent_one_fraction, parent_feature_index);

  // leaf node
  if (node < 0) {
    for (int i = 1; i <= unique_depth; ++i) {
      const double w = UnwoundPathSum(unique_path, unique_depth, i);
      const PathElement &el = unique_path[i];
      (*phi)[el.feature_index] += w*(el.one_fraction - el.zero_fraction)*leaf_value_[~node];
    }

  // internal node
  } else {
    const int hot_index = Decision(feature_values.count(split_feature_[node]) > 0 ? feature_values.at(split_feature_[node]) : 0.0f, node);
    const int cold_index = (hot_index == left_child_[node] ? right_child_[node] : left_child_[node]);
    const double w = data_count(node);
    const double hot_zero_fraction = data_count(hot_index) / w;
    const double cold_zero_fraction = data_count(cold_index) / w;
    double incoming_zero_fraction = 1;
    double incoming_one_fraction = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    int path_index = 0;
    for (; path_index <= unique_depth; ++path_index) {
      if (unique_path[path_index].feature_index == split_feature_[node]) break;
    }
    if (path_index != unique_depth + 1) {
      incoming_zero_fraction = unique_path[path_index].zero_fraction;
      incoming_one_fraction = unique_path[path_index].one_fraction;
      UnwindPath(unique_path, unique_depth, path_index);
      unique_depth -= 1;
    }

    TreeSHAPByMap(feature_values, phi, hot_index, unique_depth + 1, unique_path,
                  hot_zero_fraction*incoming_zero_fraction, incoming_one_fraction, split_feature_[node]);

    TreeSHAPByMap(feature_values, phi, cold_index, unique_depth + 1, unique_path,
                  cold_zero_fraction*incoming_zero_fraction, 0, split_feature_[node]);
  }
}

double Tree::ExpectedValue() const {
  if (num_leaves_ == 1) return LeafOutput(0);
  const double total_count = internal_count_[0];
  double exp_value = 0.0;
  for (int i = 0; i < num_leaves(); ++i) {
    exp_value += (leaf_count_[i] / total_count)*LeafOutput(i);
  }
  return exp_value;
}

void Tree::RecomputeMaxDepth() {
  if (num_leaves_ == 1) {
    max_depth_ = 0;
  } else {
    if (leaf_depth_.size() == 0) {
      RecomputeLeafDepths(0, 0);
    }
    max_depth_ = leaf_depth_[0];
    for (int i = 1; i < num_leaves(); ++i) {
      if (max_depth_ < leaf_depth_[i]) max_depth_ = leaf_depth_[i];
    }
  }
}

}  // namespace LightGBM
