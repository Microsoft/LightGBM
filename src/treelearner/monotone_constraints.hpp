/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_
#define LIGHTGBM_SRC_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_

#include <LightGBM/tree.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "split_info.hpp"

namespace LightGBM {

class LeafConstraintsBase;

struct BasicConstraint {
  double min = -std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::max();

  BasicConstraint(double min, double max) : min(min), max(max) {}

  BasicConstraint() = default;
};

struct FeatureConstraint {
  virtual void InitCumulativeConstraints(bool) const {}
  virtual void Update(int) const {}
  virtual BasicConstraint LeftToBasicConstraint() const = 0;
  virtual BasicConstraint RightToBasicConstraint() const = 0;
  virtual bool ConstraintDifferentDependingOnThreshold() const = 0;
  virtual ~FeatureConstraint() {}
};

struct ConstraintEntry {
  virtual ~ConstraintEntry() {}
  virtual void Reset() = 0;
  virtual void UpdateMin(double new_min) = 0;
  virtual void UpdateMax(double new_max) = 0;
  virtual bool UpdateMinAndReturnBoolIfChanged(double new_min) = 0;
  virtual bool UpdateMaxAndReturnBoolIfChanged(double new_max) = 0;
  virtual ConstraintEntry *clone() const = 0;

  virtual void RecomputeConstraintsIfNeeded(LeafConstraintsBase *, int, int,
                                            uint32_t) {}

  virtual FeatureConstraint *GetFeatureConstraint(int feature_index) = 0;
};

// used by both BasicLeafConstraints and IntermediateLeafConstraints
struct BasicConstraintEntry : ConstraintEntry,
                              FeatureConstraint,
                              BasicConstraint {
  bool ConstraintDifferentDependingOnThreshold() const final { return false; }

  BasicConstraintEntry *clone() const final {
    return new BasicConstraintEntry(*this);
  };

  void Reset() final {
    min = -std::numeric_limits<double>::max();
    max = std::numeric_limits<double>::max();
  }

  void UpdateMin(double new_min) final { min = std::max(new_min, min); }

  void UpdateMax(double new_max) final { max = std::min(new_max, max); }

  bool UpdateMinAndReturnBoolIfChanged(double new_min) final {
    if (new_min > min) {
      min = new_min;
      return true;
    }
    return false;
  }

  bool UpdateMaxAndReturnBoolIfChanged(double new_max) final {
    if (new_max < max) {
      max = new_max;
      return true;
    }
    return false;
  }

  BasicConstraint LeftToBasicConstraint() const final { return *this; }

  BasicConstraint RightToBasicConstraint() const final { return *this; }

  FeatureConstraint *GetFeatureConstraint(int) final { return this; }
};

struct FeatureMinOrMaxConstraints {
  std::vector<double> constraints;
  // the constraint number i is valid on the slice
  // [thresholds[i]:threshold[i+1])
  // if threshold[i+1] does not exist, then it is valid for thresholds following
  // threshold[i]
  std::vector<uint32_t> thresholds;

  FeatureMinOrMaxConstraints() {
    constraints.reserve(32);
    thresholds.reserve(32);
  }

  size_t Size() const { return thresholds.size(); }

  explicit FeatureMinOrMaxConstraints(double extremum) {
    constraints.reserve(32);
    thresholds.reserve(32);

    constraints.push_back(extremum);
    thresholds.push_back(0);
  }

  void Reset(double extremum) {
    constraints.resize(1);
    constraints[0] = extremum;
    thresholds.resize(1);
    thresholds[0] = 0;
  }

  void UpdateMin(double min) {
    for (size_t j = 0; j < constraints.size(); ++j) {
      if (min > constraints[j]) {
        constraints[j] = min;
      }
    }
  }

  void UpdateMax(double max) {
    for (size_t j = 0; j < constraints.size(); ++j) {
      if (max < constraints[j]) {
        constraints[j] = max;
      }
    }
  }
};

struct CumulativeFeatureConstraint {
  std::vector<uint32_t> thresholds_min_constraints;
  std::vector<uint32_t> thresholds_max_constraints;
  std::vector<double> cumulative_min_constraints_left_to_right;
  std::vector<double> cumulative_min_constraints_right_to_left;
  std::vector<double> cumulative_max_constraints_left_to_right;
  std::vector<double> cumulative_max_constraints_right_to_left;
  size_t index_min_constraints_left_to_right;
  size_t index_min_constraints_right_to_left;
  size_t index_max_constraints_left_to_right;
  size_t index_max_constraints_right_to_left;

  static void CumulativeExtremum(
      const double &(*extremum_function)(const double &, const double &),
      bool is_direction_from_left_to_right,
      std::vector<double>* cumulative_extremum) {
    if (cumulative_extremum->size() == 1) {
      return;
    }

#ifdef DEBUG
    CHECK_NE(cumulative_extremum->size(), 0);
#endif

    size_t n_exts = cumulative_extremum->size();
    int step = is_direction_from_left_to_right ? 1 : -1;
    size_t start = is_direction_from_left_to_right ? 0 : n_exts - 1;
    size_t end = is_direction_from_left_to_right ? n_exts - 1 : 0;

    for (auto i = start; i != end; i = i + step) {
      (*cumulative_extremum)[i + step] = extremum_function(
          (*cumulative_extremum)[i + step], (*cumulative_extremum)[i]);
    }
  }

  CumulativeFeatureConstraint() = default;

  CumulativeFeatureConstraint(FeatureMinOrMaxConstraints min_constraints,
                              FeatureMinOrMaxConstraints max_constraints,
                              bool REVERSE) {
    thresholds_min_constraints = min_constraints.thresholds;
    thresholds_max_constraints = max_constraints.thresholds;
    cumulative_min_constraints_left_to_right = min_constraints.constraints;
    cumulative_min_constraints_right_to_left = min_constraints.constraints;
    cumulative_max_constraints_left_to_right = max_constraints.constraints;
    cumulative_max_constraints_right_to_left = max_constraints.constraints;

    const double &(*min)(const double &, const double &) = std::min<double>;
    const double &(*max)(const double &, const double &) = std::max<double>;
    CumulativeExtremum(max, true, &cumulative_min_constraints_left_to_right);
    CumulativeExtremum(max, false, &cumulative_min_constraints_right_to_left);
    CumulativeExtremum(min, true, &cumulative_max_constraints_left_to_right);
    CumulativeExtremum(min, false, &cumulative_max_constraints_right_to_left);

    if (REVERSE) {
      index_min_constraints_left_to_right =
          thresholds_min_constraints.size() - 1;
      index_min_constraints_right_to_left =
          thresholds_min_constraints.size() - 1;
      index_max_constraints_left_to_right =
          thresholds_max_constraints.size() - 1;
      index_max_constraints_right_to_left =
          thresholds_max_constraints.size() - 1;
    } else {
      index_min_constraints_left_to_right = 0;
      index_min_constraints_right_to_left = 0;
      index_max_constraints_left_to_right = 0;
      index_max_constraints_right_to_left = 0;
    }
  }

  void Update(int threshold) {
    while (
        static_cast<int>(
            thresholds_min_constraints[index_min_constraints_left_to_right]) >
        threshold - 1) {
      index_min_constraints_left_to_right -= 1;
    }
    while (
        static_cast<int>(
            thresholds_min_constraints[index_min_constraints_right_to_left]) >
        threshold) {
      index_min_constraints_right_to_left -= 1;
    }
    while (
        static_cast<int>(
            thresholds_max_constraints[index_max_constraints_left_to_right]) >
        threshold - 1) {
      index_max_constraints_left_to_right -= 1;
    }
    while (
        static_cast<int>(
            thresholds_max_constraints[index_max_constraints_right_to_left]) >
        threshold) {
      index_max_constraints_right_to_left -= 1;
    }
  }

  double GetRightMin() const {
    return cumulative_min_constraints_right_to_left
        [index_min_constraints_right_to_left];
  }
  double GetRightMax() const {
    return cumulative_max_constraints_right_to_left
        [index_max_constraints_right_to_left];
  }
  double GetLeftMin() const {
    return cumulative_min_constraints_left_to_right
        [index_min_constraints_left_to_right];
  }
  double GetLeftMax() const {
    return cumulative_max_constraints_left_to_right
        [index_max_constraints_left_to_right];
  }
};

struct AdvancedFeatureConstraints : FeatureConstraint {
  FeatureMinOrMaxConstraints min_constraints;
  FeatureMinOrMaxConstraints max_constraints;
  mutable CumulativeFeatureConstraint cumulative_feature_constraint;
  bool min_constraints_to_be_recomputed = false;
  bool max_constraints_to_be_recomputed = false;

  void InitCumulativeConstraints(bool REVERSE) const final {
    cumulative_feature_constraint =
        CumulativeFeatureConstraint(min_constraints, max_constraints, REVERSE);
  }

  void Update(int threshold) const final {
    cumulative_feature_constraint.Update(threshold);
  }

  FeatureMinOrMaxConstraints &GetMinConstraints() { return min_constraints; }

  FeatureMinOrMaxConstraints &GetMaxConstraints() { return max_constraints; }

  bool ConstraintDifferentDependingOnThreshold() const final {
    return min_constraints.Size() > 1 || max_constraints.Size() > 1;
  }

  BasicConstraint RightToBasicConstraint() const final {
    return BasicConstraint(cumulative_feature_constraint.GetRightMin(),
                           cumulative_feature_constraint.GetRightMax());
  }

  BasicConstraint LeftToBasicConstraint() const final {
    return BasicConstraint(cumulative_feature_constraint.GetLeftMin(),
                           cumulative_feature_constraint.GetLeftMax());
  }

  void Reset() {
    min_constraints.Reset(-std::numeric_limits<double>::max());
    max_constraints.Reset(std::numeric_limits<double>::max());
  }

  void UpdateMax(double new_max, bool trigger_a_recompute) {
    if (trigger_a_recompute) {
      max_constraints_to_be_recomputed = true;
    }
    max_constraints.UpdateMax(new_max);
  }

  bool FeatureMaxConstraintsToBeUpdated() const {
    return max_constraints_to_be_recomputed;
  }

  bool FeatureMinConstraintsToBeUpdated() const {
    return min_constraints_to_be_recomputed;
  }

  void ResetUpdates() {
    min_constraints_to_be_recomputed = false;
    max_constraints_to_be_recomputed = false;
  }

  void UpdateMin(double new_min, bool trigger_a_recompute) {
    if (trigger_a_recompute) {
      min_constraints_to_be_recomputed = true;
    }
    min_constraints.UpdateMin(new_min);
  }
};

class LeafConstraintsBase {
 public:
  virtual ~LeafConstraintsBase() {}
  virtual const ConstraintEntry* Get(int leaf_idx) = 0;
  virtual FeatureConstraint* GetFeatureConstraint(int leaf_idx, int feature_index) = 0;
  virtual void Reset() = 0;
  virtual void BeforeSplit(int leaf, int new_leaf,
                           int8_t monotone_type) = 0;
  virtual std::vector<int> Update(
      bool is_numerical_split,
      int leaf, int new_leaf, int8_t monotone_type, double right_output,
      double left_output, int split_feature, const SplitInfo& split_info,
      const std::vector<SplitInfo>& best_split_per_leaf) = 0;

  virtual void GoUpToFindConstrainingLeaves(
      int, int,
      std::vector<int>*,
      std::vector<uint32_t>*,
      std::vector<bool>*,
      FeatureMinOrMaxConstraints*, bool ,
      uint32_t, uint32_t, uint32_t) {}

  virtual void RecomputeConstraintsIfNeeded(
      LeafConstraintsBase *constraints_,
      int feature_for_constraint, int leaf_idx, uint32_t it_end) = 0;

  inline static LeafConstraintsBase* Create(const Config* config, int num_leaves, int num_features);

  double ComputeMonotoneSplitGainPenalty(int leaf_index, double penalization) {
    int depth = tree_->leaf_depth(leaf_index);
    if (penalization >= depth + 1.) {
      return kEpsilon;
    }
    if (penalization <= 1.) {
      return 1. - penalization / pow(2., depth) + kEpsilon;
    }
    return 1. - pow(2, penalization - 1. - depth) + kEpsilon;
  }

  void ShareTreePointer(const Tree* tree) {
    tree_ = tree;
  }

 protected:
  const Tree* tree_;
};

// used by AdvancedLeafConstraints
struct AdvancedConstraintEntry : ConstraintEntry {
  std::vector<AdvancedFeatureConstraints> constraints;

  AdvancedConstraintEntry *clone() const final {
    return new AdvancedConstraintEntry(*this);
  };

  void RecomputeConstraintsIfNeeded(LeafConstraintsBase *constraints_,
                                    int feature_for_constraint, int leaf_idx,
                                    uint32_t it_end) final {
    if (constraints[feature_for_constraint]
            .FeatureMinConstraintsToBeUpdated() ||
        constraints[feature_for_constraint]
            .FeatureMaxConstraintsToBeUpdated()) {
      FeatureMinOrMaxConstraints &constraints_to_be_updated =
          constraints[feature_for_constraint].FeatureMinConstraintsToBeUpdated()
              ? constraints[feature_for_constraint].GetMinConstraints()
              : constraints[feature_for_constraint].GetMaxConstraints();

      constraints_to_be_updated.Reset(
          constraints[feature_for_constraint].FeatureMinConstraintsToBeUpdated()
              ? -std::numeric_limits<double>::max()
              : std::numeric_limits<double>::max());

      std::vector<int> features_of_splits_going_up_from_original_leaf =
          std::vector<int>();
      std::vector<uint32_t> thresholds_of_splits_going_up_from_original_leaf =
          std::vector<uint32_t>();
      std::vector<bool> was_original_leaf_right_child_of_split =
          std::vector<bool>();
      constraints_->GoUpToFindConstrainingLeaves(
          feature_for_constraint, leaf_idx,
          &features_of_splits_going_up_from_original_leaf,
          &thresholds_of_splits_going_up_from_original_leaf,
          &was_original_leaf_right_child_of_split, &constraints_to_be_updated,
          constraints[feature_for_constraint]
              .FeatureMinConstraintsToBeUpdated(),
          0, it_end, it_end);
      constraints[feature_for_constraint].ResetUpdates();
    }
  }

  // for each feature, an array of constraints needs to be stored
  explicit AdvancedConstraintEntry(int num_features) {
    constraints.resize(num_features);
  }

  void Reset() final {
    for (size_t i = 0; i < constraints.size(); ++i) {
      constraints[i].Reset();
    }
  }

  void UpdateMin(double new_min) final {
    for (size_t i = 0; i < constraints.size(); ++i) {
      constraints[i].UpdateMin(new_min, false);
    }
  }

  void UpdateMax(double new_max) final {
    for (size_t i = 0; i < constraints.size(); ++i) {
      constraints[i].UpdateMax(new_max, false);
    }
  }

  bool UpdateMinAndReturnBoolIfChanged(double new_min) final {
    for (size_t i = 0; i < constraints.size(); ++i) {
      constraints[i].UpdateMin(new_min, true);
    }
    // even if nothing changed, this could have been unconstrained so it needs
    // to be recomputed from the beginning
    return true;
  }

  bool UpdateMaxAndReturnBoolIfChanged(double new_max) final {
    for (size_t i = 0; i < constraints.size(); ++i) {
      constraints[i].UpdateMax(new_max, true);
    }
    // even if nothing changed, this could have been unconstrained so it needs
    // to be recomputed from the beginning
    return true;
  }

  FeatureConstraint *GetFeatureConstraint(int feature_index) final {
    return &constraints[feature_index];
  }
};

class BasicLeafConstraints : public LeafConstraintsBase {
 public:
  explicit BasicLeafConstraints(int num_leaves) : num_leaves_(num_leaves) {
    for (int i = 0; i < num_leaves; ++i) {
      entries_.emplace_back(new BasicConstraintEntry());
    }
  }

  void Reset() override {
    for (auto& entry : entries_) {
      entry->Reset();
    }
  }

  void RecomputeConstraintsIfNeeded(
      LeafConstraintsBase* constraints_,
      int feature_for_constraint, int leaf_idx, uint32_t it_end) override {
    entries_[~leaf_idx]->RecomputeConstraintsIfNeeded(constraints_, feature_for_constraint, leaf_idx, it_end);
  }

  void BeforeSplit(int, int, int8_t) override {}

  std::vector<int> Update(bool is_numerical_split, int leaf, int new_leaf,
                          int8_t monotone_type, double right_output,
                          double left_output, int, const SplitInfo& ,
                          const std::vector<SplitInfo>&) override {
    entries_[new_leaf].reset(entries_[leaf]->clone());
    if (is_numerical_split) {
      double mid = (left_output + right_output) / 2.0f;
      if (monotone_type < 0) {
        entries_[leaf]->UpdateMin(mid);
        entries_[new_leaf]->UpdateMax(mid);
      } else if (monotone_type > 0) {
        entries_[leaf]->UpdateMax(mid);
        entries_[new_leaf]->UpdateMin(mid);
      }
    }
    return std::vector<int>();
  }

  const ConstraintEntry* Get(int leaf_idx) override { return entries_[leaf_idx].get(); }

  FeatureConstraint* GetFeatureConstraint(int leaf_idx, int feature_index) final {
    return entries_[leaf_idx]->GetFeatureConstraint(feature_index);
  }

 protected:
  int num_leaves_;
  std::vector<std::unique_ptr<ConstraintEntry>> entries_;
};

class IntermediateLeafConstraints : public BasicLeafConstraints {
 public:
  explicit IntermediateLeafConstraints(const Config* config, int num_leaves)
      : BasicLeafConstraints(num_leaves), config_(config) {
    leaf_is_in_monotone_subtree_.resize(num_leaves_, false);
    node_parent_.resize(num_leaves_ - 1, -1);
    leaves_to_update_.reserve(num_leaves_);
  }

  void Reset() override {
    BasicLeafConstraints::Reset();
    std::fill_n(leaf_is_in_monotone_subtree_.begin(), num_leaves_, false);
    std::fill_n(node_parent_.begin(), num_leaves_ - 1, -1);
    leaves_to_update_.clear();
  }

  void BeforeSplit(int leaf, int new_leaf,
                   int8_t monotone_type) override {
    if (monotone_type != 0 || leaf_is_in_monotone_subtree_[leaf]) {
      leaf_is_in_monotone_subtree_[leaf] = true;
      leaf_is_in_monotone_subtree_[new_leaf] = true;
    }
#ifdef DEBUG
    CHECK_GE(new_leaf - 1, 0);
    CHECK_LT(static_cast<size_t>(new_leaf - 1), node_parent_.size());
#endif
    node_parent_[new_leaf - 1] = tree_->leaf_parent(leaf);
  }

  void UpdateConstraintsWithOutputs(bool is_numerical_split, int leaf,
                                    int new_leaf, int8_t monotone_type,
                                    double right_output, double left_output) {
    entries_[new_leaf].reset(entries_[leaf]->clone());
    if (is_numerical_split) {
      if (monotone_type < 0) {
        entries_[leaf]->UpdateMin(right_output);
        entries_[new_leaf]->UpdateMax(left_output);
      } else if (monotone_type > 0) {
        entries_[leaf]->UpdateMax(right_output);
        entries_[new_leaf]->UpdateMin(left_output);
      }
    }
  }

  std::vector<int> Update(bool is_numerical_split, int leaf,
                          int new_leaf, int8_t monotone_type,
                          double right_output, double left_output,
                          int split_feature, const SplitInfo& split_info,
                          const std::vector<SplitInfo>& best_split_per_leaf) final {
    leaves_to_update_.clear();
    if (leaf_is_in_monotone_subtree_[leaf]) {
      UpdateConstraintsWithOutputs(is_numerical_split, leaf, new_leaf,
                                   monotone_type, right_output, left_output);

      // Initialize variables to store information while going up the tree
      int depth = tree_->leaf_depth(new_leaf) - 1;

      std::vector<int> features_of_splits_going_up_from_original_leaf;
      std::vector<uint32_t> thresholds_of_splits_going_up_from_original_leaf;
      std::vector<bool> was_original_leaf_right_child_of_split;

      features_of_splits_going_up_from_original_leaf.reserve(depth);
      thresholds_of_splits_going_up_from_original_leaf.reserve(depth);
      was_original_leaf_right_child_of_split.reserve(depth);

      GoUpToFindLeavesToUpdate(tree_->leaf_parent(new_leaf),
                               &features_of_splits_going_up_from_original_leaf,
                               &thresholds_of_splits_going_up_from_original_leaf,
                               &was_original_leaf_right_child_of_split,
                               split_feature, split_info, split_info.threshold,
                               best_split_per_leaf);
    }
    return leaves_to_update_;
  }

  bool OppositeChildShouldBeUpdated(
      bool is_split_numerical,
      const std::vector<int>& features_of_splits_going_up_from_original_leaf,
      int inner_feature,
      const std::vector<bool>& was_original_leaf_right_child_of_split,
      bool is_in_right_child) {
    // if the split is categorical, it is not handled by this optimisation,
    // so the code will have to go down in the other child subtree to see if
    // there are leaves to update
    // even though it may sometimes be unnecessary
    if (is_split_numerical) {
      // only branches containing leaves that are contiguous to the original
      // leaf need to be updated
      // therefore, for the same feature, there is no use going down from the
      // second time going up on the right (or on the left)
      for (size_t split_idx = 0;
           split_idx < features_of_splits_going_up_from_original_leaf.size();
           ++split_idx) {
        if (features_of_splits_going_up_from_original_leaf[split_idx] ==
                inner_feature &&
            (was_original_leaf_right_child_of_split[split_idx] ==
             is_in_right_child)) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  // Recursive function that goes up the tree, and then down to find leaves that
  // have constraints to be updated
  void GoUpToFindLeavesToUpdate(
      int node_idx,
      std::vector<int>* features_of_splits_going_up_from_original_leaf,
      std::vector<uint32_t>* thresholds_of_splits_going_up_from_original_leaf,
      std::vector<bool>* was_original_leaf_right_child_of_split,
      int split_feature, const SplitInfo& split_info, uint32_t split_threshold,
      const std::vector<SplitInfo>& best_split_per_leaf) {
#ifdef DEBUG
    CHECK_GE(node_idx, 0);
    CHECK_LT(static_cast<size_t>(node_idx), node_parent_.size());
#endif
    int parent_idx = node_parent_[node_idx];
    // if not at the root
    if (parent_idx != -1) {
      int inner_feature = tree_->split_feature_inner(parent_idx);
      int feature = tree_->split_feature(parent_idx);
      int8_t monotone_type = config_->monotone_constraints[feature];
      bool is_in_right_child = tree_->right_child(parent_idx) == node_idx;
      bool is_split_numerical = tree_->IsNumericalSplit(parent_idx);

      // this is just an optimisation not to waste time going down in subtrees
      // where there won't be any leaf to update
      bool opposite_child_should_be_updated = OppositeChildShouldBeUpdated(
          is_split_numerical, *features_of_splits_going_up_from_original_leaf,
          inner_feature, *was_original_leaf_right_child_of_split,
          is_in_right_child);

      if (opposite_child_should_be_updated) {
        // if there is no monotone constraint on a split,
        // then there is no relationship between its left and right leaves' values
        if (monotone_type != 0) {
          // these variables correspond to the current split we encounter going
          // up the tree
          int left_child_idx = tree_->left_child(parent_idx);
          int right_child_idx = tree_->right_child(parent_idx);
          bool left_child_is_curr_idx = (left_child_idx == node_idx);
          int opposite_child_idx =
              (left_child_is_curr_idx) ? right_child_idx : left_child_idx;
          bool update_max_constraints_in_opposite_child_leaves =
              (monotone_type < 0) ? left_child_is_curr_idx
                                  : !left_child_is_curr_idx;

          // the opposite child needs to be updated
          // so the code needs to go down in the the opposite child
          // to see which leaves' constraints need to be updated
          GoDownToFindLeavesToUpdate(
              opposite_child_idx,
              *features_of_splits_going_up_from_original_leaf,
              *thresholds_of_splits_going_up_from_original_leaf,
              *was_original_leaf_right_child_of_split,
              update_max_constraints_in_opposite_child_leaves, split_feature,
              split_info, true, true, split_threshold, best_split_per_leaf);
        }

        // if opposite_child_should_be_updated, then it means the path to come up there was relevant,
        // i.e. that it will be helpful going down to determine which leaf
        // is actually contiguous to the original 2 leaves and should be updated
        // so the variables associated with the split need to be recorded
        was_original_leaf_right_child_of_split->push_back(
            tree_->right_child(parent_idx) == node_idx);
        thresholds_of_splits_going_up_from_original_leaf->push_back(
            tree_->threshold_in_bin(parent_idx));
        features_of_splits_going_up_from_original_leaf->push_back(
            tree_->split_feature_inner(parent_idx));
      }

      // since current node is not the root, keep going up
      GoUpToFindLeavesToUpdate(
          parent_idx, features_of_splits_going_up_from_original_leaf,
          thresholds_of_splits_going_up_from_original_leaf,
          was_original_leaf_right_child_of_split, split_feature, split_info,
          split_threshold, best_split_per_leaf);
    }
  }

  void GoDownToFindLeavesToUpdate(
      int node_idx,
      const std::vector<int>& features_of_splits_going_up_from_original_leaf,
      const std::vector<uint32_t>&
          thresholds_of_splits_going_up_from_original_leaf,
      const std::vector<bool>& was_original_leaf_right_child_of_split,
      bool update_max_constraints, int split_feature,
      const SplitInfo& split_info, bool use_left_leaf, bool use_right_leaf,
      uint32_t split_threshold,
      const std::vector<SplitInfo>& best_split_per_leaf) {
    // if leaf
    if (node_idx < 0) {
      int leaf_idx = ~node_idx;

      // splits that are not to be used shall not be updated,
      // included leaf at max depth
      if (best_split_per_leaf[leaf_idx].gain == kMinScore) {
        return;
      }

      std::pair<double, double> min_max_constraints;
      bool something_changed = false;
      // if the current leaf is contiguous with both the new right leaf and the new left leaf
      // then it may need to be greater than the max of the 2 or smaller than the min of the 2
      // otherwise, if the current leaf is contiguous with only one of the 2 new leaves,
      // then it may need to be greater or smaller than it
      if (use_right_leaf && use_left_leaf) {
        min_max_constraints =
            std::minmax(split_info.right_output, split_info.left_output);
      } else if (use_right_leaf && !use_left_leaf) {
        min_max_constraints = std::pair<double, double>(
            split_info.right_output, split_info.right_output);
      } else {
        min_max_constraints = std::pair<double, double>(split_info.left_output,
                                                        split_info.left_output);
      }

#ifdef DEBUG
      if (update_max_constraints) {
        CHECK_GE(min_max_constraints.first, tree_->LeafOutput(leaf_idx));
      } else {
        CHECK_LE(min_max_constraints.second, tree_->LeafOutput(leaf_idx));
      }
#endif
      // depending on which split made the current leaf and the original leaves contiguous,
      // either the min constraint or the max constraint of the current leaf need to be updated
      if (!update_max_constraints) {
        something_changed = entries_[leaf_idx]->UpdateMinAndReturnBoolIfChanged(
            min_max_constraints.second);
      } else {
        something_changed = entries_[leaf_idx]->UpdateMaxAndReturnBoolIfChanged(
            min_max_constraints.first);
      }
      // If constraints were not updated, then there is no need to update the leaf
      if (!something_changed) {
        return;
      }
      leaves_to_update_.push_back(leaf_idx);

    } else {  // if node
      // check if the children are contiguous with the original leaf
      std::pair<bool, bool> keep_going_left_right = ShouldKeepGoingLeftRight(
          node_idx, features_of_splits_going_up_from_original_leaf,
          thresholds_of_splits_going_up_from_original_leaf,
          was_original_leaf_right_child_of_split);
      int inner_feature = tree_->split_feature_inner(node_idx);
      uint32_t threshold = tree_->threshold_in_bin(node_idx);
      bool is_split_numerical = tree_->IsNumericalSplit(node_idx);
      bool use_left_leaf_for_update_right = true;
      bool use_right_leaf_for_update_left = true;
      // if the split is on the same feature (categorical variables not supported)
      // then depending on the threshold,
      // the current left child may not be contiguous with the original right leaf,
      // or the current right child may not be contiguous with the original left leaf
      if (is_split_numerical && inner_feature == split_feature) {
        if (threshold >= split_threshold) {
          use_left_leaf_for_update_right = false;
        }
        if (threshold <= split_threshold) {
          use_right_leaf_for_update_left = false;
        }
      }

      // go down left
      if (keep_going_left_right.first) {
        GoDownToFindLeavesToUpdate(
            tree_->left_child(node_idx),
            features_of_splits_going_up_from_original_leaf,
            thresholds_of_splits_going_up_from_original_leaf,
            was_original_leaf_right_child_of_split, update_max_constraints,
            split_feature, split_info, use_left_leaf,
            use_right_leaf_for_update_left && use_right_leaf, split_threshold,
            best_split_per_leaf);
      }
      // go down right
      if (keep_going_left_right.second) {
        GoDownToFindLeavesToUpdate(
            tree_->right_child(node_idx),
            features_of_splits_going_up_from_original_leaf,
            thresholds_of_splits_going_up_from_original_leaf,
            was_original_leaf_right_child_of_split, update_max_constraints,
            split_feature, split_info,
            use_left_leaf_for_update_right && use_left_leaf, use_right_leaf,
            split_threshold, best_split_per_leaf);
      }
    }
  }

  std::pair<bool, bool> ShouldKeepGoingLeftRight(
      int node_idx,
      const std::vector<int>& features_of_splits_going_up_from_original_leaf,
      const std::vector<uint32_t>&
          thresholds_of_splits_going_up_from_original_leaf,
      const std::vector<bool>& was_original_leaf_right_child_of_split) {
    int inner_feature = tree_->split_feature_inner(node_idx);
    uint32_t threshold = tree_->threshold_in_bin(node_idx);
    bool is_split_numerical = tree_->IsNumericalSplit(node_idx);

    bool keep_going_right = true;
    bool keep_going_left = true;
    // left and right nodes are checked to find out if they are contiguous with
    // the original leaves if so the algorithm should keep going down these nodes
    // to update constraints
    if (is_split_numerical) {
      for (size_t i = 0;
           i < features_of_splits_going_up_from_original_leaf.size(); ++i) {
        if (features_of_splits_going_up_from_original_leaf[i] ==
            inner_feature) {
          if (threshold >=
                  thresholds_of_splits_going_up_from_original_leaf[i] &&
              !was_original_leaf_right_child_of_split[i]) {
            keep_going_right = false;
            if (!keep_going_left) {
              break;
            }
          }
          if (threshold <=
                  thresholds_of_splits_going_up_from_original_leaf[i] &&
              was_original_leaf_right_child_of_split[i]) {
            keep_going_left = false;
            if (!keep_going_right) {
              break;
            }
          }
        }
      }
    }
    return std::pair<bool, bool>(keep_going_left, keep_going_right);
  }

 protected:
  const Config* config_;
  std::vector<int> leaves_to_update_;
  // add parent node information
  std::vector<int> node_parent_;
  // Keeps track of the monotone splits above the leaf
  std::vector<bool> leaf_is_in_monotone_subtree_;
};

class AdvancedLeafConstraints : public IntermediateLeafConstraints {
 public:
  AdvancedLeafConstraints(const Config *config, int num_leaves,
                          int num_features)
      : IntermediateLeafConstraints(config, num_leaves) {
    for (int i = 0; i < num_leaves; ++i) {
      entries_[i].reset(new AdvancedConstraintEntry(num_features));
    }
  }

  // at any point in time, for an index i, the constraint constraint[i] has to
  // be valid on [threshold[i]: threshold[i + 1]) (or [threshold[i]: +inf) if i
  // is the last index of the array)
  void UpdateConstraints(FeatureMinOrMaxConstraints* feature_constraint,
                         double extremum, uint32_t it_start, uint32_t it_end,
                         bool use_max_operator, uint32_t last_threshold) {
    bool start_done = false;
    bool end_done = false;
    // previous constraint have to be tracked
    // for example when adding a constraints cstr2 on thresholds [1:2),
    // on an existing constraints cstr1 on thresholds [0, +inf),
    // the thresholds and constraints must become
    // [0, 1, 2] and [cstr1, cstr2, cstr1]
    // so since we loop through thresholds only once,
    // the previous constraint that still applies needs to be recorded
    double previous_constraint = use_max_operator
      ? -std::numeric_limits<double>::max()
      : std::numeric_limits<double>::max();
    double current_constraint;
    for (size_t i = 0; i < feature_constraint->thresholds.size(); ++i) {
      current_constraint = feature_constraint->constraints[i];
      // easy case when the thresholds match
      if (feature_constraint->thresholds[i] == it_start) {
        feature_constraint->constraints[i] =
            (use_max_operator)
                ? std::max(extremum, feature_constraint->constraints[i])
                : std::min(extremum, feature_constraint->constraints[i]);
        start_done = true;
      }
      if (feature_constraint->thresholds[i] > it_start) {
        // existing constraint is updated if there is a need for it
        if (feature_constraint->thresholds[i] < it_end) {
          feature_constraint->constraints[i] =
              (use_max_operator)
                  ? std::max(extremum, feature_constraint->constraints[i])
                  : std::min(extremum, feature_constraint->constraints[i]);
        }
        // when thresholds don't match, a new threshold
        // and a new constraint may need to be inserted
        if (!start_done) {
          start_done = true;
          if ((use_max_operator && extremum > previous_constraint) ||
              (!use_max_operator && extremum < previous_constraint)) {
            feature_constraint->constraints.insert(
                feature_constraint->constraints.begin() + i, extremum);
            feature_constraint->thresholds.insert(
                feature_constraint->thresholds.begin() + i, it_start);
            ++i;
          }
        }
      }
      // easy case when the end thresholds match
      if (feature_constraint->thresholds[i] == it_end) {
        end_done = true;
        break;
      }
      // if they don't then, the previous constraint needs to be added back
      // where the current one ends
      if (feature_constraint->thresholds[i] > it_end) {
        if (i != 0 &&
            previous_constraint != feature_constraint->constraints[i - 1]) {
          feature_constraint->constraints.insert(
              feature_constraint->constraints.begin() + i, previous_constraint);
          feature_constraint->thresholds.insert(
              feature_constraint->thresholds.begin() + i, it_end);
        }
        end_done = true;
        break;
      }
      // If 2 successive constraints are the same then the second one may as
      // well be deleted
      if (i != 0 && feature_constraint->constraints[i] ==
                        feature_constraint->constraints[i - 1]) {
        feature_constraint->constraints.erase(
            feature_constraint->constraints.begin() + i);
        feature_constraint->thresholds.erase(
            feature_constraint->thresholds.begin() + i);
        previous_constraint = current_constraint;
        --i;
      }
      previous_constraint = current_constraint;
    }
    // if the loop didn't get to an index greater than it_start, it needs to be
    // added at the end
    if (!start_done) {
      if ((use_max_operator &&
           extremum > feature_constraint->constraints.back()) ||
          (!use_max_operator &&
           extremum < feature_constraint->constraints.back())) {
        feature_constraint->constraints.push_back(extremum);
        feature_constraint->thresholds.push_back(it_start);
      } else {
        end_done = true;
      }
    }
    // if we didn't get to an index after it_end, then the previous constraint
    // needs to be set back, unless it_end goes up to the last bin of the feature
    if (!end_done && it_end != last_threshold &&
        previous_constraint != feature_constraint->constraints.back()) {
      feature_constraint->constraints.push_back(previous_constraint);
      feature_constraint->thresholds.push_back(it_end);
    }
  }

  // this function is called only when computing constraints when the monotone
  // precise mode is set to true
  // it makes sure that it is worth it to visit a branch, as it could
  // not contain any relevant constraint (for example if the a branch
  // with bigger values is also constraining the original leaf, then
  // it is useless to visit the branch with smaller values)
  std::pair<bool, bool>
  LeftRightContainsRelevantInformation(bool min_constraints_to_be_updated,
                                       int feature,
                                       bool split_feature_is_inner_feature) {
    if (split_feature_is_inner_feature) {
      return std::pair<bool, bool>(true, true);
    }
    int8_t monotone_type = config_->monotone_constraints[feature];
    if (monotone_type == 0) {
      return std::pair<bool, bool>(true, true);
    }
    if ((monotone_type == -1 && min_constraints_to_be_updated) ||
        (monotone_type == 1 && !min_constraints_to_be_updated)) {
      return std::pair<bool, bool>(true, false);
    } else {
      //    Same as
      //    if ((monotone_type == 1 && min_constraints_to_be_updated) ||
      //        (monotone_type == -1 && !min_constraints_to_be_updated))
      return std::pair<bool, bool>(false, true);
    }
  }

  // this function goes down in a subtree to find the
  // constraints that would apply on the original leaf
  void GoDownToFindConstrainingLeaves(
      int feature_for_constraint, int root_monotone_feature, int node_idx,
      bool min_constraints_to_be_updated, uint32_t it_start, uint32_t it_end,
      const std::vector<int> &features_of_splits_going_up_from_original_leaf,
      const std::vector<uint32_t> &
          thresholds_of_splits_going_up_from_original_leaf,
      const std::vector<bool> &was_original_leaf_right_child_of_split,
      FeatureMinOrMaxConstraints* feature_constraint, uint32_t last_threshold) {
    double extremum;
    // if leaf, then constraints need to be updated according to its value
    if (node_idx < 0) {
      extremum = tree_->LeafOutput(~node_idx);
#ifdef DEBUG
      CHECK(it_start < it_end);
#endif
      UpdateConstraints(feature_constraint, extremum, it_start, it_end,
                        min_constraints_to_be_updated, last_threshold);
    } else {  // if node, keep going down the tree
      // check if the children are contiguous to the original leaf and therefore
      // potentially constraining
      std::pair<bool, bool> keep_going_left_right = ShouldKeepGoingLeftRight(
          node_idx, features_of_splits_going_up_from_original_leaf,
          thresholds_of_splits_going_up_from_original_leaf,
          was_original_leaf_right_child_of_split);
      int inner_feature = tree_->split_feature_inner(node_idx);
      int feature = tree_->split_feature(node_idx);
      uint32_t threshold = tree_->threshold_in_bin(node_idx);

      bool split_feature_is_inner_feature =
          (inner_feature == feature_for_constraint);
      bool split_feature_is_monotone_feature =
          (root_monotone_feature == feature_for_constraint);
      // make sure that both children contain values that could
      // potentially help determine the true constraints for the original leaf
      std::pair<bool, bool> left_right_contain_relevant_information =
          LeftRightContainsRelevantInformation(
              min_constraints_to_be_updated, feature,
              split_feature_is_inner_feature &&
                  !split_feature_is_monotone_feature);
      // if both children are contiguous to the original leaf
      // but one contains values greater than the other
      // then no need to go down in both
      if (keep_going_left_right.first &&
          (left_right_contain_relevant_information.first ||
           !keep_going_left_right.second)) {
        // update thresholds based on going left
        uint32_t new_it_end = split_feature_is_inner_feature
                                  ? std::min(threshold + 1, it_end)
                                  : it_end;
        GoDownToFindConstrainingLeaves(
            feature_for_constraint, root_monotone_feature,
            tree_->left_child(node_idx), min_constraints_to_be_updated,
            it_start, new_it_end,
            features_of_splits_going_up_from_original_leaf,
            thresholds_of_splits_going_up_from_original_leaf,
            was_original_leaf_right_child_of_split, feature_constraint,
            last_threshold);
      }
      if (keep_going_left_right.second &&
          (left_right_contain_relevant_information.second ||
           !keep_going_left_right.first)) {
        // update thresholds based on going right
        uint32_t new_it_start = split_feature_is_inner_feature
                                    ? std::max(threshold + 1, it_start)
                                    : it_start;
        GoDownToFindConstrainingLeaves(
            feature_for_constraint, root_monotone_feature,
            tree_->right_child(node_idx), min_constraints_to_be_updated,
            new_it_start, it_end,
            features_of_splits_going_up_from_original_leaf,
            thresholds_of_splits_going_up_from_original_leaf,
            was_original_leaf_right_child_of_split, feature_constraint,
            last_threshold);
      }
    }
  }

  // this function is only used if the monotone precise mode is enabled
  // it recursively goes up the tree then down to find leaf that
  // are constraining the current leaf
  void GoUpToFindConstrainingLeaves(
      int feature_for_constraint, int node_idx,
      std::vector<int>* features_of_splits_going_up_from_original_leaf,
      std::vector<uint32_t>* thresholds_of_splits_going_up_from_original_leaf,
      std::vector<bool>* was_original_leaf_right_child_of_split,
      FeatureMinOrMaxConstraints* feature_constraint,
      bool min_constraints_to_be_updated, uint32_t it_start, uint32_t it_end,
      uint32_t last_threshold) final {
    int parent_idx =
        (node_idx < 0) ? tree_->leaf_parent(~node_idx) : node_parent_[node_idx];
    // if not at the root
    if (parent_idx != -1) {
      int inner_feature = tree_->split_feature_inner(parent_idx);
      int feature = tree_->split_feature(parent_idx);
      int8_t monotone_type = config_->monotone_constraints[feature];
      bool is_in_right_child = tree_->right_child(parent_idx) == node_idx;
      bool is_split_numerical = tree_->IsNumericalSplit(parent_idx);
      uint32_t threshold = tree_->threshold_in_bin(parent_idx);

      // by going up, more information about the position of the
      // original leaf are gathered so the starting and ending
      // thresholds can be updated, which will save some time later
      if ((feature_for_constraint == inner_feature) && is_split_numerical) {
        if (is_in_right_child) {
          it_start = std::max(threshold, it_start);
        } else {
          it_end = std::min(threshold + 1, it_end);
        }
#ifdef DEBUG
        CHECK(it_start < it_end);
#endif
      }

      // this is just an optimisation not to waste time going down in subtrees
      // where there won't be any new constraining leaf
      bool opposite_child_necessary_to_update_constraints =
          OppositeChildShouldBeUpdated(
              is_split_numerical,
              *features_of_splits_going_up_from_original_leaf, inner_feature,
              *was_original_leaf_right_child_of_split, is_in_right_child);

      if (opposite_child_necessary_to_update_constraints) {
        // if there is no monotone constraint on a split,
        // then there is no relationship between its left and right leaves'
        // values
        if (monotone_type != 0) {
          int left_child_idx = tree_->left_child(parent_idx);
          int right_child_idx = tree_->right_child(parent_idx);
          bool left_child_is_curr_idx = (left_child_idx == node_idx);

          bool update_min_constraints_in_curr_child_leaf =
              (monotone_type < 0) ? left_child_is_curr_idx
                                  : !left_child_is_curr_idx;
          if (update_min_constraints_in_curr_child_leaf ==
              min_constraints_to_be_updated) {
            int opposite_child_idx =
                (left_child_is_curr_idx) ? right_child_idx : left_child_idx;

            // go down in the opposite branch to find potential
            // constraining leaves
            GoDownToFindConstrainingLeaves(
                feature_for_constraint, inner_feature, opposite_child_idx,
                min_constraints_to_be_updated, it_start, it_end,
                *features_of_splits_going_up_from_original_leaf,
                *thresholds_of_splits_going_up_from_original_leaf,
                *was_original_leaf_right_child_of_split, feature_constraint,
                last_threshold);
          }
        }
        // if opposite_child_should_be_updated, then it means the path to come
        // up there was relevant,
        // i.e. that it will be helpful going down to determine which leaf
        // is actually contiguous to the original leaf and constraining
        // so the variables associated with the split need to be recorded
        was_original_leaf_right_child_of_split->push_back(is_in_right_child);
        thresholds_of_splits_going_up_from_original_leaf->push_back(threshold);
        features_of_splits_going_up_from_original_leaf->push_back(inner_feature);
      }

      // since current node is not the root, keep going up
      if (parent_idx != 0) {
        GoUpToFindConstrainingLeaves(
            feature_for_constraint, parent_idx,
            features_of_splits_going_up_from_original_leaf,
            thresholds_of_splits_going_up_from_original_leaf,
            was_original_leaf_right_child_of_split, feature_constraint,
            min_constraints_to_be_updated, it_start, it_end, last_threshold);
      }
    }
  }
};

LeafConstraintsBase* LeafConstraintsBase::Create(const Config* config,
                                                 int num_leaves, int num_features) {
  if (config->monotone_constraints_method == "intermediate") {
    return new IntermediateLeafConstraints(config, num_leaves);
  }
  if (config->monotone_constraints_method == "advanced") {
    return new AdvancedLeafConstraints(config, num_leaves, num_features);
  }
  return new BasicLeafConstraints(num_leaves);
}

}  // namespace LightGBM
#endif  // LIGHTGBM_SRC_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_
