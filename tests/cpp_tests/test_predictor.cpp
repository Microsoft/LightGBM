/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <LightGBM/c_api.h>
#include <LightGBM/prediction_early_stop.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../src/application/predictor.hpp"

namespace LightGBM {
class PredictorTestPeer {
 public:
  static size_t DenseElements(const Predictor& predictor) {
    size_t size = 0;
    for (const auto& buffer : predictor.predict_buf_) size += buffer.size();
    return size;
  }
};

namespace {

class PredictorBufferTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_num_threads_ = LGBM_DEFAULT_NUM_THREADS;
    OMP_SET_NUM_THREADS(4);
    // A high-dimensional stump with equal leaf counts and a NaN default-left split.
    std::string model =
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\n"
        "max_feature_idx=" + std::to_string(kNumFeatures - 1) +
        "\nobjective=binary sigmoid:1\nfeature_names=";
    for (int i = 0; i < kNumFeatures; ++i) {
      model += "f ";
    }
    model += "\nfeature_infos=";
    for (int i = 0; i < kNumFeatures; ++i) {
      model += "none ";
    }
    model += "\n\nTree=0\nnum_leaves=2\nnum_cat=0\nsplit_feature=" +
        std::to_string(kNumFeatures - 1) +
        "\nsplit_gain=1\nthreshold=0.5\ndecision_type=10\n"
        "left_child=-1\nright_child=-2\nleaf_value=-2 3\nleaf_weight=1 1\n"
        "leaf_count=1 1\ninternal_value=0.5\ninternal_weight=2\ninternal_count=2\n"
        "is_linear=0\nshrinkage=1\n\nend of trees\n";
    model_text_ = model;
    boosting_.reset(Boosting::CreateBoosting("gbdt", nullptr, "cpu", 1));
    ASSERT_NE(nullptr, boosting_);
    ASSERT_TRUE(boosting_->LoadModelFromString(model.data(), model.size()));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    rows_ = {{}, {{kNumFeatures - 1, 1.0}}, {{kNumFeatures - 1, nan}},
             {{kNumFeatures - 1, 0.0}}};
    // Explicit zero entries cross the sparsity threshold without changing the score.
    for (int i = 0; i < 4; ++i) {
      std::vector<std::pair<int, double>> row;
      for (int feature = 0; feature < 2048; ++feature) {
        row.emplace_back(feature, 0.0);
      }
      row.insert(row.end(), rows_[i].begin(), rows_[i].end());
      rows_.push_back(std::move(row));
    }
  }

  void TearDown() override {
    OMP_SET_NUM_THREADS(previous_num_threads_);
  }

  static constexpr int kNumFeatures = 100008;
  int previous_num_threads_;
  std::string model_text_;
  std::unique_ptr<Boosting> boosting_;
  std::vector<std::vector<std::pair<int, double>>> rows_;
};

TEST_F(PredictorBufferTest, ReusesBuffersAcrossSparseAndDenseRows) {
  for (int num_threads : {1, 4}) {
    OMP_SET_NUM_THREADS(num_threads);
    for (const std::string storage : {"auto", "array", "map"}) {
      for (int predict_type = 0; predict_type < 4; ++predict_type) {
        Predictor predictor(boosting_.get(), 0, -1, predict_type == 1,
                            predict_type == 2, predict_type == 3, false, 10, 10.0, storage);
        const auto& predict = predictor.GetPredictFunction();
        OMP_INIT_EX();
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1)
        for (int task = 0; task < num_threads; ++task) {
          OMP_LOOP_EX_BEGIN();
          // Each thread starts with empty/hash rows, then allocates and reuses its array.
          // The second pass also checks hash rows after the array has been allocated.
          for (int repeat = 0; repeat < 2; ++repeat) {
            for (size_t row = 0; row < rows_.size(); ++row) {
              SCOPED_TRACE(testing::Message() << "type=" << predict_type << " task=" << task
                           << " repeat=" << repeat << " row=" << row);
              const bool right_leaf = row % 4 == 1;
              const double raw_score = right_leaf ? 3.0 : -2.0;
              std::vector<double> expected(predict_type == 3 ? kNumFeatures + 1 : 1, 0.0);
              if (predict_type == 3) {
                expected[kNumFeatures - 1] = raw_score - 0.5;
                expected[kNumFeatures] = 0.5;
              } else if (predict_type == 2) {
                expected[0] = right_leaf ? 1.0 : 0.0;
              } else {
                expected[0] = predict_type == 1 ? raw_score : 1.0 / (1.0 + std::exp(-raw_score));
              }
              std::vector<double> output(expected.size(), 0.0);
              predict(rows_[row], output.data());
              EXPECT_EQ(expected, output);
            }
          }
          OMP_LOOP_EX_END();
        }
        OMP_THROW_EX();
      }
    }
  }
}

TEST_F(PredictorBufferTest, SparseContributionsMatchDenseContributions) {
  Predictor predictor(boosting_.get(), 0, -1, false, false, true, false, 10, 10.0);
  for (const auto& row : rows_) {
    std::vector<std::unordered_map<int, double>> sparse_output(1);
    predictor.GetPredictSparseFunction()(row, &sparse_output);
    std::vector<double> output(kNumFeatures + 1, 0.0);
    predictor.GetPredictFunction()(row, output.data());
    std::vector<double> expanded(kNumFeatures + 1, 0.0);
    for (const auto& entry : sparse_output[0]) {
      expanded[entry.first] = entry.second;
    }
    EXPECT_EQ(output, expanded);
  }
}


TEST_F(PredictorBufferTest, LeafPredictionTakesPrecedenceOverContributions) {
  for (const std::string storage : {"auto", "array", "map"}) {
    Predictor predictor(boosting_.get(), 0, -1, true, true, true, false, 10, 10.0, storage);
    // Keep a full-sized output so a regression reports a mismatch without overwriting memory.
    std::vector<double> output(kNumFeatures + 1, -123.0);
    predictor.GetPredictFunction()(rows_[1], output.data());
    EXPECT_EQ(1.0, output[0]);
    for (size_t i = 1; i < output.size(); ++i) {
      EXPECT_EQ(-123.0, output[i]);
    }
  }
}

TEST_F(PredictorBufferTest, DenseStorageFollowsSelectedPath) {
  OMP_SET_NUM_THREADS(1);
  for (const std::string storage : {"auto", "array", "map"}) {
    Predictor predictor(boosting_.get(), 0, -1, true, false, false,
                        false, 10, 10.0, storage);
    EXPECT_EQ(storage == "array" ? kNumFeatures : 0,
              PredictorTestPeer::DenseElements(predictor));
    double output;
    predictor.GetPredictFunction()(rows_[1], &output);
    EXPECT_EQ(3.0, output);
    EXPECT_EQ(storage == "array" ? kNumFeatures : 0,
              PredictorTestPeer::DenseElements(predictor));
    predictor.GetPredictFunction()(rows_[5], &output);
    EXPECT_EQ(3.0, output);
    EXPECT_EQ(storage == "map" ? 0 : kNumFeatures,
              PredictorTestPeer::DenseElements(predictor));
    predictor.GetPredictFunction()(rows_[0], &output);
    EXPECT_EQ(-2.0, output);
  }
}

TEST_F(PredictorBufferTest, SparseContributionsDoNotAllocateDenseStorage) {
  OMP_SET_NUM_THREADS(1);
  Predictor predictor(boosting_.get(), 0, -1, false, false, true, false, 10, 10.0);
  std::vector<std::unordered_map<int, double>> output(1);
  predictor.GetPredictSparseFunction()(rows_[1], &output);
  EXPECT_EQ(0, PredictorTestPeer::DenseElements(predictor));
}

TEST_F(PredictorBufferTest, PredictionApisAcceptStorageChanges) {
  BoosterHandle handle = nullptr;
  int iterations = 0;
  ASSERT_EQ(0, LGBM_BoosterLoadModelFromString(model_text_.c_str(), &iterations, &handle));
  const int32_t offsets[] = {0, 1};
  const int32_t index = kNumFeatures - 1;
  const double value = 1.0;
  std::vector<double> dense(kNumFeatures, 0.0);
  dense.back() = value;
  for (const std::string storage : {"map", "array", "auto", "map"}) {
    const std::string parameters = "num_threads=1 predict_feature_storage=" + storage;
    int64_t length = 0;
    double result = 0;
    EXPECT_EQ(0, LGBM_BoosterPredictForCSR(handle, offsets, C_API_DTYPE_INT32,
        &index, &value, C_API_DTYPE_FLOAT64, 2, 1, kNumFeatures, C_API_PREDICT_RAW_SCORE,
        0, -1, parameters.c_str(), &length, &result));
    EXPECT_EQ(1, length);
    EXPECT_EQ(3.0, result);
    EXPECT_EQ(0, LGBM_BoosterPredictForMatSingleRow(handle, dense.data(), C_API_DTYPE_FLOAT64,
        kNumFeatures, 1, C_API_PREDICT_RAW_SCORE, 0, -1, parameters.c_str(), &length, &result));
    EXPECT_EQ(3.0, result);
    FastConfigHandle fast = nullptr;
    ASSERT_EQ(0, LGBM_BoosterPredictForMatSingleRowFastInit(handle, C_API_PREDICT_RAW_SCORE,
        0, -1, C_API_DTYPE_FLOAT64, kNumFeatures, parameters.c_str(), &fast));
    EXPECT_EQ(0, LGBM_BoosterPredictForMatSingleRowFast(fast, dense.data(), &length, &result));
    EXPECT_EQ(3.0, result);
    EXPECT_EQ(0, LGBM_FastConfigFree(fast));
  }
  EXPECT_EQ(0, LGBM_BoosterFree(handle));
}

TEST_F(PredictorBufferTest, DuplicateIndicesKeepLastValue) {
  for (const std::string storage : {"auto", "array", "map"}) {
    Predictor predictor(boosting_.get(), 0, -1, true, false, false, false, 10, 10.0, storage);
    double output;
    predictor.GetPredictFunction()({{kNumFeatures - 1, 0.0}, {kNumFeatures - 1, 1.0}}, &output);
    EXPECT_EQ(3.0, output);
    predictor.GetPredictFunction()({}, &output);
    EXPECT_EQ(-2.0, output);
  }
}

TEST_F(PredictorBufferTest, CategoricalAndLinearModelsMatchAcrossStorageModes) {
  for (bool categorical : {true, false}) {
    std::string text = model_text_;
    auto replace = [&text](const std::string& from, const std::string& to) {
      const auto pos = text.find(from);
      ASSERT_NE(std::string::npos, pos);
      text.replace(pos, from.size(), to);
    };
    if (categorical) {
      replace("num_cat=0", "num_cat=1");
      replace("threshold=0.5", "threshold=0");
      replace("decision_type=10", "decision_type=1");
      replace("is_linear=0", "cat_boundaries=0 1\ncat_threshold=2\nis_linear=0");
    } else {
      replace("is_linear=0", "is_linear=1\nleaf_const=1 2\nnum_features=1 1\nleaf_features=" +
          std::to_string(kNumFeatures - 1) + " " + std::to_string(kNumFeatures - 1) +
          "\nleaf_coeff=2 3");
    }
    std::unique_ptr<Boosting> model(Boosting::CreateBoosting("gbdt", nullptr, "cpu", 1));
    ASSERT_TRUE(model->LoadModelFromString(text.data(), text.size()));
    for (bool leaf : {false, true}) {
      Predictor array(model.get(), 0, -1, true, leaf, false, false, 10, 10.0, "array");
      Predictor map(model.get(), 0, -1, true, leaf, false, false, 10, 10.0, "map");
      for (const auto& row : rows_) {
        double expected, actual;
        array.GetPredictFunction()(row, &expected);
        map.GetPredictFunction()(row, &actual);
        EXPECT_EQ(expected, actual);
      }
    }
  }
}

TEST(PredictorConfigTest, ValidatesStorage) {
  for (const std::string storage : {"auto", "array", "map"}) {
    Config config(Config::Str2Map(("predict_feature_storage=" + storage).c_str()));
    EXPECT_EQ(storage, config.predict_feature_storage);
  }
  EXPECT_THROW(Config(Config::Str2Map("predict_feature_storage=invalid")), std::runtime_error);
}

}  // namespace
}  // namespace LightGBM
