/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <testutils.h>
#include <LightGBM/c_api.h>

#include <string>
#include <vector>

using LightGBM::TestUtils;

namespace {

void TrainSmallBooster(BoosterHandle* booster_handle_out, DatasetHandle* train_dataset_out, int* n_features_out) {
  int result = TestUtils::LoadDatasetFromExamples(
      "binary_classification/binary.train", "max_bin=15", train_dataset_out);
  ASSERT_EQ(0, result) << "LoadDatasetFromExamples result code: " << result;

  result = LGBM_BoosterCreate(*train_dataset_out, "app=binary metric=auc num_leaves=31 verbose=0", booster_handle_out);
  ASSERT_EQ(0, result) << "LGBM_BoosterCreate result code: " << result;

  for (int i = 0; i < 5; i++) {
    int produced_empty_tree;
    result = LGBM_BoosterUpdateOneIter(*booster_handle_out, &produced_empty_tree);
    ASSERT_EQ(0, result) << "LGBM_BoosterUpdateOneIter result code: " << result;
  }

  result = LGBM_BoosterGetNumFeature(*booster_handle_out, n_features_out);
  ASSERT_EQ(0, result) << "LGBM_BoosterGetNumFeature result code: " << result;
}

}  // namespace

// Regression test for https://github.com/microsoft/LightGBM/issues/7325: passing a null
// out_result buffer used to segfault (an unconditional write through the null pointer inside
// GBDT::Predict/PredictSingleRow) instead of returning a catchable error, even though the
// out_len-only-first pattern is a common convention in other prediction-sizing APIs.
TEST(Predict, ForMatNullOutResultReturnsErrorInsteadOfCrashing) {
  BoosterHandle booster_handle;
  DatasetHandle train_dataset;
  int n_features;
  TrainSmallBooster(&booster_handle, &train_dataset, &n_features);

  std::vector<double> row(n_features, 0.0);
  int64_t out_len;
  int result = LGBM_BoosterPredictForMat(
      booster_handle,
      row.data(),
      C_API_DTYPE_FLOAT64,
      1,              // nrow
      n_features,     // ncol
      1,              // is_row_major
      C_API_PREDICT_NORMAL,
      0,              // start_iteration
      -1,             // num_iteration
      "",
      &out_len,
      nullptr);        // out_result

  EXPECT_NE(0, result) << "LGBM_BoosterPredictForMat should reject a null out_result instead of crashing";
  std::string error_message(LGBM_GetLastError());
  EXPECT_NE(std::string::npos, error_message.find("out_result"))
      << "Unexpected error message: " << error_message;

  ASSERT_EQ(0, LGBM_BoosterFree(booster_handle));
  ASSERT_EQ(0, LGBM_DatasetFree(train_dataset));
}

TEST(Predict, ForMatSingleRowNullOutResultReturnsErrorInsteadOfCrashing) {
  BoosterHandle booster_handle;
  DatasetHandle train_dataset;
  int n_features;
  TrainSmallBooster(&booster_handle, &train_dataset, &n_features);

  std::vector<double> row(n_features, 0.0);
  int64_t out_len;
  int result = LGBM_BoosterPredictForMatSingleRow(
      booster_handle,
      row.data(),
      C_API_DTYPE_FLOAT64,
      n_features,     // ncol
      1,              // is_row_major
      C_API_PREDICT_NORMAL,
      0,              // start_iteration
      -1,             // num_iteration
      "",
      &out_len,
      nullptr);        // out_result

  EXPECT_NE(0, result) << "LGBM_BoosterPredictForMatSingleRow should reject a null out_result instead of crashing";
  std::string error_message(LGBM_GetLastError());
  EXPECT_NE(std::string::npos, error_message.find("out_result"))
      << "Unexpected error message: " << error_message;

  ASSERT_EQ(0, LGBM_BoosterFree(booster_handle));
  ASSERT_EQ(0, LGBM_DatasetFree(train_dataset));
}
