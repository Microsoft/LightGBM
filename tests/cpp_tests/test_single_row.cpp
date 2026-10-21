/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <LightGBM/c_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

using LightGBM::TestUtils;

void test_predict_type(int predict_type, int num_predicts) {
    // Load some test data
    int result;

    DatasetHandle train_dataset;
    result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.train", "max_bin=15", &train_dataset);
    EXPECT_EQ(0, result) << "LoadDatasetFromExamples train result code: " << result;

    BoosterHandle booster_handle;
    result = LGBM_BoosterCreate(train_dataset, "app=binary metric=auc num_leaves=31 verbose=0", &booster_handle);
    EXPECT_EQ(0, result) << "LGBM_BoosterCreate result code: " << result;

    for (int i = 0; i < 51; i++) {
        int produced_empty_tree;
        result = LGBM_BoosterUpdateOneIter(
            booster_handle,
            &produced_empty_tree);
        EXPECT_EQ(0, result) << "LGBM_BoosterUpdateOneIter result code: " << result;
    }

    int n_features;
    result = LGBM_BoosterGetNumFeature(
        booster_handle,
        &n_features);
    EXPECT_EQ(0, result) << "LGBM_BoosterGetNumFeature result code: " << result;
    EXPECT_EQ(28, n_features) << "LGBM_BoosterGetNumFeature number of features: " << n_features;

    // Run a single row prediction and compare with regular Mat prediction:
    int64_t output_size;
    result = LGBM_BoosterCalcNumPredict(
        booster_handle,
        1,
        predict_type,          // predict_type
        0,                     // start_iteration
        -1,                    // num_iteration
        &output_size);
    EXPECT_EQ(0, result) << "LGBM_BoosterCalcNumPredict result code: " << result;
    EXPECT_EQ(num_predicts, output_size) << "LGBM_BoosterCalcNumPredict output size: " << output_size;

    std::ifstream test_file("examples/binary_classification/binary.test");
    std::vector<double> test;
    double x;
    int test_set_size = 0;
    while (test_file >> x) {
        if (test_set_size % (n_features + 1) == 0) {
            // Drop the result from the dataset, we only care about checking that prediction results are equal
            // in both cases
            test_file >> x;
            test_set_size++;
        }
        test.push_back(x);
        test_set_size++;
    }
    EXPECT_EQ(test_set_size % (n_features + 1), 0) << "Test size mismatch with dataset size (%)";
    test_set_size /= (n_features + 1);
    EXPECT_EQ(test_set_size, 500) << "Improperly parsed test file (test_set_size)";
    EXPECT_EQ(test.size(), test_set_size * n_features) << "Improperly parsed test file (test len)";

    std::vector<double> mat_output(output_size * test_set_size, -1);
    int64_t written;
    result = LGBM_BoosterPredictForMat(
        booster_handle,
        &test[0],
        C_API_DTYPE_FLOAT64,
        test_set_size,         // nrow
        n_features,            // ncol
        1,                     // is_row_major
        predict_type,          // predict_type
        0,                     // start_iteration
        -1,                    // num_iteration
        "",
        &written,
        &mat_output[0]);
    EXPECT_EQ(0, result) << "LGBM_BoosterPredictForMat result code: " << result;

    // Test LGBM_BoosterPredictForMat in multi-threaded mode
    const int kNThreads = 10;
    const int numIterations = 5;
    std::vector<std::thread> predict_for_mat_threads(kNThreads);
    for (int i = 0; i < kNThreads; i++) {
        predict_for_mat_threads[i] = std::thread(
            [
                i, test_set_size, output_size, n_features,
                    test = &test[0], booster_handle, predict_type, numIterations
            ]() {
                for (int j = 0; j < numIterations; j++) {
                    int result;
                    std::vector<double> mat_output(output_size * test_set_size, -1);
                    int64_t written;
                    result = LGBM_BoosterPredictForMat(
                        booster_handle,
                        &test[0],
                        C_API_DTYPE_FLOAT64,
                        test_set_size,         // nrow
                        n_features,            // ncol
                        1,                     // is_row_major
                        predict_type,          // predict_type
                        0,                     // start_iteration
                        -1,                    // num_iteration
                        "",
                        &written,
                        &mat_output[0]);
                    EXPECT_EQ(0, result) << "LGBM_BoosterPredictForMat result code: " << result;
                }
            });
    }
    for (std::thread& t : predict_for_mat_threads) {
        t.join();
    }

    const char* fast_config_parameters[] = {"", "predict_disable_fast_lock=true"};
    for (const char* fast_config_parameter : fast_config_parameters) {
        // Now let's run with the single row fast prediction API:
        FastConfigHandle fast_configs[kNThreads];
        for (int i = 0; i < kNThreads; i++) {
            result = LGBM_BoosterPredictForMatSingleRowFastInit(
                booster_handle,
                predict_type,          // predict_type
                0,                     // start_iteration
                -1,                    // num_iteration
                C_API_DTYPE_FLOAT64,
                n_features,
                fast_config_parameter,
                &fast_configs[i]);
            EXPECT_EQ(0, result) << "LGBM_BoosterPredictForMatSingleRowFastInit result code: " << result;
        }

        std::vector<double> single_row_output(output_size * test_set_size, -1);
        std::vector<std::thread> single_row_threads(kNThreads);
        int batch_size = (test_set_size + kNThreads - 1) / kNThreads;  // round up
        for (int i = 0; i < kNThreads; i++) {
            single_row_threads[i] = std::thread(
                [
                    i, batch_size, test_set_size, output_size, n_features,
                        test = &test[0], fast_configs = &fast_configs[0], single_row_output = &single_row_output[0]
                ]() {
                    int result;
                    int64_t written;
                    for (int j = i * batch_size; j < std::min((i + 1) * batch_size, test_set_size); j++) {
                        result = LGBM_BoosterPredictForMatSingleRowFast(
                            fast_configs[i],
                            &test[j * n_features],
                            &written,
                            &single_row_output[j * output_size]);
                        EXPECT_EQ(0, result) << "LGBM_BoosterPredictForMatSingleRowFast result code: " << result;
                        EXPECT_EQ(written, output_size) << "LGBM_BoosterPredictForMatSingleRowFast unexpected written output size";
                    }
                });
        }
        for (std::thread& t : single_row_threads) {
            t.join();
        }

        EXPECT_EQ(single_row_output, mat_output) << "LGBM_BoosterPredictForMatSingleRowFast output mismatch with LGBM_BoosterPredictForMat";

        // Free all:
        for (int i = 0; i < kNThreads; i++) {
            result = LGBM_FastConfigFree(fast_configs[i]);
            EXPECT_EQ(0, result) << "LGBM_FastConfigFree result code: " << result;
        }
    }

    result = LGBM_BoosterFree(booster_handle);
    EXPECT_EQ(0, result) << "LGBM_BoosterFree result code: " << result;

    result = LGBM_DatasetFree(train_dataset);
    EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
}

TEST(SingleRow, Normal) {
    test_predict_type(C_API_PREDICT_NORMAL, 1);
}

TEST(SingleRow, Contrib) {
    test_predict_type(C_API_PREDICT_CONTRIB, 29);
}

TEST(SingleRow, CSRFastPredictionTypeAndIterationRange) {
    DatasetHandle dataset;
    ASSERT_EQ(0, TestUtils::LoadDatasetFromExamples(
        "binary_classification/binary.train", "max_bin=15", &dataset));
    BoosterHandle booster;
    ASSERT_EQ(0, LGBM_BoosterCreate(
        dataset, "objective=binary num_leaves=7 verbosity=-1 num_threads=1", &booster));
    for (int i = 0; i < 10; ++i) {
        int finished;
        ASSERT_EQ(0, LGBM_BoosterUpdateOneIter(booster, &finished));
    }
    int num_features;
    ASSERT_EQ(0, LGBM_BoosterGetNumFeature(booster, &num_features));
    const int32_t indptr[] = {0, 2};
    const int32_t indices[] = {0, 5};
    const double rows[][2] = {{0.5, 2.0}, {-1.0, 0.0}};
    for (int predict_type : {C_API_PREDICT_NORMAL, C_API_PREDICT_RAW_SCORE,
                             C_API_PREDICT_LEAF_INDEX, C_API_PREDICT_CONTRIB}) {
        for (int start_iteration : {0, 2}) {
            for (int num_iteration : {-1, 3}) {
                SCOPED_TRACE(::testing::Message() << "type=" << predict_type
                    << " start=" << start_iteration << " count=" << num_iteration);
                int64_t expected_size;
                ASSERT_EQ(0, LGBM_BoosterCalcNumPredict(
                    booster, 1, predict_type, start_iteration, num_iteration, &expected_size));
                // Accommodate all output types so an incorrect type reports a test failure.
                const int64_t capacity = std::max<int64_t>(num_features + 1, 10);
                std::vector<std::vector<double>> expected;
                for (const auto& row : rows) {
                    std::vector<double> output(capacity);
                    int64_t written;
                    ASSERT_EQ(0, LGBM_BoosterPredictForCSR(
                        booster, indptr, C_API_DTYPE_INT32, indices, row, C_API_DTYPE_FLOAT64,
                        2, 2, num_features, predict_type, start_iteration, num_iteration,
                        "num_threads=1", &written, output.data()));
                    ASSERT_EQ(expected_size, written);
                    expected.push_back(output);
                }
                FastConfigHandle fast;
                ASSERT_EQ(0, LGBM_BoosterPredictForCSRSingleRowFastInit(
                    booster, predict_type, start_iteration, num_iteration,
                    C_API_DTYPE_FLOAT64, num_features, "num_threads=1", &fast));
                // Reuse the predictor across alternating input rows.
                for (int repeat = 0; repeat < 3; ++repeat) {
                    for (int row = 0; row < 2; ++row) {
                        std::vector<double> output(capacity);
                        int64_t written;
                        ASSERT_EQ(0, LGBM_BoosterPredictForCSRSingleRowFast(
                            fast, indptr, C_API_DTYPE_INT32, indices, rows[row],
                            2, 2, &written, output.data()));
                        EXPECT_EQ(expected_size, written);
                        for (int64_t i = 0; i < expected_size; ++i) {
                            EXPECT_DOUBLE_EQ(expected[row][i], output[i]);
                        }
                    }
                }
                EXPECT_EQ(0, LGBM_FastConfigFree(fast));
            }
        }
    }
    EXPECT_EQ(0, LGBM_BoosterFree(booster));
    EXPECT_EQ(0, LGBM_DatasetFree(dataset));
}
