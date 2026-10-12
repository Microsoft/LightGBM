/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/c_api.h>
#include <LightGBM/dataset.h>

#include <iostream>
#include <string>
#include <vector>

using LightGBM::Dataset;
using LightGBM::Log;
using LightGBM::TestUtils;

void test_stream_dense(
  int8_t creation_type,
  DatasetHandle ref_dataset_handle,
  int32_t nrows,
  int32_t ncols,
  int32_t nclasses,
  int batch_count,
  const std::vector<double>* features,
  const std::vector<float>* labels,
  const std::vector<float>* weights,
  const std::vector<double>* init_scores,
  const std::vector<int32_t>* groups) {
  Log::Info("Streaming %d rows dense data with a batch size of %d", nrows, batch_count);
  DatasetHandle dataset_handle = nullptr;
  Dataset* dataset = nullptr;

  int has_weights = weights != nullptr;
  int has_init_scores = init_scores != nullptr;
  int has_queries = groups != nullptr;

  bool succeeded = true;
  std::string exceptionText("");

  try {
    int result = 0;
    switch (creation_type) {
      case 0: {
        Log::Info("Creating Dataset using LGBM_DatasetCreateFromSampledColumn, %d rows dense data with a batch size of %d", nrows, batch_count);

        // construct sample data first (use all data for convenience and since size is small)
        std::vector<std::vector<double>> sample_values(ncols);
        std::vector<std::vector<int>> sample_idx(ncols);
        const double* current_val = features->data();
        for (int32_t idx = 0; idx < nrows; ++idx) {
          for (int32_t k = 0; k < ncols; ++k) {
            if (std::fabs(*current_val) > 1e-35f || std::isnan(*current_val)) {
              sample_values[k].emplace_back(*current_val);
              sample_idx[k].emplace_back(static_cast<int>(idx));
            }
            current_val++;
          }
        }

        std::vector<int> sample_sizes;
        std::vector<double*> sample_values_ptrs;
        std::vector<int*> sample_idx_ptrs;
        for (int32_t i = 0; i < ncols; ++i) {
          sample_values_ptrs.push_back(sample_values[i].data());
          sample_idx_ptrs.push_back(sample_idx[i].data());
          sample_sizes.push_back(static_cast<int>(sample_values[i].size()));
        }

        result = LGBM_DatasetCreateFromSampledColumn(
          sample_values_ptrs.data(),
          sample_idx_ptrs.data(),
          ncols,
          sample_sizes.data(),
          nrows,
          nrows,
          nrows,
          "max_bin=15",
          &dataset_handle);
        EXPECT_EQ(0, result) << "LGBM_DatasetCreateFromSampledColumn result code: " << result;

        result = LGBM_DatasetInitStreaming(dataset_handle, has_weights, has_init_scores, has_queries, nclasses, 1, -1);
        EXPECT_EQ(0, result) << "LGBM_DatasetInitStreaming result code: " << result;
        break;
      }

      case 1:
        Log::Info("Creating Dataset using LGBM_DatasetCreateByReference, %d rows dense data with a batch size of %d", nrows, batch_count);
        result = LGBM_DatasetCreateByReference(ref_dataset_handle, nrows, &dataset_handle);
        EXPECT_EQ(0, result) << "LGBM_DatasetCreateByReference result code: " << result;
        break;
    }

    dataset = static_cast<Dataset*>(dataset_handle);

    Log::Info("Streaming dense dataset, %d rows dense data with a batch size of %d", nrows, batch_count);
    TestUtils::StreamDenseDataset(
      dataset_handle,
      nrows,
      ncols,
      nclasses,
      batch_count,
      features,
      labels,
      weights,
      init_scores,
      groups);

    dataset->FinishLoad();

    TestUtils::AssertMetadata(&dataset->metadata(),
                              labels,
                              weights,
                              init_scores,
                              groups);
  }
  catch (std::exception& ex) {
    succeeded = false;
    exceptionText = std::string(ex.what());
  }

  if (dataset_handle) {
    int result = LGBM_DatasetFree(dataset_handle);
    EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
  }

  if (!succeeded) {
    FAIL() << "Test Dense Stream failed with exception: " << exceptionText;
  }
}

void test_stream_sparse(
  int8_t creation_type,
  DatasetHandle ref_dataset_handle,
  int32_t nrows,
  int32_t ncols,
  int32_t nclasses,
  int batch_count,
  const std::vector<int32_t>* indptr,
  const std::vector<int32_t>* indices,
  const std::vector<double>* vals,
  const std::vector<float>* labels,
  const std::vector<float>* weights,
  const std::vector<double>* init_scores,
  const std::vector<int32_t>* groups) {
  Log::Info("Streaming %d rows sparse data with a batch size of %d", nrows, batch_count);
  DatasetHandle dataset_handle = nullptr;
  Dataset* dataset = nullptr;

  int has_weights = weights != nullptr;
  int has_init_scores = init_scores != nullptr;
  int has_queries = groups != nullptr;

  bool succeeded = true;
  std::string exceptionText("");

  try {
    int result = 0;
    switch (creation_type) {
      case 0: {
        Log::Info("Creating Dataset using LGBM_DatasetCreateFromSampledColumn, %d rows sparse data with a batch size of %d", nrows, batch_count);

        std::vector<std::vector<double>> sample_values(ncols);
        std::vector<std::vector<int>> sample_idx(ncols);
        for (size_t i = 0; i < indptr->size() - 1; ++i) {
          int start_index = indptr->at(i);
          int stop_index = indptr->at(i + 1);
          for (int32_t j = start_index; j < stop_index; ++j) {
            auto val = vals->at(j);
            auto idx = indices->at(j);
            if (std::fabs(val) > 1e-35f || std::isnan(val)) {
              sample_values[idx].emplace_back(val);
              sample_idx[idx].emplace_back(static_cast<int>(i));
            }
          }
        }

        std::vector<int> sample_sizes;
        std::vector<double*> sample_values_ptrs;
        std::vector<int*> sample_idx_ptrs;
        for (int32_t i = 0; i < ncols; ++i) {
          sample_values_ptrs.push_back(sample_values[i].data());
          sample_idx_ptrs.push_back(sample_idx[i].data());
          sample_sizes.push_back(static_cast<int>(sample_values[i].size()));
        }

        result = LGBM_DatasetCreateFromSampledColumn(
          sample_values_ptrs.data(),
          sample_idx_ptrs.data(),
          ncols,
          sample_sizes.data(),
          nrows,
          nrows,
          nrows,
          "max_bin=15",
          &dataset_handle);
        EXPECT_EQ(0, result) << "LGBM_DatasetCreateFromSampledColumn result code: " << result;

        dataset = static_cast<Dataset*>(dataset_handle);
        dataset->InitStreaming(nrows, has_weights, has_init_scores, has_queries, nclasses, 2, -1);
        break;
      }

      case 1:
        Log::Info("Creating Dataset using LGBM_DatasetCreateByReference, %d rows sparse data with a batch size of %d", nrows, batch_count);
        result = LGBM_DatasetCreateByReference(ref_dataset_handle, nrows, &dataset_handle);
        EXPECT_EQ(0, result) << "LGBM_DatasetCreateByReference result code: " << result;
        break;
    }

    dataset = static_cast<Dataset*>(dataset_handle);

    Log::Info("Streaming sparse dataset, %d rows sparse data with a batch size of %d", nrows, batch_count);
    TestUtils::StreamSparseDataset(
      dataset_handle,
      nrows,
      nclasses,
      batch_count,
      indptr,
      indices,
      vals,
      labels,
      weights,
      init_scores,
      groups);

    dataset->FinishLoad();

    TestUtils::AssertMetadata(&dataset->metadata(),
                              labels,
                              weights,
                              init_scores,
                              groups);
  }
  catch (std::exception& ex) {
    succeeded = false;
    exceptionText = std::string(ex.what());
  }

  if (dataset_handle) {
    int result = LGBM_DatasetFree(dataset_handle);
    EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
  }

  if (!succeeded) {
    FAIL() << "Test Sparse Stream failed with exception: " << exceptionText;
  }
}

TEST(Stream, PushDenseRowsWithMetadata) {
  // Load some test data
  DatasetHandle ref_dataset_handle;
  const char* params = "max_bin=15";
  // Use the smaller ".test" data because we don't care about the actual data and it's smaller
  int result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.test", params, &ref_dataset_handle);
  EXPECT_EQ(0, result) << "LoadDatasetFromExamples result code: " << result;

  Dataset* ref_dataset = static_cast<Dataset*>(ref_dataset_handle);
  auto noriginalrows = ref_dataset->num_data();
  Log::Info("Row count: %d", noriginalrows);
  Log::Info("Feature group count: %d", ref_dataset->num_features());

  // Add some fake initial_scores and groups so we can test streaming them
  int nclasses = 2;  // choose > 1 just to test multi-class handling
  std::vector<double> unused_init_scores;
  unused_init_scores.resize(noriginalrows * nclasses);
  std::vector<int32_t> unused_groups;
  unused_groups.assign(noriginalrows, 1);
  result = LGBM_DatasetSetField(ref_dataset_handle, "init_score", unused_init_scores.data(), noriginalrows * nclasses, 1);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField init_score result code: " << result;
  result = LGBM_DatasetSetField(ref_dataset_handle, "group", unused_groups.data(), noriginalrows, 2);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField group result code: " << result;

  // Now use the reference dataset schema to make some testable Datasets with N rows each
  int32_t nrows = 1000;
  int32_t ncols = ref_dataset->num_features();
  std::vector<double> features;
  std::vector<float> labels;
  std::vector<float> weights;
  std::vector<double> init_scores;
  std::vector<int32_t> groups;

  Log::Info("Creating random data");
  TestUtils::CreateRandomDenseData(nrows, ncols, nclasses, &features, &labels, &weights, &init_scores, &groups);

  const std::vector<int32_t> batch_counts = { 1, nrows / 100, nrows / 10, nrows };
  const std::vector<int8_t> creation_types = { 0, 1 };

  for (size_t i = 0; i < creation_types.size(); ++i) {  // from sampled data or reference
    for (size_t j = 0; j < batch_counts.size(); ++j) {
      auto type = creation_types[i];
      auto batch_count = batch_counts[j];
      test_stream_dense(type, ref_dataset_handle, nrows, ncols, nclasses, batch_count, &features, &labels, &weights, &init_scores, &groups);
    }
  }

  result = LGBM_DatasetFree(ref_dataset_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
}

TEST(Stream, PushSparseRowsWithMetadata) {
  // Load some test data
  DatasetHandle ref_dataset_handle;
  const char* params = "max_bin=15";
  // Use the smaller ".test" data because we don't care about the actual data and it's smaller
  int result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.test", params, &ref_dataset_handle);
  EXPECT_EQ(0, result) << "LoadDatasetFromExamples result code: " << result;

  Dataset* ref_dataset = static_cast<Dataset*>(ref_dataset_handle);
  auto noriginalrows = ref_dataset->num_data();
  Log::Info("Row count: %d", noriginalrows);
  Log::Info("Feature group count: %d", ref_dataset->num_features());

  // Add some fake initial_scores and groups so we can test streaming them
  int32_t nclasses = 2;
  std::vector<double> unused_init_scores;
  unused_init_scores.resize(noriginalrows * nclasses);
  std::vector<int32_t> unused_groups;
  unused_groups.assign(noriginalrows, 1);
  result = LGBM_DatasetSetField(ref_dataset_handle, "init_score", unused_init_scores.data(), noriginalrows * nclasses, 1);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField init_score result code: " << result;
  result = LGBM_DatasetSetField(ref_dataset_handle, "group", unused_groups.data(), noriginalrows, 2);
  EXPECT_EQ(0, result) << "LGBM_DatasetSetField group result code: " << result;

  // Now use the reference dataset schema to make some testable Datasets with N rows each
  int32_t nrows = 1000;
  int32_t ncols = ref_dataset->num_features();
  std::vector<int32_t> indptr;
  std::vector<int32_t> indices;
  std::vector<double> vals;
  std::vector<float> labels;
  std::vector<float> weights;
  std::vector<double> init_scores;
  std::vector<int32_t> groups;

  Log::Info("Creating random data");
  float sparse_percent = .1f;
  TestUtils::CreateRandomSparseData(nrows, ncols, nclasses, sparse_percent, &indptr, &indices, &vals, &labels, &weights, &init_scores, &groups);

  const std::vector<int32_t> batch_counts = { 1, nrows / 100, nrows / 10, nrows };
  const std::vector<int8_t> creation_types = { 0, 1 };

  for (size_t i = 0; i < creation_types.size(); ++i) {  // from sampled data or reference
    for (size_t j = 0; j < batch_counts.size(); ++j) {
      auto type = creation_types[i];
      auto batch_count = batch_counts[j];
      test_stream_sparse(type, ref_dataset_handle, nrows, ncols, nclasses, batch_count, &indptr, &indices, &vals, &labels, &weights, &init_scores, &groups);
    }
  }

  result = LGBM_DatasetFree(ref_dataset_handle);
  EXPECT_EQ(0, result) << "LGBM_DatasetFree result code: " << result;
}
