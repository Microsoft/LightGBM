/*!
 * Copyright (c) 2026-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Oliver Borchert
 */

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <gtest/gtest.h>

#include <LightGBM/c_api.h>

#include <vector>

#include <nanoarrow/nanoarrow.hpp>

namespace {

nanoarrow::UniqueSchema MakePrimitiveSchema(ArrowType type) {
  nanoarrow::UniqueSchema schema;
  EXPECT_EQ(ArrowSchemaInitFromType(schema.get(), type), NANOARROW_OK);
  return schema;
}

nanoarrow::UniqueSchema MakeFloatStructSchema(int n_fields) {
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  EXPECT_EQ(ArrowSchemaSetTypeStruct(schema.get(), n_fields), NANOARROW_OK);
  for (int i = 0; i < n_fields; ++i) {
    EXPECT_EQ(ArrowSchemaSetType(schema->children[i], NANOARROW_TYPE_FLOAT), NANOARROW_OK);
  }
  return schema;
}

nanoarrow::UniqueArray MakeFloatArray(const std::vector<float>& values) {
  nanoarrow::UniqueArray array;
  EXPECT_EQ(ArrowArrayInitFromType(array.get(), NANOARROW_TYPE_FLOAT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  for (auto v : values) {
    EXPECT_EQ(ArrowArrayAppendDouble(array.get(), v), NANOARROW_OK);
  }
  EXPECT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);
  return array;
}

nanoarrow::UniqueArray MakeFloatStructArray(const struct ArrowSchema* schema,
                                            const std::vector<std::vector<float>>& columns) {
  nanoarrow::UniqueArray array;
  EXPECT_EQ(ArrowArrayInitFromSchema(array.get(), schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  const size_t n = columns[0].size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t c = 0; c < columns.size(); ++c) {
      EXPECT_EQ(ArrowArrayAppendDouble(array->children[c], columns[c][i]), NANOARROW_OK);
    }
    EXPECT_EQ(ArrowArrayFinishElement(array.get()), NANOARROW_OK);
  }
  EXPECT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);
  return array;
}

}  // namespace

TEST(ArrowDeprecatedTest, DatasetCreateFromArrow) {
  auto schema = MakeFloatStructSchema(2);
  std::vector<std::vector<float>> columns = {
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}};
  auto array = MakeFloatStructArray(schema.get(), columns);

  // Move ownership of schema and array out of the unique wrappers; the
  // deprecated API takes ownership of both.
  ArrowSchema raw_schema;
  schema.move(&raw_schema);
  std::vector<ArrowArray> raw_chunks(1);
  array.move(&raw_chunks[0]);

  DatasetHandle handle = nullptr;
  int result = LGBM_DatasetCreateFromArrow(
      static_cast<int64_t>(raw_chunks.size()), raw_chunks.data(), &raw_schema,
      "max_bin=15", nullptr, &handle);
  ASSERT_EQ(result, 0);
  ASSERT_NE(handle, nullptr);

  int num_data = 0;
  int num_feature = 0;
  ASSERT_EQ(LGBM_DatasetGetNumData(handle, &num_data), 0);
  ASSERT_EQ(LGBM_DatasetGetNumFeature(handle, &num_feature), 0);
  EXPECT_EQ(num_data, 6);
  EXPECT_EQ(num_feature, 2);

  ASSERT_EQ(LGBM_DatasetFree(handle), 0);
}

TEST(ArrowDeprecatedTest, DatasetSetFieldFromArrow) {
  // Create a small dataset from a dense matrix.
  std::vector<double> data = {1.0, 2.0,
                              3.0, 4.0,
                              5.0, 6.0,
                              7.0, 8.0};
  DatasetHandle handle = nullptr;
  ASSERT_EQ(LGBM_DatasetCreateFromMat(data.data(), C_API_DTYPE_FLOAT64, 4, 2, 1,
                                      "max_bin=15", nullptr, &handle),
            0);

  // Set the label using the deprecated Arrow API.
  std::vector<float> label_values = {0.0f, 1.0f, 0.0f, 1.0f};
  auto label_schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
  auto label_array = MakeFloatArray(label_values);

  ArrowSchema raw_schema;
  label_schema.move(&raw_schema);
  std::vector<ArrowArray> raw_chunks(1);
  label_array.move(&raw_chunks[0]);

  ASSERT_EQ(LGBM_DatasetSetFieldFromArrow(
                handle, "label", static_cast<int64_t>(raw_chunks.size()),
                raw_chunks.data(), &raw_schema),
            0);

  int out_len = 0;
  const void* out_ptr = nullptr;
  int out_type = 0;
  ASSERT_EQ(LGBM_DatasetGetField(handle, "label", &out_len, &out_ptr, &out_type), 0);
  EXPECT_EQ(out_type, C_API_DTYPE_FLOAT32);
  ASSERT_EQ(out_len, static_cast<int>(label_values.size()));
  const float* read = static_cast<const float*>(out_ptr);
  for (size_t i = 0; i < label_values.size(); ++i) {
    EXPECT_FLOAT_EQ(read[i], label_values[i]);
  }

  ASSERT_EQ(LGBM_DatasetFree(handle), 0);
}

TEST(ArrowDeprecatedTest, BoosterPredictForArrow) {
  // Train a tiny booster.
  const int nrow = 8;
  const int ncol = 2;
  std::vector<double> data = {1.0, 1.0,
                              2.0, 2.0,
                              3.0, 3.0,
                              4.0, 4.0,
                              5.0, 5.0,
                              6.0, 6.0,
                              7.0, 7.0,
                              8.0, 8.0};
  std::vector<float> labels = {0, 0, 0, 0, 1, 1, 1, 1};

  DatasetHandle dataset = nullptr;
  ASSERT_EQ(LGBM_DatasetCreateFromMat(data.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1,
                                      "max_bin=15", nullptr, &dataset),
            0);
  ASSERT_EQ(LGBM_DatasetSetField(dataset, "label", labels.data(),
                                 static_cast<int>(labels.size()), C_API_DTYPE_FLOAT32),
            0);

  BoosterHandle booster = nullptr;
  ASSERT_EQ(LGBM_BoosterCreate(dataset,
                               "objective=binary metric=auc num_leaves=3 verbose=-1",
                               &booster),
            0);
  for (int i = 0; i < 3; ++i) {
    int finished = 0;
    ASSERT_EQ(LGBM_BoosterUpdateOneIter(booster, &finished), 0);
  }

  // Predict using the deprecated Arrow API.
  auto schema = MakeFloatStructSchema(ncol);
  std::vector<std::vector<float>> columns = {
      {1.0f, 4.0f, 8.0f},
      {1.0f, 4.0f, 8.0f}};
  auto array = MakeFloatStructArray(schema.get(), columns);

  ArrowSchema raw_schema;
  schema.move(&raw_schema);
  std::vector<ArrowArray> raw_chunks(1);
  array.move(&raw_chunks[0]);

  const int n_predict_rows = static_cast<int>(columns[0].size());
  std::vector<double> arrow_out(n_predict_rows, 0.0);
  int64_t arrow_written = 0;
  ASSERT_EQ(LGBM_BoosterPredictForArrow(
                booster, static_cast<int64_t>(raw_chunks.size()), raw_chunks.data(),
                &raw_schema, C_API_PREDICT_NORMAL, 0, -1, "", &arrow_written,
                arrow_out.data()),
            0);
  ASSERT_EQ(arrow_written, n_predict_rows);

  // Compare against LGBM_BoosterPredictForMat with equivalent data.
  std::vector<double> mat_data = {1.0, 1.0,
                                  4.0, 4.0,
                                  8.0, 8.0};
  std::vector<double> mat_out(n_predict_rows, 0.0);
  int64_t mat_written = 0;
  ASSERT_EQ(LGBM_BoosterPredictForMat(booster, mat_data.data(), C_API_DTYPE_FLOAT64,
                                      n_predict_rows, ncol, 1, C_API_PREDICT_NORMAL, 0,
                                      -1, "", &mat_written, mat_out.data()),
            0);
  ASSERT_EQ(mat_written, n_predict_rows);
  for (int i = 0; i < n_predict_rows; ++i) {
    EXPECT_DOUBLE_EQ(arrow_out[i], mat_out[i]);
  }

  ASSERT_EQ(LGBM_BoosterFree(booster), 0);
  ASSERT_EQ(LGBM_DatasetFree(dataset), 0);
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
