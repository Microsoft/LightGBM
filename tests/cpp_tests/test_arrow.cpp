/*!
 * Copyright (c) 2023-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2023-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Oliver Borchert
 */

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include <nanoarrow/nanoarrow.hpp>

#include "../../src/arrow/array.hpp"

using LightGBM::ArrowChunkedArray;

namespace {

// Build an ArrowArrayStream from a schema and a list of chunk arrays. Takes ownership of the
// passed schema and chunks.
nanoarrow::UniqueArrayStream MakeStream(nanoarrow::UniqueSchema schema,
                                        std::vector<nanoarrow::UniqueArray> chunks) {
  nanoarrow::UniqueArrayStream stream;
  nanoarrow::VectorArrayStream(schema.get(), std::move(chunks)).ToArrayStream(stream.get());
  return stream;
}

nanoarrow::UniqueSchema MakePrimitiveSchema(ArrowType type) {
  nanoarrow::UniqueSchema schema;
  EXPECT_EQ(ArrowSchemaInitFromType(schema.get(), type), NANOARROW_OK);
  return schema;
}

nanoarrow::UniqueSchema MakeStructSchema(const std::vector<ArrowType>& field_types) {
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  EXPECT_EQ(ArrowSchemaSetTypeStruct(schema.get(), field_types.size()), NANOARROW_OK);
  for (size_t i = 0; i < field_types.size(); ++i) {
    EXPECT_EQ(ArrowSchemaSetType(schema->children[i], field_types[i]), NANOARROW_OK);
  }
  return schema;
}

template <typename T>
nanoarrow::UniqueArray MakePrimitiveArray(ArrowType type, const std::vector<T>& values,
                                          const std::vector<int64_t>& null_indices = {},
                                          int64_t offset = 0) {
  nanoarrow::UniqueArray array;
  EXPECT_EQ(ArrowArrayInitFromType(array.get(), type), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  size_t null_idx_pos = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    if (null_idx_pos < null_indices.size() &&
        null_indices[null_idx_pos] == static_cast<int64_t>(i)) {
      EXPECT_EQ(ArrowArrayAppendNull(array.get(), 1), NANOARROW_OK);
      ++null_idx_pos;
    } else {
      if (type == NANOARROW_TYPE_BOOL) {
        EXPECT_EQ(ArrowArrayAppendInt(array.get(), values[i] ? 1 : 0), NANOARROW_OK);
      } else {
        EXPECT_EQ(ArrowArrayAppendDouble(array.get(), static_cast<double>(values[i])),
                  NANOARROW_OK);
      }
    }
  }
  EXPECT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);

  // Apply slicing offset (tests the consumer's handling of `array->offset`).
  if (offset > 0) {
    array->offset += offset;
    array->length -= offset;
  }
  return array;
}

}  // namespace

TEST(ArrowChunkedArrayTest, GetLength) {
  // Single chunk
  {
    auto schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
    std::vector<nanoarrow::UniqueArray> chunks;
    chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {1, 2}));
    ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());
    ASSERT_EQ(chunked_array.get_length(), 2);
  }

  // Multiple chunks
  {
    auto schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
    std::vector<nanoarrow::UniqueArray> chunks;
    chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {1, 2}));
    chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {3, 4, 5, 6}));
    ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());
    ASSERT_EQ(chunked_array.get_length(), 6);
  }

  // Sliced chunk via offset
  {
    auto schema = MakePrimitiveSchema(NANOARROW_TYPE_BOOL);
    std::vector<nanoarrow::UniqueArray> chunks;
    chunks.emplace_back(
        MakePrimitiveArray<bool>(NANOARROW_TYPE_BOOL, {true, false, true, true}, {}, 1));
    ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());
    ASSERT_EQ(chunked_array.get_length(), 3);
  }
}

TEST(ArrowChunkedArrayTest, GetFields) {
  auto schema = MakeStructSchema({NANOARROW_TYPE_FLOAT, NANOARROW_TYPE_FLOAT});

  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  std::vector<float> dat1 = {1, 2, 3};
  std::vector<float> dat2 = {4, 5, 6};
  for (size_t i = 0; i < dat1.size(); ++i) {
    ASSERT_EQ(ArrowArrayAppendDouble(array->children[0], dat1[i]), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayAppendDouble(array->children[1], dat2[i]), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayFinishElement(array.get()), NANOARROW_OK);
  }
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);

  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(std::move(array));
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  ASSERT_EQ(chunked_array.get_length(), 3);
  ASSERT_EQ(chunked_array.get_num_fields(), 2);

  int32_t first0 = 0, first1 = 0;
  chunked_array.view().field(0).visit<int32_t>([&](auto v) { first0 = *v.begin(); });
  chunked_array.view().field(1).visit<int32_t>([&](auto v) { first1 = *v.begin(); });
  ASSERT_EQ(first0, 1);
  ASSERT_EQ(first1, 4);
}

TEST(ArrowChunkedArrayTest, IteratorArithmetic) {
  auto schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {1, 2}));
  chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {3, 4, 5, 6}));
  chunks.emplace_back(MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {7}));
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  chunked_array.view().visit<int32_t>([](auto v) {
    auto it = v.begin();
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 2);
    ++it;
    EXPECT_EQ(*it, 3);
    it += 2;
    EXPECT_EQ(*it, 5);
    it += 2;
    EXPECT_EQ(*it, 7);

    auto begin = v.begin();
    EXPECT_EQ(begin[0], 1);
    EXPECT_EQ(begin[1], 2);
    EXPECT_EQ(begin[2], 3);
    EXPECT_EQ(begin[6], 7);

    auto end = v.end();
    EXPECT_EQ(end - it, 1);
    EXPECT_EQ(end - v.begin(), 7);
  });
}

TEST(ArrowChunkedArrayTest, BooleanIterator) {
  auto schema = MakePrimitiveSchema(NANOARROW_TYPE_BOOL);
  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(MakePrimitiveArray<bool>(NANOARROW_TYPE_BOOL, {false, true, false}, {2}));
  chunks.emplace_back(MakePrimitiveArray<bool>(
      NANOARROW_TYPE_BOOL, {false, false, false, false, true, true, true, true, false, true}, {},
      1));
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  chunked_array.view().visit<float>([](auto v) {
    auto it = v.begin();
    // First chunk
    EXPECT_EQ(*it, 0);
    EXPECT_EQ(*(++it), 1);
    EXPECT_TRUE(std::isnan(*(++it)));

    // Second chunk
    EXPECT_EQ(*(++it), 0);
    it += 3;
    EXPECT_EQ(*it, 1);
    it += 4;
    EXPECT_EQ(*it, 0);
    EXPECT_EQ(*(++it), 1);

    EXPECT_EQ(++it, v.end());
  });
}

TEST(ArrowChunkedArrayTest, OffsetAndValidity) {
  auto schema = MakePrimitiveSchema(NANOARROW_TYPE_FLOAT);
  std::vector<nanoarrow::UniqueArray> chunks;
  chunks.emplace_back(
      MakePrimitiveArray<float>(NANOARROW_TYPE_FLOAT, {0, 1, 2, 3, 4, 5, 6}, {2, 3}, 2));
  ArrowChunkedArray chunked_array(MakeStream(std::move(schema), std::move(chunks)).get());

  chunked_array.view().visit<double>([](auto v) {
    auto it = v.begin();
    EXPECT_TRUE(std::isnan(*it));
    EXPECT_TRUE(std::isnan(*(++it)));
    EXPECT_EQ(it[2], 4);
    EXPECT_EQ(it[4], 6);
  });
}
