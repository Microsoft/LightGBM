/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <LightGBM/utils/byte_buffer.h>

#include <memory>
#include <random>

using LightGBM::ByteBuffer;


TEST(ByteBuffer, JustWorks) {
  std::unique_ptr<ByteBuffer> buffer;
  buffer.reset(new ByteBuffer());

  int cumulativeSize = 0;
  EXPECT_EQ(cumulativeSize, buffer->GetSize());

  int8_t int8Val = 34;
  cumulativeSize += sizeof(int8_t);
  buffer->Write(&int8Val, sizeof(int8_t));
  EXPECT_EQ(cumulativeSize, buffer->GetSize());
  EXPECT_EQ(int8Val, buffer->GetAt(cumulativeSize - 1));

  int16_t int16Val = 33;
  cumulativeSize += sizeof(int16_t);
  buffer->Write(&int16Val, sizeof(int16_t));
  EXPECT_EQ(cumulativeSize, buffer->GetSize());
  int16_t serializedInt16 = 0;
  char* int16Ptr = reinterpret_cast<char*>(&serializedInt16);
  for (unsigned int i = 0; i < sizeof(int16_t); i++) {
    int16Ptr[i] = buffer->GetAt(cumulativeSize - (sizeof(int16_t) - i));
  }
  EXPECT_EQ(int16Val, serializedInt16);

  int64_t int64Val = 35;
  cumulativeSize += sizeof(int64_t);
  buffer->Write(&int64Val, sizeof(int64_t));
  EXPECT_EQ(cumulativeSize, buffer->GetSize());
  int64_t serializedInt64 = 0;
  char* int64Ptr = reinterpret_cast<char*>(&serializedInt64);
  for (unsigned int i = 0; i < sizeof(int64_t); i++) {
    int64Ptr[i] = buffer->GetAt(cumulativeSize - (sizeof(int64_t) - i));
  }
  EXPECT_EQ(int64Val, serializedInt64);

  double doubleVal = 36.6;
  cumulativeSize += sizeof(double);
  buffer->Write(&doubleVal, sizeof(doubleVal));
  EXPECT_EQ(cumulativeSize, buffer->GetSize());
  double serializedDouble = 0;
  char* doublePtr = reinterpret_cast<char*>(&serializedDouble);
  for (unsigned int i = 0; i < sizeof(double); i++) {
    doublePtr[i] = buffer->GetAt(cumulativeSize - (sizeof(double) - i));
  }
  EXPECT_EQ(doubleVal, serializedDouble);

  const int charSize = 3;
  char charArrayVal[charSize] = { 'a', 'b', 'c' };
  cumulativeSize += charSize;
  buffer->Write(charArrayVal, charSize);
  EXPECT_EQ(cumulativeSize, buffer->GetSize());
  for (int i = 0; i < charSize; i++) {
    EXPECT_EQ(charArrayVal[i], buffer->GetAt(cumulativeSize - (charSize - i)));
  }

  // Test that Data() points to first value written
  EXPECT_EQ(int8Val, *buffer->Data());
}
