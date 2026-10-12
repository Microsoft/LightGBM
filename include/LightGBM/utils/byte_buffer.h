/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_BYTE_BUFFER_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_BYTE_BUFFER_H_

#include <LightGBM/export.h>
#include <LightGBM/utils/binary_writer.h>

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace LightGBM {

/*!
  * \brief An implementation for serializing binary data to an auto-expanding memory buffer
  */
struct ByteBuffer final : public BinaryWriter {
  ByteBuffer() {}

  explicit ByteBuffer(size_t initial_size) {
    buffer_.reserve(initial_size);
  }

  size_t Write(const void* data, size_t bytes) {
    const char* mem_ptr = static_cast<const char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
      buffer_.push_back(mem_ptr[i]);
    }

    return bytes;
  }

  LIGHTGBM_EXPORT void Reserve(size_t capacity) {
    buffer_.reserve(capacity);
  }

  LIGHTGBM_EXPORT size_t GetSize() const {
    return buffer_.size();
  }

  LIGHTGBM_EXPORT char GetAt(size_t index) const {
    return buffer_.at(index);
  }

  LIGHTGBM_EXPORT char* Data() {
    return buffer_.data();
  }

 private:
  std::vector<char> buffer_;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_BYTE_BUFFER_H_
