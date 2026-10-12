/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_BINARY_WRITER_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_BINARY_WRITER_H_

#include <cstdlib>
#include <vector>

namespace LightGBM {

/*!
  * \brief An interface for serializing binary data to a buffer
  */
struct BinaryWriter {
  /*!
    * \brief Append data to this binary target
    * \param data Buffer to write from
    * \param bytes Number of bytes to write from buffer
    * \return Number of bytes written
    */
  virtual size_t Write(const void* data, size_t bytes) = 0;

  /*!
    * \brief Append data to this binary target aligned on a given byte size boundary
    * \param data Buffer to write from
    * \param bytes Number of bytes to write from buffer
    * \param alignment The size of bytes to align to in whole increments
    * \return Number of bytes written
    */
  size_t AlignedWrite(const void* data, size_t bytes, size_t alignment = 8) {
    auto ret = Write(data, bytes);
    if (bytes % alignment != 0) {
      size_t padding = AlignedSize(bytes, alignment) - bytes;
      std::vector<char> tmp(padding, 0);
      ret += Write(tmp.data(), padding);
    }
    return ret;
  }

  /*!
    * \brief The aligned size of a buffer length.
    * \param bytes The number of bytes in a buffer
    * \param alignment The size of bytes to align to in whole increments
    * \return Number of aligned bytes
    */
  static size_t AlignedSize(size_t bytes, size_t alignment = 8) {
    if (bytes % alignment == 0) {
      return bytes;
    } else {
      return bytes / alignment * alignment + alignment;
    }
  }
};
}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_BINARY_WRITER_H_
