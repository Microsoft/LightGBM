/*!
 * Copyright (c) 2018-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2018-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_FILE_IO_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_FILE_IO_H_

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
 * \brief An interface for writing files from buffers
 */
struct VirtualFileWriter : BinaryWriter {
  virtual ~VirtualFileWriter() {}

  /*!
   * \brief Initialize the writer
   * \return True when the file is available for writes
   */
  virtual bool Init() = 0;

  /*!
   * \brief Create appropriate writer for filename
   * \param filename Filename of the data
   * \return File writer instance
   */
  static std::unique_ptr<VirtualFileWriter> Make(const std::string& filename);

  /*!
   * \brief Check filename existence
   * \param filename Filename of the data
   * \return True when the file exists
   */
  static bool Exists(const std::string& filename);
};

/**
 * \brief An interface for reading files into buffers
 */
struct VirtualFileReader {
  /*!
   * \brief Constructor
   * \param filename Filename of the data
   */
  virtual ~VirtualFileReader() {}
  /*!
   * \brief Initialize the reader
   * \return True when the file is available for read
   */
  virtual bool Init() = 0;
  /*!
   * \brief Read data into buffer
   * \param buffer Buffer to read data into
   * \param bytes Number of bytes to read
   * \return Number of bytes read
   */
  virtual size_t Read(void* buffer, size_t bytes) const = 0;
  /*!
   * \brief Create appropriate reader for filename
   * \param filename Filename of the data
   * \return File reader instance
   */
  static std::unique_ptr<VirtualFileReader> Make(const std::string& filename);
};

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_FILE_IO_H_
