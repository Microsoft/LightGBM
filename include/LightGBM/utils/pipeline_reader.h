/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_PIPELINE_READER_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_PIPELINE_READER_H_

#include <LightGBM/utils/file_io.h>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace LightGBM {

/*!
* \brief A pipeline file reader, use 2 threads, one read block from file, the other process the block
*/
class PipelineReader {
 public:
  /*!
  * \brief Read data from a file, use pipeline methods
  * \param filename Filename of data
  * \process_fun Process function
  */
  static size_t Read(const char* filename, int skip_bytes, const std::function<size_t(const char*, size_t)>& process_fun) {
    auto reader = VirtualFileReader::Make(filename);
    if (!reader->Init()) {
      return 0;
    }
    size_t cnt = 0;
    const size_t buffer_size =  16 * 1024 * 1024;
    // buffer used for the process_fun
    auto buffer_process = std::vector<char>(buffer_size);
    // buffer used for the file reading
    auto buffer_read = std::vector<char>(buffer_size);
    size_t read_cnt = 0;
    if (skip_bytes > 0) {
      // skip first k bytes
      read_cnt = reader->Read(buffer_process.data(), skip_bytes);
    }
    // read first block
    read_cnt = reader->Read(buffer_process.data(), buffer_size);

    size_t last_read_cnt = 0;
    while (read_cnt > 0) {
      // start read thread
      std::thread read_worker = std::thread(
        [=, &last_read_cnt, &reader, &buffer_read] {
        last_read_cnt = reader->Read(buffer_read.data(), buffer_size);
      });
      // start process
      cnt += process_fun(buffer_process.data(), read_cnt);
      // wait for read thread
      read_worker.join();
      // exchange the buffer
      std::swap(buffer_process, buffer_read);
      read_cnt = last_read_cnt;
    }
    return cnt;
  }
};

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_PIPELINE_READER_H_
