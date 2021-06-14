/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_TEXT_READER_H_
#define LIGHTGBM_UTILS_TEXT_READER_H_

#include <LightGBM/utils/log.h>
#include <LightGBM/utils/pipeline_reader.h>
#include <LightGBM/utils/random.h>

#include <string>
#include <cstdio>
#include <functional>
#include <sstream>
#include <vector>

namespace LightGBM {

const size_t kGbs = size_t(1024) * 1024 * 1024;

/*!
* \brief Read text data from file
*/
template<typename INDEX_T>
class TextReader {
 public:
  /*!
  * \brief Constructor
  * \param filename Filename of data
  * \param is_skip_first_line True if need to skip header
  */
  TextReader(const char* filename, bool is_skip_first_line, size_t progress_interval_bytes = SIZE_MAX):
    filename_(filename), is_skip_first_line_(is_skip_first_line), read_progress_interval_bytes_(progress_interval_bytes) {
    if (is_skip_first_line_) {
      auto reader = VirtualFileReader::Make(filename);
      if (!reader->Init()) {
        Log::Fatal("Could not open %s", filename);
      }
      std::stringstream str_buf;
      char read_c;
      size_t nread = reader->Read(&read_c, 1);
      while (nread == 1) {
        if (read_c == '\n' || read_c == '\r') {
          break;
        }
        str_buf << read_c;
        ++skip_bytes_;
        nread = reader->Read(&read_c, 1);
      }
      if (read_c == '\r') {
        reader->Read(&read_c, 1);
        ++skip_bytes_;
      }
      if (read_c == '\n') {
        reader->Read(&read_c, 1);
        ++skip_bytes_;
      }
      first_line_ = str_buf.str();
      Log::Debug("Skipped header \"%s\" in file %s", first_line_.c_str(), filename_);
    }
  }
  /*!
  * \brief Destructor
  */
  ~TextReader() {
    Clear();
  }
  /*!
  * \brief Clear cached data
  */
  inline void Clear() {
    lines_.clear();
    lines_.shrink_to_fit();
  }
  /*!
  * \brief return first line of data
  */
  inline std::string first_line() {
    return first_line_;
  }
  /*!
  * \brief Get text data that read from file
  * \return Text data, store in std::vector by line
  */
  inline std::vector<std::string>& Lines() { return lines_; }

  INDEX_T ReadAllAndProcess(const std::function<void(INDEX_T, const char*, size_t)>& process_fun) {
    last_line_ = "";
    INDEX_T total_cnt = 0;
    size_t bytes_read = 0;
    PipelineReader::Read(filename_, skip_bytes_,
        [&process_fun, &bytes_read, &total_cnt, this]
    (const char* buffer_process, size_t read_cnt) {
      size_t cnt = 0;
      size_t i = 0;
      size_t last_i = 0;
      // skip the break between \r and \n
      if (last_line_.size() == 0 && buffer_process[0] == '\n') {
        i = 1;
        last_i = i;
      }
      while (i < read_cnt) {
        if (buffer_process[i] == '\n' || buffer_process[i] == '\r') {
          if (last_line_.size() > 0) {
            last_line_.append(buffer_process + last_i, i - last_i);
            process_fun(total_cnt, last_line_.c_str(), last_line_.size());
            last_line_ = "";
          } else {
            process_fun(total_cnt, buffer_process + last_i, i - last_i);
          }
          ++cnt;
          ++i;
          ++total_cnt;
          // skip end of line
          while ((buffer_process[i] == '\n' || buffer_process[i] == '\r') && i < read_cnt) { ++i; }
          last_i = i;
        } else {
          ++i;
        }
      }
      if (last_i != read_cnt) {
        last_line_.append(buffer_process + last_i, read_cnt - last_i);
      }

      size_t prev_bytes_read = bytes_read;
      bytes_read += read_cnt;
      if (prev_bytes_read / read_progress_interval_bytes_ < bytes_read / read_progress_interval_bytes_) {
        Log::Debug("Read %.1f GBs from %s.", 1.0 * bytes_read / kGbs, filename_);
      }

      return cnt;
    });
    // if last line of file doesn't contain end of line
    if (last_line_.size() > 0) {
      Log::Info("Warning: last line of %s has no end of line, still using this line", filename_);
      process_fun(total_cnt, last_line_.c_str(), last_line_.size());
      ++total_cnt;
      last_line_ = "";
    }
    return total_cnt;
  }

  /*!
  * \brief Read all text data from file in memory
  * \return number of lines of text data
  */
  INDEX_T ReadAllLines() {
    return ReadAllAndProcess(
      [=](INDEX_T, const char* buffer, size_t size) {
      lines_.emplace_back(buffer, size);
    });
  }

  /*!
  * \brief Read all text data from file in memory with accumulating category encoding statistics
  * \return number of lines of text data
  */
  INDEX_T ReadAllLines(
      const std::function<void(const char*, size_t, INDEX_T)>& category_encoding_acc_func) {
    return ReadAllAndProcess(
      [=](INDEX_T row_idx, const char* buffer, size_t size) {
      category_encoding_acc_func(buffer, size, row_idx);
      lines_.emplace_back(buffer, size);
    });
  }

  std::vector<char> ReadContent(size_t* out_len) {
    std::vector<char> ret;
    *out_len = 0;
    auto reader = VirtualFileReader::Make(filename_);
    if (!reader->Init()) {
      return ret;
    }
    const size_t buffer_size = 16 * 1024 * 1024;
    auto buffer_read = std::vector<char>(buffer_size);
    size_t read_cnt = 0;
    do {
      read_cnt = reader->Read(buffer_read.data(), buffer_size);
      ret.insert(ret.end(), buffer_read.begin(), buffer_read.begin() + read_cnt);
      *out_len += read_cnt;
    } while (read_cnt > 0);
    return ret;
  }

  template <bool USE_CATEGORY_ENCODING>
  INDEX_T SampleFromFile(Random* random, INDEX_T sample_cnt, std::vector<std::string>* out_sampled_data,
    std::vector<INDEX_T>* sampled_indices,
    std::function<void(const char*, size_t, INDEX_T)> category_encoding_acc_func) {
    INDEX_T cur_sample_cnt = 0;
    if (USE_CATEGORY_ENCODING) {
      sampled_indices->clear();
    }
    return ReadAllAndProcess([=, &random, &cur_sample_cnt, &out_sampled_data]
    (INDEX_T line_idx, const char* buffer, size_t size) {
      if (USE_CATEGORY_ENCODING) {
        category_encoding_acc_func(buffer, size, line_idx);
      }
      if (cur_sample_cnt < sample_cnt) {
        out_sampled_data->emplace_back(buffer, size);
        if (USE_CATEGORY_ENCODING) {
          sampled_indices->push_back(line_idx);
        }
        ++cur_sample_cnt;
      } else {
        const size_t idx = static_cast<size_t>(random->NextInt(0, static_cast<int>(line_idx + 1)));
        if (idx < static_cast<size_t>(sample_cnt)) {
          out_sampled_data->operator[](idx) = std::string(buffer, size);
          if (USE_CATEGORY_ENCODING) {
            sampled_indices->operator[](idx) = line_idx;
          }
        }
      }
    });
  }

  /*!
  * \brief Read part of text data from file in memory, use filter_fun to filter data
  * \param filter_fun Function that perform data filter
  * \param out_used_data_indices Store line indices that read text data
  * \return The number of total data
  */
  INDEX_T ReadAndFilterLines(const std::function<bool(INDEX_T, const char*, size_t)>& filter_fun, std::vector<INDEX_T>* out_used_data_indices) {
    out_used_data_indices->clear();
    INDEX_T total_cnt = ReadAllAndProcess(
        [&filter_fun, &out_used_data_indices, this]
    (INDEX_T line_idx , const char* buffer, size_t size) {
      bool is_used = filter_fun(line_idx, buffer, size);
      if (is_used) { out_used_data_indices->push_back(line_idx); }
      if (is_used) { lines_.emplace_back(buffer, size); }
    });
    return total_cnt;
  }

  template <bool GET_SAMPLED_INDICES>
  INDEX_T SampleAndFilterFromFile(const std::function<bool(INDEX_T, const char*, size_t)>& filter_fun,
    std::vector<INDEX_T>* out_used_data_indices, Random* random, INDEX_T sample_cnt,
    std::vector<std::string>* out_sampled_data, std::vector<INDEX_T>* sampled_indices) {
    INDEX_T cur_sample_cnt = 0;
    out_used_data_indices->clear();
    INDEX_T total_cnt = ReadAllAndProcess(
        [=, &filter_fun, &out_used_data_indices, &random, &cur_sample_cnt,
         &out_sampled_data]
    (INDEX_T line_idx, const char* buffer, size_t size) {
      bool is_used = filter_fun(line_idx, buffer, size);
      if (GET_SAMPLED_INDICES) {
        sampled_indices->clear();
      }
      if (is_used) { out_used_data_indices->push_back(line_idx); }
      if (is_used) {
        if (cur_sample_cnt < sample_cnt) {
          out_sampled_data->emplace_back(buffer, size);
          if (GET_SAMPLED_INDICES) {
            sampled_indices->emplace_back(line_idx);
          }
          ++cur_sample_cnt;
        } else {
          const size_t idx = static_cast<size_t>(random->NextInt(0, static_cast<int>(out_used_data_indices->size())));
          if (idx < static_cast<size_t>(sample_cnt)) {
            out_sampled_data->operator[](idx) = std::string(buffer, size);
            if (GET_SAMPLED_INDICES) {
              sampled_indices->operator[](idx) = line_idx;
            }
          }
        }
      }
    });
    return total_cnt;
  }

  INDEX_T CountLine() {
    return ReadAllAndProcess(
      [=](INDEX_T, const char*, size_t) {
    });
  }

  INDEX_T ReadAllAndProcessParallelWithFilter(const std::function<void(INDEX_T, const std::vector<std::string>&)>& process_fun, const std::function<bool(INDEX_T, INDEX_T)>& filter_fun) {
    last_line_ = "";
    INDEX_T total_cnt = 0;
    size_t bytes_read = 0;
    INDEX_T used_cnt = 0;
    PipelineReader::Read(filename_, skip_bytes_,
        [&process_fun, &filter_fun, &total_cnt, &bytes_read, &used_cnt, this]
    (const char* buffer_process, size_t read_cnt) {
      size_t cnt = 0;
      size_t i = 0;
      size_t last_i = 0;
      INDEX_T start_idx = used_cnt;
      // skip the break between \r and \n
      if (last_line_.size() == 0 && buffer_process[0] == '\n') {
        i = 1;
        last_i = i;
      }
      while (i < read_cnt) {
        if (buffer_process[i] == '\n' || buffer_process[i] == '\r') {
          if (last_line_.size() > 0) {
            last_line_.append(buffer_process + last_i, i - last_i);
            if (filter_fun(used_cnt, total_cnt)) {
              lines_.push_back(last_line_);
              ++used_cnt;
            }
            last_line_ = "";
          } else {
            if (filter_fun(used_cnt, total_cnt)) {
              lines_.emplace_back(buffer_process + last_i, i - last_i);
              ++used_cnt;
            }
          }
          ++cnt;
          ++i;
          ++total_cnt;
          // skip end of line
          while ((buffer_process[i] == '\n' || buffer_process[i] == '\r') && i < read_cnt) { ++i; }
          last_i = i;
        } else {
          ++i;
        }
      }
      process_fun(start_idx, lines_);
      lines_.clear();
      if (last_i != read_cnt) {
        last_line_.append(buffer_process + last_i, read_cnt - last_i);
      }

      size_t prev_bytes_read = bytes_read;
      bytes_read += read_cnt;
      if (prev_bytes_read / read_progress_interval_bytes_ < bytes_read / read_progress_interval_bytes_) {
        Log::Debug("Read %.1f GBs from %s.", 1.0 * bytes_read / kGbs, filename_);
      }

      return cnt;
    });
    // if last line of file doesn't contain end of line
    if (last_line_.size() > 0) {
      Log::Info("Warning: last line of %s has no end of line, still using this line", filename_);
      if (filter_fun(used_cnt, total_cnt)) {
        lines_.push_back(last_line_);
        process_fun(used_cnt, lines_);
      }
      lines_.clear();
      ++total_cnt;
      ++used_cnt;
      last_line_ = "";
    }
    return total_cnt;
  }

  INDEX_T ReadAllAndProcessParallel(const std::function<void(INDEX_T, const std::vector<std::string>&)>& process_fun) {
    return ReadAllAndProcessParallelWithFilter(process_fun, [](INDEX_T, INDEX_T) { return true; });
  }

  INDEX_T ReadPartAndProcessParallel(const std::vector<INDEX_T>& used_data_indices, const std::function<void(INDEX_T, const std::vector<std::string>&)>& process_fun) {
    return ReadAllAndProcessParallelWithFilter(process_fun,
      [&used_data_indices](INDEX_T used_cnt, INDEX_T total_cnt) {
      if (static_cast<size_t>(used_cnt) < used_data_indices.size() && total_cnt == used_data_indices[used_cnt]) {
        return true;
      } else {
        return false;
      }
    });
  }

 private:
  /*! \brief Filename of text data */
  const char* filename_;
  /*! \brief Cache the read text data */
  std::vector<std::string> lines_;
  /*! \brief Buffer for last line */
  std::string last_line_;
  /*! \brief first line */
  std::string first_line_ = "";
  /*! \brief is skip first line */
  bool is_skip_first_line_ = false;
  size_t read_progress_interval_bytes_;
  /*! \brief is skip first line */
  int skip_bytes_ = 0;
};

}  // namespace LightGBM

#endif   // LightGBM_UTILS_TEXT_READER_H_
