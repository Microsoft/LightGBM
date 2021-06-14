/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_PARSER_BASE_H_
#define LIGHTGBM_PARSER_BASE_H_

#include <string>
#include <utility>
#include <vector>

namespace LightGBM {
/*! \brief Interface for Parser */
class Parser {
 public:
  typedef const char* (*AtofFunc)(const char* p, double* out);

  /*! \brief virtual destructor */
  virtual ~Parser() {}

  /*!
  * \brief Parse one line with label
  * \param str One line record, string format, should end with '\0'
  * \param out_features Output columns, store in (column_idx, values)
  * \param out_label Label will store to this if exists
  * \param line_idx the line index of current line
  */
  virtual void ParseOneLine(const char* str,
                            std::vector<std::pair<int, double>>* out_features,
                            double* out_label, const int line_idx = -1) const = 0;

  virtual int NumFeatures() const = 0;

  /*!
  * \brief Create an object of parser, will auto choose the format depend on file
  * \param filename One Filename of data
  * \param header Whether the data file has header
  * \param num_features Pass num_features of this data file if you know, <=0 means don't know
  * \param label_idx index of label column
  * \param precise_float_parser using precise floating point number parsing if true
  * \return Object of parser
  */
  static Parser* CreateParser(const char* filename, bool header, int num_features, int label_idx, bool precise_float_parser);

  /*! \brief Binary file token */
  static const char* binary_file_token;

  /*! \brief Check can load from binary file */
  static std::string CheckCanLoadFromBin(const char* filename);
};

// Row iterator of one column for CSC matrix
class CSC_RowIterator {
 public:
  CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx);
  virtual ~CSC_RowIterator() {}
  // return value at idx, only can access by ascent order
  virtual double Get(int idx);
  // return next non-zero pair, if index < 0, means no more data
  virtual std::pair<int, double> NextNonZero();

  virtual void Reset() {
    nonzero_idx_ = 0;
    cur_idx_ = -1;
    cur_val_ = 0.0f;
    is_end_ = false;
  }

 public:
  int nonzero_idx_ = 0;
  int cur_idx_ = -1;
  double cur_val_ = 0.0f;
  bool is_end_ = false;
  std::function<std::pair<int, double>(int idx)> iter_fun_;
};

std::function<std::pair<int, double>(int idx)>
IterateFunctionFromCSC(const void* col_ptr, int col_ptr_type, const int32_t* indices,
  const void* data, int data_type, int64_t ncol_ptr, int64_t , int col_idx);

}  // namespace LightGBM

#endif  // LightGBM_PARSER_BASE_H_
