/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_BIN_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_BIN_H_

#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/file_io.h>

#include <cstdint>
#include <limits>
#include <string>
#include <functional>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace LightGBM {

enum BinType {
  NumericalBin,
  CategoricalBin
};

enum MissingType {
  None,
  Zero,
  NaN
};

typedef double hist_t;
typedef int32_t int_hist_t;
typedef uint64_t hist_cnt_t;
// check at compile time
static_assert(sizeof(hist_t) == sizeof(hist_cnt_t), "Histogram entry size is not correct");

const size_t kHistEntrySize = 2 * sizeof(hist_t);
const size_t kInt32HistEntrySize = 2 * sizeof(int_hist_t);
const size_t kInt16HistEntrySize = 2 * sizeof(int16_t);
const int kHistOffset = 2;
const double kSparseThreshold = 0.7;

#define GET_GRAD(hist, i) hist[(i) << 1]
#define GET_HESS(hist, i) hist[((i) << 1) + 1]

inline static void HistogramSumReducer(const char* src, char* dst, int type_size, comm_size_t len) {
  comm_size_t used_size = 0;
  const hist_t* p1;
  hist_t* p2;
  while (used_size < len) {
    // convert
    p1 = reinterpret_cast<const hist_t*>(src);
    p2 = reinterpret_cast<hist_t*>(dst);
    *p2 += *p1;
    src += type_size;
    dst += type_size;
    used_size += type_size;
  }
}

inline static void Int32HistogramSumReducer(const char* src, char* dst, int type_size, comm_size_t len) {
  const int64_t* src_ptr = reinterpret_cast<const int64_t*>(src);
  int64_t* dst_ptr = reinterpret_cast<int64_t*>(dst);
  const comm_size_t steps = (len + (type_size * 2) - 1) / (type_size * 2);
  #pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS())
  for (comm_size_t i = 0; i < steps; ++i) {
    dst_ptr[i] += src_ptr[i];
  }
}

inline static void Int16HistogramSumReducer(const char* src, char* dst, int type_size, comm_size_t len) {
  const int32_t* src_ptr = reinterpret_cast<const int32_t*>(src);
  int32_t* dst_ptr = reinterpret_cast<int32_t*>(dst);
  const comm_size_t steps = (len + (type_size * 2) - 1) / (type_size * 2);
  #pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS())
  for (comm_size_t i = 0; i < steps; ++i) {
    dst_ptr[i] += src_ptr[i];
  }
}

/*! \brief This class used to convert feature values into bin,
*          and store some meta information for bin*/
class BinMapper {
 public:
  BinMapper();
  BinMapper(const BinMapper& other);
  explicit BinMapper(const void* memory);
  ~BinMapper();

  bool CheckAlign(const BinMapper& other) const {
    if (num_bin_ != other.num_bin_) {
      return false;
    }
    if (missing_type_ != other.missing_type_) {
      return false;
    }
    if (bin_type_ == BinType::NumericalBin) {
      for (int i = 0; i < num_bin_; ++i) {
        if (bin_upper_bound_[i] != other.bin_upper_bound_[i]) {
          return false;
        }
      }
    } else {
      for (int i = 0; i < num_bin_; i++) {
        if (bin_2_categorical_[i] != other.bin_2_categorical_[i]) {
          return false;
        }
      }
    }
    return true;
  }

  /*! \brief Get number of bins */
  inline int num_bin() const { return num_bin_; }

  /*! \brief Missing Type */
  inline MissingType missing_type() const { return missing_type_; }

  /*! \brief True if bin is trivial (contains only one bin) */
  inline bool is_trivial() const { return is_trivial_; }

  /*! \brief Sparsity of this bin ( num_zero_bins / num_data ) */
  inline double sparse_rate() const { return sparse_rate_; }

  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  void SaveBinaryToFile(BinaryWriter* writer) const;

  /*!
  * \brief Mapping bin into feature value
  * \param bin
  * \return Feature value of this bin
  */
  inline double BinToValue(uint32_t bin) const {
    if (bin_type_ == BinType::NumericalBin) {
      return bin_upper_bound_[bin];
    } else {
      return bin_2_categorical_[bin];
    }
  }

  /*!
  * \brief Maximum categorical value
  * \return Maximum categorical value for categorical features, 0 for numerical features
  */
  inline int MaxCatValue() const {
    if (bin_2_categorical_.size() == 0) {
      return 0;
    }
    int max_cat_value = bin_2_categorical_[0];
    for (size_t i = 1; i < bin_2_categorical_.size(); ++i) {
      if (bin_2_categorical_[i] > max_cat_value) {
        max_cat_value = bin_2_categorical_[i];
      }
    }
    return max_cat_value;
  }

  /*!
  * \brief Get sizes in byte of this object
  */
  size_t SizesInByte() const;

  /*!
  * \brief Mapping feature value into bin
  * \param value
  * \return bin for this feature value
  */
  inline uint32_t ValueToBin(double value) const;

  /*!
  * \brief Get the default bin when value is 0
  * \return default bin
  */
  inline uint32_t GetDefaultBin() const {
    return default_bin_;
  }

  inline uint32_t GetMostFreqBin() const {
    return most_freq_bin_;
  }

  /*!
  * \brief Construct feature value to bin mapper according feature values
  * \param values (Sampled) values of this feature, Note: not include zero.
  * \param num_values number of values.
  * \param total_sample_cnt number of total sample count, equal with values.size() + num_zeros
  * \param max_bin The maximal number of bin
  * \param min_data_in_bin min number of data in one bin
  * \param min_split_data
  * \param pre_filter
  * \param bin_type Type of this bin
  * \param use_missing True to enable missing value handle
  * \param zero_as_missing True to use zero as missing value
  * \param forced_upper_bounds Vector of split points that must be used (if this has size less than max_bin, remaining splits are found by the algorithm)
  */
  void FindBin(double* values, int num_values, size_t total_sample_cnt, int max_bin, int min_data_in_bin, int min_split_data, bool pre_filter, BinType bin_type,
               bool use_missing, bool zero_as_missing, const std::vector<double>& forced_upper_bounds);

  /*!
  * \brief Serializing this object to buffer
  * \param buffer The destination
  */
  void CopyTo(char* buffer) const;

  /*!
  * \brief Deserializing this object from buffer
  * \param buffer The source
  */
  void CopyFrom(const char* buffer);

  /*!
  * \brief Get bin types
  */
  inline BinType bin_type() const { return bin_type_; }

  /*!
  * \brief Get bin info
  */
  inline std::string bin_info_string() const {
    if (bin_type_ == BinType::CategoricalBin) {
      return Common::Join(bin_2_categorical_, ":");
    } else {
      std::stringstream str_buf;
      str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
      str_buf << '[' << min_val_ << ':' << max_val_ << ']';
      return str_buf.str();
    }
  }

 private:
  /*! \brief Number of bins */
  int num_bin_;
  MissingType missing_type_;
  /*! \brief Store upper bound for each bin */
  std::vector<double> bin_upper_bound_;
  /*! \brief True if this feature is trivial */
  bool is_trivial_;
  /*! \brief Sparse rate of this bins( num_bin0/num_data ) */
  double sparse_rate_;
  /*! \brief Type of this bin */
  BinType bin_type_;
  /*! \brief Mapper from categorical to bin */
  std::unordered_map<int, unsigned int> categorical_2_bin_;
  /*! \brief Mapper from bin to categorical */
  std::vector<int> bin_2_categorical_;
  /*! \brief minimal feature value */
  double min_val_;
  /*! \brief maximum feature value */
  double max_val_;
  /*! \brief bin value of feature value 0 */
  uint32_t default_bin_;

  uint32_t most_freq_bin_;
};

/*! \brief Iterator for one bin column */
class BinIterator {
 public:
  /*!
  * \brief Get bin data on specific row index
  * \param idx Index of this data
  * \return Bin data
  */
  virtual uint32_t Get(data_size_t idx) = 0;
  virtual uint32_t RawGet(data_size_t idx) = 0;
  virtual void Reset(data_size_t idx) = 0;
  virtual ~BinIterator() = default;
};

/*!
* \brief Interface for bin data. This class will store bin data for one feature.
*        unlike OrderedBin, this class will store data by original order.
*        Note that it may cause cache misses when construct histogram,
*        but it doesn't need to re-order operation, So it will be faster than OrderedBin for dense feature
*/
class Bin {
 public:
  /*! \brief virtual destructor */
  virtual ~Bin() {}
  /*!
  * \brief Initialize for pushing.  By default, no action needed.
  * \param num_thread The number of external threads that will be calling the push APIs
  * \param omp_max_threads The maximum number of OpenMP threads to allocate for
  */
  virtual void InitStreaming(uint32_t /*num_thread*/, int32_t /*omp_max_threads*/) { }
  /*!
  * \brief Push one record
  * \param tid Thread id
  * \param idx Index of record
  * \param value bin value of record
  */
  virtual void Push(int tid, data_size_t idx, uint32_t value) = 0;

  virtual void CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) = 0;
  /*!
  * \brief Get bin iterator of this bin for specific feature
  * \param min_bin min_bin of current used feature
  * \param max_bin max_bin of current used feature
  * \param most_freq_bin
  * \return Iterator of this bin
  */
  virtual BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const = 0;

  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  virtual void SaveBinaryToFile(BinaryWriter* writer) const = 0;

  /*!
  * \brief Load from memory
  * \param memory
  * \param local_used_indices
  */
  virtual void LoadFromMemory(const void* memory,
    const std::vector<data_size_t>& local_used_indices) = 0;

  /*!
  * \brief Get sizes in byte of this object
  */
  virtual size_t SizesInByte() const = 0;

  /*! \brief Number of all data */
  virtual data_size_t num_data() const = 0;

  /*! \brief Get data pointer */
  virtual void* get_data() = 0;

  virtual void ReSize(data_size_t num_data) = 0;

  /*!
  * \brief Construct histogram of this feature,
  *        Note: We use ordered_gradients and ordered_hessians to improve cache hit chance
  *        The naive solution is using gradients[data_indices[i]] for data_indices[i] to get gradients,
           which is not cache friendly, since the access of memory is not continuous.
  *        ordered_gradients and ordered_hessians are preprocessed, and they are re-ordered by data_indices.
  *        Ordered_gradients[i] is aligned with data_indices[i]'s gradients (same for ordered_hessians).
  * \param data_indices Used data indices in current leaf
  * \param start start index in data_indices
  * \param end end index in data_indices
  * \param ordered_gradients Pointer to gradients, the data_indices[i]-th data's gradient is ordered_gradients[i]
  * \param ordered_hessians Pointer to hessians, the data_indices[i]-th data's hessian is ordered_hessians[i]
  * \param out Output Result
  */
  virtual void ConstructHistogram(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const = 0;

  virtual void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const = 0;

  virtual void ConstructHistogramInt8(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const = 0;

  virtual void ConstructHistogramInt8(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const = 0;

  virtual void ConstructHistogramInt16(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const = 0;

  virtual void ConstructHistogramInt16(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const = 0;

  virtual void ConstructHistogramInt32(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const = 0;

  virtual void ConstructHistogramInt32(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const = 0;

  /*!
  * \brief Construct histogram of this feature,
  *        Note: We use ordered_gradients and ordered_hessians to improve cache hit chance
  *        The naive solution is using gradients[data_indices[i]] for data_indices[i] to get gradients,
  which is not cache friendly, since the access of memory is not continuous.
  *        ordered_gradients and ordered_hessians are preprocessed, and they are re-ordered by data_indices.
  *        Ordered_gradients[i] is aligned with data_indices[i]'s gradients (same for ordered_hessians).
  * \param data_indices Used data indices in current leaf
  * \param start start index in data_indices
  * \param end end index in data_indices
  * \param ordered_gradients Pointer to gradients, the data_indices[i]-th data's gradient is ordered_gradients[i]
  * \param out Output Result
  */
  virtual void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                  const score_t* ordered_gradients, hist_t* out) const = 0;

  virtual void ConstructHistogram(data_size_t start, data_size_t end,
                                  const score_t* ordered_gradients, hist_t* out) const = 0;

  virtual void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const = 0;

  virtual void ConstructHistogramInt8(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const = 0;

  virtual void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const = 0;

  virtual void ConstructHistogramInt16(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const = 0;

  virtual void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const = 0;

  virtual void ConstructHistogramInt32(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const = 0;

  virtual data_size_t Split(uint32_t min_bin, uint32_t max_bin,
                            uint32_t default_bin, uint32_t most_freq_bin,
                            MissingType missing_type, bool default_left,
                            uint32_t threshold, const data_size_t* data_indices,
                            data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const = 0;

  virtual data_size_t SplitCategorical(
      uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin,
      const uint32_t* threshold, int num_threshold,
      const data_size_t* data_indices, data_size_t cnt,
      data_size_t* lte_indices, data_size_t* gt_indices) const = 0;

  virtual data_size_t Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const = 0;

  virtual data_size_t SplitCategorical(
      uint32_t max_bin, uint32_t most_freq_bin, const uint32_t* threshold,
      int num_threshold, const data_size_t* data_indices, data_size_t cnt,
      data_size_t* lte_indices, data_size_t* gt_indices) const = 0;

  /*!
  * \brief After pushed all feature data, call this could have better refactor for bin data
  */
  virtual void FinishLoad() = 0;

  /*!
  * \brief Create object for bin data of one feature, used for dense feature
  * \param num_data Total number of data
  * \param num_bin Number of bin
  * \return The bin data object
  */
  static Bin* CreateDenseBin(data_size_t num_data, int num_bin);

  /*!
  * \brief Create object for bin data of one feature, used for sparse feature
  * \param num_data Total number of data
  * \param num_bin Number of bin
  * \return The bin data object
  */
  static Bin* CreateSparseBin(data_size_t num_data, int num_bin);

  /*!
  * \brief Deep copy the bin
  */
  virtual Bin* Clone() = 0;

  virtual const void* GetColWiseData(uint8_t* bit_type, bool* is_sparse, std::vector<BinIterator*>* bin_iterator, const int num_threads) const = 0;

  virtual const void* GetColWiseData(uint8_t* bit_type, bool* is_sparse, BinIterator** bin_iterator) const = 0;
};


class MultiValBin {
 public:
  virtual ~MultiValBin() {}

  virtual data_size_t num_data() const = 0;

  virtual int32_t num_bin() const = 0;

  virtual double num_element_per_row() const = 0;

  virtual const std::vector<uint32_t>& offsets() const = 0;

  virtual void PushOneRow(int tid, data_size_t idx, const std::vector<uint32_t>& values) = 0;

  virtual void CopySubrow(const MultiValBin* full_bin,
                          const data_size_t* used_indices,
                          data_size_t num_used_indices) = 0;

  virtual MultiValBin* CreateLike(data_size_t num_data, int num_bin,
                                  int num_feature,
                                  double estimate_element_per_row,
                                  const std::vector<uint32_t>& offsets) const = 0;

  virtual void CopySubcol(const MultiValBin* full_bin,
                          const std::vector<int>& used_feature_index,
                          const std::vector<uint32_t>& lower,
                          const std::vector<uint32_t>& upper,
                          const std::vector<uint32_t>& delta) = 0;

  virtual void ReSize(data_size_t num_data, int num_bin, int num_feature,
                      double estimate_element_per_row, const std::vector<uint32_t>& offsets) = 0;

  virtual void CopySubrowAndSubcol(
      const MultiValBin* full_bin, const data_size_t* used_indices,
      data_size_t num_used_indices, const std::vector<int>& used_feature_index,
      const std::vector<uint32_t>& lower, const std::vector<uint32_t>& upper,
      const std::vector<uint32_t>& delta) = 0;

  virtual void ConstructHistogram(const data_size_t* data_indices,
                                  data_size_t start, data_size_t end,
                                  const score_t* gradients,
                                  const score_t* hessians,
                                  hist_t* out) const = 0;

  virtual void ConstructHistogram(data_size_t start, data_size_t end,
                                  const score_t* gradients,
                                  const score_t* hessians,
                                  hist_t* out) const = 0;

  virtual void ConstructHistogramOrdered(const data_size_t* data_indices,
                                         data_size_t start, data_size_t end,
                                         const score_t* ordered_gradients,
                                         const score_t* ordered_hessians,
                                         hist_t* out) const = 0;

  virtual void ConstructHistogramInt32(const data_size_t* data_indices,
                                       data_size_t start, data_size_t end,
                                       const score_t* gradients,
                                       const score_t* hessians,
                                       hist_t* out) const = 0;

  virtual void ConstructHistogramInt32(data_size_t start, data_size_t end,
                                       const score_t* gradients,
                                       const score_t* hessians,
                                       hist_t* out) const = 0;

  virtual void ConstructHistogramOrderedInt32(const data_size_t* data_indices,
                                              data_size_t start, data_size_t end,
                                              const score_t* ordered_gradients,
                                              const score_t* ordered_hessians,
                                              hist_t* out) const = 0;

  virtual void ConstructHistogramInt16(const data_size_t* data_indices,
                                       data_size_t start, data_size_t end,
                                       const score_t* gradients,
                                       const score_t* hessians,
                                       hist_t* out) const = 0;

  virtual void ConstructHistogramInt16(data_size_t start, data_size_t end,
                                       const score_t* gradients,
                                       const score_t* hessians,
                                       hist_t* out) const = 0;

  virtual void ConstructHistogramOrderedInt16(const data_size_t* data_indices,
                                              data_size_t start, data_size_t end,
                                              const score_t* ordered_gradients,
                                              const score_t* ordered_hessians,
                                              hist_t* out) const = 0;

  virtual void ConstructHistogramInt8(const data_size_t* data_indices,
                                      data_size_t start, data_size_t end,
                                      const score_t* gradients,
                                      const score_t* hessians,
                                      hist_t* out) const = 0;

  virtual void ConstructHistogramInt8(data_size_t start, data_size_t end,
                                      const score_t* gradients,
                                      const score_t* hessians,
                                      hist_t* out) const = 0;

  virtual void ConstructHistogramOrderedInt8(const data_size_t* data_indices,
                                             data_size_t start, data_size_t end,
                                             const score_t* ordered_gradients,
                                             const score_t* ordered_hessians,
                                             hist_t* out) const = 0;

  virtual void FinishLoad() = 0;

  virtual bool IsSparse() = 0;

  static MultiValBin* CreateMultiValBin(data_size_t num_data, int num_bin,
                                        int num_feature, double sparse_rate, const std::vector<uint32_t>& offsets);

  static MultiValBin* CreateMultiValDenseBin(data_size_t num_data, int num_bin,
                                             int num_feature, const std::vector<uint32_t>& offsets);

  static MultiValBin* CreateMultiValSparseBin(data_size_t num_data, int num_bin, double estimate_element_per_row);

  static constexpr double multi_val_bin_sparse_threshold = 0.25f;

  virtual MultiValBin* Clone() = 0;

  #ifdef USE_CUDA
  virtual const void* GetRowWiseData(uint8_t* bit_type,
    size_t* total_size,
    bool* is_sparse,
    const void** out_data_ptr,
    uint8_t* data_ptr_bit_type) const = 0;
  #endif  // USE_CUDA
};

inline uint32_t BinMapper::ValueToBin(double value) const {
  if (std::isnan(value)) {
    if (bin_type_ == BinType::CategoricalBin) {
      return 0;
    } else if (missing_type_ == MissingType::NaN) {
      return num_bin_ - 1;
    } else {
      value = 0.0f;
    }
  }
  if (bin_type_ == BinType::NumericalBin) {
    // binary search to find bin
    int l = 0;
    int r = num_bin_ - 1;
    if (missing_type_ == MissingType::NaN) {
      r -= 1;
    }
    while (l < r) {
      int m = (r + l - 1) / 2;
      if (value <= bin_upper_bound_[m]) {
        r = m;
      } else {
        l = m + 1;
      }
    }
    return l;
  } else {
    int int_value = static_cast<int>(value);
    // convert negative value to NaN bin
    if (int_value < 0) {
      return 0;
    }
    if (categorical_2_bin_.count(int_value)) {
      return categorical_2_bin_.at(int_value);
    } else {
      return 0;
    }
  }
}

}  // namespace LightGBM

#endif   // LIGHTGBM_INCLUDE_LIGHTGBM_BIN_H_
