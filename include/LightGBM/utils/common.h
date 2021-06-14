/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_UTILS_COMMON_H_
#define LIGHTGBM_UTILS_COMMON_H_

#if ((defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__)))
#include <LightGBM/utils/common_legacy_solaris.h>
#endif
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <limits>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if (!((defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__))))
#define FMT_HEADER_ONLY
#include "../../../external_libs/fmt/include/fmt/format.h"
#endif
#include "../../../external_libs/fast_double_parser/include/fast_double_parser.h"

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#endif

#if defined(_MSC_VER)
#include <malloc.h>
#elif MM_MALLOC
#include <mm_malloc.h>
// https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
// https://www.oreilly.com/library/view/mac-os-x/0596003560/ch05s01s02.html
#elif defined(__GNUC__) && defined(HAVE_MALLOC_H)
  #include <malloc.h>
  #define _mm_malloc(a, b) memalign(b, a)
  #define _mm_free(a) free(a)
#else
#include <stdlib.h>
#define _mm_malloc(a, b) malloc(a)
#define _mm_free(a) free(a)
#endif

namespace LightGBM {

namespace Common {

/*!
* Imbues the stream with the C locale.
*/
static void C_stringstream(std::stringstream &ss) {
  ss.imbue(std::locale::classic());
}

inline static char tolower(char in) {
  if (in <= 'Z' && in >= 'A')
    return in - ('Z' - 'z');
  return in;
}

inline static std::string Trim(std::string str) {
  if (str.empty()) {
    return str;
  }
  str.erase(str.find_last_not_of(" \f\n\r\t\v") + 1);
  str.erase(0, str.find_first_not_of(" \f\n\r\t\v"));
  return str;
}

inline static std::string RemoveQuotationSymbol(std::string str) {
  if (str.empty()) {
    return str;
  }
  str.erase(str.find_last_not_of("'\"") + 1);
  str.erase(0, str.find_first_not_of("'\""));
  return str;
}

inline static bool StartsWith(const std::string& str, const std::string prefix) {
  if (str.substr(0, prefix.size()) == prefix) {
    return true;
  } else {
    return false;
  }
}

inline static std::vector<std::string> Split(const char* c_str, char delimiter) {
  std::vector<std::string> ret;
  std::string str(c_str);
  size_t i = 0;
  size_t pos = 0;
  while (pos < str.length()) {
    if (str[pos] == delimiter) {
      if (i < pos) {
        ret.push_back(str.substr(i, pos - i));
      }
      ++pos;
      i = pos;
    } else {
      ++pos;
    }
  }
  if (i < pos) {
    ret.push_back(str.substr(i));
  }
  return ret;
}

inline static std::vector<std::string> SplitBrackets(const char* c_str, char left_delimiter, char right_delimiter) {
  std::vector<std::string> ret;
  std::string str(c_str);
  size_t i = 0;
  size_t pos = 0;
  bool open = false;
  while (pos < str.length()) {
    if (str[pos] == left_delimiter) {
      open = true;
      ++pos;
      i = pos;
    } else if (str[pos] == right_delimiter && open) {
      if (i < pos) {
        ret.push_back(str.substr(i, pos - i));
      }
      open = false;
      ++pos;
    } else {
      ++pos;
    }
  }
  return ret;
}

inline static std::vector<std::string> SplitLines(const char* c_str) {
  std::vector<std::string> ret;
  std::string str(c_str);
  size_t i = 0;
  size_t pos = 0;
  while (pos < str.length()) {
    if (str[pos] == '\n' || str[pos] == '\r') {
      if (i < pos) {
        ret.push_back(str.substr(i, pos - i));
      }
      // skip the line endings
      while (str[pos] == '\n' || str[pos] == '\r') ++pos;
      // new begin
      i = pos;
    } else {
      ++pos;
    }
  }
  if (i < pos) {
    ret.push_back(str.substr(i));
  }
  return ret;
}

inline static std::vector<std::string> Split(const char* c_str, const char* delimiters) {
  std::vector<std::string> ret;
  std::string str(c_str);
  size_t i = 0;
  size_t pos = 0;
  while (pos < str.length()) {
    bool met_delimiters = false;
    for (int j = 0; delimiters[j] != '\0'; ++j) {
      if (str[pos] == delimiters[j]) {
        met_delimiters = true;
        break;
      }
    }
    if (met_delimiters) {
      if (i < pos) {
        ret.push_back(str.substr(i, pos - i));
      }
      ++pos;
      i = pos;
    } else {
      ++pos;
    }
  }
  if (i < pos) {
    ret.push_back(str.substr(i));
  }
  return ret;
}

template<typename T>
inline static const char* Atoi(const char* p, T* out) {
  int sign;
  T value;
  while (*p == ' ') {
    ++p;
  }
  sign = 1;
  if (*p == '-') {
    sign = -1;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  for (value = 0; *p >= '0' && *p <= '9'; ++p) {
    value = value * 10 + (*p - '0');
  }
  *out = static_cast<T>(sign * value);
  while (*p == ' ') {
    ++p;
  }
  return p;
}

template<typename T>
inline static double Pow(T base, int power) {
  if (power < 0) {
    return 1.0 / Pow(base, -power);
  } else if (power == 0) {
    return 1;
  } else if (power % 2 == 0) {
    return Pow(base*base, power / 2);
  } else if (power % 3 == 0) {
    return Pow(base*base*base, power / 3);
  } else {
    return base * Pow(base, power - 1);
  }
}

inline static const char* Atof(const char* p, double* out) {
  int frac;
  double sign, value, scale;
  *out = NAN;
  // Skip leading white space, if any.
  while (*p == ' ') {
    ++p;
  }
  // Get sign, if any.
  sign = 1.0;
  if (*p == '-') {
    sign = -1.0;
    ++p;
  } else if (*p == '+') {
    ++p;
  }

  // is a number
  if ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E') {
    // Get digits before decimal point or exponent, if any.
    for (value = 0.0; *p >= '0' && *p <= '9'; ++p) {
      value = value * 10.0 + (*p - '0');
    }

    // Get digits after decimal point, if any.
    if (*p == '.') {
      double right = 0.0;
      int nn = 0;
      ++p;
      while (*p >= '0' && *p <= '9') {
        right = (*p - '0') + right * 10.0;
        ++nn;
        ++p;
      }
      value += right / Pow(10.0, nn);
    }

    // Handle exponent, if any.
    frac = 0;
    scale = 1.0;
    if ((*p == 'e') || (*p == 'E')) {
      uint32_t expon;
      // Get sign of exponent, if any.
      ++p;
      if (*p == '-') {
        frac = 1;
        ++p;
      } else if (*p == '+') {
        ++p;
      }
      // Get digits of exponent, if any.
      for (expon = 0; *p >= '0' && *p <= '9'; ++p) {
        expon = expon * 10 + (*p - '0');
      }
      if (expon > 308) expon = 308;
      // Calculate scaling factor.
      while (expon >= 50) { scale *= 1E50; expon -= 50; }
      while (expon >= 8) { scale *= 1E8;  expon -= 8; }
      while (expon > 0) { scale *= 10.0; expon -= 1; }
    }
    // Return signed and scaled floating point result.
    *out = sign * (frac ? (value / scale) : (value * scale));
  } else {
    size_t cnt = 0;
    while (*(p + cnt) != '\0' && *(p + cnt) != ' '
           && *(p + cnt) != '\t' && *(p + cnt) != ','
           && *(p + cnt) != '\n' && *(p + cnt) != '\r'
           && *(p + cnt) != ':') {
      ++cnt;
    }
    if (cnt > 0) {
      std::string tmp_str(p, cnt);
      std::transform(tmp_str.begin(), tmp_str.end(), tmp_str.begin(), Common::tolower);
      if (tmp_str == std::string("na") || tmp_str == std::string("nan") ||
          tmp_str == std::string("null")) {
        *out = NAN;
      } else if (tmp_str == std::string("inf") || tmp_str == std::string("infinity")) {
        *out = sign * 1e308;
      } else {
        Log::Fatal("Unknown token %s in data file", tmp_str.c_str());
      }
      p += cnt;
    }
  }

  while (*p == ' ') {
    ++p;
  }

  return p;
}

// Use fast_double_parse and strtod (if parse failed) to parse double.
inline static const char* AtofPrecise(const char* p, double* out) {
  const char* end = fast_double_parser::parse_number(p, out);

  if (end != nullptr) {
    return end;
  }

  // Rare path: Not in RFC 7159 format. Possible "inf", "nan", etc. Fallback to standard library:
  char* end2;
  errno = 0;  // This is Required before calling strtod.
  *out = std::strtod(p, &end2);  // strtod is locale aware.
  if (end2 == p) {
    Log::Fatal("no conversion to double for: %s", p);
  }
  if (errno == ERANGE) {
    Log::Fatal("convert to double got underflow or overflow: %s", p);
  }
  return end2;
}

inline static bool AtoiAndCheck(const char* p, int* out) {
  const char* after = Atoi(p, out);
  if (*after != '\0') {
    return false;
  }
  return true;
}

inline static bool AtofAndCheck(const char* p, double* out) {
  const char* after = Atof(p, out);
  if (*after != '\0') {
    return false;
  }
  return true;
}

inline static const char* SkipSpaceAndTab(const char* p) {
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  return p;
}

inline static const char* SkipReturn(const char* p) {
  while (*p == '\n' || *p == '\r' || *p == ' ') {
    ++p;
  }
  return p;
}

template<typename T, typename T2>
inline static std::vector<T2> ArrayCast(const std::vector<T>& arr) {
  std::vector<T2> ret(arr.size());
  for (size_t i = 0; i < arr.size(); ++i) {
    ret[i] = static_cast<T2>(arr[i]);
  }
  return ret;
}

template<typename T, bool is_float>
struct __StringToTHelper {
  T operator()(const std::string& str) const {
    T ret = 0;
    Atoi(str.c_str(), &ret);
    return ret;
  }
};

template<typename T>
struct __StringToTHelper<T, true> {
  T operator()(const std::string& str) const {
    return static_cast<T>(std::stod(str));
  }
};

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, char delimiter) {
  std::vector<std::string> strs = Split(str.c_str(), delimiter);
  std::vector<T> ret;
  ret.reserve(strs.size());
  __StringToTHelper<T, std::is_floating_point<T>::value> helper;
  for (const auto& s : strs) {
    ret.push_back(helper(s));
  }
  return ret;
}

template<typename T>
inline static std::vector<std::vector<T>> StringToArrayofArrays(
    const std::string& str, char left_bracket, char right_bracket, char delimiter) {
  std::vector<std::string> strs = SplitBrackets(str.c_str(), left_bracket, right_bracket);
  std::vector<std::vector<T>> ret;
  for (const auto& s : strs) {
    ret.push_back(StringToArray<T>(s, delimiter));
  }
  return ret;
}

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, int n) {
  if (n == 0) {
    return std::vector<T>();
  }
  std::vector<std::string> strs = Split(str.c_str(), ' ');
  CHECK_EQ(strs.size(), static_cast<size_t>(n));
  std::vector<T> ret;
  ret.reserve(strs.size());
  __StringToTHelper<T, std::is_floating_point<T>::value> helper;
  for (const auto& s : strs) {
    ret.push_back(helper(s));
  }
  return ret;
}

template<typename T, bool is_float>
struct __StringToTHelperFast {
  const char* operator()(const char*p, T* out) const {
    return Atoi(p, out);
  }
};

template<typename T>
struct __StringToTHelperFast<T, true> {
  const char* operator()(const char*p, T* out) const {
    double tmp = 0.0f;
    auto ret = Atof(p, &tmp);
    *out = static_cast<T>(tmp);
    return ret;
  }
};

template<typename T>
inline static std::vector<T> StringToArrayFast(const std::string& str, int n) {
  if (n == 0) {
    return std::vector<T>();
  }
  auto p_str = str.c_str();
  __StringToTHelperFast<T, std::is_floating_point<T>::value> helper;
  std::vector<T> ret(n);
  for (int i = 0; i < n; ++i) {
    p_str = helper(p_str, &ret[i]);
  }
  return ret;
}

template<typename T>
inline static std::string Join(const std::vector<T>& strs, const char* delimiter, const bool force_C_locale = false) {
  if (strs.empty()) {
    return std::string("");
  }
  std::stringstream str_buf;
  if (force_C_locale) {
    C_stringstream(str_buf);
  }
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << strs[0];
  for (size_t i = 1; i < strs.size(); ++i) {
    str_buf << delimiter;
    str_buf << strs[i];
  }
  return str_buf.str();
}

template<>
inline std::string Join<int8_t>(const std::vector<int8_t>& strs, const char* delimiter, const bool force_C_locale) {
  if (strs.empty()) {
    return std::string("");
  }
  std::stringstream str_buf;
  if (force_C_locale) {
    C_stringstream(str_buf);
  }
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << static_cast<int16_t>(strs[0]);
  for (size_t i = 1; i < strs.size(); ++i) {
    str_buf << delimiter;
    str_buf << static_cast<int16_t>(strs[i]);
  }
  return str_buf.str();
}

template<typename T>
inline static std::string Join(const std::vector<T>& strs, size_t start, size_t end, const char* delimiter, const bool force_C_locale = false) {
  if (end - start <= 0) {
    return std::string("");
  }
  start = std::min(start, static_cast<size_t>(strs.size()) - 1);
  end = std::min(end, static_cast<size_t>(strs.size()));
  std::stringstream str_buf;
  if (force_C_locale) {
    C_stringstream(str_buf);
  }
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << strs[start];
  for (size_t i = start + 1; i < end; ++i) {
    str_buf << delimiter;
    str_buf << strs[i];
  }
  return str_buf.str();
}

inline static int64_t Pow2RoundUp(int64_t x) {
  int64_t t = 1;
  for (int i = 0; i < 64; ++i) {
    if (t >= x) {
      return t;
    }
    t <<= 1;
  }
  return 0;
}

/*!
 * \brief Do inplace softmax transformation on p_rec
 * \param p_rec The input/output vector of the values.
 */
inline static void Softmax(std::vector<double>* p_rec) {
  std::vector<double> &rec = *p_rec;
  double wmax = rec[0];
  for (size_t i = 1; i < rec.size(); ++i) {
    wmax = std::max(rec[i], wmax);
  }
  double wsum = 0.0f;
  for (size_t i = 0; i < rec.size(); ++i) {
    rec[i] = std::exp(rec[i] - wmax);
    wsum += rec[i];
  }
  for (size_t i = 0; i < rec.size(); ++i) {
    rec[i] /= static_cast<double>(wsum);
  }
}

inline static void Softmax(const double* input, double* output, int len) {
  double wmax = input[0];
  for (int i = 1; i < len; ++i) {
    wmax = std::max(input[i], wmax);
  }
  double wsum = 0.0f;
  for (int i = 0; i < len; ++i) {
    output[i] = std::exp(input[i] - wmax);
    wsum += output[i];
  }
  for (int i = 0; i < len; ++i) {
    output[i] /= static_cast<double>(wsum);
  }
}

template<typename T>
std::vector<const T*> ConstPtrInVectorWrapper(const std::vector<std::unique_ptr<T>>& input) {
  std::vector<const T*> ret;
  for (auto t = input.begin(); t !=input.end(); ++t) {
    ret.push_back(t->get());
  }
  return ret;
}

template<typename T1, typename T2>
inline static void SortForPair(std::vector<T1>* keys, std::vector<T2>* values, size_t start, bool is_reverse = false) {
  std::vector<std::pair<T1, T2>> arr;
  auto& ref_key = *keys;
  auto& ref_value = *values;
  for (size_t i = start; i < keys->size(); ++i) {
    arr.emplace_back(ref_key[i], ref_value[i]);
  }
  if (!is_reverse) {
    std::stable_sort(arr.begin(), arr.end(), [](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
      return a.first < b.first;
    });
  } else {
    std::stable_sort(arr.begin(), arr.end(), [](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
      return a.first > b.first;
    });
  }
  for (size_t i = start; i < arr.size(); ++i) {
    ref_key[i] = arr[i].first;
    ref_value[i] = arr[i].second;
  }
}

template <typename T>
inline static std::vector<T*> Vector2Ptr(std::vector<std::vector<T>>* data) {
  std::vector<T*> ptr(data->size());
  auto& ref_data = *data;
  for (size_t i = 0; i < data->size(); ++i) {
    ptr[i] = ref_data[i].data();
  }
  return ptr;
}

template <typename T>
inline static std::vector<int> VectorSize(const std::vector<std::vector<T>>& data) {
  std::vector<int> ret(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    ret[i] = static_cast<int>(data[i].size());
  }
  return ret;
}

inline static double AvoidInf(double x) {
  if (std::isnan(x)) {
    return 0.0;
  } else if (x >= 1e300) {
    return 1e300;
  } else if (x <= -1e300) {
    return -1e300;
  } else {
    return x;
  }
}

inline static float AvoidInf(float x) {
  if (std::isnan(x)) {
    return 0.0f;
  } else if (x >= 1e38) {
    return 1e38f;
  } else if (x <= -1e38) {
    return -1e38f;
  } else {
    return x;
  }
}

template<typename _Iter> inline
static typename std::iterator_traits<_Iter>::value_type* IteratorValType(_Iter) {
  return (0);
}

template<typename _RanIt, typename _Pr, typename _VTRanIt> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred, _VTRanIt*) {
  size_t len = _Last - _First;
  const size_t kMinInnerLen = 1024;
  int num_threads = OMP_NUM_THREADS();
  if (len <= kMinInnerLen || num_threads <= 1) {
    std::sort(_First, _Last, _Pred);
    return;
  }
  size_t inner_size = (len + num_threads - 1) / num_threads;
  inner_size = std::max(inner_size, kMinInnerLen);
  num_threads = static_cast<int>((len + inner_size - 1) / inner_size);
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < num_threads; ++i) {
    size_t left = inner_size*i;
    size_t right = left + inner_size;
    right = std::min(right, len);
    if (right > left) {
      std::sort(_First + left, _First + right, _Pred);
    }
  }
  // Buffer for merge.
  std::vector<_VTRanIt> temp_buf(len);
  _RanIt buf = temp_buf.begin();
  size_t s = inner_size;
  // Recursive merge
  while (s < len) {
    int loop_size = static_cast<int>((len + s * 2 - 1) / (s * 2));
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < loop_size; ++i) {
      size_t left = i * 2 * s;
      size_t mid = left + s;
      size_t right = mid + s;
      right = std::min(len, right);
      if (mid >= right) { continue; }
      std::copy(_First + left, _First + mid, buf + left);
      std::merge(buf + left, buf + mid, _First + mid, _First + right, _First + left, _Pred);
    }
    s *= 2;
  }
}

template<typename _RanIt, typename _Pr> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred) {
  return ParallelSort(_First, _Last, _Pred, IteratorValType(_First));
}

// Check that all y[] are in interval [ymin, ymax] (end points included); throws error if not
template <typename T>
inline static void CheckElementsIntervalClosed(const T *y, T ymin, T ymax, int ny, const char *callername) {
  auto fatal_msg = [&y, &ymin, &ymax, &callername](int i) {
    std::ostringstream os;
    os << "[%s]: does not tolerate element [#%i = " << y[i] << "] outside [" << ymin << ", " << ymax << "]";
    Log::Fatal(os.str().c_str(), callername, i);
  };
  for (int i = 1; i < ny; i += 2) {
    if (y[i - 1] < y[i]) {
      if (y[i - 1] < ymin) {
        fatal_msg(i - 1);
      } else if (y[i] > ymax) {
        fatal_msg(i);
      }
    } else {
      if (y[i - 1] > ymax) {
        fatal_msg(i - 1);
      } else if (y[i] < ymin) {
        fatal_msg(i);
      }
    }
  }
  if (ny & 1) {  // odd
    if (y[ny - 1] < ymin || y[ny - 1] > ymax) {
      fatal_msg(ny - 1);
    }
  }
}

// One-pass scan over array w with nw elements: find min, max and sum of elements;
// this is useful for checking weight requirements.
template <typename T1, typename T2>
inline static void ObtainMinMaxSum(const T1 *w, int nw, T1 *mi, T1 *ma, T2 *su) {
  T1 minw;
  T1 maxw;
  T1 sumw;
  int i;
  if (nw & 1) {  // odd
    minw = w[0];
    maxw = w[0];
    sumw = w[0];
    i = 2;
  } else {  // even
    if (w[0] < w[1]) {
      minw = w[0];
      maxw = w[1];
    } else {
      minw = w[1];
      maxw = w[0];
    }
    sumw = w[0] + w[1];
    i = 3;
  }
  for (; i < nw; i += 2) {
    if (w[i - 1] < w[i]) {
      minw = std::min(minw, w[i - 1]);
      maxw = std::max(maxw, w[i]);
    } else {
      minw = std::min(minw, w[i]);
      maxw = std::max(maxw, w[i - 1]);
    }
    sumw += w[i - 1] + w[i];
  }
  if (mi != nullptr) {
    *mi = minw;
  }
  if (ma != nullptr) {
    *ma = maxw;
  }
  if (su != nullptr) {
    *su = static_cast<T2>(sumw);
  }
}

inline static std::vector<uint32_t> EmptyBitset(int n) {
  int size = n / 32;
  if (n % 32 != 0) ++size;
  return std::vector<uint32_t>(size);
}

template<typename T>
inline static void InsertBitset(std::vector<uint32_t>* vec, const T val) {
  auto& ref_v = *vec;
  int i1 = val / 32;
  int i2 = val % 32;
  if (static_cast<int>(vec->size()) < i1 + 1) {
    vec->resize(i1 + 1, 0);
  }
  ref_v[i1] |= (1 << i2);
}

template<typename T>
inline static std::vector<uint32_t> ConstructBitset(const T* vals, int n) {
  std::vector<uint32_t> ret;
  for (int i = 0; i < n; ++i) {
    int i1 = vals[i] / 32;
    int i2 = vals[i] % 32;
    if (static_cast<int>(ret.size()) < i1 + 1) {
      ret.resize(i1 + 1, 0);
    }
    ret[i1] |= (1 << i2);
  }
  return ret;
}

template<typename T>
inline static bool FindInBitset(const uint32_t* bits, int n, T pos) {
  int i1 = pos / 32;
  if (i1 >= n) {
    return false;
  }
  int i2 = pos % 32;
  return (bits[i1] >> i2) & 1;
}

inline static bool CheckDoubleEqualOrdered(double a, double b) {
  double upper = std::nextafter(a, INFINITY);
  return b <= upper;
}

inline static double GetDoubleUpperBound(double a) {
  return std::nextafter(a, INFINITY);
}

inline static size_t GetLine(const char* str) {
  auto start = str;
  while (*str != '\0' && *str != '\n' && *str != '\r') {
    ++str;
  }
  return str - start;
}

inline static const char* SkipNewLine(const char* str) {
  if (*str == '\r') {
    ++str;
  }
  if (*str == '\n') {
    ++str;
  }
  return str;
}

template <typename T>
static int Sign(T x) {
  return (x > T(0)) - (x < T(0));
}

template <typename T>
static T SafeLog(T x) {
  if (x > 0) {
    return std::log(x);
  } else {
    return -INFINITY;
  }
}

inline bool CheckAllowedJSON(const std::string& s) {
  unsigned char char_code;
  for (auto c : s) {
    char_code = static_cast<unsigned char>(c);
    if (char_code == 34      // "
        || char_code == 44   // ,
        || char_code == 58   // :
        || char_code == 91   // [
        || char_code == 93   // ]
        || char_code == 123  // {
        || char_code == 125  // }
        ) {
      return false;
    }
  }
  return true;
}

inline int RoundInt(double x) {
  return static_cast<int>(x + 0.5f);
}

template <typename T, std::size_t N = 32>
class AlignmentAllocator {
 public:
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T* pointer;
  typedef const T* const_pointer;

  typedef T& reference;
  typedef const T& const_reference;

  inline AlignmentAllocator() throw() {}

  template <typename T2>
  inline AlignmentAllocator(const AlignmentAllocator<T2, N>&) throw() {}

  inline ~AlignmentAllocator() throw() {}

  inline pointer adress(reference r) {
    return &r;
  }

  inline const_pointer adress(const_reference r) const {
    return &r;
  }

  inline pointer allocate(size_type n) {
    return (pointer)_mm_malloc(n * sizeof(value_type), N);
  }

  inline void deallocate(pointer p, size_type) {
    _mm_free(p);
  }

  inline void construct(pointer p, const value_type& wert) {
    new (p) value_type(wert);
  }

  inline void destroy(pointer p) {
    p->~value_type();
  }

  inline size_type max_size() const throw() {
    return size_type(-1) / sizeof(value_type);
  }

  template <typename T2>
  struct rebind {
    typedef AlignmentAllocator<T2, N> other;
  };

  bool operator!=(const AlignmentAllocator<T, N>& other) const {
    return !(*this == other);
  }

  // Returns true if and only if storage allocated from *this
  // can be deallocated from other, and vice versa.
  // Always returns true for stateless allocators.
  bool operator==(const AlignmentAllocator<T, N>&) const {
    return true;
  }
};

class Timer {
 public:
  Timer() {
#ifdef TIMETAG
    int num_threads = OMP_NUM_THREADS();
    start_time_.resize(num_threads);
    stats_.resize(num_threads);
#endif  // TIMETAG
  }

  ~Timer() { Print(); }

#ifdef TIMETAG
  void Start(const std::string& name) {
    auto tid = omp_get_thread_num();
    start_time_[tid][name] = std::chrono::steady_clock::now();
  }

  void Stop(const std::string& name) {
    auto cur_time = std::chrono::steady_clock::now();
    auto tid = omp_get_thread_num();
    if (stats_[tid].find(name) == stats_[tid].end()) {
      stats_[tid][name] = std::chrono::duration<double, std::milli>(0);
    }
    stats_[tid][name] += cur_time - start_time_[tid][name];
  }

#else
  void Start(const std::string&) {}

  void Stop(const std::string&) {}
#endif  // TIMETAG

  void Print() const {
#ifdef TIMETAG
    std::unordered_map<std::string, std::chrono::duration<double, std::milli>>
        stats(stats_[0].begin(), stats_[0].end());
    for (size_t i = 1; i < stats_.size(); ++i) {
      for (auto it = stats_[i].begin(); it != stats_[i].end(); ++it) {
        if (stats.find(it->first) == stats.end()) {
          stats[it->first] = it->second;
        } else {
          stats[it->first] += it->second;
        }
      }
    }
    std::map<std::string, std::chrono::duration<double, std::milli>> ordered(
        stats.begin(), stats.end());
    for (auto it = ordered.begin(); it != ordered.end(); ++it) {
      Log::Info("%s costs:\t %f", it->first.c_str(), it->second * 1e-3);
    }
#endif  // TIMETAG
  }
#ifdef TIMETAG
  std::vector<
      std::unordered_map<std::string, std::chrono::steady_clock::time_point>>
      start_time_;
  std::vector<std::unordered_map<std::string,
                                 std::chrono::duration<double, std::milli>>>
      stats_;
#endif  // TIMETAG
};

// Note: this class is not thread-safe, don't use it inside omp blocks
class FunctionTimer {
 public:
#ifdef TIMETAG
  FunctionTimer(const std::string& name, Timer& timer) : timer_(timer) {
    timer.Start(name);
    name_ = name;
  }

  ~FunctionTimer() { timer_.Stop(name_); }

 private:
  std::string name_;
  Timer& timer_;
#else
  FunctionTimer(const std::string&, Timer&) {}
#endif  // TIMETAG
};

}  // namespace Common

extern Common::Timer global_timer;


/*!
* Provides locale-independent alternatives to Common's methods.
* Essential to make models robust to locale settings.
*/
namespace CommonC {

template<typename T>
inline static std::string Join(const std::vector<T>& strs, const char* delimiter) {
  return LightGBM::Common::Join(strs, delimiter, true);
}

template<typename T>
inline static std::string Join(const std::vector<T>& strs, size_t start, size_t end, const char* delimiter) {
  return LightGBM::Common::Join(strs, start, end, delimiter, true);
}

inline static const char* Atof(const char* p, double* out) {
  return LightGBM::Common::Atof(p, out);
}

template<typename T, bool is_float>
struct __StringToTHelperFast {
  const char* operator()(const char*p, T* out) const {
    return LightGBM::Common::Atoi(p, out);
  }
};

/*!
* \warning Beware that ``Common::Atof`` in ``__StringToTHelperFast``,
*          has **less** floating point precision than ``__StringToTHelper``.
*          Both versions are kept to maintain bit-for-bit the "legacy" LightGBM behaviour in terms of precision.
*          Check ``StringToArrayFast`` and ``StringToArray`` for more details on this.
*/
template<typename T>
struct __StringToTHelperFast<T, true> {
  const char* operator()(const char*p, T* out) const {
    double tmp = 0.0f;
    auto ret = Atof(p, &tmp);
    *out = static_cast<T>(tmp);
    return ret;
  }
};

template<typename T, bool is_float>
struct __StringToTHelper {
  T operator()(const std::string& str) const {
    T ret = 0;
    LightGBM::Common::Atoi(str.c_str(), &ret);
    return ret;
  }
};

/*!
* \warning Beware that ``Common::Atof`` in ``__StringToTHelperFast``,
*          has **less** floating point precision than ``__StringToTHelper``.
*          Both versions are kept to maintain bit-for-bit the "legacy" LightGBM behaviour in terms of precision.
*          Check ``StringToArrayFast`` and ``StringToArray`` for more details on this.
* \note It is possible that ``fast_double_parser::parse_number`` is faster than ``Common::Atof``.
*/
template<typename T>
struct __StringToTHelper<T, true> {
  T operator()(const std::string& str) const {
    double tmp;

    const char* end = Common::AtofPrecise(str.c_str(), &tmp);
    if (end == str.c_str()) {
        Log::Fatal("Failed to parse double: %s", str.c_str());
    }

    return static_cast<T>(tmp);
  }
};


/*!
* \warning Beware that due to internal use of ``Common::Atof`` in ``__StringToTHelperFast``,
*          this method has less precision for floating point numbers than ``StringToArray``,
*          which calls ``__StringToTHelper``.
*          As such, ``StringToArrayFast`` and ``StringToArray`` are not equivalent!
*          Both versions were kept to maintain bit-for-bit the "legacy" LightGBM behaviour in terms of precision.
*/
template<typename T>
inline static std::vector<T> StringToArrayFast(const std::string& str, int n) {
  if (n == 0) {
    return std::vector<T>();
  }
  auto p_str = str.c_str();
  __StringToTHelperFast<T, std::is_floating_point<T>::value> helper;
  std::vector<T> ret(n);
  for (int i = 0; i < n; ++i) {
    p_str = helper(p_str, &ret[i]);
  }
  return ret;
}

/*!
* \warning Do not replace calls to this method by ``StringToArrayFast``.
*          This method is more precise for floating point numbers.
*          Check ``StringToArrayFast`` for more details.
*/
template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, int n) {
  if (n == 0) {
    return std::vector<T>();
  }
  std::vector<std::string> strs = LightGBM::Common::Split(str.c_str(), ' ');
  CHECK_EQ(strs.size(), static_cast<size_t>(n));
  std::vector<T> ret;
  ret.reserve(strs.size());
  __StringToTHelper<T, std::is_floating_point<T>::value> helper;
  for (const auto& s : strs) {
    ret.push_back(helper(s));
  }
  return ret;
}

/*!
* \warning Do not replace calls to this method by ``StringToArrayFast``.
*          This method is more precise for floating point numbers.
*          Check ``StringToArrayFast`` for more details.
*/
template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, char delimiter) {
  std::vector<std::string> strs = LightGBM::Common::Split(str.c_str(), delimiter);
  std::vector<T> ret;
  ret.reserve(strs.size());
  __StringToTHelper<T, std::is_floating_point<T>::value> helper;
  for (const auto& s : strs) {
    ret.push_back(helper(s));
  }
  return ret;
}

#if (!((defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__))))
/*!
* Safely formats a value onto a buffer according to a format string and null-terminates it.
*
* \note It checks that the full value was written or forcefully aborts.
*       This safety check serves to prevent incorrect internal API usage.
*       Correct usage will never incur in this problem:
*         - The received buffer size shall be sufficient at all times for the input format string and value.
*/
template <typename T>
inline static void format_to_buf(char* buffer, const size_t buf_len, const char* format, const T value) {
    auto result = fmt::format_to_n(buffer, buf_len, format, value);
    if (result.size >= buf_len) {
      Log::Fatal("Numerical conversion failed. Buffer is too small.");
    }
    buffer[result.size] = '\0';
}

template<typename T, bool is_float, bool high_precision>
struct __TToStringHelper {
  void operator()(T value, char* buffer, size_t buf_len) const {
    format_to_buf(buffer, buf_len, "{}", value);
  }
};

template<typename T>
struct __TToStringHelper<T, true, false> {
  void operator()(T value, char* buffer, size_t buf_len) const {
    format_to_buf(buffer, buf_len, "{:g}", value);
  }
};

template<typename T>
struct __TToStringHelper<T, true, true> {
  void operator()(T value, char* buffer, size_t buf_len) const {
    format_to_buf(buffer, buf_len, "{:.17g}", value);
  }
};

/*!
* Converts an array to a string with values separated by the space character.
* This method replaces Common's ``ArrayToString`` and ``ArrayToStringFast`` functionality
* and is locale-independent.
* 
* \note If ``high_precision_output`` is set to true,
*       floating point values are output with more digits of precision.
*/
template<bool high_precision_output = false, typename T>
inline static std::string ArrayToString(const std::vector<T>& arr, size_t n) {
  if (arr.empty() || n == 0) {
    return std::string("");
  }
  __TToStringHelper<T, std::is_floating_point<T>::value, high_precision_output> helper;
  const size_t buf_len = high_precision_output ? 32 : 16;
  std::vector<char> buffer(buf_len);
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  helper(arr[0], buffer.data(), buf_len);
  str_buf << buffer.data();
  for (size_t i = 1; i < std::min(n, arr.size()); ++i) {
    helper(arr[i], buffer.data(), buf_len);
    str_buf << ' ' << buffer.data();
  }
  return str_buf.str();
}

/*!
* Converts an unordered_map to a string with key-value pairs separated by the specified delimiter.
* Each key-value pair is represented by <key>:<value> with a colon in between.
* This method is locale-independent.
*
* \note If ``high_precision_output`` is set to true,
*       floating point values are output with more digits of precision.
*/
template<bool key_high_precision_output = false, bool value_high_precision_output = false,
  typename KEY_T, typename VALUE_T>
inline static std::string UnorderedMapToString(const std::unordered_map<KEY_T, VALUE_T>& map,
  const char pair_sep, const char delimiter) {
  if (map.empty()) {
    return std::string("");
  }
  __TToStringHelper<KEY_T, std::is_floating_point<KEY_T>::value, key_high_precision_output> key_helper;
  __TToStringHelper<VALUE_T, std::is_floating_point<VALUE_T>::value, value_high_precision_output> value_helper;
  const size_t key_buf_len = key_high_precision_output ? 32 : 16;
  const size_t value_buf_len = value_high_precision_output ? 32 : 16;
  std::vector<char> key_buffer(key_buf_len);
  std::vector<char> value_buffer(value_buf_len);
  std::stringstream str_buf;
  Common::C_stringstream(str_buf);
  auto iter = map.begin();
  key_helper(iter->first, key_buffer.data(), key_buf_len);
  str_buf << key_buffer.data();
  str_buf << pair_sep;
  value_helper(iter->second, value_buffer.data(), value_buf_len);
  str_buf << value_buffer.data();

  ++iter;
  for (; iter != map.end(); ++iter) {
    str_buf << delimiter;
    key_helper(iter->first, key_buffer.data(), key_buf_len);
    str_buf << key_buffer.data();
    str_buf << pair_sep;
    value_helper(iter->second, value_buffer.data(), value_buf_len);
    str_buf << value_buffer.data();
  }
  return str_buf.str();
}

#endif  // (!((defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__))))


}  // namespace CommonC


}  // namespace LightGBM

#endif  // LIGHTGBM_UTILS_COMMON_H_
