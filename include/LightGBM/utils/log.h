/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_LOG_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_LOG_H_

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef LGB_R_BUILD

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif

#ifndef R_USE_C99_IN_CXX
#define R_USE_C99_IN_CXX
#endif

#include <R_ext/Print.h>
extern "C" void R_FlushConsole(void);
#endif

namespace LightGBM {

#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL thread_local
#endif

#ifndef CHECK
#define CHECK(condition)                                                    \
  if (!(condition))                                                         \
    Log::Fatal("Check failed: " #condition " at %s, line %d .\n", __FILE__, \
               __LINE__);
#endif

#ifndef CHECK_EQ
#define CHECK_EQ(a, b) CHECK((a) == (b))
#endif

#ifndef CHECK_NE
#define CHECK_NE(a, b) CHECK((a) != (b))
#endif

#ifndef CHECK_GE
#define CHECK_GE(a, b) CHECK((a) >= (b))
#endif

#ifndef CHECK_LE
#define CHECK_LE(a, b) CHECK((a) <= (b))
#endif

#ifndef CHECK_GT
#define CHECK_GT(a, b) CHECK((a) > (b))
#endif

#ifndef CHECK_LT
#define CHECK_LT(a, b) CHECK((a) < (b))
#endif

#ifndef CHECK_NOTNULL
#define CHECK_NOTNULL(pointer)                                         \
  if ((pointer) == nullptr)                                            \
    LightGBM::Log::Fatal(#pointer " Can't be NULL at %s, line %d .\n", \
                         __FILE__, __LINE__);
#endif

enum class LogLevel : int {
  Fatal = -1,
  Warning = 0,
  Info = 1,
  Debug = 2,
};

/*!
 * \brief A static Log class
 */
class Log {
 public:
  using Callback = void (*)(const char *);
  /*!
   * \brief Resets the minimal log level. It is INFO by default.
   * \param level The new minimal log level.
   */
  static void ResetLogLevel(LogLevel level) { GetLevel() = level; }

  static void ResetCallBack(Callback callback) { GetLogCallBack() = callback; }

  static void Debug(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Debug, "Debug", format, val);
    va_end(val);
  }
  static void Info(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Info, "Info", format, val);
    va_end(val);
  }
  static void Warning(const char *format, ...) {
    va_list val;
    va_start(val, format);
    Write(LogLevel::Warning, "Warning", format, val);
    va_end(val);
  }
  static void Fatal(const char *format, ...) {
    va_list val;
    const size_t kBufSize = 1024;
    char str_buf[kBufSize];
    va_start(val, format);
#ifdef _MSC_VER
    vsnprintf_s(str_buf, kBufSize, format, val);
#else
    vsnprintf(str_buf, kBufSize, format, val);
#endif
    va_end(val);

// R code should write back to R's error stream,
// otherwise to stderr
#ifndef LGB_R_BUILD
    fprintf(stderr, "[LightGBM] [Fatal] %s\n", str_buf);
    fflush(stderr);
#else
    REprintf("[LightGBM] [Fatal] %s\n", str_buf);
    R_FlushConsole();
#endif
    throw std::runtime_error(std::string(str_buf));
  }

 private:
  static void Write(LogLevel level, const char *level_str, const char *format,
                    va_list val) {
    if (level <= GetLevel()) {  // omit the message with low level
// R code should write back to R's output stream,
// otherwise to stdout
#ifndef LGB_R_BUILD
      if (GetLogCallBack() == nullptr) {
        printf("[LightGBM] [%s] ", level_str);
        vprintf(format, val);
        printf("\n");
        fflush(stdout);
      } else {
        const size_t kBufSize = 512;
        char buf[kBufSize];
        snprintf(buf, kBufSize, "[LightGBM] [%s] ", level_str);
        GetLogCallBack()(buf);
        vsnprintf(buf, kBufSize, format, val);
        GetLogCallBack()(buf);
        GetLogCallBack()("\n");
      }
#else
      Rprintf("[LightGBM] [%s] ", level_str);
      Rvprintf(format, val);
      Rprintf("\n");
      R_FlushConsole();
#endif
    }
  }

  // a trick to use static variable in header file.
  // May be not good, but avoid to use an additional cpp file
  static LogLevel &GetLevel() {
    static THREAD_LOCAL LogLevel level = LogLevel::Info;
    return level;
  }

  static Callback &GetLogCallBack() {
    static THREAD_LOCAL Callback callback = nullptr;
    return callback;
  }
};

}  // namespace LightGBM
#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_LOG_H_
