/*
 * yamc_shared_lock.hpp
 *
 * MIT License
 *
 * Copyright (c) 2017 yohhoy
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_YAMC_SHARED_LOCK_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_YAMC_SHARED_LOCK_HPP_

#include <cassert>
#include <chrono>
#include <mutex>
#include <system_error>
#include <utility>  // std::swap

/*
 * std::shared_lock in C++14 Standard Library
 *
 * - yamc::shared_lock<Mutex>
 */
namespace yamc {

template <typename Mutex>
class shared_lock {
  void locking_precondition(const char* emsg) {
    if (pm_ == nullptr) {
      throw std::system_error(
          std::make_error_code(std::errc::operation_not_permitted), emsg);
    }
    if (owns_) {
      throw std::system_error(
          std::make_error_code(std::errc::resource_deadlock_would_occur), emsg);
    }
  }

 public:
  using mutex_type = Mutex;

  shared_lock() noexcept = default;

  explicit shared_lock(mutex_type* m) {
    m->lock_shared();
    pm_ = m;
    owns_ = true;
  }

  shared_lock(const mutex_type& m, std::defer_lock_t) noexcept {
    pm_ = &m;
    owns_ = false;
  }

  shared_lock(const mutex_type& m, std::try_to_lock_t) {
    pm_ = &m;
    owns_ = m.try_lock_shared();
  }

  shared_lock(const mutex_type& m, std::adopt_lock_t) {
    pm_ = &m;
    owns_ = true;
  }

  template <typename Clock, typename Duration>
  shared_lock(const mutex_type& m,
              const std::chrono::time_point<Clock, Duration>& abs_time) {
    pm_ = &m;
    owns_ = m.try_lock_shared_until(abs_time);
  }

  template <typename Rep, typename Period>
  shared_lock(const mutex_type& m,
              const std::chrono::duration<Rep, Period>& rel_time) {
    pm_ = &m;
    owns_ = m.try_lock_shared_for(rel_time);
  }

  ~shared_lock() {
    if (owns_) {
      assert(pm_ != nullptr);
      pm_->unlock_shared();
    }
  }

  shared_lock(const shared_lock&) = delete;
  shared_lock& operator=(const shared_lock&) = delete;

  shared_lock(shared_lock&& rhs) noexcept {
    if (pm_ && owns_) {
      pm_->unlock_shared();
    }
    pm_ = rhs.pm_;
    owns_ = rhs.owns_;
    rhs.pm_ = nullptr;
    rhs.owns_ = false;
  }

  shared_lock& operator=(shared_lock&& rhs) noexcept {
    if (pm_ && owns_) {
      pm_->unlock_shared();
    }
    pm_ = rhs.pm_;
    owns_ = rhs.owns_;
    rhs.pm_ = nullptr;
    rhs.owns_ = false;
    return *this;
  }

  void lock() {
    locking_precondition("shared_lock::lock");
    pm_->lock_shared();
    owns_ = true;
  }

  bool try_lock() {
    locking_precondition("shared_lock::try_lock");
    return (owns_ = pm_->try_lock_shared());
  }

  template <typename Rep, typename Period>
  bool try_lock_for(const std::chrono::duration<Rep, Period>& rel_time) {
    locking_precondition("shared_lock::try_lock_for");
    return (owns_ = pm_->try_lock_shared_for(rel_time));
  }

  template <typename Clock, typename Duration>
  bool try_lock_until(
      const std::chrono::time_point<Clock, Duration>& abs_time) {
    locking_precondition("shared_lock::try_lock_until");
    return (owns_ = pm_->try_lock_shared_until(abs_time));
  }

  void unlock() {
    assert(pm_ != nullptr);
    if (!owns_) {
      throw std::system_error(
          std::make_error_code(std::errc::operation_not_permitted),
          "shared_lock::unlock");
    }
    pm_->unlock_shared();
    owns_ = false;
  }

  void swap(shared_lock& sl) noexcept {
    std::swap(pm_, sl.pm_);
    std::swap(owns_, sl.owns_);
  }

  mutex_type* release() noexcept {
    mutex_type* result = pm_;
    pm_ = nullptr;
    owns_ = false;
    return result;
  }

  bool owns_lock() const noexcept { return owns_; }

  explicit operator bool() const noexcept { return owns_; }

  mutex_type* mutex() const noexcept { return pm_; }

 private:
  mutex_type* pm_ = nullptr;
  bool owns_ = false;
};

}  // namespace yamc

namespace std {

/// std::swap() specialization for yamc::shared_lock<Mutex> type
template <typename Mutex>
void swap(yamc::shared_lock<Mutex>& lhs,
          yamc::shared_lock<Mutex>& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace std

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_YAMC_YAMC_SHARED_LOCK_HPP_
