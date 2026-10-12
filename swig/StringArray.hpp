/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Alberto Ferreira
 */
#ifndef LIGHTGBM_SWIG_STRINGARRAY_HPP_
#define LIGHTGBM_SWIG_STRINGARRAY_HPP_

#include <algorithm>
#include <new>
#include <string>
#include <vector>

/**
 * Container that manages an array of fixed-length strings.
 *
 * To be compatible with SWIG's `various.i` extension module,
 * the array of pointers to char* must be NULL-terminated:
 *   [char*, char*, char*, ..., NULL]
 * This implies that the length of this array is bigger
 * by 1 element than the number of char* it stores.
 * I.e., _num_elements == _array.size()-1
 *
 * The class also takes care of allocation of the underlying
 * char* memory.
 */
class StringArray {
 public:
    StringArray(size_t num_elements, size_t string_size)
      : _string_size(string_size),
        _array(num_elements + 1, nullptr) {
        _allocate_strings(num_elements, string_size);
    }

    ~StringArray() {
        _release_strings();
    }

    /**
     * Returns the pointer to the raw array.
     * Notice its size is greater than the number of stored strings by 1.
     *
     * @return char** pointer to raw data (null-terminated).
     */
    char **data() noexcept {
        return _array.data();
    }

    /**
     * Return char* from the array of size _string_size+1.
     * Notice the last element in _array is already
     * considered out of bounds.
     *
     * @param index Index of the element to retrieve.
     * @return pointer or nullptr if index is out of bounds.
     */
    char *getitem(size_t index) noexcept {
        if (_in_bounds(index))
            return _array[index];
        else
            return nullptr;
    }

    /**
     * Safely copies the full content data
     * into one of the strings in the array.
     * If that is not possible, returns error (-1).
     *
     * @param index index of the string in the array.
     * @param content content to store
     *
     * @return In case index results in out of bounds access,
     * or content + 1 (null-terminator byte) doesn't fit
     * into the target string (_string_size), it errors out
     * and returns -1.
     */
    int setitem(size_t index, const std::string &content) noexcept {
        if (_in_bounds(index) && content.size() < _string_size) {
            std::strcpy(_array[index], content.c_str());  // NOLINT
            return 0;
        } else {
            return -1;
        }
    }

    /**
     * @return number of stored strings.
     */
    size_t get_num_elements() noexcept {
        return _array.size() - 1;
    }

 private:
    /**
     * Returns true if and only if within bounds.
     * Notice that it excludes the last element of _array (NULL).
     *
     * @param index index of the element
     * @return bool true if within bounds
     */
    bool _in_bounds(size_t index) noexcept {
        return index < get_num_elements();
    }

    /**
     * Allocate an array of fixed-length strings.
     *
     * Since a NULL-terminated array is required by SWIG's `various.i`,
     * the size of the array is actually `num_elements + 1` but only
     * num_elements are filled.
     *
     * @param num_elements Number of strings to store in the array.
     * @param string_size The size of each string in the array.
     */
    void _allocate_strings(size_t num_elements, size_t string_size) {
        for (size_t i = 0; i < num_elements; ++i) {
            // Leave space for \0 terminator:
            _array[i] = new (std::nothrow) char[string_size + 1];

            // Check memory allocation:
            if (!_array[i]) {
                _release_strings();
                throw std::bad_alloc();
            }
        }
    }

    /**
     * Deletes the allocated strings.
     */
    void _release_strings() noexcept {
        std::for_each(_array.begin(), _array.end(), [](char* c) { delete[] c; });
    }

    const size_t _string_size;
    std::vector<char*> _array;
};

#endif  // LIGHTGBM_SWIG_STRINGARRAY_HPP_
