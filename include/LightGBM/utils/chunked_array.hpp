/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Alberto Ferreira
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_CHUNKED_ARRAY_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_CHUNKED_ARRAY_HPP_

#include <LightGBM/utils/log.h>

#include <stdint.h>

#include <algorithm>
#include <new>
#include <vector>


namespace LightGBM {

/**
 * Container that manages a dynamic array of fixed-length chunks.
 *
 * The class also takes care of allocation & release of the underlying
 * memory. It can be used with either a high or low-level API.
 *
 * The high-level API allocates chunks as needed, manages addresses automatically and keeps
 * track of number of inserted elements, but is not thread-safe (this is ok as usually input is a streaming iterator).
 * For parallel input sources the low-level API must be used.
 *
 * Note: When using this for `LGBM_DatasetCreateFromMats` use a
 *       chunk_size multiple of #num_cols for your dataset, so each chunk
 *       contains "complete" instances.
 *
 * === High-level insert API intro ===
 *
 * The easiest way to use is:
 *  0. ChunkedArray(chunk_size)  # Choose appropriate size
 *  1. add(value)                # as many times as you want (will generate chunks as needed)
 *  2. data() or void_data()     # retrieves a T** or void** pointer (useful for `LGBM_DatasetCreateFromMats`).
 *
 * Useful query methods (all O(1)):
 *  - get_add_count()   # total count of added elements.
 *  - get_chunks_count()  # how many chunks are currently allocated.
 *  - get_current_chunk_added_count()  # for the last add() chunk, how many items there are.
 *  - get_chunk_size()    # get constant chunk_size from constructor call.
 *
 *  With those you can generate int32_t sizes[]. Last chunk can be smaller than chunk_size, so, for any i:
 *    - sizes[i<last] = get_chunk_size()
 *    - sizes[i==last] = get_add_count()
 *
 *
 * === Low-level insert API intro ===
 *
 * For advanced usage - useful for inserting in parallel - one can also:
 *  1. call new_chunk() at any time for as many chunks as needed.  (thread-UNsafe)
 *  2. call setitem(chunk, idx, value) to insert each value.       (thread-safe)
 *
 */
template <class T>
class ChunkedArray {
 public:
    explicit ChunkedArray(size_t chunk_size)
      : _chunk_size(chunk_size), _last_chunk_idx(0), _last_idx_in_last_chunk(0) {
      if (chunk_size == 0) {
        Log::Fatal("ChunkedArray chunk size must be larger than 0!");
      }
       new_chunk();
    }

    ~ChunkedArray() {
        release();
    }

    /**
     * Adds a value to the chunks sequentially.
     * If the last chunk is full it creates a new one and appends to it.
     *
     * @param value value to insert.
     */
    void add(T value) {
        if (!within_bounds(_last_chunk_idx, _last_idx_in_last_chunk)) {
            new_chunk();
            ++_last_chunk_idx;
            _last_idx_in_last_chunk = 0;
        }

        CHECK_EQ(setitem(_last_chunk_idx, _last_idx_in_last_chunk, value), 0);
        ++_last_idx_in_last_chunk;
    }

    /**
     * @return Number of add() calls.
     */
    size_t get_add_count() const {
        return _last_chunk_idx * _chunk_size + _last_idx_in_last_chunk;
    }

    /**
     * @return Number of allocated chunks.
     */
    size_t get_chunks_count() const {
        return _chunks.size();
    }

    /**
     * @return Number of elemends add()'ed in the last chunk.
     */
    size_t get_last_chunk_add_count() const {
        return _last_idx_in_last_chunk;
    }

    /**
     * Getter for the chunk size set at the constructor.
     *
     * @return Return the size of chunks.
     */
    size_t get_chunk_size() const {
        return _chunk_size;
    }

    /**
     * Returns the pointer to the raw chunks data.
     *
     * @return T** pointer to raw data.
     */
    T **data() noexcept {
        return _chunks.data();
    }

    /**
     * Returns the pointer to the raw chunks data, but cast to void**.
     * This is so ``LGBM_DatasetCreateFromMats`` accepts it.
     *
     * @return void** pointer to raw data.
     */
    void **data_as_void() noexcept {
        return reinterpret_cast<void**>(_chunks.data());
    }

    /**
     * Coalesces (copies chunked data) to a contiguous array of the same type.
     * It assumes that ``other`` has enough space to receive that data.
     *
     * @param other array with elements T of size >= this->get_add_count().
     * @param all_valid_addresses
     *            If true exports values from all valid addresses independently of add() count.
     *            Otherwise, exports only up to `get_add_count()` addresses.
     */
    void coalesce_to(T *other, bool all_valid_addresses = false) const {
        const size_t full_chunks = this->get_chunks_count() - 1;

        // Copy full chunks:
        size_t i = 0;
        for (size_t chunk = 0; chunk < full_chunks; ++chunk) {
            T* chunk_ptr = _chunks[chunk];
            for (size_t in_chunk_idx = 0; in_chunk_idx < _chunk_size; ++in_chunk_idx) {
                other[i++] = chunk_ptr[in_chunk_idx];
            }
        }
        // Copy filled values from last chunk only:
        const size_t last_chunk_elems_to_copy = all_valid_addresses ? _chunk_size : this->get_last_chunk_add_count();
        T* chunk_ptr = _chunks[full_chunks];
        for (size_t in_chunk_idx = 0; in_chunk_idx < last_chunk_elems_to_copy; ++in_chunk_idx) {
            other[i++] = chunk_ptr[in_chunk_idx];
        }
    }

    /**
     * Return value from array of chunks.
     *
     * @param chunk_index index of the chunk
     * @param index_within_chunk index within chunk
     * @param on_fail_value sentinel value. If out of bounds returns that value.
     *
     * @return pointer or nullptr if index is out of bounds.
     */
    T getitem(size_t chunk_index, size_t index_within_chunk, T on_fail_value) const noexcept {
        if (within_bounds(chunk_index, index_within_chunk))
            return _chunks[chunk_index][index_within_chunk];
        else
            return on_fail_value;
    }

    /**
     * Sets the value at a specific address in one of the chunks.
     *
     * @param chunk_index index of the chunk
     * @param index_within_chunk index within chunk
     * @param value value to store
     *
     * @return 0 = success, -1 = out of bounds access.
     */
    int setitem(size_t chunk_index, size_t index_within_chunk, T value) noexcept {
        if (within_bounds(chunk_index, index_within_chunk)) {
            _chunks[chunk_index][index_within_chunk] = value;
            return 0;
        } else {
            return -1;
        }
    }

    /**
     * To reset storage call this.
     * Will release existing resources and prepare for reuse.
     */
    void clear() noexcept {
        release();
        new_chunk();
    }

    /**
     * Deletes all the allocated chunks.
     * Do not use container after this! See ``clear()`` instead.
     */
    void release() noexcept {
        std::for_each(_chunks.begin(), _chunks.end(), [](T* c) { delete[] c; });
        _chunks.clear();
        _chunks.shrink_to_fit();
        _last_chunk_idx = 0;
        _last_idx_in_last_chunk = 0;
    }

    /**
     * As the array is dynamic, checks whether a given address is currently within bounds.
     *
     * @param chunk_index index of the chunk
     * @param index_within_chunk index within that chunk
     * @return true if that chunk is already allocated and index_within_chunk < chunk size.
     */
    inline bool within_bounds(size_t chunk_index, size_t index_within_chunk) const {
        return (chunk_index < _chunks.size()) && (index_within_chunk < _chunk_size);
    }

    /**
     * Adds a new chunk to the array of chunks. Not thread-safe.
     */
    void new_chunk() {
        _chunks.push_back(new (std::nothrow) T[_chunk_size]);

        // Check memory allocation success:
        if (!_chunks[_chunks.size() - 1]) {
            release();
            Log::Fatal("Memory exhausted! Cannot allocate new ChunkedArray chunk.");
        }
    }

 private:
    const size_t _chunk_size;
    std::vector<T*> _chunks;

    // For the add() interface & some of the get_*() queries:
    size_t _last_chunk_idx;  //<! Index of chunks
    size_t _last_idx_in_last_chunk;  //<! Index within chunk
};


}  // namespace LightGBM

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_UTILS_CHUNKED_ARRAY_HPP_
