/*!
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Oliver Borchert
 */

#ifndef LIGHTGBM_SRC_ARROW_ARRAY_HPP_
#define LIGHTGBM_SRC_ARROW_ARRAY_HPP_

#include <LightGBM/arrow.h>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <nanoarrow/nanoarrow.hpp>

namespace LightGBM {

/* ------------------------------------------- UTILS ------------------------------------------- */

inline struct ArrowSchemaView MakeSchemaView(const struct ArrowSchema* schema) {
  struct ArrowSchemaView view;
  struct ArrowError error;
  auto ret = ArrowSchemaViewInit(&view, schema, &error);
  if (ret != NANOARROW_OK) {
    throw nanoarrow::Exception("Failed to initialize ArrowSchemaView: " +
                               std::string(ArrowErrorMessage(&error)));
  }
  return view;
}

/* --------------------------------------------------------------------------------------------- */
/*                                         CHUNKED ARRAY                                         */
/* --------------------------------------------------------------------------------------------- */

class ArrowChunkedArray {
  enum ArrowType type_;
  nanoarrow::UniqueSchema schema_;
  std::vector<nanoarrow::UniqueArray> chunks_;

 public:
  class View;
  template <typename ArrowT, typename OutputT>
  class Visitor;
  template <typename ArrowT, typename OutputT>
  class Iterator;

  /**
   * @brief Construct a new chunked Arrow array from an array stream.
   * The stream is consumed and the chunked Arrow array takes ownership of the schema and all
   * chunks. Upon destruction, the release callback is called for the schema and all chunks.
   *
   * @param stream The Arrow array stream to consume.
   */
  explicit ArrowChunkedArray(ArrowArrayStream* stream) {
    nanoarrow::UniqueArrayStream stream_(stream);

    // Extract the schema
    auto ret = stream_->get_schema(stream_.get(), schema_.get());
    if (ret != NANOARROW_OK) {
      throw nanoarrow::Exception("Failed to get schema from Arrow array stream: " +
                                 std::string(stream_->get_last_error(stream_.get())));
    }

    // Turn the schema into a type
    type_ = MakeSchemaView(schema_.get()).type;

    // Extract all chunks
    while (true) {
      nanoarrow::UniqueArray chunk;
      auto ret = stream_->get_next(stream_.get(), chunk.get());
      if (ret != NANOARROW_OK) {
        throw nanoarrow::Exception("Failed to get next chunk from Arrow array stream: " +
                                   std::string(stream_->get_last_error(stream_.get())));
      }
      if (chunk->release == nullptr) break;
      chunks_.emplace_back(std::move(chunk));
    }
  }

  /**
   * @brief Construct a new chunked Arrow array from a list of Arrow arrays and a schema.
   * The chunked Arrow array takes ownership of the schema and all chunks. Upon destruction,
   * the release callback is called for the schema and all chunks.
   *
   * @param n_chunks The number of Arrow arrays.
   * @param chunks Pointer to the list of Arrow arrays.
   * @param schema Pointer to the schema of all Arrow arrays.
   */
  explicit ArrowChunkedArray(int64_t n_chunks, struct ArrowArray* chunks,
                             struct ArrowSchema* schema) {
    // Take ownership of schema
    schema_ = nanoarrow::UniqueSchema(schema);
    type_ = MakeSchemaView(schema_.get()).type;

    // Take ownership of chunks
    chunks_.reserve(n_chunks);
    for (int64_t i = 0; i < n_chunks; ++i) {
      chunks_.emplace_back(&chunks[i]);
    }
  }

  /**
   * @brief Whether the chunked array is a struct and has multiple fields.
   * A struct chunked array is typically interpreted as a table.
   */
  bool is_struct() const { return type_ == NANOARROW_TYPE_STRUCT; }

  /**
   * @brief Get the length of the chunked array as the sum of all chunk lengths.
   *
   * @return int64_t The total number of elements.
   */
  int64_t get_length() const {
    int64_t length = 0;
    for (const auto& chunk : chunks_) {
      length += chunk->length;
    }
    return length;
  }

  /**
   * @brief Get the number of fields in the chunked array.
   *
   * @return int64_t The number of fields.
   */
  int64_t get_num_fields() const {
    if (!is_struct()) {
      Log::Fatal("Expected struct type for array, got %s", ArrowTypeString(type_));
    }
    return schema_.get()->n_children;
  }

  /**
   * @brief Obtain a view on the chunked array to visit its values.
   *
   * @return View The view on the array.
   */
  View view() const {
    std::vector<const struct ArrowArray*> chunk_ptrs;
    chunk_ptrs.reserve(chunks_.size());
    for (const auto& chunk : chunks_) {
      // Skip empty chunks to avoid additional complexity in the iterator
      if (chunk->length == 0) continue;
      chunk_ptrs.push_back(chunk.get());
    }
    return View(type_, schema_.get(), std::move(chunk_ptrs));
  }

  /* ------------------------------------------------------------------------------------------- */
  /*                                             VIEW                                            */
  /* ------------------------------------------------------------------------------------------- */

  class View {
    friend class ArrowChunkedArray;

    enum ArrowType type_;
    const struct ArrowSchema* schema_;
    std::vector<const struct ArrowArray*> chunks_;

    View(enum ArrowType type, const struct ArrowSchema* schema,
         std::vector<const struct ArrowArray*> chunks)
        : type_(type), schema_(schema), chunks_(std::move(chunks)) {}

   public:
    explicit View(std::vector<View> views) {
      type_ = views[0].type_;
      schema_ = views[0].schema_;
      for (auto it = views.begin(), end = views.end(); it != end; ++it) {
        if (it->type_ != type_) {
          Log::Fatal("All views must have the same type, but got %s and %s",
                     ArrowTypeString(it->type_), ArrowTypeString(type_));
        }
        chunks_.insert(chunks_.end(), it->chunks_.begin(), it->chunks_.end());
      }
    }

    /**
     * @brief Obtain a view on the field at the given index.
     * This method assumes that the view has a type "struct".
     *
     * @param field_idx The index of the field to view.
     * @return View A view on the field at the given index.
     */
    View field(int64_t field_idx) const {
      if (type_ != NANOARROW_TYPE_STRUCT) {
        Log::Fatal("Expected struct type for array, got %s", ArrowTypeString(type_));
      }

      std::vector<const struct ArrowArray*> chunk_ptrs;
      chunk_ptrs.reserve(chunks_.size());
      for (const auto& chunk : chunks_) {
        chunk_ptrs.push_back(chunk->children[field_idx]);
      }

      auto type = MakeSchemaView(schema_->children[field_idx]).type;
      return View(type, schema_->children[field_idx], std::move(chunk_ptrs));
    }

    /**
     * @brief Visit the chunked array with a visitor created from the schema type.
     * The visitor allows accessing all values from the chunked array, casting to the desired
     * output type.
     *
     * @tparam OutputT The desired output type.
     * @tparam F The type of the visitor function, which must be invocable with a
     *           `Visitor<ArrowT, OutputT>`
     * @param f The visitor function to invoke with the created visitor.
     * @return decltype(auto) The result of invoking the visitor function with the created visitor.
     */
    template <typename OutputT, typename F>
    decltype(auto) visit(F&& f) const {
      // Switch on the schema type and construct the appropriate visitor
      switch (type_) {
        case NANOARROW_TYPE_INT8:
          return f(Visitor<int8_t, OutputT>(chunks_));
        case NANOARROW_TYPE_INT16:
          return f(Visitor<int16_t, OutputT>(chunks_));
        case NANOARROW_TYPE_INT32:
          return f(Visitor<int32_t, OutputT>(chunks_));
        case NANOARROW_TYPE_INT64:
          return f(Visitor<int64_t, OutputT>(chunks_));
        case NANOARROW_TYPE_UINT8:
          return f(Visitor<uint8_t, OutputT>(chunks_));
        case NANOARROW_TYPE_UINT16:
          return f(Visitor<uint16_t, OutputT>(chunks_));
        case NANOARROW_TYPE_UINT32:
          return f(Visitor<uint32_t, OutputT>(chunks_));
        case NANOARROW_TYPE_UINT64:
          return f(Visitor<uint64_t, OutputT>(chunks_));
        case NANOARROW_TYPE_FLOAT:
          return f(Visitor<float, OutputT>(chunks_));
        case NANOARROW_TYPE_DOUBLE:
          return f(Visitor<double, OutputT>(chunks_));
        case NANOARROW_TYPE_BOOL:
          return f(Visitor<bool, OutputT>(chunks_));
        default:
          Log::Fatal("Unsupported Arrow type: %s", ArrowTypeString(type_));
      }
    }
  };

  /* ------------------------------------------------------------------------------------------- */
  /*                                           VISITOR                                           */
  /* ------------------------------------------------------------------------------------------- */

  template <typename ArrowT, typename OutputT>
  class Visitor {
    friend class View;
    friend class Iterator<ArrowT, OutputT>;

    std::vector<const struct ArrowArray*> chunks_;
    std::vector<int64_t> chunk_offsets_;

    explicit Visitor(std::vector<const struct ArrowArray*> chunks) : chunks_(chunks) {
      // Derive chunk offsets
      chunk_offsets_.reserve(chunks.size() + 1);
      chunk_offsets_.push_back(0);
      for (const auto& chunk : chunks) {
        chunk_offsets_.push_back(chunk_offsets_.back() + chunk->length);
      }
    }

    ArrowT is_valid(int64_t chunk_idx, int64_t element_idx) const {
      auto arr = chunks_[chunk_idx];
      return arr->buffers[0] == nullptr ||
             ArrowBitGet(static_cast<const uint8_t*>(arr->buffers[0]), arr->offset + element_idx);
    }

    ArrowT get(int64_t chunk_idx, int64_t element_idx) const {
      auto arr = chunks_[chunk_idx];
      auto idx = arr->offset + element_idx;
      if constexpr (std::is_same_v<ArrowT, bool>) {
        return ArrowBitGet(static_cast<const uint8_t*>(arr->buffers[1]), idx);
      } else {
        return static_cast<const ArrowT*>(arr->buffers[1])[idx];
      }
    }

   public:
    Iterator<ArrowT, OutputT> begin() const { return Iterator<ArrowT, OutputT>(*this, 0, 0); }
    Iterator<ArrowT, OutputT> end() const {
      return Iterator<ArrowT, OutputT>(*this, chunks_.size(), 0);
    }
  };

  /* ------------------------------------------------------------------------------------------- */
  /*                                           ITERATOR                                          */
  /* ------------------------------------------------------------------------------------------- */

  template <typename ArrowT, typename OutputT>
  class Iterator {
    friend class Visitor<ArrowT, OutputT>;

    const Visitor<ArrowT, OutputT>& visitor_;
    int64_t chunk_idx_;
    int64_t element_idx_;

    Iterator(const Visitor<ArrowT, OutputT>& visitor, int64_t chunk_idx, int64_t element_idx)
        : visitor_(visitor), chunk_idx_(chunk_idx), element_idx_(element_idx) {}

    int64_t full_offset() const { return visitor_.chunk_offsets_[chunk_idx_] + element_idx_; }

   public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = int64_t;
    using value_type = OutputT;
    using pointer = value_type*;
    using reference = value_type&;

    /* --------------------------------------- OPERATORS --------------------------------------- */

   public:
    Iterator<ArrowT, OutputT>& operator++() {
      if (element_idx_ + 1 >= visitor_.chunks_[chunk_idx_]->length) {
        element_idx_ = 0;
        chunk_idx_++;
      } else {
        element_idx_++;
      }
      return *this;
    }

    Iterator<ArrowT, OutputT>& operator+=(int64_t c) {
      while (element_idx_ + c >= visitor_.chunks_[chunk_idx_]->length) {
        c -= visitor_.chunks_[chunk_idx_]->length - element_idx_;
        element_idx_ = 0;
        chunk_idx_++;
      }
      element_idx_ += c;
      return *this;
    }

    Iterator<ArrowT, OutputT>& operator--() {
      if (element_idx_ == 0) {
        chunk_idx_--;
        element_idx_ = visitor_.chunks_[chunk_idx_]->length - 1;
      } else {
        element_idx_--;
      }
      return *this;
    }

    Iterator<ArrowT, OutputT>& operator-=(int64_t c) {
      while (c > element_idx_) {
        c -= element_idx_ + 1;
        chunk_idx_--;
        element_idx_ = visitor_.chunks_[chunk_idx_]->length - 1;
      }
      element_idx_ -= c;
      return *this;
    }

    friend int64_t operator-(const Iterator<ArrowT, OutputT>& a,
                             const Iterator<ArrowT, OutputT>& b) {
      auto full_offset_a = a.full_offset();
      auto full_offset_b = b.full_offset();
      return full_offset_a - full_offset_b;
    }

    friend bool operator==(const Iterator<ArrowT, OutputT>& a,
                           const Iterator<ArrowT, OutputT>& b) {
      return a.chunk_idx_ == b.chunk_idx_ && a.element_idx_ == b.element_idx_;
    }

    friend bool operator!=(const Iterator<ArrowT, OutputT>& a,
                           const Iterator<ArrowT, OutputT>& b) {
      return !(a == b);
    }

    /* ----------------------------------------- VALUE ----------------------------------------- */

   private:
    static constexpr OutputT null_default() {
      if constexpr (std::is_floating_point_v<OutputT>) {
        return std::numeric_limits<OutputT>::quiet_NaN();
      } else {
        return OutputT{0};
      }
    }

    OutputT get(int64_t chunk_idx, int64_t element_idx) const {
      if (visitor_.is_valid(chunk_idx, element_idx)) {
        return static_cast<OutputT>(visitor_.get(chunk_idx, element_idx));
      } else {
        return null_default();
      }
    }

   public:
    OutputT operator*() const { return this->get(chunk_idx_, element_idx_); }

    OutputT operator[](int64_t c) const {
      if (visitor_.chunks_.size() == 1) {
        return this->get(0, c);
      }
      auto it =
          std::upper_bound(visitor_.chunk_offsets_.begin(), visitor_.chunk_offsets_.end(), c);
      auto chunk_idx = std::distance(visitor_.chunk_offsets_.begin(), it) - 1;
      auto element_idx = c - visitor_.chunk_offsets_[chunk_idx];
      return this->get(chunk_idx, element_idx);
    }
  };
};

};  // namespace LightGBM

#endif  // LIGHTGBM_SRC_ARROW_ARRAY_HPP_
