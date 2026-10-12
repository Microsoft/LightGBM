/*!
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Oliver Borchert
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_ARROW_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_ARROW_H_

// The C data interface (ARROW_C_DATA_INTERFACE) is taken from
// https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions

// The C stream interface (ARROW_C_STREAM_INTERFACE) is taken from
// https://arrow.apache.org/docs/format/CStreamInterface.html#structure-definition

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  // Release callback
  void (*release)(struct ArrowSchema*);
  // Opaque producer-specific data
  void* private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  // Release callback
  void (*release)(struct ArrowArray*);
  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
  // Callbacks providing stream functionality
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
  const char* (*get_last_error)(struct ArrowArrayStream*);

  // Release callback
  void (*release)(struct ArrowArrayStream*);

  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE

#ifdef __cplusplus
}
#endif
#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_ARROW_H_
