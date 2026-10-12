/*!
 * Copyright (c) 2018-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2018-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
/*!
 * This SWIG interface extension provides support to
 * the pointer manipulation methods present in the standard
 * SWIG wrappers, but with support for larger arrays.
 *
 * SWIG provides this in https://github.com/swig/swig/blob/master/Lib/carrays.i
 * but the standard methods only provide arrays with up to
 * max(int32_t) elements.
 *
 * The `long_array_functions` wrappers extend this
 * to arrays of size max(int64_t) instead of max(int32_t).
 */

%pointer_functions(uint8_t, bytep)
%pointer_functions(int, intp)
%pointer_functions(long, longp)
%pointer_functions(double, doublep)
%pointer_functions(float, floatp)
%pointer_functions(int64_t, int64_tp)
%pointer_functions(int32_t, int32_tp)
%pointer_functions(size_t, size_tp)

%pointer_cast(int64_t *, long *, int64_t_to_long_ptr)
%pointer_cast(int64_t *, double *, int64_t_to_double_ptr)
%pointer_cast(int32_t *, int *, int32_t_to_int_ptr)
%pointer_cast(long *, int64_t *, long_to_int64_t_ptr)
%pointer_cast(double *, int64_t *, double_to_int64_t_ptr)
%pointer_cast(int *, int32_t *, int_to_int32_t_ptr)

%pointer_cast(double *, void *, double_to_voidp_ptr)
%pointer_cast(float *, void *, float_to_voidp_ptr)
%pointer_cast(int *, void *, int_to_voidp_ptr)
%pointer_cast(uint8_t *, void *, byte_to_voidp_ptr)
%pointer_cast(int32_t *, void *, int32_t_to_voidp_ptr)
%pointer_cast(int64_t *, void *, int64_t_to_voidp_ptr)

%pointer_cast(void *, double **, void_to_doublep_ptr)

/* Custom pointer manipulation template */
%define %pointer_manipulation(TYPE, NAME)
%{
  static TYPE *new_##NAME() { %}
  %{  TYPE* NAME = new TYPE; return NAME; %}
  %{}

  static void delete_##NAME(TYPE *self) { %}
  %{  if (self) delete self; %}
  %{}
  %}

TYPE *new_##NAME();
void  delete_##NAME(TYPE *self);

%enddef

%define %pointer_dereference(TYPE, NAME)
%{
  static TYPE NAME ##_value(TYPE *self) {
    TYPE NAME = *self;
    return NAME;
  }
%}

TYPE NAME##_value(TYPE *self);

%enddef

%define %pointer_handle(TYPE, NAME)
%{
  static TYPE* NAME ##_handle() { %}
  %{ TYPE* NAME = new TYPE; *NAME = (TYPE)operator new(sizeof(int*)); return NAME; %}
  %{}
%}

TYPE *NAME##_handle();

%enddef

%define %long_array_functions(TYPE,NAME)
%{
  static TYPE *new_##NAME(int64_t nelements) { %}
  %{  return new TYPE[nelements](); %}
  %{}

  static void delete_##NAME(TYPE *ary) { %}
  %{  delete [] ary; %}
  %{}

  static TYPE NAME##_getitem(TYPE *ary, int64_t index) {
    return ary[index];
  }
  static void NAME##_setitem(TYPE *ary, int64_t index, TYPE value) {
    ary[index] = value;
  }
  %}

TYPE *new_##NAME(int64_t nelements);
void delete_##NAME(TYPE *ary);
TYPE NAME##_getitem(TYPE *ary, int64_t index);
void NAME##_setitem(TYPE *ary, int64_t index, TYPE value);

%enddef

/* Custom template for arrays of pointers */
%define %ptr_array_functions(TYPE,NAME)
%{
  static TYPE **new_##NAME(int64_t nelements) { %}
  %{  return new TYPE*[nelements](); %}
  %{}

  static void delete_##NAME(TYPE **ary) { %}
  %{  delete [] ary; %}
  %{}

  static TYPE *NAME##_getitem(TYPE **ary, int64_t index) {
    return ary[index];
  }
  static void NAME##_setitem(TYPE **ary, int64_t index, TYPE *value) {
    ary[index] = value;
  }
  %}

TYPE **new_##NAME(int64_t nelements);
void delete_##NAME(TYPE **ary);
TYPE *NAME##_getitem(TYPE **ary, int64_t index);
void NAME##_setitem(TYPE **ary, int64_t index, TYPE *value);

%enddef

%long_array_functions(uint8_t, byteArray)
%long_array_functions(double, doubleArray)
%long_array_functions(float, floatArray)
%long_array_functions(int32_t, intArray)
%long_array_functions(int64_t, longArray)

%ptr_array_functions(void, voidPtrArray)
%ptr_array_functions(double, doublePtrArray)
%ptr_array_functions(int, intPtrArray)

%pointer_manipulation(void*, voidpp)

/* Allow dereferencing of void** to void* */
%pointer_dereference(void*, voidpp)

/* Allow retrieving handle to void** */
%pointer_handle(void*, voidpp)
