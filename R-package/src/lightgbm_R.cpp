/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "lightgbm_R.h"

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>

#include <R_ext/Rdynload.h>
#include <R_ext/Altrep.h>

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif

#ifndef R_USE_C99_IN_CXX
#define R_USE_C99_IN_CXX
#endif

#include <R_ext/Error.h>

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include <type_traits>

R_altrep_class_t lgb_altrepped_char_vec;
R_altrep_class_t lgb_altrepped_int_arr;
R_altrep_class_t lgb_altrepped_dbl_arr;

template <class T>
void delete_cpp_array(SEXP R_ptr) {
  T *ptr_to_cpp_obj = static_cast<T*>(R_ExternalPtrAddr(R_ptr));
  delete[] ptr_to_cpp_obj;
  R_ClearExternalPtr(R_ptr);
}

void delete_cpp_char_vec(SEXP R_ptr) {
  std::vector<char> *ptr_to_cpp_obj = static_cast<std::vector<char>*>(R_ExternalPtrAddr(R_ptr));
  delete ptr_to_cpp_obj;
  R_ClearExternalPtr(R_ptr);
}

// Note: MSVC has issues with Altrep classes, so they are disabled for it.
// See: https://github.com/lightgbm-org/LightGBM/pull/6213#issuecomment-2111025768
#ifdef _MSC_VER
#  define LGB_NO_ALTREP
#endif

#ifndef LGB_NO_ALTREP
SEXP make_altrepped_raw_vec(void *void_ptr) {
  std::unique_ptr<std::vector<char>> *ptr_to_cpp_vec = static_cast<std::unique_ptr<std::vector<char>>*>(void_ptr);
  SEXP R_ptr = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  SEXP R_raw = Rf_protect(R_new_altrep(lgb_altrepped_char_vec, R_NilValue, R_NilValue));

  R_SetExternalPtrAddr(R_ptr, ptr_to_cpp_vec->get());
  R_RegisterCFinalizerEx(R_ptr, delete_cpp_char_vec, TRUE);
  ptr_to_cpp_vec->release();

  R_set_altrep_data1(R_raw, R_ptr);
  Rf_unprotect(2);
  return R_raw;
}
#else
SEXP make_r_raw_vec(void *void_ptr) {
  std::unique_ptr<std::vector<char>> *ptr_to_cpp_vec = static_cast<std::unique_ptr<std::vector<char>>*>(void_ptr);
  R_xlen_t len = ptr_to_cpp_vec->get()->size();
  SEXP out = Rf_protect(Rf_allocVector(RAWSXP, len));
  std::copy(ptr_to_cpp_vec->get()->begin(), ptr_to_cpp_vec->get()->end(), reinterpret_cast<char*>(RAW(out)));
  Rf_unprotect(1);
  return out;
}
#define make_altrepped_raw_vec make_r_raw_vec
#endif

std::vector<char>* get_ptr_from_altrepped_raw(SEXP R_raw) {
  return static_cast<std::vector<char>*>(R_ExternalPtrAddr(R_altrep_data1(R_raw)));
}

R_xlen_t get_altrepped_raw_len(SEXP R_raw) {
  return get_ptr_from_altrepped_raw(R_raw)->size();
}

const void* get_altrepped_raw_dataptr_or_null(SEXP R_raw) {
  return get_ptr_from_altrepped_raw(R_raw)->data();
}

void* get_altrepped_raw_dataptr(SEXP R_raw, Rboolean writable) {
  return get_ptr_from_altrepped_raw(R_raw)->data();
}

#ifndef LGB_NO_ALTREP
template <class T>
R_altrep_class_t get_altrep_class_for_type() {
  if (std::is_same<T, double>::value) {
    return lgb_altrepped_dbl_arr;
  } else {
    return lgb_altrepped_int_arr;
  }
}
#else
template <class T>
SEXPTYPE get_sexptype_class_for_type() {
  if (std::is_same<T, double>::value) {
    return REALSXP;
  } else {
    return INTSXP;
  }
}

template <class T>
T* get_r_vec_ptr(SEXP x) {
  if (std::is_same<T, double>::value) {
    return static_cast<T*>(static_cast<void*>(REAL(x)));
  } else {
    return static_cast<T*>(static_cast<void*>(INTEGER(x)));
  }
}
#endif

template <class T>
struct arr_and_len {
  T *arr;
  int64_t len;
};

#ifndef LGB_NO_ALTREP
template <class T>
SEXP make_altrepped_vec_from_arr(void *void_ptr) {
  T *arr = static_cast<arr_and_len<T>*>(void_ptr)->arr;
  uint64_t len = static_cast<arr_and_len<T>*>(void_ptr)->len;
  SEXP R_ptr = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  SEXP R_len = Rf_protect(Rf_allocVector(REALSXP, 1));
  SEXP R_vec = Rf_protect(R_new_altrep(get_altrep_class_for_type<T>(), R_NilValue, R_NilValue));

  REAL(R_len)[0] = static_cast<double>(len);
  R_SetExternalPtrAddr(R_ptr, arr);
  R_RegisterCFinalizerEx(R_ptr, delete_cpp_array<T>, TRUE);

  R_set_altrep_data1(R_vec, R_ptr);
  R_set_altrep_data2(R_vec, R_len);
  Rf_unprotect(3);
  return R_vec;
}
#else
template <class T>
SEXP make_R_vec_from_arr(void *void_ptr) {
  T *arr = static_cast<arr_and_len<T>*>(void_ptr)->arr;
  uint64_t len = static_cast<arr_and_len<T>*>(void_ptr)->len;
  SEXP out = Rf_protect(Rf_allocVector(get_sexptype_class_for_type<T>(), len));
  std::copy(arr, arr + len, get_r_vec_ptr<T>(out));
  Rf_unprotect(1);
  return out;
}
#define make_altrepped_vec_from_arr make_R_vec_from_arr
#endif

R_xlen_t get_altrepped_vec_len(SEXP R_vec) {
  return static_cast<R_xlen_t>(Rf_asReal(R_altrep_data2(R_vec)));
}

const void* get_altrepped_vec_dataptr_or_null(SEXP R_vec) {
  return R_ExternalPtrAddr(R_altrep_data1(R_vec));
}

void* get_altrepped_vec_dataptr(SEXP R_vec, Rboolean writable) {
  return R_ExternalPtrAddr(R_altrep_data1(R_vec));
}

#define COL_MAJOR (0)

#define MAX_LENGTH_ERR_MSG 1024
char R_errmsg_buffer[MAX_LENGTH_ERR_MSG];
struct LGBM_R_ErrorClass { SEXP cont_token; };
void LGBM_R_save_exception_msg(const std::exception &err);
void LGBM_R_save_exception_msg(const std::string &err);

#define R_API_BEGIN() \
  try {
#define R_API_END() } \
  catch(LGBM_R_ErrorClass &cont) { R_ContinueUnwind(cont.cont_token); } \
  catch(std::exception& ex) { LGBM_R_save_exception_msg(ex); } \
  catch(std::string& ex) { LGBM_R_save_exception_msg(ex); } \
  catch(...) { Rf_error("unknown exception"); } \
  Rf_error("%s", R_errmsg_buffer); \
  return R_NilValue; /* <- won't be reached */

#define CHECK_CALL(x) \
  if ((x) != 0) { \
    throw std::runtime_error(LGBM_GetLastError()); \
  }

// These are helper functions to allow doing a stack unwind
// after an R allocation error, which would trigger a long jump.
void LGBM_R_save_exception_msg(const std::exception &err) {
  std::snprintf(R_errmsg_buffer, MAX_LENGTH_ERR_MSG, "%s\n", err.what());
}

void LGBM_R_save_exception_msg(const std::string &err) {
  std::snprintf(R_errmsg_buffer, MAX_LENGTH_ERR_MSG, "%s\n", err.c_str());
}

SEXP wrapped_R_string(void *len) {
  return Rf_allocVector(STRSXP, *(reinterpret_cast<R_xlen_t*>(len)));
}

SEXP wrapped_R_raw(void *len) {
  return Rf_allocVector(RAWSXP, *(reinterpret_cast<R_xlen_t*>(len)));
}

SEXP wrapped_R_int(void *len) {
  return Rf_allocVector(INTSXP, *(reinterpret_cast<R_xlen_t*>(len)));
}

SEXP wrapped_R_real(void *len) {
  return Rf_allocVector(REALSXP, *(reinterpret_cast<R_xlen_t*>(len)));
}

SEXP wrapped_Rf_mkChar(void *txt) {
  return Rf_mkChar(reinterpret_cast<char*>(txt));
}

void throw_R_memerr(void *ptr_cont_token, Rboolean jump) {
  if (jump) {
    LGBM_R_ErrorClass err{*(reinterpret_cast<SEXP*>(ptr_cont_token))};
    throw err;
  }
}

SEXP safe_R_string(R_xlen_t len, SEXP *cont_token) {
  return R_UnwindProtect(wrapped_R_string, reinterpret_cast<void*>(&len), throw_R_memerr, cont_token, *cont_token);
}

SEXP safe_R_raw(R_xlen_t len, SEXP *cont_token) {
  return R_UnwindProtect(wrapped_R_raw, reinterpret_cast<void*>(&len), throw_R_memerr, cont_token, *cont_token);
}

SEXP safe_R_int(R_xlen_t len, SEXP *cont_token) {
  return R_UnwindProtect(wrapped_R_int, reinterpret_cast<void*>(&len), throw_R_memerr, cont_token, *cont_token);
}

SEXP safe_R_real(R_xlen_t len, SEXP *cont_token) {
  return R_UnwindProtect(wrapped_R_real, reinterpret_cast<void*>(&len), throw_R_memerr, cont_token, *cont_token);
}

SEXP safe_R_mkChar(char *txt, SEXP *cont_token) {
  return R_UnwindProtect(wrapped_Rf_mkChar, reinterpret_cast<void*>(txt), throw_R_memerr, cont_token, *cont_token);
}

using LightGBM::Common::Split;
using LightGBM::Log;

SEXP LGBM_HandleIsNull_R(SEXP handle) {
  return Rf_ScalarLogical(R_ExternalPtrAddr(handle) == NULL);
}

void _DatasetFinalizer(SEXP handle) {
  LGBM_DatasetFree_R(handle);
}

SEXP LGBM_NullBoosterHandleError_R() {
  Rf_error(
      "Attempting to use a Booster which no longer exists and/or cannot be restored. "
      "This can happen if the Booster's finalizer was called "
      "or if this Booster was saved through saveRDS() using 'serializable=FALSE'.");
  return R_NilValue;
}

void _AssertBoosterHandleNotNull(SEXP handle) {
  if (Rf_isNull(handle) || !R_ExternalPtrAddr(handle)) {
    LGBM_NullBoosterHandleError_R();
  }
}

void _AssertDatasetHandleNotNull(SEXP handle) {
  if (Rf_isNull(handle) || !R_ExternalPtrAddr(handle)) {
    Rf_error(
      "Attempting to use a Dataset which no longer exists. "
      "This can happen if the Dataset's finalizer was called or if this Dataset was saved with saveRDS(). "
      "To avoid this error in the future, use lgb.Dataset.save() or Dataset$save_binary() to save lightgbm Datasets.");
  }
}

SEXP LGBM_DatasetCreateFromFile_R(SEXP filename,
  SEXP parameters,
  SEXP reference) {
  R_API_BEGIN();
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  DatasetHandle handle = nullptr;
  DatasetHandle ref = nullptr;
  if (!Rf_isNull(reference)) {
    ref = R_ExternalPtrAddr(reference);
  }
  const char* filename_ptr = CHAR(Rf_protect(Rf_asChar(filename)));
  const char* parameters_ptr = CHAR(Rf_protect(Rf_asChar(parameters)));
  CHECK_CALL(LGBM_DatasetCreateFromFile(filename_ptr, parameters_ptr, ref, &handle));
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  Rf_unprotect(3);
  return ret;
  R_API_END();
}

SEXP LGBM_DatasetCreateFromCSC_R(SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  SEXP parameters,
  SEXP reference) {
  R_API_BEGIN();
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  const int* p_indptr = INTEGER(indptr);
  const int* p_indices = INTEGER(indices);
  const double* p_data = REAL(data);
  int64_t nindptr = static_cast<int64_t>(Rf_asInteger(num_indptr));
  int64_t ndata = static_cast<int64_t>(Rf_asInteger(nelem));
  int64_t nrow = static_cast<int64_t>(Rf_asInteger(num_row));
  const char* parameters_ptr = CHAR(Rf_protect(Rf_asChar(parameters)));
  DatasetHandle handle = nullptr;
  DatasetHandle ref = nullptr;
  if (!Rf_isNull(reference)) {
    ref = R_ExternalPtrAddr(reference);
  }
  CHECK_CALL(LGBM_DatasetCreateFromCSC(p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, parameters_ptr, ref, &handle));
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  Rf_unprotect(2);
  return ret;
  R_API_END();
}

SEXP LGBM_DatasetCreateFromMat_R(SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP parameters,
  SEXP reference) {
  R_API_BEGIN();
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  int32_t nrow = static_cast<int32_t>(Rf_asInteger(num_row));
  int32_t ncol = static_cast<int32_t>(Rf_asInteger(num_col));
  double* p_mat = REAL(data);
  const char* parameters_ptr = CHAR(Rf_protect(Rf_asChar(parameters)));
  DatasetHandle handle = nullptr;
  DatasetHandle ref = nullptr;
  if (!Rf_isNull(reference)) {
    ref = R_ExternalPtrAddr(reference);
  }
  CHECK_CALL(LGBM_DatasetCreateFromMat(p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    parameters_ptr, ref, &handle));
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  Rf_unprotect(2);
  return ret;
  R_API_END();
}

SEXP LGBM_DatasetGetSubset_R(SEXP handle,
  SEXP used_row_indices,
  SEXP len_used_row_indices,
  SEXP parameters) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  int32_t len = static_cast<int32_t>(Rf_asInteger(len_used_row_indices));
  std::unique_ptr<int32_t[]> idxvec(new int32_t[len]);
  // convert from one-based to zero-based index
  const int *used_row_indices_ = INTEGER(used_row_indices);
#ifndef _MSC_VER
#pragma omp simd
#endif
  for (int32_t i = 0; i < len; ++i) {
    idxvec[i] = static_cast<int32_t>(used_row_indices_[i] - 1);
  }
  const char* parameters_ptr = CHAR(Rf_protect(Rf_asChar(parameters)));
  DatasetHandle res = nullptr;
  CHECK_CALL(LGBM_DatasetGetSubset(R_ExternalPtrAddr(handle),
    idxvec.get(), len, parameters_ptr,
    &res));
  R_SetExternalPtrAddr(ret, res);
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  Rf_unprotect(2);
  return ret;
  R_API_END();
}

SEXP LGBM_DatasetSetFeatureNames_R(SEXP handle,
  SEXP feature_names) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  auto vec_names = Split(CHAR(Rf_protect(Rf_asChar(feature_names))), '\t');
  int len = static_cast<int>(vec_names.size());
  std::unique_ptr<const char*[]> vec_sptr(new const char*[len]);
  for (int i = 0; i < len; ++i) {
    vec_sptr[i] = vec_names[i].c_str();
  }
  CHECK_CALL(LGBM_DatasetSetFeatureNames(R_ExternalPtrAddr(handle),
    vec_sptr.get(), len));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetGetFeatureNames_R(SEXP handle) {
  SEXP cont_token = Rf_protect(R_MakeUnwindCont());
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  SEXP feature_names;
  int len = 0;
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_ExternalPtrAddr(handle), &len));
  const size_t reserved_string_size = 256;
  std::vector<std::vector<char>> names(len);
  std::vector<char*> ptr_names(len);
  for (int i = 0; i < len; ++i) {
    names[i].resize(reserved_string_size);
    ptr_names[i] = names[i].data();
  }
  int out_len;
  size_t required_string_size;
  CHECK_CALL(
    LGBM_DatasetGetFeatureNames(
      R_ExternalPtrAddr(handle),
      len, &out_len,
      reserved_string_size, &required_string_size,
      ptr_names.data()));
  // if any feature names were larger than allocated size,
  // allow for a larger size and try again
  if (required_string_size > reserved_string_size) {
    for (int i = 0; i < len; ++i) {
      names[i].resize(required_string_size);
      ptr_names[i] = names[i].data();
    }
    CHECK_CALL(
      LGBM_DatasetGetFeatureNames(
        R_ExternalPtrAddr(handle),
        len,
        &out_len,
        required_string_size,
        &required_string_size,
        ptr_names.data()));
  }
  CHECK_EQ(len, out_len);
  feature_names = Rf_protect(safe_R_string(static_cast<R_xlen_t>(len), &cont_token));
  for (int i = 0; i < len; ++i) {
    SET_STRING_ELT(feature_names, i, safe_R_mkChar(ptr_names[i], &cont_token));
  }
  Rf_unprotect(2);
  return feature_names;
  R_API_END();
}

SEXP LGBM_DatasetSaveBinary_R(SEXP handle,
  SEXP filename) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  const char* filename_ptr = CHAR(Rf_protect(Rf_asChar(filename)));
  CHECK_CALL(LGBM_DatasetSaveBinary(R_ExternalPtrAddr(handle),
    filename_ptr));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetFree_R(SEXP handle) {
  R_API_BEGIN();
  if (!Rf_isNull(handle) && R_ExternalPtrAddr(handle)) {
    CHECK_CALL(LGBM_DatasetFree(R_ExternalPtrAddr(handle)));
    R_ClearExternalPtr(handle);
  }
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetSetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data,
  SEXP num_element) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  int len = Rf_asInteger(num_element);
  const char* name = CHAR(Rf_protect(Rf_asChar(field_name)));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    CHECK_CALL(LGBM_DatasetSetField(R_ExternalPtrAddr(handle), name, INTEGER(field_data), len, C_API_DTYPE_INT32));
  } else if (!strcmp("init_score", name)) {
    CHECK_CALL(LGBM_DatasetSetField(R_ExternalPtrAddr(handle), name, REAL(field_data), len, C_API_DTYPE_FLOAT64));
  } else {
    std::unique_ptr<float[]> vec(new float[len]);
    std::copy(REAL(field_data), REAL(field_data) + len, vec.get());
    CHECK_CALL(LGBM_DatasetSetField(R_ExternalPtrAddr(handle), name, vec.get(), len, C_API_DTYPE_FLOAT32));
  }
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetGetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  const char* name = CHAR(Rf_protect(Rf_asChar(field_name)));
  int out_len = 0;
  int out_type = 0;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_ExternalPtrAddr(handle), name, &out_len, &res, &out_type));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    auto p_data = reinterpret_cast<const int32_t*>(res);
    // convert from boundaries to size
    int *field_data_ = INTEGER(field_data);
#ifndef _MSC_VER
#pragma omp simd
#endif
    for (int i = 0; i < out_len - 1; ++i) {
      field_data_[i] = p_data[i + 1] - p_data[i];
    }
  } else if (!strcmp("init_score", name)) {
    auto p_data = reinterpret_cast<const double*>(res);
    std::copy(p_data, p_data + out_len, REAL(field_data));
  } else {
    auto p_data = reinterpret_cast<const float*>(res);
    std::copy(p_data, p_data + out_len, REAL(field_data));
  }
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetGetFieldSize_R(SEXP handle,
  SEXP field_name,
  SEXP out) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  const char* name = CHAR(Rf_protect(Rf_asChar(field_name)));
  int out_len = 0;
  int out_type = 0;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_ExternalPtrAddr(handle), name, &out_len, &res, &out_type));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    out_len -= 1;
  }
  INTEGER(out)[0] = out_len;
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetUpdateParamChecking_R(SEXP old_params,
  SEXP new_params) {
  R_API_BEGIN();
  const char* old_params_ptr = CHAR(Rf_protect(Rf_asChar(old_params)));
  const char* new_params_ptr = CHAR(Rf_protect(Rf_asChar(new_params)));
  CHECK_CALL(LGBM_DatasetUpdateParamChecking(old_params_ptr, new_params_ptr));
  Rf_unprotect(2);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetGetNumData_R(SEXP handle, SEXP out) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  int nrow;
  CHECK_CALL(LGBM_DatasetGetNumData(R_ExternalPtrAddr(handle), &nrow));
  INTEGER(out)[0] = nrow;
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetGetNumFeature_R(SEXP handle,
  SEXP out) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  int nfeature;
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_ExternalPtrAddr(handle), &nfeature));
  INTEGER(out)[0] = nfeature;
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_DatasetGetFeatureNumBin_R(SEXP handle, SEXP feature_idx, SEXP out) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(handle);
  int feature = Rf_asInteger(feature_idx);
  int nbins;
  CHECK_CALL(LGBM_DatasetGetFeatureNumBin(R_ExternalPtrAddr(handle), feature, &nbins));
  INTEGER(out)[0] = nbins;
  return R_NilValue;
  R_API_END();
}

// --- start Booster interfaces

void _BoosterFinalizer(SEXP handle) {
  LGBM_BoosterFree_R(handle);
}

SEXP LGBM_BoosterFree_R(SEXP handle) {
  R_API_BEGIN();
  if (!Rf_isNull(handle) && R_ExternalPtrAddr(handle)) {
    CHECK_CALL(LGBM_BoosterFree(R_ExternalPtrAddr(handle)));
    R_ClearExternalPtr(handle);
  }
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterCreate_R(SEXP train_data,
  SEXP parameters) {
  R_API_BEGIN();
  _AssertDatasetHandleNotNull(train_data);
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  const char* parameters_ptr = CHAR(Rf_protect(Rf_asChar(parameters)));
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterCreate(R_ExternalPtrAddr(train_data), parameters_ptr, &handle));
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  Rf_unprotect(2);
  return ret;
  R_API_END();
}

SEXP LGBM_BoosterCreateFromModelfile_R(SEXP filename) {
  R_API_BEGIN();
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  int out_num_iterations = 0;
  const char* filename_ptr = CHAR(Rf_protect(Rf_asChar(filename)));
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterCreateFromModelfile(filename_ptr, &out_num_iterations, &handle));
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  Rf_unprotect(2);
  return ret;
  R_API_END();
}

SEXP LGBM_BoosterLoadModelFromString_R(SEXP model_str) {
  R_API_BEGIN();
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  SEXP temp = NULL;
  int n_protected = 1;
  int out_num_iterations = 0;
  const char* model_str_ptr = nullptr;
  switch (TYPEOF(model_str)) {
    case RAWSXP: {
      model_str_ptr = reinterpret_cast<const char*>(RAW(model_str));
      break;
    }
    case CHARSXP: {
      model_str_ptr = reinterpret_cast<const char*>(CHAR(model_str));
      break;
    }
    case STRSXP: {
      temp = Rf_protect(STRING_ELT(model_str, 0));
      n_protected++;
      model_str_ptr = reinterpret_cast<const char*>(CHAR(temp));
    }
  }
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterLoadModelFromString(model_str_ptr, &out_num_iterations, &handle));
  R_SetExternalPtrAddr(ret, handle);
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  Rf_unprotect(n_protected);
  return ret;
  R_API_END();
}

SEXP LGBM_BoosterMerge_R(SEXP handle,
  SEXP other_handle) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  _AssertBoosterHandleNotNull(other_handle);
  CHECK_CALL(LGBM_BoosterMerge(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(other_handle)));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterAddValidData_R(SEXP handle,
  SEXP valid_data) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  _AssertDatasetHandleNotNull(valid_data);
  CHECK_CALL(LGBM_BoosterAddValidData(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(valid_data)));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterResetTrainingData_R(SEXP handle,
  SEXP train_data) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  _AssertDatasetHandleNotNull(train_data);
  CHECK_CALL(LGBM_BoosterResetTrainingData(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(train_data)));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterResetParameter_R(SEXP handle,
  SEXP parameters) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  const char* parameters_ptr = CHAR(Rf_protect(Rf_asChar(parameters)));
  CHECK_CALL(LGBM_BoosterResetParameter(R_ExternalPtrAddr(handle), parameters_ptr));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterGetNumClasses_R(SEXP handle,
  SEXP out) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int num_class;
  CHECK_CALL(LGBM_BoosterGetNumClasses(R_ExternalPtrAddr(handle), &num_class));
  INTEGER(out)[0] = num_class;
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterGetNumFeature_R(SEXP handle) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int out = 0;
  CHECK_CALL(LGBM_BoosterGetNumFeature(R_ExternalPtrAddr(handle), &out));
  return Rf_ScalarInteger(out);
  R_API_END();
}

SEXP LGBM_BoosterUpdateOneIter_R(SEXP handle) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int produced_empty_tree = 0;
  CHECK_CALL(LGBM_BoosterUpdateOneIter(R_ExternalPtrAddr(handle), &produced_empty_tree));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterUpdateOneIterCustom_R(SEXP handle,
  SEXP grad,
  SEXP hess,
  SEXP len) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int produced_empty_tree = 0;
  int int_len = Rf_asInteger(len);
  std::unique_ptr<float[]> tgrad(new float[int_len]), thess(new float[int_len]);
  std::copy(REAL(grad), REAL(grad) + int_len, tgrad.get());
  std::copy(REAL(hess), REAL(hess) + int_len, thess.get());
  CHECK_CALL(LGBM_BoosterUpdateOneIterCustom(R_ExternalPtrAddr(handle), tgrad.get(), thess.get(),
    &produced_empty_tree));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterRollbackOneIter_R(SEXP handle) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  CHECK_CALL(LGBM_BoosterRollbackOneIter(R_ExternalPtrAddr(handle)));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterGetCurrentIteration_R(SEXP handle, SEXP out) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int out_iteration;
  CHECK_CALL(LGBM_BoosterGetCurrentIteration(R_ExternalPtrAddr(handle), &out_iteration));
  INTEGER(out)[0] = out_iteration;
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterNumModelPerIteration_R(SEXP handle, SEXP out) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int models_per_iter;
  CHECK_CALL(LGBM_BoosterNumModelPerIteration(R_ExternalPtrAddr(handle), &models_per_iter));
  INTEGER(out)[0] = models_per_iter;
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterNumberOfTotalModel_R(SEXP handle, SEXP out) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int total_models;
  CHECK_CALL(LGBM_BoosterNumberOfTotalModel(R_ExternalPtrAddr(handle), &total_models));
  INTEGER(out)[0] = total_models;
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterGetUpperBoundValue_R(SEXP handle,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  double* ptr_ret = REAL(out_result);
  CHECK_CALL(LGBM_BoosterGetUpperBoundValue(R_ExternalPtrAddr(handle), ptr_ret));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterGetLowerBoundValue_R(SEXP handle,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  double* ptr_ret = REAL(out_result);
  CHECK_CALL(LGBM_BoosterGetLowerBoundValue(R_ExternalPtrAddr(handle), ptr_ret));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterGetEvalNames_R(SEXP handle) {
  SEXP cont_token = Rf_protect(R_MakeUnwindCont());
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  SEXP eval_names;
  int len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_ExternalPtrAddr(handle), &len));
  const size_t reserved_string_size = 128;
  std::vector<std::vector<char>> names(len);
  std::vector<char*> ptr_names(len);
  for (int i = 0; i < len; ++i) {
    names[i].resize(reserved_string_size);
    ptr_names[i] = names[i].data();
  }

  int out_len;
  size_t required_string_size;
  CHECK_CALL(
    LGBM_BoosterGetEvalNames(
      R_ExternalPtrAddr(handle),
      len, &out_len,
      reserved_string_size, &required_string_size,
      ptr_names.data()));
  // if any eval names were larger than allocated size,
  // allow for a larger size and try again
  if (required_string_size > reserved_string_size) {
    for (int i = 0; i < len; ++i) {
      names[i].resize(required_string_size);
      ptr_names[i] = names[i].data();
    }
    CHECK_CALL(
      LGBM_BoosterGetEvalNames(
        R_ExternalPtrAddr(handle),
        len,
        &out_len,
        required_string_size,
        &required_string_size,
        ptr_names.data()));
  }
  CHECK_EQ(out_len, len);
  eval_names = Rf_protect(safe_R_string(static_cast<R_xlen_t>(len), &cont_token));
  for (int i = 0; i < len; ++i) {
    SET_STRING_ELT(eval_names, i, safe_R_mkChar(ptr_names[i], &cont_token));
  }
  Rf_unprotect(2);
  return eval_names;
  R_API_END();
}

SEXP LGBM_BoosterGetEval_R(SEXP handle,
  SEXP data_idx,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_ExternalPtrAddr(handle), &len));
  double* ptr_ret = REAL(out_result);
  int out_len;
  CHECK_CALL(LGBM_BoosterGetEval(R_ExternalPtrAddr(handle), Rf_asInteger(data_idx), &out_len, ptr_ret));
  CHECK_EQ(out_len, len);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterGetNumPredict_R(SEXP handle,
  SEXP data_idx,
  SEXP out) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int64_t len;
  CHECK_CALL(LGBM_BoosterGetNumPredict(R_ExternalPtrAddr(handle), Rf_asInteger(data_idx), &len));
  INTEGER(out)[0] = static_cast<int>(len);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterGetPredict_R(SEXP handle,
  SEXP data_idx,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterGetPredict(R_ExternalPtrAddr(handle), Rf_asInteger(data_idx), &out_len, ptr_ret));
  return R_NilValue;
  R_API_END();
}

int GetPredictType(SEXP is_rawscore, SEXP is_leafidx, SEXP is_predcontrib) {
  int pred_type = C_API_PREDICT_NORMAL;
  if (Rf_asInteger(is_rawscore)) {
    pred_type = C_API_PREDICT_RAW_SCORE;
  }
  if (Rf_asInteger(is_leafidx)) {
    pred_type = C_API_PREDICT_LEAF_INDEX;
  }
  if (Rf_asInteger(is_predcontrib)) {
    pred_type = C_API_PREDICT_CONTRIB;
  }
  return pred_type;
}

SEXP LGBM_BoosterPredictForFile_R(SEXP handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP result_filename) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  const char* data_filename_ptr = CHAR(Rf_protect(Rf_asChar(data_filename)));
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));
  const char* result_filename_ptr = CHAR(Rf_protect(Rf_asChar(result_filename)));
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  CHECK_CALL(LGBM_BoosterPredictForFile(R_ExternalPtrAddr(handle), data_filename_ptr,
    Rf_asInteger(data_has_header), pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), parameter_ptr,
    result_filename_ptr));
  Rf_unprotect(3);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterCalcNumPredict_R(SEXP handle,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP out_len) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  int64_t len = 0;
  CHECK_CALL(LGBM_BoosterCalcNumPredict(R_ExternalPtrAddr(handle), Rf_asInteger(num_row),
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), &len));
  INTEGER(out_len)[0] = static_cast<int>(len);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterPredictForCSC_R(SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  const int* p_indptr = INTEGER(indptr);
  const int32_t* p_indices = reinterpret_cast<const int32_t*>(INTEGER(indices));
  const double* p_data = REAL(data);
  int64_t nindptr = static_cast<int64_t>(Rf_asInteger(num_indptr));
  int64_t ndata = static_cast<int64_t>(Rf_asInteger(nelem));
  int64_t nrow = static_cast<int64_t>(Rf_asInteger(num_row));
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));
  CHECK_CALL(LGBM_BoosterPredictForCSC(R_ExternalPtrAddr(handle),
    p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), parameter_ptr, &out_len, ptr_ret));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterPredictForCSR_R(SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP ncols,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForCSR(R_ExternalPtrAddr(handle),
    INTEGER(indptr), C_API_DTYPE_INT32, INTEGER(indices),
    REAL(data), C_API_DTYPE_FLOAT64,
    Rf_xlength(indptr), Rf_xlength(data), Rf_asInteger(ncols),
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration),
    parameter_ptr, &out_len, REAL(out_result)));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterPredictForCSRSingleRow_R(SEXP handle,
  SEXP indices,
  SEXP data,
  SEXP ncols,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));
  int nnz = static_cast<int>(Rf_xlength(data));
  const int indptr[] = {0, nnz};
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForCSRSingleRow(R_ExternalPtrAddr(handle),
    indptr, C_API_DTYPE_INT32, INTEGER(indices),
    REAL(data), C_API_DTYPE_FLOAT64,
    2, nnz, Rf_asInteger(ncols),
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration),
    parameter_ptr, &out_len, REAL(out_result)));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

void LGBM_FastConfigFree_wrapped(SEXP handle) {
  LGBM_FastConfigFree(static_cast<FastConfigHandle*>(R_ExternalPtrAddr(handle)));
}

SEXP LGBM_BoosterPredictForCSRSingleRowFastInit_R(SEXP handle,
  SEXP ncols,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));
  FastConfigHandle out_fastConfig;
  CHECK_CALL(LGBM_BoosterPredictForCSRSingleRowFastInit(R_ExternalPtrAddr(handle),
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration),
    C_API_DTYPE_FLOAT64, Rf_asInteger(ncols),
    parameter_ptr, &out_fastConfig));
  R_SetExternalPtrAddr(ret, out_fastConfig);
  R_RegisterCFinalizerEx(ret, LGBM_FastConfigFree_wrapped, TRUE);
  Rf_unprotect(2);
  return ret;
  R_API_END();
}

SEXP LGBM_BoosterPredictForCSRSingleRowFast_R(SEXP handle_fastConfig,
  SEXP indices,
  SEXP data,
  SEXP out_result) {
  R_API_BEGIN();
  int nnz = static_cast<int>(Rf_xlength(data));
  const int indptr[] = {0, nnz};
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForCSRSingleRowFast(R_ExternalPtrAddr(handle_fastConfig),
    indptr, C_API_DTYPE_INT32, INTEGER(indices),
    REAL(data),
    2, nnz,
    &out_len, REAL(out_result)));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterPredictForMat_R(SEXP handle,
  SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  int32_t nrow = static_cast<int32_t>(Rf_asInteger(num_row));
  int32_t ncol = static_cast<int32_t>(Rf_asInteger(num_col));
  const double* p_mat = REAL(data);
  double* ptr_ret = REAL(out_result);
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForMat(R_ExternalPtrAddr(handle),
    p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), parameter_ptr, &out_len, ptr_ret));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

struct SparseOutputPointers {
  void* indptr;
  int32_t* indices;
  void* data;
  SparseOutputPointers(void* indptr, int32_t* indices, void* data)
  : indptr(indptr), indices(indices), data(data) {}
};

void delete_SparseOutputPointers(SparseOutputPointers *ptr) {
  LGBM_BoosterFreePredictSparse(ptr->indptr, ptr->indices, ptr->data, C_API_DTYPE_INT32, C_API_DTYPE_FLOAT64);
  delete ptr;
}

SEXP LGBM_BoosterPredictSparseOutput_R(SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP is_csr,
  SEXP nrows,
  SEXP ncols,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter) {
  SEXP cont_token = Rf_protect(R_MakeUnwindCont());
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  const char* out_names[] = {"indptr", "indices", "data", ""};
  SEXP out = Rf_protect(Rf_mkNamed(VECSXP, out_names));
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));

  int64_t out_len[2];
  void *out_indptr;
  int32_t *out_indices;
  void *out_data;

  CHECK_CALL(LGBM_BoosterPredictSparseOutput(R_ExternalPtrAddr(handle),
    INTEGER(indptr), C_API_DTYPE_INT32, INTEGER(indices),
    REAL(data), C_API_DTYPE_FLOAT64,
    Rf_xlength(indptr), Rf_xlength(data),
    Rf_asLogical(is_csr)? Rf_asInteger(ncols) : Rf_asInteger(nrows),
    C_API_PREDICT_CONTRIB, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration),
    parameter_ptr,
    Rf_asLogical(is_csr)? C_API_MATRIX_TYPE_CSR : C_API_MATRIX_TYPE_CSC,
    out_len, &out_indptr, &out_indices, &out_data));

  std::unique_ptr<SparseOutputPointers, decltype(&delete_SparseOutputPointers)> pointers_struct = {
    new SparseOutputPointers(
      out_indptr,
      out_indices,
      out_data),
    &delete_SparseOutputPointers
  };

  arr_and_len<int> indptr_str{static_cast<int*>(out_indptr), out_len[1]};
  SET_VECTOR_ELT(
    out, 0,
    R_UnwindProtect(make_altrepped_vec_from_arr<int>,
      static_cast<void*>(&indptr_str), throw_R_memerr, &cont_token, cont_token));
  pointers_struct->indptr = nullptr;

  arr_and_len<int> indices_str{static_cast<int*>(out_indices), out_len[0]};
  SET_VECTOR_ELT(
    out, 1,
    R_UnwindProtect(make_altrepped_vec_from_arr<int>,
      static_cast<void*>(&indices_str), throw_R_memerr, &cont_token, cont_token));
  pointers_struct->indices = nullptr;

  arr_and_len<double> data_str{static_cast<double*>(out_data), out_len[0]};
  SET_VECTOR_ELT(
    out, 2,
    R_UnwindProtect(make_altrepped_vec_from_arr<double>,
      static_cast<void*>(&data_str), throw_R_memerr, &cont_token, cont_token));
  pointers_struct->data = nullptr;

  Rf_unprotect(3);
  return out;
  R_API_END();
}

SEXP LGBM_BoosterPredictForMatSingleRow_R(SEXP handle,
  SEXP data,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForMatSingleRow(R_ExternalPtrAddr(handle),
    REAL(data), C_API_DTYPE_FLOAT64, Rf_xlength(data), 1,
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration),
    parameter_ptr, &out_len, ptr_ret));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterPredictForMatSingleRowFastInit_R(SEXP handle,
  SEXP ncols,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  SEXP ret = Rf_protect(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
  const char* parameter_ptr = CHAR(Rf_protect(Rf_asChar(parameter)));
  FastConfigHandle out_fastConfig;
  CHECK_CALL(LGBM_BoosterPredictForMatSingleRowFastInit(R_ExternalPtrAddr(handle),
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration),
    C_API_DTYPE_FLOAT64, Rf_asInteger(ncols),
    parameter_ptr, &out_fastConfig));
  R_SetExternalPtrAddr(ret, out_fastConfig);
  R_RegisterCFinalizerEx(ret, LGBM_FastConfigFree_wrapped, TRUE);
  Rf_unprotect(2);
  return ret;
  R_API_END();
}

SEXP LGBM_BoosterPredictForMatSingleRowFast_R(SEXP handle_fastConfig,
  SEXP data,
  SEXP out_result) {
  R_API_BEGIN();
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForMatSingleRowFast(R_ExternalPtrAddr(handle_fastConfig),
    REAL(data), &out_len, REAL(out_result)));
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_BoosterSaveModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP filename,
  SEXP start_iteration) {
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  const char* filename_ptr = CHAR(Rf_protect(Rf_asChar(filename)));
  CHECK_CALL(LGBM_BoosterSaveModel(R_ExternalPtrAddr(handle), Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), Rf_asInteger(feature_importance_type), filename_ptr));
  Rf_unprotect(1);
  return R_NilValue;
  R_API_END();
}

// Note: for some reason, MSVC crashes when an error is thrown here
// if the buffer variable is defined as 'std::unique_ptr<std::vector<char>>',
// but not if it is defined as '<std::vector<char>'.
#ifndef _MSC_VER
SEXP LGBM_BoosterSaveModelToString_R(SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP start_iteration) {
  SEXP cont_token = Rf_protect(R_MakeUnwindCont());
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int64_t out_len = 0;
  int64_t buf_len = 1024 * 1024;
  int num_iter = Rf_asInteger(num_iteration);
  int start_iter = Rf_asInteger(start_iteration);
  int importance_type = Rf_asInteger(feature_importance_type);
  std::unique_ptr<std::vector<char>> inner_char_buf(new std::vector<char>(buf_len));
  CHECK_CALL(LGBM_BoosterSaveModelToString(R_ExternalPtrAddr(handle), start_iter, num_iter, importance_type, buf_len, &out_len, inner_char_buf->data()));
  inner_char_buf->resize(out_len);
  if (out_len > buf_len) {
    CHECK_CALL(LGBM_BoosterSaveModelToString(R_ExternalPtrAddr(handle), start_iter, num_iter, importance_type, out_len, &out_len, inner_char_buf->data()));
  }
  SEXP out = R_UnwindProtect(make_altrepped_raw_vec, &inner_char_buf, throw_R_memerr, &cont_token, cont_token);
  Rf_unprotect(1);
  return out;
  R_API_END();
}
#else
SEXP LGBM_BoosterSaveModelToString_R(SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP start_iteration) {
  SEXP cont_token = Rf_protect(R_MakeUnwindCont());
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  int64_t out_len = 0;
  int64_t buf_len = 1024 * 1024;
  int num_iter = Rf_asInteger(num_iteration);
  int start_iter = Rf_asInteger(start_iteration);
  int importance_type = Rf_asInteger(feature_importance_type);
  std::vector<char> inner_char_buf(buf_len);
  CHECK_CALL(LGBM_BoosterSaveModelToString(R_ExternalPtrAddr(handle), start_iter, num_iter, importance_type, buf_len, &out_len, inner_char_buf.data()));
  SEXP model_str = Rf_protect(safe_R_raw(out_len, &cont_token));
  // if the model string was larger than the initial buffer, call the function again, writing directly to the R object
  if (out_len > buf_len) {
    CHECK_CALL(LGBM_BoosterSaveModelToString(R_ExternalPtrAddr(handle), start_iter, num_iter, importance_type, out_len, &out_len, reinterpret_cast<char*>(RAW(model_str))));
  } else {
    std::copy(inner_char_buf.begin(), inner_char_buf.begin() + out_len, reinterpret_cast<char*>(RAW(model_str)));
  }
  Rf_unprotect(2);
  return model_str;
  R_API_END();
}
#endif

SEXP LGBM_BoosterDumpModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP start_iteration) {
  SEXP cont_token = Rf_protect(R_MakeUnwindCont());
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  SEXP model_str;
  int64_t out_len = 0;
  int64_t buf_len = 1024 * 1024;
  int num_iter = Rf_asInteger(num_iteration);
  int start_iter = Rf_asInteger(start_iteration);
  int importance_type = Rf_asInteger(feature_importance_type);
  std::vector<char> inner_char_buf(buf_len);
  CHECK_CALL(LGBM_BoosterDumpModel(R_ExternalPtrAddr(handle), start_iter, num_iter, importance_type, buf_len, &out_len, inner_char_buf.data()));
  // if the model string was larger than the initial buffer, allocate a bigger buffer and try again
  if (out_len > buf_len) {
    inner_char_buf.resize(out_len);
    CHECK_CALL(LGBM_BoosterDumpModel(R_ExternalPtrAddr(handle), start_iter, num_iter, importance_type, out_len, &out_len, inner_char_buf.data()));
  }
  model_str = Rf_protect(safe_R_string(static_cast<R_xlen_t>(1), &cont_token));
  SET_STRING_ELT(model_str, 0, safe_R_mkChar(inner_char_buf.data(), &cont_token));
  Rf_unprotect(2);
  return model_str;
  R_API_END();
}

SEXP LGBM_DumpParamAliases_R() {
  SEXP cont_token = Rf_protect(R_MakeUnwindCont());
  R_API_BEGIN();
  SEXP aliases_str;
  int64_t out_len = 0;
  int64_t buf_len = 1024 * 1024;
  std::vector<char> inner_char_buf(buf_len);
  CHECK_CALL(LGBM_DumpParamAliases(buf_len, &out_len, inner_char_buf.data()));
  // if aliases string was larger than the initial buffer, allocate a bigger buffer and try again
  if (out_len > buf_len) {
    inner_char_buf.resize(out_len);
    CHECK_CALL(LGBM_DumpParamAliases(out_len, &out_len, inner_char_buf.data()));
  }
  aliases_str = Rf_protect(safe_R_string(static_cast<R_xlen_t>(1), &cont_token));
  SET_STRING_ELT(aliases_str, 0, safe_R_mkChar(inner_char_buf.data(), &cont_token));
  Rf_unprotect(2);
  return aliases_str;
  R_API_END();
}

SEXP LGBM_BoosterGetLoadedParam_R(SEXP handle) {
  SEXP cont_token = Rf_protect(R_MakeUnwindCont());
  R_API_BEGIN();
  _AssertBoosterHandleNotNull(handle);
  SEXP params_str;
  int64_t out_len = 0;
  int64_t buf_len = 1024 * 1024;
  std::vector<char> inner_char_buf(buf_len);
  CHECK_CALL(LGBM_BoosterGetLoadedParam(R_ExternalPtrAddr(handle), buf_len, &out_len, inner_char_buf.data()));
  // if aliases string was larger than the initial buffer, allocate a bigger buffer and try again
  if (out_len > buf_len) {
    inner_char_buf.resize(out_len);
    CHECK_CALL(LGBM_BoosterGetLoadedParam(R_ExternalPtrAddr(handle), out_len, &out_len, inner_char_buf.data()));
  }
  params_str = Rf_protect(safe_R_string(static_cast<R_xlen_t>(1), &cont_token));
  SET_STRING_ELT(params_str, 0, safe_R_mkChar(inner_char_buf.data(), &cont_token));
  Rf_unprotect(2);
  return params_str;
  R_API_END();
}

SEXP LGBM_GetMaxThreads_R(SEXP out) {
  R_API_BEGIN();
  int num_threads;
  CHECK_CALL(LGBM_GetMaxThreads(&num_threads));
  INTEGER(out)[0] = num_threads;
  return R_NilValue;
  R_API_END();
}

SEXP LGBM_SetMaxThreads_R(SEXP num_threads) {
  R_API_BEGIN();
  int new_num_threads = Rf_asInteger(num_threads);
  CHECK_CALL(LGBM_SetMaxThreads(new_num_threads));
  return R_NilValue;
  R_API_END();
}

// .Call() calls
static const R_CallMethodDef CallEntries[] = {
  {"LGBM_HandleIsNull_R"                         , (DL_FUNC) &LGBM_HandleIsNull_R                         , 1},
  {"LGBM_DatasetCreateFromFile_R"                , (DL_FUNC) &LGBM_DatasetCreateFromFile_R                , 3},
  {"LGBM_DatasetCreateFromCSC_R"                 , (DL_FUNC) &LGBM_DatasetCreateFromCSC_R                 , 8},
  {"LGBM_DatasetCreateFromMat_R"                 , (DL_FUNC) &LGBM_DatasetCreateFromMat_R                 , 5},
  {"LGBM_DatasetGetSubset_R"                     , (DL_FUNC) &LGBM_DatasetGetSubset_R                     , 4},
  {"LGBM_DatasetSetFeatureNames_R"               , (DL_FUNC) &LGBM_DatasetSetFeatureNames_R               , 2},
  {"LGBM_DatasetGetFeatureNames_R"               , (DL_FUNC) &LGBM_DatasetGetFeatureNames_R               , 1},
  {"LGBM_DatasetSaveBinary_R"                    , (DL_FUNC) &LGBM_DatasetSaveBinary_R                    , 2},
  {"LGBM_DatasetFree_R"                          , (DL_FUNC) &LGBM_DatasetFree_R                          , 1},
  {"LGBM_DatasetSetField_R"                      , (DL_FUNC) &LGBM_DatasetSetField_R                      , 4},
  {"LGBM_DatasetGetFieldSize_R"                  , (DL_FUNC) &LGBM_DatasetGetFieldSize_R                  , 3},
  {"LGBM_DatasetGetField_R"                      , (DL_FUNC) &LGBM_DatasetGetField_R                      , 3},
  {"LGBM_DatasetUpdateParamChecking_R"           , (DL_FUNC) &LGBM_DatasetUpdateParamChecking_R           , 2},
  {"LGBM_DatasetGetNumData_R"                    , (DL_FUNC) &LGBM_DatasetGetNumData_R                    , 2},
  {"LGBM_DatasetGetNumFeature_R"                 , (DL_FUNC) &LGBM_DatasetGetNumFeature_R                 , 2},
  {"LGBM_DatasetGetFeatureNumBin_R"              , (DL_FUNC) &LGBM_DatasetGetFeatureNumBin_R              , 3},
  {"LGBM_BoosterCreate_R"                        , (DL_FUNC) &LGBM_BoosterCreate_R                        , 2},
  {"LGBM_BoosterFree_R"                          , (DL_FUNC) &LGBM_BoosterFree_R                          , 1},
  {"LGBM_BoosterCreateFromModelfile_R"           , (DL_FUNC) &LGBM_BoosterCreateFromModelfile_R           , 1},
  {"LGBM_BoosterLoadModelFromString_R"           , (DL_FUNC) &LGBM_BoosterLoadModelFromString_R           , 1},
  {"LGBM_BoosterMerge_R"                         , (DL_FUNC) &LGBM_BoosterMerge_R                         , 2},
  {"LGBM_BoosterAddValidData_R"                  , (DL_FUNC) &LGBM_BoosterAddValidData_R                  , 2},
  {"LGBM_BoosterResetTrainingData_R"             , (DL_FUNC) &LGBM_BoosterResetTrainingData_R             , 2},
  {"LGBM_BoosterResetParameter_R"                , (DL_FUNC) &LGBM_BoosterResetParameter_R                , 2},
  {"LGBM_BoosterGetNumClasses_R"                 , (DL_FUNC) &LGBM_BoosterGetNumClasses_R                 , 2},
  {"LGBM_BoosterGetNumFeature_R"                 , (DL_FUNC) &LGBM_BoosterGetNumFeature_R                 , 1},
  {"LGBM_BoosterGetLoadedParam_R"                , (DL_FUNC) &LGBM_BoosterGetLoadedParam_R                , 1},
  {"LGBM_BoosterUpdateOneIter_R"                 , (DL_FUNC) &LGBM_BoosterUpdateOneIter_R                 , 1},
  {"LGBM_BoosterUpdateOneIterCustom_R"           , (DL_FUNC) &LGBM_BoosterUpdateOneIterCustom_R           , 4},
  {"LGBM_BoosterRollbackOneIter_R"               , (DL_FUNC) &LGBM_BoosterRollbackOneIter_R               , 1},
  {"LGBM_BoosterGetCurrentIteration_R"           , (DL_FUNC) &LGBM_BoosterGetCurrentIteration_R           , 2},
  {"LGBM_BoosterNumModelPerIteration_R"          , (DL_FUNC) &LGBM_BoosterNumModelPerIteration_R          , 2},
  {"LGBM_BoosterNumberOfTotalModel_R"            , (DL_FUNC) &LGBM_BoosterNumberOfTotalModel_R            , 2},
  {"LGBM_BoosterGetUpperBoundValue_R"            , (DL_FUNC) &LGBM_BoosterGetUpperBoundValue_R            , 2},
  {"LGBM_BoosterGetLowerBoundValue_R"            , (DL_FUNC) &LGBM_BoosterGetLowerBoundValue_R            , 2},
  {"LGBM_BoosterGetEvalNames_R"                  , (DL_FUNC) &LGBM_BoosterGetEvalNames_R                  , 1},
  {"LGBM_BoosterGetEval_R"                       , (DL_FUNC) &LGBM_BoosterGetEval_R                       , 3},
  {"LGBM_BoosterGetNumPredict_R"                 , (DL_FUNC) &LGBM_BoosterGetNumPredict_R                 , 3},
  {"LGBM_BoosterGetPredict_R"                    , (DL_FUNC) &LGBM_BoosterGetPredict_R                    , 3},
  {"LGBM_BoosterPredictForFile_R"                , (DL_FUNC) &LGBM_BoosterPredictForFile_R                , 10},
  {"LGBM_BoosterCalcNumPredict_R"                , (DL_FUNC) &LGBM_BoosterCalcNumPredict_R                , 8},
  {"LGBM_BoosterPredictForCSC_R"                 , (DL_FUNC) &LGBM_BoosterPredictForCSC_R                 , 14},
  {"LGBM_BoosterPredictForCSR_R"                 , (DL_FUNC) &LGBM_BoosterPredictForCSR_R                 , 12},
  {"LGBM_BoosterPredictForCSRSingleRow_R"        , (DL_FUNC) &LGBM_BoosterPredictForCSRSingleRow_R        , 11},
  {"LGBM_BoosterPredictForCSRSingleRowFastInit_R", (DL_FUNC) &LGBM_BoosterPredictForCSRSingleRowFastInit_R, 8},
  {"LGBM_BoosterPredictForCSRSingleRowFast_R"    , (DL_FUNC) &LGBM_BoosterPredictForCSRSingleRowFast_R    , 4},
  {"LGBM_BoosterPredictSparseOutput_R"           , (DL_FUNC) &LGBM_BoosterPredictSparseOutput_R           , 10},
  {"LGBM_BoosterPredictForMat_R"                 , (DL_FUNC) &LGBM_BoosterPredictForMat_R                 , 11},
  {"LGBM_BoosterPredictForMatSingleRow_R"        , (DL_FUNC) &LGBM_BoosterPredictForMatSingleRow_R        , 9},
  {"LGBM_BoosterPredictForMatSingleRowFastInit_R", (DL_FUNC) &LGBM_BoosterPredictForMatSingleRowFastInit_R, 8},
  {"LGBM_BoosterPredictForMatSingleRowFast_R"    , (DL_FUNC) &LGBM_BoosterPredictForMatSingleRowFast_R    , 3},
  {"LGBM_BoosterSaveModel_R"                     , (DL_FUNC) &LGBM_BoosterSaveModel_R                     , 5},
  {"LGBM_BoosterSaveModelToString_R"             , (DL_FUNC) &LGBM_BoosterSaveModelToString_R             , 4},
  {"LGBM_BoosterDumpModel_R"                     , (DL_FUNC) &LGBM_BoosterDumpModel_R                     , 4},
  {"LGBM_NullBoosterHandleError_R"               , (DL_FUNC) &LGBM_NullBoosterHandleError_R               , 0},
  {"LGBM_DumpParamAliases_R"                     , (DL_FUNC) &LGBM_DumpParamAliases_R                     , 0},
  {"LGBM_GetMaxThreads_R"                        , (DL_FUNC) &LGBM_GetMaxThreads_R                        , 1},
  {"LGBM_SetMaxThreads_R"                        , (DL_FUNC) &LGBM_SetMaxThreads_R                        , 1},
  {NULL, NULL, 0}
};

LIGHTGBM_C_EXPORT void R_init_lightgbm(DllInfo *dll);

void R_init_lightgbm(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

#ifndef LGB_NO_ALTREP
  lgb_altrepped_char_vec = R_make_altraw_class("lgb_altrepped_char_vec", "lightgbm", dll);
  R_set_altrep_Length_method(lgb_altrepped_char_vec, get_altrepped_raw_len);
  R_set_altvec_Dataptr_method(lgb_altrepped_char_vec, get_altrepped_raw_dataptr);
  R_set_altvec_Dataptr_or_null_method(lgb_altrepped_char_vec, get_altrepped_raw_dataptr_or_null);

  lgb_altrepped_int_arr = R_make_altinteger_class("lgb_altrepped_int_arr", "lightgbm", dll);
  R_set_altrep_Length_method(lgb_altrepped_int_arr, get_altrepped_vec_len);
  R_set_altvec_Dataptr_method(lgb_altrepped_int_arr, get_altrepped_vec_dataptr);
  R_set_altvec_Dataptr_or_null_method(lgb_altrepped_int_arr, get_altrepped_vec_dataptr_or_null);

  lgb_altrepped_dbl_arr = R_make_altreal_class("lgb_altrepped_dbl_arr", "lightgbm", dll);
  R_set_altrep_Length_method(lgb_altrepped_dbl_arr, get_altrepped_vec_len);
  R_set_altvec_Dataptr_method(lgb_altrepped_dbl_arr, get_altrepped_vec_dataptr);
  R_set_altvec_Dataptr_or_null_method(lgb_altrepped_dbl_arr, get_altrepped_vec_dataptr_or_null);
#endif
}
