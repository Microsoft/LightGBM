/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "lightgbm_R.h"

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/text_reader.h>

#include <R_ext/Rdynload.h>

#define R_NO_REMAP
#define R_USE_C99_IN_CXX
#include <R_ext/Error.h>

#include <string>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#define COL_MAJOR (0)

#define R_API_BEGIN() \
  try {
#define R_API_END() } \
  catch(std::exception& ex) { LGBM_SetLastError(ex.what()); return R_NilValue;} \
  catch(std::string& ex) { LGBM_SetLastError(ex.c_str()); return R_NilValue; } \
  catch(...) { LGBM_SetLastError("unknown exception"); return R_NilValue;} \
  return R_NilValue;

#define CHECK_CALL(x) \
  if ((x) != 0) { \
    Rf_error(LGBM_GetLastError()); \
    return R_NilValue; \
  }

using LightGBM::Common::Split;
using LightGBM::Log;

SEXP LGBM_HandleIsNull_R(SEXP handle) {
  return Rf_ScalarLogical(R_ExternalPtrAddr(handle) == NULL);
}

void _DatasetFinalizer(SEXP handle) {
  LGBM_DatasetFree_R(handle);
}

SEXP LGBM_DatasetCreateFromFile_R(SEXP filename,
  SEXP parameters,
  SEXP reference) {
  SEXP ret;
  R_API_BEGIN();
  DatasetHandle handle = nullptr;
  DatasetHandle ref = nullptr;
  if (!Rf_isNull(reference)) {
    ref = R_ExternalPtrAddr(reference);
  }
  CHECK_CALL(LGBM_DatasetCreateFromFile(CHAR(Rf_asChar(filename)), CHAR(Rf_asChar(parameters)),
    ref, &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  UNPROTECT(1);
  return ret;
  R_API_END();
}

SEXP LGBM_DatasetCreateFromCSC_R(SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP label,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  SEXP parameters,
  SEXP reference) {
  SEXP ret;
  R_API_BEGIN();
  const int* p_indptr = INTEGER(indptr);
  const int* p_indices = INTEGER(indices);
  const double* p_data = REAL(data);
  const double* p_label = Rf_isNull(label) ? nullptr : REAL(label);

  int64_t nindptr = static_cast<int64_t>(Rf_asInteger(num_indptr));
  int64_t ndata = static_cast<int64_t>(Rf_asInteger(nelem));
  int64_t nrow = static_cast<int64_t>(Rf_asInteger(num_row));
  const float* float_p_label = nullptr;
  std::vector<float> float_label_vec;
  if (p_label != nullptr) {
    float_label_vec.resize(nrow);
    #pragma omp parallel for schedule(static) if (nrow >= 1024)
    for (int i = 0; i < nrow; ++i) {
      float_label_vec[i] = static_cast<float>(p_label[i]);
    }
    float_p_label = float_label_vec.data();
  }
  DatasetHandle handle = nullptr;
  DatasetHandle ref = nullptr;
  if (!Rf_isNull(reference)) {
    ref = R_ExternalPtrAddr(reference);
  }
  CHECK_CALL(LGBM_DatasetCreateFromCSC(p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, float_p_label, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, CHAR(Rf_asChar(parameters)), ref, &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  UNPROTECT(1);
  return ret;
  R_API_END();
}

SEXP LGBM_DatasetCreateFromMat_R(SEXP data,
  SEXP label,
  SEXP num_row,
  SEXP num_col,
  SEXP parameters,
  SEXP reference) {
  SEXP ret;
  R_API_BEGIN();
  int32_t nrow = static_cast<int32_t>(Rf_asInteger(num_row));
  int32_t ncol = static_cast<int32_t>(Rf_asInteger(num_col));
  double* p_mat = REAL(data);
  double* p_label = Rf_isNull(label) ? nullptr : REAL(label);
  const float* float_p_label = nullptr;
  std::vector<float> float_label_vec;
  if (p_label != nullptr) {
    float_label_vec.resize(nrow);
    #pragma omp parallel for schedule(static) if (nrow >= 1024)
    for (int i = 0; i < nrow; ++i) {
      float_label_vec[i] = static_cast<float>(p_label[i]);
    }
    float_p_label = float_label_vec.data();
  }
  DatasetHandle handle = nullptr;
  DatasetHandle ref = nullptr;
  if (!Rf_isNull(reference)) {
    ref = R_ExternalPtrAddr(reference);
  }
  CHECK_CALL(LGBM_DatasetCreateFromMat(p_mat, float_p_label, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    CHAR(Rf_asChar(parameters)), ref, &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  UNPROTECT(1);
  return ret;
  R_API_END();
}

SEXP LGBM_DatasetGetSubset_R(SEXP handle,
  SEXP used_row_indices,
  SEXP len_used_row_indices,
  SEXP parameters) {
  SEXP ret;
  R_API_BEGIN();
  int32_t len = static_cast<int32_t>(Rf_asInteger(len_used_row_indices));
  std::vector<int32_t> idxvec(len);
  // convert from one-based to zero-based index
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
  for (int32_t i = 0; i < len; ++i) {
    idxvec[i] = static_cast<int32_t>(INTEGER(used_row_indices)[i] - 1);
  }
  DatasetHandle res = nullptr;
  CHECK_CALL(LGBM_DatasetGetSubset(R_ExternalPtrAddr(handle),
    idxvec.data(), len, CHAR(Rf_asChar(parameters)),
    &res));
  ret = PROTECT(R_MakeExternalPtr(res, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _DatasetFinalizer, TRUE);
  UNPROTECT(1);
  return ret;
  R_API_END();
}

SEXP LGBM_DatasetSetFeatureNames_R(SEXP handle,
  SEXP feature_names) {
  R_API_BEGIN();
  auto vec_names = Split(CHAR(Rf_asChar(feature_names)), '\t');
  std::vector<const char*> vec_sptr;
  int len = static_cast<int>(vec_names.size());
  for (int i = 0; i < len; ++i) {
    vec_sptr.push_back(vec_names[i].c_str());
  }
  CHECK_CALL(LGBM_DatasetSetFeatureNames(R_ExternalPtrAddr(handle),
    vec_sptr.data(), len));
  R_API_END();
}

SEXP LGBM_DatasetGetFeatureNames_R(SEXP handle) {
  SEXP feature_names;
  R_API_BEGIN();
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
  feature_names = PROTECT(Rf_allocVector(STRSXP, len));
  for (int i = 0; i < len; ++i) {
    SET_STRING_ELT(feature_names, i, Rf_mkChar(ptr_names[i]));
  }
  UNPROTECT(1);
  return feature_names;
  R_API_END();
}

SEXP LGBM_DatasetSaveBinary_R(SEXP handle,
  SEXP filename) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetSaveBinary(R_ExternalPtrAddr(handle),
    CHAR(Rf_asChar(filename))));
  R_API_END();
}

SEXP LGBM_DatasetFree_R(SEXP handle) {
  R_API_BEGIN();
  if (!Rf_isNull(handle) && R_ExternalPtrAddr(handle)) {
    CHECK_CALL(LGBM_DatasetFree(R_ExternalPtrAddr(handle)));
    R_ClearExternalPtr(handle);
  }
  R_API_END();
}

SEXP LGBM_DatasetSetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data,
  SEXP num_element) {
  R_API_BEGIN();
  int len = Rf_asInteger(num_element);
  const char* name = CHAR(Rf_asChar(field_name));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    std::vector<int32_t> vec(len);
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
    for (int i = 0; i < len; ++i) {
      vec[i] = static_cast<int32_t>(INTEGER(field_data)[i]);
    }
    CHECK_CALL(LGBM_DatasetSetField(R_ExternalPtrAddr(handle), name, vec.data(), len, C_API_DTYPE_INT32));
  } else if (!strcmp("init_score", name)) {
    CHECK_CALL(LGBM_DatasetSetField(R_ExternalPtrAddr(handle), name, REAL(field_data), len, C_API_DTYPE_FLOAT64));
  } else {
    std::vector<float> vec(len);
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
    for (int i = 0; i < len; ++i) {
      vec[i] = static_cast<float>(REAL(field_data)[i]);
    }
    CHECK_CALL(LGBM_DatasetSetField(R_ExternalPtrAddr(handle), name, vec.data(), len, C_API_DTYPE_FLOAT32));
  }
  R_API_END();
}

SEXP LGBM_DatasetGetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data) {
  R_API_BEGIN();
  const char* name = CHAR(Rf_asChar(field_name));
  int out_len = 0;
  int out_type = 0;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_ExternalPtrAddr(handle), name, &out_len, &res, &out_type));

  if (!strcmp("group", name) || !strcmp("query", name)) {
    auto p_data = reinterpret_cast<const int32_t*>(res);
    // convert from boundaries to size
#pragma omp parallel for schedule(static, 512) if (out_len >= 1024)
    for (int i = 0; i < out_len - 1; ++i) {
      INTEGER(field_data)[i] = p_data[i + 1] - p_data[i];
    }
  } else if (!strcmp("init_score", name)) {
    auto p_data = reinterpret_cast<const double*>(res);
#pragma omp parallel for schedule(static, 512) if (out_len >= 1024)
    for (int i = 0; i < out_len; ++i) {
      REAL(field_data)[i] = p_data[i];
    }
  } else {
    auto p_data = reinterpret_cast<const float*>(res);
#pragma omp parallel for schedule(static, 512) if (out_len >= 1024)
    for (int i = 0; i < out_len; ++i) {
      REAL(field_data)[i] = p_data[i];
    }
  }
  R_API_END();
}

SEXP LGBM_DatasetGetFieldSize_R(SEXP handle,
  SEXP field_name,
  SEXP out) {
  R_API_BEGIN();
  const char* name = CHAR(Rf_asChar(field_name));
  int out_len = 0;
  int out_type = 0;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_ExternalPtrAddr(handle), name, &out_len, &res, &out_type));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    out_len -= 1;
  }
  INTEGER(out)[0] = out_len;
  R_API_END();
}

SEXP LGBM_DatasetUpdateParamChecking_R(SEXP old_params,
  SEXP new_params) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetUpdateParamChecking(CHAR(Rf_asChar(old_params)), CHAR(Rf_asChar(new_params))));
  R_API_END();
}

SEXP LGBM_DatasetGetNumData_R(SEXP handle, SEXP out) {
  int nrow;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumData(R_ExternalPtrAddr(handle), &nrow));
  INTEGER(out)[0] = nrow;
  R_API_END();
}

SEXP LGBM_DatasetGetNumFeature_R(SEXP handle,
  SEXP out) {
  int nfeature;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_ExternalPtrAddr(handle), &nfeature));
  INTEGER(out)[0] = nfeature;
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
  R_API_END();
}

SEXP LGBM_BoosterCreate_R(SEXP train_data,
  SEXP parameters) {
  SEXP ret;
  R_API_BEGIN();
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterCreate(R_ExternalPtrAddr(train_data), CHAR(Rf_asChar(parameters)), &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  UNPROTECT(1);
  return ret;
  R_API_END();
}

SEXP LGBM_BoosterCreateFromModelfile_R(SEXP filename) {
  SEXP ret;
  R_API_BEGIN();
  int out_num_iterations = 0;
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterCreateFromModelfile(CHAR(Rf_asChar(filename)), &out_num_iterations, &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  UNPROTECT(1);
  return ret;
  R_API_END();
}

SEXP LGBM_BoosterLoadModelFromString_R(SEXP model_str) {
  SEXP ret;
  R_API_BEGIN();
  int out_num_iterations = 0;
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterLoadModelFromString(CHAR(Rf_asChar(model_str)), &out_num_iterations, &handle));
  ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
  UNPROTECT(1);
  return ret;
  R_API_END();
}

SEXP LGBM_BoosterMerge_R(SEXP handle,
  SEXP other_handle) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterMerge(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(other_handle)));
  R_API_END();
}

SEXP LGBM_BoosterAddValidData_R(SEXP handle,
  SEXP valid_data) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterAddValidData(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(valid_data)));
  R_API_END();
}

SEXP LGBM_BoosterResetTrainingData_R(SEXP handle,
  SEXP train_data) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetTrainingData(R_ExternalPtrAddr(handle), R_ExternalPtrAddr(train_data)));
  R_API_END();
}

SEXP LGBM_BoosterResetParameter_R(SEXP handle,
  SEXP parameters) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetParameter(R_ExternalPtrAddr(handle), CHAR(Rf_asChar(parameters))));
  R_API_END();
}

SEXP LGBM_BoosterGetNumClasses_R(SEXP handle,
  SEXP out) {
  int num_class;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetNumClasses(R_ExternalPtrAddr(handle), &num_class));
  INTEGER(out)[0] = num_class;
  R_API_END();
}

SEXP LGBM_BoosterUpdateOneIter_R(SEXP handle) {
  int is_finished = 0;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterUpdateOneIter(R_ExternalPtrAddr(handle), &is_finished));
  R_API_END();
}

SEXP LGBM_BoosterUpdateOneIterCustom_R(SEXP handle,
  SEXP grad,
  SEXP hess,
  SEXP len) {
  int is_finished = 0;
  R_API_BEGIN();
  int int_len = Rf_asInteger(len);
  std::vector<float> tgrad(int_len), thess(int_len);
#pragma omp parallel for schedule(static, 512) if (int_len >= 1024)
  for (int j = 0; j < int_len; ++j) {
    tgrad[j] = static_cast<float>(REAL(grad)[j]);
    thess[j] = static_cast<float>(REAL(hess)[j]);
  }
  CHECK_CALL(LGBM_BoosterUpdateOneIterCustom(R_ExternalPtrAddr(handle), tgrad.data(), thess.data(), &is_finished));
  R_API_END();
}

SEXP LGBM_BoosterRollbackOneIter_R(SEXP handle) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterRollbackOneIter(R_ExternalPtrAddr(handle)));
  R_API_END();
}

SEXP LGBM_BoosterGetCurrentIteration_R(SEXP handle,
  SEXP out) {
  int out_iteration;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetCurrentIteration(R_ExternalPtrAddr(handle), &out_iteration));
  INTEGER(out)[0] = out_iteration;
  R_API_END();
}

SEXP LGBM_BoosterGetUpperBoundValue_R(SEXP handle,
  SEXP out_result) {
  R_API_BEGIN();
  double* ptr_ret = REAL(out_result);
  CHECK_CALL(LGBM_BoosterGetUpperBoundValue(R_ExternalPtrAddr(handle), ptr_ret));
  R_API_END();
}

SEXP LGBM_BoosterGetLowerBoundValue_R(SEXP handle,
  SEXP out_result) {
  R_API_BEGIN();
  double* ptr_ret = REAL(out_result);
  CHECK_CALL(LGBM_BoosterGetLowerBoundValue(R_ExternalPtrAddr(handle), ptr_ret));
  R_API_END();
}

SEXP LGBM_BoosterGetEvalNames_R(SEXP handle) {
  SEXP eval_names;
  R_API_BEGIN();
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
  eval_names = PROTECT(Rf_allocVector(STRSXP, len));
  for (int i = 0; i < len; ++i) {
    SET_STRING_ELT(eval_names, i, Rf_mkChar(ptr_names[i]));
  }
  UNPROTECT(1);
  return eval_names;
  R_API_END();
}

SEXP LGBM_BoosterGetEval_R(SEXP handle,
  SEXP data_idx,
  SEXP out_result) {
  R_API_BEGIN();
  int len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_ExternalPtrAddr(handle), &len));
  double* ptr_ret = REAL(out_result);
  int out_len;
  CHECK_CALL(LGBM_BoosterGetEval(R_ExternalPtrAddr(handle), Rf_asInteger(data_idx), &out_len, ptr_ret));
  CHECK_EQ(out_len, len);
  R_API_END();
}

SEXP LGBM_BoosterGetNumPredict_R(SEXP handle,
  SEXP data_idx,
  SEXP out) {
  R_API_BEGIN();
  int64_t len;
  CHECK_CALL(LGBM_BoosterGetNumPredict(R_ExternalPtrAddr(handle), Rf_asInteger(data_idx), &len));
  INTEGER(out)[0] = static_cast<int>(len);
  R_API_END();
}

SEXP LGBM_BoosterGetPredict_R(SEXP handle,
  SEXP data_idx,
  SEXP out_result) {
  R_API_BEGIN();
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterGetPredict(R_ExternalPtrAddr(handle), Rf_asInteger(data_idx), &out_len, ptr_ret));
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
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  CHECK_CALL(LGBM_BoosterPredictForFile(R_ExternalPtrAddr(handle), CHAR(Rf_asChar(data_filename)),
    Rf_asInteger(data_has_header), pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), CHAR(Rf_asChar(parameter)),
    CHAR(Rf_asChar(result_filename))));
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
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  int64_t len = 0;
  CHECK_CALL(LGBM_BoosterCalcNumPredict(R_ExternalPtrAddr(handle), Rf_asInteger(num_row),
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), &len));
  INTEGER(out_len)[0] = static_cast<int>(len);
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
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);

  const int* p_indptr = INTEGER(indptr);
  const int32_t* p_indices = reinterpret_cast<const int32_t*>(INTEGER(indices));
  const double* p_data = REAL(data);

  int64_t nindptr = static_cast<int64_t>(Rf_asInteger(num_indptr));
  int64_t ndata = static_cast<int64_t>(Rf_asInteger(nelem));
  int64_t nrow = static_cast<int64_t>(Rf_asInteger(num_row));
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForCSC(R_ExternalPtrAddr(handle),
    p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), CHAR(Rf_asChar(parameter)), &out_len, ptr_ret));
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
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);

  int32_t nrow = static_cast<int32_t>(Rf_asInteger(num_row));
  int32_t ncol = static_cast<int32_t>(Rf_asInteger(num_col));

  const double* p_mat = REAL(data);
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForMat(R_ExternalPtrAddr(handle),
    p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), CHAR(Rf_asChar(parameter)), &out_len, ptr_ret));

  R_API_END();
}

SEXP LGBM_BoosterSaveModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP filename) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterSaveModel(R_ExternalPtrAddr(handle), 0, Rf_asInteger(num_iteration), Rf_asInteger(feature_importance_type), CHAR(Rf_asChar(filename))));
  R_API_END();
}

SEXP LGBM_BoosterSaveModelToString_R(SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type) {
  SEXP model_str;
  R_API_BEGIN();
  int64_t out_len = 0;
  int64_t buf_len = 1024 * 1024;
  int num_iter = Rf_asInteger(num_iteration);
  int importance_type = Rf_asInteger(feature_importance_type);
  std::vector<char> inner_char_buf(buf_len);
  CHECK_CALL(LGBM_BoosterSaveModelToString(R_ExternalPtrAddr(handle), 0, num_iter, importance_type, buf_len, &out_len, inner_char_buf.data()));
  // if the model string was larger than the initial buffer, allocate a bigger buffer and try again
  if (out_len > buf_len) {
    inner_char_buf.resize(out_len);
    CHECK_CALL(LGBM_BoosterSaveModelToString(R_ExternalPtrAddr(handle), 0, num_iter, importance_type, out_len, &out_len, inner_char_buf.data()));
  }
  model_str = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(model_str, 0, Rf_mkChar(inner_char_buf.data()));
  UNPROTECT(1);
  return model_str;
  R_API_END();
}

SEXP LGBM_BoosterDumpModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type) {
  SEXP model_str;
  R_API_BEGIN();
  int64_t out_len = 0;
  int64_t buf_len = 1024 * 1024;
  int num_iter = Rf_asInteger(num_iteration);
  int importance_type = Rf_asInteger(feature_importance_type);
  std::vector<char> inner_char_buf(buf_len);
  CHECK_CALL(LGBM_BoosterDumpModel(R_ExternalPtrAddr(handle), 0, num_iter, importance_type, buf_len, &out_len, inner_char_buf.data()));
  // if the model string was larger than the initial buffer, allocate a bigger buffer and try again
  if (out_len > buf_len) {
    inner_char_buf.resize(out_len);
    CHECK_CALL(LGBM_BoosterDumpModel(R_ExternalPtrAddr(handle), 0, num_iter, importance_type, out_len, &out_len, inner_char_buf.data()));
  }
  model_str = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(model_str, 0, Rf_mkChar(inner_char_buf.data()));
  UNPROTECT(1);
  return model_str;
  R_API_END();
}

// .Call() calls
static const R_CallMethodDef CallEntries[] = {
  {"LGBM_HandleIsNull_R"              , (DL_FUNC) &LGBM_HandleIsNull_R              , 1},
  {"LGBM_DatasetCreateFromFile_R"     , (DL_FUNC) &LGBM_DatasetCreateFromFile_R     , 3},
  {"LGBM_DatasetCreateFromCSC_R"      , (DL_FUNC) &LGBM_DatasetCreateFromCSC_R      , 9},
  {"LGBM_DatasetCreateFromMat_R"      , (DL_FUNC) &LGBM_DatasetCreateFromMat_R      , 6},
  {"LGBM_DatasetGetSubset_R"          , (DL_FUNC) &LGBM_DatasetGetSubset_R          , 4},
  {"LGBM_DatasetSetFeatureNames_R"    , (DL_FUNC) &LGBM_DatasetSetFeatureNames_R    , 2},
  {"LGBM_DatasetGetFeatureNames_R"    , (DL_FUNC) &LGBM_DatasetGetFeatureNames_R    , 1},
  {"LGBM_DatasetSaveBinary_R"         , (DL_FUNC) &LGBM_DatasetSaveBinary_R         , 2},
  {"LGBM_DatasetFree_R"               , (DL_FUNC) &LGBM_DatasetFree_R               , 1},
  {"LGBM_DatasetSetField_R"           , (DL_FUNC) &LGBM_DatasetSetField_R           , 4},
  {"LGBM_DatasetGetFieldSize_R"       , (DL_FUNC) &LGBM_DatasetGetFieldSize_R       , 3},
  {"LGBM_DatasetGetField_R"           , (DL_FUNC) &LGBM_DatasetGetField_R           , 3},
  {"LGBM_DatasetUpdateParamChecking_R", (DL_FUNC) &LGBM_DatasetUpdateParamChecking_R, 2},
  {"LGBM_DatasetGetNumData_R"         , (DL_FUNC) &LGBM_DatasetGetNumData_R         , 2},
  {"LGBM_DatasetGetNumFeature_R"      , (DL_FUNC) &LGBM_DatasetGetNumFeature_R      , 2},
  {"LGBM_BoosterCreate_R"             , (DL_FUNC) &LGBM_BoosterCreate_R             , 2},
  {"LGBM_BoosterFree_R"               , (DL_FUNC) &LGBM_BoosterFree_R               , 1},
  {"LGBM_BoosterCreateFromModelfile_R", (DL_FUNC) &LGBM_BoosterCreateFromModelfile_R, 1},
  {"LGBM_BoosterLoadModelFromString_R", (DL_FUNC) &LGBM_BoosterLoadModelFromString_R, 1},
  {"LGBM_BoosterMerge_R"              , (DL_FUNC) &LGBM_BoosterMerge_R              , 2},
  {"LGBM_BoosterAddValidData_R"       , (DL_FUNC) &LGBM_BoosterAddValidData_R       , 2},
  {"LGBM_BoosterResetTrainingData_R"  , (DL_FUNC) &LGBM_BoosterResetTrainingData_R  , 2},
  {"LGBM_BoosterResetParameter_R"     , (DL_FUNC) &LGBM_BoosterResetParameter_R     , 2},
  {"LGBM_BoosterGetNumClasses_R"      , (DL_FUNC) &LGBM_BoosterGetNumClasses_R      , 2},
  {"LGBM_BoosterUpdateOneIter_R"      , (DL_FUNC) &LGBM_BoosterUpdateOneIter_R      , 1},
  {"LGBM_BoosterUpdateOneIterCustom_R", (DL_FUNC) &LGBM_BoosterUpdateOneIterCustom_R, 4},
  {"LGBM_BoosterRollbackOneIter_R"    , (DL_FUNC) &LGBM_BoosterRollbackOneIter_R    , 1},
  {"LGBM_BoosterGetCurrentIteration_R", (DL_FUNC) &LGBM_BoosterGetCurrentIteration_R, 2},
  {"LGBM_BoosterGetUpperBoundValue_R" , (DL_FUNC) &LGBM_BoosterGetUpperBoundValue_R , 2},
  {"LGBM_BoosterGetLowerBoundValue_R" , (DL_FUNC) &LGBM_BoosterGetLowerBoundValue_R , 2},
  {"LGBM_BoosterGetEvalNames_R"       , (DL_FUNC) &LGBM_BoosterGetEvalNames_R       , 1},
  {"LGBM_BoosterGetEval_R"            , (DL_FUNC) &LGBM_BoosterGetEval_R            , 3},
  {"LGBM_BoosterGetNumPredict_R"      , (DL_FUNC) &LGBM_BoosterGetNumPredict_R      , 3},
  {"LGBM_BoosterGetPredict_R"         , (DL_FUNC) &LGBM_BoosterGetPredict_R         , 3},
  {"LGBM_BoosterPredictForFile_R"     , (DL_FUNC) &LGBM_BoosterPredictForFile_R     , 10},
  {"LGBM_BoosterCalcNumPredict_R"     , (DL_FUNC) &LGBM_BoosterCalcNumPredict_R     , 8},
  {"LGBM_BoosterPredictForCSC_R"      , (DL_FUNC) &LGBM_BoosterPredictForCSC_R      , 14},
  {"LGBM_BoosterPredictForMat_R"      , (DL_FUNC) &LGBM_BoosterPredictForMat_R      , 11},
  {"LGBM_BoosterSaveModel_R"          , (DL_FUNC) &LGBM_BoosterSaveModel_R          , 4},
  {"LGBM_BoosterSaveModelToString_R"  , (DL_FUNC) &LGBM_BoosterSaveModelToString_R  , 3},
  {"LGBM_BoosterDumpModel_R"          , (DL_FUNC) &LGBM_BoosterDumpModel_R          , 3},
  {NULL, NULL, 0}
};

LIGHTGBM_C_EXPORT void R_init_lightgbm(DllInfo *dll);

void R_init_lightgbm(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
