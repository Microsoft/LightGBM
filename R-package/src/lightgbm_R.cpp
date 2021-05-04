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

using LightGBM::Common::Join;
using LightGBM::Common::Split;
using LightGBM::Log;

LGBM_SE EncodeChar(LGBM_SE dest, const char* src, SEXP buf_len, SEXP actual_len, size_t str_len) {
  if (str_len > INT32_MAX) {
    Log::Fatal("Don't support large string in R-package");
  }
  INTEGER(actual_len)[0] = static_cast<int>(str_len);
  if (Rf_asInteger(buf_len) < static_cast<int>(str_len)) {
    return dest;
  }
  auto ptr = R_CHAR_PTR(dest);
  std::memcpy(ptr, src, str_len);
  return dest;
}

SEXP LGBM_GetLastError_R() {
  SEXP out;
  out = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(out, 0, Rf_mkChar(LGBM_GetLastError()));
  UNPROTECT(1);
  return out;
}

SEXP LGBM_DatasetCreateFromFile_R(SEXP filename,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out) {
  R_API_BEGIN();
  DatasetHandle handle = nullptr;
  CHECK_CALL(LGBM_DatasetCreateFromFile(CHAR(Rf_asChar(filename)), R_CHAR_PTR(parameters),
    R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_DatasetCreateFromCSC_R(SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out) {
  R_API_BEGIN();
  const int* p_indptr = INTEGER(indptr);
  const int* p_indices = INTEGER(indices);
  const double* p_data = REAL(data);

  int64_t nindptr = static_cast<int64_t>(Rf_asInteger(num_indptr));
  int64_t ndata = static_cast<int64_t>(Rf_asInteger(nelem));
  int64_t nrow = static_cast<int64_t>(Rf_asInteger(num_row));
  DatasetHandle handle = nullptr;
  CHECK_CALL(LGBM_DatasetCreateFromCSC(p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, R_CHAR_PTR(parameters), R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_DatasetCreateFromMat_R(SEXP data,
  SEXP num_row,
  SEXP num_col,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out) {
  R_API_BEGIN();
  int32_t nrow = static_cast<int32_t>(Rf_asInteger(num_row));
  int32_t ncol = static_cast<int32_t>(Rf_asInteger(num_col));
  double* p_mat = REAL(data);
  DatasetHandle handle = nullptr;
  CHECK_CALL(LGBM_DatasetCreateFromMat(p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    R_CHAR_PTR(parameters), R_GET_PTR(reference), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_DatasetGetSubset_R(LGBM_SE handle,
  SEXP used_row_indices,
  SEXP len_used_row_indices,
  LGBM_SE parameters,
  LGBM_SE out) {
  R_API_BEGIN();
  int len = Rf_asInteger(len_used_row_indices);
  std::vector<int> idxvec(len);
  // convert from one-based to  zero-based index
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
  for (int i = 0; i < len; ++i) {
    idxvec[i] = INTEGER(used_row_indices)[i] - 1;
  }
  DatasetHandle res = nullptr;
  CHECK_CALL(LGBM_DatasetGetSubset(R_GET_PTR(handle),
    idxvec.data(), len, R_CHAR_PTR(parameters),
    &res));
  R_SET_PTR(out, res);
  R_API_END();
}

SEXP LGBM_DatasetSetFeatureNames_R(LGBM_SE handle,
  SEXP feature_names) {
  R_API_BEGIN();
  auto vec_names = Split(CHAR(Rf_asChar(feature_names)), '\t');
  std::vector<const char*> vec_sptr;
  int len = static_cast<int>(vec_names.size());
  for (int i = 0; i < len; ++i) {
    vec_sptr.push_back(vec_names[i].c_str());
  }
  CHECK_CALL(LGBM_DatasetSetFeatureNames(R_GET_PTR(handle),
    vec_sptr.data(), len));
  R_API_END();
}

SEXP LGBM_DatasetGetFeatureNames_R(LGBM_SE handle,
  SEXP buf_len,
  SEXP actual_len,
  LGBM_SE feature_names) {
  R_API_BEGIN();
  int len = 0;
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_GET_PTR(handle), &len));
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
      R_GET_PTR(handle),
      len, &out_len,
      reserved_string_size, &required_string_size,
      ptr_names.data()));
  CHECK_EQ(len, out_len);
  CHECK_GE(reserved_string_size, required_string_size);
  auto merge_str = Join<char*>(ptr_names, "\t");
  EncodeChar(feature_names, merge_str.c_str(), buf_len, actual_len, merge_str.size() + 1);
  R_API_END();
}

SEXP LGBM_DatasetSaveBinary_R(LGBM_SE handle,
  SEXP filename) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetSaveBinary(R_GET_PTR(handle),
    CHAR(Rf_asChar(filename))));
  R_API_END();
}

SEXP LGBM_DatasetFree_R(LGBM_SE handle) {
  R_API_BEGIN();
  if (R_GET_PTR(handle) != nullptr) {
    CHECK_CALL(LGBM_DatasetFree(R_GET_PTR(handle)));
    R_SET_PTR(handle, nullptr);
  }
  R_API_END();
}

SEXP LGBM_DatasetSetField_R(LGBM_SE handle,
  SEXP field_name,
  SEXP field_data,
  SEXP num_element) {
  R_API_BEGIN();
  int len = static_cast<int>(Rf_asInteger(num_element));
  const char* name = CHAR(Rf_asChar(field_name));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    std::vector<int32_t> vec(len);
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
    for (int i = 0; i < len; ++i) {
      vec[i] = static_cast<int32_t>(INTEGER(field_data)[i]);
    }
    CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, vec.data(), len, C_API_DTYPE_INT32));
  } else if (!strcmp("init_score", name)) {
    CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, REAL(field_data), len, C_API_DTYPE_FLOAT64));
  } else {
    std::vector<float> vec(len);
#pragma omp parallel for schedule(static, 512) if (len >= 1024)
    for (int i = 0; i < len; ++i) {
      vec[i] = static_cast<float>(REAL(field_data)[i]);
    }
    CHECK_CALL(LGBM_DatasetSetField(R_GET_PTR(handle), name, vec.data(), len, C_API_DTYPE_FLOAT32));
  }
  R_API_END();
}

SEXP LGBM_DatasetGetField_R(LGBM_SE handle,
  SEXP field_name,
  SEXP field_data) {
  R_API_BEGIN();
  const char* name = CHAR(Rf_asChar(field_name));
  int out_len = 0;
  int out_type = 0;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_GET_PTR(handle), name, &out_len, &res, &out_type));

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

SEXP LGBM_DatasetGetFieldSize_R(LGBM_SE handle,
  SEXP field_name,
  SEXP out) {
  R_API_BEGIN();
  const char* name = CHAR(Rf_asChar(field_name));
  int out_len = 0;
  int out_type = 0;
  const void* res;
  CHECK_CALL(LGBM_DatasetGetField(R_GET_PTR(handle), name, &out_len, &res, &out_type));
  if (!strcmp("group", name) || !strcmp("query", name)) {
    out_len -= 1;
  }
  INTEGER(out)[0] = static_cast<int>(out_len);
  R_API_END();
}

SEXP LGBM_DatasetUpdateParamChecking_R(LGBM_SE old_params,
  LGBM_SE new_params) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetUpdateParamChecking(R_CHAR_PTR(old_params), R_CHAR_PTR(new_params)));
  R_API_END();
}

SEXP LGBM_DatasetGetNumData_R(LGBM_SE handle, SEXP out) {
  int nrow;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumData(R_GET_PTR(handle), &nrow));
  INTEGER(out)[0] = static_cast<int>(nrow);
  R_API_END();
}

SEXP LGBM_DatasetGetNumFeature_R(LGBM_SE handle,
  SEXP out) {
  int nfeature;
  R_API_BEGIN();
  CHECK_CALL(LGBM_DatasetGetNumFeature(R_GET_PTR(handle), &nfeature));
  INTEGER(out)[0] = static_cast<int>(nfeature);
  R_API_END();
}

// --- start Booster interfaces

SEXP LGBM_BoosterFree_R(LGBM_SE handle) {
  R_API_BEGIN();
  if (R_GET_PTR(handle) != nullptr) {
    CHECK_CALL(LGBM_BoosterFree(R_GET_PTR(handle)));
    R_SET_PTR(handle, nullptr);
  }
  R_API_END();
}

SEXP LGBM_BoosterCreate_R(LGBM_SE train_data,
  LGBM_SE parameters,
  LGBM_SE out) {
  R_API_BEGIN();
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterCreate(R_GET_PTR(train_data), R_CHAR_PTR(parameters), &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_BoosterCreateFromModelfile_R(SEXP filename,
  LGBM_SE out) {
  R_API_BEGIN();
  int out_num_iterations = 0;
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterCreateFromModelfile(CHAR(Rf_asChar(filename)), &out_num_iterations, &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_BoosterLoadModelFromString_R(SEXP model_str,
  LGBM_SE out) {
  R_API_BEGIN();
  int out_num_iterations = 0;
  BoosterHandle handle = nullptr;
  CHECK_CALL(LGBM_BoosterLoadModelFromString(CHAR(Rf_asChar(model_str)), &out_num_iterations, &handle));
  R_SET_PTR(out, handle);
  R_API_END();
}

SEXP LGBM_BoosterMerge_R(LGBM_SE handle,
  LGBM_SE other_handle) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterMerge(R_GET_PTR(handle), R_GET_PTR(other_handle)));
  R_API_END();
}

SEXP LGBM_BoosterAddValidData_R(LGBM_SE handle,
  LGBM_SE valid_data) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterAddValidData(R_GET_PTR(handle), R_GET_PTR(valid_data)));
  R_API_END();
}

SEXP LGBM_BoosterResetTrainingData_R(LGBM_SE handle,
  LGBM_SE train_data) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetTrainingData(R_GET_PTR(handle), R_GET_PTR(train_data)));
  R_API_END();
}

SEXP LGBM_BoosterResetParameter_R(LGBM_SE handle,
  LGBM_SE parameters) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterResetParameter(R_GET_PTR(handle), R_CHAR_PTR(parameters)));
  R_API_END();
}

SEXP LGBM_BoosterGetNumClasses_R(LGBM_SE handle,
  SEXP out) {
  int num_class;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetNumClasses(R_GET_PTR(handle), &num_class));
  INTEGER(out)[0] = static_cast<int>(num_class);
  R_API_END();
}

SEXP LGBM_BoosterUpdateOneIter_R(LGBM_SE handle) {
  int is_finished = 0;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterUpdateOneIter(R_GET_PTR(handle), &is_finished));
  R_API_END();
}

SEXP LGBM_BoosterUpdateOneIterCustom_R(LGBM_SE handle,
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
  CHECK_CALL(LGBM_BoosterUpdateOneIterCustom(R_GET_PTR(handle), tgrad.data(), thess.data(), &is_finished));
  R_API_END();
}

SEXP LGBM_BoosterRollbackOneIter_R(LGBM_SE handle) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterRollbackOneIter(R_GET_PTR(handle)));
  R_API_END();
}

SEXP LGBM_BoosterGetCurrentIteration_R(LGBM_SE handle,
  SEXP out) {
  int out_iteration;
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterGetCurrentIteration(R_GET_PTR(handle), &out_iteration));
  INTEGER(out)[0] = static_cast<int>(out_iteration);
  R_API_END();
}

SEXP LGBM_BoosterGetUpperBoundValue_R(LGBM_SE handle,
  SEXP out_result) {
  R_API_BEGIN();
  double* ptr_ret = REAL(out_result);
  CHECK_CALL(LGBM_BoosterGetUpperBoundValue(R_GET_PTR(handle), ptr_ret));
  R_API_END();
}

SEXP LGBM_BoosterGetLowerBoundValue_R(LGBM_SE handle,
  SEXP out_result) {
  R_API_BEGIN();
  double* ptr_ret = REAL(out_result);
  CHECK_CALL(LGBM_BoosterGetLowerBoundValue(R_GET_PTR(handle), ptr_ret));
  R_API_END();
}

SEXP LGBM_BoosterGetEvalNames_R(LGBM_SE handle,
  SEXP buf_len,
  SEXP actual_len,
  LGBM_SE eval_names) {
  R_API_BEGIN();
  int len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_GET_PTR(handle), &len));

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
      R_GET_PTR(handle),
      len, &out_len,
      reserved_string_size, &required_string_size,
      ptr_names.data()));
  CHECK_EQ(out_len, len);
  CHECK_GE(reserved_string_size, required_string_size);
  auto merge_names = Join<char*>(ptr_names, "\t");
  EncodeChar(eval_names, merge_names.c_str(), buf_len, actual_len, merge_names.size() + 1);
  R_API_END();
}

SEXP LGBM_BoosterGetEval_R(LGBM_SE handle,
  SEXP data_idx,
  SEXP out_result) {
  R_API_BEGIN();
  int len;
  CHECK_CALL(LGBM_BoosterGetEvalCounts(R_GET_PTR(handle), &len));
  double* ptr_ret = REAL(out_result);
  int out_len;
  CHECK_CALL(LGBM_BoosterGetEval(R_GET_PTR(handle), Rf_asInteger(data_idx), &out_len, ptr_ret));
  CHECK_EQ(out_len, len);
  R_API_END();
}

SEXP LGBM_BoosterGetNumPredict_R(LGBM_SE handle,
  SEXP data_idx,
  SEXP out) {
  R_API_BEGIN();
  int64_t len;
  CHECK_CALL(LGBM_BoosterGetNumPredict(R_GET_PTR(handle), Rf_asInteger(data_idx), &len));
  INTEGER(out)[0] = static_cast<int>(len);
  R_API_END();
}

SEXP LGBM_BoosterGetPredict_R(LGBM_SE handle,
  SEXP data_idx,
  SEXP out_result) {
  R_API_BEGIN();
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterGetPredict(R_GET_PTR(handle), Rf_asInteger(data_idx), &out_len, ptr_ret));
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

SEXP LGBM_BoosterPredictForFile_R(LGBM_SE handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  LGBM_SE parameter,
  SEXP result_filename) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);
  CHECK_CALL(LGBM_BoosterPredictForFile(R_GET_PTR(handle), CHAR(Rf_asChar(data_filename)),
    Rf_asInteger(data_has_header), pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), R_CHAR_PTR(parameter),
    CHAR(Rf_asChar(result_filename))));
  R_API_END();
}

SEXP LGBM_BoosterCalcNumPredict_R(LGBM_SE handle,
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
  CHECK_CALL(LGBM_BoosterCalcNumPredict(R_GET_PTR(handle), Rf_asInteger(num_row),
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), &len));
  INTEGER(out_len)[0] = static_cast<int>(len);
  R_API_END();
}

SEXP LGBM_BoosterPredictForCSC_R(LGBM_SE handle,
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
  LGBM_SE parameter,
  SEXP out_result) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);

  const int* p_indptr = INTEGER(indptr);
  const int* p_indices = INTEGER(indices);
  const double* p_data = REAL(data);

  int64_t nindptr = Rf_asInteger(num_indptr);
  int64_t ndata = Rf_asInteger(nelem);
  int64_t nrow = Rf_asInteger(num_row);
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForCSC(R_GET_PTR(handle),
    p_indptr, C_API_DTYPE_INT32, p_indices,
    p_data, C_API_DTYPE_FLOAT64, nindptr, ndata,
    nrow, pred_type,  Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), R_CHAR_PTR(parameter), &out_len, ptr_ret));
  R_API_END();
}

SEXP LGBM_BoosterPredictForMat_R(LGBM_SE handle,
  SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  LGBM_SE parameter,
  SEXP out_result) {
  R_API_BEGIN();
  int pred_type = GetPredictType(is_rawscore, is_leafidx, is_predcontrib);

  int32_t nrow = Rf_asInteger(num_row);
  int32_t ncol = Rf_asInteger(num_col);

  const double* p_mat = REAL(data);
  double* ptr_ret = REAL(out_result);
  int64_t out_len;
  CHECK_CALL(LGBM_BoosterPredictForMat(R_GET_PTR(handle),
    p_mat, C_API_DTYPE_FLOAT64, nrow, ncol, COL_MAJOR,
    pred_type, Rf_asInteger(start_iteration), Rf_asInteger(num_iteration), R_CHAR_PTR(parameter), &out_len, ptr_ret));

  R_API_END();
}

SEXP LGBM_BoosterSaveModel_R(LGBM_SE handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP filename) {
  R_API_BEGIN();
  CHECK_CALL(LGBM_BoosterSaveModel(R_GET_PTR(handle), 0, Rf_asInteger(num_iteration), Rf_asInteger(feature_importance_type), CHAR(Rf_asChar(filename))));
  R_API_END();
}

SEXP LGBM_BoosterSaveModelToString_R(LGBM_SE handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP buffer_len,
  SEXP actual_len,
  LGBM_SE out_str) {
  R_API_BEGIN();
  int64_t out_len = 0;
  int64_t buf_len = static_cast<int64_t>(Rf_asInteger(buffer_len));
  std::vector<char> inner_char_buf(buf_len);
  CHECK_CALL(LGBM_BoosterSaveModelToString(R_GET_PTR(handle), 0, Rf_asInteger(num_iteration), Rf_asInteger(feature_importance_type), buf_len, &out_len, inner_char_buf.data()));
  EncodeChar(out_str, inner_char_buf.data(), buffer_len, actual_len, static_cast<size_t>(out_len));
  R_API_END();
}

SEXP LGBM_BoosterDumpModel_R(LGBM_SE handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP buffer_len,
  SEXP actual_len,
  LGBM_SE out_str) {
  R_API_BEGIN();
  int64_t out_len = 0;
  int64_t buf_len = static_cast<int64_t>(Rf_asInteger(buffer_len));
  std::vector<char> inner_char_buf(buf_len);
  CHECK_CALL(LGBM_BoosterDumpModel(R_GET_PTR(handle), 0, Rf_asInteger(num_iteration), Rf_asInteger(feature_importance_type), buf_len, &out_len, inner_char_buf.data()));
  EncodeChar(out_str, inner_char_buf.data(), buffer_len, actual_len, static_cast<size_t>(out_len));
  R_API_END();
}

// .Call() calls
static const R_CallMethodDef CallEntries[] = {
  {"LGBM_GetLastError_R"              , (DL_FUNC) &LGBM_GetLastError_R              , 0},
  {"LGBM_DatasetCreateFromFile_R"     , (DL_FUNC) &LGBM_DatasetCreateFromFile_R     , 4},
  {"LGBM_DatasetCreateFromCSC_R"      , (DL_FUNC) &LGBM_DatasetCreateFromCSC_R      , 9},
  {"LGBM_DatasetCreateFromMat_R"      , (DL_FUNC) &LGBM_DatasetCreateFromMat_R      , 6},
  {"LGBM_DatasetGetSubset_R"          , (DL_FUNC) &LGBM_DatasetGetSubset_R          , 5},
  {"LGBM_DatasetSetFeatureNames_R"    , (DL_FUNC) &LGBM_DatasetSetFeatureNames_R    , 2},
  {"LGBM_DatasetGetFeatureNames_R"    , (DL_FUNC) &LGBM_DatasetGetFeatureNames_R    , 4},
  {"LGBM_DatasetSaveBinary_R"         , (DL_FUNC) &LGBM_DatasetSaveBinary_R         , 2},
  {"LGBM_DatasetFree_R"               , (DL_FUNC) &LGBM_DatasetFree_R               , 1},
  {"LGBM_DatasetSetField_R"           , (DL_FUNC) &LGBM_DatasetSetField_R           , 4},
  {"LGBM_DatasetGetFieldSize_R"       , (DL_FUNC) &LGBM_DatasetGetFieldSize_R       , 3},
  {"LGBM_DatasetGetField_R"           , (DL_FUNC) &LGBM_DatasetGetField_R           , 3},
  {"LGBM_DatasetUpdateParamChecking_R", (DL_FUNC) &LGBM_DatasetUpdateParamChecking_R, 2},
  {"LGBM_DatasetGetNumData_R"         , (DL_FUNC) &LGBM_DatasetGetNumData_R         , 2},
  {"LGBM_DatasetGetNumFeature_R"      , (DL_FUNC) &LGBM_DatasetGetNumFeature_R      , 2},
  {"LGBM_BoosterCreate_R"             , (DL_FUNC) &LGBM_BoosterCreate_R             , 3},
  {"LGBM_BoosterFree_R"               , (DL_FUNC) &LGBM_BoosterFree_R               , 1},
  {"LGBM_BoosterCreateFromModelfile_R", (DL_FUNC) &LGBM_BoosterCreateFromModelfile_R, 2},
  {"LGBM_BoosterLoadModelFromString_R", (DL_FUNC) &LGBM_BoosterLoadModelFromString_R, 2},
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
  {"LGBM_BoosterGetEvalNames_R"       , (DL_FUNC) &LGBM_BoosterGetEvalNames_R       , 4},
  {"LGBM_BoosterGetEval_R"            , (DL_FUNC) &LGBM_BoosterGetEval_R            , 3},
  {"LGBM_BoosterGetNumPredict_R"      , (DL_FUNC) &LGBM_BoosterGetNumPredict_R      , 3},
  {"LGBM_BoosterGetPredict_R"         , (DL_FUNC) &LGBM_BoosterGetPredict_R         , 3},
  {"LGBM_BoosterPredictForFile_R"     , (DL_FUNC) &LGBM_BoosterPredictForFile_R     , 10},
  {"LGBM_BoosterCalcNumPredict_R"     , (DL_FUNC) &LGBM_BoosterCalcNumPredict_R     , 8},
  {"LGBM_BoosterPredictForCSC_R"      , (DL_FUNC) &LGBM_BoosterPredictForCSC_R      , 14},
  {"LGBM_BoosterPredictForMat_R"      , (DL_FUNC) &LGBM_BoosterPredictForMat_R      , 11},
  {"LGBM_BoosterSaveModel_R"          , (DL_FUNC) &LGBM_BoosterSaveModel_R          , 4},
  {"LGBM_BoosterSaveModelToString_R"  , (DL_FUNC) &LGBM_BoosterSaveModelToString_R  , 6},
  {"LGBM_BoosterDumpModel_R"          , (DL_FUNC) &LGBM_BoosterDumpModel_R          , 6},
  {NULL, NULL, 0}
};

LIGHTGBM_C_EXPORT void R_init_lightgbm(DllInfo *dll);

void R_init_lightgbm(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
