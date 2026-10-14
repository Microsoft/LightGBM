/*!
 * Copyright (c) 2018-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2018-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
/* lightgbmlib.i */
%module lightgbmlib
%ignore LGBM_BoosterSaveModelToString;
%ignore LGBM_BoosterGetEvalNames;
%ignore LGBM_BoosterGetFeatureNames;
%{
/* Includes the header in the wrapper code */
#include "../include/LightGBM/export.h"
#include "../include/LightGBM/utils/log.h"
#include "../include/LightGBM/utils/common.h"
#include "../include/LightGBM/c_api.h"
%}

%include "various.i"
%include "carrays.i"
%include "cpointer.i"
%include "stdint.i"

/* Note: instead of using array_functions for string array we apply a typemap instead.
   Future char** parameter names should be added to the typemap.
*/
%apply char **STRING_ARRAY { char **feature_names, char **out_strs }

/* header files */
%include "../include/LightGBM/export.h"
%include "../include/LightGBM/c_api.h"

%typemap(in, numinputs = 0) JNIEnv *jenv %{
  $1 = jenv;
%}

%newobject LGBM_BoosterSaveModelToStringSWIG;
%newobject LGBM_BoosterDumpModelSWIG;

%inline %{
  char * LGBM_BoosterSaveModelToStringSWIG(BoosterHandle handle,
                                           int start_iteration,
                                           int num_iteration,
                                           int feature_importance_type,
                                           int64_t buffer_len,
                                           int64_t* out_len) {
    char* dst = new char[buffer_len];
    int result = LGBM_BoosterSaveModelToString(handle, start_iteration, num_iteration, feature_importance_type, buffer_len, out_len, dst);
    // Reallocate to use larger length
    if (*out_len > buffer_len) {
      delete [] dst;
      int64_t realloc_len = *out_len;
      dst = new char[realloc_len];
      result = LGBM_BoosterSaveModelToString(handle, start_iteration, num_iteration, feature_importance_type, realloc_len, out_len, dst);
    }
    if (result != 0) {
      delete [] dst;
      return nullptr;
    }
    return dst;
  }

  char * LGBM_BoosterDumpModelSWIG(BoosterHandle handle,
                                   int start_iteration,
                                   int num_iteration,
                                   int feature_importance_type,
                                   int64_t buffer_len,
                                   int64_t* out_len) {
    char* dst = new char[buffer_len];
    int result = LGBM_BoosterDumpModel(handle, start_iteration, num_iteration, feature_importance_type, buffer_len, out_len, dst);
    // Reallocate to use larger length
    if (*out_len > buffer_len) {
      delete [] dst;
      int64_t realloc_len = *out_len;
      dst = new char[realloc_len];
      result = LGBM_BoosterDumpModel(handle, start_iteration, num_iteration, feature_importance_type, realloc_len, out_len, dst);
    }
    if (result != 0) {
      delete [] dst;
      return nullptr;
    }
    return dst;
  }

  int LGBM_BoosterPredictForMatSingle(JNIEnv *jenv,
                                      jdoubleArray data,
                                      BoosterHandle handle,
                                      int data_type,
                                      int ncol,
                                      int is_row_major,
                                      int predict_type,
                                      int start_iteration,
                                      int num_iteration,
                                      const char* parameter,
                                      int64_t* out_len,
                                      double* out_result) {
    double* data0 = (double*)jenv->GetPrimitiveArrayCritical(data, 0);

    int ret = LGBM_BoosterPredictForMatSingleRow(handle, data0, data_type, ncol, is_row_major, predict_type, start_iteration,
                                                 num_iteration, parameter, out_len, out_result);

    jenv->ReleasePrimitiveArrayCritical(data, data0, JNI_ABORT);

    return ret;
  }

  /*! \brief Even faster variant of `LGBM_BoosterPredictForMatSingle`.
   *
   * Uses `LGBM_BoosterPredictForMatSingleRowFast` which is faster
   * than `LGBM_BoosterPredictForMatSingleRow` and the trick of
   * `LGBM_BoosterPredictForMatSingle` to capture the Java data array
   * using `GetPrimitiveArrayCritical`, which can yield faster access
   * to the array if the JVM passes the actual address to the C++ side
   * instead of performing a copy.
   */
  int LGBM_BoosterPredictForMatSingleRowFastCriticalSWIG(JNIEnv *jenv,
                                                         jdoubleArray data,
                                                         FastConfigHandle handle,
                                                         int64_t* out_len,
                                                         double* out_result) {
    double* data0 = (double*)jenv->GetPrimitiveArrayCritical(data, 0);

    int ret = LGBM_BoosterPredictForMatSingleRowFast(handle, data0, out_len, out_result);

    jenv->ReleasePrimitiveArrayCritical(data, data0, JNI_ABORT);

    return ret;
  }

  int LGBM_BoosterPredictForCSRSingle(JNIEnv *jenv,
                                      jintArray indices,
                                      jdoubleArray values,
                                      int numNonZeros,
                                      BoosterHandle handle,
                                      int indptr_type,
                                      int data_type,
                                      int64_t nelem,
                                      int64_t num_col,
                                      int predict_type,
                                      int start_iteration,
                                      int num_iteration,
                                      const char* parameter,
                                      int64_t* out_len,
                                      double* out_result) {
    // Alternatives
    // - GetIntArrayElements: performs copy
    // - GetDirectBufferAddress: fails on wrapped array
    // Some words of warning for GetPrimitiveArrayCritical
    // https://stackoverflow.com/questions/23258357/whats-the-trade-off-between-using-getprimitivearraycritical-and-getprimitivety

    jboolean isCopy;
    int* indices0 = (int*)jenv->GetPrimitiveArrayCritical(indices, &isCopy);
    double* values0 = (double*)jenv->GetPrimitiveArrayCritical(values, &isCopy);

    int32_t ind[2] = { 0, numNonZeros };

    int ret = LGBM_BoosterPredictForCSRSingleRow(handle, ind, indptr_type, indices0, values0, data_type, 2,
                                                 nelem, num_col, predict_type, start_iteration, num_iteration, parameter, out_len, out_result);

    jenv->ReleasePrimitiveArrayCritical(values, values0, JNI_ABORT);
    jenv->ReleasePrimitiveArrayCritical(indices, indices0, JNI_ABORT);

    return ret;
  }

  /*! \brief Even faster variant of `LGBM_BoosterPredictForCSRSingle`.
   *
   * Uses `LGBM_BoosterPredictForCSRSingleRowFast` which is faster
   * than `LGBM_BoosterPredictForMatSingleRow` and the trick of
   * `LGBM_BoosterPredictForCSRSingle` to capture the Java data array
   * using `GetPrimitiveArrayCritical`, which can yield faster access
   * to the array if the JVM passes the actual address to the C++ side
   * instead of performing a copy.
   */
  int LGBM_BoosterPredictForCSRSingleRowFastCriticalSWIG(JNIEnv *jenv,
                                                         jintArray indices,
                                                         jdoubleArray values,
                                                         int numNonZeros,
                                                         FastConfigHandle handle,
                                                         int indptr_type,
                                                         int64_t nelem,
                                                         int64_t* out_len,
                                                         double* out_result) {
    // Alternatives
    // - GetIntArrayElements: performs copy
    // - GetDirectBufferAddress: fails on wrapped array
    // Some words of warning for GetPrimitiveArrayCritical
    // https://stackoverflow.com/questions/23258357/whats-the-trade-off-between-using-getprimitivearraycritical-and-getprimitivety

    jboolean isCopy;
    int* indices0 = (int*)jenv->GetPrimitiveArrayCritical(indices, &isCopy);
    double* values0 = (double*)jenv->GetPrimitiveArrayCritical(values, &isCopy);

    int32_t ind[2] = { 0, numNonZeros };

    int ret = LGBM_BoosterPredictForCSRSingleRowFast(handle, ind, indptr_type, indices0, values0, 2,
                                                     nelem, out_len, out_result);

    jenv->ReleasePrimitiveArrayCritical(values, values0, JNI_ABORT);
    jenv->ReleasePrimitiveArrayCritical(indices, indices0, JNI_ABORT);

    return ret;
  }

  #include <functional>
  #include <vector>

  struct CSRDirect {
          jintArray indices;
          jdoubleArray values;
          int* indices0;
          double* values0;
          int size;
  };

  int LGBM_DatasetCreateFromCSRSpark(JNIEnv *jenv,
                                     jobjectArray arrayOfSparseVector,
                                     int num_rows,
                                     int64_t num_col,
                                     const char* parameters,
                                     const DatasetHandle reference,
                                     DatasetHandle* out) {
    jclass sparseVectorClass = jenv->FindClass("org/apache/spark/ml/linalg/SparseVector");
    jmethodID sparseVectorIndices = jenv->GetMethodID(sparseVectorClass, "indices", "()[I");
    jmethodID sparseVectorValues = jenv->GetMethodID(sparseVectorClass, "values", "()[D");

    std::vector<CSRDirect> jniCache;
    jniCache.reserve(num_rows);

    // this needs to be done ahead of time as row_func is invoked from multiple threads
    // these threads would have to be registered with the JVM and also unregistered.
    // It is not clear if that can be achieved with OpenMP
    for (int i = 0; i < num_rows; i++) {
      // get the row
      jobject objSparseVec = jenv->GetObjectArrayElement(arrayOfSparseVector, i);

      // get the size, indices and values
      auto indices = (jintArray)jenv->CallObjectMethod(objSparseVec, sparseVectorIndices);
      if (jenv->ExceptionCheck()) {
        return -1;
      }
      auto values = (jdoubleArray)jenv->CallObjectMethod(objSparseVec, sparseVectorValues);
      if (jenv->ExceptionCheck()) {
        return -1;
      }
      int size = jenv->GetArrayLength(indices);

      // Note: when testing on larger data (e.g. 288k rows per partition and 36mio rows total)
      // using GetPrimitiveArrayCritical resulted in a dead-lock
      // lock arrays
      // int* indices0 = (int*)jenv->GetPrimitiveArrayCritical(indices, 0);
      // double* values0 = (double*)jenv->GetPrimitiveArrayCritical(values, 0);
      // in test-usecase an alternative to GetPrimitiveArrayCritical as it performs copies
      int* indices0 = (int *)jenv->GetIntArrayElements(indices, 0);
      double* values0 = jenv->GetDoubleArrayElements(values, 0);

      jniCache.push_back({indices, values, indices0, values0, size});
    }

    // type is important here as we want a std::function, rather than a lambda
    std::function<void(int idx, std::vector<std::pair<int, double>>& ret)> row_func = [&](int row_num, std::vector<std::pair<int, double>>& ret) {
      auto& jc = jniCache[row_num];
      ret.clear();  // reset size, but not free()
      ret.reserve(jc.size);  // make sure we have enough allocated

      // copy data
      int* indices0p = jc.indices0;
      double* values0p = jc.values0;
      int* indices0e = indices0p + jc.size;

      for (; indices0p != indices0e; ++indices0p, ++values0p)
        ret.emplace_back(*indices0p, *values0p);
    };

    int ret = LGBM_DatasetCreateFromCSRFunc(&row_func, num_rows, num_col, parameters, reference, out);

    for (auto& jc : jniCache) {
      // jenv->ReleasePrimitiveArrayCritical(jc.values, jc.values0, JNI_ABORT);
      // jenv->ReleasePrimitiveArrayCritical(jc.indices, jc.indices0, JNI_ABORT);
      jenv->ReleaseDoubleArrayElements(jc.values, jc.values0, JNI_ABORT);
      jenv->ReleaseIntArrayElements(jc.indices, (jint *)jc.indices0, JNI_ABORT);
    }

    return ret;
  }
%}


%include "pointer_manipulation.i"
%include "StringArray_API_extensions.i"
%include "ChunkedArray_API_extensions.i"
