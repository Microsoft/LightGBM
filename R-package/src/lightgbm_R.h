/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_R_PACKAGE_SRC_LIGHTGBM_R_H_
#define LIGHTGBM_R_PACKAGE_SRC_LIGHTGBM_R_H_

#include <LightGBM/c_api.h>

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif

#ifndef R_USE_C99_IN_CXX
#define R_USE_C99_IN_CXX
#endif

#include <Rinternals.h>

/*!
* \brief check if an R external pointer (like a Booster or Dataset handle) is a null pointer
* \param handle handle for a Booster, Dataset, or Predictor
* \return R logical, TRUE if the handle is a null pointer
*/
LIGHTGBM_C_EXPORT SEXP LGBM_HandleIsNull_R(
  SEXP handle
);

/*!
* \brief Throw a standardized error message when encountering a null Booster handle
* \return No return, will throw an error
*/
LIGHTGBM_C_EXPORT SEXP LGBM_NullBoosterHandleError_R();

// --- start Dataset interface

/*!
* \brief load Dataset from file like the command_line LightGBM does
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other Dataset, nullptr means not used
* \return Dataset handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromFile_R(
  SEXP filename,
  SEXP parameters,
  SEXP reference
);

/*!
* \brief create a Dataset from Compressed Sparse Column (CSC) format
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param num_indptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param parameters additional parameters
* \param reference used to align bin mapper with other Dataset, nullptr means not used
* \return Dataset handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromCSC_R(
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  SEXP parameters,
  SEXP reference
);

/*!
* \brief create Dataset from dense matrix
* \param data matrix data
* \param num_row number of rows
* \param num_col number columns
* \param parameters additional parameters
* \param reference used to align bin mapper with other Dataset, nullptr means not used
* \return Dataset handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromMat_R(
  SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP parameters,
  SEXP reference
);

/*!
* \brief Create subset of a Dataset
* \param handle handle of full Dataset
* \param used_row_indices Indices used in subset
* \param len_used_row_indices length of Indices used in subset
* \param parameters additional parameters
* \return Dataset handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetSubset_R(
  SEXP handle,
  SEXP used_row_indices,
  SEXP len_used_row_indices,
  SEXP parameters
);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return R character vector of feature names
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSetFeatureNames_R(
  SEXP handle,
  SEXP feature_names
);

/*!
* \brief get feature names from Dataset
* \param handle Dataset handle
* \return an R character vector with feature names from the Dataset or NULL if no feature names
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetFeatureNames_R(
  SEXP handle
);

/*!
* \brief save Dataset to binary file
* \param handle an instance of Dataset
* \param filename file name
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSaveBinary_R(
  SEXP handle,
  SEXP filename
);

/*!
* \brief free Dataset
* \param handle an instance of Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetFree_R(
  SEXP handle
);

/*!
* \brief set vector to a content in info
*        Note: group and group_id only work for C_API_DTYPE_INT32
*              label and weight only work for C_API_DTYPE_FLOAT32
* \param handle an instance of Dataset
* \param field_name field name, can be label, weight, group, group_id
* \param field_data pointer to vector
* \param num_element number of element in field_data
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSetField_R(
  SEXP handle,
  SEXP field_name,
  SEXP field_data,
  SEXP num_element
);

/*!
* \brief get size of info vector from Dataset
* \param handle an instance of Dataset
* \param field_name field name
* \param out size of info vector from Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetFieldSize_R(
  SEXP handle,
  SEXP field_name,
  SEXP out
);

/*!
* \brief get info vector from Dataset
* \param handle an instance of Dataset
* \param field_name field name
* \param field_data pointer to vector
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetField_R(
  SEXP handle,
  SEXP field_name,
  SEXP field_data
);

/*!
 * \brief Raise errors for attempts to update Dataset parameters.
 *        Some parameters cannot be updated after construction.
 * \param old_params Current Dataset parameters
 * \param new_params New Dataset parameters
 * \return R NULL value
 */
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetUpdateParamChecking_R(
  SEXP old_params,
  SEXP new_params
);

/*!
* \brief get number of data.
* \param handle the handle to the Dataset
* \param out The address to hold number of data
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetNumData_R(
  SEXP handle,
  SEXP out
);

/*!
* \brief get number of features
* \param handle the handle to the Dataset
* \param out The output of number of features
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetNumFeature_R(
  SEXP handle,
  SEXP out
);

/*!
* \brief get number of bins for feature
* \param handle the handle to the Dataset
* \param feature the index of the feature
* \param out The output of number of bins
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetFeatureNumBin_R(
  SEXP handle,
  SEXP feature,
  SEXP out
);

// --- start Booster interfaces

/*!
* \brief create a new boosting learner
* \param train_data training Dataset
* \param parameters format: 'key1=value1 key2=value2'
* \return Booster handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCreate_R(
  SEXP train_data,
  SEXP parameters
);

/*!
* \brief free Booster
* \param handle handle to be freed
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterFree_R(
  SEXP handle
);

/*!
* \brief load an existing Booster from model file
* \param filename filename of model
* \return Booster handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCreateFromModelfile_R(
  SEXP filename
);

/*!
* \brief load an existing Booster from a string
* \param model_str string containing the model
* \return Booster handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterLoadModelFromString_R(
  SEXP model_str
);

/*!
* \brief Get parameters as JSON string.
* \param handle Booster handle
* \return R character vector (length=1) with parameters in JSON format
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetLoadedParam_R(
  SEXP handle
);

/*!
* \brief Merge model in two Boosters to first handle
* \param handle handle primary Booster handle, will merge other handle to this
* \param other_handle secondary Booster handle
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterMerge_R(
  SEXP handle,
  SEXP other_handle
);

/*!
* \brief Add new validation to Booster
* \param handle Booster handle
* \param valid_data validation Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterAddValidData_R(
  SEXP handle,
  SEXP valid_data
);

/*!
* \brief Reset training data for Booster
* \param handle Booster handle
* \param train_data training Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterResetTrainingData_R(
  SEXP handle,
  SEXP train_data
);

/*!
* \brief Reset config for current Booster
* \param handle Booster handle
* \param parameters format: 'key1=value1 key2=value2'
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterResetParameter_R(
  SEXP handle,
  SEXP parameters
);

/*!
* \brief Get number of classes
* \param handle Booster handle
* \param out number of classes
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetNumClasses_R(
  SEXP handle,
  SEXP out
);

/*!
* \brief Get number of features.
* \param handle Booster handle
* \return Total number of features, as R integer
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetNumFeature_R(
  SEXP handle
);

/*!
* \brief update the model in one round
* \param handle Booster handle
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterUpdateOneIter_R(
  SEXP handle
);

/*!
* \brief update the model, by directly specifying gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle Booster handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \param len length of grad/hess
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterUpdateOneIterCustom_R(
  SEXP handle,
  SEXP grad,
  SEXP hess,
  SEXP len
);

/*!
* \brief Rollback one iteration
* \param handle Booster handle
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterRollbackOneIter_R(
  SEXP handle
);

/*!
* \brief Get iteration of current boosting rounds
* \param handle Booster handle
* \param out iteration of boosting rounds
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetCurrentIteration_R(
  SEXP handle,
  SEXP out
);

/*!
 * \brief Get number of trees per iteration
 * \param handle Booster handle
 * \param out Number of trees per iteration
 * \return R NULL value
 */
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterNumModelPerIteration_R(
    SEXP handle,
    SEXP out
);

/*!
 * \brief Get total number of trees
 * \param handle Booster handle
 * \param out Total number of trees of Booster
 * \return R NULL value
 */
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterNumberOfTotalModel_R(
    SEXP handle,
    SEXP out
);

/*!
* \brief Get model upper bound value.
* \param handle Handle of Booster
* \param[out] out_results Result pointing to max value
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetUpperBoundValue_R(
    SEXP handle,
    SEXP out_result
);

/*!
* \brief Get model lower bound value.
* \param handle Handle of Booster
* \param[out] out_results Result pointing to min value
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetLowerBoundValue_R(
    SEXP handle,
    SEXP out_result
);

/*!
* \brief Get names of eval metrics
* \param handle Handle of booster
* \return R character vector with names of eval metrics
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetEvalNames_R(
  SEXP handle
);

/*!
* \brief get evaluation for training data and validation data
* \param handle Booster handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result float array containing result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetEval_R(
  SEXP handle,
  SEXP data_idx,
  SEXP out_result
);

/*!
* \brief Get number of prediction for training data and validation data
* \param handle Booster handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out size of predict
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetNumPredict_R(
  SEXP handle,
  SEXP data_idx,
  SEXP out
);

/*!
* \brief Get prediction for training data and validation data.
*        This can be used to support customized eval function
* \param handle Booster handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result, used to store predict result, should pre-allocate memory
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetPredict_R(
  SEXP handle,
  SEXP data_idx,
  SEXP out_result
);

/*!
* \brief make prediction for file
* \param handle Booster handle
* \param data_filename filename of data file
* \param data_has_header data file has header or not
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration Start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param result_filename filename of file to write predictions to
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForFile_R(
  SEXP handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP result_filename
);

/*!
* \brief Get number of prediction
* \param handle Booster handle
* \param num_row number of rows in input
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration Start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out_len length of prediction
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCalcNumPredict_R(
  SEXP handle,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP out_len
);

/*!
* \brief make prediction for a new Dataset
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
*               for feature contributions, its length is equal to num_class * num_data * (num_features + 1)
* \param handle Booster handle
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param num_indptr number of cols in the matrix + 1
* \param nelem number of non-zero elements in the matrix
* \param num_row number of rows
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForCSC_R(
  SEXP handle,
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
  SEXP out_result
);

/*!
* \brief make prediction for a new Dataset
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
*               for feature contributions, its length is equal to num_class * num_data * (num_features + 1)
* \param handle Booster handle
* \param indptr array with the index pointer of the data in CSR format
* \param indices array with the non-zero indices of the data in CSR format
* \param data array with the non-zero values of the data in CSR format
* \param ncols number of columns in the data
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForCSR_R(
  SEXP handle,
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
  SEXP out_result
);

/*!
* \brief make prediction for a single row of data
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class
*               for leaf index, its length is equal to num_class * num_iteration
*               for feature contributions, its length is equal to num_class * (num_features + 1)
* \param handle Booster handle
* \param indices array corresponding to the indices of the columns with non-zero values of the row to predict on
* \param data array corresponding to the non-zero values of row to predict on
* \param ncols number of columns in the data
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForCSRSingleRow_R(
  SEXP handle,
  SEXP indices,
  SEXP data,
  SEXP ncols,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result
);

/*!
* \brief Initialize and return a fast configuration handle to use with ``LGBM_BoosterPredictForCSRSingleRowFast_R``.
* \param handle Booster handle
* \param ncols number columns in the data
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \return Fast configuration handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForCSRSingleRowFastInit_R(
  SEXP handle,
  SEXP ncols,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter
);

/*!
* \brief make prediction for a single row of data
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class
*               for leaf index, its length is equal to num_class * num_iteration
*               for feature contributions, its length is equal to num_class * (num_features + 1)
* \param handle_fastConfig Fast configuration handle
* \param indices array corresponding to the indices of the columns with non-zero values of the row to predict on
* \param data array corresponding to the non-zero values of row to predict on
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForCSRSingleRowFast_R(
  SEXP handle_fastConfig,
  SEXP indices,
  SEXP data,
  SEXP out_result
);

/*!
* \brief make feature contribution prediction for a new Dataset
* \param handle Booster handle
* \param indptr array with the index pointer of the data in CSR or CSC format
* \param indices array with the non-zero indices of the data in CSR or CSC format
* \param data array with the non-zero values of the data in CSR or CSC format
* \param is_csr whether the input data is in CSR format or not (pass FALSE for CSC)
* \param nrows number of rows in the data
* \param ncols number of columns in the data
* \param start_iteration start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \return An R list with entries "indptr", "indices", "data", constituting the
*         feature contributions in sparse format, in the same storage order as
*         the input data.
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictSparseOutput_R(
  SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP is_csr,
  SEXP nrows,
  SEXP ncols,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter
);

/*!
* \brief make prediction for a new Dataset
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
*               for feature contributions, its length is equal to num_class * num_data * (num_features + 1)
* \param handle Booster handle
* \param data pointer to the data space
* \param num_row number of rows
* \param num_col number columns
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForMat_R(
  SEXP handle,
  SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result
);

/*!
* \brief make prediction for a single row of data
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class
*               for leaf index, its length is equal to num_class * num_iteration
*               for feature contributions, its length is equal to num_class * (num_features + 1)
* \param handle Booster handle
* \param data array corresponding to the row to predict on
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForMatSingleRow_R(
  SEXP handle,
  SEXP data,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result
);

/*!
* \brief Initialize and return a fast configuration handle to use with ``LGBM_BoosterPredictForMatSingleRowFast_R``.
* \param handle Booster handle
* \param ncols number columns in the data
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \return Fast configuration handle
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForMatSingleRowFastInit_R(
  SEXP handle,
  SEXP ncols,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter
);

/*!
* \brief make prediction for a single row of data
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class
*               for leaf index, its length is equal to num_class * num_iteration
*               for feature contributions, its length is equal to num_class * (num_features + 1)
* \param handle_fastConfig Fast configuration handle
* \param data array corresponding to the row to predict on
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForMatSingleRowFast_R(
  SEXP handle_fastConfig,
  SEXP data,
  SEXP out_result
);

/*!
* \brief save model into file
* \param handle Booster handle
* \param num_iteration, <= 0 means save all
* \param feature_importance_type type of feature importance, 0: split, 1: gain
* \param filename file name
* \param start_iteration Starting iteration (0 based)
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterSaveModel_R(
  SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP filename,
  SEXP start_iteration
);

/*!
* \brief create string containing model
* \param handle Booster handle
* \param num_iteration, <= 0 means save all
* \param feature_importance_type type of feature importance, 0: split, 1: gain
* \param start_iteration Starting iteration (0 based)
* \return R character vector (length=1) with model string
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterSaveModelToString_R(
  SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP start_iteration
);

/*!
* \brief dump model to JSON
* \param handle Booster handle
* \param num_iteration, <= 0 means save all
* \param feature_importance_type type of feature importance, 0: split, 1: gain
* \param start_iteration Index of starting iteration (0 based)
* \return R character vector (length=1) with model JSON
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterDumpModel_R(
  SEXP handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP start_iteration
);

/*!
* \brief Dump parameter aliases to JSON
* \return R character vector (length=1) with aliases JSON
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DumpParamAliases_R();

/*!
* \brief Get current maximum number of threads used by LightGBM routines in this process.
* \param[out] out current maximum number of threads used by LightGBM. -1 means defaulting to omp_get_num_threads().
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_GetMaxThreads_R(
  SEXP out
);


/*!
* \brief Set maximum number of threads used by LightGBM routines in this process.
* \param num_threads maximum number of threads used by LightGBM. -1 means defaulting to omp_get_num_threads().
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_SetMaxThreads_R(
  SEXP num_threads
);

#endif  // LIGHTGBM_R_PACKAGE_SRC_LIGHTGBM_R_H_
