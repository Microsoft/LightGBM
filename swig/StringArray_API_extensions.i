/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
/**
 * This SWIG interface extension provides support to
 * allocate, return and manage arrays of strings through
 * the class StringArray.
 *
 * This is then used to generate wrappers that return newly-allocated
 * arrays of strings, so the user can them access them easily as a String[]
 * on the Java side by a single call to StringArray::data(), and even manipulate
 * them.
 *
 * It also implements working wrappers to:
 *  - LGBM_BoosterGetEvalNames    (re-implemented with new API)
 *  - LGBM_BoosterGetFeatureNames (original non-wrapped version didn't work).
 * where the wrappers names end with "SWIG".
 */


#include <memory>

%include "./StringArray.i"

%inline %{

    #define API_OK_OR_VALUE(api_return, return_value) if (api_return == -1) return return_value
    #define API_OK_OR_NULL(api_return) API_OK_OR_VALUE(api_return, nullptr)

    /**
     * @brief Wraps LGBM_BoosterGetEvalNames.
     *
     * In case of success a new StringArray is created and returned,
     * which you're responsible for freeing,
     * @see StringArrayHandle_free().
     * In case of failure such resource is freed and nullptr is returned.
     * Check for that case with null (lightgbmlib) or 0 (lightgbmlibJNI).
     *
     * @param handle Booster handle
     * @return StringArrayHandle with the eval names (or nullptr in case of error)
     */
    StringArrayHandle LGBM_BoosterGetEvalNamesSWIG(BoosterHandle handle)
    {
        int eval_counts;
        size_t string_size;
        std::unique_ptr<StringArray> strings(nullptr);

        // Retrieve required allocation space:
        API_OK_OR_NULL(LGBM_BoosterGetEvalNames(handle,
                                                0, &eval_counts,
                                                0, &string_size,
                                                nullptr));

        try {
            strings.reset(new StringArray(eval_counts, string_size));
        } catch (std::bad_alloc &/*e*/) {
            LGBM_SetLastError("Failure to allocate memory.");
            return nullptr;
        }

        API_OK_OR_NULL(LGBM_BoosterGetEvalNames(handle,
                                                eval_counts, &eval_counts,
                                                string_size, &string_size,
                                                strings->data()));

        return strings.release();
    }

    /**
     * @brief Wraps LGBM_BoosterGetFeatureNames.
     *
     * Allocates a new StringArray. You must free it yourself if it succeeds.
     * @see StringArrayHandle_free().
     * In case of failure such resource is freed and nullptr is returned.
     * Check for that case with null (lightgbmlib) or 0 (lightgbmlibJNI).
     *
     * @param handle Booster handle
     * @return StringArrayHandle with the feature names (or nullptr in case of error)
     */
    StringArrayHandle LGBM_BoosterGetFeatureNamesSWIG(BoosterHandle handle)
    {
        int num_features;
        size_t max_feature_name_size;
        std::unique_ptr<StringArray> strings(nullptr);

        // Retrieve required allocation space:
        API_OK_OR_NULL(LGBM_BoosterGetFeatureNames(handle,
                                                   0, &num_features,
                                                   0, &max_feature_name_size,
                                                   nullptr));

        try {
            strings.reset(new StringArray(num_features, max_feature_name_size));
        } catch (std::bad_alloc &/*e*/) {
            LGBM_SetLastError("Failure to allocate memory.");
            return nullptr;
        }

        API_OK_OR_NULL(LGBM_BoosterGetFeatureNames(handle,
                                                   num_features, &num_features,
                                                   max_feature_name_size, &max_feature_name_size,
                                                   strings->data()));

        return strings.release();
    }


    /**
     * @brief Wraps LGBM_DatasetGetFeatureNames. Has the same limitations as a
     * LGBM_BoosterGetFeatureNames:
     *
     * Allocates a new StringArray. You must free it yourself if it succeeds.
     * @see StringArrayHandle_free().
     * In case of failure such resource is freed and nullptr is returned.
     * Check for that case with null (lightgbmlib) or 0 (lightgbmlibJNI).
     *
     * @param handle Booster handle
     * @return StringArrayHandle with the feature names (or nullptr in case of error)
     */
    StringArrayHandle LGBM_DatasetGetFeatureNamesSWIG(BoosterHandle handle)
    {
        int num_features;
        size_t max_feature_name_size;
        std::unique_ptr<StringArray> strings(nullptr);

        // Retrieve required allocation space:
        API_OK_OR_NULL(LGBM_DatasetGetFeatureNames(handle,
                                                   0, &num_features,
                                                   0, &max_feature_name_size,
                                                   nullptr));
        try {
            strings.reset(new StringArray(num_features, max_feature_name_size));
        } catch (std::bad_alloc &/*e*/) {
            LGBM_SetLastError("Failure to allocate memory.");
            return nullptr;
        }

        API_OK_OR_NULL(LGBM_DatasetGetFeatureNames(handle,
                                                   num_features, &num_features,
                                                   max_feature_name_size, &max_feature_name_size,
                                                   strings->data()));

        return strings.release();
    }

%}
