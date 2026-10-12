/*!
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Alberto Ferreira
 */
/**
 * This wraps the StringArray.hpp class for SWIG usage,
 * adding the basic C-style wrappers needed to make it
 * usable for the users of the low-level lightgbmJNI API.
 */

%{
#include "../swig/StringArray.hpp"
%}

// Use SWIG's `various.i` to get a String[] directly in one call:
%apply char **STRING_ARRAY {char **StringArrayHandle_get_strings};

%inline %{

    typedef void* StringArrayHandle;

    /**
     * @brief Creates a new StringArray and returns its handle.
     *
     * @param num_strings number of strings to store.
     * @param string_size the maximum number of characters that can be stored in each string.
     * @return StringArrayHandle or nullptr in case of allocation failure.
     */
    StringArrayHandle StringArrayHandle_create(size_t num_strings, size_t string_size) {
        try {
            return new StringArray(num_strings, string_size);
        } catch (std::bad_alloc &/*e*/) {
            return nullptr;
        }
    }

    /**
     * @brief Free the StringArray object.
     *
     * @param handle StringArray handle.
     */
    void StringArrayHandle_free(StringArrayHandle handle)
    {
        delete reinterpret_cast<StringArray *>(handle);
    }

    /**
     * @brief Return the raw pointer to the array of strings.
     * Wrapped in Java into String[] automatically.
     *
     * @param handle StringArray handle.
     * @return Raw pointer to the string array which `various.i` maps to String[].
     */
    char **StringArrayHandle_get_strings(StringArrayHandle handle)
    {
        return reinterpret_cast<StringArray *>(handle)->data();
    }

    /**
     * For the end user to extract a specific string from the StringArray object.
     *
     * @param handle StringArray handle.
     * @param index index of the string to retrieve from the array.
     * @return raw pointer to string at index, or nullptr if out of bounds.
     */
    char *StringArrayHandle_get_string(StringArrayHandle handle, int index)
    {
        return reinterpret_cast<StringArray *>(handle)->getitem(index);
    }

    /**
     * @brief Replaces one string of the array at index with the new content.
     *
     * @param handle StringArray handle.
     * @param index Index of the string to replace
     * @param new_content The content to replace
     * @return 0 (success) or -1 (error) in case of out of bounds index or too large content.
     */
    int StringArrayHandle_set_string(StringArrayHandle handle, size_t index, const char* new_content)
    {
        return reinterpret_cast<StringArray *>(handle)->setitem(index, std::string(new_content));
    }

    /**
     * @brief Retrieve the number of strings in the StringArray.
     *
     * @param handle StringArray handle.
     * @return number of strings that the array stores.
     */
    size_t StringArrayHandle_get_num_elements(StringArrayHandle handle)
    {
        return reinterpret_cast<StringArray *>(handle)->get_num_elements();
    }

%}
