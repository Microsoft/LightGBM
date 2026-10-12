/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
/**
 * Wrap chunked_array.hpp class for SWIG usage.
 *
 * Author: Alberto Ferreira
 */

%{
#include "../include/LightGBM/utils/chunked_array.hpp"
%}

%include "../include/LightGBM/utils/chunked_array.hpp"

using LightGBM::ChunkedArray;

%template(int32ChunkedArray) ChunkedArray<int32_t>;
/* Unfortunately, for the time being,
 * SWIG has issues generating the overloads to coalesce_to()
 * for larger integral types
 * so we won't support that for now:
 */
//%template(int64ChunkedArray) ChunkedArray<int64_t>;
%template(floatChunkedArray) ChunkedArray<float>;
%template(doubleChunkedArray) ChunkedArray<double>;
