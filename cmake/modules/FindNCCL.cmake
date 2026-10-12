#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tries to find NCCL headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(NCCL)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  NCCL_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the NCCL installation.
#              The environment variable NCCL_ROOT overrides this variable.
#
# This module defines
#  NCCL_FOUND, whether nccl has been found
#  NCCL_INCLUDE_DIR, directory containing header
#  NCCL_LIBRARY, directory containing nccl library
#  NCCL_LIB_NAME, nccl library name
#  USE_NCCL_LIB_PATH, when set, NCCL_LIBRARY path is also inspected for the
#                     location of the nccl library. This would disable
#                     switching between static and shared.
#
# It creates an imported target 'NCCL::NCCL' for convenience in linking.
#
# This module assumes that the CUDA toolkit has already been found.
#

if(NCCL_LIBRARY)
  if(NOT USE_NCCL_LIB_PATH)
    # Don't cache NCCL_LIBRARY to enable switching between static and shared.
    unset(NCCL_LIBRARY CACHE)
  endif()
endif()

if(BUILD_WITH_SHARED_NCCL)
  # libnccl.so
  set(NCCL_LIB_NAME nccl)
  set(_nccl_lib_type "SHARED")
else()
  # libnccl_static.a
  set(NCCL_LIB_NAME nccl_static)
  set(_nccl_lib_type "STATIC")
endif()

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS $ENV{NCCL_ROOT}/include ${NCCL_ROOT}/include)

find_library(NCCL_LIBRARY
  NAMES ${NCCL_LIB_NAME}
  HINTS $ENV{NCCL_ROOT}/lib ${NCCL_ROOT}/lib)

message(STATUS "Using nccl library: ${NCCL_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG
                                  NCCL_INCLUDE_DIR NCCL_LIBRARY)

mark_as_advanced(
  NCCL_INCLUDE_DIR
  NCCL_LIBRARY
)

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
  add_library(NCCL::NCCL ${_nccl_lib_type} IMPORTED)
  set_target_properties(NCCL::NCCL PROPERTIES
    IMPORTED_LOCATION "${NCCL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
endif()

unset(_nccl_lib_type)
