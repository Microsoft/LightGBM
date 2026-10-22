# Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
# Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

cmake_minimum_required(VERSION 3.28)

option(USE_MPI "Enable MPI-based distributed learning" OFF)
option(USE_OPENMP "Enable OpenMP" ON)
option(USE_GPU "Enable GPU-accelerated training" OFF)
option(USE_SWIG "Enable SWIG to generate Java API" OFF)
option(USE_TIMETAG "Set to ON to output time costs" OFF)
option(USE_CUDA "Enable CUDA-accelerated training " OFF)
option(USE_ROCM "Enable ROCm-accelerated training " OFF)
option(USE_DEBUG "Set to ON for Debug mode" OFF)
option(USE_SANITIZER "Use sanitizer flags" OFF)
set(
  ENABLED_SANITIZERS
  "address" "leak" "undefined"
  CACHE
  STRING
  "Semicolon separated list of sanitizer names, e.g., 'address;leak'. \
Supported sanitizers are address, leak, undefined and thread."
)
option(USE_HOMEBREW_FALLBACK "(macOS-only) also look in 'brew --prefix' for libraries (e.g. OpenMP)" ON)
option(BUILD_CLI "Build the 'lightgbm' command-line interface in addition to lib_lightgbm" ON)
option(BUILD_CPP_TEST "Build C++ tests with Google Test" OFF)
option(BUILD_STATIC_LIB "Build static library" OFF)
option(INSTALL_HEADERS "Install headers to CMAKE_INSTALL_PREFIX (e.g. '/usr/local/include')" ON)
option(__BUILD_FOR_PYTHON "Set to ON if building lib_lightgbm for use with the Python-package" OFF)
option(__BUILD_FOR_R "Set to ON if building lib_lightgbm for use with the R-package" OFF)
option(__INTEGRATE_OPENCL "Set to ON if building LightGBM with the OpenCL ICD Loader and its dependencies included" OFF)

# If using Visual Studio generators, always target v10.x of the Windows SDK.
# Doing this avoids lookups that could fall back to very old versions, e.g. by finding
# outdated registry entries.
# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION.html
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(CMAKE_SYSTEM_VERSION 10.0 CACHE INTERNAL "target Windows SDK version" FORCE)
endif()

project(lightgbm LANGUAGES C CXX)

include(cmake/Utils.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

#-- Sanitizer
if(USE_SANITIZER)
  if(MSVC)
    message(FATAL_ERROR "Sanitizers are not supported with MSVC.")
  endif()
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif()

if(__INTEGRATE_OPENCL)
  set(__INTEGRATE_OPENCL ON CACHE BOOL "" FORCE)
  set(USE_GPU OFF CACHE BOOL "" FORCE)
  message(STATUS "Building library with integrated OpenCL components")
endif()

if(__BUILD_FOR_PYTHON OR __BUILD_FOR_R OR USE_SWIG)
    # the SWIG wrapper, the Python and R packages don't require the CLI
    set(BUILD_CLI OFF)
    # installing the SWIG wrapper, the R and Python packages shouldn't place LightGBM's headers
    # outside of where the package is installed
    set(INSTALL_HEADERS OFF)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.2")
    message(FATAL_ERROR "Insufficient gcc version (${CMAKE_CXX_COMPILER_VERSION})")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.8")
    message(FATAL_ERROR "Insufficient Clang version (${CMAKE_CXX_COMPILER_VERSION})")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1.0")
    message(FATAL_ERROR "Insufficient AppleClang version (${CMAKE_CXX_COMPILER_VERSION})")
  endif()
elseif(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "Insufficient MSVC version (${MSVC_VERSION})")
  endif()
endif()

if(USE_SWIG)
  find_package(SWIG REQUIRED)
  find_package(Java REQUIRED)
  find_package(JNI REQUIRED)
  include(UseJava)
  include(UseSWIG)
  set(SWIG_CXX_EXTENSION "cxx")
  set(SWIG_EXTRA_LIBRARIES "")
  set(SWIG_JAVA_EXTRA_FILE_EXTENSIONS ".java" "JNI.java")
  set(SWIG_MODULE_JAVA_LANGUAGE "JAVA")
  set(SWIG_MODULE_JAVA_SWIG_LANGUAGE_FLAG "java")
  set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/java")
  include_directories(Java_INCLUDE_DIRS)
  include_directories(JNI_INCLUDE_DIRS)
  include_directories($ENV{JAVA_HOME}/include)
  if(WIN32)
      set(LGBM_SWIG_DESTINATION_DIR "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/windows/x86_64")
      include_directories($ENV{JAVA_HOME}/include/win32)
  elseif(APPLE)
      set(LGBM_SWIG_DESTINATION_DIR "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/osx/x86_64")
      include_directories($ENV{JAVA_HOME}/include/darwin)
  else()
      set(LGBM_SWIG_DESTINATION_DIR "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/linux/x86_64")
      include_directories($ENV{JAVA_HOME}/include/linux)
  endif()
  file(MAKE_DIRECTORY "${LGBM_SWIG_DESTINATION_DIR}")
endif()

set(EIGEN_DIR "${PROJECT_SOURCE_DIR}/external_libs/eigen")
include_directories(${EIGEN_DIR})

# See https://gitlab.com/libeigen/eigen/-/blob/master/COPYING.README
add_definitions(-DEIGEN_MPL2_ONLY)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

set(FAST_DOUBLE_PARSER_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external_libs/fast_double_parser/include")
include_directories(${FAST_DOUBLE_PARSER_INCLUDE_DIR})

set(FMT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external_libs/fmt/include")
include_directories(${FMT_INCLUDE_DIR})

if(BUILD_CPP_TEST AND MSVC)
  # Use /MT flag to statically link the C runtime. Must be set before
  # add_subdirectory(nanoarrow) so the bundled library uses the same runtime
  # as the test executable; otherwise linking fails with unresolved __imp_*
  # symbols (e.g. __imp_realloc).
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# nanoarrow is only needed by the Arrow-based C API entry points, which are
# disabled for the R package (the R API never calls them). Skipping the
# subdirectory keeps the package tarball free of nanoarrow sources, which
# would otherwise trigger CRAN warnings about diagnostic-suppressing pragmas.
if(NOT __BUILD_FOR_R)
  add_subdirectory("${PROJECT_SOURCE_DIR}/external_libs/nanoarrow" EXCLUDE_FROM_ALL)
  if(NOT BUILD_STATIC_LIB)
    set_target_properties(nanoarrow_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endif()

if(__BUILD_FOR_R)
    find_package(LibR REQUIRED)
    message(STATUS "LIBR_EXECUTABLE: ${LIBR_EXECUTABLE}")
    message(STATUS "LIBR_INCLUDE_DIRS: ${LIBR_INCLUDE_DIRS}")
    message(STATUS "LIBR_LIBS_DIR: ${LIBR_LIBS_DIR}")
    message(STATUS "LIBR_CORE_LIBRARY: ${LIBR_CORE_LIBRARY}")
    include_directories(${LIBR_INCLUDE_DIRS})
    add_definitions(-DLGB_R_BUILD)
endif()

if(USE_TIMETAG)
    add_definitions(-DTIMETAG)
endif()

if(USE_DEBUG)
    add_definitions(-DDEBUG)
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DUSE_MPI)
else()
    add_definitions(-DUSE_SOCKET)
endif()

if(USE_CUDA)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    find_package(CUDAToolkit 11.8 REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      compute_cmake_cuda_archs("${CUDAToolkit_VERSION}")
    endif()
    enable_language(CUDA)
    set(USE_OPENMP ON CACHE BOOL "CUDA requires OpenMP" FORCE)
endif()

if(USE_ROCM)
    enable_language(HIP)
    set(USE_OPENMP ON CACHE BOOL "ROCm requires OpenMP" FORCE)
endif()

if(USE_OPENMP)
    if(APPLE)
        find_package(OpenMP)
        if(NOT OpenMP_FOUND)
            if(USE_HOMEBREW_FALLBACK)
                # libomp 15.0+ from brew is keg-only, so have to search in other locations.
                # See https://github.com/Homebrew/homebrew-core/issues/112107#issuecomment-1278042927.
                execute_process(COMMAND brew --prefix libomp
                            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
                set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
                set(OpenMP_C_LIB_NAMES omp)
                set(OpenMP_CXX_LIB_NAMES omp)
                set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
            endif()
            find_package(OpenMP REQUIRED)
        endif()
    else()
        find_package(OpenMP REQUIRED)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(USE_GPU)
    set(BOOST_COMPUTE_HEADER_DIR ${PROJECT_SOURCE_DIR}/external_libs/compute/include)
    include_directories(${BOOST_COMPUTE_HEADER_DIR})
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    message(STATUS "OpenCL include directory: " ${OpenCL_INCLUDE_DIRS})
    if(WIN32)
        set(Boost_USE_STATIC_LIBS ON)
    endif()
    find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
    if(WIN32)
        # disable autolinking in boost
        add_definitions(-DBOOST_ALL_NO_LIB)
    endif()
    include_directories(${Boost_INCLUDE_DIRS})
    add_definitions(-DUSE_GPU)
endif()

if(__INTEGRATE_OPENCL)
    if(APPLE)
        message(FATAL_ERROR "Integrated OpenCL build is not available on macOS")
    else()
        include(cmake/IntegratedOpenCL.cmake)
        add_definitions(-DUSE_GPU)
    endif()
endif()

if(USE_CUDA)
    find_package(NCCL REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS} -Xcompiler=-fPIC -Xcompiler=-Wall")
    if(USE_DEBUG)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
    else()
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
    endif()
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

    add_definitions(-DUSE_CUDA)

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

if(USE_ROCM)
    find_package(HIP)
    include_directories(${HIP_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${OpenMP_CXX_FLAGS} -fPIC -Wall")

    # avoid warning: unused variable 'mask' due to __shfl_down_sync work-around
    set(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-unused-variable")
    # avoid warning: 'hipHostAlloc' is deprecated: use hipHostMalloc instead
    set(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-deprecated-declarations")
    # avoid many warnings about missing overrides
    set(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-inconsistent-missing-override")
    # avoid warning: shift count >= width of type in feature_histogram.hpp
    set(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-shift-count-overflow")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DISABLED_WARNINGS}")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${DISABLED_WARNINGS}")

    if(USE_DEBUG)
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0")
    else()
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
    endif()
    message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")

    # Building for ROCm almost always means USE_CUDA.
    # Exceptions to this will be guarded by USE_ROCM.
    add_definitions(-DUSE_CUDA)
    add_definitions(-DUSE_ROCM)
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <xmmintrin.h>
int main() {
  int a = 0;
  _mm_prefetch(&a, _MM_HINT_NTA);
  return 0;
}
" MM_PREFETCH)

if(${MM_PREFETCH})
  message(STATUS "Using _mm_prefetch")
  add_definitions(-DMM_PREFETCH)
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <mm_malloc.h>
int main() {
  char *a = (char*)_mm_malloc(8, 16);
  _mm_free(a);
  return 0;
}
" MM_MALLOC)

if(${MM_MALLOC})
  message(STATUS "Using _mm_malloc")
  add_definitions(-DMM_MALLOC)
endif()

if(UNIX OR MINGW OR CYGWIN)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pthread -Wextra -Wall -Wno-ignored-attributes -Wno-unknown-pragmas -Wno-return-type"
  )
  if(MINGW)
    # ignore this warning: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95353
    set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow"
    )
  endif()
  if(USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  endif()
  if(USE_SWIG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
  endif()
  if(NOT USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
  endif()
  if(__BUILD_FOR_R AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cast-function-type")
  endif()
endif()

if(WIN32 AND MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

# Check if inet_pton is already available, to avoid conflicts with the implementation in LightGBM.
# As of 2022, MinGW started including a definition of inet_pton.
if(WIN32)
  include(CheckSymbolExists)
  list(APPEND CMAKE_REQUIRED_LIBRARIES "ws2_32")
  check_symbol_exists(inet_pton "ws2tcpip.h" WIN_INET_PTON_FOUND)
  if(WIN_INET_PTON_FOUND)
    add_definitions(-DWIN_HAS_INET_PTON)
  endif()
  list(REMOVE_ITEM CMAKE_REQUIRED_LIBRARIES "ws2_32")
endif()

if(MSVC)
    # compiling 'fmt' on MSVC: "Unicode support requires compiling with /utf-8"
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP /utf-8")
    if(__BUILD_FOR_R)
        # MSVC does not like this commit:
        # https://github.com/wch/r-source/commit/fb52ac1a610571fcb8ac92d886b9fefcffaa7d48
        #
        # and raises "error C3646: 'private_data_c': unknown override specifier"
        add_definitions(-DR_LEGACY_RCOMPLEX)
    endif()
    if(USE_DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy")
    endif()
else()
    if(NOT BUILD_STATIC_LIB)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
    if(NOT USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
    endif()
endif()

set(LightGBM_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories(${LightGBM_HEADER_DIR})

if(USE_MPI)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

set(
    LGBM_SOURCES
      src/boosting/boosting.cpp
      src/boosting/gbdt_model_text.cpp
      src/boosting/gbdt_prediction.cpp
      src/boosting/gbdt.cpp
      src/boosting/prediction_early_stop.cpp
      src/boosting/sample_strategy.cpp
      src/io/bin.cpp
      src/io/config_auto.cpp
      src/io/config.cpp
      src/io/dataset_loader.cpp
      src/io/dataset.cpp
      src/io/file_io.cpp
      src/io/json11.cpp
      src/io/metadata.cpp
      src/io/parser.cpp
      src/io/train_share_states.cpp
      src/io/tree.cpp
      src/io/pairwise_lambdarank_bin.cpp
      src/io/pairwise_ranking_feature_group.cpp
      src/metric/dcg_calculator.cpp
      src/metric/metric.cpp
      src/network/linker_topo.cpp
      src/network/linkers_mpi.cpp
      src/network/linkers_socket.cpp
      src/network/network.cpp
      src/objective/objective_function.cpp
      src/objective/rank_objective.cpp
      src/treelearner/data_parallel_tree_learner.cpp
      src/treelearner/feature_histogram.cpp
      src/treelearner/feature_parallel_tree_learner.cpp
      src/treelearner/gpu_tree_learner.cpp
      src/treelearner/gradient_discretizer.cpp
      src/treelearner/linear_tree_learner.cpp
      src/treelearner/serial_tree_learner.cpp
      src/treelearner/tree_learner.cpp
      src/treelearner/voting_parallel_tree_learner.cpp
      src/utils/openmp_wrapper.cpp
)
set(
    LGBM_CUDA_SOURCES
      src/boosting/cuda/cuda_score_updater.cpp
      src/boosting/cuda/cuda_score_updater.cu
      src/boosting/cuda/nccl_gbdt.cpp
      src/metric/cuda/cuda_binary_metric.cpp
      src/metric/cuda/cuda_pointwise_metric.cpp
      src/metric/cuda/cuda_regression_metric.cpp
      src/metric/cuda/cuda_pointwise_metric.cu
      src/objective/cuda/cuda_binary_objective.cpp
      src/objective/cuda/cuda_multiclass_objective.cpp
      src/objective/cuda/cuda_rank_objective.cpp
      src/objective/cuda/cuda_regression_objective.cpp
      src/objective/cuda/cuda_binary_objective.cu
      src/objective/cuda/cuda_multiclass_objective.cu
      src/objective/cuda/cuda_rank_objective.cu
      src/objective/cuda/cuda_regression_objective.cu
      src/treelearner/cuda/cuda_best_split_finder.cpp
      src/treelearner/cuda/cuda_data_partition.cpp
      src/treelearner/cuda/cuda_histogram_constructor.cpp
      src/treelearner/cuda/cuda_leaf_splits.cpp
      src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp
      src/treelearner/cuda/cuda_best_split_finder.cu
      src/treelearner/cuda/cuda_data_partition.cu
      src/treelearner/cuda/cuda_gradient_discretizer.cu
      src/treelearner/cuda/cuda_histogram_constructor.cu
      src/treelearner/cuda/cuda_leaf_splits.cu
      src/treelearner/cuda/cuda_single_gpu_tree_learner.cu
      src/io/cuda/cuda_column_data.cu
      src/io/cuda/cuda_tree.cu
      src/io/cuda/cuda_column_data.cpp
      src/io/cuda/cuda_metadata.cpp
      src/io/cuda/cuda_row_data.cpp
      src/io/cuda/cuda_tree.cpp
      src/cuda/cuda_utils.cpp
      src/cuda/cuda_algorithms.cu
)

if(USE_CUDA OR USE_ROCM)
  list(APPEND LGBM_SOURCES ${LGBM_CUDA_SOURCES})
endif()

if(USE_ROCM)
  set(CU_FILES "")
  foreach(file IN LISTS LGBM_CUDA_SOURCES)
      string(REGEX MATCH "\\.cu$" is_cu_file "${file}")
      if(is_cu_file)
          list(APPEND CU_FILES "${file}")
      endif()
  endforeach()
  set_source_files_properties(${CU_FILES} PROPERTIES LANGUAGE HIP)
endif()

add_library(lightgbm_objs OBJECT ${LGBM_SOURCES})
if(NOT __BUILD_FOR_R)
  target_link_libraries(lightgbm_objs PRIVATE nanoarrow_static)
endif()

if(BUILD_CLI)
    add_executable(lightgbm src/main.cpp src/application/application.cpp)
    target_link_libraries(lightgbm PRIVATE lightgbm_objs)
endif()

set(API_SOURCES "src/c_api.cpp")
# Only build the R part of the library if building for
# use with the R-package
if(__BUILD_FOR_R)
  list(APPEND API_SOURCES "src/lightgbm_R.cpp")
endif()

add_library(lightgbm_capi_objs OBJECT ${API_SOURCES})
if(NOT __BUILD_FOR_R)
  target_link_libraries(lightgbm_capi_objs PRIVATE nanoarrow_static)
endif()

if(BUILD_STATIC_LIB)
  add_library(_lightgbm STATIC)
else()
  add_library(_lightgbm SHARED)
endif()

# R expects libraries of the form <project>.{dll,dylib,so}, not lib_<project>.{dll,dylib,so}
if(__BUILD_FOR_R)
  set_target_properties(
    _lightgbm
    PROPERTIES
      PREFIX ""
      OUTPUT_NAME "lightgbm"
  )
endif()

# LightGBM headers include openmp, cuda, R etc. headers,
# thus PUBLIC is required for building _lightgbm_swig target.
target_link_libraries(_lightgbm PUBLIC lightgbm_capi_objs lightgbm_objs)

if(MSVC AND NOT __BUILD_FOR_R)
  set_target_properties(_lightgbm PROPERTIES OUTPUT_NAME "lib_lightgbm")
endif()

if(USE_SWIG)
  set_property(SOURCE swig/lightgbmlib.i PROPERTY CPLUSPLUS ON)
  list(APPEND swig_options -package com.microsoft.ml.lightgbm)
  set_property(SOURCE swig/lightgbmlib.i PROPERTY SWIG_FLAGS "${swig_options}")
  swig_add_library(_lightgbm_swig LANGUAGE java SOURCES swig/lightgbmlib.i)
  swig_link_libraries(_lightgbm_swig _lightgbm)
  set_target_properties(
    _lightgbm_swig
    PROPERTIES
      # needed to ensure Linux build does not have lib prefix specified twice, e.g. liblib_lightgbm_swig
      PREFIX ""
      # needed in some versions of CMake for VS and MinGW builds to ensure output dll has lib prefix
      OUTPUT_NAME "lib_lightgbm_swig"
  )
  if(WIN32)
    set(LGBM_SWIG_LIB_DESTINATION_PATH "${LGBM_SWIG_DESTINATION_DIR}/lib_lightgbm_swig.dll")
    if(MINGW OR CYGWIN)
        set(LGBM_LIB_SOURCE_PATH "${PROJECT_SOURCE_DIR}/lib_lightgbm.dll")
        set(LGBM_SWIG_LIB_SOURCE_PATH "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.dll")
    else()
        set(LGBM_LIB_SOURCE_PATH "${PROJECT_SOURCE_DIR}/Release/lib_lightgbm.dll")
        set(LGBM_SWIG_LIB_SOURCE_PATH "${PROJECT_SOURCE_DIR}/Release/lib_lightgbm_swig.dll")
    endif()
  elseif(APPLE)
    set(LGBM_LIB_SOURCE_PATH "${PROJECT_SOURCE_DIR}/lib_lightgbm.dylib")
    set(LGBM_SWIG_LIB_SOURCE_PATH "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.jnilib")
    set(LGBM_SWIG_LIB_DESTINATION_PATH "${LGBM_SWIG_DESTINATION_DIR}/lib_lightgbm_swig.dylib")
  else()
    set(LGBM_LIB_SOURCE_PATH "${PROJECT_SOURCE_DIR}/lib_lightgbm.so")
    set(LGBM_SWIG_LIB_SOURCE_PATH "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.so")
    set(LGBM_SWIG_LIB_DESTINATION_PATH "${LGBM_SWIG_DESTINATION_DIR}/lib_lightgbm_swig.so")
  endif()
  add_custom_command(
      TARGET _lightgbm_swig
      POST_BUILD
      COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
      COMMAND
        "${CMAKE_COMMAND}"
        -E
        copy_if_different
        "${LGBM_LIB_SOURCE_PATH}"
        "${LGBM_SWIG_DESTINATION_DIR}"
      COMMAND
        "${CMAKE_COMMAND}"
        -E
        copy_if_different
        "${LGBM_SWIG_LIB_SOURCE_PATH}"
        "${LGBM_SWIG_LIB_DESTINATION_PATH}"
      COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com
    )
endif()

if(USE_MPI)
  target_link_libraries(lightgbm_objs PUBLIC ${MPI_CXX_LIBRARIES})
endif()

if(USE_OPENMP)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(lightgbm_objs PUBLIC OpenMP::OpenMP_CXX)
    # c_api headers also includes OpenMP headers, thus compiling
    # lightgbm_capi_objs needs include directory for OpenMP.
    # Specifying OpenMP in target_link_libraries will get include directory
    # requirements for compilation.
    # This uses CMake's Transitive Usage Requirements. Refer to CMake doc:
    # https://cmake.org/cmake/help/v3.16/manual/cmake-buildsystem.7.html#transitive-usage-requirements
    target_link_libraries(lightgbm_capi_objs PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(USE_GPU)
  target_link_libraries(lightgbm_objs PUBLIC ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
endif()

if(USE_CUDA)
  target_link_libraries(lightgbm_objs PUBLIC NCCL::NCCL)
endif()

if(USE_ROCM)
  find_package(rccl)
  target_link_libraries(lightgbm_objs PUBLIC ${RCCL_LIBRARY})
endif()

if(__INTEGRATE_OPENCL)
  # targets OpenCL and Boost are added in IntegratedOpenCL.cmake
  add_dependencies(lightgbm_objs OpenCL Boost)
  # variables INTEGRATED_OPENCL_* are set in IntegratedOpenCL.cmake
  target_include_directories(lightgbm_objs PRIVATE ${INTEGRATED_OPENCL_INCLUDES})
  target_compile_definitions(lightgbm_objs PRIVATE ${INTEGRATED_OPENCL_DEFINITIONS})
  target_link_libraries(lightgbm_objs PUBLIC ${INTEGRATED_OPENCL_LIBRARIES} ${CMAKE_DL_LIBS})
endif()

if(USE_CUDA)

  set_target_properties(
    lightgbm_objs
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
  )

  set_target_properties(
    _lightgbm
    PROPERTIES
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  if(BUILD_CLI)
    set_target_properties(
      lightgbm
      PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
  endif()
endif()

if(USE_ROCM)
  target_link_libraries(lightgbm_objs PUBLIC hip::host)
endif()

if(WIN32)
    if(MINGW OR CYGWIN)
      target_link_libraries(lightgbm_objs PUBLIC ws2_32 iphlpapi)
    endif()
endif()

if(__BUILD_FOR_R)
  # utils/log.h and capi uses R headers, thus both object libraries need to link
  # with R lib.
  if(MSVC)
    set(R_LIB ${LIBR_MSVC_CORE_LIBRARY})
  else()
    set(R_LIB ${LIBR_CORE_LIBRARY})
  endif()
  target_link_libraries(lightgbm_objs PUBLIC ${R_LIB})
  target_link_libraries(lightgbm_capi_objs PUBLIC ${R_LIB})
endif()

#-- Google C++ tests
if(BUILD_CPP_TEST)
  find_package(GTest CONFIG)
  if(NOT GTEST_FOUND)
    message(STATUS "Did not find Google Test in the system root. Fetching Google Test now...")
    include(FetchContent)
# lint_cmake: -readability/wonkycase
    FetchContent_Declare(
# lint_cmake: +readability/wonkycase
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
    )
# lint_cmake: -readability/wonkycase
    FetchContent_MakeAvailable(googletest)
# lint_cmake: +readability/wonkycase
    add_library(GTest::GTest ALIAS gtest)
  endif()

  set(LightGBM_TEST_HEADER_DIR ${PROJECT_SOURCE_DIR}/tests/cpp_tests)
  include_directories(${LightGBM_TEST_HEADER_DIR})

  set(
    CPP_TEST_SOURCES
      tests/cpp_tests/test_array_args.cpp
      tests/cpp_tests/test_arrow.cpp
      tests/cpp_tests/test_arrow_deprecated.cpp
      tests/cpp_tests/test_byte_buffer.cpp
      tests/cpp_tests/test_chunked_array.cpp
      tests/cpp_tests/test_common.cpp
      tests/cpp_tests/test_main.cpp
      tests/cpp_tests/test_serialize.cpp
      tests/cpp_tests/test_single_row.cpp
      tests/cpp_tests/test_split_info.cpp
      tests/cpp_tests/test_stream.cpp
      tests/cpp_tests/testutils.cpp
    )
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
  endif()
  add_executable(testlightgbm ${CPP_TEST_SOURCES})
  target_link_libraries(testlightgbm PRIVATE lightgbm_objs lightgbm_capi_objs nanoarrow_static GTest::GTest)
endif()

if(BUILD_CLI)
    install(
      TARGETS lightgbm
      RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    )
endif()

if(__BUILD_FOR_PYTHON)
    set(CMAKE_INSTALL_PREFIX "lightgbm")
endif()

# The macOS linker puts an absolute path to linked libraries in lib_lightgbm.dylib.
# This block overrides that information for LightGBM's OpenMP dependency, to allow
# finding that library in more places.
#
# This reduces the risk of runtime issues resulting from multiple {libgomp,libiomp,libomp}.dylib being loaded.
#
if(APPLE AND USE_OPENMP AND NOT BUILD_STATIC_LIB)
  # store path to {libgomp,libiomp,libomp}.dylib found at build time in a variable
  get_target_property(
    OpenMP_LIBRARY_LOCATION
    OpenMP::OpenMP_CXX
    INTERFACE_LINK_LIBRARIES
  )
  # get just the filename of that path
  # (to deal with the possibility that it might be 'libomp.dylib' or 'libgomp.dylib' or 'libiomp.dylib')
  get_filename_component(
    OpenMP_LIBRARY_NAME
    ${OpenMP_LIBRARY_LOCATION}
    NAME
  )
  # get directory of that path
  get_filename_component(
    OpenMP_LIBRARY_DIR
    ${OpenMP_LIBRARY_LOCATION}
    DIRECTORY
  )
  # get exact name of the library in a variable
  get_target_property(
    __LIB_LIGHTGBM_OUTPUT_NAME
    _lightgbm
    OUTPUT_NAME
  )
  if(NOT __LIB_LIGHTGBM_OUTPUT_NAME)
    set(__LIB_LIGHTGBM_OUTPUT_NAME "lib_lightgbm")
  endif()

  if(CMAKE_SHARED_LIBRARY_SUFFIX_CXX)
    set(
      __LIB_LIGHTGBM_FILENAME "${__LIB_LIGHTGBM_OUTPUT_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX_CXX}"
      CACHE INTERNAL "lightgbm shared library filename"
    )
  else()
    set(
      __LIB_LIGHTGBM_FILENAME "${__LIB_LIGHTGBM_OUTPUT_NAME}.dylib"
      CACHE INTERNAL "lightgbm shared library filename"
    )
  endif()

  # Override the absolute path to OpenMP with a relative one using @rpath.
  #
  # This also ensures that if a {libgomp,libiomp,libomp}.dylib has already been loaded, it'll just use that.
  add_custom_command(
    TARGET _lightgbm
    POST_BUILD
      COMMAND
        install_name_tool
        -change
        ${OpenMP_LIBRARY_LOCATION}
        "@rpath/${OpenMP_LIBRARY_NAME}"
        "${__LIB_LIGHTGBM_FILENAME}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Replacing hard-coded OpenMP install_name with '@rpath/${OpenMP_LIBRARY_NAME}'..."
  )
  # add RPATH entries to ensure the loader looks in the following, in the following order:
  #
  #   - (R-only) ${LIBR_LIBS_DIR}    (wherever R for macOS stores vendored third-party libraries)
  #   - ${OpenMP_LIBRARY_DIR}        (wherever find_package(OpenMP) found OpenMP at build time)
  #   - /opt/homebrew/opt/libomp/lib (where 'brew install' / 'brew link' puts libomp.dylib)
  #   - /opt/local/lib/libomp        (where 'port install' puts libomp.dylib)
  #

  # with some compilers, OpenMP ships with the compiler (e.g. libgomp with gcc)
  list(APPEND __omp_install_rpaths "${OpenMP_LIBRARY_DIR}")

  # with clang, libomp doesn't ship with the compiler and might be supplied separately
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      list(
        APPEND __omp_install_rpaths
          "/opt/homebrew/opt/libomp/lib"
          "/opt/local/lib/libomp"
      )
      # It appears that CRAN's macOS binaries compiled with -fopenmp have install names
      # of the form:
      #
      #   /Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/lib/libomp.dylib
      #
      # That corresponds to the libomp.dylib that ships with the R framework for macOS, available
      # from https://cran.r-project.org/bin/macosx/.
      #
      # That absolute-path install name leads to that library being loaded unconditionally.
      #
      # That can result in e.g. 'library(data.table)' loading R's libomp.dylib and 'library(lightgbm)' loading
      # Homebrew's. Having 2 loaded in the same process can lead to segfaults and unpredictable behavior.
      #
      # This can't be easily avoided by forcing R-package builds in LightGBM to use R's libomp.dylib
      # at build time... LightGBM's CMake uses find_package(OpenMP), and R for macOS only provides the
      # library, not CMake config files for it.
      #
      # Best we can do, to allow CMake-based builds of the R-package here to continue to work
      # alongside CRAN-prepared binaries of other packages with OpenMP dependencies, is to
      # ensure that R's library directory is the first place the loader searches for
      # libomp.dylib when clang is used.
      #
      # ref: https://github.com/lightgbm-org/LightGBM/issues/6628
      #
      if(__BUILD_FOR_R)
        list(PREPEND __omp_install_rpaths "${LIBR_LIBS_DIR}")
      endif()
  endif()
  set_target_properties(
    _lightgbm
    PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "${__omp_install_rpaths}"
      INSTALL_RPATH_USE_LINK_PATH FALSE
  )
endif()

install(
  TARGETS _lightgbm
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

if(INSTALL_HEADERS)
    install(
      DIRECTORY ${LightGBM_HEADER_DIR}/LightGBM
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include
    )
    install(
      FILES ${FAST_DOUBLE_PARSER_INCLUDE_DIR}/fast_double_parser.h
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/LightGBM/utils
    )
    install(
      DIRECTORY ${FMT_INCLUDE_DIR}/
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/LightGBM/utils
      FILES_MATCHING PATTERN "*.h"
    )
endif()
