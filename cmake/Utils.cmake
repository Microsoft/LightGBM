# Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
# Copyright (c) 2026 The XGBoost developers.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

# Generate CMAKE_CUDA_ARCHITECTURES form a list of architectures.
#
# lint_cmake: -linelength
# Adopted from XGBoost, see https://github.com/dmlc/xgboost/blob/41591eeff5b10caabe82d65fc7164b447eabd249/cmake/Utils.cmake#L62
# lint_cmake: +linelength
#
function(compute_cmake_cuda_archs _cuda_version)

  # Set up defaults based on CUDA version
  # Remember to update arch-specific tunings when supporting new archs.
  #
  # Reference for mapping of CUDA toolkit component versions to supported architectures ("compute capabilities"):
  # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
  #
  if(_cuda_version VERSION_GREATER_EQUAL "13.0")
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 120)
  elseif(_cuda_version VERSION_GREATER_EQUAL "12.9")
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 100 120)
  elseif(_cuda_version VERSION_GREATER_EQUAL "12.8")
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 80 86 89 90 100 120)
  elseif(_cuda_version VERSION_GREATER_EQUAL "11.8")
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 80 86 89 90)
  else()
    message(FATAL_ERROR
      "No default architecture list configured for CUDA version '${_cuda_version}'. "
      "You can avoid this error and attempt building by manually passing '-DCMAKE_CUDA_ARCHITECTURES'."
    )
  endif()

  # For the last architectures, generate both:
  #
  #   - SASS via '-real', for performance
  #   - PTX via '-virtual', for forward compatibility
  #
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "([0-9]+)-real" "\\0;\\1-virtual" AT -1)
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endfunction()
