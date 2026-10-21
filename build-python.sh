#!/bin/sh

# [description]
#
#     Prepare a source distribution (sdist) or built distribution (wheel)
#     of the Python-package, and optionally install it.
#
# [usage]
#
#     # build sdist and put it in dist/
#     sh ./build-python.sh sdist
#
#     # build wheel and put it in dist/
#     sh ./build-python.sh bdist_wheel [OPTIONS]
#
#     # compile lib_lightgbm and install the Python-package wrapping it
#     sh ./build-python.sh install [OPTIONS]
#
#     # install the Python-package using a pre-compiled lib_lightgbm
#     # (assumes lib_lightgbm.{dll,so} is located at the root of the repo)
#     sh ./build-python.sh install --precompile
#
# [options]
#
#     --boost-dir=FILEPATH
#                                   Directory with Boost package configuration file.
#     --boost-include-dir=FILEPATH
#                                   Directory containing Boost headers.
#     --boost-librarydir=FILEPATH
#                                   Preferred Boost library directory.
#     --boost-root=FILEPATH
#                                   Boost preferred installation prefix.
#     --opencl-include-dir=FILEPATH
#                                   OpenCL include directory.
#     --opencl-library=FILEPATH
#                                   Path to OpenCL library.
#     --bit32
#                                   Compile 32-bit version.
#     --cuda
#                                   Compile CUDA version.
#     --gpu
#                                   Compile GPU version.
#     --integrated-opencl
#                                   Compile integrated OpenCL version.
#     --mingw
#                                   Compile with MinGW.
#     --mpi
#                                   Compile MPI version.
#     --no-isolation
#                                   Assume all build and install dependencies are already installed,
#                                   don't go to the internet to get them.
#     --nomp
#                                   Compile version without OpenMP support.
#     --precompile
#                                   Use precompiled library.
#                                   Only used with 'install' command.
#     --rocm
#                                   Compile ROCm version.
#     --time-costs
#                                   Compile version that outputs time costs for different internal routines.
#     --user
#                                   Install into user-specific instead of global site-packages directory.
#                                   Only used with 'install' command.

set -e -u

echo "[INFO] building lightgbm"

# Default values of arguments
INSTALL="false"
BUILD_SDIST="false"
BUILD_WHEEL="false"

PIP_INSTALL_ARGS=""
BUILD_ARGS=""
PRECOMPILE="false"

while [ $# -gt 0 ]; do
  case "$1" in
    ############################
    # sub-commands of setup.py #
    ############################
    install)
      INSTALL="true"
      ;;
    sdist)
      BUILD_SDIST="true"
      ;;
    bdist_wheel)
      BUILD_WHEEL="true"
      ;;
    ############################
    # customized library paths #
    ############################
    --boost-dir|--boost-dir=*)
        if echo "$1" | grep -q '^*=*$';
            then shift;
        fi
        BOOST_DIR="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.Boost_DIR='${BOOST_DIR}'"
        ;;
    --boost-include-dir|--boost-include-dir=*)
        if echo "$1" | grep -q '^*=*$';
            then shift;
        fi
        BOOST_INCLUDE_DIR="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.Boost_INCLUDE_DIR='${BOOST_INCLUDE_DIR}'"
        ;;
    --boost-librarydir|--boost-librarydir=*)
        if echo "$1" | grep -q '^*=*$';
            then shift;
        fi
        BOOST_LIBRARY_DIR="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.BOOST_LIBRARYDIR='${BOOST_LIBRARY_DIR}'"
        ;;
    --boost-root|--boost-root=*)
        if echo "$1" | grep -q '^*=*$';
            then shift;
        fi
        BOOST_ROOT="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.Boost_ROOT='${BOOST_ROOT}'"
        ;;
    --opencl-include-dir|--opencl-include-dir=*)
        if echo "$1" | grep -q '^*=*$';
            then shift;
        fi
        OPENCL_INCLUDE_DIR="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.OpenCL_INCLUDE_DIR='${OPENCL_INCLUDE_DIR}'"
        ;;
    --opencl-library|--opencl-library=*)
        if echo "$1" | grep -q '^*=*$';
            then shift;
        fi
        OPENCL_LIBRARY="${1#*=}"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.OpenCL_LIBRARY='${OPENCL_LIBRARY}'"
        ;;
    #########
    # flags #
    #########
    --bit32)
        echo "[INFO] Attempting to build 32-bit version of LightGBM, which is only supported on Windows with Visual Studio."
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.args=-AWin32"
        ;;
    --cuda)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_CUDA=ON"
        ;;
    --rocm)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_ROCM=ON"
        ;;
    --gpu)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_GPU=ON"
        ;;
    --integrated-opencl)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.__INTEGRATE_OPENCL=ON"
        ;;
    --mingw)
        # ref: https://stackoverflow.com/a/45104058/3986677
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.CMAKE_SH=CMAKE_SH-NOTFOUND"
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.args=-G'MinGW Makefiles'"
        ;;
    --mpi)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_MPI=ON"
        ;;
    --no-isolation)
        BUILD_ARGS="${BUILD_ARGS} --no-isolation"
        PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS} --no-build-isolation"
        ;;
    --nomp)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_OPENMP=OFF"
        ;;
    --precompile)
        PRECOMPILE="true"
        ;;
    --time-costs)
        BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.USE_TIMETAG=ON"
        ;;
    --user)
        PIP_INSTALL_ARGS="${PIP_INSTALL_ARGS} --user"
        ;;
    *)
        echo "[ERROR] invalid argument '${1}'. Aborting"
        exit 1
        ;;
  esac
  shift
done

# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
if [ -n "${CUDAARCHS:-}" ]; then
    BUILD_ARGS="${BUILD_ARGS} --config-setting=cmake.define.CMAKE_CUDA_ARCHITECTURES=${CUDAARCHS}"
fi

if ! python -m build --version >/dev/null; then
    echo "'build' is required to build 'lightgbm'. Install it with 'pip install build' or similar."
    exit 1
fi

# create a new directory that just contains the files needed
# to build the Python-package
create_isolated_source_dir() {
    rm -rf \
        ./lightgbm-python \
        ./lightgbm \
        ./python-package/build \
        ./python-package/build_cpp \
        ./python-package/compile \
        ./python-package/dist \
        ./python-package/lightgbm.egg-info

    cp -R ./python-package ./lightgbm-python

    cp LICENSE ./lightgbm-python/
    cp VERSION.txt ./lightgbm-python/lightgbm/VERSION.txt

    cp -R ./cmake ./lightgbm-python
    cp CMakeLists.txt ./lightgbm-python
    cp -R ./include ./lightgbm-python
    cp -R ./src ./lightgbm-python
    cp -R ./swig ./lightgbm-python

    # include only specific files from external_libs, to keep the package
    # small and avoid redistributing code with licenses incompatible with
    # LightGBM's license

    ##############
    # fast_float #
    ##############
    # Ship fast_float as the single amalgamated header generated by its own
    # 'script/amalgamate.py' (with all license notices embedded as comments),
    # instead of copying its individual files. This keeps the sdist's file
    # count from growing by more files than necessary.
    FAST_FLOAT_HEADER_DIR="$(pwd)/lightgbm-python/external_libs/fast_float/include/fast_float"
    mkdir -p "${FAST_FLOAT_HEADER_DIR}"
    (
        cd external_libs/fast_float
        python script/amalgamate.py \
            --license TRIPLE \
            --output "${FAST_FLOAT_HEADER_DIR}/fast_float.h"
    )

    #######
    # fmt #
    #######
    mkdir -p ./lightgbm-python/external_libs/fmt
    cp \
        external_libs/fmt/CMakeLists.txt \
        ./lightgbm-python/external_libs/fmt/CMakeLists.txt
    cp \
        external_libs/fmt/LICENSE* \
        ./lightgbm-python/external_libs/fmt/

    mkdir -p ./lightgbm-python/external_libs/fmt/include/fmt
    cp \
        external_libs/fmt/include/fmt/*.h \
        ./lightgbm-python/external_libs/fmt/include/fmt/

    #########
    # Eigen #
    #########
    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen
    cp \
        external_libs/eigen/CMakeLists.txt \
        ./lightgbm-python/external_libs/eigen/CMakeLists.txt

    modules="Cholesky Core Dense Eigenvalues Geometry Householder Jacobi LU QR SVD"
    for eigen_module in ${modules}; do
        cp \
            "external_libs/eigen/Eigen/${eigen_module}" \
            "./lightgbm-python/external_libs/eigen/Eigen/${eigen_module}"
        if [ "${eigen_module}" != "Dense" ]; then
            mkdir -p "./lightgbm-python/external_libs/eigen/Eigen/src/${eigen_module}/"
            cp \
                -R \
                "external_libs/eigen/Eigen/src/${eigen_module}"/* \
                "./lightgbm-python/external_libs/eigen/Eigen/src/${eigen_module}/"
        fi
    done

    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen/misc
    cp \
        -R \
        external_libs/eigen/Eigen/src/misc \
        ./lightgbm-python/external_libs/eigen/Eigen/src/misc/

    mkdir -p ./lightgbm-python/external_libs/eigen/Eigen/plugins
    cp \
        -R \
        external_libs/eigen/Eigen/src/plugins \
        ./lightgbm-python/external_libs/eigen/Eigen/src/plugins/

    ###################
    # compute (Boost) #
    ###################
    mkdir -p ./lightgbm-python/external_libs/compute
    cp \
        -R \
        external_libs/compute/include \
        ./lightgbm-python/external_libs/compute/include/

    #############
    # nanoarrow #
    #############
    mkdir -p ./lightgbm-python/external_libs/nanoarrow
    cp \
        external_libs/nanoarrow/CMakeLists.txt \
        external_libs/nanoarrow/LICENSE.txt \
        external_libs/nanoarrow/NOTICE.txt \
        ./lightgbm-python/external_libs/nanoarrow/
    cp -R \
        external_libs/nanoarrow/cmake \
        ./lightgbm-python/external_libs/nanoarrow/cmake/
    mkdir -p ./lightgbm-python/external_libs/nanoarrow/src/nanoarrow
    cp \
        external_libs/nanoarrow/src/nanoarrow/nanoarrow.h \
        external_libs/nanoarrow/src/nanoarrow/nanoarrow.hpp \
        external_libs/nanoarrow/src/nanoarrow/nanoarrow_config.h.in \
        ./lightgbm-python/external_libs/nanoarrow/src/nanoarrow/
    mkdir -p ./lightgbm-python/external_libs/nanoarrow/src/nanoarrow/common
    cp -R \
        external_libs/nanoarrow/src/nanoarrow/common/*.h \
        ./lightgbm-python/external_libs/nanoarrow/src/nanoarrow/common
    cp -R \
        external_libs/nanoarrow/src/nanoarrow/common/*.c \
        ./lightgbm-python/external_libs/nanoarrow/src/nanoarrow/common
    mkdir -p ./lightgbm-python/external_libs/nanoarrow/src/nanoarrow/hpp
    cp -R \
        external_libs/nanoarrow/src/nanoarrow/hpp/*.hpp \
        ./lightgbm-python/external_libs/nanoarrow/src/nanoarrow/hpp
}

create_isolated_source_dir

cd ./lightgbm-python

# if 'install' was passed, choose the type of package to build and install
if test "${INSTALL}" = true; then
    if test "${PRECOMPILE}" = true; then
        BUILD_SDIST=false
        BUILD_WHEEL=true
        rm -rf \
            ./cmake \
            ./CMakeLists.txt \
            ./external_libs \
            ./include \
            ./src \
            ./swig
        # avoid trying to recompile, just use hatchling and copy in relevant files
        sed -i.bak -e '/start:build-system/,/end:build-system/d' pyproject.toml

# replace build backend configuration
cat >> ./pyproject.toml <<EOF

[build-system]
requires = ["hatchling>=1.27.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
# do not consider .gitignore when choosing files to include / exclude
ignore-vcs = true
packages = ["lightgbm"]

EOF
        mkdir -p ./lightgbm/lib
        if test -f ../lib_lightgbm.so; then
            echo "[INFO] found pre-compiled lib_lightgbm.so"
            cp ../lib_lightgbm.so ./lightgbm/lib/lib_lightgbm.so
        elif test -f ../lib_lightgbm.dylib; then
            echo "[INFO] found pre-compiled lib_lightgbm.dylib"
            cp ../lib_lightgbm.dylib ./lightgbm/lib/lib_lightgbm.dylib
        elif test -f ../lib_lightgbm.dll; then
            echo "[INFO] found pre-compiled lib_lightgbm.dll"
            cp ../lib_lightgbm.dll ./lightgbm/lib/lib_lightgbm.dll
        elif test -f ../Release/lib_lightgbm.dll; then
            echo "[INFO] found pre-compiled Release/lib_lightgbm.dll"
            cp ../Release/lib_lightgbm.dll ./lightgbm/lib/lib_lightgbm.dll
        elif test -f ../windows/x64/DLL/lib_lightgbm.dll; then
            echo "[INFO] found pre-compiled windows/x64/DLL/lib_lightgbm.dll"
            cp ../windows/x64/DLL/lib_lightgbm.dll ./lightgbm/lib/lib_lightgbm.dll
            cp ../windows/x64/DLL/lib_lightgbm.lib ./lightgbm/lib/lib_lightgbm.lib
        elif test -f ../windows/x64/Debug_DLL/lib_lightgbm.dll; then
            echo "[INFO] found pre-compiled windows/x64/Debug_DLL/lib_lightgbm.dll"
            cp ../windows/x64/Debug_DLL/lib_lightgbm.dll ./lightgbm/lib/lib_lightgbm.dll
            cp ../windows/x64/Debug_DLL/lib_lightgbm.lib ./lightgbm/lib/lib_lightgbm.lib
        else
            echo "[ERROR] cannot find pre-compiled library. Aborting"
            exit 1
        fi
        rm -f ./*.bak
    fi

    # at this point, if 'install' was passed but the package type wasn't indicated, prefer wheel
    if test "${BUILD_SDIST}" = false && test "${BUILD_WHEEL}" = false; then
        echo "[INFO] 'install' passed but no package type ('bdist_wheel', 'sdist') chosen. Defaulting to 'bdist_wheel'."
        BUILD_SDIST="false"
        BUILD_WHEEL="true"
    fi
fi

if test "${BUILD_SDIST}" = true; then
    echo "[INFO] --- building sdist ---"
    rm -f ../dist/*.tar.gz
    # use xargs to work with args that contain whitespaces
    # note that empty echo string leads to that xargs doesn't run the command
    # in some implementations of xargs
    # ref: https://stackoverflow.com/a/8296746
    echo "--sdist --outdir ../dist ${BUILD_ARGS} ." | xargs python -m build
fi

if test "${BUILD_WHEEL}" = true; then
    echo "[INFO] --- building wheel ---"
    rm -f ../dist/*.whl || true
    # use xargs to work with args that contain whitespaces
    # note that empty echo string leads to that xargs doesn't run the command
    # in some implementations of xargs
    # ref: https://stackoverflow.com/a/8296746
    echo "--wheel --outdir ../dist ${BUILD_ARGS} ." | xargs python -m build
fi

if test "${INSTALL}" = true; then
    echo "[INFO] --- installing lightgbm ---"
    cd ..
    if test "${BUILD_WHEEL}" = true; then
        PACKAGE_FILE="$(echo dist/lightgbm*.whl)"
    else
        PACKAGE_FILE="$(echo dist/lightgbm*.tar.gz)"
    fi
    # shellcheck disable=SC2086
    python -m pip install \
        ${PIP_INSTALL_ARGS} \
        --force-reinstall \
        --no-cache-dir \
        --no-deps \
        "${PACKAGE_FILE}"
fi

echo "[INFO] cleaning up"
rm -rf ./lightgbm-python
