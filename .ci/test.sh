#!/bin/bash

set -e -E -o -u pipefail

# defaults
CONDA_ENV="test-env"
IN_UBUNTU_BASE_CONTAINER=${IN_UBUNTU_BASE_CONTAINER:-"false"}
METHOD=${METHOD:-""}
PRODUCES_ARTIFACTS=${PRODUCES_ARTIFACTS:-"false"}
SANITIZERS=${SANITIZERS:-""}

ARCH=$(uname -m)

LGB_VER=$(head -n 1 "${BUILD_DIRECTORY}/VERSION.txt")

# create the artifact upload directory if it doesn't exist yet
mkdir -p "${BUILD_ARTIFACTSTAGINGDIRECTORY}"

if [[ $OS_NAME == "macos" ]] && [[ $COMPILER == "gcc" ]]; then
    export CXX=g++-14
    export CC=gcc-14
elif [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "clang" ]]; then
    export CXX=clang++
    export CC=clang
elif [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "clang-17" ]]; then
    export CXX=clang++-17
    export CC=clang-17
fi

if [[ $IN_UBUNTU_BASE_CONTAINER == "true" ]]; then
    export LANG="en_US.UTF-8"
    export LC_ALL="en_US.UTF-8"
fi

# Setting MACOSX_DEPLOYMENT_TARGET prevents CMake from building against too-new
# macOS features, and helps tools like Python build tools determine the appropriate
# wheel compatibility tags.
#
# ref:
#   * https://cmake.org/cmake/help/latest/envvar/MACOSX_DEPLOYMENT_TARGET.html
#   * https://github.com/scikit-build/scikit-build-core/blob/acb7d0346e4a05bcb47a4ea3939c705ab71e3145/src/scikit_build_core/builder/macos.py#L36
if [[ $ARCH == "x86_64" ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.15
else
    export MACOSX_DEPLOYMENT_TARGET=12.0
fi

if [[ "${TASK}" == "r-package" ]]; then
    bash "${BUILD_DIRECTORY}/.ci/test-r-package.sh" || exit 1
    exit 0
fi

cd "${BUILD_DIRECTORY}"

if [[ $TASK == "swig" ]]; then
    cmake -B build -S . -DUSE_SWIG=ON
    cmake --build build -j4 || exit 1
    if [[ $OS_NAME == "linux" ]] && [[ $COMPILER == "gcc" ]]; then
        objdump -T ./lib_lightgbm.so > ./objdump.log || exit 1
        objdump -T ./lib_lightgbm_swig.so >> ./objdump.log || exit 1
        ./.ci/check-dynamic-dependencies.sh ./objdump.log || exit 1
    fi
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        cp ./build/lightgbmlib.jar "${BUILD_ARTIFACTSTAGINGDIRECTORY}/lightgbmlib_${OS_NAME}.jar"
    fi
    exit 0
fi

if [[ "$TASK" == "cpp-tests" ]]; then
    cmake_args=(
        -DBUILD_CPP_TEST=ON
        -DUSE_DEBUG=ON
    )
    if [[ $METHOD == "with-sanitizers" ]]; then
        cmake_args+=("-DUSE_SANITIZER=ON")
        if [[ -n $SANITIZERS ]]; then
            cmake_args+=("-DENABLED_SANITIZERS=$SANITIZERS")
        fi
    fi
    cmake -B build -S . "${cmake_args[@]}"
    cmake --build build --target testlightgbm -j4 || exit 1
    ./testlightgbm || exit 1
    exit 0
fi

# From Python 3.13 onwards, CPython packages on conda-forge have build strings
# with formats like "*_cp314".
#
# Have to be specific here (no trailing wildcard) to avoid unintentionally pulling
# in free-threaded builds (e.g. "*_cp314t").
PYTHON_MAJOR_VERSION="${PYTHON_VERSION%%.*}"
PYTHON_MINOR_VERSION="${PYTHON_VERSION#*.}"
if (( PYTHON_MAJOR_VERSION > 3 || (PYTHON_MAJOR_VERSION == 3 && PYTHON_MINOR_VERSION > 12) )); then
    PYTHON_ABI_TAG="cp${PYTHON_VERSION//./}"
else
    PYTHON_ABI_TAG="cpython"
fi
CONDA_PYTHON_REQUIREMENT="python=${PYTHON_VERSION}[build=*_${PYTHON_ABI_TAG}]"

if [[ $TASK == "if-else" ]]; then
    conda create -q -y -n "${CONDA_ENV}" "${CONDA_PYTHON_REQUIREMENT}" numpy
    # shellcheck disable=SC1091
    source activate "${CONDA_ENV}"
    cmake -B build -S . || exit 1
    cmake --build build --target lightgbm -j4 || exit 1
    cd "$BUILD_DIRECTORY/tests/cpp_tests"
    ../../lightgbm config=train.conf convert_model_language=cpp convert_model=../../src/boosting/gbdt_prediction.cpp
    ../../lightgbm config=predict.conf output_result=origin.pred
    ../../lightgbm config=predict.conf output_result=ifelse.pred
    python test.py
    exit 0
fi

PYTHON_ENV_MANAGER="conda"
if [[ "${PYTHON_VERSION}" == "3.10" ]]; then
    PYTHON_ENV_MANAGER="pixi"
    CI_PIXI_ENV="py310"
fi

# 'pixi' is used for end-of-life Python versions
if [[ "${PYTHON_ENV_MANAGER}" == "pixi" ]]; then
    eval "$(pixi shell-hook --locked -e "${CI_PIXI_ENV}")"
else
    CONDA_REQUIREMENT_FILE="${BUILD_DIRECTORY}/.ci/conda-envs/ci-core.txt"
    conda create \
        -q \
        -y \
        -n "${CONDA_ENV}" \
        --file "${CONDA_REQUIREMENT_FILE}" \
        "${CONDA_PYTHON_REQUIREMENT}" \
    || exit 1

    # print output of 'conda list', to help in submitting bug reports
    echo "conda list:"
    conda list -n ${CONDA_ENV}

    # shellcheck disable=SC1091
    source activate $CONDA_ENV
fi

cd "${BUILD_DIRECTORY}"

if [[ $TASK == "sdist" ]]; then
    sh ./build-python.sh sdist || exit 1
    sh .ci/check-python-dists.sh ./dist || exit 1
    pip install -v --no-deps "./dist/lightgbm-${LGB_VER}.tar.gz" || exit 1
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        cp "./dist/lightgbm-${LGB_VER}.tar.gz" "${BUILD_ARTIFACTSTAGINGDIRECTORY}" || exit 1
    fi
    pytest -ra ./tests/python_package_test || exit 1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    if [[ $OS_NAME == "macos" ]]; then
        sh ./build-python.sh bdist_wheel || exit 1
        sh .ci/check-python-dists.sh ./dist || exit 1
        if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
            cp "$(echo "dist/lightgbm-${LGB_VER}-py3-none-macosx"*.whl)" "${BUILD_ARTIFACTSTAGINGDIRECTORY}" || exit 1
        fi
    else
        BUILD_PYTHON_FLAGS=()
        if [[ "${ARCH}" != "ppc64le" ]]; then
            BUILD_PYTHON_FLAGS+=(--integrated-opencl)
        fi

        sh ./build-python.sh bdist_wheel "${BUILD_PYTHON_FLAGS[@]}" || exit 1

        # print some debugging logs about the wheel's GLIBC version and dependencies on shared libraries
        pip install 'auditwheel>=6.5.1'
        auditwheel show ./dist/lightgbm*.whl

        # pass through 'auditwheel repair' to set the appropriate wheel tags.
        #
        # intentionally avoid vendoring libgomp, to reduce the risk of multiple OpenMP libraries
        # being loaded in the same process.
        auditwheel repair \
            --exclude 'libgomp.so*' \
            --lib-sdir '' \
            --wheel-dir dist-fixed/ \
            ./dist/lightgbm*.whl

        # overwrite the original wheel with the new one
        rm ./dist/lightgbm*.whl
        mv ./dist-fixed/lightgbm*.whl ./dist

        # check wheel properties
        sh .ci/check-python-dists.sh ./dist || exit 1

        if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
            # hard-code expected tag so CI will fail if 'auditwheel repair' has a surprising result (e.g. newer
            # manylinux tag than we intended)
            if [[ $ARCH == "x86_64" ]]; then
                PLATFORM="manylinux_2_27_x86_64.manylinux_2_28_x86_64"
            elif [[ $ARCH == "ppc64le" ]]; then
                PLATFORM="manylinux_2_27_ppc64le.manylinux_2_28_ppc64le"
            else
                PLATFORM="manylinux2014_aarch64.manylinux_2_17_aarch64"
            fi
            cp "dist/lightgbm-${LGB_VER}-py3-none-${PLATFORM}.whl" "${BUILD_ARTIFACTSTAGINGDIRECTORY}" || exit 1
        fi
        if [[ "${ARCH}" != "ppc64le" ]]; then
            # Make sure we can do both CPU and GPU; see tests/python_package_test/test_dual.py
            export LIGHTGBM_TEST_DUAL_CPU_GPU=1
        fi
    fi
    pip install -v --no-deps ./dist/*.whl || exit 1
    pytest -ra ./tests || exit 1
    exit 0
fi

if [[ $TASK == "gpu" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "gpu";/' ./include/LightGBM/config.h
    grep -q 'std::string device_type = "gpu"' ./include/LightGBM/config.h || exit 1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        sh ./build-python.sh sdist || exit 1
        sh .ci/check-python-dists.sh ./dist || exit 1
        pip install \
            -v \
            --no-deps \
            --config-settings=cmake.define.USE_GPU=ON \
            "./dist/lightgbm-${LGB_VER}.tar.gz" \
        || exit 1
        pytest -ra ./tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        sh ./build-python.sh bdist_wheel --gpu || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install -v --no-deps "$(echo "./dist/lightgbm-${LGB_VER}"*.whl)" || exit 1
        pytest -ra ./tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        cmake -B build -S . -DUSE_GPU=ON
    fi
elif [[ $TASK == "cuda" ]]; then
    sed -i'.bak' 's/std::string device_type = "cpu";/std::string device_type = "cuda";/' ./include/LightGBM/config.h
    grep -q 'std::string device_type = "cuda"' ./include/LightGBM/config.h || exit 1  # make sure that changes were really done
    # by default ``gpu_use_dp=false`` for efficiency. change to ``true`` here for exact results in ci tests
    sed -i'.bak' 's/gpu_use_dp = false;/gpu_use_dp = true;/' ./include/LightGBM/config.h
    grep -q 'gpu_use_dp = true' ./include/LightGBM/config.h || exit 1  # make sure that changes were really done
    if [[ $METHOD == "pip" ]]; then
        sh ./build-python.sh sdist || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install \
            -v \
            --no-deps \
            --config-settings=cmake.define.USE_CUDA=ON \
            "./dist/lightgbm-${LGB_VER}.tar.gz" \
        || exit 1
        pytest -ra ./tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        sh ./build-python.sh bdist_wheel --cuda || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install -v --no-deps "$(echo "./dist/lightgbm-${LGB_VER}"*.whl)" || exit 1
        pytest -ra ./tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        # we want at least 1 CI job testing that manual override of CMAKE_CUDA_ARCHITECTURES works
        cmake \
            -B build \
            -S . \
            -DUSE_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES="native"
    fi
elif [[ $TASK == "mpi" ]]; then
    if [[ $METHOD == "pip" ]]; then
        sh ./build-python.sh sdist || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install \
            -v \
            --no-deps \
            --config-settings=cmake.define.USE_MPI=ON \
            "./dist/lightgbm-${LGB_VER}.tar.gz" \
        || exit 1
        pytest -ra ./tests/python_package_test || exit 1
        exit 0
    elif [[ $METHOD == "wheel" ]]; then
        sh ./build-python.sh bdist_wheel --mpi || exit 1
        sh ./.ci/check-python-dists.sh ./dist || exit 1
        pip install -v --no-deps "$(echo "./dist/lightgbm-${LGB_VER}"*.whl)" || exit 1
        pytest -ra ./tests || exit 1
        exit 0
    elif [[ $METHOD == "source" ]]; then
        cmake -B build -S . -DUSE_MPI=ON -DUSE_DEBUG=ON
    fi
else
    cmake -B build -S .
fi

cmake --build build --target _lightgbm -j4 || exit 1

sh ./build-python.sh install --precompile || exit 1

# This MPI skip can be removed when those jobs do not use conda.
# ref: https://github.com/lightgbm-org/LightGBM/issues/7355#issuecomment-5578260518
if [[ "${TASK}" != "mpi" ]]; then
    echo "testing imports with 'python -O1'"
    PYTHONOPTIMIZE=1 python -c "import lightgbm; print(lightgbm.__version__)"

    echo "testing imports with 'python -O2'"
    PYTHONOPTIMIZE=2 python -c "import lightgbm; print(lightgbm.__version__)"
fi

echo "running tests"
pytest -ra ./tests || exit 1

if [[ $TASK == "regular" ]]; then
    if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
        if [[ $OS_NAME == "macos" ]]; then
            cp ./lib_lightgbm.dylib "${BUILD_ARTIFACTSTAGINGDIRECTORY}/lib_lightgbm.dylib"
        else
            if [[ $COMPILER == "gcc" ]]; then
                objdump -T ./lib_lightgbm.so > ./objdump.log || exit 1
                ./.ci/check-dynamic-dependencies.sh ./objdump.log || exit 1
            fi
            cp ./lib_lightgbm.so "${BUILD_ARTIFACTSTAGINGDIRECTORY}/lib_lightgbm.so"
        fi
    fi
    cd "$BUILD_DIRECTORY/examples/python-guide"
    sed -i'.bak' '/import lightgbm as lgb/a\
import matplotlib\
matplotlib.use\(\"Agg\"\)\
' plot_example.py  # prevent interactive window mode
    sed -i'.bak' 's/graph.render(view=True)/graph.render(view=False)/' plot_example.py
    # install optional plotting libraries
    # (not necessary for pixi-managed environments, where they're just installed by default)
    if [[ "${PYTHON_ENV_MANAGER}" != "pixi" ]]; then
        conda install -q -y -n $CONDA_ENV \
            'h5py>=3.10' \
            'ipywidgets>=8.1.2' \
            'notebook>=7.1.2'
    fi
    for f in *.py **/*.py; do python "${f}" || exit 1; done  # run all examples
    cd "$BUILD_DIRECTORY/examples/python-guide/notebooks"
    sed -i'.bak' 's/INTERACTIVE = False/assert False, \\"Interactive mode disabled\\"/' interactive_plot_example.ipynb
    jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace ./*.ipynb || exit 1  # run all notebooks

    # importing the library should succeed even if all optional dependencies are not present
    if [[ "${PYTHON_ENV_MANAGER}" != "pixi" ]]; then
        conda uninstall -q -n $CONDA_ENV --force --yes \
            cffi \
            dask \
            distributed \
            joblib \
            matplotlib-base \
            pandas \
            polars \
            psutil \
            pyarrow \
            python-graphviz \
            scikit-learn || exit 1
    fi
    python -c "import lightgbm" || exit 1
fi
