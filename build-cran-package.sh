#!/bin/sh

# [description]
#     Prepare a source distribution of the R-package
#     to be submitted to CRAN.
#
# [arguments]
#
#     --r-executable Customize the R executable used by `R CMD build`.
#                    Useful if building the R-package in an environment with
#                    non-standard builds of R, such as those provided in
#                    https://github.com/wch/r-debug.
#
#     --no-build-vignettes Pass this flag to skip creating vignettes.
#                          You might want to do this to avoid installing
#                          vignette-only dependencies, or to avoid
#                          portability issues.
#
# [usage]
#
#     # default usage
#     sh build-cran-package.sh
#
#     # custom R build
#     sh build-cran-package.sh --r-executable=RDvalgrind
#
#     # skip vignette building
#     sh build-cran-package.sh --no-build-vignettes

set -e -u

# Default values of arguments
BUILD_VIGNETTES=true
LGB_R_EXECUTABLE=R

while [ $# -gt 0 ]; do
  case "$1" in
    --r-executable=*)
      LGB_R_EXECUTABLE="${1#*=}"
      ;;
    --no-build-vignettes*)
      BUILD_VIGNETTES=false
      ;;
    *)
      echo "invalid argument '${1}'"
      exit 1
      ;;
  esac
  shift
done

echo "Building lightgbm with R executable: ${LGB_R_EXECUTABLE}"

ORIG_WD="$(pwd)"
TEMP_R_DIR="$(pwd)/lightgbm_r"

if test -d "${TEMP_R_DIR}"; then
    rm -r "${TEMP_R_DIR}"
fi
mkdir -p "${TEMP_R_DIR}"

# R packages cannot have versions like 3.0.0rc1, but
# 3.0.0-1 is acceptable
LGB_VERSION=$(head -1 ./VERSION.txt | sed "s/rc/-/g")

# move relevant files
cp -R R-package/* "${TEMP_R_DIR}"
cp -R include "${TEMP_R_DIR}/src/"
cp -R src/* "${TEMP_R_DIR}/src/"

if ${BUILD_VIGNETTES} ; then
    cp docs/logo/LightGBM_logo_black_text.svg "${TEMP_R_DIR}/vignettes/"
fi

mkdir -p "${TEMP_R_DIR}/src/include/LightGBM/utils/fast_float"
cp \
    external_libs/fast_float/include/fast_float/*.h \
    "${TEMP_R_DIR}/src/include/LightGBM/utils/fast_float"

mkdir -p "${TEMP_R_DIR}/src/include/LightGBM/utils/fmt"
cp \
    external_libs/fmt/include/fmt/*.h \
    "${TEMP_R_DIR}/src/include/LightGBM/utils/fmt"

# including only specific files from Eigen, to keep the R-package
# small and avoid redistributing code with licenses incompatible with
# LightGBM's license
EIGEN_R_DIR="${TEMP_R_DIR}/src/include/Eigen"
mkdir -p "${EIGEN_R_DIR}"

modules="Cholesky Core Dense Eigenvalues Geometry Householder Jacobi LU QR SVD"
for eigen_module in ${modules}; do
    cp "external_libs/eigen/Eigen/${eigen_module}" "${EIGEN_R_DIR}/${eigen_module}"
    if [ "${eigen_module}" != "Dense" ]; then
        mkdir -p "${EIGEN_R_DIR}/src/${eigen_module}/"
        cp -R "external_libs/eigen/Eigen/src/${eigen_module}"/* "${EIGEN_R_DIR}/src/${eigen_module}/"
    fi
done

mkdir -p "${EIGEN_R_DIR}/src/misc"
cp -R external_libs/eigen/Eigen/src/misc/* "${EIGEN_R_DIR}/src/misc/"

mkdir -p "${EIGEN_R_DIR}/src/plugins"
cp -R external_libs/eigen/Eigen/src/plugins/* "${EIGEN_R_DIR}/src/plugins/"

cd "${TEMP_R_DIR}"

    # Remove files not needed for CRAN
    echo "Removing files not needed for CRAN"
    rm src/install.libs.R
    rm -r inst/
    rm -r pkgdown/
    rm cran-comments.md
    rm AUTOCONF_UBUNTU_VERSION
    rm recreate-configure.sh

    # files only used by the lightgbm CLI aren't needed for
    # the R-package
    rm src/application/application.cpp
    rm src/include/LightGBM/application.h
    rm src/main.cpp

    # Remove 'region', 'endregion', and 'warning' pragmas.
    # This won't change the correctness of the code. CRAN does
    # not allow you to use compiler flag '-Wno-unknown-pragmas' or
    # pragmas that suppress warnings.
    echo "Removing unknown pragmas in headers"
    find . \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) -exec \
      sed \
        -i.bak \
        -e 's/^.*#pragma clang diagnostic.*$//' \
        -e 's/^.*#pragma diag_suppress.*$//' \
        -e 's/^.*#pragma GCC diagnostic.*$//' \
        -e 's/^.*#pragma region.*$//' \
        -e 's/^.*#pragma endregion.*$//' \
        -e 's/^.*#pragma warning.*$//' \
        {} +

    # 'processx' is listed as a 'Suggests' dependency in DESCRIPTION
    # because it is used in install.libs.R, a file that is not
    # included in the CRAN distribution of the package
    sed \
        -i.bak \
        '/processx/d' \
        DESCRIPTION

    echo "Cleaning sed backup files"
    find . -name '*.bak' -exec rm {} \;

cd "${ORIG_WD}"

if ${BUILD_VIGNETTES} ; then
    "${LGB_R_EXECUTABLE}" CMD build \
        --keep-empty-dirs \
        lightgbm_r

    echo "removing object files created by vignettes"
    rm -rf ./_tmp
    mkdir _tmp
    TARBALL_NAME="lightgbm_${LGB_VERSION}.tar.gz"
    mv "${TARBALL_NAME}" _tmp/

    echo "untarring ${TARBALL_NAME}"
    cd _tmp
        tar -xf "${TARBALL_NAME}" > /dev/null 2>&1
        rm -f "${TARBALL_NAME}"
        echo "done untarring ${TARBALL_NAME}"

        # Object files are left behind from compiling the library to generate vignettes.
        # Approaches like using tar --exclude=*.so to exclude them are not portable
        # (for example, don't work with some versions of tar on Windows).
        #
        # Removing them manually here removes the need to use tar --exclude.
        #
        # For background, see https://github.com/lightgbm-org/LightGBM/pull/3946#pullrequestreview-799415812.
        rm -f ./lightgbm/src/*.o
        rm -f ./lightgbm/src/boosting/*.o
        rm -f ./lightgbm/src/io/*.o
        rm -f ./lightgbm/src/metric/*.o
        rm -f ./lightgbm/src/network/*.o
        rm -f ./lightgbm/src/objective/*.o
        rm -f ./lightgbm/src/treelearner/*.o
        rm -f ./lightgbm/src/utils/*.o

        echo "re-tarring ${TARBALL_NAME}"
        # --no-xattrs is the default in GNU tar but not some distributions of BSD tar.
        # Enable it here to avoid errors on macOS.
        # ref: https://stackoverflow.com/a/74373784/3986677
        tar \
            -cz \
            --no-xattrs \
            -f "${TARBALL_NAME}" \
            lightgbm \
        > /dev/null 2>&1
        mv "${TARBALL_NAME}" ../
    cd ..
    echo "Done creating ${TARBALL_NAME}"

    rm -rf ./_tmp
else
    "${LGB_R_EXECUTABLE}" CMD build \
        --keep-empty-dirs \
        --no-build-vignettes \
        lightgbm_r
fi

echo "Done building R-package"
