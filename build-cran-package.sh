#!/bin/sh

# [description]
#     Prepare a source distribution of the R package
#     to be submitted to CRAN.
#
# [usage]
#     sh build-cran-package.sh

set -e

ORIG_WD=$(pwd)
TEMP_R_DIR=$(pwd)/lightgbm_r

if test -d ${TEMP_R_DIR}; then
    rm -r ${TEMP_R_DIR}
fi
mkdir -p ${TEMP_R_DIR}

CURRENT_DATE=$(date +'%Y-%m-%d')

# R packages cannot have versions like 3.0.0rc1, but
# 3.0.0-1 is acceptable
LGB_VERSION=$(cat VERSION.txt | sed "s/rc/-/g")

# move relevant files
cp -R R-package/* ${TEMP_R_DIR}
cp docs/logo/LightGBM_logo_black_text.svg ${TEMP_R_DIR}/vignettes/
cp -R include ${TEMP_R_DIR}/src/
cp -R src/* ${TEMP_R_DIR}/src/

cp \
    external_libs/fast_double_parser/include/fast_double_parser.h \
    ${TEMP_R_DIR}/src/include/LightGBM

mkdir -p ${TEMP_R_DIR}/src/include/LightGBM/fmt
cp \
    external_libs/fmt/include/fmt/*.h \
    ${TEMP_R_DIR}/src/include/LightGBM/fmt/

# including only specific files from Eigen, to keep the R package
# small and avoid redistributing code with licenses incompatible with
# LightGBM's license
EIGEN_R_DIR=${TEMP_R_DIR}/src/include/Eigen
mkdir -p ${EIGEN_R_DIR}

modules="Cholesky Core Dense Eigenvalues Geometry Householder Jacobi LU QR SVD"
for eigen_module in ${modules}; do
    cp external_libs/eigen/Eigen/${eigen_module} ${EIGEN_R_DIR}/${eigen_module}
    if [ ${eigen_module} != "Dense" ]; then
        mkdir -p ${EIGEN_R_DIR}/src/${eigen_module}/
        cp -R external_libs/eigen/Eigen/src/${eigen_module}/* ${EIGEN_R_DIR}/src/${eigen_module}/
    fi
done

mkdir -p ${EIGEN_R_DIR}/src/misc
cp -R external_libs/eigen/Eigen/src/misc/* ${EIGEN_R_DIR}/src/misc/

mkdir -p ${EIGEN_R_DIR}/src/plugins
cp -R external_libs/eigen/Eigen/src/plugins/* ${EIGEN_R_DIR}/src/plugins/

cd ${TEMP_R_DIR}

    # Remove files not needed for CRAN
    echo "Removing files not needed for CRAN"
    rm src/install.libs.R
    rm -r inst/
    rm -r pkgdown/
    rm cran-comments.md
    rm AUTOCONF_UBUNTU_VERSION
    rm recreate-configure.sh

    # files only used by the lightgbm CLI aren't needed for
    # the R package
    rm src/application/application.cpp
    rm src/include/LightGBM/application.h
    rm src/main.cpp

    # configure.ac and DESCRIPTION have placeholders for version
    # and date so they don't have to be updated manually
    sed -i.bak -e "s/~~VERSION~~/${LGB_VERSION}/" configure.ac
    sed -i.bak -e "s/~~VERSION~~/${LGB_VERSION}/" DESCRIPTION
    sed -i.bak -e "s/~~DATE~~/${CURRENT_DATE}/" DESCRIPTION

    # Remove 'region', 'endregion', and 'warning' pragmas.
    # This won't change the correctness of the code. CRAN does
    # not allow you to use compiler flag '-Wno-unknown-pragmas' or
    # pragmas that suppress warnings.
    echo "Removing unknown pragmas in headers"
    for file in $(find . -name '*.h' -o -name '*.hpp' -o -name '*.cpp'); do
      sed \
        -i.bak \
        -e 's/^.*#pragma clang diagnostic.*$//' \
        -e 's/^.*#pragma diag_suppress.*$//' \
        -e 's/^.*#pragma GCC diagnostic.*$//' \
        -e 's/^.*#pragma region.*$//' \
        -e 's/^.*#pragma endregion.*$//' \
        -e 's/^.*#pragma warning.*$//' \
        "${file}"
    done
    find . -name '*.h.bak' -o -name '*.hpp.bak' -o -name '*.cpp.bak' -exec rm {} \;

    sed \
        -i.bak \
        -e 's/\.\..*fmt\/format\.h/LightGBM\/fmt\/format\.h/' \
        src/include/LightGBM/utils/common.h

    sed \
        -i.bak \
        -e 's/\.\..*fast_double_parser\.h/LightGBM\/fast_double_parser\.h/' \
        src/include/LightGBM/utils/common.h

    # When building an R package with 'configure', it seems
    # you're guaranteed to get a shared library called
    #  <packagename>.so/dll. The package source code expects
    # 'lib_lightgbm.so', not 'lightgbm.so', to comply with the way
    # this project has historically handled installation
    echo "Changing lib_lightgbm to lightgbm"
    for file in R/*.R; do
        sed \
            -i.bak \
            -e 's/lib_lightgbm/lightgbm/' \
            "${file}"
    done
    sed \
        -i.bak \
        -e 's/lib_lightgbm/lightgbm/' \
        NAMESPACE

    # 'processx' is listed as a 'Suggests' dependency in DESCRIPTION
    # because it is used in install.libs.R, a file that is not
    # included in the CRAN distribution of the package
    sed \
        -i.bak \
        '/processx/d' \
        DESCRIPTION

    echo "Cleaning sed backup files"
    rm R/*.R.bak
    rm NAMESPACE.bak

cd ${ORIG_WD}

R CMD build \
    --keep-empty-dirs \
    lightgbm_r

echo "removing object files created by vignettes"
rm -rf ./_tmp
mkdir _tmp
TARBALL_NAME=lightgbm_${LGB_VERSION}.tar.gz
mv ${TARBALL_NAME} _tmp/

cd _tmp
    tar -xvf ${TARBALL_NAME}
    rm -rf ${TARBALL_NAME}
cd ..

tar \
    -czv \
    -C ./_tmp \
    --exclude=*.a \
    --exclude=*.dll \
    --exclude=*.o \
    --exclude=*.so \
    --exclude=*.tar.gz \
    -f ${TARBALL_NAME} \
    lightgbm

rm -rf ./_tmp

echo "Done building R package"
