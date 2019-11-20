#!/bin/bash

# set up R environment
R_LIB_PATH=~/Rlib
mkdir -p $R_LIB_PATH
echo "R_LIBS=$R_LIB_PATH" > ${HOME}/.Renviron
echo 'options(repos = "https://cran.rstudio.com")' > ${HOME}/.Rprofile
export PATH="$R_LIB_PATH/R/bin:$PATH"

# installing precompiled R for Ubuntu
# https://cran.r-project.org/bin/linux/ubuntu/#installation
# adding steps from https://stackoverflow.com/a/56378217/3986677 to get latest version
#
# This only needs to get run on Travis because R environment for Linux
# used by Azure pipelines is set up in https://github.com/guolinke/lightgbm-ci-docker
if [[ $TRAVIS == "true" ]] &&  [[ $OS_NAME == "linux" ]]; then
    sudo apt-get update
    sudo add-apt-repository \
        "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/"
    sudo apt-key adv \
        --keyserver keyserver.ubuntu.com \
        --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
    sudo apt-get update
    sudo apt-get install \
        --no-install-recommends \
        -y \
            r-base-dev=${R_TRAVIS_LINUX_VERSION} \
            texlive-latex-recommended \
            texlive-fonts-recommended \
            texlive-fonts-extra \
            qpdf \
            || exit -1
fi

# Installing R precompiled for Mac OS 10.11 or higher
if [[ $OS_NAME == "macos" ]]; then
    wget -q https://cran.r-project.org/bin/macosx/R-${R_MAC_VERSION}.pkg -O R.pkg
    sudo installer \
        -pkg $(pwd)/R.pkg \
        -target /

    # Fix "duplicate libomp versions" issue on Mac
    if [[ $AZURE == "true" ]]; then
        for LIBOMP_ALIAS in libgomp.dylib libiomp5.dylib libomp.dylib; do
            sudo ln -sf \
                "$(brew --cellar libomp)"/*/lib/libomp.dylib \
                $CONDA_PREFIX/lib/$LIBOMP_ALIAS \
            || exit -1;
        done
    fi
fi

conda install \
    -y \
    -q \
    --no-deps \
        pandoc

# Manually install Depends and Imports libraries + 'testthat'
# to avoid a CI-time dependency on devtools (for devtools::install_deps())
Rscript -e "install.packages(c('data.table', 'jsonlite', 'Matrix', 'R6', 'testthat'))" || exit -1

cd ${BUILD_DIRECTORY}
Rscript build_r.R

PKG_TARBALL=$(ls | grep '^lightgbm_.*\.tar\.gz$')
LOG_FILE_NAME="lightgbm.Rcheck/00check.log"

# suppress R CMD check warning from Suggests dependencies not being available
export _R_CHECK_FORCE_SUGGESTS_=0

# fails tests if either ERRORs or WARNINGs are thrown by
# R CMD CHECK
R CMD check ${PKG_TARBALL} \
    --as-cran \
    --no-manual \
|| exit -1

if grep -q -R "WARNING" "$LOG_FILE_NAME"; then
    echo "WARNINGS have been found by R CMD check!"
    exit -1
fi

exit 0
