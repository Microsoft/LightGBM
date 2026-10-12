#!/bin/bash

set -e -E -u -o pipefail

# defaults
ARCH=$(uname -m)

# set up R environment
export CRAN_MIRROR="https://cran.rstudio.com"
export R_LIB_PATH=~/Rlib
mkdir -p $R_LIB_PATH
export R_LIBS=$R_LIB_PATH
export PATH="$R_LIB_PATH/R/bin:$PATH"

# reference for configuring 'R CMD check' via environment variables:
# https://cran.r-project.org/doc/manuals/r-devel/R-ints.html

# don't fail builds for long-running examples unless they're very long.
# See https://github.com/lightgbm-org/LightGBM/issues/4049#issuecomment-793412254.
export _R_CHECK_EXAMPLE_TIMING_THRESHOLD_=30

if [[ "${OS_NAME}" == "macos" ]] && [[ "${ARCH}" == "x86_64" ]]; then
    # suppress 'R CMD check --as-cran' notes like this:
    #
    #    * checking compilation flags used ... NOTE
    #    Compilation used the following non-portable flag(s):
    #    ‘-march=core2’ ‘-mssse3’
    #
    export _R_CHECK_COMPILATION_FLAGS_KNOWN_="-march=core2 -mssse3"
fi

# Get details needed for installing R components
R_MAJOR_VERSION="${R_VERSION%.*}"
if [[ "${R_MAJOR_VERSION}" == "4" ]]; then
    export R_MAC_VERSION=4.3.1
    export R_MAC_PKG_URL=${CRAN_MIRROR}/bin/macosx/big-sur-${ARCH}/base/R-${R_MAC_VERSION}-${ARCH}.pkg
    export R_LINUX_VERSION="4.3.1-1.2204.0"
    export R_APT_REPO="jammy-cran40/"
else
    echo "Unrecognized R version: ${R_VERSION}"
    exit 1
fi

# installing precompiled R for Ubuntu
# https://cran.r-project.org/bin/linux/ubuntu/#installation
# adding steps from https://stackoverflow.com/a/56378217/3986677 to get latest version
#
# `devscripts` is required for 'checkbashisms' (https://github.com/r-lib/actions/issues/111)
if [[ $OS_NAME == "linux" ]]; then
    mkdir -p ~/.gnupg
    echo "disable-ipv6" >> ~/.gnupg/dirmngr.conf
    sudo apt-key adv \
        --homedir ~/.gnupg \
        --keyserver keyserver.ubuntu.com \
        --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 || exit 1
    sudo add-apt-repository \
        "deb ${CRAN_MIRROR}/bin/linux/ubuntu ${R_APT_REPO}" || exit 1
    sudo apt-get update
    sudo apt-get install \
        --no-install-recommends \
        -y \
            devscripts \
            libuv1-dev \
            r-base-core=${R_LINUX_VERSION} \
            r-base-dev=${R_LINUX_VERSION} \
            texinfo \
            texlive-latex-extra \
            texlive-latex-recommended \
            texlive-fonts-recommended \
            texlive-fonts-extra \
            tidy \
            qpdf \
            || exit 1

    if [[ $R_BUILD_TYPE == "cran" ]]; then
        sudo apt-get install \
            --no-install-recommends \
            -y \
                "autoconf=$(cat R-package/AUTOCONF_UBUNTU_VERSION)" \
                automake \
                || exit 1
    fi
fi

# Installing R precompiled for Mac OS 10.11 or higher
if [[ $OS_NAME == "macos" ]]; then

    # install BasicTeX
    curl -sL "https://mirror.ctan.org/systems/mac/mactex/BasicTeX.pkg" -o BasicTeX.pkg || exit 1
    sudo installer \
        -pkg "$(pwd)/BasicTeX.pkg" \
        -target / || exit 1

    export PATH="/Library/TeX/texbin:$PATH"
    sudo tlmgr --verify-repo=none update --self || exit 1
    sudo tlmgr --verify-repo=none install inconsolata helvetic rsfs || exit 1

    if [[ "${ARCH}" == "arm64" ]]; then
        brew_packages+=(
            checkbashisms
            qpdf
        )
        if [[ $R_BUILD_TYPE == "cran" ]]; then
            brew_packages+=(automake)
        fi

        # install Homebrew packages
        brew update-reset --auto-update
        brew update --auto-update
        brew install "${brew_packages[@]}" || exit 1

        # install R
        curl -sL "${R_MAC_PKG_URL}" -o R.pkg || exit 1
        sudo installer \
            -pkg "$(pwd)/R.pkg" \
            -target / || exit 1
    else
        echo "setting up 'pixi' environment"
        # conda-forge's activation scripts rely on being able to reference shell
        # variables that may not be defined, which can result in errors like
        # 'AR: unbound variable' when run in bash with 'set -u'.
        set +u
        eval "$(pixi shell-hook --locked -e 'r-macos-intel')"
        set -u
        echo "done setting up 'pixi' environment"
    fi

    # install tidy v5.8.0
    # ref: https://groups.google.com/g/r-sig-mac/c/7u_ivEj4zhM
    TIDY_URL=https://github.com/htacg/tidy-html5/releases/download/5.8.0/tidy-5.8.0-macos-x86_64+arm64.pkg
    curl -sL ${TIDY_URL} -o tidy.pkg
    sudo installer \
        -pkg "$(pwd)/tidy.pkg" \
        -target /

    # ensure that this newer version of 'tidy' is used by 'R CMD check'
    # ref: https://cran.r-project.org/doc/manuals/R-exts.html#Checking-packages
    export R_TIDYCMD=/usr/local/bin/tidy
fi

# install dependencies, unless in an activated 'pixi' environment (which should already have them)
if [[ -z "${PIXI_IN_SHELL:-}" ]]; then
    # {Matrix} needs {lattice}, so this needs to run before manually installing {Matrix}.
    # This should be unnecessary on R >=4.4.0
    # ref: https://github.com/lightgbm-org/LightGBM/issues/6433
    Rscript --vanilla -e "install.packages('lattice', repos = '${CRAN_MIRROR}', lib = '${R_LIB_PATH}')"

    # manually install {Matrix}, as {Matrix}=1.7-0 raised its R floor all the way to R 4.4.0
    # ref: https://github.com/lightgbm-org/LightGBM/issues/6433
    Rscript --vanilla -e "install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-5.tar.gz', repos = NULL, lib = '${R_LIB_PATH}')"

    # Manually install dependencies to avoid a CI-time dependency on devtools (for devtools::install_deps())
    Rscript --vanilla ./.ci/install-r-deps.R --build --test --exclude=Matrix || exit 1
fi

cd "${BUILD_DIRECTORY}"
PKG_TARBALL="lightgbm_$(head -1 VERSION.txt).tar.gz"
BUILD_LOG_FILE="lightgbm.Rcheck/00install.out"
LOG_FILE_NAME="lightgbm.Rcheck/00check.log"
if [[ $R_BUILD_TYPE == "cmake" ]]; then
    Rscript build_r.R -j4 --skip-install || exit 1
elif [[ $R_BUILD_TYPE == "cran" ]]; then

    # on Linux, we recreate configure in CI to test if
    # a change in a PR has changed configure.ac
    if [[ $OS_NAME == "linux" ]]; then
        ./R-package/recreate-configure.sh

        num_files_changed=$(
            git diff --name-only | wc -l
        )
        if [[ ${num_files_changed} -gt 0 ]]; then
            echo "'configure' in the R-package has changed. Please recreate it and commit the changes."
            echo "Changed files:"
            git diff --compact-summary
            echo "See R-package/README.md for details on how to recreate this script."
            echo ""
            exit 1
        fi
    fi

    ./build-cran-package.sh || exit 1

    # Test CRAN source .tar.gz in a directory that is not this repo or below it.
    # When people install.packages('lightgbm'), they won't have the LightGBM
    # git repo around. This is to protect against the use of relative paths
    # like ../../CMakeLists.txt that would only work if you are in the repo
    R_CMD_CHECK_DIR="${HOME}/tmp-r-cmd-check/"
    mkdir -p "${R_CMD_CHECK_DIR}"
    mv "${PKG_TARBALL}" "${R_CMD_CHECK_DIR}"
    cd "${R_CMD_CHECK_DIR}"
fi

if [[ $PRODUCES_ARTIFACTS == "true" ]]; then
    cp "${PKG_TARBALL}" "${BUILD_ARTIFACTSTAGINGDIRECTORY}/lightgbm-${LGB_VER}-r-cran.tar.gz"
fi

declare -i allowed_notes=0
bash "${BUILD_DIRECTORY}/.ci/run-r-cmd-check.sh" \
    "${PKG_TARBALL}" \
    "${allowed_notes}"

# ensure 'grep --count' doesn't cause failures
set +e

used_correct_r_version=$(
    cat $LOG_FILE_NAME \
    | grep --count "using R version ${R_VERSION}"
)
if [[ $used_correct_r_version -ne 1 ]]; then
    echo "Unexpected R version was used. Expected '${R_VERSION}'."
    exit 1
fi

if [[ $R_BUILD_TYPE == "cmake" ]]; then
    passed_correct_r_version_to_cmake=$(
        cat $BUILD_LOG_FILE \
        | grep --count "R version passed into FindLibR.cmake: ${R_VERSION}"
    )
    if [[ $passed_correct_r_version_to_cmake -ne 1 ]]; then
        echo "Unexpected R version was passed into cmake. Expected '${R_VERSION}'."
        exit 1
    fi
fi

# this check makes sure that CI builds of the package actually use OpenMP
if [[ $OS_NAME == "macos" ]] && [[ $R_BUILD_TYPE == "cran" ]]; then
    omp_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E "checking whether OpenMP will work .*yes"
    )
elif [[ $R_BUILD_TYPE == "cmake" ]]; then
    omp_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E ".*Found OpenMP: TRUE.*"
    )
else
    omp_working=1
fi
if [[ $omp_working -ne 1 ]]; then
    echo "OpenMP was not found"
    exit 1
fi

# this check makes sure that CI builds of the package
# actually use MM_PREFETCH preprocessor definition
#
# _mm_prefetch will not work on arm64 architecture
# ref: https://github.com/lightgbm-org/LightGBM/issues/4124
if [[ $ARCH != "arm64" ]]; then
    if [[ $R_BUILD_TYPE == "cran" ]]; then
        mm_prefetch_working=$(
            cat $BUILD_LOG_FILE \
            | grep --count -E "checking whether MM_PREFETCH work.*yes"
        )
    else
        mm_prefetch_working=$(
            cat $BUILD_LOG_FILE \
            | grep --count -E ".*Performing Test MM_PREFETCH - Success"
        )
    fi
    if [[ $mm_prefetch_working -ne 1 ]]; then
        echo "MM_PREFETCH test was not passed"
        exit 1
    fi
fi

# this check makes sure that CI builds of the package
# actually use MM_MALLOC preprocessor definition
if [[ $R_BUILD_TYPE == "cran" ]]; then
    mm_malloc_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E "checking whether MM_MALLOC work.*yes"
    )
else
    mm_malloc_working=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E ".*Performing Test MM_MALLOC - Success"
    )
fi
if [[ $mm_malloc_working -ne 1 ]]; then
    echo "MM_MALLOC test was not passed"
    exit 1
fi

# this check makes sure that no "warning: unknown pragma ignored" logs
# reach the user leading them to believe that something went wrong
if [[ $R_BUILD_TYPE == "cran" ]]; then
    pragma_warning_present=$(
        cat $BUILD_LOG_FILE \
        | grep --count -E "warning: unknown pragma ignored"
    )
    if [[ $pragma_warning_present -ne 0 ]]; then
        echo "Unknown pragma warning is present, pragmas should have been removed before build"
        exit 1
    fi
fi
