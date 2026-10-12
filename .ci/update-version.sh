#!/bin/sh
#
# [description]
#     Update files in source control based on the content of 'VERSION.txt'.
#
# [usage]
#
#     update-version.sh

set -e -u

LGB_VERSION=$(head -1 ./VERSION.txt)
LGB_VERSION_NO_RC=$(echo "${LGB_VERSION}" | sed 's/rc/-/g')

# in-place 'sed' that's compatible with GNU sed and BSD sed (the one bundled with macOS)
update_file() {
    TARGET_FILE=$1
    SED_EXPRESSION=$2
    UPDATED_CONTENTS=$(sed "${SED_EXPRESSION}" "${TARGET_FILE}")
    printf '%s\n' "${UPDATED_CONTENTS}" > "${TARGET_FILE}"
}

update_file \
    ./.appveyor.yml \
    "s|^version: .*$|version: ${LGB_VERSION}.{build}|"

update_file \
    ./python-package/pyproject.toml \
    "s|^version = \"[0-9a-z.]+\"$|version = \"${LGB_VERSION}\"|"

# R packages cannot have versions like 3.0.0rc1, but 3.0.0-1 is acceptable
update_file \
    ./R-package/DESCRIPTION \
    "s|^Version: .*$|Version: ${LGB_VERSION_NO_RC}|"

update_file \
    ./R-package/configure.ac \
    "s|^AC_INIT.*$|AC_INIT([lightgbm], [${LGB_VERSION_NO_RC}], [], [lightgbm], [])|"
