#!/bin/sh

set -e -u

DIST_DIR=${1}

# defaults
METHOD=${METHOD:-""}
TASK=${TASK:-""}

echo "checking Python-package distributions in '${DIST_DIR}'"

pip install \
    -qq \
    check-wheel-contents \
    twine || exit 1

echo "twine check..."
twine check --strict "$(echo "${DIST_DIR}"/*)" || exit 1

if { test "${TASK}" = "bdist" || test "${METHOD}" = "wheel"; }; then
    echo "check-wheel-contents..."
    check-wheel-contents "$(echo "${DIST_DIR}"/*.whl)" || exit 1
fi

PY_MINOR_VER=$(python -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR_VER" -gt 7 ]; then
    echo "pydistcheck..."
    pip install 'pydistcheck>=0.9.1'
    if { test "${TASK}" = "cuda" || test "${METHOD}" = "wheel"; }; then
        pydistcheck \
            --inspect \
            --ignore 'compiled-objects-have-debug-symbols'\
            --ignore 'distro-too-large-compressed' \
            --max-allowed-size-uncompressed '550M' \
            --max-allowed-files 806 \
            "$(echo "${DIST_DIR}"/*)" || exit 1
    elif { test "$(uname -m)" = "aarch64"; }; then
        pydistcheck \
            --inspect \
            --ignore 'compiled-objects-have-debug-symbols' \
            --max-allowed-size-compressed '5M' \
            --max-allowed-size-uncompressed '15M' \
            --max-allowed-files 806 \
            "$(echo "${DIST_DIR}"/*)" || exit 1
    else
        pydistcheck \
            --inspect \
            --max-allowed-size-compressed '5M' \
            --max-allowed-size-uncompressed '15M' \
            --max-allowed-files 806 \
            "$(echo "${DIST_DIR}"/*)" || exit 1
    fi
else
    echo "skipping pydistcheck (does not support Python 3.${PY_MINOR_VER})"
fi

echo "done checking Python-package distributions"
