#!/bin/bash

# Check that lib_lightgbm has exactly the expected global symbols.
#
# [usage]
#
#    # lib_lightgbm.{dll,dylib,so}: exports exactly the symbols in .ci/expected-symbols.txt
#    .ci/check-abi.sh lib_lightgbm.*
#
#    # R package's shared library: no unexpected symbols (some std:: templates and other
#    # things may come through due to R's linking process which we can't fully control)
#    .ci/check-abi.sh --r-package lightgbm.so
#

set -e -u -o pipefail

R_PACKAGE="false"
LIBRARY_FILE=""
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --r-package)
            R_PACKAGE="true"
            ;;
        *)
            LIBRARY_FILE="${1}"
            ;;
    esac
    shift
done

echo ""
echo "checking exported symbols in '${LIBRARY_FILE}'"

# get a list of defined, externally-visible symbols, one name per line.
if [[ "$(uname -s)" == "Darwin" ]]; then
    # Mach-O prefixes C symbol names with '_'
    actual_symbols=$(
        nm -g -U "${LIBRARY_FILE}" \
        | awk '{print $NF}' \
        | sed 's/^_//' \
        | LC_ALL=C sort
    )
else
    actual_symbols=$(
        nm --dynamic --extern-only --defined-only "${LIBRARY_FILE}" \
        | awk '{print $NF}' \
        | LC_ALL=C sort
    )
fi

if [[ "${R_PACKAGE}" == "true" ]]; then
    # NOTES: 'R_init_lightgbm' being exported is checked by 'R CMD check'
    unexpected_symbols=$(echo "${actual_symbols}" | grep -E '^Arrow|LGBM_|^_Z[A-Z]+8LightGBM|json11' || true)
    if [[ -n "${unexpected_symbols}" ]]; then
        echo "ERROR: '${LIBRARY_FILE}' exports symbols that should be hidden:"
        echo "${unexpected_symbols}"
        exit 1
    fi
    echo "OK: '${LIBRARY_FILE}' does not export any unexpected symbols"
    exit 0
fi

EXPECTED_SYMBOLS_FILE="$(dirname "${BASH_SOURCE[0]}")/expected-symbols.txt"
expected_symbols=$(cat "${EXPECTED_SYMBOLS_FILE}")

if ! diff -u --label "expected" --label "actual" <(echo "${expected_symbols}") <(echo "${actual_symbols}"); then
    echo ""
    echo "ERROR: symbols exported from '${LIBRARY_FILE}' do not match the expected symbols (${EXPECTED_SYMBOLS_FILE})."
    echo "Lines starting with '-' are expected but missing, lines starting with '+' are exported but not expected."
    echo "If the change to the public API is intentional, update '${EXPECTED_SYMBOLS_FILE}'."
    exit 1
fi

echo "OK: '${LIBRARY_FILE}' exports exactly the $(echo "${expected_symbols}" | wc -l | tr -d ' ') expected symbol(s)"
echo ""
