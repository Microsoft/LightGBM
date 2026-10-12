#!/bin/bash

set -e -u

echo "checking that all OpenMP pragmas specify num_threads()"
get_omp_pragmas_without_num_threads() {
    git grep \
        -n \
        'pragma omp parallel' \
        -- '*.c' '*.cc' '*.cpp' '*.cu' '*.cuh' '*.h' '*.hpp' '*.tpp' \
    | grep -v ' num_threads'
}

# 'grep' returns a non-0 exit code if 0 lines were found.
# Turning off '-e -o pipefail' options here so that bash doesn't
# consider this a failure and stop execution of the script.
#
# ref: https://www.gnu.org/software/grep/manual/html_node/Exit-Status.html
set +e
PROBLEMATIC_LINES=$(
    get_omp_pragmas_without_num_threads
)
set -e
if test "${PROBLEMATIC_LINES}" != ""; then
    get_omp_pragmas_without_num_threads
    echo "Found '#pragma omp parallel' not using explicit num_threads() configuration. Fix those."
    echo "For details, see https://www.openmp.org/spec-html/5.0/openmpse14.html#x54-800002.6"
    exit 1
fi
echo "done checking OpenMP pragmas"
