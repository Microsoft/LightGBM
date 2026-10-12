#!/bin/bash

# [description]
#     Collect and download artifacts from all workflow runs for a commit.
#
# [usage]
#     ./download-artifacts.sh <COMMIT_ID>
#

set -e -u -E -o pipefail

COMMIT_ID="${1}"
OUTPUT_DIR="./release-artifacts"

get-latest-run-id() {
     gh run list                      \
        --repo "lightgbm-org/LightGBM"   \
        --commit "${1}"               \
        --workflow "${2}"             \
        --json 'createdAt,databaseId' \
        --jq 'sort_by(.createdAt) | reverse | .[0] | .databaseId'
}

# ensure directory for storing artifacts exists
echo "preparing to download artifacts for commit '${COMMIT_ID}' to '${OUTPUT_DIR}'"
mkdir -p "${OUTPUT_DIR}"

# get core artifacts
echo "downloading core artifacts"
gh run download \
    --repo "lightgbm-org/LightGBM" \
    --dir "${OUTPUT_DIR}" \
    "$(get-latest-run-id "${COMMIT_ID}" 'build.yml')"
echo "done downloading core artifacts"

# get python-package artifacts
echo "downloading python-package artifacts and NuGet package"
gh run download \
    --repo "lightgbm-org/LightGBM" \
    --dir "${OUTPUT_DIR}" \
    "$(get-latest-run-id "${COMMIT_ID}" 'python_package.yml')"
echo "done downloading python-package artifacts and NuGet package"

# get R-package artifacts
echo "downloading R-package artifacts"
gh run download \
    --repo "lightgbm-org/LightGBM" \
    --dir "${OUTPUT_DIR}" \
    "$(get-latest-run-id "${COMMIT_ID}" 'r_package.yml')"
echo "done downloading R-package artifacts"

# get SWIG artifacts
echo "downloading SWIG artifacts"
gh run download \
    --repo "lightgbm-org/LightGBM" \
    --dir "${OUTPUT_DIR}" \
    "$(get-latest-run-id "${COMMIT_ID}" 'swig.yml')"
echo "done downloading SWIG artifacts"

# 'gh run download' unpackages into nested directories like {artifact-name}/{file}.
#
# This moves all files to the top level and then deletes those {artifact-name}/ directories,
# to make it easier to bulk upload all files to a release.
echo "flattening directory structure"
find "${OUTPUT_DIR}" -type f -mindepth 2 -exec mv -i '{}' "${OUTPUT_DIR}" \;
find "${OUTPUT_DIR}" -type d -mindepth 1 -exec rm -r '{}' \+

echo "downloaded artifacts:"
find "${OUTPUT_DIR}" -type f
