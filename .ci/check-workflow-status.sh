#!/bin/bash

# [description]
#
#   Look for the last run of a given GitHub Actions workflow on a given branch.
#   If there's never been one (as might be the case with optional workflows like valgrind),
#   exit with 0.
#
#   Otherwise, check the status of that latest run.
#   If it wasn't successful, exit with a non-0 exit code.
#
# [usage]
#
#     check-workflow-status.sh <BRANCH> <WORKFLOW_FILE>
#
# BRANCH: name of a branch involved in a pull request.
#
# WORKFLOW_FILE: filename (e.g. 'r_valgrind.yml') defining the GitHub Actions workflow.
#

set -e -u -o pipefail

BRANCH="${1}"
WORKFLOW_FILE="${2}"
PR_NUMBER="${3}"

# Limit how much data is pulled from the API and needs to be parsed locally.
OLDEST_ALLOWED_RUN_DATE=$(date --date='7 days ago' '+%F')

echo "Searching for latest run of '${WORKFLOW_FILE}' on branch '${BRANCH}' "

LATEST_RUN_ID=$(
    gh run list  \
        --repo 'lightgbm-org/LightGBM' \
        --event 'workflow_dispatch' \
        --created ">= ${OLDEST_ALLOWED_RUN_DATE}" \
        --workflow "${WORKFLOW_FILE}" \
        --json 'createdAt,databaseId,name' \
        --jq "sort_by(.createdAt) | reverse | map(select(.name | contains (\"pr=${PR_NUMBER}\"))) | .[0] | .databaseId"
)

if [[ "${LATEST_RUN_ID}" == "" ]]; then
    echo "No runs of '${WORKFLOW_FILE}' found on branch from pull request ${PR_NUMBER} (on or after ${OLDEST_ALLOWED_RUN_DATE})."
    exit 0
fi

echo "Checking status of workflow run '${LATEST_RUN_ID}'"
gh run view \
    --repo "lightgbm-org/LightGBM" \
    --exit-status \
    "${LATEST_RUN_ID}"
