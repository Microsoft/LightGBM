#!/bin/bash
#
# [description]
#     Rerun specified workflow for given pull request.
#
# [usage]
#     rerun-workflow.sh <WORKFLOW_ID> <PR_BRANCH>
#
# WORKFLOW_ID: Identifier (config name of ID) of a workflow to be rerun.
#
# PR_BRANCH: Name of pull request's branch.

set -e -E -u -o pipefail

if [ -z "$GITHUB_ACTIONS" ]; then
  echo "Must be run inside GitHub Actions CI"
  exit 1
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 <WORKFLOW_ID> <PR_BRANCH>"
  exit 1
fi

workflow_id=$1
pr_branch=$2

# --branch for some GitHub GLI commands does not respect the difference between forks and branches
# on the main repo. While some parts of the GitHub API refer to the branch of a workflow
# as '{org}:{branch}' for branches from forks, others expect only '{branch}'.
#
# This expansion trims a leading '{org}:' from 'pr_branch' if one is present.
pr_branch_no_fork_prefix="${pr_branch/*:/}"

RUN_ID=$(
  gh run list                              \
    --repo 'lightgbm-org/LightGBM'            \
    --workflow "${workflow_id}"            \
    --event "pull_request"                 \
    --branch "${pr_branch_no_fork_prefix}" \
    --json 'createdAt,databaseId'          \
    --jq 'sort_by(.createdAt) | reverse | .[0] | .databaseId'
)

if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: failed to find a run of workflow '${workflow_id}' for branch '${pr_branch}'"
  exit 1
fi

echo "Re-running workflow '${workflow_id}' (run ID ${RUN_ID})"

gh run rerun                  \
  --repo 'lightgbm-org/LightGBM' \
  "${RUN_ID}"
