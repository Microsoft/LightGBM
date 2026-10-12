#!/bin/bash
#
# [description]
#     Post a comment to a pull request.
#
# [usage]
#     append-comment.sh <PULL_REQUEST_ID> <BODY>
#
# PULL_REQUEST_ID: ID of PR to post the comment on.
#
# BODY: Text of the comment to be posted.

set -e -E -u -o pipefail

if [ -z "$GITHUB_ACTIONS" ]; then
  echo "Must be run inside GitHub Actions CI"
  exit 1
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 <PULL_REQUEST_ID> <BODY>"
  exit 1
fi

pr_id=$1
body=$2

body=${body/failure/failure ❌}
body=${body/error/failure ❌}
body=${body/cancelled/failure ❌}
body=${body/timed_out/failure ❌}
body=${body/success/success ✔️}
data=$(
  jq -n \
    --argjson body "\"$body\"" \
    '{"body": $body}'
)
curl -sL \
  --fail \
  -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -d "$data" \
  "${GITHUB_API_URL}/repos/lightgbm-org/LightGBM/issues/${pr_id}/comments"
