#!/bin/bash

set -e -E -u -o pipefail

CONFIG_FILE=$(pwd)/docker/.hadolint.yaml

while IFS= read -r -d '' dockerfile; do
    echo ""
    echo "linting '${dockerfile}'"
    docker run \
        --rm \
        -v "${CONFIG_FILE}":/.config/hadolint.yaml \
        -i \
        hadolint/hadolint \
    < "${dockerfile}" || exit 1;
    echo "done linting '${dockerfile}'"
done < <(find ./docker -type f -name 'dockerfile*' -print0)
