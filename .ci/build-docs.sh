#!/bin/bash

set -e -E -u -o pipefail

conda env create \
    -q \
    --name test-env \
    --file ./docs/env.yml \
|| exit 1

# shellcheck disable=SC1091
source activate test-env

make -C docs html || exit 1
