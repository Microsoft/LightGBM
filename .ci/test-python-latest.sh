#!/bin/bash

set -e -E -u -o pipefail

echo "installing lightgbm and its dependencies"
pip install \
    --prefer-binary \
    --upgrade \
    -r ./.ci/pip-envs/requirements-latest.txt \
    -r ./.ci/pip-envs/requirements-test.txt \
    dist/*.whl

echo "installed package versions:"
pip freeze

echo ""
echo "running tests"
pytest tests/c_api_test/
pytest tests/python_package_test/
