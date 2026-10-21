#!/bin/bash

set -e -E -u -o pipefail

echo "creating a venv"
python -m venv .minimal-env

# shellcheck disable=SC1091
source .minimal-env/bin/activate

echo "installing lightgbm and its dependencies"
pip install \
    --prefer-binary \
    --upgrade \
    -r ./.ci/pip-envs/requirements-test.txt \
    dist/*.whl

echo "installed package versions:"
pip list

echo "confirming that library is importable"
python -c "import lightgbm"

echo ""
echo "running tests"
pytest \
    -ra \
    --cov=lightgbm \
    --cov-fail-under=30 \
    tests/python_package_test/test_basic.py
