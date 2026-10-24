#!/bin/bash

set -e -E -u -o pipefail

rm -f ./_FIRST_RUN.flag

export PATH="${CONDA}/bin:${PATH}"

curl \
    -sL \
    -o "${HOME}/miniforge.sh" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(arch).sh"

/bin/bash "${HOME}/miniforge.sh" -b -p "${CONDA}"

conda config --set always_yes yes --set changeps1 no
conda update -q -y conda

conda env create \
    --name docs-env \
    --file env.yml || exit 1

# shellcheck disable=SC1091
source activate docs-env
make clean html || exit 1

echo "Done building docs. Open docs/_build/html/index.html in a web browser to view them."
