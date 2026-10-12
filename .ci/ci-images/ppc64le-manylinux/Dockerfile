FROM quay.io/pypa/manylinux_2_28_ppc64le

# use 'bash' for RUN steps
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV CONDA=/root/miniforge
ENV PATH="${CONDA}/bin:${PATH}"

RUN <<EOF
# set up conda
curl \
    -sL \
    -o miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh"

sh miniforge.sh -b -p "${CONDA}"

rm ./miniforge.sh
conda clean --yes --all
EOF
