# Using LightGBM via Docker

This directory contains `Dockerfile`s to make it easy to build and run LightGBM via [Docker](https://www.docker.com/).

These builds of LightGBM all train on the CPU. For GPU-enabled builds, see [the gpu/ directory](./gpu).

## Installing Docker

Follow the general installation instructions [on the Docker site](https://docs.docker.com/get-started/get-docker/):

* [macOS](https://docs.docker.com/desktop/setup/install/mac-install/)
* [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
* [Windows](https://docs.docker.com/desktop/setup/install/windows-install/)

## Using CLI Version of LightGBM via Docker

Build an image with the LightGBM CLI.

```shell
mkdir lightgbm-docker
cd lightgbm-docker
wget https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/docker/dockerfile-cli
docker build \
    -t lightgbm-cli \
    -f dockerfile-cli \
    .
```

Once that completes, the built image can be used to run the CLI in a container.
To try it out, run the following.

```shell
# configure the CLI
cat << EOF > train.conf
task = train
objective = binary
data = binary.train
num_trees = 10
output_model = LightGBM-CLI-model.txt
EOF

# get training data
curl -O https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/examples/binary_classification/binary.train

# train, and save model to a text file
docker run \
  --rm \
  --volume "${PWD}":/opt/training \
  --workdir /opt/training \
  lightgbm-cli \
  config=train.conf
```

After this runs, a LightGBM model can be found at `LightGBM-CLI-model.txt`.

For more details on how to configure and use the LightGBM CLI, see https://lightgbm.readthedocs.io/en/latest/Quick-Start.html.

## Running the Python-package Container

Build an image with the LightGBM Python-package installed.

```shell
mkdir lightgbm-docker
cd lightgbm-docker
wget https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/docker/dockerfile-python
docker build \
    -t lightgbm-python \
    -f dockerfile-python \
    .
```

Once that completes, the built image can be used to run LightGBM's Python-package in a container.
Run the following to produce a model using the Python-package.

```shell
# get training data
curl -O https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/examples/binary_classification/binary.train

# create training script
cat << EOF > train.py
import lightgbm as lgb
import numpy as np
params = {
    "objective": "binary",
    "num_trees": 10
}

bst = lgb.train(
    train_set=lgb.Dataset("binary.train"),
    params=params
)
bst.save_model("LightGBM-python-model.txt")
EOF

# run training in a container
docker run \
    --rm \
    --volume "${PWD}":/opt/training \
    --workdir /opt/training \
    lightgbm-python \
    python train.py
```

After this runs, a LightGBM model can be found at `LightGBM-python-model.txt`.

Or run an interactive Python session in a container.

```shell
docker run \
    --rm \
    --volume "${PWD}":/opt/training \
    --workdir /opt/training \
    -it lightgbm-python \
    python
```

## Running the R-package Container

Build an image with the LightGBM R-package installed.

```shell
mkdir lightgbm-docker
cd lightgbm-docker
wget https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/docker/dockerfile-r

docker build \
    -t lightgbm-r \
    -f dockerfile-r \
    .
```

Once that completes, the built image can be used to run LightGBM's R-package in a container.
Run the following to produce a model using the R-package.

```shell
# get training data
curl -O https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/examples/binary_classification/binary.train

# create training script
cat << EOF > train.R
library(lightgbm)
params <- list(
    objective = "binary"
    , num_trees = 10L
)

bst <- lgb.train(
    data = lgb.Dataset("binary.train"),
    params = params
)
lgb.save(bst, "LightGBM-R-model.txt")
EOF

# run training in a container
docker run \
    --rm \
    --volume "${PWD}":/opt/training \
    --workdir /opt/training \
    lightgbm-r \
    Rscript train.R
```

After this runs, a LightGBM model can be found at `LightGBM-R-model.txt`.

Run the following to get an interactive R session in a container.

```shell
docker run \
    --rm \
    -it lightgbm-r \
    R
```

To use [RStudio](https://posit.co/products/open-source/rstudio), an interactive development environment, run the following.

```shell
docker run \
    --rm \
    --env PASSWORD="lightgbm" \
    -p 8787:8787 \
    lightgbm-r
```

Then navigate to `localhost:8787` in your local web browser, and log in with username `rstudio` and password `lightgbm`.

To target a different R version, pass any [valid rocker/verse tag](https://hub.docker.com/r/rocker/verse/tags) to `docker build`.

For example, to test LightGBM with R 4.5:

```shell
docker build \
    -t lightgbm-r-45 \
    -f dockerfile-r \
    --build-arg R_VERSION=4.5 \
    .
```
