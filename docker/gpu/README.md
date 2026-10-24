# Tiny Distroless Dockerfile for LightGBM GPU CLI-only Version

`dockerfile-cli-only-distroless.gpu` - A multi-stage build based on the `nvidia/cuda:*-devel-*` (build) and `distroless/cc-debian12` (production) images. LightGBM (CLI-only) can be utilized in GPU and CPU modes. The resulting image size is around 15 MB.

---

# Small Dockerfile for LightGBM GPU CLI-only Version

`dockerfile-cli-only.gpu` - A multi-stage build based on the `nvidia/cuda:*-devel-*` (build) and `nvidia/cuda:*-base-*` (runtime) images. LightGBM (CLI-only) can be utilized in GPU and CPU modes. The resulting image size is around 100 MB.

---

# Dockerfile for LightGBM GPU Version with Python

`dockerfile.gpu` - A docker file with LightGBM utilizing the NVIDIA Container Toolkit. The file is based on the `nvidia/cuda:*-devel-*` image.
LightGBM can be utilized in GPU and CPU modes and via Python.

## Contents

- LightGBM (cpu + gpu)
- Python + scikit-learn, notebooks, pandas, matplotlib

Running the container starts a Jupyter Notebook at `localhost:8888`.

## Requirements

Requires docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on host machine.

## Quickstart

### Build Docker Image

```sh
mkdir lightgbm-docker
cd lightgbm-docker
wget https://raw.githubusercontent.com/lightgbm-org/LightGBM/main/docker/gpu/dockerfile.gpu
docker build -f dockerfile.gpu -t lightgbm-gpu .
```

### Run Image

```sh
docker run --gpus all --rm -d --name lightgbm-gpu -p 8888:8888 -v /home:/home lightgbm-gpu
```

### Attach with Command Line Access (if required)

```sh
docker exec -it lightgbm-gpu bash
```

### Jupyter Notebook

Jupyter prints a URL with a login token when it starts. Get it from the container logs, then open it in a browser.

```sh
docker logs lightgbm-gpu
```
