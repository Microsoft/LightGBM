LightGBM GPU Tutorial
=====================

The purpose of this document is to give you a quick step-by-step tutorial on GPU training.

We will use the GPU instance on `Microsoft Azure cloud computing platform`_ for demonstration,
but you can use any machine with modern AMD or NVIDIA GPUs.

GPU Setup
---------

You need to launch a ``NV`` type instance on Azure (available in East US, North Central US, South Central US, West Europe and Southeast Asia zones)
and select Ubuntu 16.04 LTS as the operating system.

For testing, the smallest ``NV6`` type virtual machine is sufficient, which includes 1/2 M60 GPU, with 8 GB memory, 180 GB/s memory bandwidth and 4,825 GFLOPS peak computation power.
Don't use the ``NC`` type instance as the GPUs (K80) are based on an older architecture (Kepler).

First we need to install minimal NVIDIA drivers and OpenCL development environment:

::

    sudo apt-get update
    sudo apt-get install --no-install-recommends nvidia-375
    sudo apt-get install --no-install-recommends nvidia-opencl-icd-375 nvidia-opencl-dev opencl-headers

After installing the drivers you need to restart the server.

::

    sudo init 6

After about 30 seconds, the server should be up again.

If you are using an AMD GPU, you should download and install the `AMDGPU-Pro`_ driver and also install packages ``ocl-icd-libopencl1`` and ``ocl-icd-opencl-dev``.

Build LightGBM
--------------

Now install necessary building tools and dependencies:

::

    sudo apt-get install --no-install-recommends git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev

The ``NV6`` GPU instance has a 320 GB ultra-fast SSD mounted at ``/mnt``.
Let's use it as our workspace (skip this if you are using your own machine):

::

    sudo mkdir -p /mnt/workspace
    sudo chown $(whoami):$(whoami) /mnt/workspace
    cd /mnt/workspace

Now we are ready to checkout LightGBM and compile it with GPU support:

::

    git clone --recursive https://github.com/lightgbm-org/LightGBM
    cd LightGBM
    cmake -B build -S . -DUSE_GPU=1
    # if you have installed NVIDIA CUDA to a customized location, you should specify paths to OpenCL headers and library like the following:
    # cmake -B build -S . -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/
    cmake --build build -j$(nproc)

You will see two binaries are generated, ``lightgbm`` and ``lib_lightgbm.so``.

If you are building on macOS, you probably need to remove macro ``BOOST_COMPUTE_USE_OFFLINE_CACHE`` in ``src/treelearner/gpu_tree_learner.h`` to avoid a known crash bug in Boost.Compute.

Install Python Interface (optional)
-----------------------------------

If you want to use the Python interface of LightGBM, you can install it now (along with some necessary Python-package dependencies):

::

    sudo apt-get -y install python3-pip python3-venv
    sudo -H pip install numpy scipy scikit-learn -U
    sudo sh ./build-python.sh install --precompile

You need to set an additional parameter ``"device" : "gpu"`` (along with your other options like ``learning_rate``, ``num_leaves``, etc) to use GPU in Python.

You can read our `Python-package Examples`_ for more information on how to use the Python interface.

Dataset Preparation
-------------------

Using the following commands to prepare the Higgs dataset:

::

    git clone https://github.com/guolinke/boosting_tree_benchmarks.git
    cd boosting_tree_benchmarks/data
    wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    gunzip HIGGS.csv.gz
    python higgs2libsvm.py
    cd ../..
    ln -s boosting_tree_benchmarks/data/higgs.train
    ln -s boosting_tree_benchmarks/data/higgs.test

Now we create a configuration file for LightGBM by running the following commands (please copy the entire block and run it as a whole):

::

    cat > lightgbm_gpu.conf <<EOF
    max_bin = 63
    num_leaves = 255
    num_iterations = 50
    learning_rate = 0.1
    tree_learner = serial
    task = train
    is_training_metric = false
    min_data_in_leaf = 1
    min_sum_hessian_in_leaf = 100
    ndcg_eval_at = 1,3,5,10
    device = gpu
    gpu_platform_id = 0
    gpu_device_id = 0
    EOF
    echo "num_threads=$(nproc)" >> lightgbm_gpu.conf

GPU is enabled in the configuration file we just created by setting ``device=gpu``.
In this configuration we use the first GPU installed on the system (``gpu_platform_id=0`` and ``gpu_device_id=0``). If ``gpu_platform_id`` or ``gpu_device_id`` is not set, the default platform and GPU will be selected.
You might have multiple platforms (AMD/Intel/NVIDIA) or GPUs. You can use the `clinfo`_ utility to identify the GPUs on each platform. On Ubuntu, you can install ``clinfo`` by executing ``sudo apt-get install clinfo``. If you have a discrete GPU by AMD/NVIDIA and an integrated GPU by Intel, make sure to select the correct ``gpu_platform_id`` to use the discrete GPU.

Run Your First Learning Task on GPU
-----------------------------------

Now we are ready to start GPU training!

First we want to verify the GPU works correctly.
Run the following command to train on GPU, and take a note of the AUC after 50 iterations:

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train valid=higgs.test objective=binary metric=auc

Now train the same dataset on CPU using the following command. You should observe a similar AUC:

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train valid=higgs.test objective=binary metric=auc device=cpu

Now we can make a speed test on GPU without calculating AUC after each iteration.

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=binary metric=auc

Speed test on CPU:

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=binary metric=auc device=cpu

You should observe over three times speedup on this GPU.

The GPU acceleration can be used on other tasks/metrics (regression, multi-class classification, ranking, etc) as well.
For example, we can train the Higgs dataset on GPU as a regression task:

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=regression_l2 metric=l2

Also, you can compare the training speed with CPU:

::

    ./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=regression_l2 metric=l2 device=cpu

Further Reading
---------------

- `GPU Tuning Guide and Performance Comparison <./GPU-Performance.rst>`__

- `GPU SDK Correspondence and Device Targeting Table <./GPU-Targets.rst>`__

Reference
---------

Please kindly cite the following article in your publications if you find the GPU acceleration useful:

Huan Zhang, Si Si and Cho-Jui Hsieh. "`GPU Acceleration for Large-scale Tree Boosting`_." SysML Conference, 2018.

.. _Microsoft Azure cloud computing platform: https://azure.microsoft.com/

.. _AMDGPU-Pro: https://www.amd.com/en/support.html

.. _Python-package Examples: https://github.com/lightgbm-org/LightGBM/tree/main/examples/python-guide

.. _GPU Acceleration for Large-scale Tree Boosting: https://arxiv.org/abs/1706.08359

.. _clinfo: https://github.com/Oblomov/clinfo
