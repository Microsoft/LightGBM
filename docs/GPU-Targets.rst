GPU SDK Correspondence and Device Targeting Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU Targets Table
=================

OpenCL is a universal massively parallel programming framework that targets multiple backends (GPU, CPU, FPGA, etc).
Basically, to use a device from a vendor, you have to install drivers from that specific vendor.
Intel's and AMD's OpenCL runtime also include x86 CPU target support.
NVIDIA's OpenCL runtime only supports NVIDIA GPU (no CPU support).
In general, OpenCL CPU backends are quite slow, and should be used for testing and debugging only.

You can find below a table of correspondence:

+---------------------------+-----------------+-----------------+-----------------+--------------+
| SDK                       | CPU Intel/AMD   | GPU Intel       | GPU AMD         | GPU NVIDIA   |
+===========================+=================+=================+=================+==============+
| `Intel SDK for OpenCL`_   | Supported       | Supported       | Not Supported   | Not Supported|
+---------------------------+-----------------+-----------------+-----------------+--------------+
| AMD APP SDK \*            | Supported       | Not Supported   | Supported       | Not Supported|
+---------------------------+-----------------+-----------------+-----------------+--------------+
| `PoCL`_                   | Supported       | Not Supported   | Supported       | Not Supported|
+---------------------------+-----------------+-----------------+-----------------+--------------+
| `NVIDIA CUDA Toolkit`_    | Not Supported   | Not Supported   | Not Supported   | Supported    |
+---------------------------+-----------------+-----------------+-----------------+--------------+

Legend:

\* AMD APP SDK is deprecated. On Windows, OpenCL is included in AMD graphics driver. On Linux, newer generation AMD cards are supported by the `ROCm`_ driver. You can download an archived copy of AMD APP SDK from our GitHub repo (`for Linux`_ and `for Windows`_).


--------------

Query OpenCL Devices in Your System
===================================

Your system might have multiple GPUs from different vendors ("platforms") installed. Setting up LightGBM GPU device requires two parameters: `OpenCL Platform ID <./Parameters.rst#gpu_platform_id>`__ (``gpu_platform_id``) and `OpenCL Device ID <./Parameters.rst#gpu_device_id>`__ (``gpu_device_id``). Generally speaking, each vendor provides an OpenCL platform, and devices from the same vendor have different device IDs under that platform. For example, if your system has an Intel integrated GPU and two discrete GPUs from AMD, you will have two OpenCL platforms (with ``gpu_platform_id=0`` and ``gpu_platform_id=1``). If the platform 0 is Intel, it has one device (``gpu_device_id=0``) representing the Intel GPU; if the platform 1 is AMD, it has two devices (``gpu_device_id=0``, ``gpu_device_id=1``) representing the two AMD GPUs. If you have a discrete GPU by AMD/NVIDIA and an integrated GPU by Intel, make sure to select the correct ``gpu_platform_id`` to use the discrete GPU as it usually provides better performance.

On Windows, OpenCL devices can be queried using `GPUCapsViewer`_, under the OpenCL tab. Note that the platform and device IDs reported by this utility start from 1. So you should minus the reported IDs by 1.

On Linux, OpenCL devices can be listed using the ``clinfo`` command. On Ubuntu, you can install ``clinfo`` by executing ``sudo apt-get install clinfo``.


Examples
===============

We provide test R code below, but you can use the language of your choice with the examples of your choices:

.. code:: r

    library(lightgbm)
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    train$data[, 1] <- 1:6513
    dtrain <- lgb.Dataset(train$data, label = train$label)
    data(agaricus.test, package = "lightgbm")
    test <- agaricus.test
    dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
    valids <- list(test = dtest)

    params <- list(objective = "regression",
                   metric = "rmse",
                   device = "gpu",
                   gpu_platform_id = 0,
                   gpu_device_id = 0,
                   nthread = 1,
                   boost_from_average = FALSE,
                   num_tree_per_iteration = 10,
                   max_bin = 32)
    model <- lgb.train(params,
                       dtrain,
                       2,
                       valids,
                       min_data = 1,
                       learning_rate = 1,
                       early_stopping_rounds = 10)

Make sure you list the OpenCL devices in your system and set ``gpu_platform_id`` and ``gpu_device_id`` correctly. In the following examples, our system has 1 GPU platform (``gpu_platform_id = 0``) from AMD APP SDK. The first device ``gpu_device_id = 0`` is a GPU device (AMD Oland), and the second device ``gpu_device_id = 1`` is the x86 CPU backend.

Example of using GPU (``gpu_platform_id = 0`` and ``gpu_device_id = 0`` in our system):

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 0,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using GPU Device: Oland, Vendor: Advanced Micro Devices, Inc.
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transferred to GPU in 0.004211 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and depth=8
    [1]:    test's rmse:1.10643e-17
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and depth=5
    [2]:    test's rmse:0

Running on OpenCL CPU backend devices is in generally slow, and we observe crashes on some Windows and macOS systems. Make sure you check the ``Using GPU Device`` line in the log and it is not using a CPU. The above log shows that we are using ``Oland`` GPU from AMD and not CPU.

Example of using CPU (``gpu_platform_id = 0``, ``gpu_device_id = 1``). The GPU device reported is ``Intel(R) Core(TM) i7-4600U CPU``, so it is using the CPU backend rather than a real GPU.

.. code:: r

    > params <- list(objective = "regression",
    +                metric = "rmse",
    +                device = "gpu",
    +                gpu_platform_id = 0,
    +                gpu_device_id = 1,
    +                nthread = 1,
    +                boost_from_average = FALSE,
    +                num_tree_per_iteration = 10,
    +                max_bin = 32)
    > model <- lgb.train(params,
    +                    dtrain,
    +                    2,
    +                    valids,
    +                    min_data = 1,
    +                    learning_rate = 1,
    +                    early_stopping_rounds = 10)
    [LightGBM] [Info] This is the GPU trainer!!
    [LightGBM] [Info] Total Bins 232
    [LightGBM] [Info] Number of data: 6513, number of used features: 116
    [LightGBM] [Info] Using requested OpenCL platform 0 device 1
    [LightGBM] [Info] Using GPU Device: Intel(R) Core(TM) i7-4600U CPU @ 2.10GHz, Vendor: GenuineIntel
    [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...
    [LightGBM] [Info] GPU programs have been built
    [LightGBM] [Info] Size of histogram bin entry: 12
    [LightGBM] [Info] 40 dense feature groups (0.12 MB) transferred to GPU in 0.004540 secs. 76 sparse feature groups.
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=16 and depth=8
    [1]:    test's rmse:1.10643e-17
    [LightGBM] [Info] No further splits with positive gain, best gain: -inf
    [LightGBM] [Info] Trained a tree with leaves=7 and depth=5
    [2]:    test's rmse:0


Known issues:

- Using a bad combination of ``gpu_platform_id`` and ``gpu_device_id`` can potentially lead to a **crash** due to OpenCL driver issues on some machines (you will lose your entire session content). Beware of it.

- On some systems, if you have integrated graphics card (Intel HD Graphics) and a dedicated graphics card (AMD, NVIDIA), the dedicated graphics card will automatically override the integrated graphics card. The workaround is to disable your dedicated graphics card to be able to use your integrated graphics card.

.. _Intel SDK for OpenCL: https://software.intel.com/en-us/articles/opencl-drivers

.. _ROCm: https://rocmdocs.amd.com/en/latest/

.. _for Linux: https://github.com/lightgbm-org/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2

.. _for Windows: https://github.com/lightgbm-org/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe

.. _NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

.. _clinfo: https://github.com/Oblomov/clinfo

.. _GPUCapsViewer: https://www.ozone3d.net/gpu_caps_viewer/

.. _PoCL: https://portablecl.org/
