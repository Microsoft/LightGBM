Installation Guide
==================

Versioning
~~~~~~~~~~

LightGBM releases use a 3-part version number, with this format:

.. code::

   {major}.{minor}.{patch}

This version follows a scheme called Intended Effort Versioning ("Effver" for short).
Changes to a component of the version indicate how much effort it will likely take to update
code using a previous version.

* ``major`` = updating will require significant effort
* ``minor`` = some effort
* ``patch`` = no or very little effort

This means that **new minor versions can contain breaking changes**, but these are typically
small or limited to less-frequently-used parts of the project.

For more details on why LightGBM uses EffVer instead of other schemes like semantic versioning,
see https://jacobtomlinson.dev/effver/.

Nightly Packages
~~~~~~~~~~~~~~~~

When built from source on an unreleased commit, the package version takes the following form:

.. code::

   {major}.{minor}.{patch}.99

That ``.99`` is added to ensure that a version built from an unreleased commit is considered "newer"
than all previous releases, and "older" than all future releases.

.. _nightly-builds:

To download such artifacts, run the following from the root of this repository.

.. code:: sh

   bash .ci/download-artifacts.sh ${COMMIT_ID}

Where ``COMMIT_ID`` is the full commit SHA pointing to a commit on ``main``.
The artifacts can then be found in the ``release-artifacts/`` directory.

For the Python package, nightly packages are also available via installers like ``pip``.
See `the python-package documentation`_ for details.

General Installation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~

All instructions below are aimed at compiling the 64-bit version of LightGBM.
It is worth compiling the 32-bit version only in very rare special cases involving environmental limitations.
The 32-bit version is slow and untested, so use it at your own risk and don't forget to adjust some of the commands below when installing.

By default, instructions below will use **VS Build Tools** or **make** tool to compile the code.
It it possible to use `Ninja`_ tool instead of make on all platforms, but VS Build Tools cannot be replaced with Ninja.
You can add ``-G Ninja`` to CMake flags to use Ninja.

By default, instructions below will produce a shared library file and an executable file with command-line interface.
You can add ``-DBUILD_CLI=OFF`` to CMake flags to disable the executable compilation.

If you need to build a static library instead of a shared one, you can add ``-DBUILD_STATIC_LIB=ON`` to CMake flags.

By default, instructions below will place header files into system-wide folder.
You can add ``-DINSTALL_HEADERS=OFF`` to CMake flags to disable headers installation.

By default, on macOS, CMake is looking into Homebrew standard folders for finding dependencies (e.g. OpenMP).
You can add ``-DUSE_HOMEBREW_FALLBACK=OFF`` to CMake flags to disable this behaviour.

Users who want to perform benchmarking can make LightGBM output time costs for different internal routines by adding ``-DUSE_TIMETAG=ON`` to CMake flags.

It is possible to build LightGBM in debug mode.
In this mode all compiler optimizations are disabled and LightGBM performs more checks internally.
To enable debug mode you can add ``-DUSE_DEBUG=ON`` to CMake flags or choose ``Debug_*`` configuration (e.g. ``Debug_DLL``, ``Debug_mpi``) in Visual Studio depending on how you are building LightGBM.

.. _sanitizers:

In addition to the debug mode, LightGBM can be built with compiler sanitizers.
To enable them add ``-DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address;leak;undefined"`` to CMake flags.
These values refer to the following supported sanitizers:

- ``address`` - AddressSanitizer (ASan);
- ``leak`` - LeakSanitizer (LSan);
- ``undefined`` - UndefinedBehaviorSanitizer (UBSan);
- ``thread`` - ThreadSanitizer (TSan).

Please note, that ThreadSanitizer cannot be used together with other sanitizers.
For more info and additional sanitizers' parameters please refer to the `following docs`_.
It is very useful to build `C++ unit tests <#build-c-unit-tests>`__ with sanitizers.

.. contents:: **Contents**
    :depth: 1
    :local:
    :backlinks: none

Windows
~~~~~~~

On Windows, LightGBM can be built using

- **Visual Studio**;
- **CMake** and **VS Build Tools**;
- **CMake** and **MinGW**.

Visual Studio (or VS Build Tools)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With GUI
********

1. Install `Visual Studio`_.

2. Navigate to one of the releases at https://github.com/lightgbm-org/LightGBM/releases, download ``LightGBM-complete_source_code_zip.zip``, and unzip it.

3. Go to ``LightGBM-complete_source_code_zip/windows`` folder.

4. Open ``LightGBM.sln`` file with **Visual Studio**, choose ``Release`` configuration if you need executable file or ``DLL`` configuration if you need shared library and click ``Build`` -> ``Build Solution (Ctrl+Shift+B)``.

   If you have errors about **Platform Toolset**, go to ``Project`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the toolset installed on your machine.

   If you have errors about **Windows SDK Version**, go to ``Project`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the SDK installed on your machine.

The ``.exe`` file will be in ``LightGBM-complete_source_code_zip/windows/x64/Release`` folder.
The ``.dll`` file will be in ``LightGBM-complete_source_code_zip/windows/x64/DLL`` folder.

From Command Line
*****************

1. Install `Git for Windows`_, `CMake`_ and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** is already installed).

2. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -A x64
     cmake --build build --target ALL_BUILD --config Release

The ``.exe`` and ``.dll`` files will be in ``LightGBM/Release`` folder.

MinGW-w64
^^^^^^^^^

1. Install `Git for Windows`_, `CMake`_ and `MinGW-w64`_.

2. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -G "MinGW Makefiles"
     cmake --build build -j4

The ``.exe`` and ``.dll`` files will be in ``LightGBM/`` folder.

**Note**: You may need to run the ``cmake -B build -S . -G "MinGW Makefiles"`` one more time or add ``-DCMAKE_SH=CMAKE_SH-NOTFOUND`` to CMake flags if you encounter the ``sh.exe was found in your PATH`` error.

It is recommended that you use **Visual Studio** since it has better multithreading efficiency in **Windows** for many-core systems
(see `Question 4 <./FAQ.rst#i-am-using-windows-should-i-use-visual-studio-or-mingw-for-compiling-lightgbm>`__ and `Question 8 <./FAQ.rst#cpu-usage-is-low-like-10-in-windows-when-using-lightgbm-on-very-large-datasets-with-many-core-systems>`__).

Linux
~~~~~

On Linux, LightGBM can be built using

- **CMake** and **gcc**;
- **CMake** and **Clang**.

After compilation the executable and ``.so`` files will be in ``LightGBM/`` folder.

gcc
^^^

1. Install `CMake`_ and **gcc**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S .
     cmake --build build -j4

Clang
^^^^^

1. Install `CMake`_, **Clang** and **OpenMP**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=clang++-14 CC=clang-14  # replace "14" with version of Clang installed on your machine
     cmake -B build -S .
     cmake --build build -j4

macOS
~~~~~

On macOS, LightGBM can be installed using

- **Homebrew**;
- **MacPorts**;

or can be built using

- **CMake** and **Apple Clang**;
- **CMake** and **gcc**.

Install Using ``Homebrew``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: sh

  brew install lightgbm

Refer to https://formulae.brew.sh/formula/lightgbm for more details.

Install Using ``MacPorts``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: sh

  sudo port install LightGBM

Refer to https://ports.macports.org/port/LightGBM/ for more details.

**Note**: Port for LightGBM is not maintained by LightGBM's maintainers.

Build from GitHub
^^^^^^^^^^^^^^^^^

After compilation the executable and ``.dylib`` files will be in ``LightGBM/`` folder.

Apple Clang
***********

1. Install `CMake`_ and **OpenMP**:

   .. code:: sh

     brew install cmake libomp

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S .
     cmake --build build -j4

gcc
***

1. Install `CMake`_ and **gcc**:

   .. code:: sh

     brew install cmake gcc

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     cmake -B build -S .
     cmake --build build -j4

Docker
~~~~~~

Refer to `Docker folder <https://github.com/lightgbm-org/LightGBM/tree/main/docker>`__.

Build Threadless Version (not Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default build version of LightGBM is based on OpenMP.
You can build LightGBM without OpenMP support but it is **strongly not recommended**.

Windows
^^^^^^^

On Windows, a version of LightGBM without OpenMP support can be built using

- **Visual Studio**;
- **CMake** and **VS Build Tools**;
- **CMake** and **MinGW**.

Visual Studio (or VS Build Tools)
*********************************

With GUI
--------

1. Install `Visual Studio`_.

2. Navigate to one of the releases at https://github.com/lightgbm-org/LightGBM/releases, download ``LightGBM-complete_source_code_zip.zip``, and unzip it.

3. Go to ``LightGBM-complete_source_code_zip/windows`` folder.

4. Open ``LightGBM.sln`` file with **Visual Studio**, choose ``Release`` configuration if you need executable file or ``DLL`` configuration if you need shared library.

5. Go to ``Project`` -> ``Properties`` -> ``Configuration Properties`` -> ``C/C++`` -> ``Language`` and change the ``OpenMP Support`` property to ``No (/openmp-)``.

6. Get back to the project's main screen and click ``Build`` -> ``Build Solution (Ctrl+Shift+B)``.

   If you have errors about **Platform Toolset**, go to ``Project`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the toolset installed on your machine.

   If you have errors about **Windows SDK Version**, go to ``Project`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the SDK installed on your machine.

The ``.exe`` file will be in ``LightGBM-complete_source_code_zip/windows/x64/Release`` folder.
The ``.dll`` file will be in ``LightGBM-complete_source_code_zip/windows/x64/DLL`` folder.

From Command Line
-----------------

1. Install `Git for Windows`_, `CMake`_ and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** is already installed).

2. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -A x64 -DUSE_OPENMP=OFF
     cmake --build build --target ALL_BUILD --config Release

The ``.exe`` and ``.dll`` files will be in ``LightGBM/Release`` folder.

MinGW-w64
*********

1. Install `Git for Windows`_, `CMake`_ and `MinGW-w64`_.

2. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -G "MinGW Makefiles" -DUSE_OPENMP=OFF
     cmake --build build -j4

The ``.exe`` and ``.dll`` files will be in ``LightGBM/`` folder.

**Note**: You may need to run the ``cmake -B build -S . -G "MinGW Makefiles" -DUSE_OPENMP=OFF`` one more time or add ``-DCMAKE_SH=CMAKE_SH-NOTFOUND`` to CMake flags if you encounter the ``sh.exe was found in your PATH`` error.

Linux
^^^^^

On Linux, a version of LightGBM without OpenMP support can be built using

- **CMake** and **gcc**;
- **CMake** and **Clang**.

After compilation the executable and ``.so`` files will be in ``LightGBM/`` folder.

gcc
***

1. Install `CMake`_ and **gcc**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_OPENMP=OFF
     cmake --build build -j4

Clang
*****

1. Install `CMake`_ and **Clang**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=clang++-14 CC=clang-14  # replace "14" with version of Clang installed on your machine
     cmake -B build -S . -DUSE_OPENMP=OFF
     cmake --build build -j4

macOS
^^^^^

On macOS, a version of LightGBM without OpenMP support can be built using

- **CMake** and **Apple Clang**;
- **CMake** and **gcc**.

After compilation the executable and ``.dylib`` files will be in ``LightGBM/`` folder.

Apple Clang
***********

1. Install `CMake`_:

   .. code:: sh

     brew install cmake

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_OPENMP=OFF
     cmake --build build -j4

gcc
***

1. Install `CMake`_ and **gcc**:

   .. code:: sh

     brew install cmake gcc

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     cmake -B build -S . -DUSE_OPENMP=OFF
     cmake --build build -j4

Build MPI Version
~~~~~~~~~~~~~~~~~

The default build version of LightGBM is based on socket. LightGBM also supports MPI.
`MPI`_ is a high performance communication approach with `RDMA`_ support.

If you need to run a distributed learning application with high performance communication, you can build the LightGBM with MPI support.

Windows
^^^^^^^

On Windows, an MPI version of LightGBM can be built using

- **MS MPI** and **Visual Studio**;
- **MS MPI**, **CMake** and **VS Build Tools**.

**Note**: Building MPI version by **MinGW** is not supported due to the miss of MPI library in it.

With GUI
********

1. You need to install `MS MPI`_ first. Both ``msmpisdk.msi`` and ``msmpisetup.exe`` are needed.

2. Install `Visual Studio`_.

3. Navigate to one of the releases at https://github.com/lightgbm-org/LightGBM/releases, download ``LightGBM-complete_source_code_zip.zip``, and unzip it.

4. Go to ``LightGBM-complete_source_code_zip/windows`` folder.

5. Open ``LightGBM.sln`` file with **Visual Studio**, choose ``Release_mpi`` configuration and click ``Build`` -> ``Build Solution (Ctrl+Shift+B)``.

   If you have errors about **Platform Toolset**, go to ``Project`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the toolset installed on your machine.

   If you have errors about **Windows SDK Version**, go to ``Project`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the SDK installed on your machine.

The ``.exe`` file will be in ``LightGBM-complete_source_code_zip/windows/x64/Release_mpi`` folder.

From Command Line
*****************

1. You need to install `MS MPI`_ first. Both ``msmpisdk.msi`` and ``msmpisetup.exe`` are needed.

2. Install `Git for Windows`_, `CMake`_ and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** is already installed).

3. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -A x64 -DUSE_MPI=ON
     cmake --build build --target ALL_BUILD --config Release

The ``.exe`` and ``.dll`` files will be in ``LightGBM/Release`` folder.

Linux
^^^^^

On Linux, an MPI version of LightGBM can be built using

- **CMake**, **gcc** and **Open MPI**;
- **CMake**, **Clang** and **Open MPI**.

After compilation the executable and ``.so`` files will be in ``LightGBM/`` folder.

gcc
***

1. Install `CMake`_, **gcc** and `Open MPI`_.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_MPI=ON
     cmake --build build -j4

Clang
*****

1. Install `CMake`_, **Clang**, **OpenMP** and `Open MPI`_.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=clang++-14 CC=clang-14  # replace "14" with version of Clang installed on your machine
     cmake -B build -S . -DUSE_MPI=ON
     cmake --build build -j4

macOS
^^^^^

On macOS, an MPI version of LightGBM can be built using

- **CMake**, **Open MPI** and **Apple Clang**;
- **CMake**, **Open MPI** and **gcc**.

After compilation the executable and ``.dylib`` files will be in ``LightGBM/`` folder.

Apple Clang
***********

1. Install `CMake`_, **OpenMP** and `Open MPI`_:

   .. code:: sh

     brew install cmake libomp open-mpi

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_MPI=ON
     cmake --build build -j4

gcc
***

1. Install `CMake`_, `Open MPI`_ and  **gcc**:

   .. code:: sh

     brew install cmake open-mpi gcc

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     cmake -B build -S . -DUSE_MPI=ON
     cmake --build build -j4

Build GPU Version
~~~~~~~~~~~~~~~~~

Windows
^^^^^^^

On Windows, a GPU version of LightGBM (``device_type=gpu``) can be built using

- **OpenCL**, **Boost**, **CMake** and **VS Build Tools**;
- **OpenCL**, **Boost**, **CMake** and **MinGW**.

If you use **MinGW**, the build procedure is similar to the build on Linux.

Following procedure is for the **MSVC** (Microsoft Visual C++) build.

1. Install `Git for Windows`_, `CMake`_ and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** is installed).

2. Install **OpenCL** for Windows. The installation depends on the brand (NVIDIA, AMD, Intel) of your GPU card.

   - For running on Intel, get `Intel SDK for OpenCL`_.

   - For running on AMD, get AMD APP SDK.

   - For running on NVIDIA, get `CUDA Toolkit`_.

   Further reading and correspondence table: `GPU SDK Correspondence and Device Targeting Table <./GPU-Targets.rst>`__.

3. Install `Boost Binaries`_.

   **Note**: Match your Visual C++ version:

   Visual Studio 2017 -> ``msvc-14.1-64.exe``,

   Visual Studio 2019 -> ``msvc-14.2-64.exe``,

   Visual Studio 2022 -> ``msvc-14.3-64.exe``.

4. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -A x64 -DUSE_GPU=ON -DBOOST_ROOT=C:/local/boost_1_63_0 -DBOOST_LIBRARYDIR=C:/local/boost_1_63_0/lib64-msvc-14.3
     # if you have installed NVIDIA CUDA to a customized location, you should specify paths to OpenCL headers and library like the following:
     # cmake -B build -S . -A x64 -DUSE_GPU=ON -DBOOST_ROOT=C:/local/boost_1_63_0 -DBOOST_LIBRARYDIR=C:/local/boost_1_63_0/lib64-msvc-14.3 -DOpenCL_LIBRARY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64/OpenCL.lib" -DOpenCL_INCLUDE_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include"
     cmake --build build --target ALL_BUILD --config Release

   **Note**: ``C:/local/boost_1_63_0`` and ``C:/local/boost_1_63_0/lib64-msvc-14.3`` are locations of your **Boost** binaries (assuming you've downloaded 1.63.0 version for Visual Studio 2022).

The ``.exe`` and ``.dll`` files will be in ``LightGBM/Release`` folder.

Linux
^^^^^

On Linux, a GPU version of LightGBM (``device_type=gpu``) can be built using

- **CMake**, **OpenCL**, **Boost** and **gcc**;
- **CMake**, **OpenCL**, **Boost** and **Clang**.

**OpenCL** headers and libraries are usually provided by GPU manufacture.
The generic OpenCL ICD packages (for example, Debian packages ``ocl-icd-libopencl1``, ``ocl-icd-opencl-dev``, ``pocl-opencl-icd``) can also be used.

Required **Boost** libraries (Boost.Align, Boost.System, Boost.Filesystem, Boost.Chrono) should be provided by the following Debian packages: ``libboost-dev``, ``libboost-system-dev``, ``libboost-filesystem-dev``, ``libboost-chrono-dev``.

After compilation the executable and ``.so`` files will be in ``LightGBM/`` folder.

gcc
***

1. Install `CMake`_, **gcc**, **OpenCL** and **Boost**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_GPU=ON
     # if you have installed NVIDIA CUDA to a customized location, you should specify paths to OpenCL headers and library like the following:
     # cmake -B build -S . -DUSE_GPU=ON -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/
     cmake --build build -j4

Clang
*****

1. Install `CMake`_, **Clang**, **OpenMP**, **OpenCL** and **Boost**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=clang++-14 CC=clang-14  # replace "14" with version of Clang installed on your machine
     cmake -B build -S . -DUSE_GPU=ON
     # if you have installed NVIDIA CUDA to a customized location, you should specify paths to OpenCL headers and library like the following:
     # cmake -B build -S . -DUSE_GPU=ON -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/
     cmake --build build -j4

macOS
^^^^^

The GPU version is not supported on macOS.

Docker
^^^^^^

Refer to `GPU Docker folder <https://github.com/lightgbm-org/LightGBM/tree/main/docker/gpu>`__.

Build CUDA Version
~~~~~~~~~~~~~~~~~~

The `original GPU version <#build-gpu-version>`__ of LightGBM (``device_type=gpu``) is based on OpenCL, and only computes histograms on GPUs, with other parts of training in CPUs.

The CUDA-based version (``device_type=cuda``) is a separate implementation that runs significantly faster by putting all the training process on GPUs. It also supports multi-GPU, and multi-node multi-GPU training.
Use this version in Linux environments with an NVIDIA GPU with compute capability 6.0 or higher.

Windows
^^^^^^^

The CUDA version is not supported on Windows.
Use the `GPU version <#build-gpu-version>`__ (``device_type=gpu``) for GPU acceleration on Windows.

Linux
^^^^^

On Linux, a CUDA version of LightGBM can be built using

- **CMake**, **gcc** and **CUDA**;
- **CMake**, **Clang** and **CUDA**.

Please refer to `this detailed guide`_ for **CUDA** libraries installation.

After compilation the executable and ``.so`` files will be in ``LightGBM/`` folder.

.. note::

   By default, the library will be built with support for a hard-coded list of GPU architectures
   based on the detected CUDA Toolkit version.

   To build the library with support for more architectures, set ``CMAKE_CUDA_ARCHITECTURES``.

   .. code:: sh

      # example: all Blackwell arches, including DGX Spark
      cmake -B build -S . -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="100;120;121-real;121-virtual"

      # example: just the local GPU
      cmake -B build -S . -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="native"

gcc
***

1. Install `CMake`_, **gcc** and **CUDA**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_CUDA=ON
     cmake --build build -j4

Clang
*****

1. Install `CMake`_, **Clang**, **OpenMP** and **CUDA**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=clang++-14 CC=clang-14  # replace "14" with version of Clang installed on your machine
     cmake -B build -S . -DUSE_CUDA=ON
     cmake --build build -j4

macOS
^^^^^

The CUDA version is not supported on macOS.

Build ROCm Version
~~~~~~~~~~~~~~~~~~

The `original GPU version <#build-gpu-version>`__ of LightGBM (``device_type=gpu``) is based on OpenCL.

The ROCm-based version (``device_type=cuda``) is a separate implementation. Yes, the ROCm version reuses the ``device_type=cuda`` as a convenience for users.  Use this version in Linux environments with an AMD GPU.

Windows
^^^^^^^

The ROCm version is not supported on Windows.
Use the `GPU version <#build-gpu-version>`__ (``device_type=gpu``) for GPU acceleration on Windows.

Linux
^^^^^

On Linux, a ROCm version of LightGBM can be built using

- **CMake**, **gcc** and **ROCm**;
- **CMake**, **Clang** and **ROCm**.

Please refer to `the ROCm docs`_ for **ROCm** libraries installation.

After compilation the executable and ``.so`` files will be in ``LightGBM/`` folder.

gcc
***

1. Install `CMake`_, **gcc** and **ROCm**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_ROCM=ON
     cmake --build build -j4

Clang
*****

1. Install `CMake`_, **Clang**, **OpenMP** and **ROCm**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=clang++-14 CC=clang-14  # replace "14" with version of Clang installed on your machine
     cmake -B build -S . -DUSE_ROCM=ON
     cmake --build build -j4

macOS
^^^^^

The ROCm version is not supported on macOS.

Build Java Wrapper
~~~~~~~~~~~~~~~~~~

Using the following instructions you can generate a JAR file containing the LightGBM `C API <./Development-Guide.rst#c-api>`__ wrapped by **SWIG**.

After compilation the ``.jar`` file will be in ``LightGBM/build`` folder.

Windows
^^^^^^^

On Windows, a Java wrapper of LightGBM can be built using

- **Java**, **SWIG**, **CMake** and **VS Build Tools**;
- **Java**, **SWIG**, **CMake** and **MinGW**.

VS Build Tools
**************

1. Install `Git for Windows`_, `CMake`_ and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** is already installed).

2. Install `SWIG`_ and **Java** (also make sure that ``JAVA_HOME`` environment variable is set properly).

3. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -A x64 -DUSE_SWIG=ON
     cmake --build build --target ALL_BUILD --config Release

MinGW-w64
*********

1. Install `Git for Windows`_, `CMake`_ and `MinGW-w64`_.

2. Install `SWIG`_ and **Java** (also make sure that ``JAVA_HOME`` environment variable is set properly).

3. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -G "MinGW Makefiles" -DUSE_SWIG=ON
     cmake --build build -j4

**Note**: You may need to run the ``cmake -B build -S . -G "MinGW Makefiles" -DUSE_SWIG=ON`` one more time or add ``-DCMAKE_SH=CMAKE_SH-NOTFOUND`` to CMake flags if you encounter the ``sh.exe was found in your PATH`` error.

It is recommended to use **VS Build Tools (Visual Studio)** since it has better multithreading efficiency in **Windows** for many-core systems
(see `Question 4 <./FAQ.rst#i-am-using-windows-should-i-use-visual-studio-or-mingw-for-compiling-lightgbm>`__ and `Question 8 <./FAQ.rst#cpu-usage-is-low-like-10-in-windows-when-using-lightgbm-on-very-large-datasets-with-many-core-systems>`__).

Linux
^^^^^

On Linux, a Java wrapper of LightGBM can be built using

- **CMake**, **gcc**, **Java** and **SWIG**;
- **CMake**, **Clang**, **Java** and **SWIG**.

gcc
***

1. Install `CMake`_, **gcc**, `SWIG`_ and **Java** (also make sure that ``JAVA_HOME`` environment variable is set properly).

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_SWIG=ON
     cmake --build build -j4

Clang
*****

1. Install `CMake`_, **Clang**, **OpenMP**, `SWIG`_ and **Java** (also make sure that ``JAVA_HOME`` environment variable is set properly).

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=clang++-14 CC=clang-14  # replace "14" with version of Clang installed on your machine
     cmake -B build -S . -DUSE_SWIG=ON
     cmake --build build -j4

macOS
^^^^^

On macOS, a Java wrapper of LightGBM can be built using

- **CMake**, **Java**, **SWIG** and **Apple Clang**;
- **CMake**, **Java**, **SWIG** and **gcc**.

Apple Clang
***********

1. Install `CMake`_, **Java** (also make sure that ``JAVA_HOME`` environment variable is set properly), `SWIG`_ and **OpenMP**:

   .. code:: sh

     brew install cmake openjdk swig libomp
     export JAVA_HOME="$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home/"

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DUSE_SWIG=ON
     cmake --build build -j4

gcc
***

1. Install `CMake`_, **Java** (also make sure that ``JAVA_HOME`` environment variable is set properly), `SWIG`_ and **gcc**:

   .. code:: sh

     brew install cmake openjdk swig gcc
     export JAVA_HOME="$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home/"

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     cmake -B build -S . -DUSE_SWIG=ON
     cmake --build build -j4

Build Python-package
~~~~~~~~~~~~~~~~~~~~

Refer to `the python-package documentation`_.

Build R-package
~~~~~~~~~~~~~~~

Refer to `R-package folder <https://github.com/lightgbm-org/LightGBM/tree/main/R-package>`__.

Build C++ Unit Tests
~~~~~~~~~~~~~~~~~~~~

Windows
^^^^^^^

On Windows, C++ unit tests of LightGBM can be built using

- **CMake** and **VS Build Tools**;
- **CMake** and **MinGW**.

VS Build Tools
**************

1. Install `Git for Windows`_, `CMake`_ and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** is already installed).

2. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -A x64 -DBUILD_CPP_TEST=ON
     cmake --build build --target testlightgbm --config Debug

The ``.exe`` file will be in ``LightGBM/Debug`` folder.

MinGW-w64
*********

1. Install `Git for Windows`_, `CMake`_ and `MinGW-w64`_.

2. Run the following commands:

   .. code:: console

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -G "MinGW Makefiles" -DBUILD_CPP_TEST=ON
     cmake --build build --target testlightgbm -j4

The ``.exe`` file will be in ``LightGBM/`` folder.

**Note**: You may need to run the ``cmake -B build -S . -G "MinGW Makefiles" -DBUILD_CPP_TEST=ON`` one more time or add ``-DCMAKE_SH=CMAKE_SH-NOTFOUND`` to CMake flags if you encounter the ``sh.exe was found in your PATH`` error.

Linux
^^^^^

On Linux, a C++ unit tests of LightGBM can be built using

- **CMake** and **gcc**;
- **CMake** and **Clang**.

After compilation the executable file will be in ``LightGBM/`` folder.

gcc
***

1. Install `CMake`_ and **gcc**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DBUILD_CPP_TEST=ON
     cmake --build build --target testlightgbm -j4

Clang
*****

1. Install `CMake`_, **Clang** and **OpenMP**.

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=clang++-14 CC=clang-14  # replace "14" with version of Clang installed on your machine
     cmake -B build -S . -DBUILD_CPP_TEST=ON
     cmake --build build --target testlightgbm -j4

macOS
^^^^^

On macOS, a C++ unit tests of LightGBM can be built using

- **CMake** and **Apple Clang**;
- **CMake** and **gcc**.

After compilation the executable file will be in ``LightGBM/`` folder.

Apple Clang
***********

1. Install `CMake`_ and **OpenMP**:

   .. code:: sh

     brew install cmake libomp

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     cmake -B build -S . -DBUILD_CPP_TEST=ON
     cmake --build build --target testlightgbm -j4

gcc
***

1. Install `CMake`_ and **gcc**:

   .. code:: sh

     brew install cmake gcc

2. Run the following commands:

   .. code:: sh

     git clone --recursive https://github.com/lightgbm-org/LightGBM
     cd LightGBM
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     cmake -B build -S . -DBUILD_CPP_TEST=ON
     cmake --build build --target testlightgbm -j4

.. _Visual Studio: https://visualstudio.microsoft.com/downloads/

.. _Git for Windows: https://git-scm.com/download/win

.. _CMake: https://cmake.org/

.. _VS Build Tools: https://visualstudio.microsoft.com/downloads/

.. _MinGW-w64: https://www.mingw-w64.org/downloads/

.. _MPI: https://en.wikipedia.org/wiki/Message_Passing_Interface

.. _RDMA: https://en.wikipedia.org/wiki/Remote_direct_memory_access

.. _MS MPI: https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi-release-notes

.. _Open MPI: https://www.open-mpi.org/

.. _Intel SDK for OpenCL: https://software.intel.com/en-us/articles/opencl-drivers

.. _CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

.. _Boost Binaries: https://sourceforge.net/projects/boost/files/boost-binaries/

.. _SWIG: https://www.swig.org/download.html

.. _this detailed guide: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

.. _the python-package documentation: https://github.com/lightgbm-org/LightGBM/tree/main/python-package

.. _the ROCm docs: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

.. _following docs: https://github.com/google/sanitizers/wiki

.. _Ninja: https://ninja-build.org
