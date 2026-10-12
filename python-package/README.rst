LightGBM Python-package
=======================

|License| |Python Versions| |PyPI Version| |PyPI Downloads| |conda Version| |conda Downloads| |API Docs|

Installation
------------

Preparation
'''''''''''

32-bit Python is not supported.
Please install 64-bit version.
If you have a strong need to install with 32-bit Python, refer to `Build 32-bit Version with 32-bit Python section <#build-32-bit-version-with-32-bit-python>`__.

|

Install from `PyPI <https://pypi.org/project/lightgbm>`_
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: sh

    pip install lightgbm

Compiled library that is included in the wheel file supports both **GPU** (don't confuse with CUDA version) and **CPU** versions out of the box.
This feature is available only for **Windows** and **Linux** currently.
To use **GPU** version you only need to install OpenCL Runtime libraries.
For NVIDIA and AMD GPU they are included in the ordinary drivers for your graphics card, so no action is required.
If you would like your AMD or Intel CPU to act like a GPU (for testing and debugging),
you can install `AMD APP SDK <https://github.com/lightgbm-org/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe>`_ on **Windows** and `PoCL <https://portablecl.org>`_ on **Linux**.
Many modern Linux distributions provide packages for PoCL, look for ``pocl-opencl-icd`` on Debian-based distributions and ``pocl`` on RedHat-based distributions.

For **Windows** users, `VC runtime <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist>`_ is needed if **Visual Studio** is not installed.

For **macOS** users, the **OpenMP** library is needed.
You can install it by the following command: ``brew install libomp``.

|

Install Nightly Packages
''''''''''''''''''''''''

Python packages are built on each new commit to ``main`` and uploaded to https://anaconda.org/lightgbm-packages.

Only the latest development version is available there, and can be installed like this:

.. code:: sh

   pip install --no-deps --index-url https://pypi.anaconda.org/lightgbm-packages/simple lightgbm

|

Use LightGBM with PyArrow
*************************

To install all dependencies needed to use ``PyArrow`` in LightGBM, append ``[arrow]``.

.. code:: sh

    pip install 'lightgbm[arrow]'

|

Use LightGBM with Polars
*************************

To install all dependencies needed to use ``polars`` in LightGBM, append ``[polars]``.

.. code:: sh

    pip install 'lightgbm[polars]'

|

Use LightGBM with Dask
**********************

Warning: Dask-package is only tested on macOS and Linux.

To install all dependencies needed to use ``lightgbm.dask``, append ``[dask]``.

.. code:: sh

    pip install 'lightgbm[dask]'

|

Use LightGBM with pandas
************************

To install all dependencies needed to use ``pandas`` in LightGBM, append ``[pandas]``.

.. code:: sh

    pip install 'lightgbm[pandas]'

|

Use LightGBM Plotting Capabilities
**********************************

To install all dependencies needed to use ``lightgbm.plotting``, append ``[plotting]``.

.. code:: sh

    pip install 'lightgbm[plotting]'

|

Use LightGBM with scikit-learn
******************************

To install all dependencies needed to use ``lightgbm.sklearn``, append ``[scikit-learn]``.

.. code:: sh

    pip install 'lightgbm[scikit-learn]'

|

Build from Sources
******************

.. code:: sh

    pip install lightgbm --no-binary lightgbm

For **macOS** users, you can perform installation either with **Apple Clang** or **gcc**.

- In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in `Installation Guide <https://github.com/lightgbm-org/LightGBM/blob/main/docs/Installation-Guide.rst#apple-clang>`__) first.

- In case you prefer **gcc**, you need to install it (details for installation can be found in `Installation Guide <https://github.com/lightgbm-org/LightGBM/blob/main/docs/Installation-Guide.rst#gcc-1>`__) and specify compilers by running ``export CXX=g++-7 CC=gcc-7`` (replace "7" with version of **gcc** installed on your machine) first.

For **Windows** users, **Visual Studio** (or `VS Build Tools <https://visualstudio.microsoft.com/downloads/>`_) is needed.

|

Build Threadless Version
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_OPENMP=OFF

All requirements, except the **OpenMP** requirement, from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

It is **strongly not recommended** to use this version of LightGBM!

|

Build MPI Version
~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_MPI=ON

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

For **Windows** users, compilation with **MinGW-w64** is not supported.

**MPI** libraries are needed: details for installation can be found in `Installation Guide <https://github.com/lightgbm-org/LightGBM/blob/main/docs/Installation-Guide.rst#build-mpi-version>`__.

|

Build GPU Version
~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

For **macOS** users, the GPU version is not supported.

**Boost** and **OpenCL** are needed: details for installation can be found in `Installation Guide <https://github.com/lightgbm-org/LightGBM/blob/main/docs/Installation-Guide.rst#build-gpu-version>`__.
Almost always you also need to pass ``OpenCL_INCLUDE_DIR``, ``OpenCL_LIBRARY`` options for **Linux** and ``BOOST_ROOT``, ``BOOST_LIBRARYDIR`` options for **Windows** to **CMake** via ``pip`` options, like

.. code:: sh

    pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON --config-settings=cmake.define.OpenCL_INCLUDE_DIR="/usr/local/cuda/include/" --config-settings=cmake.define.OpenCL_LIBRARY="/usr/local/cuda/lib64/libOpenCL.so"

All available options that can be passed via ``cmake.define.{option}``.

- BOOST_ROOT

- Boost_DIR

- Boost_INCLUDE_DIR

- BOOST_LIBRARYDIR

- OpenCL_INCLUDE_DIR

- OpenCL_LIBRARY

For more details see `FindBoost <https://cmake.org/cmake/help/latest/module/FindBoost.html>`__ and `FindOpenCL <https://cmake.org/cmake/help/latest/module/FindOpenCL.html>`__.

Don't confuse with `CUDA version <#build-cuda-version>`__.
To use the GPU version within Python, pass ``{"device": "gpu"}`` respectively in parameters.

|

Build CUDA Version
~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON


By default, the library will be built with support for a hard-coded list of GPU architectures
based on the detected CUDA Toolkit version.

To build the library with support for more architectures, set ``CMAKE_CUDA_ARCHITECTURES``.

.. code:: sh

    # example: all Blackwell arches, including DGX Spark
    pip install \
      --no-binary lightgbm \
      --config-settings=cmake.define.USE_CUDA=ON \
      --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES='100;120;121-real;121-virtual' \
      'lightgbm>=4.7.0'

    # example: just the local GPU
    pip install \
      --no-binary lightgbm \
      --config-settings=cmake.define.USE_CUDA=ON \
      --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES='native' \
      'lightgbm>=4.7.0'

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

For **macOS** and **Windows** users, the CUDA version is not supported.

**CUDA** library is needed: details for installation can be found in `Installation Guide <https://github.com/lightgbm-org/LightGBM/blob/main/docs/Installation-Guide.rst#build-cuda-version>`__.

Don't confuse with `GPU version <#build-gpu-version>`__.
To use the CUDA version within Python, pass ``{"device": "cuda"}`` respectively in parameters.

|

Build with MinGW-w64 on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.CMAKE_SH=CMAKE_SH-NOTFOUND --config-settings=cmake.args="-GMinGW Makefiles"

`MinGW-w64 <https://www.mingw-w64.org/>`_ should be installed first.

It is recommended to use **Visual Studio** for its better multithreading efficiency in **Windows** for many-core systems
(see `Question 4 <https://github.com/lightgbm-org/LightGBM/blob/main/docs/FAQ.rst#4-i-am-using-windows-should-i-use-visual-studio-or-mingw-for-compiling-lightgbm>`__
and `Question 8 <https://github.com/lightgbm-org/LightGBM/blob/main/docs/FAQ.rst#8-cpu-usage-is-low-like-10-in-windows-when-using-lightgbm-on-very-large-datasets-with-many-core-systems>`__).

|

Build 32-bit Version with 32-bit Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --no-binary lightgbm --config-settings=cmake.args="-AWin32"

For **Windows** users, compilation with **MinGW-w64** is not supported.

For **macOS** and **Linux** users, the 32-bit version is not supported.

It is **strongly not recommended** to use this version of LightGBM!

|

Build with Time Costs Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_TIMETAG=ON

Use this option to make LightGBM output time costs for different internal routines, to investigate and benchmark its performance.

|

Install from `conda-forge channel <https://anaconda.org/conda-forge/lightgbm>`_
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

``lightgbm`` conda packages are available from the ``conda-forge`` channel.

.. code:: sh

    conda install -c conda-forge lightgbm

These packages support **CPU**, **GPU** and **CUDA** versions out of the box.

**GPU**-enabled version is available only for **Windows** and **Linux** currently.

**CUDA**-enabled version (since ``lightgbm>=4.4.0``) is available only for **Linux** currently and will be automatically selected if you are on a system where CUDA is installed.

|

Install from GitHub
'''''''''''''''''''

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

.. code:: sh

    git clone --recursive https://github.com/lightgbm-org/LightGBM.git
    cd LightGBM
    # export CXX=g++-14 CC=gcc-14  # macOS users, if you decided to compile with gcc, don't forget to specify compilers
    sh ./build-python.sh install

Note: ``sudo`` (or administrator rights in **Windows**) may be needed to perform the command.
Run ``sh ./build-python.sh install --user`` to install into user-specific instead of global site-packages directory.

Run ``sh ./build-python.sh install --no-isolation`` to assume all build and install dependencies are already installed, don't go to the internet to get them.

|

Run ``sh ./build-python.sh install --nomp`` to disable **OpenMP** support.
All requirements from `Build Threadless Version section <#build-threadless-version>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --mpi`` to enable **MPI** support.
All requirements from `Build MPI Version section <#build-mpi-version>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --gpu`` to enable GPU support.
All requirements from `Build GPU Version section <#build-gpu-version>`__ apply for this installation option as well.
To pass additional options to **CMake** use the following syntax: ``sh ./build-python.sh install --gpu --opencl-include-dir="/usr/local/cuda/include/"``,
see `Build GPU Version section <#build-gpu-version>`__ for the complete list of them.

Run ``sh ./build-python.sh install --cuda`` to enable CUDA support.
All requirements from `Build CUDA Version section <#build-cuda-version>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --mingw``, if you want to use **MinGW-w64** on **Windows** instead of **Visual Studio**.
All requirements from `Build with MinGW-w64 on Windows section <#build-with-mingw-w64-on-windows>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --bit32``, if you want to use 32-bit version.
All requirements from `Build 32-bit Version with 32-bit Python section <#build-32-bit-version-with-32-bit-python>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --time-costs``, if you want to output time costs for different internal routines.
All requirements from `Build with Time Costs Output section <#build-with-time-costs-output>`__ apply for this installation option as well.

|

If you get any errors during installation or due to any other reasons,
you may want to build dynamic library from sources by any method you prefer
(see `Installation Guide <https://github.com/lightgbm-org/LightGBM/blob/main/docs/Installation-Guide.rst>`__).
For example, you can use ``MSBuild`` tool and `solution file <https://github.com/lightgbm-org/LightGBM/blob/main/windows/LightGBM.sln>`__ from the repo.

.. code:: sh

  MSBuild.exe windows/LightGBM.sln /p:Configuration=DLL /p:Platform=x64 /p:PlatformToolset=v143

After compiling dynamic library just run ``sh ./build-python.sh install --precompile`` to install the Python-package using that library.

|

Build Wheel File
****************

You can run ``sh ./build-python.sh bdist_wheel`` to build a wheel file but not install it.

That script requires some dependencies like ``build``, ``scikit-build-core``, and ``wheel``.
In environments with restricted or no internet access, install those tools and then pass ``--no-isolation``.

.. code:: sh

  sh ./build-python.sh bdist_wheel --no-isolation

Troubleshooting
---------------

Refer to `FAQ <https://github.com/lightgbm-org/LightGBM/tree/main/docs/FAQ.rst>`_.

Examples
--------

Refer to the walk through examples in `Python guide folder <https://github.com/lightgbm-org/LightGBM/tree/main/examples/python-guide>`_.

Supported Python Versions
-------------------------

This project supports all Python versions until they reach end-of-life.
For details on the support calendar for Python versions, see https://devguide.python.org/versions/.

Development Guide
-----------------

To check that a contribution to the package matches its style expectations, run the following from the root of the repo.

.. code:: sh

    pre-commit run --all-files

To run the tests locally and compute test coverage, install the Python package using one of the options mentioned above.
Then run the following from the root of the repo.

.. code:: sh

    pytest \
        --cov=lightgbm \
        --cov-report="term" \
        --cov-report="html:htmlcov" \
        tests/python_package_test/

Then open `htmlcov/index.html` to view a clickable coverage report.

.. |License| image:: https://img.shields.io/github/license/lightgbm-org/lightgbm.svg
   :target: https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/lightgbm.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/lightgbm
.. |PyPI Version| image:: https://img.shields.io/pypi/v/lightgbm.svg?logo=pypi&logoColor=white
   :target: https://pypi.org/project/lightgbm
.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/lightgbm?logo=pypi&logoColor=white&label=pypi%20downloads
   :target: https://pepy.tech/project/lightgbm
.. |conda Version| image:: https://img.shields.io/conda/vn/conda-forge/lightgbm?logo=conda-forge&logoColor=white&label=conda
   :target: https://anaconda.org/conda-forge/lightgbm
.. |conda Downloads| image:: https://img.shields.io/conda/d/conda-forge/lightgbm?logo=conda-forge&logoColor=white&label=conda%20downloads
   :target: https://anaconda.org/conda-forge/lightgbm/files
.. |API Docs| image:: https://readthedocs.org/projects/lightgbm/badge/?version=latest
   :target: https://lightgbm.readthedocs.io/en/latest/Python-API.html
