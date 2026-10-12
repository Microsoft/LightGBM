.. role:: raw-html(raw)
    :format: html

LightGBM FAQ
############

.. contents:: LightGBM Frequently Asked Questions
    :depth: 1
    :local:
    :backlinks: none

------

Please post questions, feature requests, and bug reports at https://github.com/lightgbm-org/LightGBM/issues.

This project is mostly maintained by volunteers, so please be patient.
If your request is time-sensitive or more than a month goes by without a response, please tag the maintainers below for help.

-  `@guolinke <https://github.com/guolinke>`__ **Guolin Ke**
-  `@shiyu1994 <https://github.com/shiyu1994>`__ **Yu Shi**
-  `@jameslamb <https://github.com/jameslamb>`__ **James Lamb**
-  `@jmoralez <https://github.com/jmoralez>`__ **José Morales**
-  `@borchero <https://github.com/borchero>`__ **Oliver Borchert**
-  `@mayer79 <https://github.com/mayer79>`__ **Michael Mayer**

--------------

General LightGBM Questions
==========================

.. contents::
    :local:
    :backlinks: none

1. Where do I find more details about LightGBM parameters?
----------------------------------------------------------

Take a look at `Parameters <./Parameters.rst>`__.

2. On datasets with millions of features, training does not start (or starts after a very long time).
-----------------------------------------------------------------------------------------------------

Use a smaller value for ``bin_construct_sample_cnt`` and a larger value for ``min_data``.

3. When running LightGBM on a large dataset, my computer runs out of RAM.
-------------------------------------------------------------------------

**Multiple Solutions**: set the ``histogram_pool_size`` parameter to the MB you want to use for LightGBM (histogram\_pool\_size + dataset size = approximately RAM used),
lower ``num_leaves`` or lower ``max_bin`` (see `lightgbm-org/LightGBM#562 <https://github.com/lightgbm-org/LightGBM/issues/562>`__).

4. I am using Windows. Should I use Visual Studio or MinGW for compiling LightGBM?
----------------------------------------------------------------------------------

Visual Studio `performs best for LightGBM <https://github.com/lightgbm-org/LightGBM/issues/542>`__.

5. When using LightGBM GPU, I cannot reproduce results over several runs.
-------------------------------------------------------------------------

This is normal and expected behaviour, but you may try to use ``gpu_use_dp = true`` for reproducibility
(see `lightgbm-org/LightGBM#560 <https://github.com/lightgbm-org/LightGBM/pull/560#issuecomment-304561654>`__).
You may also use the CPU version.

6. Bagging is not reproducible when changing the number of threads.
-------------------------------------------------------------------

:raw-html:`<strike>`
LightGBM bagging is multithreaded, so its output depends on the number of threads used.
There is `no workaround currently <https://github.com/lightgbm-org/LightGBM/issues/632>`__.
:raw-html:`</strike>`

Starting from `#2804 <https://github.com/lightgbm-org/LightGBM/pull/2804>`__ bagging result doesn't depend on the number of threads.
So this issue should be solved in the latest version.

7. I tried to use Random Forest mode, and LightGBM crashes!
-----------------------------------------------------------

This is expected behaviour for arbitrary parameters. To enable Random Forest,
you must use ``bagging_fraction`` and ``feature_fraction`` different from 1, along with a ``bagging_freq``.
`This thread <https://github.com/lightgbm-org/LightGBM/issues/691>`__ includes an example.

8. CPU usage is low (like 10%) in Windows when using LightGBM on very large datasets with many-core systems.
------------------------------------------------------------------------------------------------------------

Please use `Visual Studio <https://visualstudio.microsoft.com/downloads/>`__
as it may be `10x faster than MinGW <https://github.com/lightgbm-org/LightGBM/issues/749>`__ especially for very large trees.

9. When I'm trying to specify a categorical column with the ``categorical_feature`` parameter, I get the following sequence of warnings, but there are no negative values in the column.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. code-block:: console

   [LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
   [LightGBM] [Warning] There are no meaningful features, as all feature values are constant.

The column you're trying to pass via ``categorical_feature`` likely contains very large values.
Categorical features in LightGBM are limited by int32 range,
so you cannot pass values that are greater than ``Int32.MaxValue`` (2147483647) as categorical features (see `lightgbm-org/LightGBM#1359 <https://github.com/lightgbm-org/LightGBM/issues/1359>`__).
You should convert them to integers ranging from zero to the number of categories first.

10. LightGBM crashes randomly with the error like: ``Initializing libiomp5.dylib, but found libomp.dylib already initialized.``
-------------------------------------------------------------------------------------------------------------------------------

.. code-block:: console

   OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
   OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.

**Possible Cause**: This error means that you have multiple OpenMP libraries installed on your machine and they conflict with each other.
(File extensions in the error message may differ depending on the operating system).

If you are using Python distributed by Conda, then it is highly likely that the error is caused by the ``numpy`` package from Conda which includes the ``mkl`` package which in turn conflicts with the system-wide library.
In this case you can update the ``numpy`` package in Conda or replace the Conda's OpenMP library instance with system-wide one by creating a symlink to it in Conda environment folder ``$CONDA_PREFIX/lib``.

**Solution**: Assuming you are using macOS with Homebrew, the command which overwrites OpenMP library files in the current active Conda environment with symlinks to the system-wide library ones installed by Homebrew:

.. code-block:: bash

   ln -sf `ls -d "$(brew --cellar libomp)"/*/lib`/* $CONDA_PREFIX/lib

The described above fix worked fine before the release of OpenMP 8.0.0 version.
Starting from 8.0.0 version, Homebrew formula for OpenMP includes ``-DLIBOMP_INSTALL_ALIASES=OFF`` option which leads to that the fix doesn't work anymore.
However, you can create symlinks to library aliases manually:

.. code-block:: bash

   for LIBOMP_ALIAS in libgomp.dylib libiomp5.dylib libomp.dylib; do sudo ln -sf "$(brew --cellar libomp)"/*/lib/libomp.dylib $CONDA_PREFIX/lib/$LIBOMP_ALIAS; done

Another workaround would be removing MKL optimizations from Conda's packages completely:

.. code-block:: bash

    conda install nomkl

If this is not your case, then you should find conflicting OpenMP library installations on your own and leave only one of them.

11. LightGBM hangs when multithreading (OpenMP) and using forking in Linux at the same time.
--------------------------------------------------------------------------------------------

Use ``nthreads=1`` to disable multithreading of LightGBM. There is a bug with OpenMP which hangs forked sessions
with multithreading activated. A more expensive solution is to use new processes instead of using fork, however,
keep in mind it is creating new processes where you have to copy memory and load libraries (example: if you want to
fork 16 times your current process, then you will require to make 16 copies of your dataset in memory)
(see `lightgbm-org/LightGBM#1789 <https://github.com/lightgbm-org/LightGBM/issues/1789#issuecomment-433713383>`__).

An alternative, if multithreading is really necessary inside the forked sessions, would be to compile LightGBM with
Intel toolchain. Intel compilers are unaffected by this bug.

For C/C++ users, any OpenMP feature cannot be used before the fork happens. If an OpenMP feature is used before the
fork happens (example: using OpenMP for forking), OpenMP will hang inside the forked sessions. Use new processes instead
and copy memory as required by creating new processes instead of forking (or, use Intel compilers).

Cloud platform container services may cause LightGBM to hang, if they use Linux fork to run multiple containers on a
single instance. For example, LightGBM hangs in AWS Batch array jobs, which `use the ECS agent
<https://aws.amazon.com/batch/faqs/>`__ to manage multiple running jobs. Setting ``nthreads=1`` mitigates the issue.

12. Why is early stopping not enabled by default in LightGBM?
-------------------------------------------------------------

Early stopping involves choosing a validation set, a special type of holdout which is used to evaluate the current state of the model after each iteration to see if training can stop.

In ``LightGBM``, `we have decided to require that users specify this set directly <./Parameters.rst#valid>`_. Many options exist for splitting training data into training, test, and validation sets.

The appropriate splitting strategy depends on the task and domain of the data, information that a modeler has but which ``LightGBM`` as a general-purpose tool does not.

13. Does LightGBM support direct loading data from zero-based or one-based LibSVM format file?
----------------------------------------------------------------------------------------------

LightGBM supports loading data from zero-based LibSVM format file directly.

14. Why CMake cannot find the compiler when compiling LightGBM with MinGW?
--------------------------------------------------------------------------

.. code-block:: bash

    CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
    CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage

This is a known issue of CMake when using MinGW. The easiest solution is to run again your ``cmake`` command to bypass the one time stopper from CMake. Or you can upgrade your version of CMake to at least version 3.17.0.

See `lightgbm-org/LightGBM#3060 <https://github.com/lightgbm-org/LightGBM/issues/3060#issuecomment-626338538>`__ for more details.

15. Where can I find LightGBM's logo to use it in my presentation?
------------------------------------------------------------------

You can find LightGBM's logo in different file formats and resolutions `here <https://github.com/lightgbm-org/LightGBM/tree/main/docs/logo>`__.

16. LightGBM crashes randomly or operating system hangs during or after running LightGBM.
-----------------------------------------------------------------------------------------

**Possible Cause**: This behavior may indicate that you have multiple OpenMP libraries installed on your machine and they conflict with each other, similarly to the ``FAQ #10``.

If you are using any Python-package that depends on ``threadpoolctl``, you also may see the following warning in your logs in this case:

.. code-block:: console

    /root/miniconda/envs/test-env/lib/python3.8/site-packages/threadpoolctl.py:546: RuntimeWarning:
    Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at
    the same time. Both libraries are known to be incompatible and this
    can cause random crashes or deadlocks on Linux when loaded in the
    same Python program.
    Using threadpoolctl may cause crashes or deadlocks. For more
    information and possible workarounds, please see
        https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md

Detailed description of conflicts between multiple OpenMP instances is provided in the `following document <https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md>`__.

**Solution**: Assuming you are using LightGBM Python-package and conda as a package manager, we strongly recommend using ``conda-forge`` channel as the only source of all your Python package installations because it contains built-in patches to workaround OpenMP conflicts. Some other workarounds are listed `here <https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md>`__ under the "Workarounds for Intel OpenMP and LLVM OpenMP case" section.

If this is not your case, then you should find conflicting OpenMP library installations on your own and leave only one of them.

17. Loading LightGBM fails like: ``cannot allocate memory in static TLS block``
-------------------------------------------------------------------------------

When loading LightGBM, you may encounter errors like the following.

.. code-block:: console

   lib/libgomp.so.1: cannot allocate memory in static TLS block

This most commonly happens on aarch64 Linux systems.

``gcc``'s OpenMP library (``libgomp.so``) tries to allocate a small amount of static thread-local storage ("TLS")
when it's dynamically loaded.

That error can happen when the loader isn't able to find a large enough block of memory.

On aarch64 Linux, processes and loaded libraries share the same pool of static TLS,
which makes such failures more likely. See these discussions:

* https://bugzilla.redhat.com/show_bug.cgi?id=1722181#c6
* https://github.com/pytorch/pytorch/issues/2575

If you are experiencing this issue when using the ``lightgbm`` Python-package, try upgrading
to at least ``v4.6.0``.

For older versions of the Python-package, or for other LightGBM APIs, this issue can
often be avoided by loading ``libgomp.so.1``. That can be done directly by setting environment
variable ``LD_PRELOAD``, like this:

.. code-block:: console

    export LD_PRELOAD=/root/miniconda3/envs/test-env/lib/libgomp.so.1

It can also be done indirectly by changing the order that other libraries are loaded
into processes, which varies by programming language and application type.

For more details, see these discussions:

* https://github.com/lightgbm-org/LightGBM/pull/6654#issuecomment-2352014275
* https://github.com/lightgbm-org/LightGBM/issues/6509
* https://maskray.me/blog/2021-02-14-all-about-thread-local-storage
* https://bugzilla.redhat.com/show_bug.cgi?id=1722181#c6

------

R-package
=========

.. contents::
    :local:
    :backlinks: none

1. Any training command using LightGBM does not work after an error occurred during the training of a previous LightGBM model.
------------------------------------------------------------------------------------------------------------------------------

In older versions of the R-package (prior to ``v3.3.0``), this could happen occasionally and the solution was to run ``lgb.unloader(wipe = TRUE)`` to remove all LightGBM-related objects. Some conversation about this could be found in `lightgbm-org/LightGBM#698 <https://github.com/lightgbm-org/LightGBM/issues/698>`__.

That is no longer necessary as of ``v3.3.0``, and function ``lgb.unloader()`` has since been removed from the R-package.

2. I used ``setinfo()``, tried to print my ``lgb.Dataset``, and now the R console froze!
----------------------------------------------------------------------------------------

As of at least LightGBM v3.3.0, this issue has been resolved and printing a ``Dataset`` object does not cause the console to freeze.

In older versions, avoid printing the ``Dataset`` after calling ``setinfo()``.

As of LightGBM v4.0.0, ``setinfo()`` has been replaced by a new method, ``set_field()``.

3. ``error in data.table::data.table()...argument 2 is NULL``.
--------------------------------------------------------------

If you are experiencing this error when running ``lightgbm``, you may be facing the same issue reported in `#2715 <https://github.com/lightgbm-org/LightGBM/issues/2715>`_ and later in `#2989 <https://github.com/lightgbm-org/LightGBM/pull/2989#issuecomment-614374151>`_. We have seen that in some situations, using ``data.table`` 1.11.x results in this error. To get around this, you can upgrade your version of ``data.table`` to at least version 1.12.0.

4. ``package/dependency ‘Matrix’ is not available ...``
-------------------------------------------------------

In April 2024, ``Matrix==1.7-0`` was published to CRAN.
That version had a floor of ``R (>=4.4.0)``.
``{Matrix}`` is a hard runtime dependency of ``{lightgbm}``, so on any version of R older than ``4.4.0``, running ``install.packages("lightgbm")`` results in something like the following.

.. code-block:: text

    package ‘Matrix’ is not available for this version of R

To fix that without upgrading to R 4.4.0 or greater, manually install an older version of ``{Matrix}``.

.. code-block:: R

    install.packages('https://cran.r-project.org/src/contrib/Archive/Matrix/Matrix_1.6-5.tar.gz', repos = NULL)

------

Python-package
==============

.. contents::
    :local:
    :backlinks: none

1. ``Error: setup script specifies an absolute path`` when installing from GitHub using ``python setup.py install``.
--------------------------------------------------------------------------------------------------------------------

.. note::
    As of v4.0.0, ``lightgbm`` does not support directly invoking ``setup.py``.
    This answer refers only to versions of ``lightgbm`` prior to v4.0.0.

.. code-block:: console

   error: Error: setup script specifies an absolute path:
   /Users/Microsoft/LightGBM/python-package/lightgbm/../../lib_lightgbm.so
   setup() arguments must *always* be /-separated paths relative to the setup.py directory, *never* absolute paths.

This error should be solved in latest version.
If you still meet this error, try to remove ``lightgbm.egg-info`` folder in your Python-package and reinstall,
or check `this thread on stackoverflow <https://stackoverflow.com/questions/18085571/pip-install-error-setup-script-specifies-an-absolute-path>`__.

2. Error messages: ``Cannot ... before construct dataset``.
-----------------------------------------------------------

I see error messages like...

.. code-block:: console

   Cannot get/set label/weight/init_score/group/num_data/num_feature before construct dataset

but I've already constructed a dataset by some code like:

.. code-block:: python

    train = lightgbm.Dataset(X_train, y_train)

or error messages like

.. code-block:: console

    Cannot set predictor/reference/categorical feature after freed raw data, set free_raw_data=False when construct Dataset to avoid this.

**Solution**: Because LightGBM constructs bin mappers to build trees, and train and valid Datasets within one Booster share the same bin mappers,
categorical features and feature names etc., the Dataset objects are constructed when constructing a Booster.
If you set ``free_raw_data=True`` (default), the raw data (with Python data struct) will be freed.
So, if you want to:

-  get label (or weight/init\_score/group/data) before constructing a dataset, it's same as get ``self.label``;

-  set label (or weight/init\_score/group) before constructing a dataset, it's same as ``self.label=some_label_array``;

-  get num\_data (or num\_feature) before constructing a dataset, you can get data with ``self.data``.
   Then, if your data is ``numpy.ndarray``, use some code like ``self.data.shape``. But do not do this after subsetting the Dataset, because you'll get always ``None``;

-  set predictor (or reference/categorical feature) after constructing a dataset,
   you should set ``free_raw_data=False`` or init a Dataset object with the same raw data.

3. I encounter segmentation faults (segfaults) randomly after installing LightGBM from PyPI using ``pip install lightgbm``.
---------------------------------------------------------------------------------------------------------------------------

We are doing our best to provide universal wheels which have high running speed and are compatible with any hardware, OS, compiler, etc. at the same time.
However, sometimes it's just impossible to guarantee the possibility of usage of LightGBM in any specific environment (see `lightgbm-org/LightGBM#1743 <https://github.com/lightgbm-org/LightGBM/issues/1743>`__).

Therefore, the first thing you should try in case of segfaults is **compiling from the source** using ``pip install --no-binary lightgbm lightgbm``.
For the OS-specific prerequisites see https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst.

Also, feel free to post a new issue in our GitHub repository. We always look at each case individually and try to find a root cause.

4. I would like to install LightGBM from conda. What channel should I choose?
-----------------------------------------------------------------------------

We strongly recommend installation from the ``conda-forge`` channel and not from the ``default`` one.

For some specific examples, see `this comment <https://github.com/lightgbm-org/LightGBM/issues/4948#issuecomment-1013766397>`__.

In addition, as of ``lightgbm==4.4.0``, the ``conda-forge`` package automatically supports CUDA-based GPU acceleration.

5. How do I subclass ``scikit-learn`` estimators?
-------------------------------------------------

For ``lightgbm <= 4.5.0``, copy all of the constructor arguments from the corresponding
``lightgbm`` class into the constructor of your custom estimator.

For later versions, just ensure that the constructor of your custom estimator calls ``super().__init__()``.

Consider the example below, which implements a regressor that allows creation of truncated predictions.
This pattern will work with ``lightgbm > 4.5.0``.

.. code-block:: python

    import numpy as np
    from lightgbm import LGBMRegressor
    from sklearn.datasets import make_regression

    class TruncatedRegressor(LGBMRegressor):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def predict(self, X, max_score: float = np.inf):
            preds = super().predict(X)
            np.clip(preds, a_min=None, a_max=max_score, out=preds)
            return preds

    X, y = make_regression(n_samples=1_000, n_features=4)

    reg_trunc = TruncatedRegressor().fit(X, y)

    preds = reg_trunc.predict(X)
    print(f"mean: {preds.mean():.2f}, max: {preds.max():.2f}")
    # mean: -6.81, max: 345.10

    preds_trunc = reg_trunc.predict(X, max_score=preds.mean())
    print(f"mean: {preds_trunc.mean():.2f}, max: {preds_trunc.max():.2f}")
    # mean: -56.50, max: -6.81
