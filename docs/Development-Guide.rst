Development Guide
=================

Algorithms
----------

Refer to `Features <./Features.rst>`__ for understanding of important algorithms used in LightGBM.

Classes and Code Structure
--------------------------

Important Classes
~~~~~~~~~~~~~~~~~

+-------------------------+----------------------------------------------------------------------------------------+
| Class                   | Description                                                                            |
+=========================+========================================================================================+
| ``Application``         | The entrance of application, including training and prediction logic                   |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Bin``                 | Data structure used for storing feature discrete values (converted from float values)  |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Boosting``            | Boosting interface (GBDT, DART, etc.)                                                  |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Config``              | Stores parameters and configurations                                                   |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Dataset``             | Stores information of dataset                                                          |
+-------------------------+----------------------------------------------------------------------------------------+
| ``DatasetLoader``       | Used to construct dataset                                                              |
+-------------------------+----------------------------------------------------------------------------------------+
| ``FeatureGroup``        | Stores the data of feature, could be multiple features                                 |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Metric``              | Evaluation metrics                                                                     |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Network``             | Network interfaces and communication algorithms                                        |
+-------------------------+----------------------------------------------------------------------------------------+
| ``ObjectiveFunction``   | Objective functions used to train                                                      |
+-------------------------+----------------------------------------------------------------------------------------+
| ``Tree``                | Stores information of tree model                                                       |
+-------------------------+----------------------------------------------------------------------------------------+
| ``TreeLearner``         | Used to learn trees                                                                    |
+-------------------------+----------------------------------------------------------------------------------------+

Code Structure
~~~~~~~~~~~~~~

+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| Path                | Description                                                                                                                        |
+=====================+====================================================================================================================================+
| ./include           | Header files                                                                                                                       |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./include/utils     | Some common functions                                                                                                              |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/application   | Implementations of training and prediction logic                                                                                   |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/boosting      | Implementations of Boosting                                                                                                        |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/io            | Implementations of IO related classes, including ``Bin``, ``Config``, ``Dataset``, ``DatasetLoader``, ``Feature`` and ``Tree``     |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/metric        | Implementations of metrics                                                                                                         |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/network       | Implementations of network functions                                                                                               |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/objective     | Implementations of objective functions                                                                                             |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+
| ./src/treelearner   | Implementations of tree learners                                                                                                   |
+---------------------+------------------------------------------------------------------------------------------------------------------------------------+

Documents API
-------------

Refer to `docs README <./README.rst>`__.

C API
-----

Refer to `C API <./C-API.rst>`__ or the comments in `c\_api.h <https://github.com/lightgbm-org/LightGBM/blob/main/include/LightGBM/c_api.h>`__ file, from which the documentation is generated.

Linting
-------

Every commit in the repository is tested with multiple static analyzers.

When developing locally, run some of them using ``pre-commit`` <https://pre-commit.com/>.

.. code-block:: bash

    pre-commit run --all-files

That command will check for some issues and automatically reformat the code.

Tests
-----

C++ unit tests are located in the ``./tests/cpp_tests`` folder and written with the help of Google Test framework.
To run tests locally first refer to the `Installation Guide <./Installation-Guide.rst#build-c-unit-tests>`__ for how to build tests and then simply run compiled executable file.
It is highly recommended to build tests with `sanitizers <./Installation-Guide.rst#sanitizers>`__.

High Level Language Package
---------------------------

See the implementations at `Python-package <https://github.com/lightgbm-org/LightGBM/tree/main/python-package>`__ and `R-package <https://github.com/lightgbm-org/LightGBM/tree/main/R-package>`__.

Questions
---------

Refer to `FAQ <./FAQ.rst>`__.

Also feel free to open `issues <https://github.com/lightgbm-org/LightGBM/issues>`__ if you met problems.
