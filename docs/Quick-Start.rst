Quick Start
===========

This is a quick start guide for LightGBM CLI version.

Follow the `Installation Guide <./Installation-Guide.rst>`__ to install LightGBM first.

**List of other helpful links**

-  `Parameters <./Parameters.rst>`__

-  `Parameters Tuning <./Parameters-Tuning.rst>`__

-  `Python-package Quick Start <./Python-Intro.rst>`__

-  `Python API <./Python-API.rst>`__

Training Data Format
--------------------

LightGBM supports input data files with `CSV`_, `TSV`_ and `LibSVM`_ (zero-based) formats.

Files could be both with and without `headers <./Parameters.rst#header>`__.

`Label column <./Parameters.rst#label_column>`__ could be specified both by index and by name.

Some columns could be `ignored <./Parameters.rst#ignore_column>`__.

Categorical Feature Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LightGBM can use categorical features directly (without one-hot encoding).
The experiment on `Expo data`_ shows about 8x speed-up compared with one-hot encoding.

For the setting details, please refer to the ``categorical_feature`` `parameter <./Parameters.rst#categorical_feature>`__.

Weight and Query/Group Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LightGBM also supports weighted training, it needs an additional `weight data <./Parameters.rst#weight-data>`__.
And it needs an additional `query data <./Parameters.rst#query-data>`_ for ranking task.

Also, `weight <./Parameters.rst#weight_column>`__ and `query <./Parameters.rst#group_column>`__ data could be specified as columns in training data in the same manner as label.

Parameters Quick Look
---------------------

The parameters format is ``key1=value1 key2=value2 ...``.

Parameters can be set both in config file and command line.
If one parameter appears in both command line and config file, LightGBM will use the parameter from the command line.

The most important parameters which new users should take a look at are located into `Core Parameters <./Parameters.rst#core-parameters>`__
and the top of `Learning Control Parameters <./Parameters.rst#learning-control-parameters>`__
sections of the full detailed list of `LightGBM's parameters <./Parameters.rst>`__.

Run LightGBM
------------

::

    lightgbm config=your_config_file other_args ...

Parameters can be set both in the config file and command line, and the parameters in command line have higher priority than in the config file.
For example, the following command line will keep ``num_trees=10`` and ignore the same parameter in the config file.

::

    lightgbm config=train.conf num_trees=10

Examples
--------

-  `Binary Classification <https://github.com/lightgbm-org/LightGBM/tree/main/examples/binary_classification>`__

-  `Regression <https://github.com/lightgbm-org/LightGBM/tree/main/examples/regression>`__

-  `Lambdarank <https://github.com/lightgbm-org/LightGBM/tree/main/examples/lambdarank>`__

-  `Distributed Learning <https://github.com/lightgbm-org/LightGBM/tree/main/examples/parallel_learning>`__

.. _CSV: https://en.wikipedia.org/wiki/Comma-separated_values

.. _TSV: https://en.wikipedia.org/wiki/Tab-separated_values

.. _LibSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

.. _Expo data: https://community.amstat.org/jointscsg-section/dataexpo/dataexpo2009
