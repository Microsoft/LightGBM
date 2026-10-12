Experiments
===========

Comparison Experiment
---------------------

For the detailed experiment scripts and output logs, please refer to this `repo`_.

History
^^^^^^^

08 Mar, 2020: update according to the latest master branch (`1b97eaf <https://github.com/dmlc/xgboost/commit/1b97eaf7a74315bfa2c132d59f937a35408bcfd1>`__ for XGBoost, `bcad692 <https://github.com/lightgbm-org/LightGBM/commit/bcad692e263e0317cab11032dd017c78f9e58e5f>`__ for LightGBM). (``xgboost_exact`` is not updated for it is too slow.)

27 Feb, 2017: first version.

Data
^^^^

We used 5 datasets to conduct our comparison experiments. Details of data are listed in the following table:

+-----------+-----------------------+---------------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| Data      | Task                  | Link                                                                            | #Train\_Set | #Feature | Comments                                     |
+===========+=======================+=================================================================================+=============+==========+==============================================+
| Higgs     | Binary classification | `link <https://archive.ics.uci.edu/dataset/280/higgs>`__                        | 10,500,000  | 28       | last 500,000 samples were used as test set   |
+-----------+-----------------------+---------------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| Yahoo LTR | Learning to rank      | `link <https://proceedings.mlr.press/v14/chapelle11a.html>`__                   | 473,134     | 700      | set1.train as train, set1.test as test       |
+-----------+-----------------------+---------------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| MS LTR    | Learning to rank      | `link <https://www.microsoft.com/en-us/research/project/mslr/>`__               | 2,270,296   | 137      | {S1,S2,S3} as train set, {S5} as test set    |
+-----------+-----------------------+---------------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| Expo      | Binary classification | `link <https://community.amstat.org/jointscsg-section/dataexpo/dataexpo2009>`__ | 11,000,000  | 700      | last 1,000,000 samples were used as test set |
+-----------+-----------------------+---------------------------------------------------------------------------------+-------------+----------+----------------------------------------------+
| Allstate  | Binary classification | `link <https://www.kaggle.com/c/ClaimPredictionChallenge>`__                    | 13,184,290  | 4228     | last 1,000,000 samples were used as test set |
+-----------+-----------------------+---------------------------------------------------------------------------------+-------------+----------+----------------------------------------------+

Environment
^^^^^^^^^^^

We ran all experiments on a single Linux server (Azure ND24s) with the following specifications:

+------------------+-----------------+---------------------+
| OS               | CPU             | Memory              |
+==================+=================+=====================+
| Ubuntu 16.04 LTS | 2 \* E5-2690 v4 | 448GB               |
+------------------+-----------------+---------------------+

Baseline
^^^^^^^^

We used `xgboost`_ as a baseline.

Both xgboost and LightGBM were built with OpenMP support.

Settings
^^^^^^^^

We set up total 3 settings for experiments. The parameters of these settings are:

1. xgboost:

   .. code:: text

       eta = 0.1
       max_depth = 8
       num_round = 500
       nthread = 16
       tree_method = exact
       min_child_weight = 100

2. xgboost\_hist (using histogram based algorithm):

   .. code:: text

       eta = 0.1
       num_round = 500
       nthread = 16
       min_child_weight = 100
       tree_method = hist
       grow_policy = lossguide
       max_depth = 0
       max_leaves = 255

3. LightGBM:

   .. code:: text

       learning_rate = 0.1
       num_leaves = 255
       num_trees = 500
       num_threads = 16
       min_data_in_leaf = 0
       min_sum_hessian_in_leaf = 100

xgboost grows trees depth-wise and controls model complexity by ``max_depth``.
LightGBM uses a leaf-wise algorithm instead and controls model complexity by ``num_leaves``.
So we cannot compare them in the exact same model setting. For the tradeoff, we use xgboost with ``max_depth=8``, which will have max number leaves to 255, to compare with LightGBM with ``num_leaves=255``.

Other parameters are default values.

Result
^^^^^^

Speed
'''''

We compared speed using only the training task without any test or metric output. We didn't count the time for IO.
For the ranking tasks, since XGBoost and LightGBM implement different ranking objective functions, we used ``regression`` objective for speed benchmark, for the fair comparison.

The following table is the comparison of time cost:

+-----------+-----------+---------------+---------------+
| Data      | xgboost   | xgboost\_hist | LightGBM      |
+===========+===========+===============+===============+
| Higgs     | 3794.34 s | 165.575 s     | **130.094 s** |
+-----------+-----------+---------------+---------------+
| Yahoo LTR | 674.322 s | 131.462 s     | **76.229 s**  |
+-----------+-----------+---------------+---------------+
| MS LTR    | 1251.27 s | 98.386 s      | **70.417 s**  |
+-----------+-----------+---------------+---------------+
| Expo      | 1607.35 s | 137.65 s      | **62.607 s**  |
+-----------+-----------+---------------+---------------+
| Allstate  | 2867.22 s | 315.256 s     | **148.231 s** |
+-----------+-----------+---------------+---------------+

LightGBM ran faster than xgboost on all experiment data sets.

Accuracy
''''''''

We computed all accuracy metrics only on the test data set.

+-----------+-----------------+----------+-------------------+--------------+
| Data      | Metric          | xgboost  | xgboost\_hist     | LightGBM     |
+===========+=================+==========+===================+==============+
| Higgs     | AUC             | 0.839593 | 0.845314          | **0.845724** |
+-----------+-----------------+----------+-------------------+--------------+
| Yahoo LTR | NDCG\ :sub:`1`  | 0.719748 | 0.720049          | **0.732981** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`3`  | 0.717813 | 0.722573          | **0.735689** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`5`  | 0.737849 | 0.740899          | **0.75352**  |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`10` | 0.78089  | 0.782957          | **0.793498** |
+-----------+-----------------+----------+-------------------+--------------+
| MS LTR    | NDCG\ :sub:`1`  | 0.483956 | 0.485115          | **0.517767** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`3`  | 0.467951 | 0.47313           | **0.501063** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`5`  | 0.472476 | 0.476375          | **0.504648** |
|           +-----------------+----------+-------------------+--------------+
|           | NDCG\ :sub:`10` | 0.492429 | 0.496553          | **0.524252** |
+-----------+-----------------+----------+-------------------+--------------+
| Expo      | AUC             | 0.756713 | 0.776224          | **0.776935** |
+-----------+-----------------+----------+-------------------+--------------+
| Allstate  | AUC             | 0.607201 | **0.609465**      |  0.609072    |
+-----------+-----------------+----------+-------------------+--------------+

Memory Consumption
''''''''''''''''''

We monitored RES while running training task. And we set ``two_round=true`` (this will increase data-loading time and
reduce peak memory usage but not affect training speed or accuracy) in LightGBM to reduce peak memory usage.

+-----------+---------+---------------+--------------------+--------------------+
| Data      | xgboost | xgboost\_hist | LightGBM (col-wise)|LightGBM (row-wise) |
+===========+=========+===============+====================+====================+
| Higgs     | 4.853GB | 7.335GB       | **0.897GB**        |     1.401GB        |
+-----------+---------+---------------+--------------------+--------------------+
| Yahoo LTR | 1.907GB | 4.023GB       | **1.741GB**        |     2.161GB        |
+-----------+---------+---------------+--------------------+--------------------+
| MS LTR    | 5.469GB | 7.491GB       | **0.940GB**        |     1.296GB        |
+-----------+---------+---------------+--------------------+--------------------+
| Expo      | 1.553GB | 2.606GB       | **0.555GB**        |     0.711GB        |
+-----------+---------+---------------+--------------------+--------------------+
| Allstate  | 6.237GB | 12.090GB      | **1.116GB**        |     1.755GB        |
+-----------+---------+---------------+--------------------+--------------------+

Parallel Experiment
-------------------

History
^^^^^^^

27 Feb, 2017: first version.

Data
^^^^

We used a terabyte click log dataset to conduct parallel experiments. Details are listed in following table:

+--------+-----------------------+---------+---------------+----------+
| Data   | Task                  | Link    | #Data         | #Feature |
+========+=======================+=========+===============+==========+
| Criteo | Binary classification | `link`_ | 1,700,000,000 | 67       |
+--------+-----------------------+---------+---------------+----------+

This data contains 13 integer features and 26 categorical features for 24 days of click logs.
We statisticized the click-through rate (CTR) and count for these 26 categorical features from the first ten days.
Then we used next ten days' data, after replacing the categorical features by the corresponding CTR and count, as training data.
The processed training data have a total of 1.7 billions records and 67 features.

Environment
^^^^^^^^^^^

We ran our experiments on 16 Windows servers with the following specifications:

+---------------------+-----------------+---------------------+-------------------------------------------+
| OS                  | CPU             | Memory              | Network Adapter                           |
+=====================+=================+=====================+===========================================+
| Windows Server 2012 | 2 \* E5-2670 v2 | DDR3 1600Mhz, 256GB | Mellanox ConnectX-3, 54Gbps, RDMA support |
+---------------------+-----------------+---------------------+-------------------------------------------+

Settings
^^^^^^^^

.. code:: text

    learning_rate = 0.1
    num_leaves = 255
    num_trees = 100
    num_thread = 16
    tree_learner = data

We used data parallel here because this data is large in ``#data`` but small in ``#feature``. Other parameters were default values.

Results
^^^^^^^

+----------+---------------+---------------------------+
| #Machine | Time per Tree | Memory Usage(per Machine) |
+==========+===============+===========================+
| 1        | 627.8 s       | 176GB                     |
+----------+---------------+---------------------------+
| 2        | 311 s         | 87GB                      |
+----------+---------------+---------------------------+
| 4        | 156 s         | 43GB                      |
+----------+---------------+---------------------------+
| 8        | 80 s          | 22GB                      |
+----------+---------------+---------------------------+
| 16       | 42 s          | 11GB                      |
+----------+---------------+---------------------------+

The results show that LightGBM achieves a linear speedup with distributed learning.

GPU Experiments
---------------

Refer to `GPU Performance <./GPU-Performance.rst>`__.

.. _repo: https://github.com/guolinke/boosting_tree_benchmarks

.. _xgboost: https://github.com/dmlc/xgboost

.. _link: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
