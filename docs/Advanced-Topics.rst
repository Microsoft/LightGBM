Advanced Topics
===============

Missing Value Handle
--------------------

-  LightGBM enables the missing value handle by default. Disable it by setting ``use_missing=false``.

-  LightGBM uses NA (NaN) to represent missing values by default. Change it to use zero by setting ``zero_as_missing=true``.

-  When ``zero_as_missing=false`` (default), the unrecorded values in sparse matrices (and LightSVM) are treated as zeros.

-  When ``zero_as_missing=true``, NA and zeros (including unrecorded values in sparse matrices (and LightSVM)) are treated as missing.

Categorical Feature Support
---------------------------

-  LightGBM offers good accuracy with integer-encoded categorical features. LightGBM offers the following approaches to deal with categorical features:

   -  Method 1: Applies `Fisher (1958) <https://www.tandfonline.com/doi/abs/10.1080/01621459.1958.10501479>`__ to find the optimal split over categories as `described here <./Features.rst#optimal-split-for-categorical-features>`__.

   -  Method 2: Encoding categorical features into numerical values. We provide two encoding options:

      -  **Target encoding**: encode the categorical feature value by the mean of labels of data with the same feature value in the training set. It is easy to overfit the training data if the encoded value of a training data point uses the label of that training data point itself. So we would randomly divide the training data into folds, and when calculating the target encoding for data in one fold, we only consider data in other folds.

      -  **Count encoding**: encode the categorical feature value by the total number of data with the same feature value in the training set.
   
   These methods often perform better than one-hot encoding.

-  Use ``categorical_feature`` to specify the categorical features.
   Refer to the parameter ``categorical_feature`` in `Parameters <./Parameters.rst#categorical_feature>`__.

-  Categorical features must be encoded as non-negative integers (``int``) less than ``Int32.MaxValue`` (2147483647).
   It is best to use a contiguous range of integers started from zero.

-  Use ``category_encoders`` to specify the methods used to deal with categorical features. We use

   -  ``raw`` to indicate method 1.

   -  ``target[:prior]`` to indicate target encoding in method 2. The ``prior`` is a real number used to smooth the calculation of encoded values. So ``target[:prior]`` is calculated as: ``(sum_label + prior * prior_weight) / (count + prior_weight)``. Here ``sum_label`` is the sum of labels of data in the training set with the same categorical feature value, ``count`` is the total number of data with the same feature value in the training set (the value of count encoding), and ``prior_weight`` is a hyper-parameter. If the prior value is missing, we use the mean of all labels of training data as default prior.

   -  ``count`` to indicate count encoding in method 2.

   Note that the aforementioned methods can be used simultaneously. Different methods are separated by commas.
   For example ``category_encoders=target:0.5,target:count,raw`` will enable using splits with method 1, and in addition, convert each categorical feature into 3 numerical features. The first one uses target encoding with prior ``0.5``. The second one uses target encoding with default prior, which is the mean of labels of the training data. The third one uses count encoding.
   When ``category_encoders`` is empty, ``raw`` will be used by default. The numbers and names of features will be changed when ``category_encoders`` is not ``raw``.
   Suppose the original name of a feature is ``NAME``, the naming rules of its target and count encoding features are:

   -  For the encoder ``target`` (without user specified prior), it will be named as ``NAME_label_mean_prior_target_encoding_<label_mean>``, where ``<label_mean>`` is the mean of all labels in the training set.

   -  For the encoder ``target:<prior>`` (with user specified prior), it will be named as ``NAME_target_encoding_<prior>``.

   -  For the encoder ``count``, it will be named as ``NAME_count_encoding``.

   Use ``get_feature_name()`` of Python Booster class or ``feature_name()`` of Python Dataset class after training to get the actual feature names used when ``category_encoders`` is set.

-  Use ``num_target_encoding_folds`` to specify the number of folds to divide the training data when using target encoding.

-  Use ``prior_weight`` to specify the weight of prior in target encoding calculation. Higher value will enforce more regularization on target encoding.

-  When using method 1 (in other words, ``raw`` is enabled in ``category_encoders``), use ``min_data_per_group``, ``cat_smooth`` to deal with over-fitting (when ``#data`` is small or ``#category`` is large).

LambdaRank
----------

-  The label should be of type ``int``, such that larger numbers correspond to higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect).

-  Use ``label_gain`` to set the gain(weight) of ``int`` label.

-  Use ``lambdarank_truncation_level`` to truncate the max DCG.

Cost Efficient Gradient Boosting
--------------------------------

`Cost Efficient Gradient Boosting <https://papers.nips.cc/paper/6753-cost-efficient-gradient-boosting.pdf>`_ (CEGB)  makes it possible to penalise boosting based on the cost of obtaining feature values.
CEGB penalises learning in the following ways:

- Each time a tree is split, a penalty of ``cegb_penalty_split`` is applied.
- When a feature is used for the first time, ``cegb_penalty_feature_coupled`` is applied. This penalty can be different for each feature and should be specified as one ``double`` per feature.
- When a feature is used for the first time for a data row, ``cegb_penalty_feature_lazy`` is applied. Like ``cegb_penalty_feature_coupled``, this penalty is specified as one ``double`` per feature.

Each of the penalties above is scaled by ``cegb_tradeoff``.
Using this parameter, it is possible to change the overall strength of the CEGB penalties by changing only one parameter.

Parameters Tuning
-----------------

-  Refer to `Parameters Tuning <./Parameters-Tuning.rst>`__.

.. _Parallel Learning:

Distributed Learning
--------------------

-  Refer to `Distributed Learning Guide <./Parallel-Learning-Guide.rst>`__.

GPU Support
-----------

-  Refer to `GPU Tutorial <./GPU-Tutorial.rst>`__ and `GPU Targets <./GPU-Targets.rst>`__.

Recommendations for gcc Users (MinGW, \*nix)
--------------------------------------------

-  Refer to `gcc Tips <./gcc-Tips.rst>`__.
