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

-  LightGBM offers good accuracy with integer-encoded categorical features. LightGBM applies
   `Fisher (1958) <https://www.jstor.org/stable/2281952>`_
   to find the optimal split over categories as
   `described here <./Features.rst#optimal-split-for-categorical-features>`_. This often performs better than one-hot encoding.

-  Use ``categorical_feature`` to specify the categorical features.
   Refer to the parameter ``categorical_feature`` in `Parameters <./Parameters.rst#categorical_feature>`__.

-  Categorical features will be cast to ``int32`` (integer codes will be extracted from categorical or enum dtypes in the Python-package) so they must be encoded as non-negative integers (negative values will be treated as missing)
   less than ``Int32.MaxValue`` (2147483647).
   It is best to use a contiguous range of integers started from zero.
   Floating point numbers in categorical features will be rounded towards 0.

-  For dataframe inputs with categorical or enum dtypes, LightGBM will
   align categories to those observed during training before converting them to integer values.
   This ensures consistent encoding between training and prediction without additional preprocessing.

-  With ``categorical_feature="auto"`` (the default), LightGBM auto-detects **unordered**
   categorical columns of a dataframe input and treats them as categorical features. Ordered
   categoricals are instead treated as ordinal numeric features (standard ``value < threshold``
   splits on the integer codes), since a declared order between categories matches numeric-split
   semantics better than partition-based categorical splits. Pass the column name explicitly via
   ``categorical_feature=["col"]`` to force categorical handling of an ordered column. The
   "ordered" check is normalized across dataframe backends via
   `narwhals.is_ordered_categorical <https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#is_ordered_categorical>`__:

   -  **pandas**: ``Categorical(ordered=False)`` → categorical; ``ordered=True`` → numeric.
   -  **polars**: ``Categorical(...)`` → categorical (or numeric on polars < 2.0 when constructed
      with the now-removed ``ordering="physical"`` argument); ``Enum([...])`` → numeric.
   -  **pyarrow**: ``dictionary(..., ordered=False)`` → categorical; ``ordered=True`` → numeric.

-  At ``predict()`` time, categories not seen during training will be treated as missing values.

-  Use ``min_data_per_group``, ``cat_smooth`` to deal with over-fitting (when ``#data`` is small or ``#category`` is large).

-  For a categorical feature with high cardinality (``#category`` is large), it often works best to
   treat the feature as numeric, either by simply ignoring the categorical interpretation of the integers or
   by embedding the categories in a low-dimensional numeric space.

LambdaRank
----------

-  The label should be of type ``int``, such that larger numbers correspond to higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect).

-  Use ``label_gain`` to set the gain(weight) of ``int`` label.

-  Use ``lambdarank_truncation_level`` to truncate the max DCG.

Cost Efficient Gradient Boosting
--------------------------------

`Cost Efficient Gradient Boosting <https://proceedings.neurips.cc/paper/2017/hash/4fac9ba115140ac4f1c22da82aa0bc7f-Abstract.html>`_ (CEGB)  makes it possible to penalise boosting based on the cost of obtaining feature values.
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

Support for Position Bias Treatment
------------------------------------

Often the relevance labels provided in Learning-to-Rank tasks might be derived from implicit user feedback (e.g., clicks) and therefore might be biased due to their position/location on the screen when having been presented to a user.
LightGBM can make use of positional data.

For example, consider the case where you expect that the first 3 results from a search engine will be visible in users' browsers without scrolling, and all other results for a query would require scrolling.

LightGBM could be told to account for the position bias from results being "above the fold" by providing a ``positions`` array encoded as follows:

::

    0
    0
    0
    1
    1
    0
    0
    0
    1
    ...

Where ``0 = "above the fold"`` and ``1 = "requires scrolling"``.
The specific values are not important, as long as they are consistent across all observations in the training data.
An encoding like ``100 = "above the fold"`` and ``17 = "requires scrolling"`` would result in exactly the same trained model.

In that way, ``positions`` in LightGBM's API are similar to a categorical feature.
Just as with non-ordinal categorical features, an integer representation is just used for memory and computational efficiency... LightGBM does not care about the absolute or relative magnitude of the values.

Unlike a categorical feature, however, ``positions`` are used to adjust the target to reduce the bias in predictions made by the trained model.

The position file corresponds with training data file line by line, and has one position per line. And if the name of training data file is ``train.txt``, the position file should be named as ``train.txt.position`` and placed in the same folder as the data file.
In this case, LightGBM will load the position file automatically if it exists. The positions can also be specified through the ``Dataset`` constructor when using Python API. If the positions are specified in both approaches, the ``.position`` file will be ignored.

Currently, implemented is an approach to model position bias by using an idea of Generalized Additive Models (`GAM <https://en.wikipedia.org/wiki/Generalized_additive_model>`_) to linearly decompose the document score ``s`` into the sum of a relevance component ``f`` and a positional component ``g``:  ``s(x, pos) = f(x) + g(pos)`` where the former component depends on the original query-document features and the latter depends on the position of an item.
During the training, the compound scoring function ``s(x, pos)`` is fit with a standard ranking algorithm (e.g., LambdaMART) which boils down to jointly learning the relevance component ``f(x)`` (it is later returned as an unbiased model) and the position factors ``g(pos)`` that help better explain the observed (biased) labels.
Similar score decomposition ideas have previously been applied for classification & pointwise ranking tasks with assumptions of binary labels and binary relevance (a.k.a. "two-tower" models, refer to the papers: `Towards Disentangling Relevance and Bias in Unbiased Learning to Rank <https://arxiv.org/abs/2212.13937>`_, `PAL: a position-bias aware learning framework for CTR prediction in live recommender systems <https://dl.acm.org/doi/10.1145/3298689.3347033>`_, `A General Framework for Debiasing in CTR Prediction <https://arxiv.org/abs/2112.02767>`_).
In LightGBM, we adapt this idea to general pairwise Lerarning-to-Rank with arbitrary ordinal relevance labels.
Besides, GAMs have been used in the context of explainable ML (`Accurate Intelligible Models with Pairwise Interactions <https://www.cs.cornell.edu/~yinlou/projects/gam/>`_) to linearly decompose the contribution of each feature (and possibly their pairwise interactions) to the overall score, for subsequent analysis and interpretation of their effects in the trained models.
