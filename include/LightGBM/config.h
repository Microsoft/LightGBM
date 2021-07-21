/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \note
 * - desc and descl2 fields must be written in reStructuredText format;
 * - nested sections can be placed only at the bottom of parent's section;
 * - [doc-only] tag indicates that only documentation for this param should be generated and all other actions are performed manually;
 * - [no-save] tag indicates that this param should not be saved into a model text representation.
 */
#ifndef LIGHTGBM_CONFIG_H_
#define LIGHTGBM_CONFIG_H_

#include <LightGBM/export.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace LightGBM {

/*! \brief Types of tasks */
enum TaskType {
  kTrain, kPredict, kConvertModel, KRefitTree, kSaveBinary
};
const int kDefaultNumLeaves = 31;

struct Config {
 public:
  std::string ToString() const;
  /*!
  * \brief Get string value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetString(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, std::string* out);

  /*!
  * \brief Get int value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetInt(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, int* out);

  /*!
  * \brief Get double value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetDouble(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, double* out);

  /*!
  * \brief Get bool value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetBool(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, bool* out);

  static void KV2Map(std::unordered_map<std::string, std::string>* params, const char* kv);
  static std::unordered_map<std::string, std::string> Str2Map(const char* parameters);

  #pragma region Parameters

  #pragma region Core Parameters

  // [no-save]
  // [doc-only]
  // alias = config_file
  // desc = path of config file
  // desc = **Note**: can be used only in CLI version
  std::string config = "";

  // [no-save]
  // [doc-only]
  // type = enum
  // default = train
  // options = train, predict, convert_model, refit
  // alias = task_type
  // desc = ``train``, for training, aliases: ``training``
  // desc = ``predict``, for prediction, aliases: ``prediction``, ``test``
  // desc = ``convert_model``, for converting model file into if-else format, see more information in `Convert Parameters <#convert-parameters>`__
  // desc = ``refit``, for refitting existing models with new data, aliases: ``refit_tree``
  // desc = ``save_binary``, load train (and validation) data then save dataset to binary file. Typical usage: ``save_binary`` first, then run multiple ``train`` tasks in parallel using the saved binary file
  // desc = **Note**: can be used only in CLI version; for language-specific packages you can use the correspondent functions
  TaskType task = TaskType::kTrain;

  // [doc-only]
  // type = enum
  // options = regression, regression_l1, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass, multiclassova, cross_entropy, cross_entropy_lambda, lambdarank, rank_xendcg
  // alias = objective_type, app, application
  // desc = regression application
  // descl2 = ``regression``, L2 loss, aliases: ``regression_l2``, ``l2``, ``mean_squared_error``, ``mse``, ``l2_root``, ``root_mean_squared_error``, ``rmse``
  // descl2 = ``regression_l1``, L1 loss, aliases: ``l1``, ``mean_absolute_error``, ``mae``
  // descl2 = ``huber``, `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`__
  // descl2 = ``fair``, `Fair loss <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520>`__
  // descl2 = ``poisson``, `Poisson regression <https://en.wikipedia.org/wiki/Poisson_regression>`__
  // descl2 = ``quantile``, `Quantile regression <https://en.wikipedia.org/wiki/Quantile_regression>`__
  // descl2 = ``mape``, `MAPE loss <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`__, aliases: ``mean_absolute_percentage_error``
  // descl2 = ``gamma``, Gamma regression with log-link. It might be useful, e.g., for modeling insurance claims severity, or for any target that might be `gamma-distributed <https://en.wikipedia.org/wiki/Gamma_distribution#Occurrence_and_applications>`__
  // descl2 = ``tweedie``, Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any target that might be `tweedie-distributed <https://en.wikipedia.org/wiki/Tweedie_distribution#Occurrence_and_applications>`__
  // desc = binary classification application
  // descl2 = ``binary``, binary `log loss <https://en.wikipedia.org/wiki/Cross_entropy>`__ classification (or logistic regression)
  // descl2 = requires labels in {0, 1}; see ``cross-entropy`` application for general probability labels in [0, 1]
  // desc = multi-class classification application
  // descl2 = ``multiclass``, `softmax <https://en.wikipedia.org/wiki/Softmax_function>`__ objective function, aliases: ``softmax``
  // descl2 = ``multiclassova``, `One-vs-All <https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest>`__ binary objective function, aliases: ``multiclass_ova``, ``ova``, ``ovr``
  // descl2 = ``num_class`` should be set as well
  // desc = cross-entropy application
  // descl2 = ``cross_entropy``, objective function for cross-entropy (with optional linear weights), aliases: ``xentropy``
  // descl2 = ``cross_entropy_lambda``, alternative parameterization of cross-entropy, aliases: ``xentlambda``
  // descl2 = label is anything in interval [0, 1]
  // desc = ranking application
  // descl2 = ``lambdarank``, `lambdarank <https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf>`__ objective. `label_gain <#label_gain>`__ can be used to set the gain (weight) of ``int`` label and all values in ``label`` must be smaller than number of elements in ``label_gain``
  // descl2 = ``rank_xendcg``, `XE_NDCG_MART <https://arxiv.org/abs/1911.09798>`__ ranking objective function, aliases: ``xendcg``, ``xe_ndcg``, ``xe_ndcg_mart``, ``xendcg_mart``
  // descl2 = ``rank_xendcg`` is faster than and achieves the similar performance as ``lambdarank``
  // descl2 = label should be ``int`` type, and larger number represents the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)
  std::string objective = "regression";

  // [doc-only]
  // type = enum
  // alias = boosting_type, boost
  // options = gbdt, rf, dart, goss
  // desc = ``gbdt``, traditional Gradient Boosting Decision Tree, aliases: ``gbrt``
  // desc = ``rf``, Random Forest, aliases: ``random_forest``
  // desc = ``dart``, `Dropouts meet Multiple Additive Regression Trees <https://arxiv.org/abs/1505.01866>`__
  // desc = ``goss``, Gradient-based One-Side Sampling
  // descl2 = **Note**: internally, LightGBM uses ``gbdt`` mode for the first ``1 / learning_rate`` iterations
  // desc = ``mvs``, Minimal variance sampling <https://arxiv.org/abs/1910.13204>__
  std::string boosting = "gbdt";

  // alias = train, train_data, train_data_file, data_filename
  // desc = path of training data, LightGBM will train from this data
  // desc = **Note**: can be used only in CLI version
  std::string data = "";

  // alias = test, valid_data, valid_data_file, test_data, test_data_file, valid_filenames
  // default = ""
  // desc = path(s) of validation/test data, LightGBM will output metrics for these data
  // desc = support multiple validation data, separated by ``,``
  // desc = **Note**: can be used only in CLI version
  std::vector<std::string> valid;

  // alias = num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, num_boost_round, n_estimators
  // check = >=0
  // desc = number of boosting iterations
  // desc = **Note**: internally, LightGBM constructs ``num_class * num_iterations`` trees for multi-class classification problems
  int num_iterations = 100;

  // alias = shrinkage_rate, eta
  // check = >0.0
  // desc = shrinkage rate
  // desc = in ``dart``, it also affects on normalization weights of dropped trees
  double learning_rate = 0.1;

  // default = 31
  // alias = num_leaf, max_leaves, max_leaf
  // check = >1
  // check = <=131072
  // desc = max number of leaves in one tree
  int num_leaves = kDefaultNumLeaves;

  // [doc-only]
  // type = enum
  // options = serial, feature, data, voting
  // alias = tree, tree_type, tree_learner_type
  // desc = ``serial``, single machine tree learner
  // desc = ``feature``, feature parallel tree learner, aliases: ``feature_parallel``
  // desc = ``data``, data parallel tree learner, aliases: ``data_parallel``
  // desc = ``voting``, voting parallel tree learner, aliases: ``voting_parallel``
  // desc = refer to `Distributed Learning Guide <./Parallel-Learning-Guide.rst>`__ to get more details
  std::string tree_learner = "serial";

  // alias = num_thread, nthread, nthreads, n_jobs
  // desc = number of threads for LightGBM
  // desc = ``0`` means default number of threads in OpenMP
  // desc = for the best speed, set this to the number of **real CPU cores**, not the number of threads (most CPUs use `hyper-threading <https://en.wikipedia.org/wiki/Hyper-threading>`__ to generate 2 threads per CPU core)
  // desc = do not set it too large if your dataset is small (for instance, do not use 64 threads for a dataset with 10,000 rows)
  // desc = be aware a task manager or any similar CPU monitoring tool might report that cores not being fully utilized. **This is normal**
  // desc = for distributed learning, do not use all CPU cores because this will cause poor performance for the network communication
  // desc = **Note**: please **don't** change this during training, especially when running multiple jobs simultaneously by external packages, otherwise it may cause undesirable errors
  int num_threads = 0;

  // [doc-only]
  // type = enum
  // options = cpu, gpu, cuda
  // alias = device
  // desc = device for the tree learning, you can use GPU to achieve the faster learning
  // desc = **Note**: it is recommended to use the smaller ``max_bin`` (e.g. 63) to get the better speed up
  // desc = **Note**: for the faster speed, GPU uses 32-bit float point to sum up by default, so this may affect the accuracy for some tasks. You can set ``gpu_use_dp=true`` to enable 64-bit float point, but it will slow down the training
  // desc = **Note**: refer to `Installation Guide <./Installation-Guide.rst#build-gpu-version>`__ to build LightGBM with GPU support
  std::string device_type = "cpu";

  // [doc-only]
  // alias = random_seed, random_state
  // default = None
  // desc = this seed is used to generate other seeds, e.g. ``data_random_seed``, ``feature_fraction_seed``, etc.
  // desc = by default, this seed is unused in favor of default values of other seeds
  // desc = this seed has lower priority in comparison with other seeds, which means that it will be overridden, if you set other seeds explicitly
  int seed = 0;

  // desc = used only with ``cpu`` device type
  // desc = setting this to ``true`` should ensure the stable results when using the same data and the same parameters (and different ``num_threads``)
  // desc = when you use the different seeds, different LightGBM versions, the binaries compiled by different compilers, or in different systems, the results are expected to be different
  // desc = you can `raise issues <https://github.com/microsoft/LightGBM/issues>`__ in LightGBM GitHub repo when you meet the unstable results
  // desc = **Note**: setting this to ``true`` may slow down the training
  // desc = **Note**: to avoid potential instability due to numerical issues, please set ``force_col_wise=true`` or ``force_row_wise=true`` when setting ``deterministic=true``
  bool deterministic = false;

  #pragma endregion

  #pragma region Learning Control Parameters

  // desc = used only with ``cpu`` device type
  // desc = set this to ``true`` to force col-wise histogram building
  // desc = enabling this is recommended when:
  // descl2 = the number of columns is large, or the total number of bins is large
  // descl2 = ``num_threads`` is large, e.g. ``> 20``
  // descl2 = you want to reduce memory cost
  // desc = **Note**: when both ``force_col_wise`` and ``force_row_wise`` are ``false``, LightGBM will firstly try them both, and then use the faster one. To remove the overhead of testing set the faster one to ``true`` manually
  // desc = **Note**: this parameter cannot be used at the same time with ``force_row_wise``, choose only one of them
  bool force_col_wise = false;

  // desc = used only with ``cpu`` device type
  // desc = set this to ``true`` to force row-wise histogram building
  // desc = enabling this is recommended when:
  // descl2 = the number of data points is large, and the total number of bins is relatively small
  // descl2 = ``num_threads`` is relatively small, e.g. ``<= 16``
  // descl2 = you want to use small ``bagging_fraction`` or ``goss`` boosting to speed up
  // desc = **Note**: setting this to ``true`` will double the memory cost for Dataset object. If you have not enough memory, you can try setting ``force_col_wise=true``
  // desc = **Note**: when both ``force_col_wise`` and ``force_row_wise`` are ``false``, LightGBM will firstly try them both, and then use the faster one. To remove the overhead of testing set the faster one to ``true`` manually
  // desc = **Note**: this parameter cannot be used at the same time with ``force_col_wise``, choose only one of them
  bool force_row_wise = false;

  // alias = hist_pool_size
  // desc = max cache size in MB for historical histogram
  // desc = ``< 0`` means no limit
  double histogram_pool_size = -1.0;

  // desc = limit the max depth for tree model. This is used to deal with over-fitting when ``#data`` is small. Tree still grows leaf-wise
  // desc = ``<= 0`` means no limit
  int max_depth = -1;

  // alias = min_data_per_leaf, min_data, min_child_samples
  // check = >=0
  // desc = minimal number of data in one leaf. Can be used to deal with over-fitting
  // desc = **Note**: this is an approximation based on the Hessian, so occasionally you may observe splits which produce leaf nodes that have less than this many observations
  int min_data_in_leaf = 20;

  // alias = min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight
  // check = >=0.0
  // desc = minimal sum hessian in one leaf. Like ``min_data_in_leaf``, it can be used to deal with over-fitting
  double min_sum_hessian_in_leaf = 1e-3;

  // alias = sub_row, subsample, bagging
  // check = >0.0
  // check = <=1.0
  // desc = like ``feature_fraction``, but this will randomly select part of data without resampling
  // desc = can be used to speed up training
  // desc = can be used to deal with over-fitting
  // desc = **Note**: to enable bagging, ``bagging_freq`` should be set to a non zero value as well
  double bagging_fraction = 1.0;

  // alias = pos_sub_row, pos_subsample, pos_bagging
  // check = >0.0
  // check = <=1.0
  // desc = used only in ``binary`` application
  // desc = used for imbalanced binary classification problem, will randomly sample ``#pos_samples * pos_bagging_fraction`` positive samples in bagging
  // desc = should be used together with ``neg_bagging_fraction``
  // desc = set this to ``1.0`` to disable
  // desc = **Note**: to enable this, you need to set ``bagging_freq`` and ``neg_bagging_fraction`` as well
  // desc = **Note**: if both ``pos_bagging_fraction`` and ``neg_bagging_fraction`` are set to ``1.0``,  balanced bagging is disabled
  // desc = **Note**: if balanced bagging is enabled, ``bagging_fraction`` will be ignored
  double pos_bagging_fraction = 1.0;

  // alias = neg_sub_row, neg_subsample, neg_bagging
  // check = >0.0
  // check = <=1.0
  // desc = used only in ``binary`` application
  // desc = used for imbalanced binary classification problem, will randomly sample ``#neg_samples * neg_bagging_fraction`` negative samples in bagging
  // desc = should be used together with ``pos_bagging_fraction``
  // desc = set this to ``1.0`` to disable
  // desc = **Note**: to enable this, you need to set ``bagging_freq`` and ``pos_bagging_fraction`` as well
  // desc = **Note**: if both ``pos_bagging_fraction`` and ``neg_bagging_fraction`` are set to ``1.0``,  balanced bagging is disabled
  // desc = **Note**: if balanced bagging is enabled, ``bagging_fraction`` will be ignored
  double neg_bagging_fraction = 1.0;

  // default = 1e-4
  // check = >0.0
  // desc = used in MVS boosting. If ``mvs_adaptive == true`` then this value is ignored.
  // desc = used only in ``mvs``
  double mvs_lambda = 1e-4;

  // default = false
  // desc = use adaptive variant of mvs boosting
  // desc = used only in ``mvs``
  bool mvs_adaptive = false;

  // default = 256000
  // check = >0
  // desc = used in MVS boosting training. If dataset size is greater than ``mvs_max_sequential_size``, then threshold
  // desc = for MVS is chosen for each thread independently.
  // desc = used only in ``mvs``
  // desc = **Note**: on small dataset setting this parameter less than size of dataset may produce results depending on number of threads
  int mvs_max_sequential_size = 256000;

  // alias = subsample_freq
  // desc = frequency for bagging
  // desc = ``0`` means disable bagging; ``k`` means perform bagging at every ``k`` iteration. Every ``k``-th iteration, LightGBM will randomly select ``bagging_fraction * 100 %`` of the data to use for the next ``k`` iterations
  // desc = **Note**: to enable bagging, ``bagging_fraction`` should be set to value smaller than ``1.0`` as well
  int bagging_freq = 0;

  // alias = bagging_fraction_seed
  // desc = random seed for bagging
  int bagging_seed = 3;

  // alias = sub_feature, colsample_bytree
  // check = >0.0
  // check = <=1.0
  // desc = LightGBM will randomly select a subset of features on each iteration (tree) if ``feature_fraction`` is smaller than ``1.0``. For example, if you set it to ``0.8``, LightGBM will select 80% of features before training each tree
  // desc = can be used to speed up training
  // desc = can be used to deal with over-fitting
  double feature_fraction = 1.0;

  // alias = sub_feature_bynode, colsample_bynode
  // check = >0.0
  // check = <=1.0
  // desc = LightGBM will randomly select a subset of features on each tree node if ``feature_fraction_bynode`` is smaller than ``1.0``. For example, if you set it to ``0.8``, LightGBM will select 80% of features at each tree node
  // desc = can be used to deal with over-fitting
  // desc = **Note**: unlike ``feature_fraction``, this cannot speed up training
  // desc = **Note**: if both ``feature_fraction`` and ``feature_fraction_bynode`` are smaller than ``1.0``, the final fraction of each node is ``feature_fraction * feature_fraction_bynode``
  double feature_fraction_bynode = 1.0;

  // desc = random seed for ``feature_fraction``
  int feature_fraction_seed = 2;

  // alias = extra_tree
  // desc = use extremely randomized trees
  // desc = if set to ``true``, when evaluating node splits LightGBM will check only one randomly-chosen threshold for each feature
  // desc = can be used to speed up training
  // desc = can be used to deal with over-fitting
  bool extra_trees = false;

  // desc = random seed for selecting thresholds when ``extra_trees`` is true
  int extra_seed = 6;

  // alias = early_stopping_rounds, early_stopping, n_iter_no_change
  // desc = will stop training if one metric of one validation data doesn't improve in last ``early_stopping_round`` rounds
  // desc = ``<= 0`` means disable
  // desc = can be used to speed up training
  int early_stopping_round = 0;

  // desc = LightGBM allows you to provide multiple evaluation metrics. Set this to ``true``, if you want to use only the first metric for early stopping
  bool first_metric_only = false;

  // alias = max_tree_output, max_leaf_output
  // desc = used to limit the max output of tree leaves
  // desc = ``<= 0`` means no constraint
  // desc = the final max output of leaves is ``learning_rate * max_delta_step``
  double max_delta_step = 0.0;

  // alias = reg_alpha
  // check = >=0.0
  // desc = L1 regularization
  double lambda_l1 = 0.0;

  // alias = reg_lambda, lambda
  // check = >=0.0
  // desc = L2 regularization
  double lambda_l2 = 0.0;

  // check = >=0.0
  // desc = linear tree regularization, corresponds to the parameter ``lambda`` in Eq. 3 of `Gradient Boosting with Piece-Wise Linear Regression Trees <https://arxiv.org/pdf/1802.05640.pdf>`__
  double linear_lambda = 0.0;

  // alias = min_split_gain
  // check = >=0.0
  // desc = the minimal gain to perform split
  // desc = can be used to speed up training
  double min_gain_to_split = 0.0;

  // alias = rate_drop
  // check = >=0.0
  // check = <=1.0
  // desc = used only in ``dart``
  // desc = dropout rate: a fraction of previous trees to drop during the dropout
  double drop_rate = 0.1;

  // desc = used only in ``dart``
  // desc = max number of dropped trees during one boosting iteration
  // desc = ``<=0`` means no limit
  int max_drop = 50;

  // check = >=0.0
  // check = <=1.0
  // desc = used only in ``dart``
  // desc = probability of skipping the dropout procedure during a boosting iteration
  double skip_drop = 0.5;

  // desc = used only in ``dart``
  // desc = set this to ``true``, if you want to use xgboost dart mode
  bool xgboost_dart_mode = false;

  // desc = used only in ``dart``
  // desc = set this to ``true``, if you want to use uniform drop
  bool uniform_drop = false;

  // desc = used only in ``dart``
  // desc = random seed to choose dropping models
  int drop_seed = 4;

  // check = >=0.0
  // check = <=1.0
  // desc = used only in ``goss``
  // desc = the retain ratio of large gradient data
  double top_rate = 0.2;

  // check = >=0.0
  // check = <=1.0
  // desc = used only in ``goss``
  // desc = the retain ratio of small gradient data
  double other_rate = 0.1;

  // check = >0
  // desc = minimal number of data per categorical group
  int min_data_per_group = 100;

  // check = >0
  // desc = used for the categorical features
  // desc = limit number of split points considered for categorical features. See `the documentation on how LightGBM finds optimal splits for categorical features <./Features.rst#optimal-split-for-categorical-features>`_ for more details
  // desc = can be used to speed up training
  int max_cat_threshold = 32;

  // check = >=0.0
  // desc = used for the categorical features
  // desc = L2 regularization in categorical split
  double cat_l2 = 10.0;

  // check = >=0.0
  // desc = used for the categorical features
  // desc = this can reduce the effect of noises in categorical features, especially for categories with few data
  double cat_smooth = 10.0;

  // check = >0
  // desc = when number of categories of one feature smaller than or equal to ``max_cat_to_onehot``, one-vs-other split algorithm will be used
  int max_cat_to_onehot = 4;

  // alias = topk
  // check = >0
  // desc = used only in ``voting`` tree learner, refer to `Voting parallel <./Parallel-Learning-Guide.rst#choose-appropriate-parallel-algorithm>`__
  // desc = set this to larger value for more accurate result, but it will slow down the training speed
  int top_k = 20;

  // type = multi-int
  // alias = mc, monotone_constraint
  // default = None
  // desc = used for constraints of monotonic features
  // desc = ``1`` means increasing, ``-1`` means decreasing, ``0`` means non-constraint
  // desc = you need to specify all features in order. For example, ``mc=-1,0,1`` means decreasing for 1st feature, non-constraint for 2nd feature and increasing for the 3rd feature
  std::vector<int8_t> monotone_constraints;

  // type = enum
  // alias = monotone_constraining_method, mc_method
  // options = basic, intermediate, advanced
  // desc = used only if ``monotone_constraints`` is set
  // desc = monotone constraints method
  // descl2 = ``basic``, the most basic monotone constraints method. It does not slow the library at all, but over-constrains the predictions
  // descl2 = ``intermediate``, a `more advanced method <https://hal.archives-ouvertes.fr/hal-02862802/document>`__, which may slow the library very slightly. However, this method is much less constraining than the basic method and should significantly improve the results
  // descl2 = ``advanced``, an `even more advanced method <https://hal.archives-ouvertes.fr/hal-02862802/document>`__, which may slow the library. However, this method is even less constraining than the intermediate method and should again significantly improve the results
  std::string monotone_constraints_method = "basic";

  // alias = monotone_splits_penalty, ms_penalty, mc_penalty
  // check = >=0.0
  // desc = used only if ``monotone_constraints`` is set
  // desc = `monotone penalty <https://hal.archives-ouvertes.fr/hal-02862802/document>`__: a penalization parameter X forbids any monotone splits on the first X (rounded down) level(s) of the tree. The penalty applied to monotone splits on a given depth is a continuous, increasing function the penalization parameter
  // desc = if ``0.0`` (the default), no penalization is applied
  double monotone_penalty = 0.0;

  // type = multi-double
  // alias = feature_contrib, fc, fp, feature_penalty
  // default = None
  // desc = used to control feature's split gain, will use ``gain[i] = max(0, feature_contri[i]) * gain[i]`` to replace the split gain of i-th feature
  // desc = you need to specify all features in order
  std::vector<double> feature_contri;

  // alias = fs, forced_splits_filename, forced_splits_file, forced_splits
  // desc = path to a ``.json`` file that specifies splits to force at the top of every decision tree before best-first learning commences
  // desc = ``.json`` file can be arbitrarily nested, and each split contains ``feature``, ``threshold`` fields, as well as ``left`` and ``right`` fields representing subsplits
  // desc = categorical splits are forced in a one-hot fashion, with ``left`` representing the split containing the feature value and ``right`` representing other values
  // desc = **Note**: the forced split logic will be ignored, if the split makes gain worse
  // desc = see `this file <https://github.com/microsoft/LightGBM/tree/master/examples/binary_classification/forced_splits.json>`__ as an example
  std::string forcedsplits_filename = "";

  // check = >=0.0
  // check = <=1.0
  // desc = decay rate of ``refit`` task, will use ``leaf_output = refit_decay_rate * old_leaf_output + (1.0 - refit_decay_rate) * new_leaf_output`` to refit trees
  // desc = used only in ``refit`` task in CLI version or as argument in ``refit`` function in language-specific package
  double refit_decay_rate = 0.9;

  // check = >=0.0
  // desc = cost-effective gradient boosting multiplier for all penalties
  double cegb_tradeoff = 1.0;

  // check = >=0.0
  // desc = cost-effective gradient-boosting penalty for splitting a node
  double cegb_penalty_split = 0.0;

  // type = multi-double
  // default = 0,0,...,0
  // desc = cost-effective gradient boosting penalty for using a feature
  // desc = applied per data point
  std::vector<double> cegb_penalty_feature_lazy;

  // type = multi-double
  // default = 0,0,...,0
  // desc = cost-effective gradient boosting penalty for using a feature
  // desc = applied once per forest
  std::vector<double> cegb_penalty_feature_coupled;

  // check = >= 0.0
  // desc = controls smoothing applied to tree nodes
  // desc = helps prevent overfitting on leaves with few samples
  // desc = if set to zero, no smoothing is applied
  // desc = if ``path_smooth > 0`` then ``min_data_in_leaf`` must be at least ``2``
  // desc = larger values give stronger regularization
  // descl2 = the weight of each node is ``(n / path_smooth) * w + w_p / (n / path_smooth + 1)``, where ``n`` is the number of samples in the node, ``w`` is the optimal node weight to minimise the loss (approximately ``-sum_gradients / sum_hessians``), and ``w_p`` is the weight of the parent node
  // descl2 = note that the parent output ``w_p`` itself has smoothing applied, unless it is the root node, so that the smoothing effect accumulates with the tree depth
  double path_smooth = 0;

  // desc = controls which features can appear in the same branch
  // desc = by default interaction constraints are disabled, to enable them you can specify
  // descl2 = for CLI, lists separated by commas, e.g. ``[0,1,2],[2,3]``
  // descl2 = for Python-package, list of lists, e.g. ``[[0, 1, 2], [2, 3]]``
  // descl2 = for R-package, list of character or numeric vectors, e.g. ``list(c("var1", "var2", "var3"), c("var3", "var4"))`` or ``list(c(1L, 2L, 3L), c(3L, 4L))``. Numeric vectors should use 1-based indexing, where ``1L`` is the first feature, ``2L`` is the second feature, etc
  // desc = any two features can only appear in the same branch only if there exists a constraint containing both features
  std::string interaction_constraints = "";

  // alias = verbose
  // desc = controls the level of LightGBM's verbosity
  // desc = ``< 0``: Fatal, ``= 0``: Error (Warning), ``= 1``: Info, ``> 1``: Debug
  int verbosity = 1;

  // [no-save]
  // alias = model_input, model_in
  // desc = filename of input model
  // desc = for ``prediction`` task, this model will be applied to prediction data
  // desc = for ``train`` task, training will be continued from this model
  // desc = **Note**: can be used only in CLI version
  std::string input_model = "";

  // [no-save]
  // alias = model_output, model_out
  // desc = filename of output model in training
  // desc = **Note**: can be used only in CLI version
  std::string output_model = "LightGBM_model.txt";

  // desc = the feature importance type in the saved model file
  // desc = ``0``: count-based feature importance (numbers of splits are counted); ``1``: gain-based feature importance (values of gain are counted)
  // desc = **Note**: can be used only in CLI version
  int saved_feature_importance_type = 0;

  // [no-save]
  // alias = save_period
  // desc = frequency of saving model file snapshot
  // desc = set this to positive value to enable this function. For example, the model file will be snapshotted at each iteration if ``snapshot_freq=1``
  // desc = **Note**: can be used only in CLI version
  int snapshot_freq = -1;

  #pragma endregion

  #pragma region IO Parameters

  #pragma region Dataset Parameters

  // alias = linear_trees
  // desc = fit piecewise linear gradient boosting tree
  // descl2 = tree splits are chosen in the usual way, but the model at each leaf is linear instead of constant
  // descl2 = the linear model at each leaf includes all the numerical features in that leaf's branch
  // descl2 = categorical features are used for splits as normal but are not used in the linear models
  // descl2 = missing values should not be encoded as ``0``. Use ``np.nan`` for Python, ``NA`` for the CLI, and ``NA``, ``NA_real_``, or ``NA_integer_`` for R
  // descl2 = it is recommended to rescale data before training so that features have similar mean and standard deviation
  // descl2 = **Note**: only works with CPU and ``serial`` tree learner
  // descl2 = **Note**: ``regression_l1`` objective is not supported with linear tree boosting
  // descl2 = **Note**: setting ``linear_tree=true`` significantly increases the memory use of LightGBM
  // descl2 = **Note**: if you specify ``monotone_constraints``, constraints will be enforced when choosing the split points, but not when fitting the linear models on leaves
  bool linear_tree = false;

  // check = >1
  // desc = max number of bins that feature values will be bucketed in
  // desc = small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)
  // desc = LightGBM will auto compress memory according to ``max_bin``. For example, LightGBM will use ``uint8_t`` for feature value if ``max_bin=255``
  int max_bin = 255;

  // type = multi-int
  // default = None
  // desc = max number of bins for each feature
  // desc = if not specified, will use ``max_bin`` for all features
  std::vector<int32_t> max_bin_by_feature;

  // check = >0
  // desc = minimal number of data inside one bin
  // desc = use this to avoid one-data-one-bin (potential over-fitting)
  int min_data_in_bin = 3;

  // alias = subsample_for_bin
  // check = >0
  // desc = number of data that sampled to construct feature discrete bins
  // desc = setting this to larger value will give better training result, but may increase data loading time
  // desc = set this to larger value if data is very sparse
  // desc = **Note**: don't set this to small values, otherwise, you may encounter unexpected errors and poor accuracy
  int bin_construct_sample_cnt = 200000;

  // alias = data_seed
  // desc = random seed for sampling data to construct histogram bins
  int data_random_seed = 1;

  // alias = is_sparse, enable_sparse, sparse
  // desc = used to enable/disable sparse optimization
  bool is_enable_sparse = true;

  // alias = is_enable_bundle, bundle
  // desc = set this to ``false`` to disable Exclusive Feature Bundling (EFB), which is described in `LightGBM: A Highly Efficient Gradient Boosting Decision Tree <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree>`__
  // desc = **Note**: disabling this may cause the slow training speed for sparse datasets
  bool enable_bundle = true;

  // desc = set this to ``false`` to disable the special handle of missing value
  bool use_missing = true;

  // desc = set this to ``true`` to treat all zero as missing values (including the unshown values in LibSVM / sparse matrices)
  // desc = set this to ``false`` to use ``na`` for representing missing values
  bool zero_as_missing = false;

  // desc = set this to ``true`` (the default) to tell LightGBM to ignore the features that are unsplittable based on ``min_data_in_leaf``
  // desc = as dataset object is initialized only once and cannot be changed after that, you may need to set this to ``false`` when searching parameters with ``min_data_in_leaf``, otherwise features are filtered by ``min_data_in_leaf`` firstly if you don't reconstruct dataset object
  // desc = **Note**: setting this to ``false`` may slow down the training
  bool feature_pre_filter = true;

  // alias = is_pre_partition
  // desc = used for distributed learning (excluding the ``feature_parallel`` mode)
  // desc = ``true`` if training data are pre-partitioned, and different machines use different partitions
  bool pre_partition = false;

  // alias = two_round_loading, use_two_round_loading
  // desc = set this to ``true`` if data file is too big to fit in memory
  // desc = by default, LightGBM will map data file to memory and load features from memory. This will provide faster data loading speed, but may cause run out of memory error when the data file is very big
  // desc = **Note**: works only in case of loading data directly from file
  bool two_round = false;

  // alias = has_header
  // desc = set this to ``true`` if input data has header
  // desc = **Note**: works only in case of loading data directly from file
  bool header = false;

  // type = int or string
  // alias = label
  // desc = used to specify the label column
  // desc = use number for index, e.g. ``label=0`` means column\_0 is the label
  // desc = add a prefix ``name:`` for column name, e.g. ``label=name:is_click``
  // desc = **Note**: works only in case of loading data directly from file
  std::string label_column = "";

  // type = int or string
  // alias = weight
  // desc = used to specify the weight column
  // desc = use number for index, e.g. ``weight=0`` means column\_0 is the weight
  // desc = add a prefix ``name:`` for column name, e.g. ``weight=name:weight``
  // desc = **Note**: works only in case of loading data directly from file
  // desc = **Note**: index starts from ``0`` and it doesn't count the label column when passing type is ``int``, e.g. when label is column\_0, and weight is column\_1, the correct parameter is ``weight=0``
  std::string weight_column = "";

  // type = int or string
  // alias = group, group_id, query_column, query, query_id
  // desc = used to specify the query/group id column
  // desc = use number for index, e.g. ``query=0`` means column\_0 is the query id
  // desc = add a prefix ``name:`` for column name, e.g. ``query=name:query_id``
  // desc = **Note**: works only in case of loading data directly from file
  // desc = **Note**: data should be grouped by query\_id, for more information, see `Query Data <#query-data>`__
  // desc = **Note**: index starts from ``0`` and it doesn't count the label column when passing type is ``int``, e.g. when label is column\_0 and query\_id is column\_1, the correct parameter is ``query=0``
  std::string group_column = "";

  // type = multi-int or string
  // alias = ignore_feature, blacklist
  // desc = used to specify some ignoring columns in training
  // desc = use number for index, e.g. ``ignore_column=0,1,2`` means column\_0, column\_1 and column\_2 will be ignored
  // desc = add a prefix ``name:`` for column name, e.g. ``ignore_column=name:c1,c2,c3`` means c1, c2 and c3 will be ignored
  // desc = **Note**: works only in case of loading data directly from file
  // desc = **Note**: index starts from ``0`` and it doesn't count the label column when passing type is ``int``
  // desc = **Note**: despite the fact that specified columns will be completely ignored during the training, they still should have a valid format allowing LightGBM to load file successfully
  std::string ignore_column = "";

  // type = multi-int or string
  // alias = cat_feature, categorical_column, cat_column
  // desc = used to specify categorical features
  // desc = use number for index, e.g. ``categorical_feature=0,1,2`` means column\_0, column\_1 and column\_2 are categorical features
  // desc = add a prefix ``name:`` for column name, e.g. ``categorical_feature=name:c1,c2,c3`` means c1, c2 and c3 are categorical features
  // desc = **Note**: only supports categorical with ``int`` type (not applicable for data represented as pandas DataFrame in Python-package)
  // desc = **Note**: index starts from ``0`` and it doesn't count the label column when passing type is ``int``
  // desc = **Note**: all values should be less than ``Int32.MaxValue`` (2147483647)
  // desc = **Note**: using large values could be memory consuming. Tree decision rule works best when categorical features are presented by consecutive integers starting from zero
  // desc = **Note**: all negative values will be treated as **missing values**
  // desc = **Note**: the output cannot be monotonically constrained with respect to a categorical feature
  std::string categorical_feature = "";

  // desc = path to a ``.json`` file that specifies bin upper bounds for some or all features
  // desc = ``.json`` file should contain an array of objects, each containing the word ``feature`` (integer feature index) and ``bin_upper_bound`` (array of thresholds for binning)
  // desc = see `this file <https://github.com/microsoft/LightGBM/tree/master/examples/regression/forced_bins.json>`__ as an example
  std::string forcedbins_filename = "";

  // [no-save]
  // alias = is_save_binary, is_save_binary_file
  // desc = if ``true``, LightGBM will save the dataset (including validation data) to a binary file. This speed ups the data loading for the next time
  // desc = **Note**: ``init_score`` is not saved in binary file
  // desc = **Note**: can be used only in CLI version; for language-specific packages you can use the correspondent function
  bool save_binary = false;

  // desc = use precise floating point number parsing for text parser (e.g. CSV, TSV, LibSVM input)
  // desc = **Note**: setting this to ``true`` may lead to much slower text parsing
  bool precise_float_parser = false;

  #pragma endregion

  #pragma region Predict Parameters

  // [no-save]
  // desc = used only in ``prediction`` task
  // desc = used to specify from which iteration to start the prediction
  // desc = ``<= 0`` means from the first iteration
  int start_iteration_predict = 0;

  // [no-save]
  // desc = used only in ``prediction`` task
  // desc = used to specify how many trained iterations will be used in prediction
  // desc = ``<= 0`` means no limit
  int num_iteration_predict = -1;

  // [no-save]
  // alias = is_predict_raw_score, predict_rawscore, raw_score
  // desc = used only in ``prediction`` task
  // desc = set this to ``true`` to predict only the raw scores
  // desc = set this to ``false`` to predict transformed scores
  bool predict_raw_score = false;

  // [no-save]
  // alias = is_predict_leaf_index, leaf_index
  // desc = used only in ``prediction`` task
  // desc = set this to ``true`` to predict with leaf index of all trees
  bool predict_leaf_index = false;

  // [no-save]
  // alias = is_predict_contrib, contrib
  // desc = used only in ``prediction`` task
  // desc = set this to ``true`` to estimate `SHAP values <https://arxiv.org/abs/1706.06060>`__, which represent how each feature contributes to each prediction
  // desc = produces ``#features + 1`` values where the last value is the expected value of the model output over the training data
  // desc = **Note**: if you want to get more explanation for your model's predictions using SHAP values like SHAP interaction values, you can install `shap package <https://github.com/slundberg/shap>`__
  // desc = **Note**: unlike the shap package, with ``predict_contrib`` we return a matrix with an extra column, where the last column is the expected value
  // desc = **Note**: this feature is not implemented for linear trees
  bool predict_contrib = false;

  // [no-save]
  // desc = used only in ``prediction`` task
  // desc = control whether or not LightGBM raises an error when you try to predict on data with a different number of features than the training data
  // desc = if ``false`` (the default), a fatal error will be raised if the number of features in the dataset you predict on differs from the number seen during training
  // desc = if ``true``, LightGBM will attempt to predict on whatever data you provide. This is dangerous because you might get incorrect predictions, but you could use it in situations where it is difficult or expensive to generate some features and you are very confident that they were never chosen for splits in the model
  // desc = **Note**: be very careful setting this parameter to ``true``
  bool predict_disable_shape_check = false;

  // [no-save]
  // desc = used only in ``prediction`` task
  // desc = used only in ``classification`` and ``ranking`` applications
  // desc = if ``true``, will use early-stopping to speed up the prediction. May affect the accuracy
  // desc = **Note**: cannot be used with ``rf`` boosting type or custom objective function
  bool pred_early_stop = false;

  // [no-save]
  // desc = used only in ``prediction`` task
  // desc = the frequency of checking early-stopping prediction
  int pred_early_stop_freq = 10;

  // [no-save]
  // desc = used only in ``prediction`` task
  // desc = the threshold of margin in early-stopping prediction
  double pred_early_stop_margin = 10.0;

  // [no-save]
  // alias = predict_result, prediction_result, predict_name, prediction_name, pred_name, name_pred
  // desc = used only in ``prediction`` task
  // desc = filename of prediction result
  // desc = **Note**: can be used only in CLI version
  std::string output_result = "LightGBM_predict_result.txt";

  #pragma endregion

  #pragma region Convert Parameters

  // [no-save]
  // desc = used only in ``convert_model`` task
  // desc = only ``cpp`` is supported yet; for conversion model to other languages consider using `m2cgen <https://github.com/BayesWitnesses/m2cgen>`__ utility
  // desc = if ``convert_model_language`` is set and ``task=train``, the model will be also converted
  // desc = **Note**: can be used only in CLI version
  std::string convert_model_language = "";

  // [no-save]
  // alias = convert_model_file
  // desc = used only in ``convert_model`` task
  // desc = output filename of converted model
  // desc = **Note**: can be used only in CLI version
  std::string convert_model = "gbdt_prediction.cpp";

  #pragma endregion

  #pragma endregion

  #pragma region Objective Parameters

  // desc = used only in ``rank_xendcg`` objective
  // desc = random seed for objectives, if random process is needed
  int objective_seed = 5;

  // check = >0
  // alias = num_classes
  // desc = used only in ``multi-class`` classification application
  int num_class = 1;

  // alias = unbalance, unbalanced_sets
  // desc = used only in ``binary`` and ``multiclassova`` applications
  // desc = set this to ``true`` if training data are unbalanced
  // desc = **Note**: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
  // desc = **Note**: this parameter cannot be used at the same time with ``scale_pos_weight``, choose only **one** of them
  bool is_unbalance = false;

  // check = >0.0
  // desc = used only in ``binary`` and ``multiclassova`` applications
  // desc = weight of labels with positive class
  // desc = **Note**: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
  // desc = **Note**: this parameter cannot be used at the same time with ``is_unbalance``, choose only **one** of them
  double scale_pos_weight = 1.0;

  // check = >0.0
  // desc = used only in ``binary`` and ``multiclassova`` classification and in ``lambdarank`` applications
  // desc = parameter for the sigmoid function
  double sigmoid = 1.0;

  // desc = used only in ``regression``, ``binary``, ``multiclassova`` and ``cross-entropy`` applications
  // desc = adjusts initial score to the mean of labels for faster convergence
  bool boost_from_average = true;

  // desc = used only in ``regression`` application
  // desc = used to fit ``sqrt(label)`` instead of original values and prediction result will be also automatically converted to ``prediction^2``
  // desc = might be useful in case of large-range labels
  bool reg_sqrt = false;

  // check = >0.0
  // desc = used only in ``huber`` and ``quantile`` ``regression`` applications
  // desc = parameter for `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`__ and `Quantile regression <https://en.wikipedia.org/wiki/Quantile_regression>`__
  double alpha = 0.9;

  // check = >0.0
  // desc = used only in ``fair`` ``regression`` application
  // desc = parameter for `Fair loss <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520>`__
  double fair_c = 1.0;

  // check = >0.0
  // desc = used only in ``poisson`` ``regression`` application
  // desc = parameter for `Poisson regression <https://en.wikipedia.org/wiki/Poisson_regression>`__ to safeguard optimization
  double poisson_max_delta_step = 0.7;

  // check = >=1.0
  // check = <2.0
  // desc = used only in ``tweedie`` ``regression`` application
  // desc = used to control the variance of the tweedie distribution
  // desc = set this closer to ``2`` to shift towards a **Gamma** distribution
  // desc = set this closer to ``1`` to shift towards a **Poisson** distribution
  double tweedie_variance_power = 1.5;

  // check = >0
  // desc = used only in ``lambdarank`` application
  // desc = controls the number of top-results to focus on during training, refer to "truncation level" in the Sec. 3 of `LambdaMART paper <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf>`__
  // desc = this parameter is closely related to the desirable cutoff ``k`` in the metric **NDCG@k** that we aim at optimizing the ranker for. The optimal setting for this parameter is likely to be slightly higher than ``k`` (e.g., ``k + 3``) to include more pairs of documents to train on, but perhaps not too high to avoid deviating too much from the desired target metric **NDCG@k**
  int lambdarank_truncation_level = 30;

  // desc = used only in ``lambdarank`` application
  // desc = set this to ``true`` to normalize the lambdas for different queries, and improve the performance for unbalanced data
  // desc = set this to ``false`` to enforce the original lambdarank algorithm
  bool lambdarank_norm = true;

  // type = multi-double
  // default = 0,1,3,7,15,31,63,...,2^30-1
  // desc = used only in ``lambdarank`` application
  // desc = relevant gain for labels. For example, the gain of label ``2`` is ``3`` in case of default label gains
  // desc = separate by ``,``
  std::vector<double> label_gain;

  #pragma endregion

  #pragma region Metric Parameters

  // [doc-only]
  // alias = metrics, metric_types
  // default = ""
  // type = multi-enum
  // desc = metric(s) to be evaluated on the evaluation set(s)
  // descl2 = ``""`` (empty string or not specified) means that metric corresponding to specified ``objective`` will be used (this is possible only for pre-defined objective functions, otherwise no evaluation metric will be added)
  // descl2 = ``"None"`` (string, **not** a ``None`` value) means that no metric will be registered, aliases: ``na``, ``null``, ``custom``
  // descl2 = ``l1``, absolute loss, aliases: ``mean_absolute_error``, ``mae``, ``regression_l1``
  // descl2 = ``l2``, square loss, aliases: ``mean_squared_error``, ``mse``, ``regression_l2``, ``regression``
  // descl2 = ``rmse``, root square loss, aliases: ``root_mean_squared_error``, ``l2_root``
  // descl2 = ``quantile``, `Quantile regression <https://en.wikipedia.org/wiki/Quantile_regression>`__
  // descl2 = ``mape``, `MAPE loss <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`__, aliases: ``mean_absolute_percentage_error``
  // descl2 = ``huber``, `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`__
  // descl2 = ``fair``, `Fair loss <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520>`__
  // descl2 = ``poisson``, negative log-likelihood for `Poisson regression <https://en.wikipedia.org/wiki/Poisson_regression>`__
  // descl2 = ``gamma``, negative log-likelihood for **Gamma** regression
  // descl2 = ``gamma_deviance``, residual deviance for **Gamma** regression
  // descl2 = ``tweedie``, negative log-likelihood for **Tweedie** regression
  // descl2 = ``ndcg``, `NDCG <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`__, aliases: ``lambdarank``, ``rank_xendcg``, ``xendcg``, ``xe_ndcg``, ``xe_ndcg_mart``, ``xendcg_mart``
  // descl2 = ``map``, `MAP <https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/>`__, aliases: ``mean_average_precision``
  // descl2 = ``auc``, `AUC <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`__
  // descl2 = ``average_precision``, `average precision score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>`__
  // descl2 = ``binary_logloss``, `log loss <https://en.wikipedia.org/wiki/Cross_entropy>`__, aliases: ``binary``
  // descl2 = ``binary_error``, for one sample: ``0`` for correct classification, ``1`` for error classification
  // descl2 = ``auc_mu``, `AUC-mu <http://proceedings.mlr.press/v97/kleiman19a/kleiman19a.pdf>`__
  // descl2 = ``multi_logloss``, log loss for multi-class classification, aliases: ``multiclass``, ``softmax``, ``multiclassova``, ``multiclass_ova``, ``ova``, ``ovr``
  // descl2 = ``multi_error``, error rate for multi-class classification
  // descl2 = ``cross_entropy``, cross-entropy (with optional linear weights), aliases: ``xentropy``
  // descl2 = ``cross_entropy_lambda``, "intensity-weighted" cross-entropy, aliases: ``xentlambda``
  // descl2 = ``kullback_leibler``, `Kullback-Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`__, aliases: ``kldiv``
  // desc = support multiple metrics, separated by ``,``
  std::vector<std::string> metric;

  // [no-save]
  // check = >0
  // alias = output_freq
  // desc = frequency for metric output
  // desc = **Note**: can be used only in CLI version
  int metric_freq = 1;

  // [no-save]
  // alias = training_metric, is_training_metric, train_metric
  // desc = set this to ``true`` to output metric result over training dataset
  // desc = **Note**: can be used only in CLI version
  bool is_provide_training_metric = false;

  // type = multi-int
  // default = 1,2,3,4,5
  // alias = ndcg_eval_at, ndcg_at, map_eval_at, map_at
  // desc = used only with ``ndcg`` and ``map`` metrics
  // desc = `NDCG <https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG>`__ and `MAP <https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/>`__ evaluation positions, separated by ``,``
  std::vector<int> eval_at;

  // check = >0
  // desc = used only with ``multi_error`` metric
  // desc = threshold for top-k multi-error metric
  // desc = the error on each sample is ``0`` if the true class is among the top ``multi_error_top_k`` predictions, and ``1`` otherwise
  // descl2 = more precisely, the error on a sample is ``0`` if there are at least ``num_classes - multi_error_top_k`` predictions strictly less than the prediction on the true class
  // desc = when ``multi_error_top_k=1`` this is equivalent to the usual multi-error metric
  int multi_error_top_k = 1;

  // type = multi-double
  // default = None
  // desc = used only with ``auc_mu`` metric
  // desc = list representing flattened matrix (in row-major order) giving loss weights for classification errors
  // desc = list should have ``n * n`` elements, where ``n`` is the number of classes
  // desc = the matrix co-ordinate ``[i, j]`` should correspond to the ``i * n + j``-th element of the list
  // desc = if not specified, will use equal weights for all classes
  std::vector<double> auc_mu_weights;

  #pragma endregion

  #pragma region Network Parameters

  // check = >0
  // alias = num_machine
  // desc = the number of machines for distributed learning application
  // desc = this parameter is needed to be set in both **socket** and **mpi** versions
  int num_machines = 1;

  // check = >0
  // default = 12400 (random for Dask-package)
  // alias = local_port, port
  // desc = TCP listen port for local machines
  // desc = **Note**: don't forget to allow this port in firewall settings before training
  int local_listen_port = 12400;

  // check = >0
  // desc = socket time-out in minutes
  int time_out = 120;

  // alias = machine_list_file, machine_list, mlist
  // desc = path of file that lists machines for this distributed learning application
  // desc = each line contains one IP and one port for one machine. The format is ``ip port`` (space as a separator)
  // desc = **Note**: can be used only in CLI version
  std::string machine_list_filename = "";

  // alias = workers, nodes
  // desc = list of machines in the following format: ``ip1:port1,ip2:port2``
  std::string machines = "";

  #pragma endregion

  #pragma region GPU Parameters

  // desc = OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform
  // desc = ``-1`` means the system-wide default platform
  // desc = **Note**: refer to `GPU Targets <./GPU-Targets.rst#query-opencl-devices-in-your-system>`__ for more details
  int gpu_platform_id = -1;

  // desc = OpenCL device ID in the specified platform. Each GPU in the selected platform has a unique device ID
  // desc = ``-1`` means the default device in the selected platform
  // desc = **Note**: refer to `GPU Targets <./GPU-Targets.rst#query-opencl-devices-in-your-system>`__ for more details
  int gpu_device_id = -1;

  // desc = set this to ``true`` to use double precision math on GPU (by default single precision is used)
  // desc = **Note**: can be used only in OpenCL implementation, in CUDA implementation only double precision is currently supported
  bool gpu_use_dp = false;

  // check = >0
  // desc = number of GPUs
  // desc = **Note**: can be used only in CUDA implementation
  int num_gpu = 1;

  #pragma endregion

  #pragma endregion

  size_t file_load_progress_interval_bytes = size_t(10) * 1024 * 1024 * 1024;

  bool is_parallel = false;
  bool is_data_based_parallel = false;
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params);
  static const std::unordered_map<std::string, std::string>& alias_table();
  static const std::unordered_set<std::string>& parameter_set();
  std::vector<std::vector<double>> auc_mu_weights_matrix;
  std::vector<std::vector<int>> interaction_constraints_vector;

 private:
  void CheckParamConflict();
  void GetMembersFromString(const std::unordered_map<std::string, std::string>& params);
  std::string SaveMembersToString() const;
  void GetAucMuWeights();
  void GetInteractionConstraints();
};

inline bool Config::GetString(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, std::string* out) {
  if (params.count(name) > 0 && !params.at(name).empty()) {
    *out = params.at(name);
    return true;
  }
  return false;
}

inline bool Config::GetInt(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, int* out) {
  if (params.count(name) > 0 && !params.at(name).empty()) {
    if (!Common::AtoiAndCheck(params.at(name).c_str(), out)) {
      Log::Fatal("Parameter %s should be of type int, got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

inline bool Config::GetDouble(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, double* out) {
  if (params.count(name) > 0 && !params.at(name).empty()) {
    if (!Common::AtofAndCheck(params.at(name).c_str(), out)) {
      Log::Fatal("Parameter %s should be of type double, got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

inline bool Config::GetBool(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, bool* out) {
  if (params.count(name) > 0 && !params.at(name).empty()) {
    std::string value = params.at(name);
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("false") || value == std::string("-")) {
      *out = false;
    } else if (value == std::string("true") || value == std::string("+")) {
      *out = true;
    } else {
      Log::Fatal("Parameter %s should be \"true\"/\"+\" or \"false\"/\"-\", got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

struct ParameterAlias {
  static void KeyAliasTransform(std::unordered_map<std::string, std::string>* params) {
    std::unordered_map<std::string, std::string> tmp_map;
    for (const auto& pair : *params) {
      auto alias = Config::alias_table().find(pair.first);
      if (alias != Config::alias_table().end()) {  // found alias
        auto alias_set = tmp_map.find(alias->second);
        if (alias_set != tmp_map.end()) {  // alias already set
                                           // set priority by length & alphabetically to ensure reproducible behavior
          if (alias_set->second.size() < pair.first.size() ||
            (alias_set->second.size() == pair.first.size() && alias_set->second < pair.first)) {
            Log::Warning("%s is set with %s=%s, %s=%s will be ignored. Current value: %s=%s",
                         alias->second.c_str(), alias_set->second.c_str(), params->at(alias_set->second).c_str(),
                         pair.first.c_str(), pair.second.c_str(), alias->second.c_str(), params->at(alias_set->second).c_str());
          } else {
            Log::Warning("%s is set with %s=%s, will be overridden by %s=%s. Current value: %s=%s",
                         alias->second.c_str(), alias_set->second.c_str(), params->at(alias_set->second).c_str(),
                         pair.first.c_str(), pair.second.c_str(), alias->second.c_str(), pair.second.c_str());
            tmp_map[alias->second] = pair.first;
          }
        } else {  // alias not set
          tmp_map.emplace(alias->second, pair.first);
        }
      } else if (Config::parameter_set().find(pair.first) == Config::parameter_set().end()) {
        Log::Warning("Unknown parameter: %s", pair.first.c_str());
      }
    }
    for (const auto& pair : tmp_map) {
      auto alias = params->find(pair.first);
      if (alias == params->end()) {  // not find
        params->emplace(pair.first, params->at(pair.second));
        params->erase(pair.second);
      } else {
        Log::Warning("%s is set=%s, %s=%s will be ignored. Current value: %s=%s",
                     pair.first.c_str(), alias->second.c_str(), pair.second.c_str(), params->at(pair.second).c_str(),
                     pair.first.c_str(), alias->second.c_str());
      }
    }
  }
};

inline std::string ParseObjectiveAlias(const std::string& type) {
  if (type == std::string("regression") || type == std::string("regression_l2")
    || type == std::string("mean_squared_error") || type == std::string("mse") || type == std::string("l2")
    || type == std::string("l2_root") || type == std::string("root_mean_squared_error") || type == std::string("rmse")) {
    return "regression";
  } else if (type == std::string("regression_l1") || type == std::string("mean_absolute_error")
    || type == std::string("l1") || type == std::string("mae")) {
    return "regression_l1";
  } else if (type == std::string("multiclass") || type == std::string("softmax")) {
    return "multiclass";
  } else if (type == std::string("multiclassova") || type == std::string("multiclass_ova") || type == std::string("ova") || type == std::string("ovr")) {
    return "multiclassova";
  } else if (type == std::string("xentropy") || type == std::string("cross_entropy")) {
    return "cross_entropy";
  } else if (type == std::string("xentlambda") || type == std::string("cross_entropy_lambda")) {
    return "cross_entropy_lambda";
  } else if (type == std::string("mean_absolute_percentage_error") || type == std::string("mape")) {
    return "mape";
  } else if (type == std::string("rank_xendcg") || type == std::string("xendcg") || type == std::string("xe_ndcg")
             || type == std::string("xe_ndcg_mart") || type == std::string("xendcg_mart")) {
    return "rank_xendcg";
  } else if (type == std::string("none") || type == std::string("null") || type == std::string("custom") || type == std::string("na")) {
    return "custom";
  }
  return type;
}

inline std::string ParseMetricAlias(const std::string& type) {
  if (type == std::string("regression") || type == std::string("regression_l2") || type == std::string("l2") || type == std::string("mean_squared_error") || type == std::string("mse")) {
    return "l2";
  } else if (type == std::string("l2_root") || type == std::string("root_mean_squared_error") || type == std::string("rmse")) {
    return "rmse";
  } else if (type == std::string("regression_l1") || type == std::string("l1") || type == std::string("mean_absolute_error") || type == std::string("mae")) {
    return "l1";
  } else if (type == std::string("binary_logloss") || type == std::string("binary")) {
    return "binary_logloss";
  } else if (type == std::string("ndcg") || type == std::string("lambdarank") || type == std::string("rank_xendcg")
             || type == std::string("xendcg") || type == std::string("xe_ndcg") || type == std::string("xe_ndcg_mart") || type == std::string("xendcg_mart")) {
    return "ndcg";
  } else if (type == std::string("map") || type == std::string("mean_average_precision")) {
    return "map";
  } else if (type == std::string("multi_logloss") || type == std::string("multiclass") || type == std::string("softmax") || type == std::string("multiclassova") || type == std::string("multiclass_ova") || type == std::string("ova") || type == std::string("ovr")) {
    return "multi_logloss";
  } else if (type == std::string("xentropy") || type == std::string("cross_entropy")) {
    return "cross_entropy";
  } else if (type == std::string("xentlambda") || type == std::string("cross_entropy_lambda")) {
    return "cross_entropy_lambda";
  } else if (type == std::string("kldiv") || type == std::string("kullback_leibler")) {
    return "kullback_leibler";
  } else if (type == std::string("mean_absolute_percentage_error") || type == std::string("mape")) {
    return "mape";
  } else if (type == std::string("none") || type == std::string("null") || type == std::string("custom") || type == std::string("na")) {
    return "custom";
  }
  return type;
}

}   // namespace LightGBM

#endif   // LightGBM_CONFIG_H_
