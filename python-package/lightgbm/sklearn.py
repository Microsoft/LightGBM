# coding: utf-8
"""Scikit-learn wrapper interface for LightGBM."""

import copy
import warnings
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import narwhals as nw
import narwhals.dependencies as nwd
import narwhals.typing as nwt
import numpy as np
import scipy.sparse

from .basic import (
    _MULTICLASS_OBJECTIVES,
    Booster,
    Dataset,
    LGBMDeprecationWarning,
    LightGBMError,
    _choose_param_value,
    _ConfigAliases,
    _LGBM_BoosterBestScoreType,
    _LGBM_CategoricalFeatureConfiguration,
    _LGBM_EvalFunctionResultType,
    _LGBM_FeatureNameConfiguration,
    _LGBM_GroupType,
    _LGBM_InitScoreType,
    _LGBM_LabelType,
    _LGBM_PredictReturnType,
    _LGBM_WeightType,
    _log_warning,
)
from .callback import _EvalResultDict, record_evaluation
from .compat import (
    SKLEARN_CHECK_SAMPLE_WEIGHT_HAS_ALLOW_ZERO_WEIGHTS_ARG,
    SKLEARN_INSTALLED,
    LGBMNotFittedError,
    _LGBMAssertAllFinite,
    _LGBMCheckClassificationTargets,
    _LGBMCheckSampleWeight,
    _LGBMClassifierBase,
    _LGBMComputeSampleWeight,
    _LGBMCpuCount,
    _LGBMLabelEncoder,
    _LGBMModelBase,
    _LGBMRegressorBase,
    _LGBMValidateData,
    _sklearn_version,
    pd_DataFrame,
)
from .engine import train

if TYPE_CHECKING:
    from .compat import _sklearn_Tags


__all__ = [
    "LGBMClassifier",
    "LGBMModel",
    "LGBMRanker",
    "LGBMRegressor",
]

_LGBM_ScikitMatrixLike = Union[
    List[Union[List[float], List[int]]],
    np.ndarray,
    pd_DataFrame,
    nwt.IntoDataFrame,
    scipy.sparse.spmatrix,
]
_LGBM_ScikitCustomObjectiveFunction = Union[
    # f(labels, preds)
    Callable[
        [Optional[np.ndarray], np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ],
    # f(labels, preds, weights)
    Callable[
        [Optional[np.ndarray], np.ndarray, Optional[np.ndarray]],
        Tuple[np.ndarray, np.ndarray],
    ],
    # f(labels, preds, weights, group)
    Callable[
        [Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
        Tuple[np.ndarray, np.ndarray],
    ],
]
_LGBM_ScikitCustomEvalFunction = Union[
    # f(labels, preds)
    Callable[
        [Optional[np.ndarray], np.ndarray],
        _LGBM_EvalFunctionResultType,
    ],
    Callable[
        [Optional[np.ndarray], np.ndarray],
        List[_LGBM_EvalFunctionResultType],
    ],
    # f(labels, preds, weights)
    Callable[
        [Optional[np.ndarray], np.ndarray, Optional[np.ndarray]],
        _LGBM_EvalFunctionResultType,
    ],
    Callable[
        [Optional[np.ndarray], np.ndarray, Optional[np.ndarray]],
        List[_LGBM_EvalFunctionResultType],
    ],
    # f(labels, preds, weights, group)
    Callable[
        [Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
        _LGBM_EvalFunctionResultType,
    ],
    Callable[
        [Optional[np.ndarray], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
        List[_LGBM_EvalFunctionResultType],
    ],
]
_LGBM_ScikitEvalMetricType = Union[
    str,
    _LGBM_ScikitCustomEvalFunction,
    List[Union[str, _LGBM_ScikitCustomEvalFunction]],
]
_LGBM_ScikitValidSet = Tuple[_LGBM_ScikitMatrixLike, _LGBM_LabelType]


def _get_group_from_constructed_dataset(dataset: Dataset) -> Optional[np.ndarray]:
    group = dataset.get_group()
    error_msg = (
        "Estimators in lightgbm.sklearn should only retrieve query groups from a constructed Dataset. "
        "If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/lightgbm-org/LightGBM/issues."
    )
    assert group is None or isinstance(group, np.ndarray), error_msg
    return group


def _get_label_from_constructed_dataset(dataset: Dataset) -> np.ndarray:
    label = dataset.get_label()
    error_msg = (
        "Estimators in lightgbm.sklearn should only retrieve labels from a constructed Dataset. "
        "If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/lightgbm-org/LightGBM/issues."
    )
    assert isinstance(label, np.ndarray), error_msg
    return label


def _get_weight_from_constructed_dataset(dataset: Dataset) -> Optional[np.ndarray]:
    weight = dataset.get_weight()
    error_msg = (
        "Estimators in lightgbm.sklearn should only retrieve weights from a constructed Dataset. "
        "If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/lightgbm-org/LightGBM/issues."
    )
    assert weight is None or isinstance(weight, np.ndarray), error_msg
    return weight


class _ObjectiveFunctionWrapper:
    """Proxy class for objective function."""

    def __init__(self, func: _LGBM_ScikitCustomObjectiveFunction):
        """Construct a proxy class.

        This class transforms objective function to match objective function with signature ``new_func(preds, dataset)``
        as expected by ``lightgbm.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with following signatures:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
            or ``func(y_true, y_pred, weight, group)``
            and returns (grad, hess):

                y_true : numpy 1-D array of shape = [n_samples]
                    The target values.
                y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                    The predicted values.
                    Predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task.
                weight : numpy 1-D array of shape = [n_samples]
                    The weight of samples. Weights should be non-negative.
                group : numpy 1-D array
                    Group/query data.
                    Only used in the learning-to-rank task.
                    sum(group) = n_samples.
                    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
                grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape [n_samples, n_classes] (for multi-class task)
                    The value of the first order derivative (gradient) of the loss
                    with respect to the elements of y_pred for each sample point.
                hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                    The value of the second order derivative (Hessian) of the loss
                    with respect to the elements of y_pred for each sample point.

        .. note::

            For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
            and grad and hess should be returned in the same format.
        """
        self.func = func

    def __call__(
        self,
        preds: np.ndarray,
        dataset: Dataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of preds for each sample point.
        hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of preds for each sample point.
        """
        labels = _get_label_from_constructed_dataset(dataset)
        argc = len(signature(self.func).parameters)
        if argc == 2:
            grad, hess = self.func(labels, preds)  # type: ignore[call-arg]
            return grad, hess

        weight = _get_weight_from_constructed_dataset(dataset)
        if argc == 3:
            grad, hess = self.func(labels, preds, weight)  # type: ignore[call-arg]
            return grad, hess

        if argc == 4:
            group = _get_group_from_constructed_dataset(dataset)
            return self.func(labels, preds, weight, group)  # type: ignore[call-arg]

        raise TypeError(f"Self-defined objective function should have 2, 3 or 4 arguments, got {argc}")


class _EvalFunctionWrapper:
    """Proxy class for evaluation function."""

    def __init__(self, func: _LGBM_ScikitCustomEvalFunction):
        """Construct a proxy class.

        This class transforms evaluation function to match evaluation function with signature ``new_func(preds, dataset)``
        as expected by ``lightgbm.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with following signatures:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
            or ``func(y_true, y_pred, weight, group)``
            and returns (metric_name, metric_value, maximize) or
            list of (metric_name, metric_value, maximize):

                y_true : numpy 1-D array of shape = [n_samples]
                    The target values.
                y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array shape = [n_samples, n_classes] (for multi-class task)
                    The predicted values.
                    In case of custom ``objective``, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                weight : numpy 1-D array of shape = [n_samples]
                    The weight of samples. Weights should be non-negative.
                group : numpy 1-D array
                    Group/query data.
                    Only used in the learning-to-rank task.
                    sum(group) = n_samples.
                    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
                metric_name : str
                    Unique identifier for the metric (e.g. "custom_adjusted_mse").
                metric_value : float
                    Value of the evaluation metric.
                maximize : bool
                    Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        self.func = func

    def __call__(
        self,
        preds: np.ndarray,
        dataset: Dataset,
    ) -> Union[_LGBM_EvalFunctionResultType, List[_LGBM_EvalFunctionResultType]]:
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        metric_name : str
            Unique identifier for the metric (e.g. "custom_adjusted_mse").
        metric_value : float
            Value of the evaluation metric.
        maximize : bool
            Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        labels = _get_label_from_constructed_dataset(dataset)
        argc = len(signature(self.func).parameters)
        if argc == 2:
            return self.func(labels, preds)  # type: ignore[call-arg]

        weight = _get_weight_from_constructed_dataset(dataset)
        if argc == 3:
            return self.func(labels, preds, weight)  # type: ignore[call-arg]

        if argc == 4:
            group = _get_group_from_constructed_dataset(dataset)
            return self.func(labels, preds, weight, group)  # type: ignore[call-arg]

        raise TypeError(f"Self-defined eval function should have 2, 3 or 4 arguments, got {argc}")


def _extract_evaluation_meta_data(
    *,
    collection: Optional[Union[Dict[Any, Any], List[Any]]],
    name: str,
    i: int,
) -> Optional[Any]:
    """Try to extract the ith element of one of the ``eval_*`` inputs."""
    if collection is None:
        return None
    elif isinstance(collection, list):
        # It's possible, for example, to pass 3 eval sets through `eval_set`,
        # but only 1 init_score through `eval_init_score`.
        #
        # This if-else accounts for that possibility.
        if len(collection) > i:
            return collection[i]
        else:
            return None
    elif isinstance(collection, dict):
        return collection.get(i, None)
    else:
        raise TypeError(f"{name} should be dict or list")


def _validate_eval_set_Xy(
    *,
    eval_set: Optional[List[_LGBM_ScikitValidSet]],
    eval_X: Optional[Union[_LGBM_ScikitMatrixLike, Tuple[_LGBM_ScikitMatrixLike]]],
    eval_y: Optional[Union[_LGBM_LabelType, Tuple[_LGBM_LabelType]]],
) -> Optional[List[_LGBM_ScikitValidSet]]:
    """Validate eval args.

    Returns
    -------
    eval_set
    """
    if eval_set is not None:
        msg = "The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead."
        warnings.warn(msg, category=LGBMDeprecationWarning, stacklevel=2)
        if eval_X is not None or eval_y is not None:
            raise ValueError("Specify either 'eval_set' or 'eval_X' and 'eval_y', but not both.")
        if isinstance(eval_set, tuple):
            return [eval_set]
        else:
            return eval_set
    if (eval_X is None) != (eval_y is None):
        raise ValueError("You must specify eval_X and eval_y, not just one of them.")
    if eval_set is None and eval_X is not None:
        if isinstance(eval_X, tuple) != isinstance(eval_y, tuple):
            raise ValueError("If eval_X is a tuple, y_val must be a tuple of same length, and vice versa.")
        if isinstance(eval_X, tuple) and isinstance(eval_y, tuple):
            if len(eval_X) != len(eval_y):
                raise ValueError("If eval_X is a tuple, y_val must be a tuple of same length, and vice versa.")
        if isinstance(eval_X, tuple) and isinstance(eval_y, tuple):
            eval_set = list(zip(eval_X, eval_y, strict=True))
        else:
            eval_set = [(eval_X, eval_y)]
    return eval_set


class LGBMModel(_LGBMModelBase):
    """Implementation of the scikit-learn API for LightGBM."""

    def __init__(
        self,
        *,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        **kwargs: Any,
    ):
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        if not SKLEARN_INSTALLED:
            raise LightGBMError(
                "scikit-learn is required for lightgbm.sklearn. "
                "You must install scikit-learn and restart your session to use this module."
            )

        self.boosting_type = boosting_type
        self.objective = objective
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self._Booster: Optional[Booster] = None
        self._evals_result: _EvalResultDict = {}
        self._best_score: _LGBM_BoosterBestScoreType = {}
        self._best_iteration: int = -1
        self._other_params: Dict[str, Any] = {}
        self._objective = objective
        self.class_weight = class_weight
        self._class_weight: Optional[Union[Dict, str]] = None
        self._class_map: Optional[Dict[int, int]] = None
        self._fitted_with_feature_names: bool = False
        self._n_features: int = -1
        self._n_features_in: int = -1
        self._classes: Optional[np.ndarray] = None
        self._n_classes: int = -1
        self.set_params(**kwargs)

    # scikit-learn 1.6 introduced an __sklearn__tags() method intended to replace _more_tags().
    # _more_tags() can be removed whenever lightgbm's minimum supported scikit-learn version
    # is >=1.6.
    # ref: https://github.com/lightgbm-org/LightGBM/pull/6651
    def _more_tags(self) -> Dict[str, Any]:
        check_sample_weight_str = (
            "In LightGBM, setting a sample's weight to 0 can produce a different result than omitting the sample. "
            "Such samples intentionally still affect count-based measures like 'min_data_in_leaf' "
            "(https://github.com/lightgbm-org/LightGBM/issues/5626#issuecomment-1712706678) and the estimated distribution "
            "of features for Dataset construction (see https://github.com/lightgbm-org/LightGBM/issues/5553)."
        )
        # "check_sample_weight_equivalence" can be removed when lightgbm's
        # minimum supported scikit-learn version is at least 1.6
        # ref: https://github.com/scikit-learn/scikit-learn/pull/30137
        xfail_checks = {
            "check_no_attributes_set_in_init": (
                "scikit-learn incorrectly asserts that private attributes "
                "cannot be set in __init__: "
                "(see https://github.com/lightgbm-org/LightGBM/issues/2628)"
            ),
            "check_all_zero_sample_weights_error": (
                "Beginning in scikit-learn 1.9, by default estimators are expected to reject "
                "sample weight arrays that are all-0. LightGBM intentionally accepts such arrays. "
                "LightGBM supports some operations where training on an all-0-weight input could make sense, "
                "like batch updates with training continuation or manual model creation with forced splits."
            ),
            "check_sample_weight_equivalence": check_sample_weight_str,
            "check_sample_weight_equivalence_on_dense_data": check_sample_weight_str,
            "check_sample_weight_equivalence_on_sparse_data": check_sample_weight_str,
        }
        # "check_decision_proba_consistency" can be removed when lightgbm's
        # minimum supported scikit-learn version is at least 1.2
        sklearn_major, sklearn_minor, *_ = _sklearn_version.split(".")
        if (int(sklearn_major), int(sklearn_minor)) < (1, 2):
            xfail_checks["check_decision_proba_consistency"] = (
                "decision_function() returns raw margins while predict_proba() applies sigmoid in C++ "
                "independently, causing different tie structures after rounding. "
                "scikit-learn >= 1.2 relaxed this check to accept monotonically consistent scores."
            )
        return {
            "allow_nan": True,
            "X_types": ["2darray", "sparse", "1dlabels"],
            "_xfail_checks": xfail_checks,
        }

    @staticmethod
    def _update_sklearn_tags_from_dict(
        *,
        tags: "_sklearn_Tags",
        tags_dict: Dict[str, Any],
    ) -> "_sklearn_Tags":
        """Update ``sklearn.utils.Tags`` inherited from ``scikit-learn`` base classes.

        ``scikit-learn`` 1.6 introduced a dataclass-based interface for estimator tags.
        ref: https://github.com/scikit-learn/scikit-learn/pull/29677

        This method handles updating that instance based on the value in ``self._more_tags()``.
        """
        tags.input_tags.allow_nan = tags_dict["allow_nan"]
        tags.input_tags.sparse = "sparse" in tags_dict["X_types"]
        tags.target_tags.one_d_labels = "1dlabels" in tags_dict["X_types"]
        return tags

    def __sklearn_tags__(self) -> Optional["_sklearn_Tags"]:
        # _LGBMModelBase.__sklearn_tags__() cannot be called unconditionally,
        # because that method isn't defined for scikit-learn<1.6
        if not hasattr(_LGBMModelBase, "__sklearn_tags__"):
            err_msg = (
                "__sklearn_tags__() should not be called when using scikit-learn<1.6. "
                f"Detected version: {_sklearn_version}"
            )
            raise AttributeError(err_msg)

        # take whatever tags are provided by BaseEstimator, then modify
        # them with LightGBM-specific values
        return self._update_sklearn_tags_from_dict(
            tags=super().__sklearn_tags__(),
            tags_dict=self._more_tags(),
        )

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "fitted_", False)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        # Based on: https://github.com/dmlc/xgboost/blob/bd92b1c9c0db3e75ec3dfa513e1435d518bb535d/python-package/xgboost/sklearn.py#L941
        # which was based on: https://stackoverflow.com/questions/59248211
        #
        # `get_params()` flows like this:
        #
        # 0. Get parameters in subclass (self.__class__) first, by using inspect.
        # 1. Get parameters in all parent classes (especially `LGBMModel`).
        # 2. Get whatever was passed via `**kwargs`.
        # 3. Merge them.
        #
        # This needs to accommodate being called recursively in the following
        # inheritance graphs (and similar for classification and ranking):
        #
        #   DaskLGBMRegressor -> LGBMRegressor     -> LGBMModel -> BaseEstimator
        #   (custom subclass) -> LGBMRegressor     -> LGBMModel -> BaseEstimator
        #                        LGBMRegressor     -> LGBMModel -> BaseEstimator
        #                        (custom subclass) -> LGBMModel -> BaseEstimator
        #                                             LGBMModel -> BaseEstimator
        #
        params = super().get_params(deep=deep)
        cp = copy.copy(self)
        # If the immediate parent defines get_params(), use that.
        if callable(getattr(cp.__class__.__bases__[0], "get_params", None)):
            cp.__class__ = cp.__class__.__bases__[0]
        # Otherwise, skip it and assume the next class will have it.
        # This is here primarily for cases where the first class in MRO is a scikit-learn mixin.
        else:
            cp.__class__ = cp.__class__.__bases__[1]
        params.update(cp.__class__.get_params(cp, deep))
        params.update(self._other_params)
        return params

    def set_params(self, **params: Any) -> "LGBMModel":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Parameter names with their new values.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
            self._other_params[key] = value
        return self

    def _process_params(self, stage: str) -> Dict[str, Any]:
        """Process the parameters of this estimator based on its type, parameter aliases, etc.

        Parameters
        ----------
        stage : str
            Name of the stage (can be ``fit`` or ``predict``) this method is called from.

        Returns
        -------
        processed_params : dict
            Processed parameter names mapped to their values.
        """
        assert stage in {"fit", "predict"}
        params = self.get_params()

        params.pop("objective", None)
        for alias in _ConfigAliases.get("objective"):
            if alias in params:
                obj = params.pop(alias)
                _log_warning(f"Found '{alias}' in params. Will use it instead of 'objective' argument")
                if stage == "fit":
                    self._objective = obj
        if stage == "fit":
            if self._objective is None:
                if isinstance(self, LGBMRegressor):
                    self._objective = "regression"
                elif isinstance(self, LGBMClassifier):
                    if self._n_classes > 2:
                        self._objective = "multiclass"
                    else:
                        self._objective = "binary"
                elif isinstance(self, LGBMRanker):
                    self._objective = "lambdarank"
                else:
                    raise ValueError("Unknown LGBMModel type.")
        if callable(self._objective):
            if stage == "fit":
                params["objective"] = _ObjectiveFunctionWrapper(self._objective)
            else:
                params["objective"] = "None"
        else:
            params["objective"] = self._objective

        params.pop("importance_type", None)
        params.pop("n_estimators", None)
        params.pop("class_weight", None)

        if isinstance(params["random_state"], np.random.RandomState):
            params["random_state"] = params["random_state"].randint(np.iinfo(np.int32).max)
        elif isinstance(params["random_state"], np.random.Generator):
            params["random_state"] = int(params["random_state"].integers(np.iinfo(np.int32).max))
        if self._n_classes > 2:
            for alias in _ConfigAliases.get("num_class"):
                params.pop(alias, None)
            params["num_class"] = self._n_classes
        if hasattr(self, "_eval_at"):
            eval_at = self._eval_at
            for alias in _ConfigAliases.get("eval_at"):
                if alias in params:
                    _log_warning(f"Found '{alias}' in params. Will use it instead of 'eval_at' argument")
                    eval_at = params.pop(alias)
            params["eval_at"] = eval_at

        # register default metric for consistency with callable eval_metric case
        original_metric = self._objective if isinstance(self._objective, str) else None
        if original_metric is None:
            # try to deduce from class instance
            if isinstance(self, LGBMRegressor):
                original_metric = "l2"
            elif isinstance(self, LGBMClassifier):
                original_metric = "multi_logloss" if self._n_classes > 2 else "binary_logloss"
            elif isinstance(self, LGBMRanker):
                original_metric = "ndcg"

        # overwrite default metric by explicitly set metric
        params = _choose_param_value("metric", params, original_metric)

        # use joblib conventions for negative n_jobs, just like scikit-learn
        # at predict time, this is handled later due to the order of parameter updates
        if stage == "fit":
            params = _choose_param_value("num_threads", params, self.n_jobs)
            params["num_threads"] = self._process_n_jobs(params["num_threads"])

        return params

    def _process_n_jobs(self, n_jobs: Optional[int]) -> int:
        """Convert special values of n_jobs to their actual values according to the formulas that apply.

        Parameters
        ----------
        n_jobs : int or None
            The original value of n_jobs, potentially having special values such as 'None' or
            negative integers.

        Returns
        -------
        n_jobs : int
            The value of n_jobs with special values converted to actual number of threads.
        """
        if n_jobs is None:
            n_jobs = _LGBMCpuCount(only_physical_cores=True)
        elif n_jobs < 0:
            n_jobs = max(_LGBMCpuCount(only_physical_cores=False) + 1 + n_jobs, 1)
        return n_jobs

    def fit(
        self,
        X: _LGBM_ScikitMatrixLike,
        y: _LGBM_LabelType,
        sample_weight: Optional[_LGBM_WeightType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        group: Optional[_LGBM_GroupType] = None,
        eval_set: Optional[List[_LGBM_ScikitValidSet]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_LGBM_WeightType]] = None,
        eval_class_weight: Optional[List[float]] = None,
        eval_init_score: Optional[List[_LGBM_InitScoreType]] = None,
        eval_group: Optional[List[_LGBM_GroupType]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        feature_name: _LGBM_FeatureNameConfiguration = "auto",
        categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[str, Path, Booster, "LGBMModel"]] = None,
        *,
        eval_X: Optional[Union[_LGBM_ScikitMatrixLike, Tuple[_LGBM_ScikitMatrixLike]]] = None,
        eval_y: Optional[Union[_LGBM_LabelType, Tuple[_LGBM_LabelType]]] = None,
    ) -> "LGBMModel":
        """
        Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : numpy array, pandas DataFrame, pyarrow Table, polars DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
            Input feature matrix.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        y : numpy array, pandas DataFrame, pandas Series, list of int or float, pyarrow ChunkedArray or polars Series of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        sample_weight : numpy array, pandas Series, list of int or float, pyarrow ChunkedArray, polars Series of shape = [n_samples] or None, optional (default=None)
            Weights of training data. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : numpy array, pandas DataFrame, pandas Series, list of int or float, list of lists, pyarrow ChunkedArray, pyarrow Table, polars Series, polars DataFrame of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task) or shape = [n_samples, n_classes] (for multi-class task) or None, optional (default=None)
            Init score of training data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        group : numpy array, pandas Series, pyarrow ChunkedArray, polars Series, list of int or float, or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        eval_set : list or None, optional (default=None)
            .. deprecated:: 4.7.0
                A list of (X, y) tuple pairs to use as validation sets.
                Use ``eval_X`` and ``eval_y`` instead.
        eval_names : list of str, or None, optional (default=None)
            Unique identifiers for each evaluation dataset.
            Should be the same length as ``eval_set`` / ``eval_X``.
        eval_sample_weight : list of array (same types as ``sample_weight`` supports), or None, optional (default=None)
            Weights of eval data. Weights should be non-negative.
        eval_class_weight : list or None, optional (default=None)
            Class weights of eval data.
        eval_init_score : list of array (same types as ``init_score`` supports), or None, optional (default=None)
            Init score of eval data.
        eval_group : list of array (same types as ``group`` supports), or None, optional (default=None)
            Group data of eval data.
        eval_metric : str, callable, list or None, optional (default=None)
            If str, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note below for more details.
            If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
        feature_name : list of str, or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        init_model : str, pathlib.Path, Booster, LGBMModel or None, optional (default=None)
            Filename of LightGBM model, Booster instance or LGBMModel instance used for continue training.
        eval_X : numpy array, pandas DataFrame, pyarrow Table, polars DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features], or tuple of such inputs, or None, optional (default=None)
            Feature matrix or tuple thereof, e.g. ``(X_val0, X_val1)``, to use as validation sets.

            .. versionadded:: 4.7.0

        eval_y : numpy array, pandas DataFrame, pandas Series, list of int or float, pyarrow ChunkedArray or polars Series of shape = [n_samples], or tuple of such inputs, or None, optional (default=None)
            Target values or tuple thereof, e.g. ``(y_val0, y_val1)``, to use as validation sets.

            .. versionadded:: 4.7.0

        Returns
        -------
        self : LGBMModel
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (metric_name, metric_value, maximize) or
        list of (metric_name, metric_value, maximize):

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                In case of custom ``objective``, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            metric_name : str
                Unique identifier for the metric (e.g. "custom_adjusted_mse").
            metric_value : float
                Value of the evaluation metric.
            maximize : bool
                Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        params = self._process_params(stage="fit")

        # Do not modify original args in fit function
        # Refer to https://github.com/lightgbm-org/LightGBM/pull/2619
        eval_metric_list: List[Union[str, _LGBM_ScikitCustomEvalFunction]]
        if eval_metric is None:
            eval_metric_list = []
        elif isinstance(eval_metric, list):
            eval_metric_list = copy.deepcopy(eval_metric)
        else:
            eval_metric_list = [copy.deepcopy(eval_metric)]

        # Separate built-in from callable evaluation metrics
        eval_metrics_callable = [_EvalFunctionWrapper(f) for f in eval_metric_list if callable(f)]
        eval_metrics_builtin = [m for m in eval_metric_list if isinstance(m, str)]

        # concatenate metric from params (or default if not provided in params) and eval_metric
        params["metric"] = [params["metric"]] if isinstance(params["metric"], (str, type(None))) else params["metric"]
        params["metric"] = [e for e in eval_metrics_builtin if e not in params["metric"]] + params["metric"]
        params["metric"] = [metric for metric in params["metric"] if metric is not None]

        if isinstance(X, pd_DataFrame):
            _X, _y = X, y
            self.n_features_in_ = _X.shape[1]
        elif nwd.is_into_dataframe(X):
            _X, _y = X, y
            self.n_features_in_ = nw.from_native(X).shape[1]
        else:
            # NOTE: _LGBMValidateData() is also responsible for setting n_features_in_
            _X, _y = _LGBMValidateData(
                self,
                X,
                y,
                reset=True,
                # allow any input type (this validation is done further down, in lgb.Dataset())
                accept_sparse=True,
                # do not raise an error if Inf of NaN values are found (LightGBM handles these internally)
                ensure_all_finite=False,
                # raise an error on 0-row and 1-row inputs
                ensure_min_samples=2,
            )
            if sample_weight is not None:
                if SKLEARN_CHECK_SAMPLE_WEIGHT_HAS_ALLOW_ZERO_WEIGHTS_ARG:
                    sample_weight = _LGBMCheckSampleWeight(sample_weight, _X, allow_all_zero_weights=True)
                else:
                    sample_weight = _LGBMCheckSampleWeight(sample_weight, _X)

        if self._class_weight is None:
            self._class_weight = self.class_weight
        if self._class_weight is not None:
            class_sample_weight = _LGBMComputeSampleWeight(self._class_weight, y)
            if sample_weight is None or len(sample_weight) == 0:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)  # type: ignore[arg-type]

        train_set = Dataset(
            data=_X,
            label=_y,
            weight=sample_weight,
            group=group,
            init_score=init_score,
            categorical_feature=categorical_feature,
            feature_name=feature_name,
            params=params,
        )

        valid_sets: List[Dataset] = []
        eval_set = _validate_eval_set_Xy(eval_set=eval_set, eval_X=eval_X, eval_y=eval_y)
        if eval_set is not None:
            # check eval_group (only relevant for ranking tasks)
            if eval_group is not None:
                if len(eval_group) != len(eval_set):
                    raise ValueError(
                        f"Length of eval_group ({len(eval_group)}) not equal to length of eval_set ({len(eval_set)})"
                    )

            for i, valid_data in enumerate(eval_set):
                # reduce cost for prediction training data
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = _extract_evaluation_meta_data(
                        collection=eval_sample_weight,
                        name="eval_sample_weight",
                        i=i,
                    )
                    valid_class_weight = _extract_evaluation_meta_data(
                        collection=eval_class_weight,
                        name="eval_class_weight",
                        i=i,
                    )
                    if valid_class_weight is not None:
                        if isinstance(valid_class_weight, dict) and self._class_map is not None:
                            valid_class_weight = {self._class_map[k]: v for k, v in valid_class_weight.items()}
                        valid_class_sample_weight = _LGBMComputeSampleWeight(valid_class_weight, valid_data[1])
                        if valid_weight is None or len(valid_weight) == 0:
                            valid_weight = valid_class_sample_weight
                        else:
                            valid_weight = np.multiply(valid_weight, valid_class_sample_weight)
                    valid_init_score = _extract_evaluation_meta_data(
                        collection=eval_init_score,
                        name="eval_init_score",
                        i=i,
                    )
                    valid_group = _extract_evaluation_meta_data(
                        collection=eval_group,
                        name="eval_group",
                        i=i,
                    )
                    valid_set = Dataset(
                        data=valid_data[0],
                        label=valid_data[1],
                        weight=valid_weight,
                        group=valid_group,
                        init_score=valid_init_score,
                        categorical_feature="auto",
                        params=params,
                    )

                valid_sets.append(valid_set)

        if isinstance(init_model, LGBMModel):
            init_model = init_model.booster_

        if callbacks is None:
            callbacks = []
        else:
            callbacks = copy.copy(callbacks)  # don't use deepcopy here to allow non-serializable objects

        evals_result: _EvalResultDict = {}
        callbacks.append(record_evaluation(evals_result))

        self._Booster = train(
            params=params,
            train_set=train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=eval_names,
            feval=eval_metrics_callable,  # type: ignore[arg-type]
            init_model=init_model,
            callbacks=callbacks,
        )

        # This populates the property self.n_features_, the number of features in the fitted model,
        # and so should only be set after fitting.
        #
        # The related property self._n_features_in, which populates self.n_features_in_,
        # is set BEFORE fitting.
        self._n_features = self._Booster.num_feature()

        # This attribute informs self.features_in_, but isn't set until here
        # because Dataset.construct(), called by train(), is responsible for updating it.
        self._fitted_with_feature_names = train_set._has_non_default_feature_names

        self._evals_result = evals_result
        self._best_iteration = self._Booster.best_iteration
        self._best_score = self._Booster.best_score

        self.fitted_ = True

        # free dataset
        self._Booster.free_dataset()
        del train_set, valid_sets
        return self

    def predict(
        self,
        X: _LGBM_ScikitMatrixLike,
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> _LGBM_PredictReturnType:
        """
        Return the predicted value for each sample.

        Parameters
        ----------
        X : numpy array, pandas DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_result : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        X_leaves : array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("Estimator not fitted, call fit before exploiting the model.")
        if not isinstance(X, pd_DataFrame) and not nwd.is_into_dataframe(X):
            X = _LGBMValidateData(
                self,
                X,
                # 'y' being omitted = run scikit-learn's check_array() instead of check_X_y()
                #
                # Prevent scikit-learn from deleting or modifying attributes like 'feature_names_in_' and 'n_features_in_'.
                # These shouldn't be changed at predict() time.
                reset=False,
                # allow any input type (this validation is done further down, in lgb.Dataset())
                accept_sparse=True,
                # do not raise an error if Inf of NaN values are found (LightGBM handles these internally)
                ensure_all_finite=False,
                # raise an error on 0-row inputs
                ensure_min_samples=1,
            )
        # retrieve original params that possibly can be used in both training and prediction
        # and then overwrite them (considering aliases) with params that were passed directly in prediction
        predict_params = self._process_params(stage="predict")
        for alias in _ConfigAliases.get_by_alias(
            "data",
            "X",
            "raw_score",
            "start_iteration",
            "num_iteration",
            "pred_leaf",
            "pred_contrib",
            *kwargs.keys(),
        ):
            predict_params.pop(alias, None)
        predict_params.update(kwargs)

        # number of threads can have values with special meaning which is only applied
        # in the scikit-learn interface, these should not reach the c++ side as-is
        predict_params = _choose_param_value("num_threads", predict_params, self.n_jobs)
        predict_params["num_threads"] = self._process_n_jobs(predict_params["num_threads"])

        return self._Booster.predict(  # type: ignore[union-attr]
            X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **predict_params,
        )

    @property
    def n_features_(self) -> int:
        """:obj:`int`: The number of features of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No n_features found. Need to call fit beforehand.")
        return self._n_features

    @property
    def n_features_in_(self) -> int:
        """:obj:`int`: The number of features of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No n_features_in found. Need to call fit beforehand.")
        return self._n_features_in

    @n_features_in_.setter
    def n_features_in_(self, value: int) -> None:
        """Set number of features found in passed-in dataset.

        Starting with ``scikit-learn`` 1.6, ``scikit-learn`` expects to be able to directly
        set this property in functions like ``validate_data()``.

        .. note::

            Do not call ``estimator.n_features_in_ = some_int`` or anything else that invokes
            this method. It is only here for compatibility with ``scikit-learn`` validation
            functions used internally in ``lightgbm``.
        """
        self._n_features_in = value

    @property
    def best_score_(self) -> _LGBM_BoosterBestScoreType:
        """:obj:`dict`: The best score of fitted model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No best_score found. Need to call fit beforehand.")
        return self._best_score

    @property
    def best_iteration_(self) -> int:
        """:obj:`int`: The best iteration of fitted model if ``early_stopping()`` callback has been specified."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError(
                "No best_iteration found. Need to call fit with early_stopping callback beforehand."
            )
        return self._best_iteration

    @property
    def objective_(self) -> Union[str, _LGBM_ScikitCustomObjectiveFunction]:
        """:obj:`str` or :obj:`callable`: The concrete objective used while fitting this model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No objective found. Need to call fit beforehand.")
        return self._objective  # type: ignore[return-value]

    @property
    def n_estimators_(self) -> int:
        """:obj:`int`: True number of boosting iterations performed.

        This might be less than parameter ``n_estimators`` if early stopping was enabled or
        if boosting stopped early due to limits on complexity like ``min_gain_to_split``.

        .. versionadded:: 4.0.0
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No n_estimators found. Need to call fit beforehand.")
        return self._Booster.current_iteration()  # type: ignore

    @property
    def n_iter_(self) -> int:
        """:obj:`int`: True number of boosting iterations performed.

        This might be less than parameter ``n_estimators`` if early stopping was enabled or
        if boosting stopped early due to limits on complexity like ``min_gain_to_split``.

        .. versionadded:: 4.0.0
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No n_iter found. Need to call fit beforehand.")
        return self._Booster.current_iteration()  # type: ignore

    @property
    def booster_(self) -> Booster:
        """Booster: The underlying Booster of this model."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No booster found. Need to call fit beforehand.")
        return self._Booster  # type: ignore[return-value]

    @property
    def evals_result_(self) -> _EvalResultDict:
        """:obj:`dict`: The evaluation results if validation sets have been specified."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No results found. Need to call fit with eval_set beforehand.")
        return self._evals_result

    @property
    def feature_importances_(self) -> np.ndarray:
        """:obj:`array` of shape = [n_features]: The feature importances (the higher, the more important).

        .. note::

            ``importance_type`` attribute is passed to the function
            to configure the type of importance values to be extracted.
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No feature_importances found. Need to call fit beforehand.")
        return self._Booster.feature_importance(importance_type=self.importance_type)  # type: ignore[union-attr]

    @property
    def feature_name_(self) -> List[str]:
        """:obj:`list` of shape = [n_features]: The names of features.

        .. note::

            If input does not contain feature names, they will be added during fitting in the format ``Column_0``, ``Column_1``, ..., ``Column_N``.
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No feature_name found. Need to call fit beforehand.")
        return self._Booster.feature_name()  # type: ignore[union-attr]

    @property
    def feature_names_in_(self) -> np.ndarray:
        """:obj:`array` of shape = [n_features]: scikit-learn compatible version of ``.feature_name_``.

        Only available when training data had feature names (e.g. a pandas DataFrame).
        When training was done with data without feature names (e.g. a numpy array),
        accessing this attribute raises ``AttributeError``.

        .. versionadded:: 4.5.0
        """
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No feature_names_in_ found. Need to call fit beforehand.")
        if not self._fitted_with_feature_names:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute 'feature_names_in_'. "
                "The training data did not have feature names "
                "(e.g. was a numpy array rather than a pandas DataFrame)."
            )
        return np.array(self.feature_name_)

    @feature_names_in_.deleter
    def feature_names_in_(self) -> None:
        """Intercept calls to delete ``feature_names_in_``.

        Some code paths in ``scikit-learn`` try to delete the ``feature_names_in_`` attribute
        on estimators when a new training dataset that doesn't have features is passed.
        LightGBM has custom handling of feature names and has chosen to opt out of this behavior.

        However, that behavior is coupled to ``scikit-learn`` automatically updating
        ``n_features_in_`` in those same code paths, which is necessary for compliance
        with its API (via argument ``reset`` to functions like ``validate_data()`` and
        ``check_array()``).

        .. note::

            Do not call ``del estimator.feature_names_in_`` or anything else that invokes
            this method. It is only here for compatibility with ``scikit-learn`` validation
            functions used internally in ``lightgbm``.
        """
        pass


class LGBMRegressor(_LGBMRegressorBase, LGBMModel):
    """LightGBM regressor."""

    # NOTE: all args from LGBMModel.__init__() are intentionally repeated here for
    #       docs, help(), and tab completion.
    def __init__(
        self,
        *,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        **kwargs: Any,
    ) -> None:
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs,
        )

    def _more_tags(self) -> Dict[str, Any]:
        # handle the case where RegressorMixin possibly provides _more_tags()
        if callable(getattr(_LGBMRegressorBase, "_more_tags", None)):
            tags = _LGBMRegressorBase._more_tags(self)
        else:
            tags = {}
        # override those with LightGBM-specific preferences
        tags.update(LGBMModel._more_tags(self))
        return tags

    def __sklearn_tags__(self) -> "_sklearn_Tags":
        return super().__sklearn_tags__()

    def fit(  # type: ignore[override]
        self,
        X: _LGBM_ScikitMatrixLike,
        y: _LGBM_LabelType,
        sample_weight: Optional[_LGBM_WeightType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        eval_set: Optional[List[_LGBM_ScikitValidSet]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_LGBM_WeightType]] = None,
        eval_init_score: Optional[List[_LGBM_InitScoreType]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        feature_name: _LGBM_FeatureNameConfiguration = "auto",
        categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[str, Path, Booster, LGBMModel]] = None,
        *,
        eval_X: Optional[Union[_LGBM_ScikitMatrixLike, Tuple[_LGBM_ScikitMatrixLike]]] = None,
        eval_y: Optional[Union[_LGBM_LabelType, Tuple[_LGBM_LabelType]]] = None,
    ) -> "LGBMRegressor":
        """
        Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : numpy array, pandas DataFrame, pyarrow Table, polars DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
            Input feature matrix.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        y : numpy array, pandas DataFrame, pandas Series, list of int or float, pyarrow ChunkedArray or polars Series of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        sample_weight : numpy array, pandas Series, list of int or float, pyarrow ChunkedArray, polars Series of shape = [n_samples] or None, optional (default=None)
            Weights of training data. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : numpy array, pandas DataFrame, pandas Series, list of int or float, list of lists, pyarrow ChunkedArray, pyarrow Table, polars Series, polars DataFrame of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task) or shape = [n_samples, n_classes] (for multi-class task) or None, optional (default=None)
            Init score of training data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        eval_set : list or None, optional (default=None)
            .. deprecated:: 4.7.0
                A list of (X, y) tuple pairs to use as validation sets.
                Use ``eval_X`` and ``eval_y`` instead.
        eval_names : list of str, or None, optional (default=None)
            Unique identifiers for each evaluation dataset.
            Should be the same length as ``eval_set`` / ``eval_X``.
        eval_sample_weight : list of array (same types as ``sample_weight`` supports), or None, optional (default=None)
            Weights of eval data. Weights should be non-negative.
        eval_init_score : list of array (same types as ``init_score`` supports), or None, optional (default=None)
            Init score of eval data.
        eval_metric : str, callable, list or None, optional (default=None)
            If str, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note below for more details.
            If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
        feature_name : list of str, or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        init_model : str, pathlib.Path, Booster, LGBMModel or None, optional (default=None)
            Filename of LightGBM model, Booster instance or LGBMModel instance used for continue training.
        eval_X : numpy array, pandas DataFrame, pyarrow Table, polars DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features], or tuple of such inputs, or None, optional (default=None)
            Feature matrix or tuple thereof, e.g. ``(X_val0, X_val1)``, to use as validation sets.

            .. versionadded:: 4.7.0

        eval_y : numpy array, pandas DataFrame, pandas Series, list of int or float, pyarrow ChunkedArray or polars Series of shape = [n_samples], or tuple of such inputs, or None, optional (default=None)
            Target values or tuple thereof, e.g. ``(y_val0, y_val1)``, to use as validation sets.

            .. versionadded:: 4.7.0

        Returns
        -------
        self : LGBMRegressor
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (metric_name, metric_value, maximize) or
        list of (metric_name, metric_value, maximize):

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                In case of custom ``objective``, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            metric_name : str
                Unique identifier for the metric (e.g. "custom_adjusted_mse").
            metric_value : float
                Value of the evaluation metric.
            maximize : bool
                Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_X=eval_X,
            eval_y=eval_y,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )
        return self


class LGBMClassifier(_LGBMClassifierBase, LGBMModel):
    """LightGBM classifier."""

    # NOTE: all args from LGBMModel.__init__() are intentionally repeated here for
    #       docs, help(), and tab completion.
    def __init__(
        self,
        *,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        **kwargs: Any,
    ) -> None:
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs,
        )

    def _more_tags(self) -> Dict[str, Any]:
        # handle the case where ClassifierMixin possibly provides _more_tags()
        if callable(getattr(_LGBMClassifierBase, "_more_tags", None)):
            tags = _LGBMClassifierBase._more_tags(self)
        else:
            tags = {}
        # override those with LightGBM-specific preferences
        tags.update(LGBMModel._more_tags(self))
        return tags

    def __sklearn_tags__(self) -> "_sklearn_Tags":
        tags = super().__sklearn_tags__()
        if tags is not None:
            tags.classifier_tags.multi_class = True
            tags.classifier_tags.multi_label = False
        return tags

    def fit(  # type: ignore[override]
        self,
        X: _LGBM_ScikitMatrixLike,
        y: _LGBM_LabelType,
        sample_weight: Optional[_LGBM_WeightType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        eval_set: Optional[List[_LGBM_ScikitValidSet]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_LGBM_WeightType]] = None,
        eval_class_weight: Optional[List[float]] = None,
        eval_init_score: Optional[List[_LGBM_InitScoreType]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        feature_name: _LGBM_FeatureNameConfiguration = "auto",
        categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[str, Path, Booster, LGBMModel]] = None,
        *,
        eval_X: Optional[Union[_LGBM_ScikitMatrixLike, Tuple[_LGBM_ScikitMatrixLike]]] = None,
        eval_y: Optional[Union[_LGBM_LabelType, Tuple[_LGBM_LabelType]]] = None,
    ) -> "LGBMClassifier":
        """
        Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : numpy array, pandas DataFrame, pyarrow Table, polars DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
            Input feature matrix.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        y : numpy array, pandas DataFrame, pandas Series, list of int or float, pyarrow ChunkedArray or polars Series of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        sample_weight : numpy array, pandas Series, list of int or float, pyarrow ChunkedArray, polars Series of shape = [n_samples] or None, optional (default=None)
            Weights of training data. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : numpy array, pandas DataFrame, pandas Series, list of int or float, list of lists, pyarrow ChunkedArray, pyarrow Table, polars Series, polars DataFrame of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task) or shape = [n_samples, n_classes] (for multi-class task) or None, optional (default=None)
            Init score of training data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        eval_set : list or None, optional (default=None)
            .. deprecated:: 4.7.0
                A list of (X, y) tuple pairs to use as validation sets.
                Use ``eval_X`` and ``eval_y`` instead.
        eval_names : list of str, or None, optional (default=None)
            Unique identifiers for each evaluation dataset.
            Should be the same length as ``eval_set`` / ``eval_X``.
        eval_sample_weight : list of array (same types as ``sample_weight`` supports), or None, optional (default=None)
            Weights of eval data. Weights should be non-negative.
        eval_class_weight : list or None, optional (default=None)
            Class weights of eval data.
        eval_init_score : list of array (same types as ``init_score`` supports), or None, optional (default=None)
            Init score of eval data.
        eval_metric : str, callable, list or None, optional (default=None)
            If str, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note below for more details.
            If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
        feature_name : list of str, or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        init_model : str, pathlib.Path, Booster, LGBMModel or None, optional (default=None)
            Filename of LightGBM model, Booster instance or LGBMModel instance used for continue training.
        eval_X : numpy array, pandas DataFrame, pyarrow Table, polars DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features], or tuple of such inputs, or None, optional (default=None)
            Feature matrix or tuple thereof, e.g. ``(X_val0, X_val1)``, to use as validation sets.

            .. versionadded:: 4.7.0

        eval_y : numpy array, pandas DataFrame, pandas Series, list of int or float, pyarrow ChunkedArray or polars Series of shape = [n_samples], or tuple of such inputs, or None, optional (default=None)
            Target values or tuple thereof, e.g. ``(y_val0, y_val1)``, to use as validation sets.

            .. versionadded:: 4.7.0

        Returns
        -------
        self : LGBMClassifier
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (metric_name, metric_value, maximize) or
        list of (metric_name, metric_value, maximize):

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                In case of custom ``objective``, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            metric_name : str
                Unique identifier for the metric (e.g. "custom_adjusted_mse").
            metric_value : float
                Value of the evaluation metric.
            maximize : bool
                Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        _LGBMAssertAllFinite(y)
        _LGBMCheckClassificationTargets(y)
        self._le = _LGBMLabelEncoder().fit(y)
        _y = self._le.transform(y)
        self._class_map = dict(zip(self._le.classes_, self._le.transform(self._le.classes_), strict=True))
        if isinstance(self.class_weight, dict):
            self._class_weight = {self._class_map[k]: v for k, v in self.class_weight.items()}

        self._classes = self._le.classes_
        self._n_classes = len(self._classes)  # type: ignore[arg-type]
        if self.objective is None:
            self._objective = None

        # adjust eval metrics to match whether binary or multiclass
        # classification is being performed
        if not callable(eval_metric):
            if isinstance(eval_metric, list):
                eval_metric_list = eval_metric
            elif isinstance(eval_metric, str):
                eval_metric_list = [eval_metric]
            else:
                eval_metric_list = []
            if self.__is_multiclass:
                for index, metric in enumerate(eval_metric_list):
                    if metric in {"logloss", "binary_logloss"}:
                        eval_metric_list[index] = "multi_logloss"
                    elif metric in {"error", "binary_error"}:
                        eval_metric_list[index] = "multi_error"
            else:
                for index, metric in enumerate(eval_metric_list):
                    if metric in {"logloss", "multi_logloss"}:
                        eval_metric_list[index] = "binary_logloss"
                    elif metric in {"error", "multi_error"}:
                        eval_metric_list[index] = "binary_error"
            eval_metric = eval_metric_list

        # do not modify args, as it causes errors in model selection tools
        valid_sets: Optional[List[_LGBM_ScikitValidSet]] = None
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            valid_sets = []
            for valid_x, valid_y in eval_set:
                if valid_x is X and valid_y is y:
                    valid_sets.append((valid_x, _y))
                else:
                    valid_sets.append((valid_x, self._le.transform(valid_y)))

        if eval_y is not None:
            if isinstance(eval_y, tuple):
                eval_y = tuple(self._le.transform(valid_y) for valid_y in eval_y)
            else:
                eval_y = self._le.transform(eval_y)

        super().fit(
            X,
            _y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=valid_sets,
            eval_names=eval_names,
            eval_X=eval_X,
            eval_y=eval_y,
            eval_sample_weight=eval_sample_weight,
            eval_class_weight=eval_class_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )
        return self

    def predict(
        self,
        X: _LGBM_ScikitMatrixLike,
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> _LGBM_PredictReturnType:
        """
        Return the predicted value for each sample.

        Parameters
        ----------
        X : numpy array, pandas DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_result : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        X_leaves : array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        result = self.predict_proba(
            X=X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )
        if callable(self._objective) or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)

    def predict_proba(
        self,
        X: _LGBM_ScikitMatrixLike,
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> _LGBM_PredictReturnType:
        """
        Return the predicted probability for each class for each sample.

        Parameters
        ----------
        X : numpy array, pandas DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_probability : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        X_leaves : array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        result = super().predict(
            X=X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )
        if callable(self._objective) and not (raw_score or pred_leaf or pred_contrib):
            _log_warning(
                "Cannot compute class probabilities or labels "
                "due to the usage of customized objective function.\n"
                "Returning raw scores instead."
            )
            return result
        elif self.__is_multiclass or raw_score or pred_leaf or pred_contrib:  # type: ignore [operator]
            return result
        else:
            error_msg = (
                "predict() should return np.ndarray when pred_contrib=False. "
                "If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/lightgbm-org/LightGBM/issues."
            )
            assert isinstance(result, np.ndarray), error_msg
            return np.vstack((1.0 - result, result)).transpose()

    def decision_function(
        self,
        X: _LGBM_ScikitMatrixLike,
        *,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> _LGBM_PredictReturnType:
        """Return the raw margin score for each sample.

        .. versionadded:: 4.7.0

        Parameters
        ----------
        X : numpy array, pandas DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
            Input features matrix.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters forwarded to ``predict()``.

        Returns
        -------
        raw_score : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        """
        return super().predict(
            X=X,
            raw_score=True,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            validate_features=validate_features,
            **kwargs,
        )

    @property
    def classes_(self) -> np.ndarray:
        """:obj:`array` of shape = [n_classes]: The class label array."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No classes found. Need to call fit beforehand.")
        return self._classes  # type: ignore[return-value]

    @property
    def n_classes_(self) -> int:
        """:obj:`int`: The number of classes."""
        if not self.__sklearn_is_fitted__():
            raise LGBMNotFittedError("No classes found. Need to call fit beforehand.")
        return self._n_classes

    @property
    def __is_multiclass(self) -> bool:
        """:obj:`bool`:  Indicator of whether the classifier is used for multiclass."""
        return self._n_classes > 2 or (isinstance(self._objective, str) and self._objective in _MULTICLASS_OBJECTIVES)


class LGBMRanker(LGBMModel):
    """LightGBM ranker.

    .. warning::

        scikit-learn doesn't support ranking applications yet,
        therefore this class is not really compatible with the sklearn ecosystem.
        Please use this class mainly for training and applying ranking models in common sklearnish way.
    """

    # NOTE: all args from LGBMModel.__init__() are intentionally repeated here for
    #       docs, help(), and tab completion.
    def __init__(
        self,
        *,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        **kwargs: Any,
    ) -> None:
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs,
        )

    def fit(  # type: ignore[override]
        self,
        X: _LGBM_ScikitMatrixLike,
        y: _LGBM_LabelType,
        sample_weight: Optional[_LGBM_WeightType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        group: Optional[_LGBM_GroupType] = None,
        eval_set: Optional[List[_LGBM_ScikitValidSet]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_LGBM_WeightType]] = None,
        eval_init_score: Optional[List[_LGBM_InitScoreType]] = None,
        eval_group: Optional[List[_LGBM_GroupType]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        eval_at: Union[List[int], Tuple[int, ...]] = (1, 2, 3, 4, 5),
        feature_name: _LGBM_FeatureNameConfiguration = "auto",
        categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[str, Path, Booster, LGBMModel]] = None,
        *,
        eval_X: Optional[Union[_LGBM_ScikitMatrixLike, Tuple[_LGBM_ScikitMatrixLike]]] = None,
        eval_y: Optional[Union[_LGBM_LabelType, Tuple[_LGBM_LabelType]]] = None,
    ) -> "LGBMRanker":
        """
        Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : numpy array, pandas DataFrame, pyarrow Table, polars DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
            Input feature matrix.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        y : numpy array, pandas DataFrame, pandas Series, list of int or float, pyarrow ChunkedArray or polars Series of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        sample_weight : numpy array, pandas Series, list of int or float, pyarrow ChunkedArray, polars Series of shape = [n_samples] or None, optional (default=None)
            Weights of training data. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : numpy array, pandas DataFrame, pandas Series, list of int or float, list of lists, pyarrow ChunkedArray, pyarrow Table, polars Series, polars DataFrame of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task) or shape = [n_samples, n_classes] (for multi-class task) or None, optional (default=None)
            Init score of training data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        group : numpy array, pandas Series, pyarrow ChunkedArray, polars Series, list of int or float, or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        eval_set : list or None, optional (default=None)
            .. deprecated:: 4.7.0
                A list of (X, y) tuple pairs to use as validation sets.
                Use ``eval_X`` and ``eval_y`` instead.
        eval_names : list of str, or None, optional (default=None)
            Unique identifiers for each evaluation dataset.
            Should be the same length as ``eval_set`` / ``eval_X``.
        eval_sample_weight : list of array (same types as ``sample_weight`` supports), or None, optional (default=None)
            Weights of eval data. Weights should be non-negative.
        eval_init_score : list of array (same types as ``init_score`` supports), or None, optional (default=None)
            Init score of eval data.
        eval_group : list of array (same types as ``group`` supports), or None, optional (default=None)
            Group data of eval data.
        eval_metric : str, callable, list or None, optional (default=None)
            If str, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note below for more details.
            If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
        eval_at : list or tuple of int, optional (default=(1, 2, 3, 4, 5))
            The evaluation positions of the specified metric.
        feature_name : list of str, or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        init_model : str, pathlib.Path, Booster, LGBMModel or None, optional (default=None)
            Filename of LightGBM model, Booster instance or LGBMModel instance used for continue training.
        eval_X : numpy array, pandas DataFrame, pyarrow Table, polars DataFrame, scipy.sparse, list of lists of int or float of shape = [n_samples, n_features], or tuple of such inputs, or None, optional (default=None)
            Feature matrix or tuple thereof, e.g. ``(X_val0, X_val1)``, to use as validation sets.

            .. versionadded:: 4.7.0

        eval_y : numpy array, pandas DataFrame, pandas Series, list of int or float, pyarrow ChunkedArray or polars Series of shape = [n_samples], or tuple of such inputs, or None, optional (default=None)
            Target values or tuple thereof, e.g. ``(y_val0, y_val1)``, to use as validation sets.

            .. versionadded:: 4.7.0

        Returns
        -------
        self : LGBMRanker
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (metric_name, metric_value, maximize) or
        list of (metric_name, metric_value, maximize):

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                In case of custom ``objective``, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            metric_name : str
                Unique identifier for the metric (e.g. "custom_adjusted_mse").
            metric_value : float
                Value of the evaluation metric.
            maximize : bool
                Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        # check group data
        if group is None:
            raise ValueError("Should set group for ranking task")

        if eval_group is None and (eval_set is not None or eval_X is not None or eval_y is not None):
            raise ValueError("eval_group cannot be None if any of eval_set, eval_X, or eval_y are provided")

        self._eval_at = eval_at
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            group=group,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_X=eval_X,
            eval_y=eval_y,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_group=eval_group,
            eval_metric=eval_metric,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )
        return self
