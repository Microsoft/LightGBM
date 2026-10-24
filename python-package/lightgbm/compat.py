# coding: utf-8
"""Compatibility library."""

import inspect
from typing import Any, List

# scikit-learn is intentionally imported first here,
# see https://github.com/lightgbm-org/LightGBM/issues/6509
"""sklearn"""
try:
    
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import _check_sample_weight

    # As of https://github.com/scikit-learn/scikit-learn/pull/32212, scikit-learn started raising an error
    # when sample weights are all 0. This argument allow_all_zero_weights can be used switch back
    # to the old behavior of allowing them.
    #
    # This can be removed when the minimum scikit-learn version supported here is v1.9.
    SKLEARN_CHECK_SAMPLE_WEIGHT_HAS_ALLOW_ZERO_WEIGHTS_ARG = (
        "allow_all_zero_weights" in inspect.signature(_check_sample_weight).parameters
    )

    try:
        from sklearn.utils.validation import validate_data
    except ImportError:
        from sklearn.utils.validation import check_array, check_X_y

        # validate_data() was added in scikit-learn 1.6, this function roughly imitates it for older versions.
        # It can be removed when lightgbm's minimum scikit-learn version is at least 1.6.
        def validate_data(
            _estimator: Any,
            X: Any,
            y: Any = "no_validation",
            accept_sparse: bool = True,
            # 'force_all_finite' was renamed to 'ensure_all_finite' in scikit-learn 1.6
            ensure_all_finite: bool = False,
            ensure_min_samples: int = 1,
            # trap other keyword arguments that only work on scikit-learn >=1.6, like 'reset'
            **ignored_kwargs: Any,
        ) -> Any:
            # it's safe to import _num_features unconditionally because:
            #
            #  * it was first added in scikit-learn 0.24.2
            #  * lightgbm cannot be used with scikit-learn versions older than that
            #  * this validate_data() re-implementation will not be called in scikit-learn>=1.6
            #
            from sklearn.utils.validation import _num_features  # noqa: PLC0415

            # _num_features() raises a TypeError on 1-dimensional input. That's a problem
            # because scikit-learn's 'check_fit1d' estimator check sets that expectation that
            # estimators must raise a ValueError when a 1-dimensional input is passed to fit().
            #
            # So here, lightgbm avoids calling _num_features() on 1-dimensional inputs.
            if hasattr(X, "shape") and len(X.shape) == 1:
                n_features_in_ = 1
            else:
                n_features_in_ = _num_features(X)

            no_val_y = isinstance(y, str) and y == "no_validation"

            # NOTE: check_X_y() calls check_array() internally, so only need to call one or the other of them here
            if no_val_y:
                X = check_array(
                    X,
                    accept_sparse=accept_sparse,
                    force_all_finite=ensure_all_finite,
                    ensure_min_samples=ensure_min_samples,
                )
            else:
                X, y = check_X_y(
                    X,
                    y,
                    accept_sparse=accept_sparse,
                    force_all_finite=ensure_all_finite,
                    ensure_min_samples=ensure_min_samples,
                )

                # this only needs to be updated at fit() time
                _estimator.n_features_in_ = n_features_in_

            # raise the same error that scikit-learn's `validate_data()` does on scikit-learn>=1.6
            if _estimator.__sklearn_is_fitted__() and _estimator._n_features != n_features_in_:
                raise ValueError(
                    f"X has {n_features_in_} features, but {_estimator.__class__.__name__} "
                    f"is expecting {_estimator._n_features} features as input."
                )

            if no_val_y:
                return X
            else:
                return X, y

    SKLEARN_INSTALLED = True
    _LGBMModelBase = BaseEstimator
    _LGBMRegressorBase = RegressorMixin
    _LGBMClassifierBase = ClassifierMixin
    LGBMNotFittedError = NotFittedError
    _LGBMCheckSampleWeight = _check_sample_weight
    _LGBMValidateData = validate_data
except ImportError:
    SKLEARN_INSTALLED = False
    SKLEARN_CHECK_SAMPLE_WEIGHT_HAS_ALLOW_ZERO_WEIGHTS_ARG = False

    class _LGBMModelBase:  # type: ignore
        """Dummy class for sklearn.base.BaseEstimator."""

        pass

    class _LGBMClassifierBase:  # type: ignore
        """Dummy class for sklearn.base.ClassifierMixin."""

        pass

    class _LGBMRegressorBase:  # type: ignore
        """Dummy class for sklearn.base.RegressorMixin."""

        pass

    LGBMNotFittedError = ValueError
    _LGBMCheckSampleWeight = None
    _LGBMValidateData = None

"""pandas"""
try:
    from pandas import CategoricalDtype as pd_CategoricalDtype
    from pandas import DataFrame as pd_DataFrame
    from pandas import Series as pd_Series
    from pandas import concat

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

    class pd_Series:  # type: ignore
        """Dummy class for pandas.Series."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class pd_DataFrame:  # type: ignore
        """Dummy class for pandas.DataFrame."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class pd_CategoricalDtype:  # type: ignore
        """Dummy class for pandas.CategoricalDtype."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    concat = None

__all__: List[str] = []
