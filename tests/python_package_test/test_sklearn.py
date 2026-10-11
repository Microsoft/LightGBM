# coding: utf-8
import inspect
import itertools
import math
import re
import sys
import textwrap
import warnings
from functools import partial
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pytest
import scipy.sparse
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_svmlight_file, make_blobs, make_multilabel_classification
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier, MultiOutputRegressor, RegressorChain
from sklearn.utils.estimator_checks import parametrize_with_checks as sklearn_parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

import lightgbm as lgb
from lightgbm.basic import LGBMDeprecationWarning
from lightgbm.compat import (
    PANDAS_INSTALLED,
    _sklearn_version,
    pd_DataFrame,
    pd_Series,
)

from .utils import (
    BuildInfo,
    assert_docstrings_equal,
    assert_silent,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_linnerud,
    logistic_sigmoid,
    make_ranking,
    make_synthetic_regression,
    np_assert_array_equal,
    sklearn_multiclass_custom_objective,
    softmax,
)

SKLEARN_MAJOR, SKLEARN_MINOR, *_ = _sklearn_version.split(".")
SKLEARN_VERSION_GTE_1_6 = (int(SKLEARN_MAJOR), int(SKLEARN_MINOR)) >= (1, 6)
SKLEARN_VERSION_GTE_1_7 = (int(SKLEARN_MAJOR), int(SKLEARN_MINOR)) >= (1, 7)

decreasing_generator = itertools.count(0, -1)
estimator_classes = (lgb.LGBMModel, lgb.LGBMClassifier, lgb.LGBMRegressor, lgb.LGBMRanker)
task_to_model_factory = {
    "ranking": lgb.LGBMRanker,
    "binary-classification": lgb.LGBMClassifier,
    "multiclass-classification": lgb.LGBMClassifier,
    "regression": lgb.LGBMRegressor,
}
all_tasks = tuple(task_to_model_factory.keys())
all_x_types = ("list2d", "numpy", "pd_DataFrame", "pa_Table", "pl_DataFrame", "scipy_csc", "scipy_csr")
all_y_types = ("list1d", "numpy", "pd_Series", "pd_DataFrame", "pa_ChunkedArray", "pl_Series")
all_group_types = ("list1d_float", "list1d_int", "numpy", "pd_Series", "pa_ChunkedArray", "pl_Series")


def _create_data(task, n_samples=100, n_features=4):
    if task == "ranking":
        X, y, g = make_ranking(n_features=4, n_samples=n_samples)
        g = np.bincount(g)
    elif task.endswith("classification"):
        if task == "binary-classification":
            centers = 2
        elif task == "multiclass-classification":
            centers = 3
        else:
            raise ValueError(f"Unknown classification task '{task}'")
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)
        g = None
    elif task == "regression":
        X, y = make_synthetic_regression(n_samples=n_samples, n_features=n_features)
        g = None
    return X, y, g


class UnpicklableCallback:
    def __reduce__(self):
        raise Exception("This class in not picklable")

    def __call__(self, env):
        env.model.attr_set_inside_callback = env.iteration * 10


class ExtendedLGBMClassifier(lgb.LGBMClassifier):
    """Class for testing that inheriting from LGBMClassifier works"""

    def __init__(self, *, some_other_param: str = "lgbm-classifier", **kwargs):
        self.some_other_param = some_other_param
        super().__init__(**kwargs)


class ExtendedLGBMRanker(lgb.LGBMRanker):
    """Class for testing that inheriting from LGBMRanker works"""

    def __init__(self, *, some_other_param: str = "lgbm-ranker", **kwargs):
        self.some_other_param = some_other_param
        super().__init__(**kwargs)


class ExtendedLGBMRegressor(lgb.LGBMRegressor):
    """Class for testing that inheriting from LGBMRegressor works"""

    def __init__(self, *, some_other_param: str = "lgbm-regressor", **kwargs):
        self.some_other_param = some_other_param
        super().__init__(**kwargs)


def custom_asymmetric_obj(y_true, y_pred):
    residual = (y_true - y_pred).astype(np.float64)
    grad = np.where(residual < 0, -2 * 10.0 * residual, -2 * residual)
    hess = np.where(residual < 0, 2 * 10.0, 2.0)
    return grad, hess


def objective_ls(y_true, y_pred):
    grad = y_pred - y_true
    hess = np.ones(len(y_true))
    return grad, hess


def logregobj(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess


def custom_dummy_obj(y_true, y_pred):
    return np.ones(y_true.shape), np.ones(y_true.shape)


def constant_metric(y_true, y_pred):
    return "error", 0, False


def decreasing_metric(y_true, y_pred):
    return ("decreasing_metric", next(decreasing_generator), False)


def mse(y_true, y_pred):
    return "custom MSE", mean_squared_error(y_true, y_pred), False


def binary_error(y_true, y_pred):
    return np.mean((y_pred > 0.5) != y_true)


def multi_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


def test_binary():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    ret = log_loss(y_test, gbm.predict_proba(X_test))
    assert ret < 0.12
    assert gbm.evals_result_["valid_0"]["binary_logloss"][gbm.best_iteration_ - 1] == pytest.approx(ret)


def test_regression():
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(n_estimators=50, verbose=-1)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    ret = mean_squared_error(y_test, gbm.predict(X_test))
    assert ret < 174
    assert gbm.evals_result_["valid_0"]["l2"][gbm.best_iteration_ - 1] == pytest.approx(ret)


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Skip due to differences in implementation details of CUDA version")
def test_multiclass():
    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    ret = multi_error(y_test, gbm.predict(X_test))
    assert ret < 0.05
    ret = multi_logloss(y_test, gbm.predict_proba(X_test))
    assert ret < 0.16
    assert gbm.evals_result_["valid_0"]["multi_logloss"][gbm.best_iteration_ - 1] == pytest.approx(ret)


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Skip due to differences in implementation details of CUDA version")
def test_lambdarank():
    rank_example_dir = Path(__file__).absolute().parents[2] / "examples" / "lambdarank"
    X_train, y_train = load_svmlight_file(str(rank_example_dir / "rank.train"))
    X_test, y_test = load_svmlight_file(str(rank_example_dir / "rank.test"))
    q_train = np.loadtxt(str(rank_example_dir / "rank.train.query"))
    q_test = np.loadtxt(str(rank_example_dir / "rank.test.query"))
    gbm = lgb.LGBMRanker(n_estimators=50)
    gbm.fit(
        X_train,
        y_train,
        group=q_train,
        eval_set=[(X_test, y_test)],
        eval_group=[q_test],
        eval_at=[1, 3],
        callbacks=[lgb.early_stopping(10), lgb.reset_parameter(learning_rate=lambda x: max(0.01, 0.1 - 0.01 * x))],
    )
    assert gbm.best_iteration_ <= 24
    assert gbm.best_score_["valid_0"]["ndcg@1"] > 0.5674
    assert gbm.best_score_["valid_0"]["ndcg@3"] > 0.578


def test_xendcg():
    xendcg_example_dir = Path(__file__).absolute().parents[2] / "examples" / "xendcg"
    X_train, y_train = load_svmlight_file(str(xendcg_example_dir / "rank.train"))
    X_test, y_test = load_svmlight_file(str(xendcg_example_dir / "rank.test"))
    q_train = np.loadtxt(str(xendcg_example_dir / "rank.train.query"))
    q_test = np.loadtxt(str(xendcg_example_dir / "rank.test.query"))
    gbm = lgb.LGBMRanker(n_estimators=50, objective="rank_xendcg", random_state=5, n_jobs=1)
    gbm.fit(
        X_train,
        y_train,
        group=q_train,
        eval_set=[(X_test, y_test)],
        eval_group=[q_test],
        eval_at=[1, 3],
        eval_metric="ndcg",
        callbacks=[lgb.early_stopping(10), lgb.reset_parameter(learning_rate=lambda x: max(0.01, 0.1 - 0.01 * x))],
    )
    assert gbm.best_iteration_ <= 24
    assert gbm.best_score_["valid_0"]["ndcg@1"] > 0.6211
    assert gbm.best_score_["valid_0"]["ndcg@3"] > 0.6253


def test_eval_at_aliases():
    rank_example_dir = Path(__file__).absolute().parents[2] / "examples" / "lambdarank"
    X_train, y_train = load_svmlight_file(str(rank_example_dir / "rank.train"))
    X_test, y_test = load_svmlight_file(str(rank_example_dir / "rank.test"))
    q_train = np.loadtxt(str(rank_example_dir / "rank.train.query"))
    q_test = np.loadtxt(str(rank_example_dir / "rank.test.query"))
    for alias in lgb.basic._ConfigAliases.get("eval_at"):
        gbm = lgb.LGBMRanker(n_estimators=5, **{alias: [1, 2, 3, 9]})
        with pytest.warns(UserWarning, match=f"Found '{alias}' in params. Will use it instead of 'eval_at' argument"):
            gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_test, y_test)], eval_group=[q_test])
        assert list(gbm.evals_result_["valid_0"].keys()) == ["ndcg@1", "ndcg@2", "ndcg@3", "ndcg@9"]


@pytest.mark.parametrize("custom_objective", [True, False])
def test_objective_aliases(custom_objective):
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    if custom_objective:
        obj = custom_dummy_obj
        metric_name = "l2"  # default one
    else:
        obj = "mape"
        metric_name = "mape"
    evals = []
    for alias in lgb.basic._ConfigAliases.get("objective"):
        gbm = lgb.LGBMRegressor(n_estimators=5, **{alias: obj})
        if alias != "objective":
            with pytest.warns(
                UserWarning, match=f"Found '{alias}' in params. Will use it instead of 'objective' argument"
            ):
                gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        else:
            gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        assert list(gbm.evals_result_["valid_0"].keys()) == [metric_name]
        evals.append(gbm.evals_result_["valid_0"][metric_name])
    evals_t = np.array(evals).T
    for i in range(evals_t.shape[0]):
        np.testing.assert_allclose(evals_t[i], evals_t[i][0])
    # check that really dummy objective was used and estimator didn't learn anything
    if custom_objective:
        np.testing.assert_allclose(evals_t, evals_t[0][0])


def test_regression_with_custom_objective():
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(n_estimators=50, verbose=-1, objective=objective_ls)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    ret = mean_squared_error(y_test, gbm.predict(X_test))
    assert ret < 174
    assert gbm.evals_result_["valid_0"]["l2"][gbm.best_iteration_ - 1] == pytest.approx(ret)


def test_binary_classification_with_custom_objective():
    X, y = load_digits(n_class=2, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1, objective=logregobj)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    # prediction result is actually not transformed (is raw) due to custom objective
    y_pred_raw = gbm.predict_proba(X_test)
    assert not np.all(y_pred_raw >= 0)
    y_pred = 1.0 / (1.0 + np.exp(-y_pred_raw))
    ret = binary_error(y_test, y_pred)
    assert ret < 0.05


def test_dart():
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(boosting_type="dart", n_estimators=50)
    gbm.fit(X_train, y_train)
    score = gbm.score(X_test, y_test)
    assert 0.8 <= score <= 1.0


def test_stacking_classifier():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    classifiers = [("gbm1", lgb.LGBMClassifier(n_estimators=3)), ("gbm2", lgb.LGBMClassifier(n_estimators=3))]
    clf = StackingClassifier(
        estimators=classifiers, final_estimator=lgb.LGBMClassifier(n_estimators=3), passthrough=True
    )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score >= 0.8
    assert score <= 1.0
    assert clf.n_features_in_ == 4  # number of input features
    assert len(clf.named_estimators_["gbm1"].feature_importances_) == 4
    assert clf.named_estimators_["gbm1"].n_features_in_ == clf.named_estimators_["gbm2"].n_features_in_
    assert clf.final_estimator_.n_features_in_ == 10  # number of concatenated features
    assert len(clf.final_estimator_.feature_importances_) == 10
    assert all(clf.named_estimators_["gbm1"].classes_ == clf.named_estimators_["gbm2"].classes_)
    assert all(clf.classes_ == clf.named_estimators_["gbm1"].classes_)


def test_stacking_regressor():
    X, y = make_synthetic_regression(n_samples=200)
    n_features = X.shape[1]
    n_input_models = 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    regressors = [("gbm1", lgb.LGBMRegressor(n_estimators=3)), ("gbm2", lgb.LGBMRegressor(n_estimators=3))]
    reg = StackingRegressor(estimators=regressors, final_estimator=lgb.LGBMRegressor(n_estimators=3), passthrough=True)
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    assert score >= 0.2
    assert score <= 1.0
    assert reg.n_features_in_ == n_features  # number of input features
    assert len(reg.named_estimators_["gbm1"].feature_importances_) == n_features
    assert reg.named_estimators_["gbm1"].n_features_in_ == reg.named_estimators_["gbm2"].n_features_in_
    assert reg.final_estimator_.n_features_in_ == n_features + n_input_models  # number of concatenated features
    assert len(reg.final_estimator_.feature_importances_) == n_features + n_input_models


def test_grid_search():
    X, y = load_iris(return_X_y=True)
    y = y.astype(str)  # utilize label encoder at it's max power
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    params = {"subsample": 0.8, "subsample_freq": 1}
    grid_params = {"boosting_type": ["rf", "gbdt"], "n_estimators": [4, 6], "reg_alpha": [0.01, 0.005]}
    evals_result = {}
    fit_params = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": constant_metric,
        "callbacks": [lgb.early_stopping(2), lgb.record_evaluation(evals_result)],
    }
    grid = GridSearchCV(estimator=lgb.LGBMClassifier(**params), param_grid=grid_params, cv=2)
    grid.fit(X_train, y_train, **fit_params)
    score = grid.score(X_test, y_test)  # utilizes GridSearchCV default refit=True
    assert grid.best_params_["boosting_type"] in ["rf", "gbdt"]
    assert grid.best_params_["n_estimators"] in [4, 6]
    assert grid.best_params_["reg_alpha"] in [0.01, 0.005]
    assert grid.best_score_ <= 1.0
    assert grid.best_estimator_.best_iteration_ == 1
    assert grid.best_estimator_.best_score_["valid_0"]["multi_logloss"] < 0.25
    assert grid.best_estimator_.best_score_["valid_0"]["error"] == 0
    assert score >= 0.2
    assert score <= 1.0
    assert evals_result == grid.best_estimator_.evals_result_


def test_random_search(rng):
    X, y = load_iris(return_X_y=True)
    y = y.astype(str)  # utilize label encoder at it's max power
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    n_iter = 3  # Number of samples
    params = {"subsample": 0.8, "subsample_freq": 1}
    param_dist = {
        "boosting_type": ["rf", "gbdt"],
        "n_estimators": rng.integers(low=3, high=10, size=(n_iter,)).tolist(),
        "reg_alpha": rng.uniform(low=0.01, high=0.06, size=(n_iter,)).tolist(),
    }
    fit_params = {"eval_set": [(X_val, y_val)], "eval_metric": constant_metric, "callbacks": [lgb.early_stopping(2)]}
    rand = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(**params), param_distributions=param_dist, cv=2, n_iter=n_iter, random_state=42
    )
    rand.fit(X_train, y_train, **fit_params)
    score = rand.score(X_test, y_test)  # utilizes RandomizedSearchCV default refit=True
    assert rand.best_params_["boosting_type"] in ["rf", "gbdt"]
    assert rand.best_params_["n_estimators"] in list(range(3, 10))
    assert rand.best_params_["reg_alpha"] >= 0.01  # Left-closed boundary point
    assert rand.best_params_["reg_alpha"] <= 0.06  # Right-closed boundary point
    assert rand.best_score_ <= 1.0
    assert rand.best_estimator_.best_score_["valid_0"]["multi_logloss"] < 0.25
    assert rand.best_estimator_.best_score_["valid_0"]["error"] == 0
    assert score >= 0.2
    assert score <= 1.0


def test_multioutput_classifier():
    n_outputs = 3
    X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=n_outputs, random_state=0)
    y = y.astype(str)  # utilize label encoder at it's max power
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    clf = MultiOutputClassifier(estimator=lgb.LGBMClassifier(n_estimators=10))
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score >= 0.2
    assert score <= 1.0
    np_assert_array_equal(np.tile(np.unique(y_train), n_outputs), np.concatenate(clf.classes_), strict=True)
    for classifier in clf.estimators_:
        assert isinstance(classifier, lgb.LGBMClassifier)
        assert isinstance(classifier.booster_, lgb.Booster)


def test_multioutput_regressor():
    bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
    X, y = bunch["data"], bunch["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    reg = MultiOutputRegressor(estimator=lgb.LGBMRegressor(n_estimators=10))
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    _, score, _ = mse(y_test, y_pred)
    assert score >= 0.2
    assert score <= 120.0
    for regressor in reg.estimators_:
        assert isinstance(regressor, lgb.LGBMRegressor)
        assert isinstance(regressor.booster_, lgb.Booster)


def test_classifier_chain():
    n_outputs = 3
    X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=n_outputs, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    order = [2, 0, 1]
    # 'base_estimator' parameter was deprecated in scikit-learn 1.7 and removed in 1.9
    #
    #  * https://github.com/scikit-learn/scikit-learn/pull/30152
    #  * https://github.com/scikit-learn/scikit-learn/pull/33750
    #
    if SKLEARN_VERSION_GTE_1_7:
        clf = ClassifierChain(estimator=lgb.LGBMClassifier(n_estimators=10), order=order, random_state=42)
    else:
        clf = ClassifierChain(base_estimator=lgb.LGBMClassifier(n_estimators=10), order=order, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score >= 0.2
    assert score <= 1.0
    np_assert_array_equal(np.tile(np.unique(y_train), n_outputs), np.concatenate(clf.classes_), strict=True)
    assert order == clf.order_
    for classifier in clf.estimators_:
        assert isinstance(classifier, lgb.LGBMClassifier)
        assert isinstance(classifier.booster_, lgb.Booster)


def test_regressor_chain():
    bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
    X, y = bunch["data"], bunch["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    order = [2, 0, 1]
    # 'base_estimator' parameter was deprecated in scikit-learn 1.7 and removed in 1.9
    #
    #  * https://github.com/scikit-learn/scikit-learn/pull/30152
    #  * https://github.com/scikit-learn/scikit-learn/pull/33750
    #
    if SKLEARN_VERSION_GTE_1_7:
        reg = RegressorChain(estimator=lgb.LGBMRegressor(n_estimators=10), order=order, random_state=42)
    else:
        reg = RegressorChain(base_estimator=lgb.LGBMRegressor(n_estimators=10), order=order, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    _, score, _ = mse(y_test, y_pred)
    assert score >= 0.2
    assert score <= 120.0
    assert order == reg.order_
    for regressor in reg.estimators_:
        assert isinstance(regressor, lgb.LGBMRegressor)
        assert isinstance(regressor.booster_, lgb.Booster)


def test_clone_and_property():
    X, y = make_synthetic_regression()
    gbm = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    gbm.fit(X, y)

    gbm_clone = clone(gbm)

    # original estimator is unaffected
    assert gbm.n_estimators == 10
    assert gbm.verbose == -1
    assert isinstance(gbm.booster_, lgb.Booster)
    assert isinstance(gbm.feature_importances_, np.ndarray)

    # new estimator is unfitted, but has the same parameters
    assert gbm_clone.__sklearn_is_fitted__() is False
    assert gbm_clone.n_estimators == 10
    assert gbm_clone.verbose == -1
    assert gbm_clone.get_params() == gbm.get_params()

    X, y = load_digits(n_class=2, return_X_y=True)
    clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    clf.fit(X, y)
    assert sorted(clf.classes_) == [0, 1]
    assert clf.n_classes_ == 2
    assert isinstance(clf.booster_, lgb.Booster)
    assert isinstance(clf.feature_importances_, np.ndarray)


@pytest.mark.parametrize("estimator", (lgb.LGBMClassifier, lgb.LGBMRegressor, lgb.LGBMRanker))  # noqa: PT007
def test_estimators_all_have_the_same_kwargs_and_defaults(estimator):
    base_spec = inspect.getfullargspec(lgb.LGBMModel)
    subclass_spec = inspect.getfullargspec(estimator)

    # should not allow for any varargs
    assert subclass_spec.varargs == base_spec.varargs
    assert subclass_spec.varargs is None

    # the only varkw should be **kwargs,
    assert subclass_spec.varkw == base_spec.varkw
    assert subclass_spec.varkw == "kwargs"

    # default values for all constructor arguments should be identical
    #
    # NOTE: if LGBMClassifier / LGBMRanker / LGBMRegressor ever override
    #       any of LGBMModel's constructor arguments, this will need to be updated
    assert subclass_spec.kwonlydefaults == base_spec.kwonlydefaults

    # only positional argument should be 'self'
    assert subclass_spec.args == base_spec.args
    assert subclass_spec.args == ["self"]
    assert subclass_spec.defaults is None

    # get_params() should be identical
    assert estimator().get_params() == lgb.LGBMModel().get_params()


def test_subclassing_get_params_works():
    expected_params = {
        "boosting_type": "gbdt",
        "class_weight": None,
        "colsample_bytree": 1.0,
        "importance_type": "split",
        "learning_rate": 0.1,
        "max_depth": -1,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "n_estimators": 100,
        "n_jobs": None,
        "num_leaves": 31,
        "objective": None,
        "random_state": None,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "subsample": 1.0,
        "subsample_for_bin": 200000,
        "subsample_freq": 0,
    }

    # Overrides, used to test that passing through **kwargs works as expected.
    #
    # why these?
    #
    #  - 'n_estimators' directly matches a keyword arg for the scikit-learn estimators
    #  - 'eta' is a parameter alias for 'learning_rate'
    overrides = {"n_estimators": 13, "eta": 0.07}

    # lightgbm-official classes
    for est in [lgb.LGBMModel, lgb.LGBMClassifier, lgb.LGBMRanker, lgb.LGBMRegressor]:
        assert est().get_params() == expected_params
        assert est(**overrides).get_params() == {
            **expected_params,
            "eta": 0.07,
            "n_estimators": 13,
            "learning_rate": 0.1,
        }

    try:
        import dask  # noqa: F401,PLC0415
    except lgb.dask._DaskImportErrorTypes:
        pass

    if "dask" in sys.modules:
        for est in [lgb.DaskLGBMClassifier, lgb.DaskLGBMRanker, lgb.DaskLGBMRegressor]:
            assert est().get_params() == {
                **expected_params,
                "client": None,
            }
            assert est(**overrides).get_params() == {
                **expected_params,
                "eta": 0.07,
                "n_estimators": 13,
                "learning_rate": 0.1,
                "client": None,
            }

    # custom sub-classes
    assert ExtendedLGBMClassifier().get_params() == {**expected_params, "some_other_param": "lgbm-classifier"}
    assert ExtendedLGBMClassifier(**overrides).get_params() == {
        **expected_params,
        "eta": 0.07,
        "n_estimators": 13,
        "learning_rate": 0.1,
        "some_other_param": "lgbm-classifier",
    }
    assert ExtendedLGBMRanker().get_params() == {
        **expected_params,
        "some_other_param": "lgbm-ranker",
    }
    assert ExtendedLGBMRanker(**overrides).get_params() == {
        **expected_params,
        "eta": 0.07,
        "n_estimators": 13,
        "learning_rate": 0.1,
        "some_other_param": "lgbm-ranker",
    }
    assert ExtendedLGBMRegressor().get_params() == {
        **expected_params,
        "some_other_param": "lgbm-regressor",
    }
    assert ExtendedLGBMRegressor(**overrides).get_params() == {
        **expected_params,
        "eta": 0.07,
        "n_estimators": 13,
        "learning_rate": 0.1,
        "some_other_param": "lgbm-regressor",
    }


@pytest.mark.parametrize("task", all_tasks)
def test_subclassing_works(task):
    # param values to make training deterministic and
    # just train a small, cheap model
    params = {
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "n_estimators": 5,
        "num_leaves": 11,
        "random_state": 708,
    }

    X, y, g = _create_data(task=task)
    if task == "ranking":
        est = lgb.LGBMRanker(**params).fit(X, y, group=g)
        est_sub = ExtendedLGBMRanker(**params).fit(X, y, group=g)
    elif task.endswith("classification"):
        est = lgb.LGBMClassifier(**params).fit(X, y)
        est_sub = ExtendedLGBMClassifier(**params).fit(X, y)
    else:
        est = lgb.LGBMRegressor(**params).fit(X, y)
        est_sub = ExtendedLGBMRegressor(**params).fit(X, y)

    np.testing.assert_allclose(est.predict(X), est_sub.predict(X))


@pytest.mark.parametrize(
    "estimator_to_task",
    [
        (lgb.LGBMClassifier, "binary-classification"),
        (ExtendedLGBMClassifier, "binary-classification"),
        (lgb.LGBMRanker, "ranking"),
        (ExtendedLGBMRanker, "ranking"),
        (lgb.LGBMRegressor, "regression"),
        (ExtendedLGBMRegressor, "regression"),
    ],
)
def test_parameter_aliases_are_handled_correctly(estimator_to_task):
    estimator, task = estimator_to_task
    # scikit-learn estimators should remember every parameter passed
    # via keyword arguments in the estimator constructor, but then
    # only pass the correct value down to LightGBM's C++ side
    params = {
        "eta": 0.08,
        "num_iterations": 3,
        "num_leaves": 5,
    }
    X, y, g = _create_data(task=task)
    mod = estimator(**params)
    if task == "ranking":
        mod.fit(X, y, group=g)
    else:
        mod.fit(X, y)

    # scikit-learn get_params()
    p = mod.get_params()
    assert p["eta"] == 0.08
    assert p["learning_rate"] == 0.1

    # lgb.Booster's 'params' attribute
    p = mod.booster_.params
    assert p["eta"] == 0.08
    assert p["learning_rate"] == 0.1

    # Config in the 'LightGBM::Booster' on the C++ side
    p = mod.booster_._get_loaded_param()
    assert p["learning_rate"] == 0.1
    assert "eta" not in p


def test_joblib(tmp_path):
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(n_estimators=10, objective=custom_asymmetric_obj, verbose=-1, importance_type="split")
    gbm.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=mse,
        callbacks=[lgb.early_stopping(5), lgb.reset_parameter(learning_rate=list(np.arange(1, 0, -0.1)))],
    )
    model_path_pkl = str(tmp_path / "lgb.pkl")
    joblib.dump(gbm, model_path_pkl)  # test model with custom functions
    gbm_pickle = joblib.load(model_path_pkl)
    assert isinstance(gbm_pickle.booster_, lgb.Booster)
    assert gbm.get_params() == gbm_pickle.get_params()
    np_assert_array_equal(gbm.feature_importances_, gbm_pickle.feature_importances_, strict=True)
    assert gbm_pickle.learning_rate == pytest.approx(0.1)
    assert callable(gbm_pickle.objective)

    for eval_set in gbm.evals_result_:
        for metric in gbm.evals_result_[eval_set]:
            np.testing.assert_allclose(gbm.evals_result_[eval_set][metric], gbm_pickle.evals_result_[eval_set][metric])
    pred_origin = gbm.predict(X_test)
    pred_pickle = gbm_pickle.predict(X_test)
    np.testing.assert_allclose(pred_origin, pred_pickle)


def test_non_serializable_objects_in_callbacks(tmp_path):
    unpicklable_callback = UnpicklableCallback()

    with pytest.raises(Exception, match="This class in not picklable"):
        joblib.dump(unpicklable_callback, tmp_path / "tmp.joblib")

    X, y = make_synthetic_regression()
    gbm = lgb.LGBMRegressor(n_estimators=5)
    gbm.fit(X, y, callbacks=[unpicklable_callback])
    assert gbm.booster_.attr_set_inside_callback == 40


@pytest.mark.parametrize("rng_constructor", [np.random.RandomState, np.random.default_rng])
def test_random_state_object(rng_constructor):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    state1 = rng_constructor(123)
    state2 = rng_constructor(123)
    clf1 = lgb.LGBMClassifier(n_estimators=10, subsample=0.5, subsample_freq=1, random_state=state1)
    clf2 = lgb.LGBMClassifier(n_estimators=10, subsample=0.5, subsample_freq=1, random_state=state2)
    # Test if random_state is properly stored
    assert clf1.random_state is state1
    assert clf2.random_state is state2
    # Test if two random states produce identical models
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    y_pred1 = clf1.predict(X_test, raw_score=True)
    y_pred2 = clf2.predict(X_test, raw_score=True)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np_assert_array_equal(clf1.feature_importances_, clf2.feature_importances_, strict=True)
    df1 = clf1.booster_.model_to_string(num_iteration=0)
    df2 = clf2.booster_.model_to_string(num_iteration=0)
    assert df1 == df2
    # Test if subsequent fits sample from random_state object and produce different models
    clf1.fit(X_train, y_train)
    y_pred1_refit = clf1.predict(X_test, raw_score=True)
    df3 = clf1.booster_.model_to_string(num_iteration=0)
    assert clf1.random_state is state1
    assert clf2.random_state is state2
    with pytest.raises(AssertionError):  # noqa: PT011
        np.testing.assert_allclose(y_pred1, y_pred1_refit)
    assert df1 != df3


def test_feature_importances_single_leaf():
    data = load_iris(return_X_y=False)
    clf = lgb.LGBMClassifier(n_estimators=10)
    clf.fit(data.data, data.target)
    importances = clf.feature_importances_
    assert len(importances) == 4


def test_feature_importances_type():
    data = load_iris(return_X_y=False)
    clf = lgb.LGBMClassifier(n_estimators=10)
    clf.fit(data.data, data.target)
    clf.set_params(importance_type="split")
    importances_split = clf.feature_importances_
    clf.set_params(importance_type="gain")
    importances_gain = clf.feature_importances_
    # Test that the largest element is NOT the same, the smallest can be the same, i.e. zero
    importance_split_top1 = sorted(importances_split, reverse=True)[0]
    importance_gain_top1 = sorted(importances_gain, reverse=True)[0]
    assert importance_split_top1 != importance_gain_top1


# why fixed seed?
# sometimes there is no difference how cols are treated (cat or not cat)
def test_pandas_categorical(rng_fixed_seed, tmp_path):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "A": rng_fixed_seed.permutation(["a", "b", "c", "d"] * 75),  # str
            "B": rng_fixed_seed.permutation([1, 2, 3] * 100),  # int
            "C": rng_fixed_seed.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),  # float
            "D": rng_fixed_seed.permutation([True, False] * 150),  # bool
            "E": pd.Categorical(rng_fixed_seed.permutation(["z", "y", "x", "w", "v"] * 60), ordered=True),
        }
    )  # str and ordered categorical
    y = rng_fixed_seed.permutation([0, 1] * 150)
    X_test = pd.DataFrame(
        {
            "A": rng_fixed_seed.permutation(["a", "b", "e"] * 20),  # unseen category
            "B": rng_fixed_seed.permutation([1, 3] * 30),
            "C": rng_fixed_seed.permutation([0.1, -0.1, 0.2, 0.2] * 15),
            "D": rng_fixed_seed.permutation([True, False] * 30),
            "E": pd.Categorical(rng_fixed_seed.permutation(["z", "y"] * 30), ordered=True),
        }
    )
    cat_cols_actual = ["A", "B", "C", "D"]
    cat_cols_to_store = cat_cols_actual + ["E"]
    X[cat_cols_actual] = X[cat_cols_actual].astype("category")
    X_test[cat_cols_actual] = X_test[cat_cols_actual].astype("category")
    cat_values = [X[col].cat.categories.tolist() for col in cat_cols_to_store]
    gbm0 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y)
    pred0 = gbm0.predict(X_test, raw_score=True)
    pred_prob = gbm0.predict_proba(X_test)[:, 1]
    gbm1 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, pd.Series(y), categorical_feature=[0])
    pred1 = gbm1.predict(X_test, raw_score=True)
    gbm2 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y, categorical_feature=["A"])
    pred2 = gbm2.predict(X_test, raw_score=True)
    gbm3 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y, categorical_feature=["A", "B", "C", "D"])
    pred3 = gbm3.predict(X_test, raw_score=True)
    categorical_model_path = tmp_path / "categorical.model"
    gbm3.booster_.save_model(categorical_model_path)
    gbm4 = lgb.Booster(model_file=categorical_model_path)
    pred4 = gbm4.predict(X_test)
    gbm5 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y, categorical_feature=["A", "B", "C", "D", "E"])
    pred5 = gbm5.predict(X_test, raw_score=True)
    gbm6 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y, categorical_feature=[])
    pred6 = gbm6.predict(X_test, raw_score=True)
    with pytest.raises(AssertionError):  # noqa: PT011
        np.testing.assert_allclose(pred0, pred1)
    with pytest.raises(AssertionError):  # noqa: PT011
        np.testing.assert_allclose(pred0, pred2)
    np.testing.assert_allclose(pred1, pred2)
    np.testing.assert_allclose(pred0, pred3)
    np.testing.assert_allclose(pred_prob, pred4)
    with pytest.raises(AssertionError):  # noqa: PT011
        np.testing.assert_allclose(pred0, pred5)  # ordered cat features aren't treated as cat features by default
    with pytest.raises(AssertionError):  # noqa: PT011
        np.testing.assert_allclose(pred0, pred6)
    assert gbm0.booster_.pandas_categorical == cat_values
    assert gbm1.booster_.pandas_categorical == cat_values
    assert gbm2.booster_.pandas_categorical == cat_values
    assert gbm3.booster_.pandas_categorical == cat_values
    assert gbm4.pandas_categorical == cat_values
    assert gbm5.booster_.pandas_categorical == cat_values
    assert gbm6.booster_.pandas_categorical == cat_values


def test_pandas_sparse(rng):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {
            "A": pd.arrays.SparseArray(rng.permutation([0, 1, 2] * 100)),
            "B": pd.arrays.SparseArray(rng.permutation([0.0, 0.1, 0.2, -0.1, 0.2] * 60)),
            "C": pd.arrays.SparseArray(rng.permutation([True, False] * 150)),
        }
    )
    y = pd.Series(pd.arrays.SparseArray(rng.permutation([0, 1] * 150)))
    X_test = pd.DataFrame(
        {
            "A": pd.arrays.SparseArray(rng.permutation([0, 2] * 30)),
            "B": pd.arrays.SparseArray(rng.permutation([0.0, 0.1, 0.2, -0.1] * 15)),
            "C": pd.arrays.SparseArray(rng.permutation([True, False] * 30)),
        }
    )
    for dtype in pd.concat([X.dtypes, X_test.dtypes, pd.Series(y.dtypes)]):
        assert isinstance(dtype, pd.SparseDtype)
    gbm = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y)
    pred_sparse = gbm.predict(X_test, raw_score=True)
    if hasattr(X_test, "sparse"):
        pred_dense = gbm.predict(X_test.sparse.to_dense(), raw_score=True)
    else:
        pred_dense = gbm.predict(X_test.to_dense(), raw_score=True)
    np.testing.assert_allclose(pred_sparse, pred_dense)


def test_predict():
    # With default params
    iris = load_iris(return_X_y=False)
    X_train, X_test, y_train, _ = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    gbm = lgb.train({"objective": "multiclass", "num_class": 3, "verbose": -1}, lgb.Dataset(X_train, y_train))
    clf = lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train)

    # Tests same probabilities
    res_engine = gbm.predict(X_test)
    res_sklearn = clf.predict_proba(X_test)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests same predictions
    res_engine = np.argmax(gbm.predict(X_test), axis=1)
    res_sklearn = clf.predict(X_test)
    np.testing.assert_equal(res_engine, res_sklearn)

    # Tests same raw scores
    res_engine = gbm.predict(X_test, raw_score=True)
    res_sklearn = clf.predict(X_test, raw_score=True)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests same leaf indices
    res_engine = gbm.predict(X_test, pred_leaf=True)
    res_sklearn = clf.predict(X_test, pred_leaf=True)
    np.testing.assert_equal(res_engine, res_sklearn)

    # Tests same feature contributions
    res_engine = gbm.predict(X_test, pred_contrib=True)
    res_sklearn = clf.predict(X_test, pred_contrib=True)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests other parameters for the prediction works
    res_engine = gbm.predict(X_test)
    res_sklearn_params = clf.predict_proba(X_test, pred_early_stop=True, pred_early_stop_margin=1.0)
    with pytest.raises(AssertionError):  # noqa: PT011
        np.testing.assert_allclose(res_engine, res_sklearn_params)

    # Tests start_iteration
    # Tests same probabilities, starting from iteration 10
    res_engine = gbm.predict(X_test, start_iteration=10)
    res_sklearn = clf.predict_proba(X_test, start_iteration=10)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests same predictions, starting from iteration 10
    res_engine = np.argmax(gbm.predict(X_test, start_iteration=10), axis=1)
    res_sklearn = clf.predict(X_test, start_iteration=10)
    np.testing.assert_equal(res_engine, res_sklearn)

    # Tests same raw scores, starting from iteration 10
    res_engine = gbm.predict(X_test, raw_score=True, start_iteration=10)
    res_sklearn = clf.predict(X_test, raw_score=True, start_iteration=10)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests same leaf indices, starting from iteration 10
    res_engine = gbm.predict(X_test, pred_leaf=True, start_iteration=10)
    res_sklearn = clf.predict(X_test, pred_leaf=True, start_iteration=10)
    np.testing.assert_equal(res_engine, res_sklearn)

    # Tests same feature contributions, starting from iteration 10
    res_engine = gbm.predict(X_test, pred_contrib=True, start_iteration=10)
    res_sklearn = clf.predict(X_test, pred_contrib=True, start_iteration=10)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests other parameters for the prediction works, starting from iteration 10
    res_engine = gbm.predict(X_test, start_iteration=10)
    res_sklearn_params = clf.predict_proba(X_test, pred_early_stop=True, pred_early_stop_margin=1.0, start_iteration=10)
    with pytest.raises(AssertionError):  # noqa: PT011
        np.testing.assert_allclose(res_engine, res_sklearn_params)

    # Test multiclass binary classification
    num_samples = 100
    num_classes = 2
    X_train = np.linspace(start=0, stop=10, num=num_samples * 3).reshape(num_samples, 3)
    y_train = np.concatenate([np.zeros(int(num_samples / 2 - 10)), np.ones(int(num_samples / 2 + 10))])

    gbm = lgb.train({"objective": "multiclass", "num_class": num_classes, "verbose": -1}, lgb.Dataset(X_train, y_train))
    clf = lgb.LGBMClassifier(objective="multiclass", num_classes=num_classes).fit(X_train, y_train)

    res_engine = gbm.predict(X_train)
    res_sklearn = clf.predict_proba(X_train)

    assert res_engine.shape == (num_samples, num_classes)
    assert res_sklearn.shape == (num_samples, num_classes)
    np.testing.assert_allclose(res_engine, res_sklearn)

    res_class_sklearn = clf.predict(X_train)
    np.testing.assert_allclose(res_class_sklearn, y_train)


def test_decision_function_and_predict_proba_consistency():
    # binary
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1).fit(X_train, y_train)
    preds_raw = clf.decision_function(X_test)
    np.testing.assert_allclose(preds_raw, clf.predict(X_test, raw_score=True))
    np.testing.assert_allclose(logistic_sigmoid(preds_raw), clf.predict_proba(X_test)[:, 1])

    # multiclass
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1).fit(X_train, y_train)
    preds_raw = clf.decision_function(X_test)
    np.testing.assert_allclose(preds_raw, clf.predict(X_test, raw_score=True))
    np.testing.assert_allclose(softmax(preds_raw), clf.predict_proba(X_test))


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrated_classifier_cv(method):
    # binary
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    deterministic_params = {
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "seed": 312,
    }
    clf = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=10, verbose=-1, **deterministic_params),
        method=method,
        cv=3,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 2)
    np.testing.assert_array_less(proba, 1.0 + 1e-9)
    np.testing.assert_array_less(-1e-9, proba)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    score = accuracy_score(y_test, clf.predict(X_test))
    assert 0.8 <= score <= 1.0

    # multiclass
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = CalibratedClassifierCV(
        lgb.LGBMClassifier(n_estimators=10, verbose=-1, **deterministic_params),
        method=method,
        cv=3,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 3)
    np.testing.assert_array_less(proba, 1.0 + 1e-9)
    np.testing.assert_array_less(-1e-9, proba)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    score = accuracy_score(y_test, clf.predict(X_test))
    assert 0.8 <= score <= 1.0


def test_predict_with_params_from_init():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    predict_params = {"pred_early_stop": True, "pred_early_stop_margin": 1.0}

    y_preds_no_params = lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train).predict(X_test, raw_score=True)

    y_preds_params_in_predict = (
        lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train).predict(X_test, raw_score=True, **predict_params)
    )
    with pytest.raises(AssertionError):  # noqa: PT011
        np.testing.assert_allclose(y_preds_no_params, y_preds_params_in_predict)

    y_preds_params_in_set_params_before_fit = (
        lgb.LGBMClassifier(verbose=-1)
        .set_params(**predict_params)
        .fit(X_train, y_train)
        .predict(X_test, raw_score=True)
    )
    np.testing.assert_allclose(y_preds_params_in_predict, y_preds_params_in_set_params_before_fit)

    y_preds_params_in_set_params_after_fit = (
        lgb.LGBMClassifier(verbose=-1)
        .fit(X_train, y_train)
        .set_params(**predict_params)
        .predict(X_test, raw_score=True)
    )
    np.testing.assert_allclose(y_preds_params_in_predict, y_preds_params_in_set_params_after_fit)

    y_preds_params_in_init = (
        lgb.LGBMClassifier(verbose=-1, **predict_params).fit(X_train, y_train).predict(X_test, raw_score=True)
    )
    np.testing.assert_allclose(y_preds_params_in_predict, y_preds_params_in_init)

    # test that params passed in predict have higher priority
    y_preds_params_overwritten = (
        lgb.LGBMClassifier(verbose=-1, **predict_params)
        .fit(X_train, y_train)
        .predict(X_test, raw_score=True, pred_early_stop=False)
    )
    np.testing.assert_allclose(y_preds_no_params, y_preds_params_overwritten)


def test_evaluate_train_set():
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    gbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    assert len(gbm.evals_result_) == 2
    assert "training" in gbm.evals_result_
    assert len(gbm.evals_result_["training"]) == 1
    assert "l2" in gbm.evals_result_["training"]
    assert "valid_1" in gbm.evals_result_
    assert len(gbm.evals_result_["valid_1"]) == 1
    assert "l2" in gbm.evals_result_["valid_1"]


def test_metrics():
    X, y = make_synthetic_regression()
    y = abs(y)
    params = {"n_estimators": 2, "verbose": -1}
    params_fit = {"X": X, "y": y, "eval_set": (X, y)}

    # no custom objective, no custom metric
    # default metric
    gbm = lgb.LGBMRegressor(**params).fit(**params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "l2" in gbm.evals_result_["training"]

    # non-default metric
    gbm = lgb.LGBMRegressor(metric="mape", **params).fit(**params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "mape" in gbm.evals_result_["training"]

    # no metric
    gbm = lgb.LGBMRegressor(metric="None", **params).fit(**params_fit)
    assert gbm.evals_result_ == {}

    # non-default metric in eval_metric
    gbm = lgb.LGBMRegressor(**params).fit(eval_metric="mape", **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "l2" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]

    # non-default metric with non-default metric in eval_metric
    gbm = lgb.LGBMRegressor(metric="gamma", **params).fit(eval_metric="mape", **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "gamma" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]

    # non-default metric with multiple metrics in eval_metric
    gbm = lgb.LGBMRegressor(metric="gamma", **params).fit(eval_metric=["l2", "mape"], **params_fit)
    assert len(gbm.evals_result_["training"]) == 3
    assert "gamma" in gbm.evals_result_["training"]
    assert "l2" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]

    # non-default metric with multiple metrics in eval_metric for LGBMClassifier
    X_classification, y_classification = load_breast_cancer(return_X_y=True)
    params_classification = {"n_estimators": 2, "verbose": -1, "objective": "binary", "metric": "binary_logloss"}
    params_fit_classification = {
        "X": X_classification,
        "y": y_classification,
        "eval_set": (X_classification, y_classification),
    }
    gbm = lgb.LGBMClassifier(**params_classification).fit(eval_metric=["fair", "error"], **params_fit_classification)
    assert len(gbm.evals_result_["training"]) == 3
    assert "fair" in gbm.evals_result_["training"]
    assert "binary_error" in gbm.evals_result_["training"]
    assert "binary_logloss" in gbm.evals_result_["training"]

    # default metric for non-default objective
    gbm = lgb.LGBMRegressor(objective="regression_l1", **params).fit(**params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "l1" in gbm.evals_result_["training"]

    # non-default metric for non-default objective
    gbm = lgb.LGBMRegressor(objective="regression_l1", metric="mape", **params).fit(**params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "mape" in gbm.evals_result_["training"]

    # no metric
    gbm = lgb.LGBMRegressor(objective="regression_l1", metric="None", **params).fit(**params_fit)
    assert gbm.evals_result_ == {}

    # non-default metric in eval_metric for non-default objective
    gbm = lgb.LGBMRegressor(objective="regression_l1", **params).fit(eval_metric="mape", **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "l1" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]

    # non-default metric with non-default metric in eval_metric for non-default objective
    gbm = lgb.LGBMRegressor(objective="regression_l1", metric="gamma", **params).fit(eval_metric="mape", **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "gamma" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]

    # non-default metric with multiple metrics in eval_metric for non-default objective
    gbm = lgb.LGBMRegressor(objective="regression_l1", metric="gamma", **params).fit(
        eval_metric=["l2", "mape"], **params_fit
    )
    assert len(gbm.evals_result_["training"]) == 3
    assert "gamma" in gbm.evals_result_["training"]
    assert "l2" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]

    # custom objective, no custom metric
    # default regression metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, **params).fit(**params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "l2" in gbm.evals_result_["training"]

    # non-default regression metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric="mape", **params).fit(**params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "mape" in gbm.evals_result_["training"]

    # multiple regression metrics for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric=["l1", "gamma"], **params).fit(**params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "l1" in gbm.evals_result_["training"]
    assert "gamma" in gbm.evals_result_["training"]

    # no metric
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric="None", **params).fit(**params_fit)
    assert gbm.evals_result_ == {}

    # default regression metric with non-default metric in eval_metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, **params).fit(eval_metric="mape", **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "l2" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]

    # non-default regression metric with metric in eval_metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric="mape", **params).fit(eval_metric="gamma", **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "mape" in gbm.evals_result_["training"]
    assert "gamma" in gbm.evals_result_["training"]

    # multiple regression metrics with metric in eval_metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric=["l1", "gamma"], **params).fit(
        eval_metric="l2", **params_fit
    )
    assert len(gbm.evals_result_["training"]) == 3
    assert "l1" in gbm.evals_result_["training"]
    assert "gamma" in gbm.evals_result_["training"]
    assert "l2" in gbm.evals_result_["training"]

    # multiple regression metrics with multiple metrics in eval_metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric=["l1", "gamma"], **params).fit(
        eval_metric=["l2", "mape"], **params_fit
    )
    assert len(gbm.evals_result_["training"]) == 4
    assert "l1" in gbm.evals_result_["training"]
    assert "gamma" in gbm.evals_result_["training"]
    assert "l2" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]

    # no custom objective, custom metric
    # default metric with custom metric
    gbm = lgb.LGBMRegressor(**params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "l2" in gbm.evals_result_["training"]
    assert "error" in gbm.evals_result_["training"]

    # non-default metric with custom metric
    gbm = lgb.LGBMRegressor(metric="mape", **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "mape" in gbm.evals_result_["training"]
    assert "error" in gbm.evals_result_["training"]

    # multiple metrics with custom metric
    gbm = lgb.LGBMRegressor(metric=["l1", "gamma"], **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_["training"]) == 3
    assert "l1" in gbm.evals_result_["training"]
    assert "gamma" in gbm.evals_result_["training"]
    assert "error" in gbm.evals_result_["training"]

    # custom metric (disable default metric)
    gbm = lgb.LGBMRegressor(metric="None", **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "error" in gbm.evals_result_["training"]

    # default metric for non-default objective with custom metric
    gbm = lgb.LGBMRegressor(objective="regression_l1", **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "l1" in gbm.evals_result_["training"]
    assert "error" in gbm.evals_result_["training"]

    # non-default metric for non-default objective with custom metric
    gbm = lgb.LGBMRegressor(objective="regression_l1", metric="mape", **params).fit(
        eval_metric=constant_metric, **params_fit
    )
    assert len(gbm.evals_result_["training"]) == 2
    assert "mape" in gbm.evals_result_["training"]
    assert "error" in gbm.evals_result_["training"]

    # multiple metrics for non-default objective with custom metric
    gbm = lgb.LGBMRegressor(objective="regression_l1", metric=["l1", "gamma"], **params).fit(
        eval_metric=constant_metric, **params_fit
    )
    assert len(gbm.evals_result_["training"]) == 3
    assert "l1" in gbm.evals_result_["training"]
    assert "gamma" in gbm.evals_result_["training"]
    assert "error" in gbm.evals_result_["training"]

    # custom metric (disable default metric for non-default objective)
    gbm = lgb.LGBMRegressor(objective="regression_l1", metric="None", **params).fit(
        eval_metric=constant_metric, **params_fit
    )
    assert len(gbm.evals_result_["training"]) == 1
    assert "error" in gbm.evals_result_["training"]

    # custom objective, custom metric
    # custom metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "error" in gbm.evals_result_["training"]

    # non-default regression metric with custom metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric="mape", **params).fit(
        eval_metric=constant_metric, **params_fit
    )
    assert len(gbm.evals_result_["training"]) == 2
    assert "mape" in gbm.evals_result_["training"]
    assert "error" in gbm.evals_result_["training"]

    # multiple regression metrics with custom metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric=["l2", "mape"], **params).fit(
        eval_metric=constant_metric, **params_fit
    )
    assert len(gbm.evals_result_["training"]) == 3
    assert "l2" in gbm.evals_result_["training"]
    assert "mape" in gbm.evals_result_["training"]
    assert "error" in gbm.evals_result_["training"]

    X, y = load_digits(n_class=3, return_X_y=True)
    params_fit = {"X": X, "y": y, "eval_set": (X, y)}

    # default metric and invalid binary metric is replaced with multiclass alternative
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric="binary_error", **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "multi_logloss" in gbm.evals_result_["training"]
    assert "multi_error" in gbm.evals_result_["training"]

    # invalid binary metric is replaced with multiclass alternative
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric="binary_error", **params_fit)
    assert gbm.objective_ == "multiclass"
    assert len(gbm.evals_result_["training"]) == 2
    assert "multi_logloss" in gbm.evals_result_["training"]
    assert "multi_error" in gbm.evals_result_["training"]

    # default metric for non-default multiclass objective
    # and invalid binary metric is replaced with multiclass alternative
    gbm = lgb.LGBMClassifier(objective="ovr", **params).fit(eval_metric="binary_error", **params_fit)
    assert gbm.objective_ == "ovr"
    assert len(gbm.evals_result_["training"]) == 2
    assert "multi_logloss" in gbm.evals_result_["training"]
    assert "multi_error" in gbm.evals_result_["training"]

    X, y = load_digits(n_class=2, return_X_y=True)
    params_fit = {"X": X, "y": y, "eval_set": (X, y)}

    # default metric and invalid multiclass metric is replaced with binary alternative
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric="multi_error", **params_fit)
    assert len(gbm.evals_result_["training"]) == 2
    assert "binary_logloss" in gbm.evals_result_["training"]
    assert "binary_error" in gbm.evals_result_["training"]

    # invalid multiclass metric is replaced with binary alternative for custom objective
    gbm = lgb.LGBMClassifier(objective=custom_dummy_obj, **params).fit(eval_metric="multi_logloss", **params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "binary_logloss" in gbm.evals_result_["training"]

    # the evaluation metric changes to multiclass metric even num classes is 2 for multiclass objective
    gbm = lgb.LGBMClassifier(objective="multiclass", num_classes=2, **params).fit(
        eval_metric="binary_logloss", **params_fit
    )
    assert len(gbm._evals_result["training"]) == 1
    assert "multi_logloss" in gbm.evals_result_["training"]

    # the evaluation metric changes to multiclass metric even num classes is 2 for ovr objective
    gbm = lgb.LGBMClassifier(objective="ovr", num_classes=2, **params).fit(eval_metric="binary_error", **params_fit)
    assert gbm.objective_ == "ovr"
    assert len(gbm.evals_result_["training"]) == 2
    assert "multi_logloss" in gbm.evals_result_["training"]
    assert "multi_error" in gbm.evals_result_["training"]


def test_multiple_eval_metrics():
    X, y = load_breast_cancer(return_X_y=True)

    params = {"n_estimators": 2, "verbose": -1, "objective": "binary", "metric": "binary_logloss"}
    params_fit = {"X": X, "y": y, "eval_set": (X, y)}

    # Verify that can receive a list of metrics, only callable
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=[constant_metric, decreasing_metric], **params_fit)
    assert len(gbm.evals_result_["training"]) == 3
    assert "error" in gbm.evals_result_["training"]
    assert "decreasing_metric" in gbm.evals_result_["training"]
    assert "binary_logloss" in gbm.evals_result_["training"]

    # Verify that can receive a list of custom and built-in metrics
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=[constant_metric, decreasing_metric, "fair"], **params_fit)
    assert len(gbm.evals_result_["training"]) == 4
    assert "error" in gbm.evals_result_["training"]
    assert "decreasing_metric" in gbm.evals_result_["training"]
    assert "binary_logloss" in gbm.evals_result_["training"]
    assert "fair" in gbm.evals_result_["training"]

    # Verify that works as expected when eval_metric is empty
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=[], **params_fit)
    assert len(gbm.evals_result_["training"]) == 1
    assert "binary_logloss" in gbm.evals_result_["training"]

    # Verify that can receive a list of metrics, only built-in
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=["fair", "error"], **params_fit)
    assert len(gbm.evals_result_["training"]) == 3
    assert "binary_logloss" in gbm.evals_result_["training"]

    # Verify that eval_metric is robust to receiving a list with None
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=["fair", "error", None], **params_fit)
    assert len(gbm.evals_result_["training"]) == 3
    assert "binary_logloss" in gbm.evals_result_["training"]


def test_nan_handle(rng):
    nrows = 100
    ncols = 10
    X = rng.standard_normal(size=(nrows, ncols))
    y = rng.standard_normal(size=(nrows,)) + np.full(nrows, 1e30)
    weight = np.zeros(nrows)
    params = {"n_estimators": 20, "verbose": -1}
    params_fit = {"X": X, "y": y, "sample_weight": weight, "eval_set": (X, y), "callbacks": [lgb.early_stopping(5)]}
    gbm = lgb.LGBMRegressor(**params).fit(**params_fit)
    np.testing.assert_allclose(gbm.evals_result_["training"]["l2"], np.nan)


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Skip due to differences in implementation details of CUDA version")
def test_first_metric_only():
    def fit_and_check(eval_set_names, metric_names, assumed_iteration, first_metric_only):
        params["first_metric_only"] = first_metric_only
        gbm = lgb.LGBMRegressor(**params).fit(**params_fit)
        assert len(gbm.evals_result_) == len(eval_set_names)
        for eval_set_name in eval_set_names:
            assert eval_set_name in gbm.evals_result_
            assert len(gbm.evals_result_[eval_set_name]) == len(metric_names)
            for metric_name in metric_names:
                assert metric_name in gbm.evals_result_[eval_set_name]

                actual = len(gbm.evals_result_[eval_set_name][metric_name])
                expected = assumed_iteration + (
                    params["early_stopping_rounds"]
                    if eval_set_name != "training" and assumed_iteration != gbm.n_estimators
                    else 0
                )
                assert expected == actual
                if eval_set_name != "training":
                    assert assumed_iteration == gbm.best_iteration_
                else:
                    assert gbm.n_estimators == gbm.best_iteration_

    X, y = make_synthetic_regression(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=72)
    params = {
        "n_estimators": 30,
        "learning_rate": 0.8,
        "num_leaves": 15,
        "verbose": -1,
        "seed": 123,
        "early_stopping_rounds": 5,
    }  # early stop should be supported via global LightGBM parameter
    params_fit = {"X": X_train, "y": y_train}

    iter_valid1_l1 = 4
    iter_valid1_l2 = 4
    iter_valid2_l1 = 2
    iter_valid2_l2 = 2
    assert len({iter_valid1_l1, iter_valid1_l2, iter_valid2_l1, iter_valid2_l2}) == 2
    iter_min_l1 = min([iter_valid1_l1, iter_valid2_l1])
    iter_min_l2 = min([iter_valid1_l2, iter_valid2_l2])
    iter_min = min([iter_min_l1, iter_min_l2])
    iter_min_valid1 = min([iter_valid1_l1, iter_valid1_l2])

    # feval
    params["metric"] = "None"
    params_fit["eval_metric"] = lambda preds, train_data: [
        decreasing_metric(preds, train_data),
        constant_metric(preds, train_data),
    ]
    params_fit["eval_set"] = (X_test1, y_test1)
    fit_and_check(["valid_0"], ["decreasing_metric", "error"], 1, False)
    fit_and_check(["valid_0"], ["decreasing_metric", "error"], 30, True)
    params_fit["eval_metric"] = lambda preds, train_data: [
        constant_metric(preds, train_data),
        decreasing_metric(preds, train_data),
    ]
    fit_and_check(["valid_0"], ["decreasing_metric", "error"], 1, True)

    # single eval_set
    params.pop("metric")
    params_fit.pop("eval_metric")
    fit_and_check(["valid_0"], ["l2"], iter_valid1_l2, False)
    fit_and_check(["valid_0"], ["l2"], iter_valid1_l2, True)

    params_fit["eval_metric"] = "l2"
    fit_and_check(["valid_0"], ["l2"], iter_valid1_l2, False)
    fit_and_check(["valid_0"], ["l2"], iter_valid1_l2, True)

    params_fit["eval_metric"] = "l1"
    fit_and_check(["valid_0"], ["l1", "l2"], iter_min_valid1, False)
    fit_and_check(["valid_0"], ["l1", "l2"], iter_valid1_l1, True)

    params_fit["eval_metric"] = ["l1", "l2"]
    fit_and_check(["valid_0"], ["l1", "l2"], iter_min_valid1, False)
    fit_and_check(["valid_0"], ["l1", "l2"], iter_valid1_l1, True)

    params_fit["eval_metric"] = ["l2", "l1"]
    fit_and_check(["valid_0"], ["l1", "l2"], iter_min_valid1, False)
    fit_and_check(["valid_0"], ["l1", "l2"], iter_valid1_l2, True)

    params_fit["eval_metric"] = ["l2", "regression", "mse"]  # test aliases
    fit_and_check(["valid_0"], ["l2"], iter_valid1_l2, False)
    fit_and_check(["valid_0"], ["l2"], iter_valid1_l2, True)

    # two eval_set
    params_fit["eval_set"] = [(X_test1, y_test1), (X_test2, y_test2)]
    params_fit["eval_metric"] = ["l1", "l2"]
    fit_and_check(["valid_0", "valid_1"], ["l1", "l2"], iter_min_l1, True)
    params_fit["eval_metric"] = ["l2", "l1"]
    fit_and_check(["valid_0", "valid_1"], ["l1", "l2"], iter_min_l2, True)

    params_fit["eval_set"] = [(X_test2, y_test2), (X_test1, y_test1)]
    params_fit["eval_metric"] = ["l1", "l2"]
    fit_and_check(["valid_0", "valid_1"], ["l1", "l2"], iter_min, False)
    fit_and_check(["valid_0", "valid_1"], ["l1", "l2"], iter_min_l1, True)
    params_fit["eval_metric"] = ["l2", "l1"]
    fit_and_check(["valid_0", "valid_1"], ["l1", "l2"], iter_min, False)
    fit_and_check(["valid_0", "valid_1"], ["l1", "l2"], iter_min_l2, True)


def test_class_weight():
    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_str = y_train.astype("str")
    y_test_str = y_test.astype("str")
    gbm = lgb.LGBMClassifier(n_estimators=10, class_weight="balanced", verbose=-1)
    gbm.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test), (X_test, y_test), (X_test, y_test), (X_test, y_test)],
        eval_class_weight=["balanced", None, "balanced", {1: 10, 4: 20}, {5: 30, 2: 40}],
    )
    for eval_set1, eval_set2 in itertools.combinations(gbm.evals_result_.keys(), 2):
        for metric in gbm.evals_result_[eval_set1]:
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_allclose,
                gbm.evals_result_[eval_set1][metric],
                gbm.evals_result_[eval_set2][metric],
            )
    gbm_str = lgb.LGBMClassifier(n_estimators=10, class_weight="balanced", verbose=-1)
    gbm_str.fit(
        X_train,
        y_train_str,
        eval_set=[
            (X_train, y_train_str),
            (X_test, y_test_str),
            (X_test, y_test_str),
            (X_test, y_test_str),
            (X_test, y_test_str),
        ],
        eval_class_weight=["balanced", None, "balanced", {"1": 10, "4": 20}, {"5": 30, "2": 40}],
    )
    for eval_set1, eval_set2 in itertools.combinations(gbm_str.evals_result_.keys(), 2):
        for metric in gbm_str.evals_result_[eval_set1]:
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_allclose,
                gbm_str.evals_result_[eval_set1][metric],
                gbm_str.evals_result_[eval_set2][metric],
            )
    for eval_set in gbm.evals_result_:
        for metric in gbm.evals_result_[eval_set]:
            np.testing.assert_allclose(gbm.evals_result_[eval_set][metric], gbm_str.evals_result_[eval_set][metric])


def test_continue_training_with_model():
    X, y = load_digits(n_class=3, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    init_gbm = lgb.LGBMClassifier(n_estimators=5).fit(X_train, y_train, eval_set=(X_test, y_test))
    gbm = lgb.LGBMClassifier(n_estimators=5).fit(X_train, y_train, eval_set=(X_test, y_test), init_model=init_gbm)
    assert len(init_gbm.evals_result_["valid_0"]["multi_logloss"]) == len(gbm.evals_result_["valid_0"]["multi_logloss"])
    assert len(init_gbm.evals_result_["valid_0"]["multi_logloss"]) == 5
    assert gbm.evals_result_["valid_0"]["multi_logloss"][-1] < init_gbm.evals_result_["valid_0"]["multi_logloss"][-1]


def test_booster_does_not_hold_datasets():
    """fit() should clear the Dataset objects stored on the Booster"""
    data = load_iris(return_X_y=False)
    clf = lgb.LGBMClassifier(n_estimators=2, max_depth=2)
    clf.fit(data.data, data.target)
    assert not hasattr(clf.booster_, "train_set")
    assert not hasattr(clf.booster_, "valid_sets")


def test_actual_number_of_trees():
    X = [[1, 2, 3], [1, 2, 3]]
    y = [1.0, 1.0]
    n_estimators = 5
    gbm = lgb.LGBMRegressor(n_estimators=n_estimators).fit(X, y)
    assert gbm.n_estimators == n_estimators
    assert gbm.n_estimators_ == 1
    assert gbm.n_iter_ == 1
    np_assert_array_equal(gbm.predict(np.array(X) * 10), y, strict=True)


def test_check_is_fitted():
    X, y = load_digits(n_class=2, return_X_y=True)
    est = lgb.LGBMModel(n_estimators=5, objective="binary")
    clf = lgb.LGBMClassifier(n_estimators=5)
    reg = lgb.LGBMRegressor(n_estimators=5)
    rnk = lgb.LGBMRanker(n_estimators=5)
    models = (est, clf, reg, rnk)
    for model in models:
        err_msg = f"This {type(model).__name__} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
        with pytest.raises(lgb.compat.LGBMNotFittedError, match=err_msg):
            check_is_fitted(model)
    est.fit(X, y)
    clf.fit(X, y)
    reg.fit(X, y)
    rnk.fit(X, y, group=np.ones(X.shape[0]))
    for model in models:
        check_is_fitted(model)


@pytest.mark.parametrize("estimator_class", estimator_classes)
@pytest.mark.parametrize("max_depth", [3, 4, 5, 8])
def test_max_depth_warning_is_never_raised(capsys, estimator_class, max_depth):
    X, y = make_blobs(n_samples=1_000, n_features=1, centers=2)
    params = {"n_estimators": 1, "max_depth": max_depth, "verbose": 0}
    if estimator_class is lgb.LGBMModel:
        estimator_class(**{**params, "objective": "binary"}).fit(X, y)
    elif estimator_class is lgb.LGBMRanker:
        estimator_class(**params).fit(X, y, group=np.ones(X.shape[0]))
    else:
        estimator_class(**params).fit(X, y)
    assert "Provided parameters constrain tree depth" not in capsys.readouterr().out


def test_verbosity_is_respected_when_using_custom_objective(capsys):
    X, y = make_synthetic_regression()
    params = {
        "objective": objective_ls,
        "nonsense": 123,
        "num_leaves": 3,
    }
    lgb.LGBMRegressor(**params, verbosity=-1, n_estimators=1).fit(X, y)
    assert capsys.readouterr().out == ""
    lgb.LGBMRegressor(**params, verbosity=0, n_estimators=1).fit(X, y)
    assert "[LightGBM] [Warning] Unknown parameter: nonsense" in capsys.readouterr().out


def test_fit_only_raises_num_rounds_warning_when_expected(capsys):
    X, y = make_synthetic_regression()
    base_kwargs = {
        "num_leaves": 5,
        "verbosity": -1,
    }

    # no warning: no aliases, all defaults
    reg = lgb.LGBMRegressor(**base_kwargs).fit(X, y)
    assert reg.n_estimators_ == 100
    assert_silent(capsys)

    # no warning: no aliases, just n_estimators
    reg = lgb.LGBMRegressor(**base_kwargs, n_estimators=2).fit(X, y)
    assert reg.n_estimators_ == 2
    assert_silent(capsys)

    # no warning: 1 alias + n_estimators (both same value)
    reg = lgb.LGBMRegressor(**base_kwargs, n_estimators=3, n_iter=3).fit(X, y)
    assert reg.n_estimators_ == 3
    assert_silent(capsys)

    # no warning: 1 alias + n_estimators (different values... value from params should win)
    reg = lgb.LGBMRegressor(**base_kwargs, n_estimators=3, n_iter=4).fit(X, y)
    assert reg.n_estimators_ == 4
    assert_silent(capsys)

    # no warning: 2 aliases (both same value)
    reg = lgb.LGBMRegressor(**base_kwargs, n_iter=3, num_iterations=3).fit(X, y)
    assert reg.n_estimators_ == 3
    assert_silent(capsys)

    # no warning: 4 aliases (all same value)
    reg = lgb.LGBMRegressor(**base_kwargs, n_iter=3, num_trees=3, nrounds=3, max_iter=3).fit(X, y)
    assert reg.n_estimators_ == 3
    assert_silent(capsys)

    # warning: 2 aliases (different values... "num_iterations" wins because it's the main param name)
    with pytest.warns(UserWarning, match="LightGBM will perform up to 5 boosting rounds"):
        reg = lgb.LGBMRegressor(**base_kwargs, num_iterations=5, n_iter=6).fit(X, y)
    assert reg.n_estimators_ == 5
    # should not be any other logs (except the warning, intercepted by pytest)
    assert_silent(capsys)

    # warning: 2 aliases (different values... first one in the order from Config::parameter2aliases() wins)
    with pytest.warns(UserWarning, match="LightGBM will perform up to 4 boosting rounds"):
        reg = lgb.LGBMRegressor(**base_kwargs, n_iter=4, max_iter=5).fit(X, y)
    assert reg.n_estimators_ == 4
    # should not be any other logs (except the warning, intercepted by pytest)
    assert_silent(capsys)


@pytest.mark.parametrize("estimator_class", estimator_classes)
def test_cannot_access_feature_names_before_fitting(estimator_class):
    model = estimator_class()
    with pytest.raises(lgb.compat.LGBMNotFittedError):  # noqa: PT011
        model.feature_name_
    with pytest.raises(lgb.compat.LGBMNotFittedError):  # noqa: PT011
        model.feature_names_in_
    with pytest.raises(lgb.compat.LGBMNotFittedError):  # noqa: PT011
        model.n_features_
    with pytest.raises(lgb.compat.LGBMNotFittedError):  # noqa: PT011
        model.n_features_in_


@pytest.mark.parametrize(
    "predict_X_type",
    [
        pytest.param("numpy", id="predict=numpy"),
        pytest.param("pd_DataFrame", id="predict=pd_DataFrame"),
        pytest.param("pa_Table", id="predict=pa_Table"),
        pytest.param("pl_DataFrame", id="predict=pl_DataFrame"),
    ],
)
@pytest.mark.parametrize(
    "fit_X_type",
    [
        pytest.param("numpy", id="fit=numpy"),
        pytest.param("pd_DataFrame", id="fit=pd_DataFrame"),
        pytest.param("pa_Table", id="fit=pa_Table"),
        pytest.param("pl_DataFrame", id="fit=pl_DataFrame"),
    ],
)
@pytest.mark.filterwarnings("error:.*feature name.*:UserWarning:sklearn")
def test_feature_names_in_and_predict_warning(
    predict_X_type,
    fit_X_type,
):
    """Test feature_names_in_ behavior and predict()-time feature name warnings.

    Should cover all combinations of fit X type, feature_name argument, and predict X type.
    Regression test for https://github.com/lightgbm-org/LightGBM/issues/6798.
    """
    if fit_X_type.startswith("pa_") or predict_X_type.startswith("pa_"):
        pa = pytest.importorskip("pyarrow")
        pd = pytest.importorskip("pandas")
    if fit_X_type.startswith("pd_") or predict_X_type.startswith("pd_"):
        pd = pytest.importorskip("pandas")
    if fit_X_type.startswith("pl_") or predict_X_type.startswith("pl_"):
        pl = pytest.importorskip("polars")

    X_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 1, 0, 1])
    n_features = X_np.shape[1]
    col_names = ["feat_0", "feat_1"]
    default_names = ["Column_0", "Column_1"]

    if fit_X_type == "numpy":
        X_fit = X_np
    elif fit_X_type == "pd_DataFrame":
        X_fit = pd.DataFrame(X_np, columns=col_names)
    elif fit_X_type == "pl_DataFrame":
        X_fit = pl.DataFrame(X_np, schema=col_names)
    else:
        X_fit = pa.Table.from_pandas(pd.DataFrame(X_np, columns=col_names))

    if predict_X_type == "numpy":
        X_predict = X_np[:2]
    elif predict_X_type == "pd_DataFrame":
        X_predict = pd.DataFrame(X_np[:2], columns=col_names)
    elif predict_X_type == "pl_DataFrame":
        X_predict = pl.DataFrame(X_np[:2], schema=col_names)
    else:
        X_predict = pa.Table.from_pandas(pd.DataFrame(X_np[:2], columns=col_names))

    # arguments to warnings.filterwarnings() that match scikit-learn's warning
    warning_kwargs = {
        "category": UserWarning,
        "module": "sklearn",
        "message": ".*feature names.*",
    }

    # input types where LightGBM supports 'feature_name="auto"'
    types_with_feat_names = {"pa_Table", "pd_DataFrame", "pl_DataFrame"}

    # case 1: no 'feature_names' passed to fit() and "feature_name='auto'" should have identical behavior
    for fit_kwargs in ({}, {"feature_name": "auto"}):
        model = lgb.LGBMClassifier(n_estimators=2, num_leaves=3).fit(X_fit, y, **fit_kwargs)

        # n_features_in_: always set after fit
        assert model.n_features_in_ == n_features

        # feature_name_: always accessible, reflects actual names used internally
        # feature_names_in_: absent when no named features, present otherwise
        if fit_X_type in types_with_feat_names:
            np_assert_array_equal(model.feature_names_in_, np.array(col_names), strict=True)
            assert model.feature_name_ == col_names
        else:
            assert model.feature_name_ == default_names
            with pytest.raises(AttributeError, match="The training data did not have feature names"):
                model.feature_names_in_

        # predict() should not raise a warning if the input did not have feature names
        if SKLEARN_VERSION_GTE_1_6:
            # fmt:off
            if (
                fit_X_type in types_with_feat_names
                and
                predict_X_type not in types_with_feat_names
            ):
            # fmt:on
                # warning may be raised (and was from at least scikit-learn 1.6)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", **warning_kwargs)
                    model.predict(X_predict)
            else:
                # expect no warning
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", **warning_kwargs)
                    model.predict(X_predict)

    # case 2: 'feature_names=custom_names' (different from column names) passed to fit()
    custom_names = ["custom_0", "custom_1"]
    model = lgb.LGBMClassifier(n_estimators=2, num_leaves=3).fit(X_fit, y, feature_name=custom_names)

    # feature names from keyword arg should be used, not any from the input data
    np_assert_array_equal(model.feature_names_in_, np.array(custom_names), strict=True)
    assert model.feature_name_ == custom_names
    np_assert_array_equal(model.feature_names_in_, np.array(custom_names), strict=True)
    assert model.n_features_in_ == n_features

    # predict() should not raise a warning if input has feature names
    if SKLEARN_VERSION_GTE_1_6:
        if predict_X_type in types_with_feat_names:
            # expect no warning
            with warnings.catch_warnings():
                warnings.filterwarnings("error", **warning_kwargs)
                model.predict(X_predict)
        else:
            # warning may be raised (and was from at least scikit-learn 1.6)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", **warning_kwargs)
                model.predict(X_predict)


# Starting with scikit-learn 1.6 (https://github.com/scikit-learn/scikit-learn/pull/30149),
# the only API for marking estimator tests as expected to fail is to pass a keyword argument
# to parametrize_with_checks(). That function didn't accept additional arguments in earlier
# versions.
#
# This block defines a patched version of parametrize_with_checks() so lightgbm's tests
# can be compatible with scikit-learn <1.6 and >=1.6.
#
# This should be removed once minimum supported scikit-learn version is at least 1.6.
def parametrize_with_checks(estimator, *args, **kwargs):
    if SKLEARN_VERSION_GTE_1_6:
        mark_decorator = sklearn_parametrize_with_checks(estimator, *args, **kwargs)
    else:
        mark_decorator = sklearn_parametrize_with_checks(estimator)

    # Some time around pytest 9.1, 'pytest' started raising a deprecation warning
    # when 'pytest.parametrize()` inputs used generators.
    #
    # see: https://github.com/scikit-learn/scikit-learn/issues/34738
    #
    # scikit-learn's `sklearn.utils.estimator_checks.parametrize_with_checks()`
    # returns a generator of checks.
    #
    # This make's LightGBM's use of `sklearn.utils.estimator_checks.parametrize_with_checks()`
    # compatible with newer and older versions of 'pytest', so tests won't be broken
    # when pytest '10.x' starts raising an exception for those inputs.
    #
    # This workaround can be removed if LightGBM drops its use of parametrize_with_checks()
    # or its minimum scikit-learn version contains https://github.com/scikit-learn/scikit-learn/pull/34448
    argnames, argvalues = mark_decorator.args
    return pytest.mark.parametrize(argnames, list(argvalues), **mark_decorator.kwargs)


def _get_expected_failed_tests(estimator):
    return estimator._more_tags()["_xfail_checks"]


@parametrize_with_checks(
    [ExtendedLGBMClassifier(), ExtendedLGBMRegressor(), lgb.LGBMClassifier(), lgb.LGBMRegressor()],
    expected_failed_checks=_get_expected_failed_tests,
)
def test_sklearn_integration(estimator, check):
    estimator.set_params(min_child_samples=1, min_data_in_bin=1)
    check(estimator)


@pytest.mark.parametrize(
    "method",
    [
        "__init__",
        "get_params",
        "predict",
        "set_params",
    ],
)
def test_estimator_docstrings_that_should_be_identical(method):
    assert_docstrings_equal(lgb.LGBMModel, method, lgb.LGBMClassifier, method)
    assert_docstrings_equal(lgb.LGBMClassifier, method, lgb.LGBMRanker, method)
    assert_docstrings_equal(lgb.LGBMRanker, method, lgb.LGBMRegressor, method)


def test_estimator_fit_docstrings_are_consistent():
    # fit()
    assert_docstrings_equal(
        lgb.LGBMModel,
        "fit",
        lgb.LGBMClassifier,
        "fit",
        expected_diff=textwrap.dedent("""\
            --- LGBMModel.fit
            +++ LGBMClassifier.fit
            @@ -41,13 +40,0 @@
            -group : numpy array, pandas Series, pyarrow ChunkedArray, polars Series, list of int or float, or None, optional (default=None)
            -    Group/query data.
            -    Only used in the learning-to-rank task.
            -    sum(group) = n_samples.
            -    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            -    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            -
            -    .. versionadded:: 4.2.0
            -        Support for ``pyarrow`` inputs
            -
            -    .. versionadded:: 4.7.0
            -        Support for ``polars`` inputs
            -
            @@ -67,2 +53,0 @@
            -eval_group : list of array (same types as ``group`` supports), or None, optional (default=None)
            -    Group data of eval data.
            @@ -105 +90 @@
            -self : LGBMModel
            +self : LGBMClassifier
        """),
    )
    assert_docstrings_equal(
        lgb.LGBMClassifier,
        "fit",
        lgb.LGBMRanker,
        "fit",
        expected_diff=textwrap.dedent("""\
            --- LGBMClassifier.fit
            +++ LGBMRanker.fit
            @@ -40,0 +41,13 @@
            +group : numpy array, pandas Series, pyarrow ChunkedArray, polars Series, list of int or float, or None, optional (default=None)
            +    Group/query data.
            +    Only used in the learning-to-rank task.
            +    sum(group) = n_samples.
            +    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            +    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            +
            +    .. versionadded:: 4.2.0
            +        Support for ``pyarrow`` inputs
            +
            +    .. versionadded:: 4.7.0
            +        Support for ``polars`` inputs
            +
            @@ -50,2 +62,0 @@
            -eval_class_weight : list or None, optional (default=None)
            -    Class weights of eval data.
            @@ -53,0 +65,2 @@
            +eval_group : list of array (same types as ``group`` supports), or None, optional (default=None)
            +    Group data of eval data.
            @@ -59,0 +73,2 @@
            +eval_at : list or tuple of int, optional (default=(1, 2, 3, 4, 5))
            +    The evaluation positions of the specified metric.
            @@ -90 +105 @@
            -self : LGBMClassifier
            +self : LGBMRanker
        """),
    )
    assert_docstrings_equal(
        lgb.LGBMRanker,
        "fit",
        lgb.LGBMRegressor,
        "fit",
        expected_diff=textwrap.dedent("""\
            --- LGBMRanker.fit
            +++ LGBMRegressor.fit
            @@ -41,13 +40,0 @@
            -group : numpy array, pandas Series, pyarrow ChunkedArray, polars Series, list of int or float, or None, optional (default=None)
            -    Group/query data.
            -    Only used in the learning-to-rank task.
            -    sum(group) = n_samples.
            -    For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            -    where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            -
            -    .. versionadded:: 4.2.0
            -        Support for ``pyarrow`` inputs
            -
            -    .. versionadded:: 4.7.0
            -        Support for ``polars`` inputs
            -
            @@ -65,2 +51,0 @@
            -eval_group : list of array (same types as ``group`` supports), or None, optional (default=None)
            -    Group data of eval data.
            @@ -73,2 +57,0 @@
            -eval_at : list or tuple of int, optional (default=(1, 2, 3, 4, 5))
            -    The evaluation positions of the specified metric.
            @@ -105 +88 @@
            -self : LGBMRanker
            +self : LGBMRegressor
        """),
    )


def test_classifier_predict_and_predict_proba_docstrings_are_consistent():
    assert_docstrings_equal(
        lgb.LGBMClassifier,
        "predict",
        lgb.LGBMClassifier,
        "predict_proba",
        expected_diff=textwrap.dedent("""\
            --- LGBMClassifier.predict
            +++ LGBMClassifier.predict_proba
            @@ -1 +1 @@
            -Return the predicted value for each sample.
            +Return the predicted probability for each class for each sample.
            @@ -38 +38 @@
            -predicted_result : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
            +predicted_probability : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
        """),
    )


@pytest.mark.parametrize("estimator_class", estimator_classes)
def test_sklearn_tags_should_correctly_reflect_lightgbm_specific_values(estimator_class):
    est = estimator_class()
    more_tags = est._more_tags()
    err_msg = "List of supported X_types has changed. Update LGBMModel.__sklearn_tags__() to match."
    assert more_tags["X_types"] == ["2darray", "sparse", "1dlabels"], err_msg
    # the try-except part of this should be removed once lightgbm's
    # minimum supported scikit-learn version is at least 1.6
    try:
        sklearn_tags = est.__sklearn_tags__()
    except AttributeError:
        # only the exact error we expected to be raised should be raised
        with pytest.raises(AttributeError, match=r"__sklearn_tags__.* should not be called"):
            est.__sklearn_tags__()
    else:
        # if no AttributeError was thrown, we must be using scikit-learn>=1.6,
        # and so the actual effects of __sklearn_tags__() should be tested
        assert sklearn_tags.input_tags.allow_nan is True
        assert sklearn_tags.input_tags.sparse is True
        assert sklearn_tags.target_tags.one_d_labels is True
        if estimator_class is lgb.LGBMClassifier:
            assert sklearn_tags.estimator_type == "classifier"
            assert sklearn_tags.classifier_tags.multi_class is True
            assert sklearn_tags.classifier_tags.multi_label is False
        elif estimator_class is lgb.LGBMRegressor:
            assert sklearn_tags.estimator_type == "regressor"


@pytest.mark.parametrize("task", all_tasks)
def test_training_succeeds_when_data_is_dataframe_and_label_is_column_array(task):
    pd = pytest.importorskip("pandas")
    X, y, g = _create_data(task)
    X = pd.DataFrame(X)
    y_col_array = y.reshape(-1, 1)
    params = {"n_estimators": 1, "num_leaves": 3, "random_state": 0}
    model_factory = task_to_model_factory[task]
    if task == "ranking":
        model_1d = model_factory(**params).fit(X, y, group=g)
        with pytest.warns(UserWarning, match="column-vector"):
            model_2d = model_factory(**params).fit(X, y_col_array, group=g)
    else:
        model_1d = model_factory(**params).fit(X, y)
        with pytest.warns(UserWarning, match="column-vector"):
            model_2d = model_factory(**params).fit(X, y_col_array)

    preds_1d = model_1d.predict(X)
    preds_2d = model_2d.predict(X)
    np_assert_array_equal(preds_1d, preds_2d, strict=True)


@pytest.mark.parametrize("use_weight", [True, False])
def test_multiclass_custom_objective(use_weight):
    centers = [[-4, -4], [4, 4], [-4, 4]]
    X, y = make_blobs(n_samples=1_000, centers=centers, random_state=42)
    weight = np.full_like(y, 2) if use_weight else None
    params = {"n_estimators": 10, "num_leaves": 7}
    builtin_obj_model = lgb.LGBMClassifier(**params)
    builtin_obj_model.fit(X, y, sample_weight=weight)
    builtin_obj_preds = builtin_obj_model.predict_proba(X)

    custom_obj_model = lgb.LGBMClassifier(objective=sklearn_multiclass_custom_objective, **params)
    custom_obj_model.fit(X, y, sample_weight=weight)
    custom_obj_preds = softmax(custom_obj_model.predict(X, raw_score=True))

    np.testing.assert_allclose(builtin_obj_preds, custom_obj_preds, rtol=0.01)
    assert not callable(builtin_obj_model.objective_)
    assert callable(custom_obj_model.objective_)


@pytest.mark.parametrize("use_weight", [True, False])
def test_multiclass_custom_eval(use_weight):
    def custom_eval(y_true, y_pred, weight):
        loss = log_loss(y_true, y_pred, sample_weight=weight)
        return "custom_logloss", loss, False

    centers = [[-4, -4], [4, 4], [-4, 4]]
    X, y = make_blobs(n_samples=1_000, centers=centers, random_state=42)
    train_test_split_func = partial(train_test_split, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split_func(X, y)
    if use_weight:
        weight = np.full_like(y, 2)
        weight_train, weight_valid = train_test_split_func(weight)
    else:
        weight_train = None
        weight_valid = None
    params = {"objective": "multiclass", "num_class": 3, "num_leaves": 7}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_names=["train", "valid"],
        eval_sample_weight=[weight_train, weight_valid],
        eval_metric=custom_eval,
    )
    eval_result = model.evals_result_
    train_ds = (X_train, y_train, weight_train)
    valid_ds = (X_valid, y_valid, weight_valid)
    for key, (X, y_true, weight) in zip(["train", "valid"], [train_ds, valid_ds], strict=True):
        np.testing.assert_allclose(eval_result[key]["multi_logloss"], eval_result[key]["custom_logloss"])
        y_pred = model.predict_proba(X)
        _, metric_value, _ = custom_eval(y_true, y_pred, weight)
        np.testing.assert_allclose(metric_value, eval_result[key]["custom_logloss"][-1])


def test_negative_n_jobs(tmp_path):
    n_threads = joblib.cpu_count()
    if n_threads <= 1:
        return None
    # 'val_minus_two' here is the expected number of threads for n_jobs=-2
    val_minus_two = n_threads - 1
    X, y = load_breast_cancer(return_X_y=True)
    # Note: according to joblib's formula, a value of n_jobs=-2 means
    # "use all but one thread" (formula: n_cpus + 1 + n_jobs)
    gbm = lgb.LGBMClassifier(n_estimators=2, verbose=-1, n_jobs=-2).fit(X, y)
    gbm.booster_.save_model(tmp_path / "model.txt")
    with open(tmp_path / "model.txt", "r") as f:
        model_txt = f.read()
    assert bool(re.search(rf"\[num_threads: {val_minus_two}\]", model_txt))


def test_default_n_jobs(tmp_path):
    n_cores = joblib.cpu_count(only_physical_cores=True)
    X, y = load_breast_cancer(return_X_y=True)
    gbm = lgb.LGBMClassifier(n_estimators=2, verbose=-1, n_jobs=None).fit(X, y)
    gbm.booster_.save_model(tmp_path / "model.txt")
    with open(tmp_path / "model.txt", "r") as f:
        model_txt = f.read()
    assert bool(re.search(rf"\[num_threads: {n_cores}\]", model_txt))


@pytest.mark.skipif(not PANDAS_INSTALLED, reason="pandas is not installed")
@pytest.mark.parametrize("task", all_tasks)
def test_validate_features(task):
    X, y, g = _create_data(task, n_features=4)
    features = ["x1", "x2", "x3", "x4"]
    df = pd_DataFrame(X, columns=features)
    model = task_to_model_factory[task](n_estimators=10, num_leaves=15, verbose=-1)
    if task == "ranking":
        model.fit(df, y, group=g)
    else:
        model.fit(df, y)
    assert model.feature_name_ == features

    # try to predict with a different feature
    df2 = df.rename(columns={"x2": "z"})
    with pytest.raises(lgb.basic.LightGBMError, match="Expected 'x2' at position 1 but found 'z'"):
        model.predict(df2, validate_features=True)

    # check that disabling the check doesn't raise the error
    model.predict(df2, validate_features=False)


# LightGBM's 'predict_disable_shape_check' mechanism is intentionally not respected by
# its scikit-learn estimators, for consistency with scikit-learn's own behavior.
@pytest.mark.parametrize("task", all_tasks)
@pytest.mark.parametrize("predict_disable_shape_check", [True, False])
def test_predict_rejects_inputs_with_incorrect_number_of_features(predict_disable_shape_check, task):
    X, y, g = _create_data(task, n_features=4)
    model_factory = task_to_model_factory[task]
    fit_kwargs = {"X": X[:, :-1], "y": y}
    if task == "ranking":
        estimator_name = "LGBMRanker"
        fit_kwargs.update({"group": g})
    elif task == "regression":
        estimator_name = "LGBMRegressor"
    else:
        estimator_name = "LGBMClassifier"

    # train on the first 3 features
    model = model_factory(n_estimators=5, num_leaves=7, verbose=-1).fit(**fit_kwargs)

    # more cols in X than features: error
    err_msg = f"X has 4 features, but {estimator_name} is expecting 3 features as input"
    with pytest.raises(ValueError, match=err_msg):
        model.predict(X, predict_disable_shape_check=predict_disable_shape_check)

    if estimator_name == "LGBMClassifier":
        with pytest.raises(ValueError, match=err_msg):
            model.predict_proba(X, predict_disable_shape_check=predict_disable_shape_check)

    # fewer cols in X than features: error
    err_msg = f"X has 2 features, but {estimator_name} is expecting 3 features as input"
    with pytest.raises(ValueError, match=err_msg):
        model.predict(X[:, :-2], predict_disable_shape_check=predict_disable_shape_check)

    if estimator_name == "LGBMClassifier":
        with pytest.raises(ValueError, match=err_msg):
            model.predict_proba(X[:, :-2], predict_disable_shape_check=predict_disable_shape_check)

    # same number of columns in both: no error
    preds = model.predict(X[:, :-1], predict_disable_shape_check=predict_disable_shape_check)
    assert preds.shape == y.shape

    if estimator_name == "LGBMClassifier":
        preds = model.predict_proba(X[:, :-1], predict_disable_shape_check=predict_disable_shape_check)
        assert preds.shape[0] == y.shape[0]


def _run_minimal_test(*, X_type, y_type, g_type, task, rng):
    if any(t.startswith("pa_") for t in [X_type, y_type, g_type]):
        pa = pytest.importorskip("pyarrow")
    if any(t.startswith("pl_") for t in [X_type, y_type, g_type]):
        pl = pytest.importorskip("polars")

    X, y, g = _create_data(task, n_samples=2_000)
    weights = np.abs(rng.standard_normal(size=(y.shape[0],)))

    if task in {"binary-classification", "regression", "ranking"}:
        init_score = np.full_like(y, np.mean(y))
    elif task == "multiclass-classification":
        init_score = np.outer(y, np.array([0.1, 0.2, 0.7]))
    else:
        raise ValueError(f"Unrecognized task '{task}'")

    X_valid = X * 2
    if X_type == "list2d":
        X = X.tolist()
    elif X_type == "scipy_csc":
        X = scipy.sparse.csc_matrix(X)
    elif X_type == "scipy_csr":
        X = scipy.sparse.csr_matrix(X)
    elif X_type == "pd_DataFrame":
        X = pd_DataFrame(X)
    elif X_type == "pa_Table":
        X = pa.Table.from_pandas(pd_DataFrame(X))
    elif X_type == "pl_DataFrame":
        X = pl.DataFrame(X)
    elif X_type != "numpy":
        raise ValueError(f"Unrecognized X_type: '{X_type}'")

    # make weights and init_score same types as y, just to avoid
    # a huge number of combinations and therefore test cases
    if y_type == "list1d":
        y = y.tolist()
        weights = weights.tolist()
        init_score = init_score.tolist()
    elif y_type == "pd_DataFrame":
        y = pd_DataFrame(y)
        weights = pd_Series(weights)
        if task == "multiclass-classification":
            init_score = pd_DataFrame(init_score)
        else:
            init_score = pd_Series(init_score)
    elif y_type == "pd_Series":
        y = pd_Series(y)
        weights = pd_Series(weights)
        if task == "multiclass-classification":
            init_score = pd_DataFrame(init_score)
        else:
            init_score = pd_Series(init_score)
    elif y_type == "pa_ChunkedArray":
        y = pa.chunked_array([y])
        weights = pa.chunked_array([weights])
        if task == "multiclass-classification":
            init_score = pa.Table.from_pandas(pd_DataFrame(init_score))
        else:
            init_score = pa.chunked_array([init_score])
    elif y_type == "pl_Series":
        y = pl.Series(y)
        weights = pl.Series(weights)
        if task == "multiclass-classification":
            init_score = pl.DataFrame(init_score)
        else:
            init_score = pl.Series(init_score)
    elif y_type != "numpy":
        raise ValueError(f"Unrecognized y_type: '{y_type}'")

    if g_type == "list1d_float":
        g = g.astype("float").tolist()
    elif g_type == "list1d_int":
        g = g.astype("int").tolist()
    elif g_type == "pd_Series":
        g = pd_Series(g)
    elif g_type == "pa_ChunkedArray":
        g = pa.chunked_array([g])
    elif g_type == "pl_Series":
        g = pl.Series(g)
    elif g_type != "numpy":
        raise ValueError(f"Unrecognized g_type: '{g_type}'")

    model = task_to_model_factory[task](n_estimators=10, verbose=-1)
    params_fit = {
        "X": X,
        "y": y,
        "sample_weight": weights,
        "init_score": init_score,
        "eval_set": [(X_valid, y)],
        "eval_sample_weight": [weights],
        "eval_init_score": [init_score],
    }
    if task == "ranking":
        params_fit["group"] = g
        params_fit["eval_group"] = [g]
    model.fit(**params_fit)

    # --- prediction accuracy --#
    preds = model.predict(X)
    if task == "binary-classification":
        assert accuracy_score(y, preds) >= 0.99
    elif task == "multiclass-classification":
        assert accuracy_score(y, preds) >= 0.99
    elif task == "regression":
        assert r2_score(y, preds) > 0.86
    elif task == "ranking":
        assert spearmanr(preds, y).correlation >= 0.99
    else:
        raise ValueError(f"Unrecognized task: '{task}'")

    # --- prediction dtypes ---#

    # default predictions:
    #
    #  * classification: int32 or int64
    #  * ranking: float64
    #  * regression: float64
    #
    if task.endswith("classification"):
        # preds go through LabelEncoder.inverse_transform() and have the same
        # dtype as model.classes_ (expected to be an integer type, but exact size
        # varies across numpy versions and operating systems)
        assert preds.dtype == model.classes_.dtype
        assert preds.dtype in (np.int32, np.int64)
    else:
        assert preds.dtype == np.float64

    # raw predictions: always float64
    preds_raw = model.predict(X, raw_score=True)
    assert preds_raw.dtype == np.float64

    # pred_contrib: always float64
    if X_type.startswith("scipy"):
        assert all(arr.dtype == np.float64 for arr in model.predict(X, pred_contrib=True))
    else:
        preds_contrib = model.predict(X, pred_contrib=True)
        assert preds_contrib.dtype == np.float64

    # pred_leavs: always int32
    preds_leaves = model.predict(X, pred_leaf=True)
    assert preds_leaves.dtype == np.int32


@pytest.mark.parametrize("X_type", all_x_types)
@pytest.mark.parametrize("y_type", all_y_types)
@pytest.mark.parametrize("task", [t for t in all_tasks if t != "ranking"])
def test_classification_and_regression_minimally_work_with_all_accepted_data_types(
    X_type,
    y_type,
    task,
    rng,
):
    if any(t.startswith("pd_") for t in [X_type, y_type]) and not PANDAS_INSTALLED:
        pytest.skip("pandas is not installed")
    _run_minimal_test(X_type=X_type, y_type=y_type, g_type="numpy", task=task, rng=rng)


@pytest.mark.parametrize("X_type", all_x_types)
@pytest.mark.parametrize("y_type", all_y_types)
@pytest.mark.parametrize("g_type", all_group_types)
def test_ranking_minimally_works_with_all_accepted_data_types(
    X_type,
    y_type,
    g_type,
    rng,
):
    if any(t.startswith("pd_") for t in [X_type, y_type, g_type]) and not PANDAS_INSTALLED:
        pytest.skip("pandas is not installed")
    _run_minimal_test(X_type=X_type, y_type=y_type, g_type=g_type, task="ranking", rng=rng)


def test_classifier_fit_detects_classes_every_time():
    rng = np.random.default_rng(seed=123)
    nrows = 1000
    ncols = 20

    X = rng.standard_normal(size=(nrows, ncols))
    y_bin = (rng.random(size=nrows) <= 0.3).astype(np.float64)
    y_multi = rng.integers(4, size=nrows)

    model = lgb.LGBMClassifier(verbose=-1)
    for _ in range(2):
        model.fit(X, y_multi)
        assert model.objective_ == "multiclass"
        model.fit(X, y_bin)
        assert model.objective_ == "binary"


def test_eval_set_deprecation():
    """Test use of eval_set raises deprecation warning."""
    X, y = make_synthetic_regression(n_samples=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    gbm = lgb.LGBMRegressor()
    msg = "The argument 'eval_set' is deprecated.*"
    with pytest.warns(LGBMDeprecationWarning, match=msg):
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])


def test_eval_set_raises():
    """Test that eval_set and eval_X raise errors where appropriate."""
    X, y = make_synthetic_regression(n_samples=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    gbm = lgb.LGBMRegressor()

    msg = "Specify either 'eval_set' or 'eval_X'.*"
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_X=X_test)
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_y=y_test)

    msg = "You must specify eval_X and eval_y, not just one of them."
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X_train, y_train, eval_X=X_test)
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X_train, y_train, eval_y=y_test)

    msg = "If eval_X is a tuple, y_val must be a tuple of same length, and vice versa."
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X_train, y_train, eval_X=(X_test, X_test), eval_y=y_test)
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X_train, y_train, eval_X=X_test, eval_y=(y_test, y_test))
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X_train, y_train, eval_X=(X_test,) * 3, eval_y=(y_test,) * 2)


def test_ranker_eval_set_raises():
    """Test that LGBMRanker raises expected errors from validating eval_group."""
    X, y, g = _create_data(task="ranking", n_samples=1_000)
    X_test, y_test, g_test = _create_data(task="ranking", n_samples=100)
    gbm = lgb.LGBMRanker()

    msg = "eval_group cannot be None if any of eval_set, eval_X, or eval_y are provided"
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X, y, group=g, eval_set=[(X_test, y_test)])
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X, y, group=g, eval_X=(X_test,))
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X, y, group=g, eval_y=(y_test,))
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X, y, group=g, eval_X=(X_test,), eval_y=(y_test,))

    msg = re.escape("Length of eval_group (1) not equal to length of eval_set (2)")
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X, y, group=g, eval_X=(X_test, X_test), eval_y=(y_test, y_test), eval_group=[g_test])
    with pytest.raises(ValueError, match=msg):
        gbm.fit(X, y, group=g, eval_set=[(X_test, y_test), (X_test, y_test)], eval_group=[g_test])


def test_eval_X_eval_y_eval_set_equivalence():
    """Test that eval_X and eval_y are equivalent to eval_set."""
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    params = {
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "seed": 708,
    }
    cbs = [lgb.early_stopping(2)]
    gbm1 = lgb.LGBMRegressor(**params)
    with pytest.warns(LGBMDeprecationWarning, match="The argument 'eval_set' is deprecated.*"):
        gbm1.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=cbs)
    gbm2 = lgb.LGBMRegressor(**params)
    gbm2.fit(X_train, y_train, eval_X=X_test, eval_y=y_test, callbacks=cbs)
    np.testing.assert_allclose(gbm1.predict(X), gbm2.predict(X))
    assert gbm1.evals_result_["valid_0"]["l2"][0] == pytest.approx(gbm2.evals_result_["valid_0"]["l2"][0])
    # 2 evaluation sets
    n = X_test.shape[0]
    X_test1, X_test2 = X_test[: n // 2], X_test[n // 2 :]
    y_test1, y_test2 = y_test[: n // 2], y_test[n // 2 :]
    gbm1 = lgb.LGBMRegressor(**params)
    with pytest.warns(LGBMDeprecationWarning, match="The argument 'eval_set' is deprecated.*"):
        gbm1.fit(X_train, y_train, eval_set=[(X_test1, y_test1), (X_test2, y_test2)], callbacks=cbs)
    gbm2 = lgb.LGBMRegressor(**params)
    gbm2.fit(X_train, y_train, eval_X=(X_test1, X_test2), eval_y=(y_test1, y_test2), callbacks=cbs)
    np.testing.assert_allclose(gbm1.predict(X), gbm2.predict(X))
    assert set(gbm2.evals_result_.keys()) == {"valid_0", "valid_1"}, (
        f"expected 2 validation sets, 'valid_0' and 'valid_1', in evals_result_, got {gbm2.evals_result_.keys()}"
    )
    assert gbm1.evals_result_["valid_0"]["l2"][0] == pytest.approx(gbm2.evals_result_["valid_0"]["l2"][0])
    assert gbm1.evals_result_["valid_1"]["l2"][0] == pytest.approx(gbm2.evals_result_["valid_1"]["l2"][0])
    assert gbm2.evals_result_["valid_0"]["l2"] != gbm2.evals_result_["valid_1"]["l2"], (
        "Evaluation results for the 2 validation sets are not different. This might mean they weren't both used."
    )


@pytest.mark.parametrize(
    "classes",
    [
        pytest.param(np.array(["down", "up"]), id="string-labels"),
        pytest.param(np.array([1, 2]), id="non-zero-based-integer-labels"),
        pytest.param(np.array([0, 2, 7, 10]), id="non-contiguous-integer-labels"),
    ],
)
@pytest.mark.parametrize("pass_as_tuples", [False, True])
def test_classifier_eval_X_eval_y_encodes_labels(classes, pass_as_tuples, rng):
    """Test that eval_y labels use the classifier's label encoding."""
    X = rng.uniform(low=0.01, high=0.06, size=(1_000, 3))
    y = rng.choice(classes, size=(X.shape[0],))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if pass_as_tuples:
        eval_X = (X_test,)
        eval_y = (y_test,)
    else:
        eval_X = X_test
        eval_y = y_test

    with mock.patch.object(lgb.LGBMModel, "fit") as mock_fit:
        model = lgb.LGBMClassifier().fit(
            X_train,
            y_train,
            eval_X=eval_X,
            eval_y=eval_y,
        )

    mock_fit.assert_called_once()
    forwarded_eval_y = mock_fit.call_args.kwargs["eval_y"]
    expected_eval_y = model._le.transform(y_test)
    if pass_as_tuples:
        assert isinstance(forwarded_eval_y, tuple)
        assert len(forwarded_eval_y) == 1
        np.testing.assert_array_equal(forwarded_eval_y[0], expected_eval_y)
    else:
        np.testing.assert_array_equal(forwarded_eval_y, expected_eval_y)
