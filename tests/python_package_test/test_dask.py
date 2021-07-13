# coding: utf-8
"""Tests for lightgbm.dask module"""

import inspect
import pickle
import random
import socket
from itertools import groupby
from os import getenv
from sys import platform

import pytest

import lightgbm as lgb

if not platform.startswith('linux'):
    pytest.skip('lightgbm.dask is currently supported in Linux environments', allow_module_level=True)
if not lgb.compat.DASK_INSTALLED:
    pytest.skip('Dask is not installed', allow_module_level=True)

import cloudpickle
import dask.array as da
import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
import sklearn.utils.estimator_checks as sklearn_checks
from dask.array.utils import assert_eq
from dask.distributed import Client, LocalCluster, default_client, wait
from pkg_resources import parse_version
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import spearmanr
from sklearn import __version__ as sk_version
from sklearn.datasets import make_blobs, make_regression

from .utils import make_ranking

sk_version = parse_version(sk_version)

tasks = ['binary-classification', 'multiclass-classification', 'regression', 'ranking']
distributed_training_algorithms = ['data', 'voting']
data_output = ['array', 'scipy_csr_matrix', 'dataframe', 'dataframe-with-categorical']
boosting_types = ['gbdt', 'dart', 'goss', 'rf']
group_sizes = [5, 5, 5, 10, 10, 10, 20, 20, 20, 50, 50]
task_to_dask_factory = {
    'regression': lgb.DaskLGBMRegressor,
    'binary-classification': lgb.DaskLGBMClassifier,
    'multiclass-classification': lgb.DaskLGBMClassifier,
    'ranking': lgb.DaskLGBMRanker
}
task_to_local_factory = {
    'regression': lgb.LGBMRegressor,
    'binary-classification': lgb.LGBMClassifier,
    'multiclass-classification': lgb.LGBMClassifier,
    'ranking': lgb.LGBMRanker
}

pytestmark = [
    pytest.mark.skipif(getenv('TASK', '') == 'mpi', reason='Fails to run with MPI interface'),
    pytest.mark.skipif(getenv('TASK', '') == 'gpu', reason='Fails to run with GPU interface')
]


@pytest.fixture(scope='module')
def cluster():
    dask_cluster = LocalCluster(n_workers=2, threads_per_worker=2, dashboard_address=None)
    yield dask_cluster
    dask_cluster.close()


@pytest.fixture(scope='module')
def cluster2():
    dask_cluster = LocalCluster(n_workers=2, threads_per_worker=2, dashboard_address=None)
    yield dask_cluster
    dask_cluster.close()


@pytest.fixture()
def listen_port():
    listen_port.port += 10
    return listen_port.port


listen_port.port = 13000


def _create_ranking_data(n_samples=100, output='array', chunk_size=50, **kwargs):
    X, y, g = make_ranking(n_samples=n_samples, random_state=42, **kwargs)
    rnd = np.random.RandomState(42)
    w = rnd.rand(X.shape[0]) * 0.01
    g_rle = np.array([len(list(grp)) for _, grp in groupby(g)])

    if output.startswith('dataframe'):
        # add target, weight, and group to DataFrame so that partitions abide by group boundaries.
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if output == 'dataframe-with-categorical':
            for i in range(5):
                col_name = f"cat_col{i}"
                cat_values = rnd.choice(['a', 'b'], X.shape[0])
                cat_series = pd.Series(
                    cat_values,
                    dtype='category'
                )
                X_df[col_name] = cat_series
        X = X_df.copy()
        X_df = X_df.assign(y=y, g=g, w=w)

        # set_index ensures partitions are based on group id.
        # See https://stackoverflow.com/questions/49532824/dask-dataframe-split-partitions-based-on-a-column-or-function.
        X_df.set_index('g', inplace=True)
        dX = dd.from_pandas(X_df, chunksize=chunk_size)

        # separate target, weight from features.
        dy = dX['y']
        dw = dX['w']
        dX = dX.drop(columns=['y', 'w'])
        dg = dX.index.to_series()

        # encode group identifiers into run-length encoding, the format LightGBMRanker is expecting
        # so that within each partition, sum(g) = n_samples.
        dg = dg.map_partitions(lambda p: p.groupby('g', sort=False).apply(lambda z: z.shape[0]))
    elif output == 'array':
        # ranking arrays: one chunk per group. Each chunk must include all columns.
        p = X.shape[1]
        dX, dy, dw, dg = [], [], [], []
        for g_idx, rhs in enumerate(np.cumsum(g_rle)):
            lhs = rhs - g_rle[g_idx]
            dX.append(da.from_array(X[lhs:rhs, :], chunks=(rhs - lhs, p)))
            dy.append(da.from_array(y[lhs:rhs]))
            dw.append(da.from_array(w[lhs:rhs]))
            dg.append(da.from_array(np.array([g_rle[g_idx]])))

        dX = da.concatenate(dX, axis=0)
        dy = da.concatenate(dy, axis=0)
        dw = da.concatenate(dw, axis=0)
        dg = da.concatenate(dg, axis=0)
    else:
        raise ValueError('Ranking data creation only supported for Dask arrays and dataframes')

    return X, y, w, g_rle, dX, dy, dw, dg


def _create_data(objective, n_samples=1_000, output='array', chunk_size=500, **kwargs):
    if objective.endswith('classification'):
        if objective == 'binary-classification':
            centers = [[-4, -4], [4, 4]]
        elif objective == 'multiclass-classification':
            centers = [[-4, -4], [4, 4], [-4, 4]]
        else:
            raise ValueError(f"Unknown classification task '{objective}'")
        X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42)
    elif objective == 'regression':
        X, y = make_regression(n_samples=n_samples, n_features=4, n_informative=2, random_state=42)
    elif objective == 'ranking':
        return _create_ranking_data(
            n_samples=n_samples,
            output=output,
            chunk_size=chunk_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown objective '{objective}'")
    rnd = np.random.RandomState(42)
    weights = rnd.random(X.shape[0]) * 0.01

    if output == 'array':
        dX = da.from_array(X, (chunk_size, X.shape[1]))
        dy = da.from_array(y, chunk_size)
        dw = da.from_array(weights, chunk_size)
    elif output.startswith('dataframe'):
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if output == 'dataframe-with-categorical':
            num_cat_cols = 2
            for i in range(num_cat_cols):
                col_name = f"cat_col{i}"
                cat_values = rnd.choice(['a', 'b'], X.shape[0])
                cat_series = pd.Series(
                    cat_values,
                    dtype='category'
                )
                X_df[col_name] = cat_series
                X = np.hstack((X, cat_series.cat.codes.values.reshape(-1, 1)))

            # make one categorical feature relevant to the target
            cat_col_is_a = X_df['cat_col0'] == 'a'
            if objective == 'regression':
                y = np.where(cat_col_is_a, y, 2 * y)
            elif objective == 'binary-classification':
                y = np.where(cat_col_is_a, y, 1 - y)
            elif objective == 'multiclass-classification':
                n_classes = 3
                y = np.where(cat_col_is_a, y, (1 + y) % n_classes)
        y_df = pd.Series(y, name='target')
        dX = dd.from_pandas(X_df, chunksize=chunk_size)
        dy = dd.from_pandas(y_df, chunksize=chunk_size)
        dw = dd.from_array(weights, chunksize=chunk_size)
    elif output == 'scipy_csr_matrix':
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(csr_matrix)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(weights, chunk_size)
        X = csr_matrix(X)
    elif output == 'scipy_csc_matrix':
        dX = da.from_array(X, chunks=(chunk_size, X.shape[1])).map_blocks(csc_matrix)
        dy = da.from_array(y, chunks=chunk_size)
        dw = da.from_array(weights, chunk_size)
        X = csc_matrix(X)
    else:
        raise ValueError(f"Unknown output type '{output}'")

    return X, y, weights, None, dX, dy, dw, None


def _r2_score(dy_true, dy_pred):
    numerator = ((dy_true - dy_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((dy_true - dy_true.mean(axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    return (1 - numerator / denominator).compute()


def _accuracy_score(dy_true, dy_pred):
    return da.average(dy_true == dy_pred).compute()


def _constant_metric(dy_true, dy_pred):
    metric_name = 'constant_metric'
    value = 0.708
    is_higher_better = False
    return metric_name, value, is_higher_better


def _pickle(obj, filepath, serializer):
    if serializer == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif serializer == 'joblib':
        joblib.dump(obj, filepath)
    elif serializer == 'cloudpickle':
        with open(filepath, 'wb') as f:
            cloudpickle.dump(obj, f)
    else:
        raise ValueError(f'Unrecognized serializer type: {serializer}')


def _unpickle(filepath, serializer):
    if serializer == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif serializer == 'joblib':
        return joblib.load(filepath)
    elif serializer == 'cloudpickle':
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)
    else:
        raise ValueError(f'Unrecognized serializer type: {serializer}')


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('task', ['binary-classification', 'multiclass-classification'])
@pytest.mark.parametrize('boosting_type', boosting_types)
@pytest.mark.parametrize('tree_learner', distributed_training_algorithms)
def test_classifier(output, task, boosting_type, tree_learner, cluster):
    with Client(cluster) as client:
        X, y, w, _, dX, dy, dw, _ = _create_data(
            objective=task,
            output=output
        )

        params = {
            "boosting_type": boosting_type,
            "tree_learner": tree_learner,
            "n_estimators": 50,
            "num_leaves": 31
        }
        if boosting_type == 'rf':
            params.update({
                'bagging_freq': 1,
                'bagging_fraction': 0.9,
            })
        elif boosting_type == 'goss':
            params['top_rate'] = 0.5

        dask_classifier = lgb.DaskLGBMClassifier(
            client=client,
            time_out=5,
            **params
        )
        dask_classifier = dask_classifier.fit(dX, dy, sample_weight=dw)
        p1 = dask_classifier.predict(dX)
        p1_raw = dask_classifier.predict(dX, raw_score=True).compute()
        p1_first_iter_raw = dask_classifier.predict(dX, start_iteration=0, num_iteration=1, raw_score=True).compute()
        p1_early_stop_raw = dask_classifier.predict(
            dX,
            pred_early_stop=True,
            pred_early_stop_margin=1.0,
            pred_early_stop_freq=2,
            raw_score=True
        ).compute()
        p1_proba = dask_classifier.predict_proba(dX).compute()
        p1_pred_leaf = dask_classifier.predict(dX, pred_leaf=True)
        p1_local = dask_classifier.to_local().predict(X)
        s1 = _accuracy_score(dy, p1)
        p1 = p1.compute()

        local_classifier = lgb.LGBMClassifier(**params)
        local_classifier.fit(X, y, sample_weight=w)
        p2 = local_classifier.predict(X)
        p2_proba = local_classifier.predict_proba(X)
        s2 = local_classifier.score(X, y)

        if boosting_type == 'rf':
            # https://github.com/microsoft/LightGBM/issues/4118
            assert_eq(s1, s2, atol=0.01)
            assert_eq(p1_proba, p2_proba, atol=0.8)
        else:
            assert_eq(s1, s2)
            assert_eq(p1, p2)
            assert_eq(p1, y)
            assert_eq(p2, y)
            assert_eq(p1_proba, p2_proba, atol=0.03)
            assert_eq(p1_local, p2)
            assert_eq(p1_local, y)

        # extra predict() parameters should be passed through correctly
        with pytest.raises(AssertionError):
            assert_eq(p1_raw, p1_first_iter_raw)

        with pytest.raises(AssertionError):
            assert_eq(p1_raw, p1_early_stop_raw)

        # pref_leaf values should have the right shape
        # and values that look like valid tree nodes
        pred_leaf_vals = p1_pred_leaf.compute()
        assert pred_leaf_vals.shape == (
            X.shape[0],
            dask_classifier.booster_.num_trees()
        )
        assert np.max(pred_leaf_vals) <= params['num_leaves']
        assert np.min(pred_leaf_vals) >= 0
        assert len(np.unique(pred_leaf_vals)) <= params['num_leaves']

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns
                if dX.dtypes[col].name == 'category'
            ]
            tree_df = dask_classifier.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[0] == '=='


@pytest.mark.parametrize('output', data_output + ['scipy_csc_matrix'])
@pytest.mark.parametrize('task', ['binary-classification', 'multiclass-classification'])
def test_classifier_pred_contrib(output, task, cluster):
    with Client(cluster) as client:
        X, y, w, _, dX, dy, dw, _ = _create_data(
            objective=task,
            output=output
        )

        params = {
            "n_estimators": 10,
            "num_leaves": 10
        }

        dask_classifier = lgb.DaskLGBMClassifier(
            client=client,
            time_out=5,
            tree_learner='data',
            **params
        )
        dask_classifier = dask_classifier.fit(dX, dy, sample_weight=dw)
        preds_with_contrib = dask_classifier.predict(dX, pred_contrib=True)

        local_classifier = lgb.LGBMClassifier(**params)
        local_classifier.fit(X, y, sample_weight=w)
        local_preds_with_contrib = local_classifier.predict(X, pred_contrib=True)

        # shape depends on whether it is binary or multiclass classification
        num_features = dask_classifier.n_features_
        num_classes = dask_classifier.n_classes_
        if num_classes == 2:
            expected_num_cols = num_features + 1
        else:
            expected_num_cols = (num_features + 1) * num_classes

        # in the special case of multi-class classification using scipy sparse matrices,
        # the output of `.predict(..., pred_contrib=True)` is a list of sparse matrices (one per class)
        #
        # since that case is so different than all other cases, check the relevant things here
        # and then return early
        if output.startswith('scipy') and task == 'multiclass-classification':
            if output == 'scipy_csr_matrix':
                expected_type = csr_matrix
            elif output == 'scipy_csc_matrix':
                expected_type = csc_matrix
            else:
                raise ValueError(f"Unrecognized output type: {output}")
            assert isinstance(preds_with_contrib, list)
            assert all(isinstance(arr, da.Array) for arr in preds_with_contrib)
            assert all(isinstance(arr._meta, expected_type) for arr in preds_with_contrib)
            assert len(preds_with_contrib) == num_classes
            assert len(preds_with_contrib) == len(local_preds_with_contrib)
            for i in range(num_classes):
                computed_preds = preds_with_contrib[i].compute()
                assert isinstance(computed_preds, expected_type)
                assert computed_preds.shape[1] == num_classes
                assert computed_preds.shape == local_preds_with_contrib[i].shape
                assert len(np.unique(computed_preds[:, -1])) == 1
                # raw scores will probably be different, but at least check that all predicted classes are the same
                pred_classes = np.argmax(computed_preds.toarray(), axis=1)
                local_pred_classes = np.argmax(local_preds_with_contrib[i].toarray(), axis=1)
                np.testing.assert_array_equal(pred_classes, local_pred_classes)
            return

        preds_with_contrib = preds_with_contrib.compute()
        if output.startswith('scipy'):
            preds_with_contrib = preds_with_contrib.toarray()

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns
                if dX.dtypes[col].name == 'category'
            ]
            tree_df = dask_classifier.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[0] == '=='

        # * shape depends on whether it is binary or multiclass classification
        # * matrix for binary classification is of the form [feature_contrib, base_value],
        #   for multi-class it's [feat_contrib_class1, base_value_class1, feat_contrib_class2, base_value_class2, etc.]
        # * contrib outputs for distributed training are different than from local training, so we can just test
        #   that the output has the right shape and base values are in the right position
        assert preds_with_contrib.shape[1] == expected_num_cols
        assert preds_with_contrib.shape == local_preds_with_contrib.shape

        if num_classes == 2:
            assert len(np.unique(preds_with_contrib[:, num_features])) == 1
        else:
            for i in range(num_classes):
                base_value_col = num_features * (i + 1) + i
                assert len(np.unique(preds_with_contrib[:, base_value_col]) == 1)


def test_find_random_open_port(cluster):
    with Client(cluster) as client:
        for _ in range(5):
            worker_address_to_port = client.run(lgb.dask._find_random_open_port)
            found_ports = worker_address_to_port.values()
            # check that found ports are different for same address (LocalCluster)
            assert len(set(found_ports)) == len(found_ports)
            # check that the ports are indeed open
            for port in found_ports:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))


def test_possibly_fix_worker_map(capsys, cluster):
    with Client(cluster) as client:
        worker_addresses = list(client.scheduler_info()["workers"].keys())

        retry_msg = 'Searching for a LightGBM training port for worker'

        # should handle worker maps without any duplicates
        map_without_duplicates = {
            worker_address: 12400 + i
            for i, worker_address in enumerate(worker_addresses)
        }
        patched_map = lgb.dask._possibly_fix_worker_map_duplicates(
            client=client,
            worker_map=map_without_duplicates
        )
        assert patched_map == map_without_duplicates
        assert retry_msg not in capsys.readouterr().out

        # should handle worker maps with duplicates
        map_with_duplicates = {
            worker_address: 12400
            for i, worker_address in enumerate(worker_addresses)
        }
        patched_map = lgb.dask._possibly_fix_worker_map_duplicates(
            client=client,
            worker_map=map_with_duplicates
        )
        assert retry_msg in capsys.readouterr().out
        assert len(set(patched_map.values())) == len(worker_addresses)


def test_training_does_not_fail_on_port_conflicts(cluster):
    with Client(cluster) as client:
        _, _, _, _, dX, dy, dw, _ = _create_data('binary-classification', output='array')

        lightgbm_default_port = 12400
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', lightgbm_default_port))
            dask_classifier = lgb.DaskLGBMClassifier(
                client=client,
                time_out=5,
                n_estimators=5,
                num_leaves=5
            )
            for _ in range(5):
                dask_classifier.fit(
                    X=dX,
                    y=dy,
                    sample_weight=dw,
                )
                assert dask_classifier.booster_


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('boosting_type', boosting_types)
@pytest.mark.parametrize('tree_learner', distributed_training_algorithms)
def test_regressor(output, boosting_type, tree_learner, cluster):
    with Client(cluster) as client:
        X, y, w, _, dX, dy, dw, _ = _create_data(
            objective='regression',
            output=output
        )

        params = {
            "boosting_type": boosting_type,
            "random_state": 42,
            "num_leaves": 31,
            "n_estimators": 20,
        }
        if boosting_type == 'rf':
            params.update({
                'bagging_freq': 1,
                'bagging_fraction': 0.9,
            })

        dask_regressor = lgb.DaskLGBMRegressor(
            client=client,
            time_out=5,
            tree=tree_learner,
            **params
        )
        dask_regressor = dask_regressor.fit(dX, dy, sample_weight=dw)
        p1 = dask_regressor.predict(dX)
        p1_pred_leaf = dask_regressor.predict(dX, pred_leaf=True)

        s1 = _r2_score(dy, p1)
        p1 = p1.compute()
        p1_raw = dask_regressor.predict(dX, raw_score=True).compute()
        p1_first_iter_raw = dask_regressor.predict(dX, start_iteration=0, num_iteration=1, raw_score=True).compute()
        p1_local = dask_regressor.to_local().predict(X)
        s1_local = dask_regressor.to_local().score(X, y)

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(X, y, sample_weight=w)
        s2 = local_regressor.score(X, y)
        p2 = local_regressor.predict(X)

        # Scores should be the same
        assert_eq(s1, s2, atol=0.01)
        assert_eq(s1, s1_local)

        # Predictions should be roughly the same.
        assert_eq(p1, p1_local)

        # pref_leaf values should have the right shape
        # and values that look like valid tree nodes
        pred_leaf_vals = p1_pred_leaf.compute()
        assert pred_leaf_vals.shape == (
            X.shape[0],
            dask_regressor.booster_.num_trees()
        )
        assert np.max(pred_leaf_vals) <= params['num_leaves']
        assert np.min(pred_leaf_vals) >= 0
        assert len(np.unique(pred_leaf_vals)) <= params['num_leaves']

        assert_eq(p1, y, rtol=0.5, atol=50.)
        assert_eq(p2, y, rtol=0.5, atol=50.)

        # extra predict() parameters should be passed through correctly
        with pytest.raises(AssertionError):
            assert_eq(p1_raw, p1_first_iter_raw)

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns
                if dX.dtypes[col].name == 'category'
            ]
            tree_df = dask_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[0] == '=='


@pytest.mark.parametrize('output', data_output)
def test_regressor_pred_contrib(output, cluster):
    with Client(cluster) as client:
        X, y, w, _, dX, dy, dw, _ = _create_data(
            objective='regression',
            output=output
        )

        params = {
            "n_estimators": 10,
            "num_leaves": 10
        }

        dask_regressor = lgb.DaskLGBMRegressor(
            client=client,
            time_out=5,
            tree_learner='data',
            **params
        )
        dask_regressor = dask_regressor.fit(dX, dy, sample_weight=dw)
        preds_with_contrib = dask_regressor.predict(dX, pred_contrib=True).compute()

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(X, y, sample_weight=w)
        local_preds_with_contrib = local_regressor.predict(X, pred_contrib=True)

        if output == "scipy_csr_matrix":
            preds_with_contrib = preds_with_contrib.toarray()

        # contrib outputs for distributed training are different than from local training, so we can just test
        # that the output has the right shape and base values are in the right position
        num_features = dX.shape[1]
        assert preds_with_contrib.shape[1] == num_features + 1
        assert preds_with_contrib.shape == local_preds_with_contrib.shape

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns
                if dX.dtypes[col].name == 'category'
            ]
            tree_df = dask_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[0] == '=='


@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('alpha', [.1, .5, .9])
def test_regressor_quantile(output, alpha, cluster):
    with Client(cluster) as client:
        X, y, w, _, dX, dy, dw, _ = _create_data(
            objective='regression',
            output=output
        )

        params = {
            "objective": "quantile",
            "alpha": alpha,
            "random_state": 42,
            "n_estimators": 10,
            "num_leaves": 10
        }

        dask_regressor = lgb.DaskLGBMRegressor(
            client=client,
            tree_learner_type='data_parallel',
            **params
        )
        dask_regressor = dask_regressor.fit(dX, dy, sample_weight=dw)
        p1 = dask_regressor.predict(dX).compute()
        q1 = np.count_nonzero(y < p1) / y.shape[0]

        local_regressor = lgb.LGBMRegressor(**params)
        local_regressor.fit(X, y, sample_weight=w)
        p2 = local_regressor.predict(X)
        q2 = np.count_nonzero(y < p2) / y.shape[0]

        # Quantiles should be right
        np.testing.assert_allclose(q1, alpha, atol=0.2)
        np.testing.assert_allclose(q2, alpha, atol=0.2)

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns
                if dX.dtypes[col].name == 'category'
            ]
            tree_df = dask_regressor.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[0] == '=='


@pytest.mark.parametrize('output', ['array', 'dataframe', 'dataframe-with-categorical'])
@pytest.mark.parametrize('group', [None, group_sizes])
@pytest.mark.parametrize('boosting_type', boosting_types)
@pytest.mark.parametrize('tree_learner', distributed_training_algorithms)
def test_ranker(output, group, boosting_type, tree_learner, cluster):
    with Client(cluster) as client:
        if output == 'dataframe-with-categorical':
            X, y, w, g, dX, dy, dw, dg = _create_data(
                objective='ranking',
                output=output,
                group=group,
                n_features=1,
                n_informative=1
            )
        else:
            X, y, w, g, dX, dy, dw, dg = _create_data(
                objective='ranking',
                output=output,
                group=group
            )

        # rebalance small dask.Array dataset for better performance.
        if output == 'array':
            dX = dX.persist()
            dy = dy.persist()
            dw = dw.persist()
            dg = dg.persist()
            _ = wait([dX, dy, dw, dg])
            client.rebalance()

        # use many trees + leaves to overfit, help ensure that Dask data-parallel strategy matches that of
        # serial learner. See https://github.com/microsoft/LightGBM/issues/3292#issuecomment-671288210.
        params = {
            "boosting_type": boosting_type,
            "random_state": 42,
            "n_estimators": 50,
            "num_leaves": 20,
            "min_child_samples": 1
        }
        if boosting_type == 'rf':
            params.update({
                'bagging_freq': 1,
                'bagging_fraction': 0.9,
            })

        dask_ranker = lgb.DaskLGBMRanker(
            client=client,
            time_out=5,
            tree_learner_type=tree_learner,
            **params
        )
        dask_ranker = dask_ranker.fit(dX, dy, sample_weight=dw, group=dg)
        rnkvec_dask = dask_ranker.predict(dX)
        rnkvec_dask = rnkvec_dask.compute()
        p1_pred_leaf = dask_ranker.predict(dX, pred_leaf=True)
        p1_raw = dask_ranker.predict(dX, raw_score=True).compute()
        p1_first_iter_raw = dask_ranker.predict(dX, start_iteration=0, num_iteration=1, raw_score=True).compute()
        p1_early_stop_raw = dask_ranker.predict(
            dX,
            pred_early_stop=True,
            pred_early_stop_margin=1.0,
            pred_early_stop_freq=2,
            raw_score=True
        ).compute()
        rnkvec_dask_local = dask_ranker.to_local().predict(X)

        local_ranker = lgb.LGBMRanker(**params)
        local_ranker.fit(X, y, sample_weight=w, group=g)
        rnkvec_local = local_ranker.predict(X)

        # distributed ranker should be able to rank decently well and should
        # have high rank correlation with scores from serial ranker.
        dcor = spearmanr(rnkvec_dask, y).correlation
        assert dcor > 0.6
        assert spearmanr(rnkvec_dask, rnkvec_local).correlation > 0.8
        assert_eq(rnkvec_dask, rnkvec_dask_local)

        # extra predict() parameters should be passed through correctly
        with pytest.raises(AssertionError):
            assert_eq(p1_raw, p1_first_iter_raw)

        with pytest.raises(AssertionError):
            assert_eq(p1_raw, p1_early_stop_raw)

        # pref_leaf values should have the right shape
        # and values that look like valid tree nodes
        pred_leaf_vals = p1_pred_leaf.compute()
        assert pred_leaf_vals.shape == (
            X.shape[0],
            dask_ranker.booster_.num_trees()
        )
        assert np.max(pred_leaf_vals) <= params['num_leaves']
        assert np.min(pred_leaf_vals) >= 0
        assert len(np.unique(pred_leaf_vals)) <= params['num_leaves']

        # be sure LightGBM actually used at least one categorical column,
        # and that it was correctly treated as a categorical feature
        if output == 'dataframe-with-categorical':
            cat_cols = [
                col for col in dX.columns
                if dX.dtypes[col].name == 'category'
            ]
            tree_df = dask_ranker.booster_.trees_to_dataframe()
            node_uses_cat_col = tree_df['split_feature'].isin(cat_cols)
            assert node_uses_cat_col.sum() > 0
            assert tree_df.loc[node_uses_cat_col, "decision_type"].unique()[0] == '=='


@pytest.mark.parametrize('task', tasks)
@pytest.mark.parametrize('output', data_output)
@pytest.mark.parametrize('eval_sizes', [[0.5, 1, 1.5], [0]])
@pytest.mark.parametrize('eval_names_prefix', ['specified', None])
def test_eval_set_no_early_stopping(task, output, eval_sizes, eval_names_prefix, cluster):
    if task == 'ranking' and output == 'scipy_csr_matrix':
        pytest.skip('LGBMRanker is not currently tested on sparse matrices')

    with Client(cluster) as client:
        # Use larger trainset to prevent premature stopping due to zero loss, causing num_trees() < n_estimators.
        # Use small chunk_size to avoid single-worker allocation of eval data partitions.
        n_samples = 1000
        chunk_size = 10
        n_eval_sets = len(eval_sizes)
        eval_set = []
        eval_sample_weight = []
        eval_class_weight = None
        eval_init_score = None

        if eval_names_prefix:
            eval_names = [f'{eval_names_prefix}_{i}' for i in range(len(eval_sizes))]
        else:
            eval_names = None

        X, y, w, g, dX, dy, dw, dg = _create_data(
            objective=task,
            n_samples=n_samples,
            output=output,
            chunk_size=chunk_size
        )

        if task == 'ranking':
            eval_metrics = ['ndcg']
            eval_at = (5, 6)
            eval_metric_names = [f'ndcg@{k}' for k in eval_at]
            eval_group = []
        else:
            # test eval_class_weight, eval_init_score on binary-classification task.
            # Note: objective's default `metric` will be evaluated in evals_result_ in addition to all eval_metrics.
            if task == 'binary-classification':
                eval_metrics = ['binary_error', 'auc']
                eval_metric_names = ['binary_logloss', 'binary_error', 'auc']
                eval_class_weight = []
                eval_init_score = []
            elif task == 'multiclass-classification':
                eval_metrics = ['multi_error']
                eval_metric_names = ['multi_logloss', 'multi_error']
            elif task == 'regression':
                eval_metrics = ['l1']
                eval_metric_names = ['l2', 'l1']

        # create eval_sets by creating new datasets or copying training data.
        for eval_size in eval_sizes:
            if eval_size == 1:
                y_e = y
                dX_e = dX
                dy_e = dy
                dw_e = dw
                dg_e = dg
            else:
                n_eval_samples = max(chunk_size, int(n_samples * eval_size))
                _, y_e, _, _, dX_e, dy_e, dw_e, dg_e = _create_data(
                    objective=task,
                    n_samples=n_eval_samples,
                    output=output,
                    chunk_size=chunk_size
                )

            eval_set.append((dX_e, dy_e))
            eval_sample_weight.append(dw_e)
            if task == 'ranking':
                eval_group.append(dg_e)

            if task == 'binary-classification':
                n_neg = np.sum(y_e == 0)
                n_pos = np.sum(y_e == 1)
                eval_class_weight.append({0: n_neg / n_pos, 1: n_pos / n_neg})
                init_score_value = np.log(np.mean(y_e) / (1 - np.mean(y_e)))
                if 'dataframe' in output:
                    d_init_score = dy_e.map_partitions(lambda x: pd.Series([init_score_value] * x.size))
                else:
                    d_init_score = dy_e.map_blocks(lambda x: np.repeat(init_score_value, x.size))

                eval_init_score.append(d_init_score)

        fit_trees = 50
        params = {
            "random_state": 42,
            "n_estimators": fit_trees,
            "num_leaves": 2
        }

        model_factory = task_to_dask_factory[task]
        dask_model = model_factory(
            client=client,
            **params
        )

        fit_params = {
            'X': dX,
            'y': dy,
            'eval_set': eval_set,
            'eval_names': eval_names,
            'eval_sample_weight': eval_sample_weight,
            'eval_init_score': eval_init_score,
            'eval_metric': eval_metrics,
            'verbose': True
        }
        if task == 'ranking':
            fit_params.update(
                {'group': dg,
                 'eval_group': eval_group,
                 'eval_at': eval_at}
            )
        elif task == 'binary-classification':
            fit_params.update({'eval_class_weight': eval_class_weight})

        if eval_sizes == [0]:
            with pytest.warns(UserWarning, match='Worker (.*) was not allocated eval_set data. Therefore evals_result_ and best_score_ data may be unreliable.'):
                dask_model.fit(**fit_params)
        else:
            dask_model = dask_model.fit(**fit_params)

            # total number of trees scales up for ova classifier.
            if task == 'multiclass-classification':
                model_trees = fit_trees * dask_model.n_classes_
            else:
                model_trees = fit_trees

            # check that early stopping was not applied.
            assert dask_model.booster_.num_trees() == model_trees
            assert dask_model.best_iteration_ is None

            # checks that evals_result_ and best_score_ contain expected data and eval_set names.
            evals_result = dask_model.evals_result_
            best_scores = dask_model.best_score_
            assert len(evals_result) == n_eval_sets
            assert len(best_scores) == n_eval_sets

            for eval_name in evals_result:
                assert eval_name in dask_model.best_score_
                if eval_names:
                    assert eval_name in eval_names

                # check that each eval_name and metric exists for all eval sets, allowing for the
                # case when a worker receives a fully-padded eval_set component which is not evaluated.
                if evals_result[eval_name] != 'not evaluated':
                    for metric in eval_metric_names:
                        assert metric in evals_result[eval_name]
                        assert metric in best_scores[eval_name]
                        assert len(evals_result[eval_name][metric]) == fit_trees


@pytest.mark.parametrize('task', ['binary-classification', 'regression', 'ranking'])
def test_eval_set_with_custom_eval_metric(task, cluster):
    with Client(cluster) as client:
        n_samples = 1000
        n_eval_samples = int(n_samples * 0.5)
        chunk_size = 10
        output = 'array'

        X, y, w, g, dX, dy, dw, dg = _create_data(
            objective=task,
            n_samples=n_samples,
            output=output,
            chunk_size=chunk_size
        )
        _, _, _, _, dX_e, dy_e, _, dg_e = _create_data(
            objective=task,
            n_samples=n_eval_samples,
            output=output,
            chunk_size=chunk_size
        )

        if task == 'ranking':
            eval_at = (5, 6)
            eval_metrics = ['ndcg', _constant_metric]
            eval_metric_names = [f'ndcg@{k}' for k in eval_at] + ['constant_metric']
        elif task == 'binary-classification':
            eval_metrics = ['binary_error', 'auc', _constant_metric]
            eval_metric_names = ['binary_logloss', 'binary_error', 'auc', 'constant_metric']
        else:
            eval_metrics = ['l1', _constant_metric]
            eval_metric_names = ['l2', 'l1', 'constant_metric']

        fit_trees = 50
        params = {
            "random_state": 42,
            "n_estimators": fit_trees,
            "num_leaves": 2
        }
        model_factory = task_to_dask_factory[task]
        dask_model = model_factory(
            client=client,
            **params
        )

        eval_set = [(dX_e, dy_e)]
        fit_params = {
            'X': dX,
            'y': dy,
            'eval_set': eval_set,
            'eval_metric': eval_metrics
        }
        if task == 'ranking':
            fit_params.update(
                {'group': dg,
                 'eval_group': [dg_e],
                 'eval_at': eval_at}
            )

        dask_model = dask_model.fit(**fit_params)

        eval_name = 'valid_0'
        evals_result = dask_model.evals_result_
        assert len(evals_result) == 1
        assert eval_name in evals_result

        for metric in eval_metric_names:
            assert metric in evals_result[eval_name]
            assert len(evals_result[eval_name][metric]) == fit_trees

        np.testing.assert_allclose(evals_result[eval_name]['constant_metric'], 0.708)


@pytest.mark.parametrize('task', tasks)
def test_training_works_if_client_not_provided_or_set_after_construction(task, cluster):
    with Client(cluster) as client:
        _, _, _, _, dX, dy, _, dg = _create_data(
            objective=task,
            output='array',
            group=None
        )
        model_factory = task_to_dask_factory[task]

        params = {
            "time_out": 5,
            "n_estimators": 1,
            "num_leaves": 2
        }

        # should be able to use the class without specifying a client
        dask_model = model_factory(**params)
        assert dask_model.client is None
        with pytest.raises(lgb.compat.LGBMNotFittedError, match='Cannot access property client_ before calling fit'):
            dask_model.client_

        dask_model.fit(dX, dy, group=dg)
        assert dask_model.fitted_
        assert dask_model.client is None
        assert dask_model.client_ == client

        preds = dask_model.predict(dX)
        assert isinstance(preds, da.Array)
        assert dask_model.fitted_
        assert dask_model.client is None
        assert dask_model.client_ == client

        local_model = dask_model.to_local()
        with pytest.raises(AttributeError):
            local_model.client
            local_model.client_

        # should be able to set client after construction
        dask_model = model_factory(**params)
        dask_model.set_params(client=client)
        assert dask_model.client == client

        with pytest.raises(lgb.compat.LGBMNotFittedError, match='Cannot access property client_ before calling fit'):
            dask_model.client_

        dask_model.fit(dX, dy, group=dg)
        assert dask_model.fitted_
        assert dask_model.client == client
        assert dask_model.client_ == client

        preds = dask_model.predict(dX)
        assert isinstance(preds, da.Array)
        assert dask_model.fitted_
        assert dask_model.client == client
        assert dask_model.client_ == client

        local_model = dask_model.to_local()
        with pytest.raises(AttributeError):
            local_model.client
            local_model.client_


@pytest.mark.parametrize('serializer', ['pickle', 'joblib', 'cloudpickle'])
@pytest.mark.parametrize('task', tasks)
@pytest.mark.parametrize('set_client', [True, False])
def test_model_and_local_version_are_picklable_whether_or_not_client_set_explicitly(serializer, task, set_client, tmp_path, cluster, cluster2):

    with Client(cluster) as client1:
        # data on cluster1
        X_1, _, _, _, dX_1, dy_1, _, dg_1 = _create_data(
            objective=task,
            output='array',
            group=None
        )

        with Client(cluster2) as client2:
            # create identical data on cluster2
            X_2, _, _, _, dX_2, dy_2, _, dg_2 = _create_data(
                objective=task,
                output='array',
                group=None
            )

            model_factory = task_to_dask_factory[task]

            params = {
                "time_out": 5,
                "n_estimators": 1,
                "num_leaves": 2
            }

            # at this point, the result of default_client() is client2 since it was the most recently
            # created. So setting client to client1 here to test that you can select a non-default client
            assert default_client() == client2
            if set_client:
                params.update({"client": client1})

            # unfitted model should survive pickling round trip, and pickling
            # shouldn't have side effects on the model object
            dask_model = model_factory(**params)
            local_model = dask_model.to_local()
            if set_client:
                assert dask_model.client == client1
            else:
                assert dask_model.client is None

            with pytest.raises(lgb.compat.LGBMNotFittedError, match='Cannot access property client_ before calling fit'):
                dask_model.client_

            assert "client" not in local_model.get_params()
            assert getattr(local_model, "client", None) is None

            tmp_file = tmp_path / "model-1.pkl"
            _pickle(
                obj=dask_model,
                filepath=tmp_file,
                serializer=serializer
            )
            model_from_disk = _unpickle(
                filepath=tmp_file,
                serializer=serializer
            )

            local_tmp_file = tmp_path / "local-model-1.pkl"
            _pickle(
                obj=local_model,
                filepath=local_tmp_file,
                serializer=serializer
            )
            local_model_from_disk = _unpickle(
                filepath=local_tmp_file,
                serializer=serializer
            )

            assert model_from_disk.client is None

            if set_client:
                assert dask_model.client == client1
            else:
                assert dask_model.client is None

            with pytest.raises(lgb.compat.LGBMNotFittedError, match='Cannot access property client_ before calling fit'):
                dask_model.client_

            # client will always be None after unpickling
            if set_client:
                from_disk_params = model_from_disk.get_params()
                from_disk_params.pop("client", None)
                dask_params = dask_model.get_params()
                dask_params.pop("client", None)
                assert from_disk_params == dask_params
            else:
                assert model_from_disk.get_params() == dask_model.get_params()
            assert local_model_from_disk.get_params() == local_model.get_params()

            # fitted model should survive pickling round trip, and pickling
            # shouldn't have side effects on the model object
            if set_client:
                dask_model.fit(dX_1, dy_1, group=dg_1)
            else:
                dask_model.fit(dX_2, dy_2, group=dg_2)
            local_model = dask_model.to_local()

            assert "client" not in local_model.get_params()
            with pytest.raises(AttributeError):
                local_model.client
                local_model.client_

            tmp_file2 = tmp_path / "model-2.pkl"
            _pickle(
                obj=dask_model,
                filepath=tmp_file2,
                serializer=serializer
            )
            fitted_model_from_disk = _unpickle(
                filepath=tmp_file2,
                serializer=serializer
            )

            local_tmp_file2 = tmp_path / "local-model-2.pkl"
            _pickle(
                obj=local_model,
                filepath=local_tmp_file2,
                serializer=serializer
            )
            local_fitted_model_from_disk = _unpickle(
                filepath=local_tmp_file2,
                serializer=serializer
            )

            if set_client:
                assert dask_model.client == client1
                assert dask_model.client_ == client1
            else:
                assert dask_model.client is None
                assert dask_model.client_ == default_client()
                assert dask_model.client_ == client2

            assert isinstance(fitted_model_from_disk, model_factory)
            assert fitted_model_from_disk.client is None
            assert fitted_model_from_disk.client_ == default_client()
            assert fitted_model_from_disk.client_ == client2

            # client will always be None after unpickling
            if set_client:
                from_disk_params = fitted_model_from_disk.get_params()
                from_disk_params.pop("client", None)
                dask_params = dask_model.get_params()
                dask_params.pop("client", None)
                assert from_disk_params == dask_params
            else:
                assert fitted_model_from_disk.get_params() == dask_model.get_params()
            assert local_fitted_model_from_disk.get_params() == local_model.get_params()

            if set_client:
                preds_orig = dask_model.predict(dX_1).compute()
                preds_loaded_model = fitted_model_from_disk.predict(dX_1).compute()
                preds_orig_local = local_model.predict(X_1)
                preds_loaded_model_local = local_fitted_model_from_disk.predict(X_1)
            else:
                preds_orig = dask_model.predict(dX_2).compute()
                preds_loaded_model = fitted_model_from_disk.predict(dX_2).compute()
                preds_orig_local = local_model.predict(X_2)
                preds_loaded_model_local = local_fitted_model_from_disk.predict(X_2)

            assert_eq(preds_orig, preds_loaded_model)
            assert_eq(preds_orig_local, preds_loaded_model_local)


def test_warns_and_continues_on_unrecognized_tree_learner(cluster):
    with Client(cluster) as client:
        X = da.random.random((1e3, 10))
        y = da.random.random((1e3, 1))
        dask_regressor = lgb.DaskLGBMRegressor(
            client=client,
            time_out=5,
            tree_learner='some-nonsense-value',
            n_estimators=1,
            num_leaves=2
        )
        with pytest.warns(UserWarning, match='Parameter tree_learner set to some-nonsense-value'):
            dask_regressor = dask_regressor.fit(X, y)

        assert dask_regressor.fitted_


@pytest.mark.parametrize('tree_learner', ['data_parallel', 'voting_parallel'])
def test_training_respects_tree_learner_aliases(tree_learner, cluster):
    with Client(cluster) as client:
        task = 'regression'
        _, _, _, _, dX, dy, dw, dg = _create_data(objective=task, output='array')
        dask_factory = task_to_dask_factory[task]
        dask_model = dask_factory(
            client=client,
            tree_learner=tree_learner,
            time_out=5,
            n_estimators=10,
            num_leaves=15
        )
        dask_model.fit(dX, dy, sample_weight=dw, group=dg)

        assert dask_model.fitted_
        assert dask_model.get_params()['tree_learner'] == tree_learner


def test_error_on_feature_parallel_tree_learner(cluster):
    with Client(cluster) as client:
        X = da.random.random((100, 10), chunks=(50, 10))
        y = da.random.random(100, chunks=50)
        X, y = client.persist([X, y])
        _ = wait([X, y])
        client.rebalance()
        dask_regressor = lgb.DaskLGBMRegressor(
            client=client,
            time_out=5,
            tree_learner='feature_parallel',
            n_estimators=1,
            num_leaves=2
        )
        with pytest.raises(lgb.basic.LightGBMError, match='Do not support feature parallel in c api'):
            dask_regressor = dask_regressor.fit(X, y)


def test_errors(cluster):
    with Client(cluster) as client:
        def f(part):
            raise Exception('foo')

        df = dd.demo.make_timeseries()
        df = df.map_partitions(f, meta=df._meta)
        with pytest.raises(Exception) as info:
            lgb.dask._train(
                client=client,
                data=df,
                label=df.x,
                params={},
                model_factory=lgb.LGBMClassifier
            )
            assert 'foo' in str(info.value)


@pytest.mark.parametrize('task', tasks)
@pytest.mark.parametrize('output', data_output)
def test_training_succeeds_even_if_some_workers_do_not_have_any_data(task, output, cluster):
    if task == 'ranking' and output == 'scipy_csr_matrix':
        pytest.skip('LGBMRanker is not currently tested on sparse matrices')

    with Client(cluster) as client:
        def collection_to_single_partition(collection):
            """Merge the parts of a Dask collection into a single partition."""
            if collection is None:
                return
            if isinstance(collection, da.Array):
                return collection.rechunk(*collection.shape)
            return collection.repartition(npartitions=1)

        X, y, w, g, dX, dy, dw, dg = _create_data(
            objective=task,
            output=output,
            group=None
        )

        dask_model_factory = task_to_dask_factory[task]
        local_model_factory = task_to_local_factory[task]

        dX = collection_to_single_partition(dX)
        dy = collection_to_single_partition(dy)
        dw = collection_to_single_partition(dw)
        dg = collection_to_single_partition(dg)

        n_workers = len(client.scheduler_info()['workers'])
        assert n_workers > 1
        assert dX.npartitions == 1

        params = {
            'time_out': 5,
            'random_state': 42,
            'num_leaves': 10
        }

        dask_model = dask_model_factory(tree='data', client=client, **params)
        dask_model.fit(dX, dy, group=dg, sample_weight=dw)
        dask_preds = dask_model.predict(dX).compute()

        local_model = local_model_factory(**params)
        if task == 'ranking':
            local_model.fit(X, y, group=g, sample_weight=w)
        else:
            local_model.fit(X, y, sample_weight=w)
        local_preds = local_model.predict(X)

        assert assert_eq(dask_preds, local_preds)


@pytest.mark.parametrize('task', tasks)
def test_network_params_not_required_but_respected_if_given(task, listen_port, cluster):
    with Client(cluster) as client:
        _, _, _, _, dX, dy, _, dg = _create_data(
            objective=task,
            output='array',
            chunk_size=10,
            group=None
        )

        dask_model_factory = task_to_dask_factory[task]

        # rebalance data to be sure that each worker has a piece of the data
        client.rebalance()

        # model 1 - no network parameters given
        dask_model1 = dask_model_factory(
            n_estimators=5,
            num_leaves=5,
        )
        dask_model1.fit(dX, dy, group=dg)
        assert dask_model1.fitted_
        params = dask_model1.get_params()
        assert 'local_listen_port' not in params
        assert 'machines' not in params

        # model 2 - machines given
        n_workers = len(client.scheduler_info()['workers'])
        open_ports = [lgb.dask._find_random_open_port() for _ in range(n_workers)]
        dask_model2 = dask_model_factory(
            n_estimators=5,
            num_leaves=5,
            machines=",".join([
                f"127.0.0.1:{port}"
                for port in open_ports
            ]),
        )

        dask_model2.fit(dX, dy, group=dg)
        assert dask_model2.fitted_
        params = dask_model2.get_params()
        assert 'local_listen_port' not in params
        assert 'machines' in params

        # model 3 - local_listen_port given
        # training should fail because LightGBM will try to use the same
        # port for multiple worker processes on the same machine
        dask_model3 = dask_model_factory(
            n_estimators=5,
            num_leaves=5,
            local_listen_port=listen_port
        )
        error_msg = "has multiple Dask worker processes running on it"
        with pytest.raises(lgb.basic.LightGBMError, match=error_msg):
            dask_model3.fit(dX, dy, group=dg)


@pytest.mark.parametrize('task', tasks)
def test_machines_should_be_used_if_provided(task, cluster):
    with Client(cluster) as client:
        _, _, _, _, dX, dy, _, dg = _create_data(
            objective=task,
            output='array',
            chunk_size=10,
            group=None
        )

        dask_model_factory = task_to_dask_factory[task]

        # rebalance data to be sure that each worker has a piece of the data
        client.rebalance()

        n_workers = len(client.scheduler_info()['workers'])
        assert n_workers > 1
        open_ports = [lgb.dask._find_random_open_port() for _ in range(n_workers)]
        dask_model = dask_model_factory(
            n_estimators=5,
            num_leaves=5,
            machines=",".join([
                f"127.0.0.1:{port}"
                for port in open_ports
            ]),
        )

        # test that "machines" is actually respected by creating a socket that uses
        # one of the ports mentioned in "machines"
        error_msg = f"Binding port {open_ports[0]} failed"
        with pytest.raises(lgb.basic.LightGBMError, match=error_msg):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', open_ports[0]))
                dask_model.fit(dX, dy, group=dg)

        # The above error leaves a worker waiting
        client.restart()

        # an informative error should be raised if "machines" has duplicates
        one_open_port = lgb.dask._find_random_open_port()
        dask_model.set_params(
            machines=",".join([
                f"127.0.0.1:{one_open_port}"
                for _ in range(n_workers)
            ])
        )
        with pytest.raises(ValueError, match="Found duplicates in 'machines'"):
            dask_model.fit(dX, dy, group=dg)


@pytest.mark.parametrize(
    "classes",
    [
        (lgb.DaskLGBMClassifier, lgb.LGBMClassifier),
        (lgb.DaskLGBMRegressor, lgb.LGBMRegressor),
        (lgb.DaskLGBMRanker, lgb.LGBMRanker)
    ]
)
def test_dask_classes_and_sklearn_equivalents_have_identical_constructors_except_client_arg(classes):
    dask_spec = inspect.getfullargspec(classes[0])
    sklearn_spec = inspect.getfullargspec(classes[1])
    assert dask_spec.varargs == sklearn_spec.varargs
    assert dask_spec.varkw == sklearn_spec.varkw
    assert dask_spec.kwonlyargs == sklearn_spec.kwonlyargs
    assert dask_spec.kwonlydefaults == sklearn_spec.kwonlydefaults

    # "client" should be the only different, and the final argument
    assert dask_spec.args[:-1] == sklearn_spec.args
    assert dask_spec.defaults[:-1] == sklearn_spec.defaults
    assert dask_spec.args[-1] == 'client'
    assert dask_spec.defaults[-1] is None


@pytest.mark.parametrize(
    "methods",
    [
        (lgb.DaskLGBMClassifier.fit, lgb.LGBMClassifier.fit),
        (lgb.DaskLGBMClassifier.predict, lgb.LGBMClassifier.predict),
        (lgb.DaskLGBMClassifier.predict_proba, lgb.LGBMClassifier.predict_proba),
        (lgb.DaskLGBMRegressor.fit, lgb.LGBMRegressor.fit),
        (lgb.DaskLGBMRegressor.predict, lgb.LGBMRegressor.predict),
        (lgb.DaskLGBMRanker.fit, lgb.LGBMRanker.fit),
        (lgb.DaskLGBMRanker.predict, lgb.LGBMRanker.predict)
    ]
)
def test_dask_methods_and_sklearn_equivalents_have_similar_signatures(methods):
    dask_spec = inspect.getfullargspec(methods[0])
    sklearn_spec = inspect.getfullargspec(methods[1])
    dask_params = inspect.signature(methods[0]).parameters
    sklearn_params = inspect.signature(methods[1]).parameters
    assert dask_spec.args == sklearn_spec.args[:len(dask_spec.args)]
    assert dask_spec.varargs == sklearn_spec.varargs
    if sklearn_spec.varkw:
        assert dask_spec.varkw == sklearn_spec.varkw[:len(dask_spec.varkw)]
    assert dask_spec.kwonlyargs == sklearn_spec.kwonlyargs
    assert dask_spec.kwonlydefaults == sklearn_spec.kwonlydefaults
    for param in dask_spec.args:
        error_msg = f"param '{param}' has different default values in the methods"
        assert dask_params[param].default == sklearn_params[param].default, error_msg


@pytest.mark.parametrize('task', tasks)
def test_training_succeeds_when_data_is_dataframe_and_label_is_column_array(task, cluster):
    with Client(cluster) as client:
        _, _, _, _, dX, dy, dw, dg = _create_data(
            objective=task,
            output='dataframe',
            group=None
        )

        model_factory = task_to_dask_factory[task]

        dy = dy.to_dask_array(lengths=True)
        dy_col_array = dy.reshape(-1, 1)
        assert len(dy_col_array.shape) == 2 and dy_col_array.shape[1] == 1

        params = {
            'n_estimators': 1,
            'num_leaves': 3,
            'random_state': 0,
            'time_out': 5
        }
        model = model_factory(**params)
        model.fit(dX, dy_col_array, sample_weight=dw, group=dg)
        assert model.fitted_


@pytest.mark.parametrize('task', tasks)
@pytest.mark.parametrize('output', data_output)
def test_init_score(task, output, cluster):
    if task == 'ranking' and output == 'scipy_csr_matrix':
        pytest.skip('LGBMRanker is not currently tested on sparse matrices')

    with Client(cluster) as client:
        _, _, _, _, dX, dy, dw, dg = _create_data(
            objective=task,
            output=output,
            group=None
        )

        model_factory = task_to_dask_factory[task]

        params = {
            'n_estimators': 1,
            'num_leaves': 2,
            'time_out': 5
        }
        init_score = random.random()
        size_factor = 1
        if task == 'multiclass-classification':
            size_factor = 3  # number of classes

        if output.startswith('dataframe'):
            init_scores = dy.map_partitions(lambda x: pd.DataFrame([[init_score] * size_factor] * x.size))
        else:
            init_scores = dy.map_blocks(lambda x: np.full((x.size, size_factor), init_score))
        model = model_factory(client=client, **params)
        model.fit(dX, dy, sample_weight=dw, init_score=init_scores, group=dg)
        # value of the root node is 0 when init_score is set
        assert model.booster_.trees_to_dataframe()['value'][0] == 0


def sklearn_checks_to_run():
    check_names = [
        "check_estimator_get_tags_default_keys",
        "check_get_params_invariance",
        "check_set_params"
    ]
    for check_name in check_names:
        check_func = getattr(sklearn_checks, check_name, None)
        if check_func:
            yield check_func


def _tested_estimators():
    for Estimator in [lgb.DaskLGBMClassifier, lgb.DaskLGBMRegressor]:
        yield Estimator()


@pytest.mark.parametrize("estimator", _tested_estimators())
@pytest.mark.parametrize("check", sklearn_checks_to_run())
def test_sklearn_integration(estimator, check, cluster):
    with Client(cluster) as client:
        estimator.set_params(local_listen_port=18000, time_out=5)
        name = type(estimator).__name__
        check(name, estimator)


# this test is separate because it takes a not-yet-constructed estimator
@pytest.mark.parametrize("estimator", list(_tested_estimators()))
def test_parameters_default_constructible(estimator):
    name = estimator.__class__.__name__
    if sk_version >= parse_version("0.24"):
        Estimator = estimator
    else:
        Estimator = estimator.__class__
    sklearn_checks.check_parameters_default_constructible(name, Estimator)


@pytest.mark.parametrize('task', tasks)
@pytest.mark.parametrize('output', data_output)
def test_predict_with_raw_score(task, output, cluster):
    if task == 'ranking' and output == 'scipy_csr_matrix':
        pytest.skip('LGBMRanker is not currently tested on sparse matrices')

    with Client(cluster) as client:
        _, _, _, _, dX, dy, _, dg = _create_data(
            objective=task,
            output=output,
            group=None
        )

        model_factory = task_to_dask_factory[task]
        params = {
            'client': client,
            'n_estimators': 1,
            'num_leaves': 2,
            'time_out': 5,
            'min_sum_hessian': 0
        }
        model = model_factory(**params)
        model.fit(dX, dy, group=dg)
        raw_predictions = model.predict(dX, raw_score=True).compute()

        trees_df = model.booster_.trees_to_dataframe()
        leaves_df = trees_df[trees_df.node_depth == 2]
        if task == 'multiclass-classification':
            for i in range(model.n_classes_):
                class_df = leaves_df[leaves_df.tree_index == i]
                assert set(raw_predictions[:, i]) == set(class_df['value'])
        else:
            assert set(raw_predictions) == set(leaves_df['value'])

        if task.endswith('classification'):
            pred_proba_raw = model.predict_proba(dX, raw_score=True).compute()
            assert_eq(raw_predictions, pred_proba_raw)
