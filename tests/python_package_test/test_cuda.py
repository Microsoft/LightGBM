# coding: utf-8
"""Tests for CUDA training correctness versus CPU."""

import numpy as np
import pytest
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import lightgbm as lgb

from .utils import BuildInfo

pytestmark = pytest.mark.skipif(not BuildInfo.has_cuda, reason="Requires a CUDA build (TASK=cuda)")


def _make_binary_data(n_samples, n_features, zeros=0.0, nans=0.0, random_state=42):
    """Binary labels from a linear predictor; zeros/NaNs can also predict y.

    After scale(), 1.2 weights the value signal and 0.8 the missingness signal.
    When zeros>0, draws are |N(5, 2)|+1 so the injected 0s sit outside the dense mass.
    """
    rng = np.random.RandomState(random_state)
    if zeros:
        X = (np.abs(rng.normal(5.0, 2.0, size=(n_samples, n_features))) + 1.0).astype(np.float32)
    else:
        X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    u = rng.uniform(size=X.shape)
    zero_mask = (u < zeros) if zeros else None
    if zero_mask is not None:
        X[zero_mask] = 0.0
    nan_mask = ((u >= zeros) & (u < zeros + nans)) if nans else None

    logits = 1.2 * scale(np.nan_to_num(X, nan=0.0) @ rng.normal(size=n_features))
    src = zero_mask if (zeros and nans) else nan_mask
    if src is not None:
        logits = logits + 0.8 * scale(src[:, : max(1, n_features // 2)].sum(axis=1))
    y = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    if nan_mask is not None:
        X[nan_mask] = np.nan
    return X, y


def _make_exclusive_nan_data(n_samples, n_groups, group_size, n_values=8, random_state=42):
    """One finite value per group so LightGBM packs the group into one column.

    n_values=8 keeps per-feature bins small enough to share a column.
    1.5 / -4.5 make y rare and driven by the last (packed) feature in each group.
    """
    rng = np.random.RandomState(random_state)
    n_features = n_groups * group_size
    X = np.full((n_samples, n_features), np.nan, dtype=np.float32)
    which = rng.randint(0, group_size, size=(n_samples, n_groups))
    vals = rng.randint(1, n_values + 1, size=(n_samples, n_groups)).astype(np.float32)
    for g in range(n_groups):
        cols = which[:, g] + g * group_size
        X[np.arange(n_samples), cols] = vals[:, g]
    later = np.zeros(n_samples, dtype=np.float64)
    for g in range(n_groups):
        later += np.nan_to_num(X[:, g * group_size + group_size - 1], nan=0.0)
    logits = 1.5 * scale(later) - 4.5
    y = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return X, y


def _train(
    X,
    y,
    device,
    max_bin,
    quantized,
    num_leaves,
    num_boost_round,
    min_data_in_leaf=20,
    # When True, also record LightGBM's own valid binary_logloss (CUDA score
    # updates on binned data) for comparison with sklearn log_loss on predict().
    internal_metric=False,
):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {
        "objective": "binary",
        "metric": "binary_logloss" if internal_metric else "None",
        "device_type": device,
        "learning_rate": 0.1,
        "num_leaves": num_leaves,
        "max_bin": max_bin,
        "use_quantized_grad": quantized,
        "num_grad_quant_bins": 4,
        "min_data_in_leaf": min_data_in_leaf,
        "num_threads": 1,
        "seed": 42,
        "deterministic": True,
        "verbose": -1,
    }
    lgb_train = lgb.Dataset(X_train, label=y_train, params={"max_bin": max_bin})
    evals_result = {}
    if internal_metric:
        lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train, params={"max_bin": max_bin})
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=lgb_eval,
            callbacks=[lgb.record_evaluation(evals_result)],
        )
    else:
        gbm = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
    pred = gbm.predict(X_test)
    return {
        "auc": roc_auc_score(y_test, pred),
        "logloss": log_loss(y_test, pred),
        "internal_logloss": evals_result["valid_0"]["binary_logloss"][-1] if internal_metric else None,
    }


def _assert_cuda_matches_cpu(
    X,
    y,
    max_bin,
    quantized,
    num_leaves,
    num_boost_round,
    min_data_in_leaf=20,
):
    params = {
        "max_bin": max_bin,
        "quantized": quantized,
        "num_leaves": num_leaves,
        "num_boost_round": num_boost_round,
        "min_data_in_leaf": min_data_in_leaf,
    }
    cpu = _train(X, y, device="cpu", **params)
    cuda = _train(X, y, device="cuda", **params)
    assert abs(cuda["auc"] - cpu["auc"]) < 0.01
    assert abs(cuda["logloss"] / cpu["logloss"] - 1.0) < 0.02


def test_cuda_quantized_multi_leaf():
    """num_leaves=31 so several splits compound stale packed leaf totals / hist-pool stride."""
    X, y = _make_binary_data(n_samples=15_000, n_features=20)
    _assert_cuda_matches_cpu(X, y, max_bin=255, quantized=True, num_leaves=31, num_boost_round=25)


def test_cuda_quantized_32bit_histogram():
    """50k rows so quantized histogram counts exceed 16-bit and must be read as int64."""
    X, y = _make_binary_data(n_samples=50_000, n_features=20)
    _assert_cuda_matches_cpu(X, y, max_bin=255, quantized=True, num_leaves=2, num_boost_round=25)


def test_cuda_missing_nan_global_split():
    """max_bin=300 takes the global-memory split finder; nans=0.25 exercises the NaN skip path."""
    X, y = _make_binary_data(n_samples=60_000, n_features=32, nans=0.25)
    _assert_cuda_matches_cpu(X, y, max_bin=300, quantized=False, num_leaves=15, num_boost_round=30)


def test_cuda_high_max_bin_histogram():
    """max_bin=768 is past FixHistogram's 512-thread block, so extra bins must be included."""
    X, y = _make_binary_data(n_samples=80_000, n_features=16)
    _assert_cuda_matches_cpu(
        X,
        y,
        max_bin=768,
        quantized=False,
        num_leaves=2,
        num_boost_round=30,
        min_data_in_leaf=100,
    )


def test_cuda_nan_eval_matches_predict():
    """Packed-column NaNs: internal logloss must match predict() (rtol=0.5)."""
    X, y = _make_exclusive_nan_data(n_samples=50_000, n_groups=16, group_size=8)
    result = _train(
        X, y, device="cuda", max_bin=255, quantized=False, num_leaves=15, num_boost_round=20, internal_metric=True
    )
    np.testing.assert_allclose(result["internal_logloss"], result["logloss"], rtol=0.5)


def test_cuda_quantized_nan_and_zero():
    """zeros=0.60 / nans=0.20 so most_freq_bin is 0 (mfb_offset==1) and NaNs are present."""
    X, y = _make_binary_data(n_samples=50_000, n_features=40, zeros=0.60, nans=0.20)
    _assert_cuda_matches_cpu(
        X,
        y,
        max_bin=255,
        quantized=True,
        num_leaves=31,
        num_boost_round=25,
        min_data_in_leaf=50,
    )
