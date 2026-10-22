# coding: utf-8
import pytest

import lightgbm as lgb

from .utils import SERIALIZERS, pickle_and_unpickle_object


def reset_feature_fraction(boosting_round):
    return 0.6 if boosting_round < 15 else 0.8


@pytest.mark.parametrize("serializer", SERIALIZERS)
def test_early_stopping_callback_is_picklable(serializer):
    rounds = 5
    callback = lgb.early_stopping(stopping_rounds=rounds)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 30
    assert callback_from_disk.before_iteration is False
    assert callback.stopping_rounds == callback_from_disk.stopping_rounds
    assert callback.stopping_rounds == rounds


def test_early_stopping_callback_rejects_invalid_stopping_rounds_with_informative_errors():
    with pytest.raises(TypeError, match="early_stopping_round should be an integer. Got 'str'"):
        lgb.early_stopping(stopping_rounds="neverrrr")


@pytest.mark.parametrize("stopping_rounds", [-10, -1, 0])
def test_early_stopping_callback_accepts_non_positive_stopping_rounds(stopping_rounds):
    cb = lgb.early_stopping(stopping_rounds=stopping_rounds)
    assert cb.enabled is False


@pytest.mark.parametrize("serializer", SERIALIZERS)
def test_log_evaluation_callback_is_picklable(serializer):
    periods = 42
    callback = lgb.log_evaluation(period=periods)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 10
    assert callback_from_disk.before_iteration is False
    assert callback.period == callback_from_disk.period
    assert callback.period == periods


@pytest.mark.parametrize("serializer", SERIALIZERS)
def test_record_evaluation_callback_is_picklable(serializer):
    results = {}
    callback = lgb.record_evaluation(eval_result=results)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 20
    assert callback_from_disk.before_iteration is False
    assert callback.eval_result == callback_from_disk.eval_result
    assert callback.eval_result is results


@pytest.mark.parametrize("serializer", SERIALIZERS)
def test_reset_parameter_callback_is_picklable(serializer):
    params = {"bagging_fraction": [0.7] * 5 + [0.6] * 5, "feature_fraction": reset_feature_fraction}
    callback = lgb.reset_parameter(**params)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 10
    assert callback_from_disk.before_iteration is True
    assert callback.kwargs == callback_from_disk.kwargs
    assert callback.kwargs == params


def test_reset_parameter_callback_with_sklearn():
    """Test that reset_parameter callback works with LGBMClassifier."""
    import numpy as np
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    model = LGBMClassifier(
        n_estimators=10,
        colsample_bytree=0.9,  # Start high
        callbacks=[lgb.reset_parameter(colsample_bytree=[0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8])],
        verbose=-1
    )
    model.fit(X, y)

    trees_df = model.booster_.trees_to_dataframe()
    unique_feature_counts = trees_df.groupby('tree_index')['split_feature'].nunique()

    assert unique_feature_counts.nunique() > 1, (
        f"reset_parameter callback did not work with LGBMClassifier. "
        f"All trees used the same number of features. Counts: {unique_feature_counts.unique()}"
    )