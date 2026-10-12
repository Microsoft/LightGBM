# coding: utf-8
from typing import Any, Dict, Optional

import numpy as np
import pytest

import lightgbm as lgb

from .utils import assert_datasets_equal, np_assert_array_equal

pl = pytest.importorskip("polars")


# ----------------------------------------------------------------------------------------------- #
#                                            UTILITIES                                            #
# ----------------------------------------------------------------------------------------------- #

_INTEGER_TYPES = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
_FLOAT_TYPES = [pl.Float32, pl.Float64]


def generate_simple_polars_frame() -> pl.DataFrame:
    values = [1, 2, 3, 4, 5]
    bool_values = [True, True, False, False, True]
    columns = {f"col_{i}": pl.Series(values, dtype=dtype) for i, dtype in enumerate(_INTEGER_TYPES + _FLOAT_TYPES)}
    columns[f"col_{len(columns)}"] = pl.Series(bool_values, dtype=pl.Boolean)
    return pl.DataFrame(columns)


def generate_nullable_polars_frame(dtype: Any) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col_0": pl.Series([1, None, 3, 4, 5], dtype=dtype),
            "col_1": pl.Series([None, 2, 3, 4, 5], dtype=dtype),
            "col_2": pl.Series([1, 2, 3, 4, None], dtype=dtype),
            "col_3": pl.Series([None, None, None, None, None], dtype=dtype),
        }
    )


def generate_dummy_polars_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt8),
            "b": pl.Series([0.5, 0.6, 0.1, 0.8, 1.5], dtype=pl.Float32),
        }
    )


def generate_random_polars_frame(
    *,
    num_columns: int,
    num_datapoints: int,
    seed: int,
    generate_nulls: bool = True,
    values: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            f"col_{i}": generate_random_polars_series(
                num_datapoints=num_datapoints, seed=seed + i, generate_nulls=generate_nulls, values=values
            )
            for i in range(num_columns)
        }
    )


def generate_random_polars_series(
    *,
    num_datapoints: int,
    seed: int,
    generate_nulls: bool = True,
    values: Optional[np.ndarray] = None,
) -> pl.Series:
    generator = np.random.default_rng(seed)
    data = (
        generator.standard_normal(num_datapoints).astype(np.float32)
        if values is None
        else generator.choice(values, size=num_datapoints, replace=True)
    )
    series = pl.Series("col", data, dtype=pl.Float32)
    if generate_nulls:
        indices = generator.choice(len(data), size=num_datapoints // 10)
        series = series.scatter(indices, None)
    return series


def dummy_dataset_params() -> Dict[str, Any]:
    return {
        "min_data_in_bin": 1,
        "min_data_in_leaf": 1,
        "force_row_wise": True,
    }


# ----------------------------------------------------------------------------------------------- #
#                                            UNIT TESTS                                           #
# ----------------------------------------------------------------------------------------------- #

# ------------------------------------------- DATASET ------------------------------------------- #


@pytest.mark.parametrize(
    ("polars_frame_fn", "dataset_params"),
    [  # Use lambda functions here to minimize memory consumption
        (generate_simple_polars_frame, dummy_dataset_params()),
        (generate_dummy_polars_frame, dummy_dataset_params()),
        (lambda: generate_nullable_polars_frame(pl.Float32), dummy_dataset_params()),
        (lambda: generate_nullable_polars_frame(pl.Int32), dummy_dataset_params()),
        (lambda: generate_random_polars_frame(num_columns=3, num_datapoints=1000, seed=42), {}),
        (lambda: generate_random_polars_frame(num_columns=100, num_datapoints=10000, seed=43), {}),
    ],
)
def test_dataset_construct_fuzzy(tmp_path, polars_frame_fn, dataset_params):
    polars_frame = polars_frame_fn()

    polars_dataset = lgb.Dataset(polars_frame, params=dataset_params)
    polars_dataset.construct()

    pandas_dataset = lgb.Dataset(polars_frame.to_pandas(), params=dataset_params)
    pandas_dataset.construct()

    assert_datasets_equal(tmp_path, polars_dataset, pandas_dataset)


def test_dataset_construct_fuzzy_boolean(tmp_path):
    boolean_data = generate_random_polars_frame(
        num_columns=10, num_datapoints=10000, seed=42, generate_nulls=False, values=np.array([True, False])
    )
    float_data = boolean_data.cast(pl.Float32)

    polars_dataset = lgb.Dataset(boolean_data)
    polars_dataset.construct()

    pandas_dataset = lgb.Dataset(float_data.to_pandas())
    pandas_dataset.construct()

    assert_datasets_equal(tmp_path, polars_dataset, pandas_dataset)


# -------------------------------------------- FIELDS ------------------------------------------- #


def test_dataset_construct_fields_fuzzy():
    polars_frame = generate_random_polars_frame(num_columns=3, num_datapoints=1000, seed=42)
    polars_labels = generate_random_polars_series(num_datapoints=1000, seed=42, generate_nulls=False)
    polars_weights = generate_random_polars_series(num_datapoints=1000, seed=42, generate_nulls=False)
    polars_init_scores = generate_random_polars_series(num_datapoints=1000, seed=44, generate_nulls=False)
    polars_groups = pl.Series("group", [300, 400, 50, 250], dtype=pl.Int32)
    polars_positions = pl.Series("position", np.random.default_rng(45).integers(0, 10, size=1000), dtype=pl.Int32)

    polars_dataset = lgb.Dataset(
        polars_frame,
        label=polars_labels,
        weight=polars_weights,
        group=polars_groups,
        init_score=polars_init_scores,
        position=polars_positions,
    )
    polars_dataset.construct()

    pandas_dataset = lgb.Dataset(
        polars_frame.to_pandas(),
        label=polars_labels.to_numpy(),
        weight=polars_weights.to_numpy(),
        group=polars_groups.to_numpy(),
        init_score=polars_init_scores.to_numpy(),
        position=polars_positions.to_numpy(),
    )
    pandas_dataset.construct()

    for field in ("label", "weight", "group", "init_score", "position"):
        np_assert_array_equal(polars_dataset.get_field(field), pandas_dataset.get_field(field), strict=True)


# -------------------------------------------- LABELS ------------------------------------------- #


@pytest.mark.parametrize("polars_type", _INTEGER_TYPES + _FLOAT_TYPES)
def test_dataset_construct_labels(polars_type):
    data = generate_dummy_polars_frame()
    labels = pl.Series("label", [0, 1, 0, 0, 1], dtype=polars_type)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_label(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("label"), strict=True)


def test_dataset_construct_labels_boolean():
    data = generate_dummy_polars_frame()
    labels = pl.Series("label", [False, True, False, False, True], dtype=pl.Boolean)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_label(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("label"), strict=True)


# ------------------------------------------- WEIGHTS ------------------------------------------- #


def test_dataset_construct_weights_none():
    data = generate_dummy_polars_frame()
    weight = pl.Series("weight", [1, 1, 1, 1, 1], dtype=pl.Float32)
    dataset = lgb.Dataset(data, weight=weight, params=dummy_dataset_params())
    dataset.construct()
    assert dataset.get_weight() is None
    assert dataset.get_field("weight") is None


@pytest.mark.parametrize("polars_type", _FLOAT_TYPES)
def test_dataset_construct_weights(polars_type):
    data = generate_dummy_polars_frame()
    weights = pl.Series("weight", [3, 0.7, 1.5, 0.5, 0.1], dtype=polars_type)
    dataset = lgb.Dataset(data, weight=weights, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([3, 0.7, 1.5, 0.5, 0.1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_weight(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("weight"), strict=True)


# -------------------------------------------- GROUPS ------------------------------------------- #


@pytest.mark.parametrize("polars_type", _INTEGER_TYPES)
def test_dataset_construct_groups(polars_type):
    data = generate_dummy_polars_frame()
    groups = pl.Series("group", [2, 3], dtype=polars_type)
    dataset = lgb.Dataset(data, group=groups, params=dummy_dataset_params())
    dataset.construct()

    expected_boundaries = np.array([0, 2, 5], dtype=np.int32)
    expected_group_sizes = np.array([2, 3], dtype=np.int32)
    np_assert_array_equal(expected_group_sizes, dataset.get_group(), strict=True)
    np_assert_array_equal(expected_boundaries, dataset.get_field("group"), strict=True)


# ------------------------------------------ POSITION ------------------------------------------- #


@pytest.mark.parametrize("polars_type", _INTEGER_TYPES)
def test_dataset_construct_position(polars_type):
    data = generate_dummy_polars_frame()
    positions = pl.Series("position", [0, 1, 2, 3, 4], dtype=polars_type)
    dataset = lgb.Dataset(data, label=[0, 1, 0, 1, 0], position=positions, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    np_assert_array_equal(expected, dataset.get_position(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("position"), strict=True)


@pytest.mark.parametrize("polars_type", _INTEGER_TYPES)
def test_dataset_construct_position_with_duplicates_and_out_of_order(polars_type):
    data = generate_dummy_polars_frame()
    positions = pl.Series("position", [15, 15, 8, 27, 15], dtype=polars_type)
    dataset = lgb.Dataset(data, label=[0, 1, 0, 1, 0], position=positions, params=dummy_dataset_params())
    dataset.construct()

    # positions are remapped on the C++ side to dense indices in first-seen order:
    # 15 -> 0, 8 -> 1, 27 -> 2
    expected = np.array([0, 0, 1, 2, 0], dtype=np.int32)
    np_assert_array_equal(expected, dataset.get_position(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("position"), strict=True)


# ----------------------------------------- INIT SCORES ----------------------------------------- #


@pytest.mark.parametrize("polars_type", _INTEGER_TYPES + _FLOAT_TYPES)
def test_dataset_construct_init_scores_array(polars_type):
    data = generate_dummy_polars_frame()
    init_scores = pl.Series("init_score", [0, 1, 2, 3, 3], dtype=polars_type)
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 2, 3, 3], dtype=np.float64)
    np_assert_array_equal(expected, dataset.get_init_score(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("init_score"), strict=True)


def test_dataset_construct_init_scores_table():
    data = generate_dummy_polars_frame()
    init_scores = pl.DataFrame(
        {
            "a": generate_random_polars_series(num_datapoints=5, seed=1, generate_nulls=False),
            "b": generate_random_polars_series(num_datapoints=5, seed=2, generate_nulls=False),
            "c": generate_random_polars_series(num_datapoints=5, seed=3, generate_nulls=False),
        }
    )
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    actual = dataset.get_init_score()
    expected = init_scores.to_numpy().astype(np.float64)
    np_assert_array_equal(expected, actual, strict=True)


# ------------------------------------------ PREDICTION ----------------------------------------- #


def assert_equal_predict_polars_pandas(booster: lgb.Booster, data: pl.DataFrame):
    pandas_data = data.to_pandas()

    p_polars = booster.predict(data)
    p_pandas = booster.predict(pandas_data)
    np_assert_array_equal(p_polars, p_pandas, strict=True)

    p_raw_polars = booster.predict(data, raw_score=True)
    p_raw_pandas = booster.predict(pandas_data, raw_score=True)
    np_assert_array_equal(p_raw_polars, p_raw_pandas, strict=True)

    p_leaf_polars = booster.predict(data, pred_leaf=True)
    p_leaf_pandas = booster.predict(pandas_data, pred_leaf=True)
    np_assert_array_equal(p_leaf_polars, p_leaf_pandas, strict=True)

    p_pred_contrib_polars = booster.predict(data, pred_contrib=True)
    p_pred_contrib_pandas = booster.predict(pandas_data, pred_contrib=True)
    np_assert_array_equal(p_pred_contrib_polars, p_pred_contrib_pandas, strict=True)

    p_first_iter_polars = booster.predict(data, start_iteration=0, num_iteration=1, raw_score=True)
    p_first_iter_pandas = booster.predict(pandas_data, start_iteration=0, num_iteration=1, raw_score=True)
    np_assert_array_equal(p_first_iter_polars, p_first_iter_pandas, strict=True)


def test_predict_regression():
    data_float = generate_random_polars_frame(num_columns=10, num_datapoints=10000, seed=42)
    data_bool = generate_random_polars_frame(
        num_columns=1, num_datapoints=10000, seed=42, generate_nulls=False, values=np.array([True, False])
    )
    data = data_float.with_columns(data_bool["col_0"].alias("col_bool"))

    dataset = lgb.Dataset(
        data,
        label=generate_random_polars_series(num_datapoints=10000, seed=43, generate_nulls=False),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_polars_pandas(booster, data)


def test_predict_binary_classification():
    data = generate_random_polars_frame(num_columns=10, num_datapoints=10000, seed=42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_polars_series(num_datapoints=10000, seed=43, generate_nulls=False, values=np.arange(2)),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_polars_pandas(booster, data)


def test_predict_multiclass_classification():
    data = generate_random_polars_frame(num_columns=10, num_datapoints=10000, seed=42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_polars_series(num_datapoints=10000, seed=43, generate_nulls=False, values=np.arange(5)),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "multiclass", "num_leaves": 7, "num_class": 5},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_polars_pandas(booster, data)


def test_predict_ranking():
    data = generate_random_polars_frame(num_columns=10, num_datapoints=10000, seed=42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_polars_series(num_datapoints=10000, seed=43, generate_nulls=False, values=np.arange(4)),
        group=np.array([1000, 2000, 3000, 4000]),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "lambdarank", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_polars_pandas(booster, data)


def test_polars_feature_name_auto():
    data = generate_dummy_polars_frame()
    dataset = lgb.Dataset(
        data,
        label=pl.Series("label", [0, 1, 0, 0, 1]),
        params=dummy_dataset_params(),
        categorical_feature=["a"],
    )
    booster = lgb.train({"num_leaves": 7}, dataset, num_boost_round=5)
    assert booster.feature_name() == ["a", "b"]


def test_polars_feature_name_manual():
    data = generate_dummy_polars_frame()
    dataset = lgb.Dataset(
        data,
        label=pl.Series("label", [0, 1, 0, 0, 1]),
        params=dummy_dataset_params(),
        feature_name=["c", "d"],
        categorical_feature=["c"],
    )
    booster = lgb.train({"num_leaves": 7}, dataset, num_boost_round=5)
    assert booster.feature_name() == ["c", "d"]


def test_get_data_polars_frame():
    from polars.testing import assert_frame_equal  # noqa: PLC0415

    original_frame = generate_simple_polars_frame()
    dataset = lgb.Dataset(original_frame, free_raw_data=False)
    dataset.construct()

    returned_data = dataset.get_data()
    assert isinstance(returned_data, pl.DataFrame)
    assert returned_data.schema == original_frame.schema
    assert returned_data.shape == original_frame.shape
    assert_frame_equal(returned_data, original_frame)


def test_get_data_polars_frame_subset(rng):
    from polars.testing import assert_frame_equal  # noqa: PLC0415

    original_frame = generate_random_polars_frame(num_columns=3, num_datapoints=1000, seed=42)
    dataset = lgb.Dataset(original_frame, free_raw_data=False)
    dataset.construct()

    subset_size = 100
    used_indices = rng.choice(a=original_frame.shape[0], size=subset_size, replace=False)
    used_indices = sorted(used_indices)

    subset_dataset = dataset.subset(used_indices).construct()
    expected_subset = original_frame[used_indices]
    subset_data = subset_dataset.get_data()

    assert isinstance(subset_data, pl.DataFrame)
    assert subset_data.schema == expected_subset.schema
    assert subset_data.shape == expected_subset.shape
    assert len(subset_data) == len(used_indices)
    assert subset_data.shape == (subset_size, 3)
    assert_frame_equal(subset_data, expected_subset)
