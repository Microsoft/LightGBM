# coding: utf-8
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import lightgbm as lgb

from .utils import assert_datasets_equal, np_assert_array_equal

pd = pytest.importorskip("pandas")


# ----------------------------------------------------------------------------------------------- #
#                                            UTILITIES                                            #
# ----------------------------------------------------------------------------------------------- #


def generate_simple_pandas_frame() -> pd.DataFrame:
    values = [1, 2, 3, 4, 5]
    bool_values = [True, True, False, False, True]
    int_dtypes = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
    float_dtypes = [np.float32, np.float64]
    columns = {f"col_{i}": pd.array(values, dtype=dtype) for i, dtype in enumerate(int_dtypes + float_dtypes)}
    columns[f"col_{len(columns)}"] = pd.array(bool_values, dtype=bool)
    return pd.DataFrame(columns)


def generate_dummy_pandas_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": pd.array([1, 2, 3, 4, 5], dtype=np.uint8),
            "b": pd.array([0.5, 0.6, 0.1, 0.8, 1.5], dtype=np.float32),
        }
    )


def generate_random_pandas_frame(
    *,
    num_columns: int,
    num_datapoints: int,
    seed: int,
    generate_nulls: bool = True,
    values: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    columns = {
        f"col_{i}": generate_random_pandas_series(
            num_datapoints=num_datapoints, seed=seed + i, generate_nulls=generate_nulls, values=values
        )
        for i in range(num_columns)
    }
    return pd.DataFrame(columns)


def generate_random_pandas_series(
    *,
    num_datapoints: int,
    seed: int,
    generate_nulls: bool = True,
    values: Optional[np.ndarray] = None,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    data = (
        generator.standard_normal(num_datapoints).astype(np.float32)
        if values is None
        else generator.choice(values, size=num_datapoints, replace=True).astype(np.float32)
    )
    if generate_nulls:
        indices = generator.choice(len(data), size=num_datapoints // 10)
        data[indices] = np.nan
    return data


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
    "pandas_frame_fn",
    [  # Use lambda functions here to minimize memory consumption
        generate_simple_pandas_frame,
        generate_dummy_pandas_frame,
        lambda: generate_random_pandas_frame(num_columns=3, num_datapoints=1000, seed=42),
        lambda: generate_random_pandas_frame(num_columns=100, num_datapoints=10000, seed=43),
    ],
)
def test_dataset_construct_fuzzy(tmp_path, pandas_frame_fn):
    df = pandas_frame_fn()

    ds1 = lgb.Dataset(df, params=dummy_dataset_params())
    ds1.construct()

    ds2 = lgb.Dataset(df.copy(), params=dummy_dataset_params())
    ds2.construct()

    assert_datasets_equal(tmp_path, ds1, ds2)


def test_dataset_construct_fuzzy_boolean(tmp_path):
    boolean_data = generate_random_pandas_frame(
        num_columns=10, num_datapoints=10000, seed=42, generate_nulls=False, values=np.array([True, False])
    )
    float_data = boolean_data.astype(np.float32)

    ds_bool = lgb.Dataset(boolean_data)
    ds_bool.construct()

    ds_float = lgb.Dataset(float_data)
    ds_float.construct()

    assert_datasets_equal(tmp_path, ds_bool, ds_float)


# -------------------------------------------- FIELDS ------------------------------------------- #


def test_dataset_construct_fields_fuzzy():
    df = generate_random_pandas_frame(num_columns=3, num_datapoints=1000, seed=42)
    labels = generate_random_pandas_series(num_datapoints=1000, seed=42, generate_nulls=False)
    weights = generate_random_pandas_series(num_datapoints=1000, seed=42, generate_nulls=False)
    init_scores = generate_random_pandas_series(num_datapoints=1000, seed=44, generate_nulls=False)
    groups = np.array([300, 400, 50, 250], dtype=np.int32)
    positions = np.random.default_rng(45).integers(0, 10, size=1000, dtype=np.int32)

    pandas_dataset = lgb.Dataset(
        df,
        label=labels,
        weight=weights,
        group=groups,
        init_score=init_scores,
        position=positions,
    )
    pandas_dataset.construct()

    numpy_dataset = lgb.Dataset(
        df.to_numpy(),
        label=labels,
        weight=weights,
        group=groups,
        init_score=init_scores,
        position=positions,
    )
    numpy_dataset.construct()

    for field in ("label", "weight", "group", "init_score", "position"):
        np_assert_array_equal(pandas_dataset.get_field(field), numpy_dataset.get_field(field), strict=True)


# -------------------------------------------- LABELS ------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, bool])
def test_dataset_construct_labels(dtype):
    data = generate_dummy_pandas_frame()
    labels = pd.Series([0, 1, 0, 0, 1], dtype=dtype)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_label(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("label"), strict=True)


# ------------------------------------------- WEIGHTS ------------------------------------------- #


def test_dataset_construct_weights_none():
    data = generate_dummy_pandas_frame()
    weight = pd.Series([1, 1, 1, 1, 1], dtype=np.float32)
    dataset = lgb.Dataset(data, weight=weight, params=dummy_dataset_params())
    dataset.construct()
    assert dataset.get_weight() is None
    assert dataset.get_field("weight") is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dataset_construct_weights(dtype):
    data = generate_dummy_pandas_frame()
    weights = pd.Series([3, 0.7, 1.5, 0.5, 0.1], dtype=dtype)
    dataset = lgb.Dataset(data, weight=weights, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([3, 0.7, 1.5, 0.5, 0.1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_weight(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("weight"), strict=True)


# -------------------------------------------- GROUPS ------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_dataset_construct_groups(dtype):
    data = generate_dummy_pandas_frame()
    groups = pd.Series([2, 3], dtype=dtype)
    dataset = lgb.Dataset(data, group=groups, params=dummy_dataset_params())
    dataset.construct()

    expected_boundaries = np.array([0, 2, 5], dtype=np.int32)
    expected_group_sizes = np.array([2, 3], dtype=np.int32)
    np_assert_array_equal(expected_group_sizes, dataset.get_group(), strict=True)
    np_assert_array_equal(expected_boundaries, dataset.get_field("group"), strict=True)


# ------------------------------------------ POSITION ------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_dataset_construct_position(dtype):
    data = generate_dummy_pandas_frame()
    positions = pd.Series([0, 1, 2, 3, 4], dtype=dtype)
    dataset = lgb.Dataset(data, label=[0, 1, 0, 1, 0], position=positions, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    np_assert_array_equal(expected, dataset.get_position(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("position"), strict=True)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_dataset_construct_position_with_duplicates_and_out_of_order(dtype):
    data = generate_dummy_pandas_frame()
    positions = pd.Series([15, 15, 8, 27, 15], dtype=dtype)
    dataset = lgb.Dataset(data, label=[0, 1, 0, 1, 0], position=positions, params=dummy_dataset_params())
    dataset.construct()

    # positions are remapped on the C++ side to dense indices in first-seen order:
    # 15 -> 0, 8 -> 1, 27 -> 2
    expected = np.array([0, 0, 1, 2, 0], dtype=np.int32)
    np_assert_array_equal(expected, dataset.get_position(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("position"), strict=True)


# ----------------------------------------- INIT SCORES ----------------------------------------- #


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_dataset_construct_init_scores_array(dtype):
    data = generate_dummy_pandas_frame()
    init_scores = pd.Series([0, 1, 2, 3, 3], dtype=dtype)
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 2, 3, 3], dtype=np.float64)
    np_assert_array_equal(expected, dataset.get_init_score(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("init_score"), strict=True)


def test_dataset_construct_init_scores_dataframe():
    data = generate_dummy_pandas_frame()
    init_scores = pd.DataFrame(
        {
            "a": generate_random_pandas_series(num_datapoints=5, seed=1, generate_nulls=False),
            "b": generate_random_pandas_series(num_datapoints=5, seed=2, generate_nulls=False),
            "c": generate_random_pandas_series(num_datapoints=5, seed=3, generate_nulls=False),
        }
    )
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    expected = init_scores.to_numpy().astype(np.float64)
    np_assert_array_equal(expected, dataset.get_init_score(), strict=True)
    np_assert_array_equal(expected, dataset.get_field("init_score"), strict=True)


# ------------------------------------------ PREDICTION ----------------------------------------- #


def test_predict_regression():
    data = generate_random_pandas_frame(num_columns=10, num_datapoints=10000, seed=42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_pandas_series(num_datapoints=10000, seed=43, generate_nulls=False),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    p_pandas = booster.predict(data)
    p_numpy = booster.predict(data.to_numpy())
    np_assert_array_equal(p_pandas, p_numpy, strict=True)


def test_predict_binary_classification():
    data = generate_random_pandas_frame(num_columns=10, num_datapoints=10000, seed=42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_pandas_series(num_datapoints=10000, seed=43, generate_nulls=False, values=np.arange(2)),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    p_pandas = booster.predict(data)
    p_numpy = booster.predict(data.to_numpy())
    np_assert_array_equal(p_pandas, p_numpy, strict=True)


def test_predict_multiclass_classification():
    data = generate_random_pandas_frame(num_columns=10, num_datapoints=10000, seed=42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_pandas_series(num_datapoints=10000, seed=43, generate_nulls=False, values=np.arange(5)),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "multiclass", "num_leaves": 7, "num_class": 5},
        dataset,
        num_boost_round=5,
    )
    p_pandas = booster.predict(data)
    p_numpy = booster.predict(data.to_numpy())
    np_assert_array_equal(p_pandas, p_numpy, strict=True)


def test_predict_ranking():
    data = generate_random_pandas_frame(num_columns=10, num_datapoints=10000, seed=42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_pandas_series(num_datapoints=10000, seed=43, generate_nulls=False, values=np.arange(4)),
        group=np.array([1000, 2000, 3000, 4000]),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "lambdarank", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    p_pandas = booster.predict(data)
    p_numpy = booster.predict(data.to_numpy())
    np_assert_array_equal(p_pandas, p_numpy, strict=True)


def test_feature_name_auto():
    data = generate_dummy_pandas_frame()
    dataset = lgb.Dataset(
        data,
        label=pd.Series([0, 1, 0, 0, 1]),
        params=dummy_dataset_params(),
        categorical_feature=["a"],
    )
    booster = lgb.train({"num_leaves": 7}, dataset, num_boost_round=5)
    assert booster.feature_name() == ["a", "b"]


def test_feature_name_manual():
    data = generate_dummy_pandas_frame()
    dataset = lgb.Dataset(
        data,
        label=pd.Series([0, 1, 0, 0, 1]),
        params=dummy_dataset_params(),
        feature_name=["c", "d"],
        categorical_feature=["c"],
    )
    booster = lgb.train({"num_leaves": 7}, dataset, num_boost_round=5)
    assert booster.feature_name() == ["c", "d"]


def test_get_data_frame():
    original_frame = generate_simple_pandas_frame()
    dataset = lgb.Dataset(original_frame, free_raw_data=False)
    dataset.construct()

    returned_data = dataset.get_data()
    assert isinstance(returned_data, pd.DataFrame)
    assert list(returned_data.columns) == list(original_frame.columns)
    assert returned_data.shape == original_frame.shape


def test_get_data_frame_subset(rng):
    original_frame = generate_random_pandas_frame(num_columns=3, num_datapoints=1000, seed=42)
    original_frame = pd.DataFrame(original_frame)
    dataset = lgb.Dataset(original_frame, free_raw_data=False)
    dataset.construct()

    subset_size = 100
    used_indices = sorted(rng.choice(a=original_frame.shape[0], size=subset_size, replace=False).tolist())

    subset_dataset = dataset.subset(used_indices).construct()
    subset_data = subset_dataset.get_data()

    assert isinstance(subset_data, pd.DataFrame)
    assert subset_data.shape == (subset_size, 3)


# ------------------------------------------- CATEGORICAL ----------------------------------------- #


def test_pandas_categorical_encoding(tmp_path):
    cat1_categories = ["a", "b", "c"]
    cat1_values = ["a", "b", "c", "b", "a"]
    cat2_categories = ["b", "c", "d"]
    cat2_values = ["b", "c", "c", "d", "d"]
    ordered_categories = ["high", "low", "mid"]
    ordered_values = ["low", "high", "mid", "high", "low"]

    df = pd.DataFrame(
        {
            "cat1": pd.Categorical(cat1_values, categories=cat1_categories, ordered=False),
            "cat2": pd.Categorical(cat2_values, categories=cat2_categories, ordered=False),
            "cat3": pd.Categorical(ordered_values, categories=ordered_categories, ordered=True),
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = [0, 1, 0, 1, 0]

    ds = lgb.Dataset(df, label=y, params=dummy_dataset_params())
    ds.construct()

    assert ds.num_data() == 5
    assert ds.num_feature() == 4
    assert ds.get_feature_name() == ["cat1", "cat2", "cat3", "num_col"]

    assert ds.categorical_feature == "auto"
    assert len(ds.pandas_categorical) == 3
    assert ds.pandas_categorical[0] == cat1_categories
    assert ds.pandas_categorical[1] == cat2_categories
    assert ds.pandas_categorical[2] == ordered_categories
    assert ds.params["categorical_column"] == [0, 1]  # ordered categorical not treated as categorical by default

    # Verify correct encodings
    ref_df = pd.DataFrame(
        {
            "cat1": [cat1_categories.index(v) for v in cat1_values],  # [0, 1, 2, 1, 0]
            "cat2": [cat2_categories.index(v) for v in cat2_values],  # [0, 1, 1, 2, 2],
            "cat3": [ordered_categories.index(v) for v in ordered_values],  # [1, 0, 2, 0, 1],
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ref_ds = lgb.Dataset(ref_df, label=y, categorical_feature=[0, 1], params=dummy_dataset_params())
    ref_ds.construct()

    assert_datasets_equal(tmp_path, ds, ref_ds)


def test_pandas_categorical_encoding_unseen_category(tmp_path):
    train_categories = ["a", "b", "c"]
    train_values = ["a", "b", "c", "a", "b"]
    valid_values = ["a", "c", "d", "d", "a"]  # "d" is unseen in training data

    train_df = pd.DataFrame({"cat_col": pd.Categorical(train_values), "num_col": [1.0, 2.0, 3.0, 4.0, 5.0]})
    valid_df = pd.DataFrame({"cat_col": pd.Categorical(valid_values), "num_col": [6.0, 7.0, 8.0, 9.0, 10.0]})

    train_ds = lgb.Dataset(train_df, label=[0, 1, 0, 1, 0], params=dummy_dataset_params())
    valid_ds = lgb.Dataset(valid_df, label=[1, 0, 1, 0, 1], reference=train_ds, params=dummy_dataset_params())
    train_ds.construct()
    valid_ds.construct()

    # Verify unseen category is encoded as NaN
    ref_valid_df = pd.DataFrame(
        {
            "cat_col": pd.Categorical(["a", "c", None, None, "a"], categories=train_categories),
            "num_col": [6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    ref_valid_ds = lgb.Dataset(ref_valid_df, label=[1, 0, 1, 0, 1], reference=train_ds, params=dummy_dataset_params())
    ref_valid_ds.construct()

    assert_datasets_equal(tmp_path, valid_ds, ref_valid_ds)


def test_categorical_encoding_registered_but_unobserved(tmp_path):
    # Define full DataFrame with all categories observed
    full_df = pd.DataFrame(
        {
            "unordered_col": pd.Categorical(["a", "b", "c", "d"]),
            "ordered_col": pd.Categorical(["e", "f", "g", "h"], ordered=True),
        }
    )

    # Slice train from full_df so all categories are preserved despite not all being observed
    train_df = full_df.iloc[[0, 2, 2]]  # ["a", "c", "c"] and ["e", "g", "g"]
    valid_df = pd.DataFrame(
        {
            "unordered_col": pd.Categorical(["a", "b", "d"]),
            "ordered_col": pd.Categorical(["h", "e", "f"], ordered=True),
        }
    )

    train_ds = lgb.Dataset(train_df, label=[0, 1, 0], params=dummy_dataset_params())
    valid_ds = lgb.Dataset(valid_df, label=[0, 1, 0], reference=train_ds, params=dummy_dataset_params())
    train_ds.construct()
    valid_ds.construct()

    assert train_ds.pandas_categorical[0] == ["a", "b", "c", "d"]
    assert train_ds.pandas_categorical[1] == ["e", "f", "g", "h"]
    assert train_ds.params["categorical_column"] == [0]  # only unordered column is treated as categorical

    # Python-side encoding: both ordered and unordered columns use all registered categories to encode
    valid_df_encoded = lgb.basic._data_from_narwhals(
        data=valid_df,
        feature_name="auto",
        categorical_feature="auto",
        pandas_categorical=train_ds.pandas_categorical,
    )[0]
    assert valid_df_encoded.iloc[:, 0].tolist() == [0.0, 1.0, 3.0]  # a -> 0, b -> 1, d -> 3
    assert valid_df_encoded.iloc[:, 1].tolist() == [3.0, 0.0, 1.0]  # h -> 3, e -> 0, f -> 1

    # C++ binning
    # - Unordered columns: only codes observed during training are binned. Unseen codes are treated as missing.
    # - Ordered columns: treats as continuous. Unseen values interpolate (e<f<g) or clip (h clipped to g).
    ref_valid_df = pd.DataFrame(
        {
            "unordered_col": pd.Categorical(["a", None, None], categories=["a", "b", "c", "d"]),
            "ordered_col": pd.Categorical(["g", "e", "g"], categories=["e", "f", "g", "h"], ordered=True),
        }
    )
    ref_valid_ds = lgb.Dataset(ref_valid_df, label=[0, 1, 0], reference=train_ds, params=dummy_dataset_params())
    ref_valid_ds.construct()

    assert_datasets_equal(tmp_path, valid_ds, ref_valid_ds)


def test_categorical_with_missing_values(tmp_path):
    categories = ["a", "b"]
    values_none = ["a", "b", None, "a", None]
    values_nan = ["b", "a", np.nan, "b", np.nan]

    df = pd.DataFrame(
        {
            "cat_none": pd.Categorical(values_none, categories=categories),
            "cat_nan": pd.Categorical(values_nan, categories=categories),
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = [0, 1, 0, 1, 0]

    ds = lgb.Dataset(df, label=y, params=dummy_dataset_params())
    ds.construct()
    assert ds.pandas_categorical == [categories, categories]

    ref_df = pd.DataFrame(
        {
            "cat_none": [0.0, 1.0, np.nan, 0.0, np.nan],
            "cat_nan": [1.0, 0.0, np.nan, 1.0, np.nan],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ref_ds = lgb.Dataset(ref_df, label=y, categorical_feature=[0, 1], params=dummy_dataset_params())
    ref_ds.construct()
    assert_datasets_equal(tmp_path, ds, ref_ds)


def test_pandas_dataset_construction_with_high_cardinality_categorical_succeeds(rng):
    X = pd.DataFrame({"x1": rng.integers(low=0, high=5_000, size=(10_000,))})
    y = rng.uniform(size=(10_000,))
    ds = lgb.Dataset(X, y, categorical_feature=["x1"])
    ds.construct()
    assert ds.num_data() == 10_000
    assert ds.num_feature() == 1


@pytest.mark.parametrize(
    "feature_name",
    [
        pytest.param(["x1"], id="feature-name"),
        pytest.param([42], id="feature-index"),
        pytest.param("auto", id="auto"),
    ],
)
@pytest.mark.parametrize("categories", ["seen", "unseen"])
def test_pandas_categorical_code_conversion_doesnt_modify_original_data(feature_name, categories, rng):
    X = rng.choice(a=["a", "b"], size=(100, 1))
    df = pd.DataFrame(X.copy(), columns=["x1"], dtype="category")
    if categories == "seen":
        pandas_categorical = [["a", "b"]]
    else:
        pandas_categorical = [["a"]]
    data = lgb.basic._data_from_narwhals(
        data=df,
        feature_name=feature_name,
        categorical_feature="auto",
        pandas_categorical=pandas_categorical,
    )[0]
    data = lgb.basic._pandas_df_to_numpy(data)
    # check that the original data wasn't modified
    np.testing.assert_equal(df["x1"], X[:, 0])
    # check that the built data has the codes
    if categories == "seen":
        # if all categories were seen during training we just take the codes
        codes = df["x1"].cat.codes
    else:
        # if we only saw 'a' during training we just replace its code
        # and leave the rest as nan
        a_code = df["x1"].cat.categories.get_loc("a")
        codes = np.where(df["x1"] == "a", a_code, np.nan)
    np.testing.assert_equal(codes, data[:, 0])


# ---------------------------------------- DTYPE VALIDATION --------------------------------------- #


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pd.Int8Dtype(), [1, 2, 3]),
        (pd.Int16Dtype(), [1, 2, 3]),
        (pd.Int32Dtype(), [1, 2, 3]),
        (pd.Int64Dtype(), [1, 2, 3]),
        (pd.UInt8Dtype(), [1, 2, 3]),
        (pd.UInt16Dtype(), [1, 2, 3]),
        (pd.UInt32Dtype(), [1, 2, 3]),
        (pd.UInt64Dtype(), [1, 2, 3]),
        (pd.Float32Dtype(), [1.0, 2.0, 3.0]),
        (pd.Float64Dtype(), [1.0, 2.0, 3.0]),
        (pd.BooleanDtype(), [True, False, True]),
        (pd.SparseDtype(), [1.0, 2.0, 3.0]),
        # Categorical dtypes are supported, but tested separately
    ],
)
def test_pandas_supported_dtypes(tmp_path, dtype, values):
    df = pd.DataFrame({"test_col": pd.Series(values, dtype=dtype), "num_col": [4.0, 5.0, 6.0]})
    y = [0, 1, 0]

    ds = lgb.Dataset(df, label=y, params=dummy_dataset_params())
    ds.construct()

    assert ds.num_data() == 3
    assert ds.num_feature() == 2
    assert ds.get_feature_name() == ["test_col", "num_col"]
    assert ds.get_label().tolist() == y

    # Verify values are preserved
    ref_df = pd.DataFrame({"test_col": values, "num_col": [4.0, 5.0, 6.0]})
    ref_ds = lgb.Dataset(ref_df, label=y, params=dummy_dataset_params())
    ref_ds.construct()

    assert_datasets_equal(tmp_path, ds, ref_ds)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pd.StringDtype(), ["a", "b", "c"]),
        (pd.DatetimeTZDtype(tz=ZoneInfo("UTC")), ["2020-01-01", "2020-01-02", "2020-01-03"]),
        (pd.PeriodDtype(freq="Y"), [pd.Period("2024"), pd.Period("2025"), pd.Period("2026")]),
        (pd.IntervalDtype(subtype="int64"), [pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(2, 3)]),
    ],
)
def test_pandas_unsupported_dtypes(dtype, values):
    df = pd.DataFrame({"test_col": pd.Series(values, dtype=dtype), "num_col": [1.0, 2.0, 3.0]})
    y = [0, 1, 0]

    with pytest.raises(ValueError, match="DataFrame dtypes must be int, float, bool, categorical or enum"):
        lgb.Dataset(df, label=y).construct()
