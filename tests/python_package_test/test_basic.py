# coding: utf-8
import filecmp
import numbers
import re
import signal
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, make_blobs
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from .utils import BuildInfo, dummy_obj, load_breast_cancer, mse_obj, np_assert_array_equal


def test_basic(tmp_path):
    X_train, X_test, y_train, y_test = train_test_split(
        *load_breast_cancer(return_X_y=True), test_size=0.1, random_state=2
    )
    feature_names = [f"Column_{i}" for i in range(X_train.shape[1])]
    feature_names[1] = "a" * 1000  # set one name to a value longer than default buffer size
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = train_data.create_valid(X_test, label=y_test)

    params = {
        "objective": "binary",
        "metric": "auc",
        "min_data": 10,
        "num_leaves": 15,
        "verbose": -1,
        "num_threads": 1,
        "max_bin": 255,
        "gpu_use_dp": True,
    }
    bst = lgb.Booster(params, train_data)
    bst.add_valid(valid_data, "valid_1")

    for i in range(20):
        bst.update()
        if i % 10 == 0:
            print(bst.eval_train(), bst.eval_valid())

    assert train_data.get_feature_name() == feature_names

    assert bst.current_iteration() == 20
    assert bst.num_trees() == 20
    assert bst.num_model_per_iteration() == 1
    if not BuildInfo.has_cuda:
        assert bst.lower_bound() == pytest.approx(-2.9040190126976606)
        assert bst.upper_bound() == pytest.approx(3.3182142872462883)

    tname = tmp_path / "svm_light.dat"
    model_file = tmp_path / "model.txt"

    bst.save_model(model_file)
    pred_from_matr = bst.predict(X_test)
    with open(tname, "w+b") as f:
        dump_svmlight_file(X_test, y_test, f)
    pred_from_file = bst.predict(tname)
    np.testing.assert_allclose(pred_from_matr, pred_from_file)

    # check saved model persistence
    bst = lgb.Booster(params, model_file=model_file)
    assert bst.feature_name() == feature_names
    pred_from_model_file = bst.predict(X_test)
    # we need to check the consistency of model file here, so test for exact equal
    np_assert_array_equal(pred_from_matr, pred_from_model_file, strict=True)

    # check early stopping is working. Make it stop very early, so the scores should be very close to zero
    pred_parameter = {"pred_early_stop": True, "pred_early_stop_freq": 5, "pred_early_stop_margin": 1.5}
    pred_early_stopping = bst.predict(X_test, **pred_parameter)
    # scores likely to be different, but prediction should still be the same
    np_assert_array_equal(np.sign(pred_from_matr), np.sign(pred_early_stopping), strict=True)

    # test that shape is checked during prediction
    bad_X_test = X_test[:, 1:]
    bad_shape_error_msg = "The number of features in data*"
    np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg, bst.predict, bad_X_test)
    np.testing.assert_raises_regex(
        lgb.basic.LightGBMError, bad_shape_error_msg, bst.predict, sparse.csr_matrix(bad_X_test)
    )
    np.testing.assert_raises_regex(
        lgb.basic.LightGBMError, bad_shape_error_msg, bst.predict, sparse.csc_matrix(bad_X_test)
    )
    with open(tname, "w+b") as f:
        dump_svmlight_file(bad_X_test, y_test, f)
    np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg, bst.predict, tname)
    with open(tname, "w+b") as f:
        dump_svmlight_file(X_test, y_test, f, zero_based=False)
    np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg, bst.predict, tname)


def test_reset_parameter_on_loaded_model(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    bst = lgb.train(
        {"objective": "binary", "learning_rate": 0.1, "num_threads": 1}, lgb.Dataset(X, label=y), num_boost_round=2
    )
    model_file = tmp_path / "model.txt"
    bst.save_model(model_file)
    loaded = lgb.Booster(model_file=model_file)
    expected = loaded.predict(X)
    assert loaded.params["learning_rate"] == 0.1
    assert "some_unrecognized_param" not in loaded.params
    assert loaded._get_loaded_param() == loaded.params
    expected_params = loaded.params.copy()

    for params in [{"some_unrecognized_param": 123456789}, {"learning_rate": 0.2}]:
        loaded.reset_parameter(params)
        expected_params.update(params)
        assert loaded.params == expected_params

        # Reload to read the current C++ configuration instead of the parameters saved at load time.
        reloaded = lgb.Booster(model_str=loaded.model_to_string())
        assert reloaded.params["learning_rate"] == expected_params["learning_rate"]
        assert "some_unrecognized_param" not in reloaded.params
        np_assert_array_equal(loaded.predict(X), expected, strict=True)


def test_booster_rollback_one_iter(rng):
    """Test that Booster.rollback_one_iter() correctly rolls back one boosting iteration."""
    X = rng.uniform(size=(100, 5))
    y = rng.integers(0, 2, size=(100,))
    X_test = rng.uniform(size=(10, 5))

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "verbose": -1,
    }
    bst = lgb.Booster(params, train_data)

    # Train for 10 iterations
    num_iterations = 10
    for _ in range(num_iterations):
        bst.update()

    assert bst.current_iteration() == num_iterations
    assert bst.num_trees() == num_iterations

    # Get predictions before rollback
    pred_before = bst.predict(X_test)

    # Rollback one iteration
    result = bst.rollback_one_iter()

    # Verify rollback decremented both iteration count and tree count
    assert bst.current_iteration() == num_iterations - 1
    assert bst.num_trees() == num_iterations - 1
    # Verify it returns self for method chaining
    assert result is bst

    # Verify predictions actually changed (proves tree was removed, not just counter)
    pred_after = bst.predict(X_test)
    assert not np.allclose(pred_before, pred_after)

    # Verify multiple rollbacks work
    bst.rollback_one_iter()
    assert bst.current_iteration() == num_iterations - 2
    assert bst.num_trees() == num_iterations - 2


class NumpySequence(lgb.Sequence):
    def __init__(self, ndarray, batch_size):
        self.ndarray = ndarray
        self.batch_size = batch_size

    def __getitem__(self, idx):
        # The simple implementation is just a single "return self.ndarray[idx]"
        # The following is for demo and testing purpose.
        if isinstance(idx, numbers.Integral):
            return self.ndarray[idx]
        elif isinstance(idx, slice):
            if not (idx.step is None or idx.step == 1):
                raise NotImplementedError("No need to implement, caller will not set step by now")
            return self.ndarray[idx.start : idx.stop]
        elif isinstance(idx, list):
            return self.ndarray[idx]
        else:
            raise TypeError(f"Sequence Index must be an integer/list/slice, got {type(idx).__name__}")

    def __len__(self):
        return len(self.ndarray)


def _create_sequence_from_ndarray(data, num_seq, batch_size):
    if num_seq == 1:
        return NumpySequence(data, batch_size)

    nrow = data.shape[0]
    seqs = []
    seq_size = nrow // num_seq
    for start in range(0, nrow, seq_size):
        end = min(start + seq_size, nrow)
        seq = NumpySequence(data[start:end], batch_size)
        seqs.append(seq)
    return seqs


@pytest.mark.parametrize("sample_count", [11, 100, None])
@pytest.mark.parametrize("batch_size", [3, None])
@pytest.mark.parametrize("include_0_and_nan", [False, True])
@pytest.mark.parametrize("num_seq", [1, 3])
def test_sequence(tmpdir, sample_count, batch_size, include_0_and_nan, num_seq, rng):
    params = {"bin_construct_sample_cnt": sample_count}

    nrow = 50
    half_nrow = nrow // 2
    ncol = 11
    data = np.arange(nrow * ncol, dtype=np.float64).reshape((nrow, ncol))

    if include_0_and_nan:
        # whole col
        data[:, 0] = 0
        data[:, 1] = np.nan

        # half col
        data[:half_nrow, 3] = 0
        data[:half_nrow, 2] = np.nan

        data[half_nrow:-2, 4] = 0
        data[:half_nrow, 4] = np.nan

    X = data[:, :-1]
    Y = data[:, -1]

    npy_bin_fname = tmpdir / "data_from_npy.bin"
    seq_bin_fname = tmpdir / "data_from_seq.bin"

    # Create dataset from numpy array directly.
    ds = lgb.Dataset(X, label=Y, params=params)
    ds.save_binary(npy_bin_fname)

    # Create dataset using Sequence.
    seqs = _create_sequence_from_ndarray(X, num_seq, batch_size)
    seq_ds = lgb.Dataset(seqs, label=Y, params=params)
    seq_ds.save_binary(seq_bin_fname)

    assert filecmp.cmp(npy_bin_fname, seq_bin_fname)

    # Test for validation set.
    # Select some random rows as valid data.
    valid_idx = (rng.random(10) * nrow).astype(np.int32)
    valid_data = data[valid_idx, :]
    valid_X = valid_data[:, :-1]
    valid_Y = valid_data[:, -1]

    valid_npy_bin_fname = tmpdir / "valid_data_from_npy.bin"
    valid_seq_bin_fname = tmpdir / "valid_data_from_seq.bin"
    valid_seq2_bin_fname = tmpdir / "valid_data_from_seq2.bin"

    valid_ds = lgb.Dataset(valid_X, label=valid_Y, params=params, reference=ds)
    valid_ds.save_binary(valid_npy_bin_fname)

    # From Dataset constructor, with dataset from numpy array.
    valid_seqs = _create_sequence_from_ndarray(valid_X, num_seq, batch_size)
    valid_seq_ds = lgb.Dataset(valid_seqs, label=valid_Y, params=params, reference=ds)
    valid_seq_ds.save_binary(valid_seq_bin_fname)
    assert filecmp.cmp(valid_npy_bin_fname, valid_seq_bin_fname)

    # From Dataset.create_valid, with dataset from sequence.
    valid_seq_ds2 = seq_ds.create_valid(valid_seqs, label=valid_Y, params=params)
    valid_seq_ds2.save_binary(valid_seq2_bin_fname)
    assert filecmp.cmp(valid_npy_bin_fname, valid_seq2_bin_fname)


@pytest.mark.parametrize("num_seq", [1, 2])
def test_sequence_get_data(num_seq, rng):
    nrow = 20
    ncol = 11
    data = np.arange(nrow * ncol, dtype=np.float64).reshape((nrow, ncol))
    X = data[:, :-1]
    Y = data[:, -1]

    seqs = _create_sequence_from_ndarray(data=X, num_seq=num_seq, batch_size=6)
    seq_ds = lgb.Dataset(seqs, label=Y, params=None, free_raw_data=False).construct()
    assert seq_ds.get_data() == seqs

    used_indices = rng.choice(a=np.arange(nrow), size=nrow // 3, replace=False)
    subset_data = seq_ds.subset(used_indices).construct()
    np_assert_array_equal(subset_data.get_data(), X[sorted(used_indices)], strict=True)


def test_chunked_dataset():
    X_train, X_test, y_train, y_test = train_test_split(
        *load_breast_cancer(return_X_y=True), test_size=0.1, random_state=2
    )

    chunk_size = X_train.shape[0] // 10 + 1
    X_train = [X_train[i * chunk_size : (i + 1) * chunk_size, :] for i in range(X_train.shape[0] // chunk_size + 1)]
    X_test = [X_test[i * chunk_size : (i + 1) * chunk_size, :] for i in range(X_test.shape[0] // chunk_size + 1)]

    train_data = lgb.Dataset(X_train, label=y_train, params={"bin_construct_sample_cnt": 100})
    valid_data = train_data.create_valid(X_test, label=y_test, params={"bin_construct_sample_cnt": 100})
    train_data.construct()
    valid_data.construct()


def test_chunked_dataset_linear():
    X_train, X_test, y_train, y_test = train_test_split(
        *load_breast_cancer(return_X_y=True), test_size=0.1, random_state=2
    )
    chunk_size = X_train.shape[0] // 10 + 1
    X_train = [X_train[i * chunk_size : (i + 1) * chunk_size, :] for i in range(X_train.shape[0] // chunk_size + 1)]
    X_test = [X_test[i * chunk_size : (i + 1) * chunk_size, :] for i in range(X_test.shape[0] // chunk_size + 1)]
    params = {"bin_construct_sample_cnt": 100, "linear_tree": True}
    train_data = lgb.Dataset(X_train, label=y_train, params=params)
    valid_data = train_data.create_valid(X_test, label=y_test, params=params)
    train_data.construct()
    valid_data.construct()


def test_save_dataset_subset_and_load_from_file(tmp_path, rng):
    data = rng.standard_normal(size=(100, 2))
    params = {"max_bin": 50, "min_data_in_bin": 10}
    ds = lgb.Dataset(data, params=params)
    ds.subset([1, 2, 3, 5, 8]).save_binary(tmp_path / "subset.bin")
    lgb.Dataset(tmp_path / "subset.bin", params=params).construct()


def test_save_binary_raises_on_truncated_write(tmp_path, rng):
    resource = pytest.importorskip("resource")
    if not hasattr(signal, "SIGXFSZ"):
        pytest.skip("SIGXFSZ is not available on this platform")

    data = rng.standard_normal(size=(1000, 20))
    ds = lgb.Dataset(data).construct()
    original_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    original_signal_handler = signal.getsignal(signal.SIGXFSZ)
    try:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (4096, original_limit[1]))
        with pytest.raises(lgb.basic.LightGBMError, match="Cannot write binary data"):
            ds.save_binary(tmp_path / "truncated.bin")
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, original_limit)
        signal.signal(signal.SIGXFSZ, original_signal_handler)


def test_subset_group():
    rank_example_dir = Path(__file__).absolute().parents[2] / "examples" / "lambdarank"
    X_train, y_train = load_svmlight_file(str(rank_example_dir / "rank.train"))
    q_train = np.loadtxt(str(rank_example_dir / "rank.train.query"))
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    assert len(lgb_train.get_group()) == 201
    subset = lgb_train.subset(list(range(10))).construct()
    subset_group = subset.get_group()
    assert len(subset_group) == 2
    assert subset_group[0] == 1
    assert subset_group[1] == 9


def test_add_features_throws_if_num_data_unequal(rng):
    X1 = rng.uniform(size=(100, 1))
    X2 = rng.uniform(size=(10, 1))
    d1 = lgb.Dataset(X1).construct()
    d2 = lgb.Dataset(X2).construct()
    with pytest.raises(
        lgb.basic.LightGBMError, match="Cannot add features from other Dataset with a different number of rows"
    ):
        d1.add_features_from(d2)


def test_add_features_throws_if_datasets_unconstructed(rng):
    X1 = rng.uniform(size=(100, 1))
    X2 = rng.uniform(size=(100, 1))
    err_msg = "Both source and target Datasets must be constructed before adding features"
    d1 = lgb.Dataset(X1)
    d2 = lgb.Dataset(X2)
    with pytest.raises(ValueError, match=err_msg):
        d1.add_features_from(d2)
    d1 = lgb.Dataset(X1).construct()
    d2 = lgb.Dataset(X2)
    with pytest.raises(ValueError, match=err_msg):
        d1.add_features_from(d2)
    d1 = lgb.Dataset(X1)
    d2 = lgb.Dataset(X2).construct()
    with pytest.raises(ValueError, match=err_msg):
        d1.add_features_from(d2)


def test_add_features_equal_data_on_alternating_used_unused(tmp_path, rng):
    X = rng.uniform(size=(100, 5))
    X[:, [1, 3]] = 0
    names = [f"col_{i}" for i in range(5)]
    for j in range(1, 5):
        d1 = lgb.Dataset(X[:, :j], feature_name=names[:j]).construct()
        d2 = lgb.Dataset(X[:, j:], feature_name=names[j:]).construct()
        d1.add_features_from(d2)
        d1name = tmp_path / "d1.txt"
        d1._dump_text(d1name)
        d = lgb.Dataset(X, feature_name=names).construct()
        dname = tmp_path / "d.txt"
        d._dump_text(dname)
        with open(d1name, "rt") as d1f:
            d1txt = d1f.read()
        with open(dname, "rt") as df:
            dtxt = df.read()
        assert dtxt == d1txt


def test_add_features_same_booster_behaviour(tmp_path, rng):
    X = rng.uniform(size=(100, 5))
    X[:, [1, 3]] = 0
    names = [f"col_{i}" for i in range(5)]
    for j in range(1, 5):
        d1 = lgb.Dataset(X[:, :j], feature_name=names[:j]).construct()
        d2 = lgb.Dataset(X[:, j:], feature_name=names[j:]).construct()
        d1.add_features_from(d2)
        d = lgb.Dataset(X, feature_name=names).construct()
        y = rng.uniform(size=(100,))
        d1.set_label(y)
        d.set_label(y)
        b1 = lgb.Booster(train_set=d1)
        b = lgb.Booster(train_set=d)
        for _ in range(10):
            b.update()
            b1.update()
        dname = tmp_path / "d.txt"
        d1name = tmp_path / "d1.txt"
        b1.save_model(d1name)
        b.save_model(dname)
        with open(dname, "rt") as df:
            dtxt = df.read()
        with open(d1name, "rt") as d1f:
            d1txt = d1f.read()
        assert dtxt == d1txt


def test_add_features_from_different_sources(rng):
    pd = pytest.importorskip("pandas")
    n_row = 100
    n_col = 5
    X = rng.uniform(size=(n_row, n_col))
    xxs = [X, sparse.csr_matrix(X), pd.DataFrame(X)]
    names = [f"col_{i}" for i in range(n_col)]
    seq = _create_sequence_from_ndarray(X, 1, 30)
    seq_ds = lgb.Dataset(seq, feature_name=names, free_raw_data=False).construct()
    npy_list_ds = lgb.Dataset(
        [X[: n_row // 2, :], X[n_row // 2 :, :]], feature_name=names, free_raw_data=False
    ).construct()
    immergeable_dds = [seq_ds, npy_list_ds]
    for x_1 in xxs:
        # test that method works even with free_raw_data=True
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=True).construct()
        d2 = lgb.Dataset(x_1, feature_name=names, free_raw_data=True).construct()
        d1.add_features_from(d2)
        assert d1.data is None

        # test that method works but sets raw data to None in case of immergeable data types
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=False).construct()
        for d2 in immergeable_dds:
            d1.add_features_from(d2)
            assert d1.data is None

        # test that method works for different data types
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=False).construct()
        res_feature_names = deepcopy(names)
        for idx, x_2 in enumerate(xxs, 2):
            original_type = type(d1.get_data())
            d2 = lgb.Dataset(x_2, feature_name=names, free_raw_data=False).construct()
            d1.add_features_from(d2)
            assert isinstance(d1.get_data(), original_type)
            assert d1.get_data().shape == (n_row, n_col * idx)
            res_feature_names += [f"D{idx}_{name}" for name in names]
            assert d1.feature_name == res_feature_names


def test_add_features_does_not_fail_if_initial_dataset_has_zero_informative_features(capsys, rng):
    arr_a = np.zeros((100, 1), dtype=np.float32)
    arr_b = rng.uniform(size=(100, 5))

    dataset_a = lgb.Dataset(arr_a, params={"verbose": 0}).construct()
    expected_msg = (
        "[LightGBM] [Warning] There are no meaningful features which satisfy "
        "the provided configuration. Decreasing Dataset parameters min_data_in_bin "
        "or min_data_in_leaf and re-constructing Dataset might resolve this warning.\n"
    )
    log_lines = capsys.readouterr().out
    assert expected_msg in log_lines

    dataset_b = lgb.Dataset(arr_b).construct()

    original_handle = dataset_a._handle.value
    dataset_a.add_features_from(dataset_b)
    assert dataset_a.num_feature() == 6
    assert dataset_a.num_data() == 100
    assert dataset_a._handle.value == original_handle


def test_cegb_affects_behavior(tmp_path, rng):
    X = rng.uniform(size=(100, 5))
    X[:, [1, 3]] = 0
    y = rng.uniform(size=(100,))
    names = [f"col_{i}" for i in range(5)]
    ds = lgb.Dataset(X, feature_name=names).construct()
    ds.set_label(y)
    base = lgb.Booster(train_set=ds)
    for _ in range(10):
        base.update()
    basename = tmp_path / "basename.txt"
    base.save_model(basename)
    with open(basename, "rt") as f:
        basetxt = f.read()
    # Set extremely harsh penalties, so CEGB will block most splits.
    cases = [
        {"cegb_penalty_feature_coupled": [50, 100, 10, 25, 30]},
        {"cegb_penalty_feature_lazy": [1, 2, 3, 4, 5]},
        {"cegb_penalty_split": 1},
    ]
    for case in cases:
        booster = lgb.Booster(train_set=ds, params=case)
        for _ in range(10):
            booster.update()
        casename = tmp_path / "casename.txt"
        booster.save_model(casename)
        with open(casename, "rt") as f:
            casetxt = f.read()
        assert basetxt != casetxt


def test_cegb_scaling_equalities(tmp_path, rng):
    X = rng.uniform(size=(100, 5))
    X[:, [1, 3]] = 0
    y = rng.uniform(size=(100,))
    names = [f"col_{i}" for i in range(5)]
    ds = lgb.Dataset(X, feature_name=names).construct()
    ds.set_label(y)
    # Compare pairs of penalties, to ensure scaling works as intended
    pairs = [
        (
            {"cegb_penalty_feature_coupled": [1, 2, 1, 2, 1]},
            {"cegb_penalty_feature_coupled": [0.5, 1, 0.5, 1, 0.5], "cegb_tradeoff": 2},
        ),
        (
            {"cegb_penalty_feature_lazy": [0.01, 0.02, 0.03, 0.04, 0.05]},
            {"cegb_penalty_feature_lazy": [0.005, 0.01, 0.015, 0.02, 0.025], "cegb_tradeoff": 2},
        ),
        ({"cegb_penalty_split": 1}, {"cegb_penalty_split": 2, "cegb_tradeoff": 0.5}),
    ]
    for p1, p2 in pairs:
        booster1 = lgb.Booster(train_set=ds, params=p1)
        booster2 = lgb.Booster(train_set=ds, params=p2)
        for _ in range(10):
            booster1.update()
            booster2.update()
        p1name = tmp_path / "p1.txt"
        # Reset booster1's parameters to p2, so the parameter section of the file matches.
        booster1.reset_parameter(p2)
        booster1.save_model(p1name)
        with open(p1name, "rt") as f:
            p1txt = f.read()
        p2name = tmp_path / "p2.txt"
        booster2.save_model(p2name)
        with open(p2name, "rt") as f:
            p2txt = f.read()
        assert p1txt == p2txt


def test_consistent_state_for_dataset_fields():
    def check_asserts(data):
        np.testing.assert_allclose(data.label, data.get_label())
        np.testing.assert_allclose(data.label, data.get_field("label"))
        assert not np.isnan(data.label[0])
        assert not np.isinf(data.label[1])
        np.testing.assert_allclose(data.weight, data.get_weight())
        np.testing.assert_allclose(data.weight, data.get_field("weight"))
        assert not np.isnan(data.weight[0])
        assert not np.isinf(data.weight[1])
        np.testing.assert_allclose(data.init_score, data.get_init_score())
        np.testing.assert_allclose(data.init_score, data.get_field("init_score"))
        assert not np.isnan(data.init_score[0])
        assert not np.isinf(data.init_score[1])
        assert np.all(np.isclose([data.label[0], data.weight[0], data.init_score[0]], data.label[0]))
        assert data.label[1] == pytest.approx(data.weight[1])
        assert data.feature_name == data.get_feature_name()

    X, y = load_breast_cancer(return_X_y=True)
    sequence = np.ones(y.shape[0])
    sequence[0] = np.nan
    sequence[1] = np.inf
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    lgb_data = lgb.Dataset(X, sequence, weight=sequence, init_score=sequence, feature_name=feature_names).construct()
    check_asserts(lgb_data)
    lgb_data = lgb.Dataset(X, y).construct()
    lgb_data.set_label(sequence)
    lgb_data.set_weight(sequence)
    lgb_data.set_init_score(sequence)
    lgb_data.set_feature_name(feature_names)
    check_asserts(lgb_data)


def test_dataset_construction_overwrites_user_provided_metadata_fields():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    position = np.array([0.0, 1.0], dtype=np.float32)
    if BuildInfo.has_cuda:
        position = None

    dtrain = lgb.Dataset(
        X,
        params={"min_data_in_bin": 1, "min_data_in_leaf": 1, "verbosity": -1},
        group=[1, 1],
        init_score=[0.312, 0.708],
        label=[1, 2],
        position=position,
        weight=[0.5, 1.5],
    )

    # unconstructed, get_* methods should return whatever was provided
    assert dtrain.group == [1, 1]
    assert dtrain.get_group() == [1, 1]
    assert dtrain.init_score == [0.312, 0.708]
    assert dtrain.get_init_score() == [0.312, 0.708]
    assert dtrain.label == [1, 2]
    assert dtrain.get_label() == [1, 2]
    if not BuildInfo.has_cuda:
        np_assert_array_equal(dtrain.position, np.array([0.0, 1.0], dtype=np.float32), strict=True)
        np_assert_array_equal(dtrain.get_position(), np.array([0.0, 1.0], dtype=np.float32), strict=True)
    assert dtrain.weight == [0.5, 1.5]
    assert dtrain.get_weight() == [0.5, 1.5]

    # before construction, get_field() should raise an exception
    for field_name in ["group", "init_score", "label", "position", "weight"]:
        with pytest.raises(Exception, match=f"Cannot get {field_name} before construct Dataset"):
            dtrain.get_field(field_name)

    # constructed, get_* methods should return numpy arrays, even when the provided
    # input was a list of floats or ints
    dtrain.construct()
    expected_group = np.array([1, 1], dtype=np.int32)
    np_assert_array_equal(dtrain.group, expected_group, strict=True)
    np_assert_array_equal(dtrain.get_group(), expected_group, strict=True)
    # get_field("group") returns a numpy array with boundaries, instead of size
    np_assert_array_equal(dtrain.get_field("group"), np.array([0, 1, 2], dtype=np.int32), strict=True)

    expected_init_score = np.array(
        [0.312, 0.708],
    )
    np_assert_array_equal(dtrain.init_score, expected_init_score, strict=True)
    np_assert_array_equal(dtrain.get_init_score(), expected_init_score, strict=True)
    np_assert_array_equal(dtrain.get_field("init_score"), expected_init_score, strict=True)

    expected_label = np.array([1, 2], dtype=np.float32)
    np_assert_array_equal(dtrain.label, expected_label, strict=True)
    np_assert_array_equal(dtrain.get_label(), expected_label, strict=True)
    np_assert_array_equal(dtrain.get_field("label"), expected_label, strict=True)

    if not BuildInfo.has_cuda:
        # NOTE: "position" is converted to int32 on the C++ side and remapped to dense
        # internal indices in encounter order. Here the input [0, 1] is already dense
        # starting from 0 in encounter order, so the remap is the identity.
        expected_position = np.array([0, 1], dtype=np.int32)
        np_assert_array_equal(dtrain.position, expected_position, strict=True)
        np_assert_array_equal(dtrain.get_position(), expected_position, strict=True)
        np_assert_array_equal(dtrain.get_field("position"), expected_position, strict=True)

    expected_weight = np.array([0.5, 1.5], dtype=np.float32)
    np_assert_array_equal(dtrain.weight, expected_weight, strict=True)
    np_assert_array_equal(dtrain.get_weight(), expected_weight, strict=True)
    np_assert_array_equal(dtrain.get_field("weight"), expected_weight, strict=True)


@pytest.mark.skipif(
    BuildInfo.has_cuda,
    reason="Positions in learning to rank is not supported in CUDA version yet",
)
def test_set_position_updates_self_position_with_remapped_int32_values():
    # Position values are remapped to dense int32 indices in the order they are first
    # encountered. With input [3, 1, 0, 2, 4, 3, 1, 0, 2, 4]:
    #   3 -> 0 (first encountered), 1 -> 1, 0 -> 2, 2 -> 3, 4 -> 4
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.float64)
    position = np.array([3, 1, 0, 2, 4, 3, 1, 0, 2, 4], dtype=np.int64)
    expected = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)

    # set via constructor
    dtrain = lgb.Dataset(
        X,
        label=y,
        position=position,
        params={"min_data_in_bin": 1, "min_data_in_leaf": 1, "verbosity": -1},
    ).construct()
    np_assert_array_equal(dtrain.position, expected, strict=True)
    np_assert_array_equal(dtrain.get_position(), expected, strict=True)
    np_assert_array_equal(dtrain.get_field("position"), expected, strict=True)

    # set via set_position() on an already-constructed Dataset
    dtrain2 = lgb.Dataset(
        X,
        label=y,
        params={"min_data_in_bin": 1, "min_data_in_leaf": 1, "verbosity": -1},
    ).construct()
    dtrain2.set_position(position)
    np_assert_array_equal(dtrain2.position, expected, strict=True)
    np_assert_array_equal(dtrain2.get_position(), expected, strict=True)
    np_assert_array_equal(dtrain2.get_field("position"), expected, strict=True)


def test_choose_param_value():
    original_params = {
        "local_listen_port": 1234,
        "port": 2222,
        "metric": "auc",
        "num_trees": 81,
        "n_iter": 13,
    }

    # should resolve duplicate aliases, and prefer the main parameter
    params = lgb.basic._choose_param_value(
        main_param_name="local_listen_port", params=original_params, default_value=5555
    )
    assert params["local_listen_port"] == 1234
    assert "port" not in params

    # should choose the highest priority alias and set that value on main param
    # if only aliases are used
    params = lgb.basic._choose_param_value(main_param_name="num_iterations", params=params, default_value=17)
    assert params["num_iterations"] == 13
    assert "num_trees" not in params
    assert "n_iter" not in params

    # should use the default if main param and aliases are missing
    params = lgb.basic._choose_param_value(main_param_name="learning_rate", params=params, default_value=0.789)
    assert params["learning_rate"] == 0.789

    # all changes should be made on copies and not modify the original
    expected_params = {
        "local_listen_port": 1234,
        "port": 2222,
        "metric": "auc",
        "num_trees": 81,
        "n_iter": 13,
    }
    assert original_params == expected_params


def test_choose_param_value_preserves_nones():
    # preserves None found for main param and still removes aliases
    params = lgb.basic._choose_param_value(
        main_param_name="num_threads",
        params={"num_threads": None, "n_jobs": 4, "objective": "regression"},
        default_value=2,
    )
    assert params == {"num_threads": None, "objective": "regression"}

    # correctly chooses value when only an alias is provided
    params = lgb.basic._choose_param_value(
        main_param_name="num_threads", params={"n_jobs": None, "objective": "regression"}, default_value=2
    )
    assert params == {"num_threads": None, "objective": "regression"}

    # adds None if that's given as the default and param not found
    params = lgb.basic._choose_param_value(
        main_param_name="min_data_in_leaf", params={"objective": "regression"}, default_value=None
    )
    assert params == {"objective": "regression", "min_data_in_leaf": None}


@pytest.mark.parametrize("objective_alias", lgb.basic._ConfigAliases.get("objective"))
def test_choose_param_value_objective(objective_alias):
    # If callable is found in objective
    params = {objective_alias: dummy_obj}
    params = lgb.basic._choose_param_value(main_param_name="objective", params=params, default_value=None)
    assert params["objective"] == dummy_obj

    # Value in params should be preferred to the default_value passed from keyword arguments
    params = {objective_alias: dummy_obj}
    params = lgb.basic._choose_param_value(main_param_name="objective", params=params, default_value=mse_obj)
    assert params["objective"] == dummy_obj

    # None of objective or its aliases in params, but default_value is callable.
    params = {}
    params = lgb.basic._choose_param_value(main_param_name="objective", params=params, default_value=mse_obj)
    assert params["objective"] == mse_obj


@pytest.mark.parametrize("collection", ["1d_np", "2d_np", "pd_float", "pd_str", "1d_list", "2d_list"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_list_to_1d_numpy(collection, dtype, rng):
    collection2y = {
        "1d_np": rng.uniform(size=(10,)),
        "2d_np": rng.uniform(size=(10, 1)),
        "pd_float": rng.uniform(size=(10,)),
        "pd_str": ["a", "b"],
        "1d_list": [1] * 10,
        "2d_list": [[1], [2]],
    }
    y = collection2y[collection]
    custom_name = "my_custom_variable"

    if collection.startswith("pd"):
        pd = pytest.importorskip("pandas")
        y = pd.Series(y)
        if pd.api.types.is_object_dtype(y):
            with pytest.raises(
                ValueError,
                match=r"pandas dtypes must be int, float or bool\.\nFields with bad pandas dtypes: 0: object",
            ):
                lgb.basic._list_to_1d_numpy(data=y, dtype=np.float32, name=custom_name)
            return
        elif pd.api.types.is_string_dtype(y):
            with pytest.raises(
                ValueError, match=r"pandas dtypes must be int, float or bool\.\nFields with bad pandas dtypes: 0: str"
            ):
                lgb.basic._list_to_1d_numpy(data=y, dtype=np.float32, name=custom_name)
            return

    if isinstance(y, np.ndarray) and len(y.shape) == 2:
        with pytest.warns(UserWarning, match="column-vector"):
            lgb.basic._list_to_1d_numpy(data=y, dtype=np.float32, name=custom_name)
        return
    elif isinstance(y, list) and isinstance(y[0], list):
        err_msg = (
            rf"Wrong type\(list\) for {custom_name}.\n"
            r"It should be list, numpy 1-D array or pandas Series"
        )
        with pytest.raises(TypeError, match=err_msg):
            lgb.basic._list_to_1d_numpy(data=y, dtype=np.float32, name=custom_name)
        return

    result = lgb.basic._list_to_1d_numpy(data=y, dtype=dtype, name=custom_name)
    assert result.size == 10
    assert result.dtype == dtype


@pytest.mark.parametrize("init_score_type", ["array", "dataframe", "list"])
def test_init_score_for_multiclass_classification(init_score_type, rng):
    init_score = [[i * 10 + j for j in range(3)] for i in range(10)]
    if init_score_type == "array":
        init_score = np.array(init_score)
    elif init_score_type == "dataframe":
        pd = pytest.importorskip("pandas")
        init_score = pd.DataFrame(init_score)
    data = rng.uniform(size=(10, 2))
    ds = lgb.Dataset(data, init_score=init_score).construct()
    np.testing.assert_equal(ds.get_field("init_score"), init_score)
    np.testing.assert_equal(ds.init_score, init_score)


def test_smoke_custom_parser(tmp_path):
    data_path = Path(__file__).absolute().parents[2] / "examples" / "binary_classification" / "binary.train"
    parser_config_file = tmp_path / "parser.ini"
    with open(parser_config_file, "w") as fout:
        fout.write('{"className": "dummy", "id": "1"}')

    data = lgb.Dataset(data_path, params={"parser_config_file": parser_config_file})
    with pytest.raises(
        lgb.basic.LightGBMError, match="Cannot find parser class 'dummy', please register first or check config format"
    ):
        data.construct()


def test_param_aliases():
    aliases = lgb.basic._ConfigAliases.aliases
    assert isinstance(aliases, dict)
    assert len(aliases) > 100
    assert all(isinstance(i, list) for i in aliases.values())
    assert all(len(i) >= 1 for i in aliases.values())
    assert all(k in v for k, v in aliases.items())
    assert lgb.basic._ConfigAliases.get("config", "task") == {"config", "config_file", "task", "task_type"}
    assert lgb.basic._ConfigAliases.get_sorted("min_data_in_leaf") == [
        "min_data_in_leaf",
        "min_data",
        "min_samples_leaf",
        "min_child_samples",
        "min_data_per_leaf",
    ]


def _bad_gradients(preds, _):
    rng = np.random.default_rng()
    # "bad" = 1 element too many
    size = (len(preds) + 1,)
    return rng.standard_normal(size=size), rng.uniform(size=size)


def _good_gradients(preds, _):
    rng = np.random.default_rng()
    return rng.standard_normal(size=preds.shape), rng.uniform(size=preds.shape)


def test_custom_objective_safety(rng):
    nrows = 100
    X = rng.standard_normal(size=(nrows, 5))
    y_binary = np.arange(nrows) % 2
    classes = [0, 1, 2]
    nclass = len(classes)
    y_multiclass = np.arange(nrows) % nclass
    ds_binary = lgb.Dataset(X, y_binary).construct()
    ds_multiclass = lgb.Dataset(X, y_multiclass).construct()
    bad_bst_binary = lgb.Booster({"objective": "none"}, ds_binary)
    good_bst_binary = lgb.Booster({"objective": "none"}, ds_binary)
    bad_bst_multi = lgb.Booster({"objective": "none", "num_class": nclass}, ds_multiclass)
    good_bst_multi = lgb.Booster({"objective": "none", "num_class": nclass}, ds_multiclass)
    good_bst_binary.update(fobj=_good_gradients)
    with pytest.raises(ValueError, match=re.escape("number of models per one iteration (1)")):
        bad_bst_binary.update(fobj=_bad_gradients)
    good_bst_multi.update(fobj=_good_gradients)
    with pytest.raises(ValueError, match=re.escape(f"number of models per one iteration ({nclass})")):
        bad_bst_multi.update(fobj=_bad_gradients)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("feature_name", [["x1", "x2"], "auto"])
def test_no_copy_when_single_float_dtype_dataframe(dtype, feature_name, rng):
    pd = pytest.importorskip("pandas")
    X = rng.uniform(size=(10, 2)).astype(dtype)
    # copy=False is necessary because starting with pandas 3.0, pd.DataFrame() creates
    # a copy of the input numpy array by default
    # ref: https://github.com/pandas-dev/pandas/issues/58913
    df = pd.DataFrame(X, copy=False)
    built_data = lgb.basic._data_from_pandas(
        data=df, feature_name=feature_name, categorical_feature="auto", pandas_categorical=None
    )[0]
    assert built_data.dtype == dtype
    assert np.shares_memory(X, built_data)


@pytest.mark.parametrize("feature_name", [["x1"], [42], "auto"])
@pytest.mark.parametrize("categories", ["seen", "unseen"])
def test_categorical_code_conversion_doesnt_modify_original_data(feature_name, categories, rng):
    pd = pytest.importorskip("pandas")
    X = rng.choice(a=["a", "b"], size=(100, 1))
    column_name = "a" if feature_name == "auto" else feature_name[0]
    df = pd.DataFrame(X.copy(), columns=[column_name], dtype="category")
    if categories == "seen":
        pandas_categorical = [["a", "b"]]
    else:
        pandas_categorical = [["a"]]
    data = lgb.basic._data_from_pandas(
        data=df,
        feature_name=feature_name,
        categorical_feature="auto",
        pandas_categorical=pandas_categorical,
    )[0]
    # check that the original data wasn't modified
    np.testing.assert_equal(df[column_name], X[:, 0])
    # check that the built data has the codes
    if categories == "seen":
        # if all categories were seen during training we just take the codes
        codes = df[column_name].cat.codes
    else:
        # if we only saw 'a' during training we just replace its code
        # and leave the rest as nan
        a_code = df[column_name].cat.categories.get_loc("a")
        codes = np.where(df[column_name] == "a", a_code, np.nan)
    np.testing.assert_equal(codes, data[:, 0])


def test_pandas_categorical_json_serialization_works(rng):
    # Some 'numpy' types aren't JSON-serializable by default, this checks
    # that having them in pandas_categorical doesn't cause issues for JSON
    # serialization of models.
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "np_float": pd.Categorical(np.array([1.1, 2.2, 3.3], dtype=np.float32)),
            "np_bool": pd.Categorical(np.array([True, False, False], dtype=np.bool_)),
            "np_int": pd.Categorical(np.array([1, 2, 3], dtype=np.int64)),
        }
    )

    # Confirm that the array dtypes also become the category dtypes,
    # with some allowance for precision differences in older pandas versions,
    # like 'np._bool_' arrays being stored as "Object" dtype.
    assert df["np_float"].dtype.categories.dtype in (np.float32, np.float64)
    assert df["np_bool"].dtype.categories.dtype in (np.bool_, np.dtype("O"))
    assert df["np_int"].dtype.categories.dtype == np.int64

    dtrain = lgb.Dataset(
        df,
        label=rng.random((df.shape[0],)),
        categorical_feature=["np_float", "np_bool", "np_int"],
    )
    dtrain.construct()

    bst = lgb.train(
        params={
            "objective": "regression",
        },
        train_set=dtrain,
        num_boost_round=2,
    )

    # JSON serialization works
    json_data = bst.dump_model()
    assert json_data["pandas_categorical"] == [
        [float(np.float32(1.1)), float(np.float32(2.2)), float(np.float32(3.3))],
        [False, True],
        [1, 2, 3],
    ]


@pytest.mark.parametrize("min_data_in_bin", [2, 10])
def test_feature_num_bin(min_data_in_bin, rng):
    X = np.vstack(
        [
            rng.uniform(size=(100,)),
            np.array([1, 2] * 50),
            np.array([0, 1, 2] * 33 + [0]),
            np.array([1, 2] * 49 + 2 * [np.nan]),
            np.zeros(100),
            rng.choice(a=[0, 1], size=(100,)),
        ]
    ).T
    n_continuous = X.shape[1] - 1
    feature_name = [f"x{i}" for i in range(n_continuous)] + ["cat1"]
    ds_kwargs = {
        "params": {"min_data_in_bin": min_data_in_bin},
        "categorical_feature": [n_continuous],  # last feature
    }
    ds = lgb.Dataset(X, feature_name=feature_name, **ds_kwargs).construct()
    expected_num_bins = [
        100 // min_data_in_bin + 1,  # extra bin for zero
        3,  # 0, 1, 2
        3,  # 0, 1, 2
        4,  # 0, 1, 2 + nan
        0,  # unused
        3,  # 0, 1 + nan
    ]
    actual_num_bins = [ds.feature_num_bin(i) for i in range(X.shape[1])]
    assert actual_num_bins == expected_num_bins
    # test using defined feature names
    bins_by_name = [ds.feature_num_bin(name) for name in feature_name]
    assert bins_by_name == expected_num_bins
    # test using default feature names
    ds_no_names = lgb.Dataset(X, **ds_kwargs).construct()
    default_names = [f"Column_{i}" for i in range(X.shape[1])]
    bins_by_default_name = [ds_no_names.feature_num_bin(name) for name in default_names]
    assert bins_by_default_name == expected_num_bins
    # check for feature indices outside of range
    num_features = X.shape[1]
    with pytest.raises(
        lgb.basic.LightGBMError,
        match=(
            f"Tried to retrieve number of bins for feature index {num_features}, "
            f"but the valid feature indices are \\[0, {num_features - 1}\\]."
        ),
    ):
        ds.feature_num_bin(num_features)


def test_feature_num_bin_with_max_bin_by_feature(rng):
    X = rng.uniform(size=(100, 3))
    max_bin_by_feature = rng.integers(low=3, high=30, size=X.shape[1])
    ds = lgb.Dataset(X, params={"max_bin_by_feature": max_bin_by_feature}).construct()
    actual_num_bins = [ds.feature_num_bin(i) for i in range(X.shape[1])]
    np.testing.assert_equal(actual_num_bins, max_bin_by_feature)


def test_set_leaf_output():
    X, y = load_breast_cancer(return_X_y=True)
    ds = lgb.Dataset(X, y)
    bst = lgb.Booster({"num_leaves": 2}, ds)
    bst.update()
    y_pred = bst.predict(X)
    for leaf_id in range(2):
        leaf_output = bst.get_leaf_output(tree_id=0, leaf_id=leaf_id)
        bst.set_leaf_output(tree_id=0, leaf_id=leaf_id, value=leaf_output + 1)
    np.testing.assert_allclose(bst.predict(X), y_pred + 1)


def test_feature_names_are_set_correctly_when_no_feature_names_passed_into_Dataset(rng):
    ds = lgb.Dataset(
        data=rng.standard_normal(size=(100, 3)),
    )
    assert ds.construct().feature_name == ["Column_0", "Column_1", "Column_2"]


def test_set_feature_name_updates_has_non_default_feature_names(rng):
    ds = lgb.Dataset(data=rng.standard_normal(size=(100, 3)), label=rng.integers(0, 2, size=100))
    assert ds._has_non_default_feature_names is False
    ds.construct()
    assert ds._has_non_default_feature_names is False
    assert ds.get_feature_name() == ["Column_0", "Column_1", "Column_2"]
    ds.set_feature_name(["a", "b", "c"])
    assert ds._has_non_default_feature_names is True
    assert ds.get_feature_name() == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("feature_name", "duplicate_name"),
    [
        (["a", "a"], "a"),
        (["_", " "], "_"),
        (["a b", "a_b"], "a_b"),
    ],
)
def test_construct_raises_informative_error_on_duplicate_feature_names(rng, feature_name, duplicate_name):
    X = rng.uniform(size=(10, len(feature_name)))
    expected_error = (
        f"After preprocessing (including replacing whitespace with '_'), multiple features named '{duplicate_name}' "
        "found in Dataset. Ensure that feature names are unique and do not contain whitespace."
    )
    with pytest.raises(lgb.basic.LightGBMError, match=f"^{re.escape(expected_error)}$"):
        lgb.Dataset(X, feature_name=feature_name).construct()


# NOTE: this intentionally contains values where num_leaves <, ==, and > (max_depth^2)
@pytest.mark.parametrize(("max_depth", "num_leaves"), [(-1, 3), (-1, 50), (5, 3), (5, 31), (5, 32), (8, 3), (8, 31)])
def test_max_depth_warning_is_not_raised_if_num_leaves_is_also_provided(capsys, num_leaves, max_depth):
    X, y = make_blobs(n_samples=1_000, n_features=1, centers=2)
    lgb.Booster(
        params={
            "objective": "binary",
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "num_iterations": 1,
            "verbose": 0,
        },
        train_set=lgb.Dataset(X, label=y),
    )
    assert "Provided parameters constrain tree depth" not in capsys.readouterr().out


# NOTE: max_depth < 5 is significant here because the default for num_leaves=31. With max_depth=5,
#       a full depth-wise tree would have 2^5 = 32 leaves.
@pytest.mark.parametrize("max_depth", [1, 2, 3, 4])
def test_max_depth_warning_is_not_raised_if_max_depth_gt_1_and_lt_5_and_num_leaves_omitted(capsys, max_depth):
    X, y = make_blobs(n_samples=1_000, n_features=1, centers=2)
    lgb.Booster(
        params={
            "objective": "binary",
            "max_depth": max_depth,
            "num_iterations": 1,
            "verbose": 0,
        },
        train_set=lgb.Dataset(X, label=y),
    )
    assert "Provided parameters constrain tree depth" not in capsys.readouterr().out


@pytest.mark.parametrize("max_depth", [5, 6, 7, 8, 9])
def test_max_depth_warning_is_raised_if_max_depth_gte_5_and_num_leaves_omitted(capsys, max_depth):
    X, y = make_blobs(n_samples=1_000, n_features=1, centers=2)
    lgb.Booster(
        params={
            "objective": "binary",
            "max_depth": max_depth,
            "num_iterations": 1,
            "verbose": 0,
        },
        train_set=lgb.Dataset(X, label=y),
    )
    expected_warning = (
        f"[LightGBM] [Warning] Provided parameters constrain tree depth (max_depth={max_depth}) without explicitly "
        f"setting 'num_leaves'. This can lead to underfitting. To resolve this warning, pass 'num_leaves' (<={2**max_depth}) "
        "in params. Alternatively, pass (max_depth=-1) and just use 'num_leaves' to constrain model complexity."
    )
    assert expected_warning in capsys.readouterr().out


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_no_copy_in_dataset_from_numpy_2d(rng, order, dtype):
    X = rng.random(size=(100, 3))
    X = np.require(X, dtype=dtype, requirements=order)
    X1d, layout = lgb.basic._np2d_to_np1d(X)
    if order == "F":
        assert layout == lgb.basic._C_API_IS_COL_MAJOR
    else:
        assert layout == lgb.basic._C_API_IS_ROW_MAJOR
    if dtype == "float32":
        assert np.shares_memory(X, X1d)
    else:
        # makes a copy
        assert not np.shares_memory(X, X1d)


def test_equal_datasets_from_row_major_and_col_major_data(tmp_path):
    # row-major dataset
    X_row, y = make_blobs(n_samples=1_000, n_features=3, centers=2)
    assert X_row.flags["C_CONTIGUOUS"]
    assert not X_row.flags["F_CONTIGUOUS"]
    ds_row = lgb.Dataset(X_row, y)
    ds_row_path = tmp_path / "ds_row.txt"
    ds_row._dump_text(ds_row_path)

    # col-major dataset
    X_col = np.asfortranarray(X_row)
    assert X_col.flags["F_CONTIGUOUS"]
    assert not X_col.flags["C_CONTIGUOUS"]
    ds_col = lgb.Dataset(X_col, y)
    ds_col_path = tmp_path / "ds_col.txt"
    ds_col._dump_text(ds_col_path)

    # check datasets are equal
    assert filecmp.cmp(ds_row_path, ds_col_path)


def test_equal_datasets_from_one_and_several_matrices_w_different_layouts(rng, tmp_path):
    # several matrices
    mats = [np.require(rng.random(size=(100, 2)), requirements=order) for order in ("C", "F", "F", "C")]
    several_path = tmp_path / "several.txt"
    lgb.Dataset(mats)._dump_text(several_path)

    # one matrix
    mat = np.vstack(mats)
    one_path = tmp_path / "one.txt"
    lgb.Dataset(mat)._dump_text(one_path)

    assert filecmp.cmp(one_path, several_path)


@pytest.mark.parametrize(
    "field_name",
    [
        "group",
        "init_score",
        pytest.param(
            "position",
            marks=pytest.mark.skipif(
                BuildInfo.has_cuda,
                reason="Positions in learning to rank is not supported in CUDA version yet",
            ),
        ),
        "weight",
    ],
)
def test_set_field_none_removes_field(rng, field_name):
    X = rng.uniform(size=(10, 1))
    d = lgb.Dataset(X).construct()

    if field_name == "group":
        field = [5, 5]
        expected = np.array([0, 5, 10], dtype=np.int32)
    elif field_name == "position":
        field = [100, 20, 100, 10, 30, 10, 30, 10, 30, 30]
        expected = np.array([0, 1, 0, 2, 3, 2, 3, 2, 3, 3], dtype=np.int32)
    else:
        field = rng.uniform(size=10)
        expected = field.astype(np.float64 if field_name == "init_score" else np.float32)

    out = d.set_field(field_name, field)
    assert out is d

    np_assert_array_equal(d.get_field(field_name), expected, strict=True)

    d.set_field(field_name, None)
    assert d.get_field(field_name) is None


def test_booster_eval_adds_new_valid_dataset() -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        *load_breast_cancer(return_X_y=True),
        test_size=0.1,
        random_state=42,
    )
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)
    booster = lgb.Booster(
        params={
            "deterministic": True,
            "force_row_wise": True,
            "objective": "binary",
            "metric": ["auc", "binary_error"],
            "num_iterations": 2,
            "num_leaves": 3,
            "num_threads": 1,
            "seed": 708,
            "verbose": -1,
        },
        train_set=train_set,
    )
    assert booster._Booster__num_dataset == 1
    assert booster.valid_sets == []

    result = booster.eval(valid_set, name="test")

    assert booster._Booster__num_dataset == 2
    assert booster.valid_sets == [valid_set]
    assert len(result) == 2
    assert isinstance(result, list)

    # first metric - AUC
    dataset_name, metric_name, metric_value, maximize = result[0]
    assert dataset_name == "test"
    assert metric_name == "auc"
    assert metric_value >= 0.50
    assert maximize is True

    # second metric - binary error
    dataset_name, metric_name, metric_value, maximize = result[1]
    assert dataset_name == "test"
    assert metric_name == "binary_error"
    assert metric_value >= 0.40
    assert maximize is False


def test_refit_correctly_handles_categorical_features_in_params(rng) -> None:
    rng = np.random.default_rng()
    X = rng.integers(1, 10, size=(1_000, 3))
    y = rng.uniform(size=(X.shape[0],))

    # Dataset with 'categorical_feature" keyword arg
    dtrain = lgb.Dataset(X, label=y, categorical_feature=[0, 2])
    bst = lgb.train(
        params={
            "num_leaves": 7,
            "verbose": -1,
        },
        train_set=dtrain,
        num_boost_round=2,
    )

    # 'categorical_column' is correctly set in params
    assert bst.params["categorical_column"] == [0, 2]

    # refit() should not raise a warning
    X_new = rng.integers(1, 10, size=(10, 3))
    y_new = rng.uniform(size=(X_new.shape[0],))
    with warnings.catch_warnings() as w:
        warnings.simplefilter("always")
        bst.refit(X_new, y_new)
        if w:
            assert not any(
                re.search(r"has been found in .*params.* and will be ignored", str(warning.message)) for warning in w
            )

    # round-trip to and from a model string
    loaded_bst = lgb.Booster(model_str=bst.model_to_string())

    # that round-trip sets Booster.params to all model parameters, using the "main"
    # ones, not any aliases
    assert loaded_bst.params["categorical_feature"] == [0, 2]
    assert "categorical_column" not in loaded_bst.params

    # case 1: 'categorical_feature' keyword arg not passed
    # result: should succeed and not warn
    loaded_bst_new = loaded_bst.refit(X_new, y_new)
    assert loaded_bst_new.params["categorical_column"] == [0, 2]

    # case 2: 'categorical_feature' keyword arg passed, but identical to what's in params
    # result: should succeed and not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loaded_bst_new = loaded_bst.refit(X_new, y_new, categorical_feature=[0, 2])
        assert loaded_bst_new.params["categorical_column"] == [0, 2]
        if w:
            assert not any(
                re.search(r"has been found in .*params.* and will be ignored", str(warning.message)) for warning in w
            )

    # case 3: 'categorical_feature' keyword arg passed, different value
    # result: informative error
    with pytest.raises(
        lgb.basic.LightGBMError,
        match=re.escape("Using refit() to change which columns are treated as categorical is not supported"),
    ):
        loaded_bst_new = loaded_bst.refit(X_new, y_new, categorical_feature=[0, 1])


def test_eval_result_no_std_dev_works():
    # can be constructed with positional args
    eval_tuple = ("dataset_2", "mape", 0.567, True)
    eval_result = lgb.EvalResult(*eval_tuple)

    # can be constructed with keyword args
    assert (
        lgb.EvalResult(
            dataset_name="dataset_2",
            metric_value=0.567,
            maximize=True,
            metric_name="mape",
        )
        == eval_result
    )

    # passes isinstance() check
    assert isinstance(eval_result, tuple)

    # length is correct
    assert len(eval_result) == 4

    # keyword and positional access works
    assert eval_result.dataset_name == "dataset_2"
    assert eval_result[0] == "dataset_2"
    assert eval_result.metric_name == "mape"
    assert eval_result[1] == "mape"
    assert eval_result.metric_value == 0.567
    assert eval_result[2] == 0.567
    assert eval_result.maximize is True
    assert eval_result[3] is True

    # trying to unpack back to 4 variables works
    dataset_name, metric_name, metric_value, maximize = eval_result
    assert dataset_name == "dataset_2"
    assert metric_name == "mape"
    assert metric_value == 0.567
    assert maximize is True

    # trying to unpack to 5 variables fails the same way other tuple unpacking fails
    with pytest.raises(ValueError, match=re.escape("not enough values to unpack (expected 5, got 4)")):
        a, b, c, d, e = eval_result

    # trying to unpack to 3 variables fails the same way other tuple unpacking fails
    with pytest.raises(ValueError, match=re.escape("too many values to unpack (expected 3)")):
        a, b, c = eval_result

    # accessing 5th element directly still works
    assert eval_result.metric_std_dev is None
    assert eval_result[4] is None

    # reports as not a cv() tuple
    assert eval_result.is_cv_result() is False


def test_eval_result_with_std_dev_works():
    # can be constructed with positional args
    eval_tuple = ("dataset_2", "mape", 2.004, True, 0.617)
    eval_result = lgb.EvalResult(*eval_tuple)

    # can be constructed with keyword args
    assert (
        lgb.EvalResult(
            dataset_name="dataset_2",
            metric_value=2.004,
            maximize=True,
            metric_name="mape",
            metric_std_dev=0.617,
        )
        == eval_result
    )

    # passes isinstance() check
    assert isinstance(eval_result, tuple)

    # length is correct
    assert len(eval_result) == 5

    # keyword and positional access works
    assert eval_result.dataset_name == "dataset_2"
    assert eval_result[0] == "dataset_2"
    assert eval_result.metric_name == "mape"
    assert eval_result[1] == "mape"
    assert eval_result.metric_value == 2.004
    assert eval_result[2] == 2.004
    assert eval_result.maximize is True
    assert eval_result[3] is True
    assert eval_result.metric_std_dev == 0.617
    assert eval_result[4] == 0.617

    # trying to unpack back to 5 variables works
    dataset_name, metric_name, metric_value, maximize, metric_std_dev = eval_result
    assert dataset_name == "dataset_2"
    assert metric_name == "mape"
    assert metric_value == 2.004
    assert maximize is True
    assert metric_std_dev == 0.617

    # trying to unpack to 6 variables fails the same way other tuple unpacking fails
    with pytest.raises(ValueError, match=re.escape("not enough values to unpack (expected 6, got 5)")):
        a, b, c, d, e, f = eval_result

    # trying to unpack to 4 variables fails the same way other tuple unpacking fails
    with pytest.raises(ValueError, match=re.escape("too many values to unpack (expected 4)")):
        a, b, c, d = eval_result

    # reports as a cv() tuple
    assert eval_result.is_cv_result() is True
