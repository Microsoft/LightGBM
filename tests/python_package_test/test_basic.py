# coding: utf-8
import os

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from lightgbm.compat import PANDAS_INSTALLED, pd_Series

from .utils import load_breast_cancer, load_iris


def test_basic(tmp_path):
    X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True),
                                                        test_size=0.1, random_state=2)
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
        "gpu_use_dp": True
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
    assert bst.lower_bound() == pytest.approx(-2.9040190126976606)
    assert bst.upper_bound() == pytest.approx(3.3182142872462883)

    tname = str(tmp_path / "svm_light.dat")
    model_file = str(tmp_path / "model.txt")

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
    np.testing.assert_array_equal(pred_from_matr, pred_from_model_file)

    # check early stopping is working. Make it stop very early, so the scores should be very close to zero
    pred_parameter = {"pred_early_stop": True, "pred_early_stop_freq": 5, "pred_early_stop_margin": 1.5}
    pred_early_stopping = bst.predict(X_test, **pred_parameter)
    # scores likely to be different, but prediction should still be the same
    np.testing.assert_array_equal(np.sign(pred_from_matr), np.sign(pred_early_stopping))

    # test that shape is checked during prediction
    bad_X_test = X_test[:, 1:]
    bad_shape_error_msg = "The number of features in data*"
    np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                   bst.predict, bad_X_test)
    np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                   bst.predict, sparse.csr_matrix(bad_X_test))
    np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                   bst.predict, sparse.csc_matrix(bad_X_test))
    with open(tname, "w+b") as f:
        dump_svmlight_file(bad_X_test, y_test, f)
    np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                   bst.predict, tname)
    with open(tname, "w+b") as f:
        dump_svmlight_file(X_test, y_test, f, zero_based=False)
    np.testing.assert_raises_regex(lgb.basic.LightGBMError, bad_shape_error_msg,
                                   bst.predict, tname)


def test_chunked_dataset():
    X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.1,
                                                        random_state=2)

    chunk_size = X_train.shape[0] // 10 + 1
    X_train = [X_train[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_train.shape[0] // chunk_size + 1)]
    X_test = [X_test[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_test.shape[0] // chunk_size + 1)]

    train_data = lgb.Dataset(X_train, label=y_train, params={"bin_construct_sample_cnt": 100})
    valid_data = train_data.create_valid(X_test, label=y_test, params={"bin_construct_sample_cnt": 100})
    train_data.construct()
    valid_data.construct()


def test_chunked_dataset_linear():
    X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.1,
                                                        random_state=2)
    chunk_size = X_train.shape[0] // 10 + 1
    X_train = [X_train[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_train.shape[0] // chunk_size + 1)]
    X_test = [X_test[i * chunk_size:(i + 1) * chunk_size, :] for i in range(X_test.shape[0] // chunk_size + 1)]
    params = {"bin_construct_sample_cnt": 100, 'linear_tree': True}
    train_data = lgb.Dataset(X_train, label=y_train, params=params)
    valid_data = train_data.create_valid(X_test, label=y_test, params=params)
    train_data.construct()
    valid_data.construct()


def test_subset_group():
    X_train, y_train = load_svmlight_file(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                       '../../examples/lambdarank/rank.train'))
    q_train = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../examples/lambdarank/rank.train.query'))
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    assert len(lgb_train.get_group()) == 201
    subset = lgb_train.subset(list(range(10))).construct()
    subset_group = subset.get_group()
    assert len(subset_group) == 2
    assert subset_group[0] == 1
    assert subset_group[1] == 9


def test_add_features_throws_if_num_data_unequal():
    X1 = np.random.random((100, 1))
    X2 = np.random.random((10, 1))
    d1 = lgb.Dataset(X1).construct()
    d2 = lgb.Dataset(X2).construct()
    with pytest.raises(lgb.basic.LightGBMError):
        d1.add_features_from(d2)


def test_add_features_throws_if_datasets_unconstructed():
    X1 = np.random.random((100, 1))
    X2 = np.random.random((100, 1))
    with pytest.raises(ValueError):
        d1 = lgb.Dataset(X1)
        d2 = lgb.Dataset(X2)
        d1.add_features_from(d2)
    with pytest.raises(ValueError):
        d1 = lgb.Dataset(X1).construct()
        d2 = lgb.Dataset(X2)
        d1.add_features_from(d2)
    with pytest.raises(ValueError):
        d1 = lgb.Dataset(X1)
        d2 = lgb.Dataset(X2).construct()
        d1.add_features_from(d2)


def test_add_features_equal_data_on_alternating_used_unused(tmp_path):
    X = np.random.random((100, 5))
    X[:, [1, 3]] = 0
    names = [f'col_{i}' for i in range(5)]
    for j in range(1, 5):
        d1 = lgb.Dataset(X[:, :j], feature_name=names[:j]).construct()
        d2 = lgb.Dataset(X[:, j:], feature_name=names[j:]).construct()
        d1.add_features_from(d2)
        d1name = str(tmp_path / "d1.txt")
        d1._dump_text(d1name)
        d = lgb.Dataset(X, feature_name=names).construct()
        dname = str(tmp_path / "d.txt")
        d._dump_text(dname)
        with open(d1name, 'rt') as d1f:
            d1txt = d1f.read()
        with open(dname, 'rt') as df:
            dtxt = df.read()
        assert dtxt == d1txt


def test_add_features_same_booster_behaviour(tmp_path):
    X = np.random.random((100, 5))
    X[:, [1, 3]] = 0
    names = [f'col_{i}' for i in range(5)]
    for j in range(1, 5):
        d1 = lgb.Dataset(X[:, :j], feature_name=names[:j]).construct()
        d2 = lgb.Dataset(X[:, j:], feature_name=names[j:]).construct()
        d1.add_features_from(d2)
        d = lgb.Dataset(X, feature_name=names).construct()
        y = np.random.random(100)
        d1.set_label(y)
        d.set_label(y)
        b1 = lgb.Booster(train_set=d1)
        b = lgb.Booster(train_set=d)
        for k in range(10):
            b.update()
            b1.update()
        dname = str(tmp_path / "d.txt")
        d1name = str(tmp_path / "d1.txt")
        b1.save_model(d1name)
        b.save_model(dname)
        with open(dname, 'rt') as df:
            dtxt = df.read()
        with open(d1name, 'rt') as d1f:
            d1txt = d1f.read()
        assert dtxt == d1txt


def test_add_features_from_different_sources():
    pd = pytest.importorskip("pandas")
    n_row = 100
    n_col = 5
    X = np.random.random((n_row, n_col))
    xxs = [X, sparse.csr_matrix(X), pd.DataFrame(X)]
    names = [f'col_{i}' for i in range(n_col)]
    for x_1 in xxs:
        # test that method works even with free_raw_data=True
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=True).construct()
        d2 = lgb.Dataset(x_1, feature_name=names, free_raw_data=True).construct()
        d1.add_features_from(d2)
        assert d1.data is None

        # test that method works but sets raw data to None in case of immergeable data types
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=False).construct()
        d2 = lgb.Dataset([X[:n_row // 2, :], X[n_row // 2:, :]],
                         feature_name=names, free_raw_data=False).construct()
        d1.add_features_from(d2)
        assert d1.data is None

        # test that method works for different data types
        d1 = lgb.Dataset(x_1, feature_name=names, free_raw_data=False).construct()
        res_feature_names = [name for name in names]
        for idx, x_2 in enumerate(xxs, 2):
            original_type = type(d1.get_data())
            d2 = lgb.Dataset(x_2, feature_name=names, free_raw_data=False).construct()
            d1.add_features_from(d2)
            assert isinstance(d1.get_data(), original_type)
            assert d1.get_data().shape == (n_row, n_col * idx)
            res_feature_names += [f'D{idx}_{name}' for name in names]
            assert d1.feature_name == res_feature_names


def test_cegb_affects_behavior(tmp_path):
    X = np.random.random((100, 5))
    X[:, [1, 3]] = 0
    y = np.random.random(100)
    names = [f'col_{i}' for i in range(5)]
    ds = lgb.Dataset(X, feature_name=names).construct()
    ds.set_label(y)
    base = lgb.Booster(train_set=ds)
    for k in range(10):
        base.update()
    basename = str(tmp_path / "basename.txt")
    base.save_model(basename)
    with open(basename, 'rt') as f:
        basetxt = f.read()
    # Set extremely harsh penalties, so CEGB will block most splits.
    cases = [{'cegb_penalty_feature_coupled': [50, 100, 10, 25, 30]},
             {'cegb_penalty_feature_lazy': [1, 2, 3, 4, 5]},
             {'cegb_penalty_split': 1}]
    for case in cases:
        booster = lgb.Booster(train_set=ds, params=case)
        for k in range(10):
            booster.update()
        casename = str(tmp_path / "casename.txt")
        booster.save_model(casename)
        with open(casename, 'rt') as f:
            casetxt = f.read()
        assert basetxt != casetxt


def test_cegb_scaling_equalities(tmp_path):
    X = np.random.random((100, 5))
    X[:, [1, 3]] = 0
    y = np.random.random(100)
    names = [f'col_{i}' for i in range(5)]
    ds = lgb.Dataset(X, feature_name=names).construct()
    ds.set_label(y)
    # Compare pairs of penalties, to ensure scaling works as intended
    pairs = [({'cegb_penalty_feature_coupled': [1, 2, 1, 2, 1]},
              {'cegb_penalty_feature_coupled': [0.5, 1, 0.5, 1, 0.5], 'cegb_tradeoff': 2}),
             ({'cegb_penalty_feature_lazy': [0.01, 0.02, 0.03, 0.04, 0.05]},
              {'cegb_penalty_feature_lazy': [0.005, 0.01, 0.015, 0.02, 0.025], 'cegb_tradeoff': 2}),
             ({'cegb_penalty_split': 1},
              {'cegb_penalty_split': 2, 'cegb_tradeoff': 0.5})]
    for (p1, p2) in pairs:
        booster1 = lgb.Booster(train_set=ds, params=p1)
        booster2 = lgb.Booster(train_set=ds, params=p2)
        for k in range(10):
            booster1.update()
            booster2.update()
        p1name = str(tmp_path / "p1.txt")
        # Reset booster1's parameters to p2, so the parameter section of the file matches.
        booster1.reset_parameter(p2)
        booster1.save_model(p1name)
        with open(p1name, 'rt') as f:
            p1txt = f.read()
        p2name = str(tmp_path / "p2.txt")
        booster2.save_model(p2name)
        with open(p2name, 'rt') as f:
            p2txt = f.read()
        assert p1txt == p2txt


def test_consistent_state_for_dataset_fields():

    def check_asserts(data):
        np.testing.assert_allclose(data.label, data.get_label())
        np.testing.assert_allclose(data.label, data.get_field('label'))
        assert not np.isnan(data.label[0])
        assert not np.isinf(data.label[1])
        np.testing.assert_allclose(data.weight, data.get_weight())
        np.testing.assert_allclose(data.weight, data.get_field('weight'))
        assert not np.isnan(data.weight[0])
        assert not np.isinf(data.weight[1])
        np.testing.assert_allclose(data.init_score, data.get_init_score())
        np.testing.assert_allclose(data.init_score, data.get_field('init_score'))
        assert not np.isnan(data.init_score[0])
        assert not np.isinf(data.init_score[1])
        assert np.all(np.isclose([data.label[0], data.weight[0], data.init_score[0]],
                                 data.label[0]))
        assert data.label[1] == pytest.approx(data.weight[1])
        assert data.feature_name == data.get_feature_name()

    X, y = load_breast_cancer(return_X_y=True)
    sequence = np.ones(y.shape[0])
    sequence[0] = np.nan
    sequence[1] = np.inf
    feature_names = [f'f{i}'for i in range(X.shape[1])]
    lgb_data = lgb.Dataset(X, sequence,
                           weight=sequence, init_score=sequence,
                           feature_name=feature_names).construct()
    check_asserts(lgb_data)
    lgb_data = lgb.Dataset(X, y).construct()
    lgb_data.set_label(sequence)
    lgb_data.set_weight(sequence)
    lgb_data.set_init_score(sequence)
    lgb_data.set_feature_name(feature_names)
    check_asserts(lgb_data)


def test_category_encoding(tmp_path):

    def test_category_encoding_inner(tmp_path, X_train, X_test, y_train, y_test, params, model_prefix):
        # checks that category_encoders works for Dataset constructor
        category_encoders_str = "target,count,target:0.5,raw"
        train_data_1 = lgb.Dataset(X_train, label=y_train, category_encoders=category_encoders_str)
        valid_data_1 = train_data_1.create_valid(X_test, label=y_test)

        categorical_feature = [fidx for fidx in range(X_train.shape[1] // 2)]
        expected_num_features = X_train.shape[1] + 3 * len(categorical_feature)

        params.update({"categorical_feature": categorical_feature})
        booster_1 = lgb.train(params, train_data_1, valid_sets=[valid_data_1], valid_names=["valid_data"], keep_training_booster=True)
        booster_non_keep_training = lgb.train(params, train_data_1, valid_sets=[valid_data_1], valid_names=["valid_data"])
        np.testing.assert_equal(train_data_1.num_feature(), expected_num_features)
        np.testing.assert_equal(valid_data_1.num_feature(), expected_num_features)
        np.testing.assert_equal(len(train_data_1.get_feature_name()), expected_num_features)
        np.testing.assert_equal(booster_1.num_feature(), expected_num_features)
        pred_1 = booster_1.predict(X_test)
        pred_non_keep_training = booster_non_keep_training.predict(X_test)
        np.testing.assert_allclose(pred_1, pred_non_keep_training)
        eval_1 = booster_1.eval(valid_data_1, "valid_data")
        pred_contrib_1 = booster_1.predict(X_test, pred_contrib=True)

        # checks that Dataset with category_encoders can be saved to and load from file
        tmp_dataset = str(tmp_path / 'category_encoding_{}_temp_dataset.bin'.format(model_prefix))
        train_data_1.save_binary(tmp_dataset)

        train_data_2 = lgb.Dataset(tmp_dataset)
        valid_data_2 = lgb.Dataset(X_test, label=y_test, reference=train_data_2)
        booster_2 = lgb.train(params, train_data_2, valid_sets=[valid_data_2], valid_names=["valid_data"], keep_training_booster=True)
        np.testing.assert_equal(train_data_2.num_feature(), expected_num_features)
        np.testing.assert_equal(valid_data_2.num_feature(), expected_num_features)
        np.testing.assert_equal(len(train_data_2.get_feature_name()), expected_num_features)
        np.testing.assert_equal(booster_2.num_feature(), expected_num_features)
        pred_2 = booster_2.predict(X_test)
        eval_2 = booster_2.eval(valid_data_2, "valid_data")
        pred_contrib_2 = booster_2.predict(X_test, pred_contrib=True)
        np.testing.assert_allclose(pred_1, pred_2)
        np.testing.assert_equal(eval_1, eval_2)
        np.testing.assert_allclose(pred_contrib_1, pred_contrib_2)

        # checks that Booster with category_encoders can be saved to and load from file
        model_file = str(tmp_path / "category_encoding_{}_model.txt".format(model_prefix))
        booster_2.save_model(model_file)
        booster_3 = lgb.Booster(params=params, model_file=model_file)
        np.testing.assert_equal(booster_3.num_feature(), expected_num_features)
        pred_3 = booster_3.predict(X_test)
        pred_contrib_3 = booster_3.predict(X_test, pred_contrib=True)
        np.testing.assert_allclose(pred_1, pred_3)
        np.testing.assert_allclose(pred_contrib_1, pred_contrib_3)

        # checks that category_encoders works in params
        train_data_4 = lgb.Dataset(X_train, label=y_train)
        valid_data_4 = lgb.Dataset(X_test, label=y_test).set_reference(train_data_4)
        params.update({"category_encoders": category_encoders_str})
        booster_4 = lgb.train(params, train_data_4, valid_sets=[valid_data_4], valid_names=["valid_data"], keep_training_booster=True)
        valid_data_to_add = lgb.Dataset(X_test, label=y_test, reference=train_data_4)
        booster_4.add_valid(valid_data_to_add, "valid_data_added")
        np.testing.assert_equal(train_data_4.num_feature(), expected_num_features)
        np.testing.assert_equal(valid_data_4.num_feature(), expected_num_features)
        np.testing.assert_equal(len(train_data_4.get_feature_name()), expected_num_features)
        np.testing.assert_equal(booster_4.num_feature(), expected_num_features)
        pred_4 = booster_4.predict(X_test)
        eval_4 = booster_4.eval(valid_data_4, "valid_data")
        eval_valid = booster_4.eval_valid()
        pred_contrib_4 = booster_4.predict(X_test, pred_contrib=True)
        np.testing.assert_allclose(pred_1, pred_4)
        np.testing.assert_equal(eval_1, eval_4)
        # expected eval_1 = [('valid_data', 'auc', 0.9686609686609686, True)]
        # expected eval_valid = [('valid_data', 'auc', 0.9686609686609686, True), ('valid_data_added', 'auc', 0.9686609686609686, True)]
        np.testing.assert_equal(eval_1[0][2], eval_valid[1][2])
        np.testing.assert_allclose(pred_contrib_1, pred_contrib_4)

        # test that target encoding with csr format works
        train_data_csr = lgb.Dataset(sparse.csr_matrix(X_train), label=y_train,
                                     category_encoders=category_encoders_str)
        valid_data_csr = lgb.Dataset(sparse.csr_matrix(X_test), label=y_test,
                                     category_encoders=category_encoders_str, reference=train_data_csr)
        booster_csr = lgb.train(params, train_data_csr, valid_sets=[valid_data_csr], valid_names=["valid_data"], keep_training_booster=True)
        np.testing.assert_equal(train_data_csr.num_feature(), expected_num_features)
        np.testing.assert_equal(valid_data_csr.num_feature(), expected_num_features)
        np.testing.assert_equal(len(train_data_csr.get_feature_name()), expected_num_features)
        np.testing.assert_equal(booster_csr.num_feature(), expected_num_features)
        pred_csr = booster_csr.predict(sparse.csr_matrix(X_test))
        eval_csr = booster_csr.eval(valid_data_csr, "valid_data")
        pred_contrib_csr = booster_csr.predict(sparse.csr_matrix(X_test), pred_contrib=True)
        if model_prefix == "multiclass":
            pred_contrib_csr = np.hstack([csr.toarray() for csr in pred_contrib_csr])
        else:
            pred_contrib_csr = pred_contrib_csr.toarray()
        np.testing.assert_allclose(pred_csr, pred_1)
        np.testing.assert_equal(eval_1, eval_csr)
        np.testing.assert_allclose(pred_contrib_1, pred_contrib_csr)

        # test that target encoding with csc format works
        train_data_csc = lgb.Dataset(sparse.csc_matrix(X_train), label=y_train,
                                     category_encoders=category_encoders_str)
        valid_data_csc = lgb.Dataset(sparse.csc_matrix(X_test), label=y_test,
                                     category_encoders=category_encoders_str, reference=train_data_csc)
        booster_csc = lgb.train(params, train_data_csc, valid_sets=[valid_data_csc], valid_names=["valid_data"])
        np.testing.assert_equal(train_data_csc.num_feature(), expected_num_features)
        np.testing.assert_equal(valid_data_csc.num_feature(), expected_num_features)
        np.testing.assert_equal(len(train_data_csc.get_feature_name()), expected_num_features)
        np.testing.assert_equal(booster_csc.num_feature(), expected_num_features)
        pred_csc = booster_csc.predict(sparse.csc_matrix(X_test))
        eval_csc = booster_csr.eval(valid_data_csc, "valid_data")
        pred_contrib_csc = booster_csc.predict(sparse.csc_matrix(X_test), pred_contrib=True)
        if model_prefix == "multiclass":
            pred_contrib_csc = np.hstack([csc.toarray() for csc in pred_contrib_csc])
        else:
            pred_contrib_csc = pred_contrib_csc.toarray()
        np.testing.assert_allclose(pred_csc, pred_1)
        np.testing.assert_equal(eval_1, eval_csc)
        np.testing.assert_allclose(pred_contrib_1, pred_contrib_csc)

    X_train, X_test, y_train, y_test = train_test_split(*load_breast_cancer(return_X_y=True),
                                                        test_size=0.1, random_state=2)

    params = {
        "objective": "binary",
        "metric": "auc",
        "min_data": 10,
        "num_leaves": 15,
        "verbose": 1,
        "max_bin": 255,
        "max_cat_to_onehot": 1
    }

    test_category_encoding_inner(tmp_path, X_train, X_test, y_train, y_test, params, "binary")

    # test target encoding under multi-class case
    X, y = load_iris(return_X_y=True)
    # convert float to int, so that we can treat them as categorical features
    X = np.array(np.array(X, dtype=np.int), dtype=np.float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,
        'verbose': 1
    }

    test_category_encoding_inner(tmp_path, X_train, X_test, y_train, y_test, params, "multiclass")


def test_choose_param_value():

    original_params = {
        "local_listen_port": 1234,
        "port": 2222,
        "metric": "auc",
        "num_trees": 81
    }

    # should resolve duplicate aliases, and prefer the main parameter
    params = lgb.basic._choose_param_value(
        main_param_name="local_listen_port",
        params=original_params,
        default_value=5555
    )
    assert params["local_listen_port"] == 1234
    assert "port" not in params

    # should choose a value from an alias and set that value on main param
    # if only an alias is used
    params = lgb.basic._choose_param_value(
        main_param_name="num_iterations",
        params=params,
        default_value=17
    )
    assert params["num_iterations"] == 81
    assert "num_trees" not in params

    # should use the default if main param and aliases are missing
    params = lgb.basic._choose_param_value(
        main_param_name="learning_rate",
        params=params,
        default_value=0.789
    )
    assert params["learning_rate"] == 0.789

    # all changes should be made on copies and not modify the original
    expected_params = {
        "local_listen_port": 1234,
        "port": 2222,
        "metric": "auc",
        "num_trees": 81
    }
    assert original_params == expected_params


@pytest.mark.skipif(not PANDAS_INSTALLED, reason='pandas is not installed')
@pytest.mark.parametrize(
    'y',
    [
        np.random.rand(10),
        np.random.rand(10, 1),
        pd_Series(np.random.rand(10)),
        pd_Series(['a', 'b']),
        [1] * 10,
        [[1], [2]]
    ])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_list_to_1d_numpy(y, dtype):
    if isinstance(y, np.ndarray) and len(y.shape) == 2:
        with pytest.warns(UserWarning, match='column-vector'):
            lgb.basic.list_to_1d_numpy(y)
        return
    elif isinstance(y, list) and isinstance(y[0], list):
        with pytest.raises(TypeError):
            lgb.basic.list_to_1d_numpy(y)
        return
    elif isinstance(y, pd_Series) and y.dtype == object:
        with pytest.raises(ValueError):
            lgb.basic.list_to_1d_numpy(y)
        return
    result = lgb.basic.list_to_1d_numpy(y, dtype=dtype)
    assert result.size == 10
    assert result.dtype == dtype
