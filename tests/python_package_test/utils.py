# coding: utf-8
import difflib
import filecmp
import os
import pickle
from functools import lru_cache
from inspect import cleandoc, getfullargspec
from pathlib import Path

import cloudpickle
import joblib
import numpy as np
import sklearn.datasets
from sklearn.utils import check_random_state

import lightgbm as lgb

SERIALIZERS = ["pickle", "joblib", "cloudpickle"]


@lru_cache(maxsize=None)
def load_breast_cancer(**kwargs):
    return sklearn.datasets.load_breast_cancer(**kwargs)


@lru_cache(maxsize=None)
def load_digits(**kwargs):
    return sklearn.datasets.load_digits(**kwargs)


@lru_cache(maxsize=None)
def load_iris(**kwargs):
    return sklearn.datasets.load_iris(**kwargs)


@lru_cache(maxsize=None)
def load_linnerud(**kwargs):
    return sklearn.datasets.load_linnerud(**kwargs)


def make_ranking(
    *, n_samples=100, n_features=20, n_informative=5, gmax=2, group=None, random_gs=False, avg_gs=10, random_state=0
):
    """Generate a learning-to-rank dataset - feature vectors grouped together with
    integer-valued graded relevance scores. Replace this with a sklearn.datasets function
    if ranking objective becomes supported in sklearn.datasets module.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Total number of documents (records) in the dataset.
    n_features : int, optional (default=20)
        Total number of features in the dataset.
    n_informative : int, optional (default=5)
        Number of features that are "informative" for ranking, as they are bias + beta * y
        where bias and beta are standard normal variates. If this is greater than n_features, the dataset will have
        n_features features, all will be informative.
    gmax : int, optional (default=2)
        Maximum graded relevance value for creating relevance/target vector. If you set this to 2, for example, all
        documents in a group will have relevance scores of either 0, 1, or 2.
    group : array-like, optional (default=None)
        1-d array or list of group sizes. When `group` is specified, this overrides n_samples, random_gs, and
        avg_gs by simply creating groups with sizes group[0], ..., group[-1].
    random_gs : bool, optional (default=False)
        True will make group sizes ~ Poisson(avg_gs), False will make group sizes == avg_gs.
    avg_gs : int, optional (default=10)
        Average number of documents (records) in each group.
    random_state : int, optional (default=0)
        Random seed.

    Returns
    -------
    X : 2-d np.ndarray of shape = [n_samples (or np.sum(group)), n_features]
        Input feature matrix for ranking objective.
    y : 1-d np.array of shape = [n_samples (or np.sum(group))]
        Integer-graded relevance scores.
    group_ids : 1-d np.array of shape = [n_samples (or np.sum(group))]
        Array of group ids, each value indicates to which group each record belongs.
    """
    rnd_generator = check_random_state(random_state)

    y_vec, group_id_vec = np.empty((0,), dtype=int), np.empty((0,), dtype=int)
    gid = 0

    # build target, group ID vectors.
    relvalues = range(gmax + 1)

    # build y/target and group-id vectors with user-specified group sizes.
    if group is not None and hasattr(group, "__len__"):
        n_samples = np.sum(group)

        for i, gsize in enumerate(group):
            y_vec = np.concatenate((y_vec, rnd_generator.choice(relvalues, size=gsize, replace=True)))
            group_id_vec = np.concatenate((group_id_vec, [i] * gsize))

    # build y/target and group-id vectors according to n_samples, avg_gs, and random_gs.
    else:
        while len(y_vec) < n_samples:
            gsize = avg_gs if not random_gs else rnd_generator.poisson(avg_gs)

            # groups should contain > 1 element for pairwise learning objective.
            if gsize < 1:
                continue

            y_vec = np.append(y_vec, rnd_generator.choice(relvalues, size=gsize, replace=True))
            group_id_vec = np.append(group_id_vec, [gid] * gsize)
            gid += 1

        y_vec, group_id_vec = y_vec[:n_samples], group_id_vec[:n_samples]

    # build feature data, X. Transform first few into informative features.
    n_informative = max(min(n_features, n_informative), 0)
    X = rnd_generator.uniform(size=(n_samples, n_features))

    for j in range(n_informative):
        bias, coef = rnd_generator.normal(size=2)
        X[:, j] = bias + coef * y_vec

    return X, y_vec, group_id_vec


@lru_cache(maxsize=None)
def make_synthetic_regression(*, n_samples=100, n_features=4, n_informative=2, random_state=42):
    return sklearn.datasets.make_regression(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative, random_state=random_state
    )


def dummy_obj(preds, train_data):
    return np.ones(preds.shape), np.ones(preds.shape)


def mse_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    grad = y_pred - y_true
    hess = np.ones(len(grad))
    return grad, hess


def softmax(x):
    row_wise_max = np.max(x, axis=1).reshape(-1, 1)
    exp_x = np.exp(x - row_wise_max)
    return exp_x / np.sum(exp_x, axis=1).reshape(-1, 1)


def logistic_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sklearn_multiclass_custom_objective(y_true, y_pred, weight=None):
    num_rows, num_class = y_pred.shape
    prob = softmax(y_pred)
    grad_update = np.zeros_like(prob)
    grad_update[np.arange(num_rows), y_true.astype(np.int32)] = -1.0
    grad = prob + grad_update
    factor = num_class / (num_class - 1)
    hess = factor * prob * (1 - prob)
    if weight is not None:
        weight2d = weight.reshape(-1, 1)
        grad *= weight2d
        hess *= weight2d
    return grad, hess


def pickle_obj(obj, filepath, serializer):
    if serializer == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    elif serializer == "joblib":
        joblib.dump(obj, filepath)
    elif serializer == "cloudpickle":
        with open(filepath, "wb") as f:
            cloudpickle.dump(obj, f)
    else:
        raise ValueError(f"Unrecognized serializer type: {serializer}")


def unpickle_obj(filepath, serializer):
    if serializer == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif serializer == "joblib":
        return joblib.load(filepath)
    elif serializer == "cloudpickle":
        with open(filepath, "rb") as f:
            return cloudpickle.load(f)
    else:
        raise ValueError(f"Unrecognized serializer type: {serializer}")


def pickle_and_unpickle_object(obj, serializer):
    with lgb.basic._TempFile() as tmp_file:
        pickle_obj(obj=obj, filepath=tmp_file.name, serializer=serializer)
        obj_from_disk = unpickle_obj(filepath=tmp_file.name, serializer=serializer)
    return obj_from_disk  # noqa: RET504


def assert_silent(capsys) -> None:
    """
    Given a ``CaptureFixture`` instance (from the ``pytest`` built-in ``capsys`` fixture),
    read the recently-captured data into a variable and assert that nothing was written
    to stdout or stderr.

    This is just here to turn 3 lines of repetitive code into 1.

    Note that this does have a side effect... ``capsys.readouterr()`` copies
    from a buffer then frees it. So it will only store into ``.out`` and ``.err`` the
    captured output since the last time that ``.readouterr()`` was called.

    ref: https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html
    """
    captured = capsys.readouterr()
    assert captured.out == "", captured.out
    assert captured.err == "", captured.err


# doing this here, at import time, to ensure it only runs once_per import
# instead of once per assertion
_numpy_testing_supports_strict_kwarg = "strict" in getfullargspec(np.testing.assert_array_equal).kwonlyargs


def np_assert_array_equal(*args, **kwargs):
    """
    np.testing.assert_array_equal() only got the kwarg ``strict`` in June 2022:
    https://github.com/numpy/numpy/pull/21595

    This function is here for testing on older Python (and therefore ``numpy``)
    """
    if not _numpy_testing_supports_strict_kwarg:
        kwargs.pop("strict")
    np.testing.assert_array_equal(*args, **kwargs)


def assert_subtree_valid(root):
    """Recursively checks the validity of a subtree rooted at `root`.

    Currently it only checks whether weights and counts are consistent between
    all parent nodes and their children.

    Parameters
    ----------
    root : dict
        A dictionary representing the root of the subtree.
        It should be produced by dump_model()

    Returns
    -------
    tuple
        A tuple containing the weight and count of the subtree rooted at `root`.
    """
    if "leaf_count" in root:
        return (root["leaf_weight"], root["leaf_count"])

    left_child = root["left_child"]
    right_child = root["right_child"]
    (l_w, l_c) = assert_subtree_valid(left_child)
    (r_w, r_c) = assert_subtree_valid(right_child)
    assert abs(root["internal_weight"] - (l_w + r_w)) <= 1e-3, (
        "root node's internal weight should be approximately the sum of its child nodes' internal weights"
    )
    assert root["internal_count"] == l_c + r_c, (
        "root node's internal count should be exactly the sum of its child nodes' internal counts"
    )
    return (root["internal_weight"], root["internal_count"])


def assert_all_trees_valid(model_dump):
    for idx, tree in enumerate(model_dump["tree_info"]):
        assert tree["tree_index"] == idx, f"tree {idx} should have tree_index={idx}. Full tree: {tree}"
        assert_subtree_valid(tree["tree_structure"])


# This mapping from CI-time environment variables is a placeholder
# until there is a more reliable way to detect which customizations
# LightGBM was built with.
#
# see https://github.com/lightgbm-org/LightGBM/issues/7273
#
class BuildInfo:
    has_cuda = os.getenv("TASK", "") == "cuda"
    has_gpu = os.getenv("TASK", "") == "gpu"
    has_mpi = os.getenv("TASK", "") == "mpi"


def assert_datasets_equal(tmp_path: Path, lhs: lgb.Dataset, rhs: lgb.Dataset) -> None:
    lhs._dump_text(tmp_path / "lhs.txt")
    rhs._dump_text(tmp_path / "rhs.txt")
    assert filecmp.cmp(tmp_path / "lhs.txt", tmp_path / "rhs.txt")


def assert_docstrings_equal(
    class1: type,
    method1: str,
    class2: type,
    method2: str,
    *,
    expected_diff: str = "",
) -> None:

    # this will fail (intentionally) if either class doesn't have the method
    doc1_docstring = cleandoc(getattr(class1, method1).__doc__)
    doc2_docstring = cleandoc(getattr(class2, method2).__doc__)

    # if they do, compare them
    diff = difflib.unified_diff(
        doc1_docstring.splitlines(keepends=True),
        doc2_docstring.splitlines(keepends=True),
        fromfile=f"{class1.__name__}.{method1}",
        tofile=f"{class2.__name__}.{method2}",
        n=0,
    )
    stringified_diff = "".join(line for line in diff)
    assert stringified_diff == expected_diff, f"docs differ:\n\n{stringified_diff}"
