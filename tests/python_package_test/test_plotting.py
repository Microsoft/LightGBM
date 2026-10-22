# coding: utf-8
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from .utils import load_breast_cancer, make_synthetic_regression


@pytest.fixture(scope="function")
def matplotlib():
    mpl = pytest.importorskip("matplotlib")
    # use non-interactive, in-memory renderer
    mpl.use("Agg")
    return mpl


@pytest.fixture(scope="module")
def breast_cancer_split():
    return train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.1, random_state=1)


def _categorical_data(category_values_lower_bound, category_values_upper_bound):
    X, y = load_breast_cancer(return_X_y=True)
    X_df = pd.DataFrame()
    rnd = np.random.RandomState(0)
    n_cat_values = rnd.randint(category_values_lower_bound, category_values_upper_bound, size=X.shape[1])
    for i in range(X.shape[1]):
        bins = np.linspace(0, 1, num=n_cat_values[i] + 1)
        X_df[f"cat_col_{i}"] = pd.qcut(X[:, i], q=bins, labels=range(n_cat_values[i])).as_unordered()
    return X_df, y


@pytest.fixture(scope="module")
def train_data(breast_cancer_split):
    X_train, _, y_train, _ = breast_cancer_split
    return lgb.Dataset(X_train, y_train)


@pytest.fixture
def params():
    return {"objective": "binary", "verbose": -1, "num_leaves": 3}


def test_plot_importance(params, breast_cancer_split, train_data, matplotlib):
    X_train, _, y_train, _ = breast_cancer_split

    gbm0 = lgb.train(params, train_data, num_boost_round=10)
    ax0 = lgb.plot_importance(gbm0)
    assert isinstance(ax0, matplotlib.axes.Axes)
    assert ax0.get_title() == "Feature importance"
    assert ax0.get_xlabel() == "Feature importance"
    assert ax0.get_ylabel() == "Features"
    assert len(ax0.patches) <= 30

    gbm1 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, verbose=-1)
    gbm1.fit(X_train, y_train)

    ax1 = lgb.plot_importance(gbm1, color="r", title="t", xlabel="x", ylabel="y")
    assert isinstance(ax1, matplotlib.axes.Axes)
    assert ax1.get_title() == "t"
    assert ax1.get_xlabel() == "x"
    assert ax1.get_ylabel() == "y"
    assert len(ax1.patches) <= 30
    for patch in ax1.patches:
        assert patch.get_facecolor() == (1.0, 0, 0, 1.0)  # red

    ax2 = lgb.plot_importance(gbm0, color=["r", "y", "g", "b"], title=None, xlabel=None, ylabel=None)
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert ax2.get_title() == ""
    assert ax2.get_xlabel() == ""
    assert ax2.get_ylabel() == ""
    assert len(ax2.patches) <= 30
    assert ax2.patches[0].get_facecolor() == (1.0, 0, 0, 1.0)  # r
    assert ax2.patches[1].get_facecolor() == (0.75, 0.75, 0, 1.0)  # y
    assert ax2.patches[2].get_facecolor() == (0, 0.5, 0, 1.0)  # g
    assert ax2.patches[3].get_facecolor() == (0, 0, 1.0, 1.0)  # b

    ax3 = lgb.plot_importance(
        gbm0, title="t @importance_type@", xlabel="x @importance_type@", ylabel="y @importance_type@"
    )
    assert isinstance(ax3, matplotlib.axes.Axes)
    assert ax3.get_title() == "t @importance_type@"
    assert ax3.get_xlabel() == "x split"
    assert ax3.get_ylabel() == "y @importance_type@"
    assert len(ax3.patches) <= 30

    ax4 = lgb.plot_importance(gbm0, title=None, xlabel=None, ylabel=None, xlim=(0, 30))
    assert isinstance(ax4, matplotlib.axes.Axes)
    assert ax4.get_title() == ""
    assert ax4.get_xlabel() == ""
    assert ax4.get_ylabel() == ""
    assert ax4.get_xlim() == (0, 30)
    assert len(ax4.patches) <= 30

    with pytest.raises(TypeError, match="xlim must be a tuple of 2 elements."):
        lgb.plot_importance(gbm0, title=None, xlabel=None, ylabel=None, xlim="not a tuple")

    ax5 = lgb.plot_importance(gbm0, title=None, xlabel=None, ylabel=None, ylim=(0, 30))
    assert isinstance(ax5, matplotlib.axes.Axes)
    assert ax5.get_title() == ""
    assert ax5.get_xlabel() == ""
    assert ax5.get_ylabel() == ""
    assert ax5.get_ylim() == (0, 30)
    assert len(ax5.patches) <= 30

    with pytest.raises(TypeError, match="ylim must be a tuple of 2 elements."):
        lgb.plot_importance(gbm0, title=None, xlabel=None, ylabel=None, ylim="not a tuple")

    ax6 = lgb.plot_importance(gbm0, title=None, xlabel=None, ylabel=None, figsize=(0, 30))
    assert isinstance(ax6, matplotlib.axes.Axes)
    assert ax6.get_title() == ""
    assert ax6.get_xlabel() == ""
    assert ax6.get_ylabel() == ""
    assert list(ax6.get_figure().get_size_inches()) == [0, 30]
    assert len(ax6.patches) <= 30

    with pytest.raises(TypeError, match="figsize must be a tuple of 2 elements."):
        lgb.plot_importance(gbm0, title=None, xlabel=None, ylabel=None, figsize="not a tuple")

    # test max_num_features parameter
    total_features = len(gbm0.feature_importance())
    assert total_features > 5, "model must have more than 5 features to test max_num_features"
    ax7 = lgb.plot_importance(gbm0, max_num_features=5)
    assert isinstance(ax7, matplotlib.axes.Axes)
    assert len(ax7.patches) == 5
    # verify the 5 displayed features are the top 5 by importance
    importance = gbm0.feature_importance()
    feature_names = gbm0.feature_name()
    sorted_pairs = sorted(zip(feature_names, importance, strict=True), key=lambda x: x[1])
    top5_names = [name for name, _ in sorted_pairs[-5:]]
    displayed_labels = [label.get_text() for label in ax7.get_yticklabels()]
    assert displayed_labels == top5_names

    gbm2 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, verbose=-1, importance_type="gain")
    gbm2.fit(X_train, y_train)

    def get_bounds_of_first_patch(axes):
        return axes.patches[0].get_extents().bounds

    first_bar1 = get_bounds_of_first_patch(lgb.plot_importance(gbm1))
    first_bar2 = get_bounds_of_first_patch(lgb.plot_importance(gbm1, importance_type="split"))
    first_bar3 = get_bounds_of_first_patch(lgb.plot_importance(gbm1, importance_type="gain"))
    first_bar4 = get_bounds_of_first_patch(lgb.plot_importance(gbm2))
    first_bar5 = get_bounds_of_first_patch(lgb.plot_importance(gbm2, importance_type="split"))
    first_bar6 = get_bounds_of_first_patch(lgb.plot_importance(gbm2, importance_type="gain"))

    assert first_bar1 == first_bar2
    assert first_bar1 == first_bar5
    assert first_bar3 == first_bar4
    assert first_bar3 == first_bar6
    assert first_bar1 != first_bar3


def test_plot_importance_zero_splits(matplotlib):
    X, y = load_breast_cancer(return_X_y=True)
    model = lgb.train(
        params={
            "min_data_in_bin": X.shape[0] + 1,
            "objective": "regression",
            "verbose": -1,
        },
        train_set=lgb.Dataset(X, label=y),
        num_boost_round=1,
    )
    with pytest.raises(ValueError, match="No non-zero feature importances found"):
        lgb.plot_importance(model)
    # ignore_zero=False should still produce a valid plot
    ax = lgb.plot_importance(model, ignore_zero=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.patches) == X.shape[1]


def test_plot_split_value_histogram(params, breast_cancer_split, train_data, matplotlib):
    X_train, _, y_train, _ = breast_cancer_split

    gbm0 = lgb.train(params, train_data, num_boost_round=10)
    ax0 = lgb.plot_split_value_histogram(gbm0, 27)
    assert isinstance(ax0, matplotlib.axes.Axes)
    assert ax0.get_title() == "Split value histogram for feature with index 27"
    assert ax0.get_xlabel() == "Feature split value"
    assert ax0.get_ylabel() == "Count"
    assert len(ax0.patches) <= 2

    gbm1 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, verbose=-1)
    gbm1.fit(X_train, y_train)

    ax1 = lgb.plot_split_value_histogram(
        gbm1,
        gbm1.booster_.feature_name()[27],
        figsize=(10, 5),
        title="Histogram for feature @index/name@ @feature@",
        xlabel="x",
        ylabel="y",
        color="r",
    )
    assert isinstance(ax1, matplotlib.axes.Axes)
    title = f"Histogram for feature name {gbm1.booster_.feature_name()[27]}"
    assert ax1.get_title() == title
    assert ax1.get_xlabel() == "x"
    assert ax1.get_ylabel() == "y"
    assert len(ax1.patches) <= 2
    for patch in ax1.patches:
        assert patch.get_facecolor() == (1.0, 0, 0, 1.0)  # red

    ax2 = lgb.plot_split_value_histogram(
        gbm0, 27, bins=10, color=["r", "y", "g", "b"], title=None, xlabel=None, ylabel=None
    )
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert ax2.get_title() == ""
    assert ax2.get_xlabel() == ""
    assert ax2.get_ylabel() == ""
    assert len(ax2.patches) == 10
    assert ax2.patches[0].get_facecolor() == (1.0, 0, 0, 1.0)  # r
    assert ax2.patches[1].get_facecolor() == (0.75, 0.75, 0, 1.0)  # y
    assert ax2.patches[2].get_facecolor() == (0, 0.5, 0, 1.0)  # g
    assert ax2.patches[3].get_facecolor() == (0, 0, 1.0, 1.0)  # b

    # test xlim parameter
    ax3 = lgb.plot_split_value_histogram(gbm0, 27, xlim=(0, 100), title=None, xlabel=None, ylabel=None)
    assert isinstance(ax3, matplotlib.axes.Axes)
    assert ax3.get_title() == ""
    assert ax3.get_xlabel() == ""
    assert ax3.get_ylabel() == ""
    assert ax3.get_xlim() == (0, 100)

    with pytest.raises(TypeError, match="xlim must be a tuple of 2 elements."):
        lgb.plot_split_value_histogram(gbm0, 27, xlim="not a tuple")

    ax4 = lgb.plot_split_value_histogram(gbm0, 27, ylim=(0, 100), title=None, xlabel=None, ylabel=None)
    assert isinstance(ax4, matplotlib.axes.Axes)
    assert ax4.get_title() == ""
    assert ax4.get_xlabel() == ""
    assert ax4.get_ylabel() == ""
    assert ax4.get_ylim() == (0, 100)

    with pytest.raises(TypeError, match="ylim must be a tuple of 2 elements."):
        lgb.plot_split_value_histogram(gbm0, 27, ylim="not a tuple")

    with pytest.raises(
        ValueError, match="Cannot plot split value histogram, because feature 0 was not used in splitting"
    ):
        lgb.plot_split_value_histogram(gbm0, 0)  # was not used in splitting


def test_plot_tree(breast_cancer_split, matplotlib):
    pytest.importorskip("graphviz")
    X_train, _, y_train, _ = breast_cancer_split
    gbm = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, verbose=-1)
    gbm.fit(X_train, y_train)

    with pytest.raises(IndexError, match="tree_index is out of range."):
        lgb.plot_tree(gbm, tree_index=83)

    ax = lgb.plot_tree(gbm, tree_index=3, figsize=(15, 8), show_info=["split_gain"])
    assert isinstance(ax, matplotlib.axes.Axes)
    w, h = ax.axes.get_figure().get_size_inches()
    assert int(w) == 15
    assert int(h) == 8


def test_create_tree_digraph(tmp_path, breast_cancer_split):
    graphviz = pytest.importorskip("graphviz")
    X_train, _, y_train, _ = breast_cancer_split
    constraints = [-1, 1] * int(X_train.shape[1] / 2)
    gbm = lgb.LGBMClassifier(
        n_estimators=10,
        num_leaves=3,
        verbose=-1,
        monotone_constraints=constraints,
        deterministic=True,
        n_jobs=1,
        force_row_wise=True,
    )
    gbm.fit(X_train, y_train)

    with pytest.raises(IndexError, match="tree_index is out of range."):
        lgb.create_tree_digraph(gbm, tree_index=83)

    # test basic graph attributes and show_info for internal nodes
    graph = lgb.create_tree_digraph(
        gbm,
        tree_index=3,
        show_info=["split_gain", "internal_value", "internal_weight"],
        name="Tree4",
        node_attr={"color": "red"},
        directory=tmp_path,
    )
    graph.render(view=False)
    assert isinstance(graph, graphviz.Digraph)
    assert graph.name == "Tree4"
    assert len(graph.node_attr) == 1
    assert graph.node_attr["color"] == "red"
    assert len(graph.graph_attr) == 0
    assert len(graph.edge_attr) == 0
    graph_body = "".join(graph.body)
    assert "leaf" in graph_body
    assert "gain" in graph_body
    assert "value" in graph_body
    assert "weight" in graph_body
    assert "#ffdddd" in graph_body
    assert "#ddffdd" in graph_body
    assert "data" not in graph_body
    assert "count" not in graph_body

    # test internal_count
    graph = lgb.create_tree_digraph(gbm, tree_index=0, show_info=["internal_count"])
    graph_body = "".join(graph.body)
    assert "count: 200" in graph_body, graph_body
    assert "count: 512" in graph_body

    # test data_percentage for internal and leaf nodes
    graph = lgb.create_tree_digraph(gbm, tree_index=0, show_info=["data_percentage"])
    graph_body = "".join(graph.body)
    assert "60.94% of data" in graph_body
    assert "9.77% of data" in graph_body
    assert "29.30% of data" in graph_body
    assert "39.06% of data" in graph_body
    assert "100.00% of data" in graph_body

    # test leaf_weight
    graph = lgb.create_tree_digraph(gbm, tree_index=0, show_info=["leaf_weight"])
    graph_body = "".join(graph.body)
    assert "72.657 weight" in graph_body
    assert "11.644 weight" in graph_body
    assert "4.931 weight" in graph_body

    # test leaf_count
    graph = lgb.create_tree_digraph(gbm, tree_index=0, show_info=["leaf_count"])
    graph_body = "".join(graph.body)
    assert "count: 312" in graph_body
    assert "count: 50" in graph_body
    assert "count: 150" in graph_body


def test_tree_with_categories_below_max_category_values(tmp_path):
    graphviz = pytest.importorskip("graphviz")
    X_train, y_train = _categorical_data(2, 10)
    params = {
        "n_estimators": 10,
        "num_leaves": 3,
        "min_data_in_bin": 1,
        "force_col_wise": True,
        "deterministic": True,
        "num_threads": 1,
        "seed": 708,
        "verbose": -1,
    }
    gbm = lgb.LGBMClassifier(**params)
    gbm.fit(X_train, y_train)

    with pytest.raises(IndexError, match="tree_index is out of range."):
        lgb.create_tree_digraph(gbm, tree_index=83)

    graph = lgb.create_tree_digraph(
        gbm,
        tree_index=3,
        show_info=["split_gain", "internal_value", "internal_weight"],
        name="Tree4",
        node_attr={"color": "red"},
        max_category_values=10,
        directory=tmp_path,
    )
    graph.render(view=False)
    assert isinstance(graph, graphviz.Digraph)
    assert graph.name == "Tree4"
    assert len(graph.node_attr) == 1
    assert graph.node_attr["color"] == "red"
    assert len(graph.graph_attr) == 0
    assert len(graph.edge_attr) == 0
    graph_body = "".join(graph.body)
    assert "leaf" in graph_body
    assert "gain" in graph_body
    assert "value" in graph_body
    assert "weight" in graph_body
    assert "data" not in graph_body
    assert "count" not in graph_body
    assert "||...||" not in graph_body


def test_tree_with_categories_above_max_category_values(tmp_path):
    graphviz = pytest.importorskip("graphviz")
    X_train, y_train = _categorical_data(20, 30)
    params = {
        "n_estimators": 10,
        "num_leaves": 3,
        "min_data_in_bin": 1,
        "force_col_wise": True,
        "deterministic": True,
        "num_threads": 1,
        "seed": 708,
        "verbose": -1,
    }
    gbm = lgb.LGBMClassifier(**params)
    gbm.fit(X_train, y_train)

    with pytest.raises(IndexError, match="tree_index is out of range."):
        lgb.create_tree_digraph(gbm, tree_index=83)

    graph = lgb.create_tree_digraph(
        gbm,
        tree_index=9,
        show_info=["split_gain", "internal_value", "internal_weight"],
        name="Tree4",
        node_attr={"color": "red"},
        max_category_values=4,
        directory=tmp_path,
    )
    graph.render(view=False)
    assert isinstance(graph, graphviz.Digraph)
    assert graph.name == "Tree4"
    assert len(graph.node_attr) == 1
    assert graph.node_attr["color"] == "red"
    assert len(graph.graph_attr) == 0
    assert len(graph.edge_attr) == 0
    graph_body = "".join(graph.body)
    assert "leaf" in graph_body
    assert "gain" in graph_body
    assert "value" in graph_body
    assert "weight" in graph_body
    assert "data" not in graph_body
    assert "count" not in graph_body
    assert "||...||" in graph_body


@pytest.mark.parametrize("use_missing", [True, False])
@pytest.mark.parametrize("zero_as_missing", [True, False])
def test_numeric_split_direction(use_missing, zero_as_missing):
    X, y = make_synthetic_regression()
    rng = np.random.RandomState(0)
    zero_mask = rng.rand(X.shape[0]) < 0.05
    X[zero_mask, :] = 0
    if use_missing:
        nan_mask = ~zero_mask & (rng.rand(X.shape[0]) < 0.1)
        X[nan_mask, :] = np.nan
    ds = lgb.Dataset(X, y)
    params = {
        "num_leaves": 127,
        "min_child_samples": 1,
        "use_missing": use_missing,
        "zero_as_missing": zero_as_missing,
    }
    bst = lgb.train(params, ds, num_boost_round=1)

    case_with_zero = X[zero_mask][[0]]
    expected_leaf_zero = bst.predict(case_with_zero, pred_leaf=True)[0]
    node = bst.dump_model()["tree_info"][0]["tree_structure"]
    while "decision_type" in node:
        direction = lgb.plotting._determine_direction_for_numeric_split(
            fval=case_with_zero[0][node["split_feature"]],
            threshold=node["threshold"],
            missing_type_str=node["missing_type"],
            default_left=node["default_left"],
        )
        node = node["left_child"] if direction == "left" else node["right_child"]
    assert node["leaf_index"] == expected_leaf_zero

    if use_missing:
        case_with_nan = X[nan_mask][[0]]
        expected_leaf_nan = bst.predict(case_with_nan, pred_leaf=True)[0]
        node = bst.dump_model()["tree_info"][0]["tree_structure"]
        while "decision_type" in node:
            direction = lgb.plotting._determine_direction_for_numeric_split(
                fval=case_with_nan[0][node["split_feature"]],
                threshold=node["threshold"],
                missing_type_str=node["missing_type"],
                default_left=node["default_left"],
            )
            node = node["left_child"] if direction == "left" else node["right_child"]
        assert node["leaf_index"] == expected_leaf_nan
        if zero_as_missing:
            # zeros treated as missing -> same leaf as NaN
            assert expected_leaf_zero == expected_leaf_nan
        else:
            # zeros are regular values -> different leaf from NaN
            assert expected_leaf_zero != expected_leaf_nan


def test_example_case_in_tree_digraph():
    pytest.importorskip("graphviz")
    rng = np.random.RandomState(0)
    x1 = rng.rand(100)
    cat = rng.randint(1, 3, size=x1.size)
    X = np.vstack([x1, cat]).T
    y = x1 + 2 * cat
    feature_name = ["x1", "cat"]
    ds = lgb.Dataset(X, y, feature_name=feature_name, categorical_feature=["cat"])

    num_round = 3
    bst = lgb.train({"num_leaves": 7}, ds, num_boost_round=num_round)
    mod = bst.dump_model()
    example_case = X[[0]]
    makes_categorical_splits = False
    seen_indices = set()
    for i in range(num_round):
        graph = lgb.create_tree_digraph(bst, example_case=example_case, tree_index=i)
        gbody = graph.body
        node = mod["tree_info"][i]["tree_structure"]
        while "decision_type" in node:  # iterate through the splits
            split_index = node["split_index"]

            node_in_graph = [n for n in gbody if f"split{split_index}" in n and "->" not in n]
            assert len(node_in_graph) == 1
            seen_indices.add(gbody.index(node_in_graph[0]))

            edge_to_node = [e for e in gbody if f"-> split{split_index}" in e]
            if node["decision_type"] == "<=":
                direction = lgb.plotting._determine_direction_for_numeric_split(
                    fval=example_case[0][node["split_feature"]],
                    threshold=node["threshold"],
                    missing_type_str=node["missing_type"],
                    default_left=node["default_left"],
                )
            else:
                makes_categorical_splits = True
                direction = lgb.plotting._determine_direction_for_categorical_split(
                    example_case[0][node["split_feature"]], node["threshold"]
                )
            node = node["left_child"] if direction == "left" else node["right_child"]
            assert "color=blue" in node_in_graph[0]
            if edge_to_node:
                assert len(edge_to_node) == 1
                assert "color=blue" in edge_to_node[0]
                seen_indices.add(gbody.index(edge_to_node[0]))
        # we're in a leaf now
        leaf_index = node["leaf_index"]
        leaf_in_graph = [n for n in gbody if f"leaf{leaf_index}" in n and "->" not in n]
        edge_to_leaf = [e for e in gbody if f"-> leaf{leaf_index}" in e]
        assert len(leaf_in_graph) == 1
        assert "color=blue" in leaf_in_graph[0]
        assert len(edge_to_leaf) == 1
        assert "color=blue" in edge_to_leaf[0]
        seen_indices.update([gbody.index(leaf_in_graph[0]), gbody.index(edge_to_leaf[0])])

        # check that the rest of the elements have black color
        remaining_elements = [e for i, e in enumerate(graph.body) if i not in seen_indices and "graph" not in e]
        assert all("color=black" in e for e in remaining_elements)

        # check that we got to the expected leaf
        expected_leaf = bst.predict(example_case, start_iteration=i, num_iteration=1, pred_leaf=True)[0]
        assert leaf_index == expected_leaf
    assert makes_categorical_splits


@pytest.mark.parametrize("input_type", ["array", "dataframe"])
def test_empty_example_case_on_tree_digraph_raises_error(input_type):
    pytest.importorskip("graphviz")
    X, y = make_synthetic_regression()
    if input_type == "dataframe":
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)
        example_case = pd.DataFrame(X[:0])
    else:
        example_case = X[:0]
    ds = lgb.Dataset(X, y)
    bst = lgb.train({"num_leaves": 3}, ds, num_boost_round=1)
    with pytest.raises(ValueError, match="example_case must have a single row."):
        lgb.create_tree_digraph(bst, tree_index=0, example_case=example_case)


def test_plot_metrics(params, breast_cancer_split, train_data, matplotlib):
    X_train, X_test, y_train, y_test = breast_cancer_split
    test_data = lgb.Dataset(X_test, y_test, reference=train_data)
    params.update({"metric": {"binary_logloss", "binary_error"}})

    evals_result0 = {}
    lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=["v1", "v2"],
        num_boost_round=10,
        callbacks=[lgb.record_evaluation(evals_result0)],
    )
    with pytest.warns(UserWarning, match="More than one metric available, picking one to plot."):
        ax0 = lgb.plot_metric(evals_result0)
    assert isinstance(ax0, matplotlib.axes.Axes)
    assert ax0.get_title() == "Metric during training"
    assert ax0.get_xlabel() == "Iterations"
    assert ax0.get_ylabel() in {"binary_logloss", "binary_error"}
    legend_items = ax0.get_legend().get_texts()
    assert len(legend_items) == 2
    assert legend_items[0].get_text() == "v1"
    assert legend_items[1].get_text() == "v2"

    ax1 = lgb.plot_metric(evals_result0, metric="binary_error")
    assert isinstance(ax1, matplotlib.axes.Axes)
    assert ax1.get_title() == "Metric during training"
    assert ax1.get_xlabel() == "Iterations"
    assert ax1.get_ylabel() == "binary_error"
    legend_items = ax1.get_legend().get_texts()
    assert len(legend_items) == 2
    assert legend_items[0].get_text() == "v1"
    assert legend_items[1].get_text() == "v2"

    ax2 = lgb.plot_metric(evals_result0, metric="binary_logloss", dataset_names=["v2"])
    assert isinstance(ax2, matplotlib.axes.Axes)
    assert ax2.get_title() == "Metric during training"
    assert ax2.get_xlabel() == "Iterations"
    assert ax2.get_ylabel() == "binary_logloss"
    legend_items = ax2.get_legend().get_texts()
    assert len(legend_items) == 1
    assert legend_items[0].get_text() == "v2"

    ax3 = lgb.plot_metric(
        evals_result0,
        metric="binary_logloss",
        dataset_names=["v1"],
        title="Metric @metric@",
        xlabel="Iterations @metric@",
        ylabel='Value of "@metric@"',
        figsize=(5, 5),
        dpi=600,
        grid=False,
    )
    assert isinstance(ax3, matplotlib.axes.Axes)
    assert ax3.get_title() == "Metric @metric@"
    assert ax3.get_xlabel() == "Iterations @metric@"
    assert ax3.get_ylabel() == 'Value of "binary_logloss"'
    legend_items = ax3.get_legend().get_texts()
    assert len(legend_items) == 1
    assert legend_items[0].get_text() == "v1"
    assert ax3.get_figure().get_figheight() == 5
    assert ax3.get_figure().get_figwidth() == 5
    assert ax3.get_figure().get_dpi() == 600
    for grid_line in ax3.get_xgridlines():
        assert not grid_line.get_visible()
    for grid_line in ax3.get_ygridlines():
        assert not grid_line.get_visible()

    evals_result1 = {}
    lgb.train(params, train_data, num_boost_round=10, callbacks=[lgb.record_evaluation(evals_result1)])
    with pytest.raises(ValueError, match="eval results cannot be empty."):
        lgb.plot_metric(evals_result1)

    gbm2 = lgb.LGBMClassifier(n_estimators=10, num_leaves=3, verbose=-1)
    gbm2.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    ax4 = lgb.plot_metric(gbm2, title=None, xlabel=None, ylabel=None)
    assert isinstance(ax4, matplotlib.axes.Axes)
    assert ax4.get_title() == ""
    assert ax4.get_xlabel() == ""
    assert ax4.get_ylabel() == ""
    legend_items = ax4.get_legend().get_texts()
    assert len(legend_items) == 1
    assert legend_items[0].get_text() == "valid_0"

    # test xlim parameter
    ax5 = lgb.plot_metric(evals_result0, metric="binary_logloss", xlim=(0, 15), title=None, xlabel=None, ylabel=None)
    assert isinstance(ax5, matplotlib.axes.Axes)
    assert ax5.get_title() == ""
    assert ax5.get_xlabel() == ""
    assert ax5.get_ylabel() == ""
    assert ax5.get_xlim() == (0, 15)

    with pytest.raises(TypeError, match="xlim must be a tuple of 2 elements."):
        lgb.plot_metric(evals_result0, metric="binary_logloss", xlim="not a tuple")

    ax6 = lgb.plot_metric(evals_result0, metric="binary_logloss", ylim=(0, 15), title=None, xlabel=None, ylabel=None)
    assert isinstance(ax6, matplotlib.axes.Axes)
    assert ax6.get_title() == ""
    assert ax6.get_xlabel() == ""
    assert ax6.get_ylabel() == ""
    assert ax6.get_ylim() == (0, 15)

    with pytest.raises(TypeError, match="ylim must be a tuple of 2 elements."):
        lgb.plot_metric(evals_result0, metric="binary_logloss", ylim="not a tuple")
