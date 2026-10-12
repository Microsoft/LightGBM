Python API
==========

.. currentmodule:: lightgbm

Data Structure API
------------------

.. autosummary::
    :toctree: pythonapi/

    Dataset
    Booster
    CVBooster
    EvalResult
    Sequence

Training API
------------

.. autosummary::
    :toctree: pythonapi/

    train
    cv

Scikit-learn API
----------------

.. autosummary::
    :toctree: pythonapi/

    LGBMModel
    LGBMClassifier
    LGBMRegressor
    LGBMRanker

Dask API
--------

.. versionadded:: 3.2.0

.. autosummary::
    :toctree: pythonapi/

    DaskLGBMClassifier
    DaskLGBMRegressor
    DaskLGBMRanker

Callbacks
---------

.. autosummary::
    :toctree: pythonapi/

    early_stopping
    log_evaluation
    record_evaluation
    reset_parameter

Plotting
--------

.. autosummary::
    :toctree: pythonapi/

    plot_importance
    plot_split_value_histogram
    plot_metric
    plot_tree
    create_tree_digraph

Utilities
---------

.. autosummary::
    :toctree: pythonapi/

    register_logger
