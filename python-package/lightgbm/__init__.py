# coding: utf-8
"""LightGBM, Light Gradient Boosting Machine.

Contributors: https://github.com/lightgbm-org/LightGBM/graphs/contributors.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

# .basic is intentionally loaded as early as possible, to dlopen() lib_lightgbm.{dll,dylib,so}
# and its dependencies as early as possible
# isort: split
from .basic import Booster, Dataset, EvalResult, Sequence, register_logger
from .callback import EarlyStopException, early_stopping, log_evaluation, record_evaluation, reset_parameter
from .engine import CVBooster, cv, train

# isort: split
from .plotting import create_tree_digraph, plot_importance, plot_metric, plot_split_value_histogram, plot_tree

# immediately import these names for type-checkers and IDEs
if TYPE_CHECKING:
    from .dask import DaskLGBMClassifier, DaskLGBMRanker, DaskLGBMRegressor
    from .sklearn import LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor


# NOTE: *args and **kwargs just added here for forward compatibility.
#       They're not strictly required.
def __getattr__(name: str, *args: Any, **kwargs: Any) -> Any:
    """Lazily load some imports.

    Allows for dependencies like 'scikit-learn' to be lazily loaded,
    even though some classes in 'lightgbm' inherit from them.

    This looks a little complex but this type of use was exactly one of the
    motivations for PEP 562 (https://peps.python.org/pep-0562/), so it should continue
    to be supported in future Python versions.
    """
    import importlib  # noqa: PLC0415

    # submodules that unconditionally require scikit-learn,
    # which we don't want to import by default for 'import lightgbm'
    _LAZY_SUBMODULES = {"dask", "sklearn"}

    # handle cases like 'import lightgbm.sklearn'
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)

    # handle cases like 'from lightgbm import LGBMClassifier'
    _DEFERRED_IMPORTS = {
        "LGBMModel": "sklearn",
        "LGBMClassifier": "sklearn",
        "LGBMRanker": "sklearn",
        "LGBMRegressor": "sklearn",
        "DaskLGBMClassifier": "dask",
        "DaskLGBMRanker": "dask",
        "DaskLGBMRegressor": "dask",
    }

    submodule = _DEFERRED_IMPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(importlib.import_module(f".{submodule}", __name__), name)
    # cache it, so __getattr__() isn't called again for this name
    globals()[name] = value
    return value


_version_path = Path(__file__).resolve().parent / "VERSION.txt"
if _version_path.is_file():
    __version__ = _version_path.read_text(encoding="utf-8").strip()

__all__ = [
    "Dataset",
    "Booster",
    "CVBooster",
    "Sequence",
    "register_logger",
    "train",
    "cv",
    "LGBMModel",
    "LGBMRegressor",
    "LGBMClassifier",
    "LGBMRanker",
    "DaskLGBMRegressor",
    "DaskLGBMClassifier",
    "DaskLGBMRanker",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
    "early_stopping",
    "EarlyStopException",
    "EvalResult",
    "plot_importance",
    "plot_split_value_histogram",
    "plot_metric",
    "plot_tree",
    "create_tree_digraph",
]
