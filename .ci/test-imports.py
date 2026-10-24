import sys

# import 'numpy' first to load system libraries
# TODO: remove when https://github.com/lightgbm-org/LightGBM/issues/7355 is resolved
import numpy as np  # noqa: F401

import lightgbm as lgb  # noqa: F401

sys.stdout.write("testing Python imports\n")

# modules that shouldn't be imported by default
lazy_imports = {
    "dask",
    "graphviz",
    "matplotlib",
}

unexpected_imports = lazy_imports.intersection(set(sys.modules))

assert unexpected_imports == set(), (
    f"Some libraries that should be loaded lazily are loaded eagerly by 'import lightgbm': {unexpected_imports}"
)

sys.stdout.write("done testing Python imports\n")
