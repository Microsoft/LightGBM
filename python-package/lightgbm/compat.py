# coding: utf-8
"""Compatibility library."""

from typing import Any, List

"""pandas"""
try:
    from pandas import CategoricalDtype as pd_CategoricalDtype
    from pandas import DataFrame as pd_DataFrame
    from pandas import Series as pd_Series
    from pandas import concat

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

    class pd_Series:  # type: ignore
        """Dummy class for pandas.Series."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class pd_DataFrame:  # type: ignore
        """Dummy class for pandas.DataFrame."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class pd_CategoricalDtype:  # type: ignore
        """Dummy class for pandas.CategoricalDtype."""

        def __init__(self, *args: Any, **kwargs: Any):
            pass

    concat = None

__all__: List[str] = []
