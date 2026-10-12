# coding: utf-8
"""Wrapper for C API of LightGBM."""

# This import causes lib_lightgbm.{dll,dylib,so} to be loaded.
# It's intentionally done here, as early as possible, to avoid issues like
# "libgomp.so.1: cannot allocate memory in static TLS block" on aarch64 Linux.
#
# For details, see the "cannot allocate memory in static TLS block" entry in docs/FAQ.rst.
from .libpath import _LIB  # isort: skip

import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple, Union

import narwhals as nw
import narwhals.dependencies as nwd
import narwhals.typing as nwt
import numpy as np
import scipy.sparse

from .compat import PANDAS_INSTALLED, concat, pd_CategoricalDtype, pd_DataFrame, pd_Series

if TYPE_CHECKING:
    from typing import Literal, TypeGuard


__all__ = [
    "Booster",
    "Dataset",
    "LGBMDeprecationWarning",
    "LightGBMError",
    "register_logger",
    "Sequence",
]

_BoosterHandle = ctypes.c_void_p
_DatasetHandle = ctypes.c_void_p
_ctypes_int_ptr = Union[
    "ctypes._Pointer[ctypes.c_int32]",
    "ctypes._Pointer[ctypes.c_int64]",
]
_ctypes_int_array = Union[
    "ctypes.Array[ctypes._Pointer[ctypes.c_int32]]",
    "ctypes.Array[ctypes._Pointer[ctypes.c_int64]]",
]
_ctypes_float_ptr = Union[
    "ctypes._Pointer[ctypes.c_float]",
    "ctypes._Pointer[ctypes.c_double]",
]
_ctypes_float_array = Union[
    "ctypes.Array[ctypes._Pointer[ctypes.c_float]]",
    "ctypes.Array[ctypes._Pointer[ctypes.c_double]]",
]
_LGBM_EvalFunctionResultType = Tuple[str, float, bool]
_LGBM_BoosterBestScoreType = Dict[str, Dict[str, float]]
_LGBM_CategoricalFeatureConfiguration = Union[List[str], List[int], "Literal['auto']"]
_LGBM_FeatureNameConfiguration = Union[List[str], "Literal['auto']"]
_LGBM_GroupType = Union[
    List[float],
    List[int],
    np.ndarray,
    pd_Series,
    nwt.IntoSeries,
]
# 'position' intentionally does not support 'list' inputs
# ref: https://github.com/lightgbm-org/LightGBM/pull/5929#discussion_r1262646998
_LGBM_PositionType = Union[
    np.ndarray,
    pd_Series,
    nwt.IntoSeries,
]
_LGBM_InitScoreType = Union[
    List[float],
    List[List[float]],
    np.ndarray,
    pd_Series,
    pd_DataFrame,
    nwt.IntoSeries,
    nwt.IntoDataFrame,
]
_LGBM_TrainDataType = Union[
    str,
    Path,
    np.ndarray,
    pd_DataFrame,
    scipy.sparse.spmatrix,
    "Sequence",
    List["Sequence"],
    List[np.ndarray],
    nwt.IntoDataFrame,
]
_LGBM_LabelType = Union[
    List[float],
    List[int],
    np.ndarray,
    pd_Series,
    pd_DataFrame,
    nwt.IntoSeries,
    nwt.IntoDataFrame,
]
_LGBM_PredictDataType = Union[
    str,
    Path,
    np.ndarray,
    pd_DataFrame,
    scipy.sparse.spmatrix,
    nwt.IntoDataFrame,
]
_LGBM_PredictReturnType = Union[
    np.ndarray,
    scipy.sparse.spmatrix,
    List[scipy.sparse.spmatrix],
]
_LGBM_PredictSparseReturnType = Union[
    scipy.sparse.spmatrix,
    List[scipy.sparse.spmatrix],
]
_LGBM_WeightType = Union[
    List[float],
    List[int],
    np.ndarray,
    pd_Series,
    nwt.IntoSeries,
]
_LGBM_SetFieldType = Union[
    List[List[float]],
    List[List[int]],
    List[float],
    List[int],
    np.ndarray,
    pd_Series,
    pd_DataFrame,
    nwt.IntoDataFrame,
]

ZERO_THRESHOLD = 1e-35

_MULTICLASS_OBJECTIVES = {"multiclass", "multiclassova", "multiclass_ova", "ova", "ovr", "softmax"}


class LightGBMError(Exception):
    """Error thrown by LightGBM."""

    pass


def _is_zero(x: float) -> bool:
    return -ZERO_THRESHOLD <= x <= ZERO_THRESHOLD


def _get_sample_count(total_nrow: int, params: str) -> int:
    sample_cnt = ctypes.c_int(0)
    _safe_call(
        _LIB.LGBM_GetSampleCount(
            ctypes.c_int32(total_nrow),
            _c_str(params),
            ctypes.byref(sample_cnt),
        )
    )
    return sample_cnt.value


def _np2d_to_np1d(mat: np.ndarray) -> Tuple[np.ndarray, int]:
    dtype: "np.typing.DTypeLike"
    if mat.dtype in (np.float32, np.float64):
        dtype = mat.dtype
    else:
        dtype = np.float32
    order: "Literal['C', 'F']"
    if mat.flags["F_CONTIGUOUS"]:
        order = "F"
        layout = _C_API_IS_COL_MAJOR
    else:
        order = "C"
        layout = _C_API_IS_ROW_MAJOR
    # ensure dtype and order, copies if either do not match
    data = np.asarray(mat, dtype=dtype, order=order)
    # flatten array without copying
    return data.ravel(order=order), layout


class _MissingType(Enum):
    NONE = "None"
    NAN = "NaN"
    ZERO = "Zero"


class _DummyLogger:
    def info(self, msg: str) -> None:
        print(msg)  # noqa: T201

    def warning(self, msg: str) -> None:
        warnings.warn(msg, stacklevel=3)


_LOGGER: Any = _DummyLogger()
_INFO_METHOD_NAME = "info"
_WARNING_METHOD_NAME = "warning"


def _has_method(logger: Any, method_name: str) -> bool:
    return callable(getattr(logger, method_name, None))


def register_logger(
    logger: Any,
    info_method_name: str = "info",
    warning_method_name: str = "warning",
) -> None:
    """Register custom logger.

    Parameters
    ----------
    logger : Any
        Custom logger.
    info_method_name : str, optional (default="info")
        Method used to log info messages.
    warning_method_name : str, optional (default="warning")
        Method used to log warning messages.
    """
    if not _has_method(logger, info_method_name) or not _has_method(logger, warning_method_name):
        raise TypeError(f"Logger must provide '{info_method_name}' and '{warning_method_name}' method")

    global _LOGGER, _INFO_METHOD_NAME, _WARNING_METHOD_NAME
    _LOGGER = logger
    _INFO_METHOD_NAME = info_method_name
    _WARNING_METHOD_NAME = warning_method_name


def _normalize_native_string(func: Callable[[str], None]) -> Callable[[str], None]:
    """Join log messages from native library which come by chunks."""
    msg_normalized: List[str] = []

    @wraps(func)
    def wrapper(msg: str) -> None:
        nonlocal msg_normalized
        if msg.strip() == "":
            msg = "".join(msg_normalized)
            msg_normalized = []
            return func(msg)
        else:
            msg_normalized.append(msg)

    return wrapper


def _log_info(msg: str) -> None:
    getattr(_LOGGER, _INFO_METHOD_NAME)(msg)


def _log_warning(msg: str) -> None:
    getattr(_LOGGER, _WARNING_METHOD_NAME)(msg)


@_normalize_native_string
def _log_native(msg: str) -> None:
    getattr(_LOGGER, _INFO_METHOD_NAME)(msg)


def _log_callback(msg: bytes) -> None:
    """Redirect logs from native library into Python."""
    _log_native(str(msg.decode("utf-8")))


# connect the Python logger to logging in lib_lightgbm
if environ.get("LIGHTGBM_BUILD_DOC", "False") != "True":
    _LIB.LGBM_GetLastError.restype = ctypes.c_char_p
    callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
    _LIB.callback = callback(_log_callback)  # type: ignore[attr-defined]
    if _LIB.LGBM_RegisterLogCallback(_LIB.callback) != 0:
        raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))


_NUMERIC_TYPES = (int, float, bool)


def _safe_call(ret: int) -> None:
    """Check the return value from C API call.

    Parameters
    ----------
    ret : int
        The return value from C API calls.
    """
    if ret != 0:
        raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))


def _is_numeric(obj: Any) -> bool:
    """Check whether object is a number or not, include numpy number, etc."""
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        # TypeError: obj is not a string or a number
        # ValueError: invalid literal
        return False


def _is_numpy_1d_array(data: Any) -> bool:
    """Check whether data is a numpy 1-D array."""
    return isinstance(data, np.ndarray) and len(data.shape) == 1


def _is_numpy_column_array(data: Any) -> bool:
    """Check whether data is a column numpy array."""
    if not isinstance(data, np.ndarray):
        return False
    shape = data.shape
    return len(shape) == 2 and shape[1] == 1


def _cast_numpy_array_to_dtype(array: np.ndarray, dtype: "np.typing.DTypeLike") -> np.ndarray:
    """Cast numpy array to given dtype."""
    if array.dtype == dtype:
        return array
    return array.astype(dtype=dtype, copy=False)


def _is_1d_list(data: Any) -> bool:
    """Check whether data is a 1-D list."""
    return isinstance(data, list) and (not data or _is_numeric(data[0]))


def _is_list_of_numpy_arrays(data: Any) -> "TypeGuard[List[np.ndarray]]":
    return isinstance(data, list) and all(isinstance(x, np.ndarray) for x in data)


def _is_list_of_sequences(data: Any) -> "TypeGuard[List[Sequence]]":
    return isinstance(data, list) and all(isinstance(x, Sequence) for x in data)


def _is_1d_collection(data: Any) -> bool:
    """Check whether data is a 1-D collection."""
    return _is_numpy_1d_array(data) or _is_numpy_column_array(data) or _is_1d_list(data) or isinstance(data, pd_Series)


def _list_to_1d_numpy(
    *,
    data: Any,
    dtype: "np.typing.DTypeLike",
    name: str,
) -> np.ndarray:
    """Convert data to numpy 1-D array."""
    if _is_numpy_1d_array(data):
        return _cast_numpy_array_to_dtype(data, dtype)
    elif _is_numpy_column_array(data):
        _log_warning("Converting column-vector to 1d array")
        array = data.ravel()
        return _cast_numpy_array_to_dtype(array, dtype)
    elif _is_1d_list(data):
        return np.asarray(data, dtype=dtype)
    elif isinstance(data, pd_Series):
        _check_for_bad_pandas_dtypes(data.to_frame().dtypes)
        return np.asarray(data, dtype=dtype)  # SparseArray should be supported as well
    else:
        raise TypeError(
            f"Wrong type({type(data).__name__}) for {name}.\nIt should be list, numpy 1-D array or pandas Series"
        )


def _is_numpy_2d_array(data: Any) -> bool:
    """Check whether data is a numpy 2-D array."""
    return isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] > 1


def _is_2d_list(data: Any) -> bool:
    """Check whether data is a 2-D list."""
    return isinstance(data, list) and len(data) > 0 and _is_1d_list(data[0])


def _is_2d_collection(data: Any) -> bool:
    """Check whether data is a 2-D collection."""
    return _is_numpy_2d_array(data) or _is_2d_list(data) or isinstance(data, pd_DataFrame)


_pycapsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_pycapsule_get_pointer.restype = ctypes.c_void_p
_pycapsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _extract_arrow_stream_capsule_pointer(pycapsule: object) -> ctypes.c_void_p:
    """Extract the raw pointer from a PyCapsule returned by __arrow_c_stream__."""
    return ctypes.c_void_p(_pycapsule_get_pointer(pycapsule, b"arrow_array_stream"))


def _data_to_2d_numpy(
    data: Any,
    dtype: "np.typing.DTypeLike",
    name: str,
) -> np.ndarray:
    """Convert data to numpy 2-D array."""
    if _is_numpy_2d_array(data):
        return _cast_numpy_array_to_dtype(data, dtype)
    if _is_2d_list(data):
        return np.array(data, dtype=dtype)
    if isinstance(data, pd_DataFrame):
        _check_for_bad_pandas_dtypes(data.dtypes)
        return _cast_numpy_array_to_dtype(data.values, dtype)
    raise TypeError(
        f"Wrong type({type(data).__name__}) for {name}.\n"
        "It should be list of lists, numpy 2-D array or pandas DataFrame"
    )


def _cfloat32_array_to_numpy(*, cptr: "ctypes._Pointer", length: int) -> np.ndarray:
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.ctypeslib.as_array(cptr, shape=(length,)).copy()
    else:
        raise RuntimeError("Expected float pointer")


def _cfloat64_array_to_numpy(*, cptr: "ctypes._Pointer", length: int) -> np.ndarray:
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.ctypeslib.as_array(cptr, shape=(length,)).copy()
    else:
        raise RuntimeError("Expected double pointer")


def _cint32_array_to_numpy(*, cptr: "ctypes._Pointer", length: int) -> np.ndarray:
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.ctypeslib.as_array(cptr, shape=(length,)).copy()
    else:
        raise RuntimeError("Expected int32 pointer")


def _cint64_array_to_numpy(*, cptr: "ctypes._Pointer", length: int) -> np.ndarray:
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int64)):
        return np.ctypeslib.as_array(cptr, shape=(length,)).copy()
    else:
        raise RuntimeError("Expected int64 pointer")


def _c_str(string: str) -> ctypes.c_char_p:
    """Convert a Python string to C string."""
    return ctypes.c_char_p(string.encode("utf-8"))


def _c_array(ctype: type, values: List[Any]) -> ctypes.Array:
    """Convert a Python array to C array."""
    return (ctype * len(values))(*values)  # type: ignore[operator]


def _to_string(x: Union[int, float, str, List]) -> str:
    if isinstance(x, list):
        val_list = ",".join(str(val) for val in x)
        return f"[{val_list}]"
    else:
        return str(x)


def _param_dict_to_str(data: Optional[Dict[str, Any]]) -> str:
    """Convert Python dictionary to string, which is passed to C API."""
    if data is None or not data:
        return ""
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or _is_numpy_1d_array(val):
            pairs.append(f"{key}={','.join(map(_to_string, val))}")
        elif isinstance(val, (str, Path, _NUMERIC_TYPES)) or _is_numeric(val):
            pairs.append(f"{key}={val}")
        elif val is not None:
            raise TypeError(f"Unknown type of parameter:{key}, got:{type(val).__name__}")
    return " ".join(pairs)


class _TempFile:
    """Proxy class to workaround errors on Windows."""

    def __enter__(self) -> "_TempFile":
        with NamedTemporaryFile(prefix="lightgbm_tmp_", delete=True) as f:
            self.name = f.name
            self.path = Path(self.name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.path.is_file():
            self.path.unlink()


# DeprecationWarning is not shown by default, so let's create our own with higher level
# ref: https://peps.python.org/pep-0565/#additional-use-case-for-futurewarning
class LGBMDeprecationWarning(FutureWarning):
    """Custom deprecation warning."""

    pass


class _ConfigAliases:
    # lazy evaluation to allow import without dynamic library, e.g., for docs generation
    aliases = None

    @staticmethod
    def _get_all_param_aliases() -> Dict[str, List[str]]:
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
        _safe_call(
            _LIB.LGBM_DumpParamAliases(
                ctypes.c_int64(buffer_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer,
            )
        )
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, re-allocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
            _safe_call(
                _LIB.LGBM_DumpParamAliases(
                    ctypes.c_int64(actual_len),
                    ctypes.byref(tmp_out_len),
                    ptr_string_buffer,
                )
            )
        return json.loads(
            string_buffer.value.decode("utf-8"), object_hook=lambda obj: {k: [k] + v for k, v in obj.items()}
        )

    @classmethod
    def get(cls, *args: str) -> Set[str]:
        if cls.aliases is None:
            cls.aliases = cls._get_all_param_aliases()
        ret = set()
        for i in args:
            ret.update(cls.get_sorted(i))
        return ret

    @classmethod
    def get_sorted(cls, name: str) -> List[str]:
        if cls.aliases is None:
            cls.aliases = cls._get_all_param_aliases()
        return cls.aliases.get(name, [name])

    @classmethod
    def get_by_alias(cls, *args: str) -> Set[str]:
        if cls.aliases is None:
            cls.aliases = cls._get_all_param_aliases()
        ret = set(args)
        for arg in args:
            for aliases in cls.aliases.values():
                if arg in aliases:
                    ret.update(aliases)
                    break
        return ret


def _choose_param_value(main_param_name: str, params: Dict[str, Any], default_value: Any) -> Dict[str, Any]:
    """Get a single parameter value, accounting for aliases.

    Parameters
    ----------
    main_param_name : str
        Name of the main parameter to get a value for. One of the keys of ``_ConfigAliases``.
    params : dict
        Dictionary of LightGBM parameters.
    default_value : Any
        Default value to use for the parameter, if none is found in ``params``.

    Returns
    -------
    params : dict
        A ``params`` dict with exactly one value for ``main_param_name``, and all aliases ``main_param_name`` removed.
        If both ``main_param_name`` and one or more aliases for it are found, the value of ``main_param_name`` will be preferred.
    """
    # avoid side effects on passed-in parameters
    params = deepcopy(params)

    aliases = _ConfigAliases.get_sorted(main_param_name)
    aliases = [a for a in aliases if a != main_param_name]

    # if main_param_name was provided, keep that value and remove all aliases
    if main_param_name in params.keys():
        for param in aliases:
            params.pop(param, None)
        return params

    # if main param name was not found, search for an alias
    for param in aliases:
        if param in params.keys():
            params[main_param_name] = params[param]
            break

    if main_param_name in params.keys():
        for param in aliases:
            params.pop(param, None)
        return params

    # neither of main_param_name, aliases were found
    params[main_param_name] = default_value

    return params


_MAX_INT32 = (1 << 31) - 1

"""Macro definition of data type in C API of LightGBM"""
_C_API_DTYPE_FLOAT32 = 0
_C_API_DTYPE_FLOAT64 = 1
_C_API_DTYPE_INT32 = 2
_C_API_DTYPE_INT64 = 3

"""Macro definition of data order in matrix"""
_C_API_IS_COL_MAJOR = 0
_C_API_IS_ROW_MAJOR = 1

"""Macro definition of prediction type in C API of LightGBM"""
_C_API_PREDICT_NORMAL = 0
_C_API_PREDICT_RAW_SCORE = 1
_C_API_PREDICT_LEAF_INDEX = 2
_C_API_PREDICT_CONTRIB = 3

"""Macro definition of sparse matrix type"""
_C_API_MATRIX_TYPE_CSR = 0
_C_API_MATRIX_TYPE_CSC = 1

"""Macro definition of feature importance type"""
_C_API_FEATURE_IMPORTANCE_SPLIT = 0
_C_API_FEATURE_IMPORTANCE_GAIN = 1

"""Data type of data field"""
_FIELD_TYPE_MAPPER = {
    "label": _C_API_DTYPE_FLOAT32,
    "weight": _C_API_DTYPE_FLOAT32,
    "init_score": _C_API_DTYPE_FLOAT64,
    "group": _C_API_DTYPE_INT32,
    "position": _C_API_DTYPE_INT32,
}

"""String name to int feature importance type mapper"""
_FEATURE_IMPORTANCE_TYPE_MAPPER = {
    "split": _C_API_FEATURE_IMPORTANCE_SPLIT,
    "gain": _C_API_FEATURE_IMPORTANCE_GAIN,
}


def _convert_from_sliced_object(data: np.ndarray) -> np.ndarray:
    """Fix the memory of multi-dimensional sliced object."""
    if isinstance(data, np.ndarray) and isinstance(data.base, np.ndarray):
        if not data.flags.c_contiguous:
            _log_warning(
                "Usage of np.ndarray subset (sliced data) is not recommended "
                "due to it will double the peak memory cost in LightGBM."
            )
            return np.copy(data)
    return data


def _c_float_array(data: np.ndarray) -> Tuple[_ctypes_float_ptr, int, np.ndarray]:
    """Get pointer of float numpy array / list."""
    if _is_1d_list(data):
        data = np.asarray(data)
    if _is_numpy_1d_array(data):
        data = _convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        ptr_data: _ctypes_float_ptr
        if data.dtype == np.float32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            type_data = _C_API_DTYPE_FLOAT32
        elif data.dtype == np.float64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            type_data = _C_API_DTYPE_FLOAT64
        else:
            raise TypeError(f"Expected np.float32 or np.float64, met type({data.dtype})")
    else:
        raise TypeError(f"Unknown type({type(data).__name__})")
    return (ptr_data, type_data, data)  # return `data` to avoid the temporary copy is freed


def _c_int_array(data: np.ndarray) -> Tuple[_ctypes_int_ptr, int, np.ndarray]:
    """Get pointer of int numpy array / list."""
    if _is_1d_list(data):
        data = np.asarray(data)
    if _is_numpy_1d_array(data):
        data = _convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        ptr_data: _ctypes_int_ptr
        if data.dtype == np.int32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            type_data = _C_API_DTYPE_INT32
        elif data.dtype == np.int64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            type_data = _C_API_DTYPE_INT64
        else:
            raise TypeError(f"Expected np.int32 or np.int64, met type({data.dtype})")
    else:
        raise TypeError(f"Unknown type({type(data).__name__})")
    return (ptr_data, type_data, data)  # return `data` to avoid the temporary copy is freed


def _is_allowed_numpy_dtype(dtype: type) -> bool:
    float128 = getattr(np, "float128", type(None))
    return issubclass(dtype, (np.integer, np.floating, np.bool_)) and not issubclass(dtype, (np.timedelta64, float128))


def _check_for_bad_pandas_dtypes(pandas_dtypes_series: pd_Series) -> None:
    bad_pandas_dtypes = [
        f"{column_name}: {pandas_dtype}"
        for column_name, pandas_dtype in pandas_dtypes_series.items()
        if not _is_allowed_numpy_dtype(pandas_dtype.type)
    ]
    if bad_pandas_dtypes:
        raise ValueError(
            f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}"
        )


def _pandas_to_numpy(
    data: pd_DataFrame,
    target_dtype: "np.typing.DTypeLike",
) -> np.ndarray:
    _check_for_bad_pandas_dtypes(data.dtypes)
    try:
        # most common case (no nullable dtypes)
        return data.to_numpy(dtype=target_dtype, copy=False)
    except TypeError:
        # 1.0 <= pd version < 1.1 and nullable dtypes, least common case
        # raises error because array is casted to type(pd.NA) and there's no na_value argument
        return data.astype(target_dtype, copy=False).values
    except ValueError:
        # data has nullable dtypes, but we can specify na_value argument and copy will be made
        return data.to_numpy(dtype=target_dtype, na_value=np.nan)


def _data_from_pandas(
    data: pd_DataFrame,
    feature_name: _LGBM_FeatureNameConfiguration,
    categorical_feature: _LGBM_CategoricalFeatureConfiguration,
    pandas_categorical: Optional[List[List]],
) -> Tuple[np.ndarray, List[str], Union[List[str], List[int]], List[List]]:
    if len(data.shape) != 2 or data.shape[0] < 1:
        raise ValueError("Input data must be 2 dimensional and non empty.")

    # take shallow copy in case we modify categorical columns
    # whole column modifications don't change the original df
    data = data.copy(deep=False)

    # determine feature names
    if feature_name == "auto":
        feature_name = [str(col) for col in data.columns]

    # determine categorical features
    cat_cols = [
        col for col, dtype in zip(data.columns, data.dtypes, strict=True) if isinstance(dtype, pd_CategoricalDtype)
    ]
    # Ordered categoricals have ordinal semantics, which means we handle them as numeric features.
    # By excluding them from ``categorical_feature``, the C++ backend uses ``value < threshold`` splits
    # on their integer codes. See docs/Advanced-Topics.rst#categorical-feature-support.
    cat_cols_not_ordered: List[str] = [col for col in cat_cols if not data[col].cat.ordered]
    if pandas_categorical is None:  # train dataset
        pandas_categorical = [list(data[col].cat.categories) for col in cat_cols]
    else:
        if len(cat_cols) != len(pandas_categorical):
            raise ValueError("train and valid dataset categorical_feature do not match.")
        for col, category in zip(cat_cols, pandas_categorical, strict=True):
            if list(data[col].cat.categories) != list(category):
                data[col] = data[col].cat.set_categories(category)
    if cat_cols:  # cat_cols is list
        data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})

    # use cat cols from DataFrame
    if categorical_feature == "auto":
        categorical_feature = cat_cols_not_ordered

    df_dtypes = [dtype.type for dtype in data.dtypes]
    # so that the target dtype considers floats
    df_dtypes.append(np.float32)
    target_dtype = np.result_type(*df_dtypes)

    return (
        _pandas_to_numpy(data, target_dtype=target_dtype),
        feature_name,
        categorical_feature,
        pandas_categorical,
    )


def _dump_pandas_categorical(
    pandas_categorical: Optional[List[List]],
    file_name: Optional[Union[str, Path]] = None,
) -> str:
    categorical_json = json.dumps(pandas_categorical)
    pandas_str = f"\npandas_categorical:{categorical_json}\n"
    if file_name is not None:
        with open(file_name, "a") as f:
            f.write(pandas_str)
    return pandas_str


def _load_pandas_categorical(
    file_name: Optional[Union[str, Path]] = None,
    model_str: Optional[str] = None,
) -> Optional[List[List]]:
    pandas_key = "pandas_categorical:"
    offset = -len(pandas_key)
    if file_name is not None:
        max_offset = -getsize(file_name)
        with open(file_name, "rb") as f:
            while True:
                offset = max(offset, max_offset)
                f.seek(offset, SEEK_END)
                lines = f.readlines()
                if len(lines) >= 2:
                    break
                offset *= 2
        last_line = lines[-1].decode("utf-8").strip()
        if not last_line.startswith(pandas_key):
            last_line = lines[-2].decode("utf-8").strip()
    elif model_str is not None:
        idx = model_str.rfind("\n", 0, offset)
        last_line = model_str[idx:].strip()
    if last_line.startswith(pandas_key):
        return json.loads(last_line[len(pandas_key) :])
    else:
        return None


class Sequence(abc.ABC):
    """
    Generic data access interface.

    Object should support the following operations:

    .. code-block::

        # Get total row number.
        >>> len(seq)
        # Random access by row index. Used for data sampling.
        >>> seq[10]
        # Range data access. Used to read data in batch when constructing Dataset.
        >>> seq[0:100]
        # Optionally specify batch_size to control range data read size.
        >>> seq.batch_size

    - With random access, **data sampling does not need to go through all data**.
    - With range data access, there's **no need to read all data into memory thus reduce memory usage**.

    .. versionadded:: 3.3.0

    Attributes
    ----------
    batch_size : int
        Default size of a batch.
    """

    batch_size = 4096  # Defaults to read 4K rows in each batch.

    @abc.abstractmethod
    def __getitem__(self, idx: Union[int, slice, List[int]]) -> np.ndarray:
        """Return data for given row index.

        A basic implementation should look like this:

        .. code-block:: python

            if isinstance(idx, numbers.Integral):
                return self._get_one_line(idx)
            elif isinstance(idx, slice):
                return np.stack([self._get_one_line(i) for i in range(idx.start, idx.stop)])
            elif isinstance(idx, list):
                # Only required if using ``Dataset.subset()``.
                return np.array([self._get_one_line(i) for i in idx])
            else:
                raise TypeError(f"Sequence index must be integer, slice or list, got {type(idx).__name__}")

        Parameters
        ----------
        idx : int, slice[int], list[int]
            Item index.

        Returns
        -------
        result : numpy 1-D array or numpy 2-D array
            1-D array if idx is int, 2-D array if idx is slice or list.
        """
        raise NotImplementedError("Sub-classes of lightgbm.Sequence must implement __getitem__()")

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return row count of this sequence."""
        raise NotImplementedError("Sub-classes of lightgbm.Sequence must implement __len__()")


class _InnerPredictor:
    """_InnerPredictor of LightGBM.

    Not exposed to user.
    Used only for prediction, usually used for continued training.

    .. note::

        Can be converted from Booster, but cannot be converted to Booster.
    """

    def __init__(
        self,
        booster_handle: _BoosterHandle,
        pandas_categorical: Optional[List[List]],
        pred_parameter: Dict[str, Any],
        manage_handle: bool,
    ):
        """Initialize the _InnerPredictor.

        Parameters
        ----------
        booster_handle : object
            Handle of Booster.
        pandas_categorical : list of list, or None
            If provided, list of categories for ``pandas`` categorical columns.
            Where the ``i``th element of the list contains the categories for the ``i``th categorical feature.
        pred_parameter : dict
            Other parameters for the prediction.
        manage_handle : bool
            If ``True``, free the corresponding Booster on the C++ side when this Python object is deleted.
        """
        self._handle = booster_handle
        self.__is_manage_handle = manage_handle
        self.pandas_categorical = pandas_categorical
        self.pred_parameter = _param_dict_to_str(pred_parameter)

        out_num_class = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterGetNumClasses(
                self._handle,
                ctypes.byref(out_num_class),
            )
        )
        self.num_class = out_num_class.value

    @classmethod
    def from_booster(
        cls,
        booster: "Booster",
        pred_parameter: Dict[str, Any],
    ) -> "_InnerPredictor":
        """Initialize an ``_InnerPredictor`` from a ``Booster``.

        Parameters
        ----------
        booster : Booster
            Booster.
        pred_parameter : dict
            Other parameters for the prediction.
        """
        return cls(
            booster_handle=booster._handle,
            pandas_categorical=booster.pandas_categorical,
            pred_parameter=pred_parameter,
            manage_handle=False,
        )

    @classmethod
    def from_model_file(
        cls,
        model_file: Union[str, Path],
        pred_parameter: Dict[str, Any],
    ) -> "_InnerPredictor":
        """Initialize an ``_InnerPredictor`` from a text file containing a LightGBM model.

        Parameters
        ----------
        model_file : str or pathlib.Path
            Path to the model file.
        pred_parameter : dict
            Other parameters for the prediction.
        """
        booster_handle = ctypes.c_void_p()
        out_num_iterations = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterCreateFromModelfile(
                _c_str(str(model_file)),
                ctypes.byref(out_num_iterations),
                ctypes.byref(booster_handle),
            )
        )
        return cls(
            booster_handle=booster_handle,
            pandas_categorical=_load_pandas_categorical(file_name=model_file),
            pred_parameter=pred_parameter,
            manage_handle=True,
        )

    def __del__(self) -> None:
        try:
            if self.__is_manage_handle:
                _safe_call(_LIB.LGBM_BoosterFree(self._handle))
        except AttributeError:
            pass

    def __getstate__(self) -> Dict[str, Any]:
        this = self.__dict__.copy()
        this.pop("handle", None)
        this.pop("_handle", None)
        return this

    def predict(
        self,
        data: _LGBM_PredictDataType,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
    ) -> _LGBM_PredictReturnType:
        """Predict logic.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, scipy.sparse, pyarrow Table or polars DataFrame
            Data source for prediction.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM).

        .. versionadded:: 4.2.0
            Support for ``pyarrow`` inputs

        .. versionadded:: 4.7.0
            Support for ``polars`` inputs

        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
        num_iteration : int, optional (default=-1)
            Iteration used for prediction.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        data_has_header : bool, optional (default=False)
            Whether data has header.
            Used only for txt data.
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.

            .. versionadded:: 4.0.0

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        if isinstance(data, Dataset):
            raise TypeError("Cannot use Dataset instance for prediction, please use raw data instead")
        if isinstance(data, pd_DataFrame) and validate_features:
            data_names = [str(x) for x in data.columns]
            ptr_names = (ctypes.c_char_p * len(data_names))()
            ptr_names[:] = [x.encode("utf-8") for x in data_names]
            _safe_call(
                _LIB.LGBM_BoosterValidateFeatureNames(
                    self._handle,
                    ptr_names,
                    ctypes.c_int(len(data_names)),
                )
            )

        if isinstance(data, pd_DataFrame):
            data = _data_from_pandas(
                data=data,
                feature_name="auto",
                categorical_feature="auto",
                pandas_categorical=self.pandas_categorical,
            )[0]

        predict_type = _C_API_PREDICT_NORMAL
        if raw_score:
            predict_type = _C_API_PREDICT_RAW_SCORE
        if pred_leaf:
            predict_type = _C_API_PREDICT_LEAF_INDEX
        if pred_contrib:
            predict_type = _C_API_PREDICT_CONTRIB

        if isinstance(data, (str, Path)):
            with _TempFile() as f:
                _safe_call(
                    _LIB.LGBM_BoosterPredictForFile(
                        self._handle,
                        _c_str(str(data)),
                        ctypes.c_int(data_has_header),
                        ctypes.c_int(predict_type),
                        ctypes.c_int(start_iteration),
                        ctypes.c_int(num_iteration),
                        _c_str(self.pred_parameter),
                        _c_str(f.name),
                    )
                )
                preds = np.loadtxt(f.name, dtype=np.float64)
                nrow = preds.shape[0]
        elif isinstance(data, scipy.sparse.csr_matrix):
            # TODO: remove 'type: ignore[assignment]' when https://github.com/lightgbm-org/LightGBM/pull/6348 is resolved.
            preds, nrow = self.__pred_for_csr(  # type: ignore[assignment]
                csr=data,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        elif isinstance(data, scipy.sparse.csc_matrix):
            # TODO: remove 'type: ignore[assignment]' when https://github.com/lightgbm-org/LightGBM/pull/6348 is resolved.
            preds, nrow = self.__pred_for_csc(  # type: ignore[assignment]
                csc=data,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        elif isinstance(data, np.ndarray):
            preds, nrow = self.__pred_for_np2d(
                mat=data,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        elif nwd.is_into_dataframe(data):
            preds, nrow = self.__pred_for_narwhals(
                data=nw.from_native(data),
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        elif isinstance(data, list):
            try:
                data = np.array(data)
            except BaseException as err:
                raise ValueError("Cannot convert data list to numpy array.") from err
            preds, nrow = self.__pred_for_np2d(
                mat=data,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        else:
            try:
                _log_warning("Converting data to scipy sparse matrix.")
                csr = scipy.sparse.csr_matrix(data)
            except BaseException as err:
                raise TypeError(f"Cannot predict data for type {type(data).__name__}") from err
            # TODO: remove 'type: ignore[assignment]' when https://github.com/lightgbm-org/LightGBM/pull/6348 is resolved.
            preds, nrow = self.__pred_for_csr(  # type: ignore[assignment]
                csr=csr,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        if pred_leaf:
            preds = preds.astype(np.int32)
        is_sparse = isinstance(preds, (list, scipy.sparse.spmatrix))
        if not is_sparse and (preds.size != nrow or pred_leaf or pred_contrib):
            if preds.size % nrow == 0:
                preds = preds.reshape(nrow, -1)
            else:
                raise ValueError(f"Length of predict result ({preds.size}) cannot be divide nrow ({nrow})")
        return preds

    def __get_num_preds(
        self,
        *,
        start_iteration: int,
        num_iteration: int,
        nrow: int,
        predict_type: int,
    ) -> int:
        """Get size of prediction result."""
        if nrow > _MAX_INT32:
            raise LightGBMError(
                "LightGBM cannot perform prediction for data "
                f"with number of rows greater than MAX_INT32 ({_MAX_INT32}).\n"
                "You can split your data into chunks "
                "and then concatenate predictions for them"
            )
        n_preds = ctypes.c_int64(0)
        _safe_call(
            _LIB.LGBM_BoosterCalcNumPredict(
                self._handle,
                ctypes.c_int(nrow),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.byref(n_preds),
            )
        )
        return n_preds.value

    def __inner_predict_np2d(
        self,
        mat: np.ndarray,
        start_iteration: int,
        num_iteration: int,
        predict_type: int,
        preds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        data, layout = _np2d_to_np1d(mat)
        ptr_data, type_ptr_data, _ = _c_float_array(data)
        n_preds = self.__get_num_preds(
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            nrow=mat.shape[0],
            predict_type=predict_type,
        )
        if preds is None:
            preds = np.empty(n_preds, dtype=np.float64)
        elif len(preds.shape) != 1 or len(preds) != n_preds:
            raise ValueError("Wrong length of pre-allocated predict array")
        out_num_preds = ctypes.c_int64(0)
        _safe_call(
            _LIB.LGBM_BoosterPredictForMat(
                self._handle,
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int32(mat.shape[0]),
                ctypes.c_int32(mat.shape[1]),
                ctypes.c_int(layout),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                _c_str(self.pred_parameter),
                ctypes.byref(out_num_preds),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, mat.shape[0]

    def __pred_for_np2d(
        self,
        mat: np.ndarray,
        start_iteration: int,
        num_iteration: int,
        predict_type: int,
    ) -> Tuple[np.ndarray, int]:
        """Predict for a 2-D numpy matrix."""
        if len(mat.shape) != 2:
            raise ValueError("Input numpy.ndarray or list must be 2 dimensional")

        nrow = mat.shape[0]
        if nrow > _MAX_INT32:
            sections = np.arange(_MAX_INT32, nrow, _MAX_INT32)
            # __get_num_preds() cannot work with nrow > MAX_INT32, so calculate overall number of predictions piecemeal
            n_preds = [
                self.__get_num_preds(
                    start_iteration=start_iteration,
                    num_iteration=num_iteration,
                    nrow=int(i),
                    predict_type=predict_type,
                )
                for i in np.diff([0] + list(sections) + [nrow])
            ]
            n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
            preds = np.empty(sum(n_preds), dtype=np.float64)
            for chunk, (start_idx_pred, end_idx_pred) in zip(
                np.array_split(mat, sections), zip(n_preds_sections, n_preds_sections[1:], strict=True), strict=True
            ):
                # avoid memory consumption by arrays concatenation operations
                self.__inner_predict_np2d(
                    mat=chunk,
                    start_iteration=start_iteration,
                    num_iteration=num_iteration,
                    predict_type=predict_type,
                    preds=preds[start_idx_pred:end_idx_pred],
                )
            return preds, nrow
        else:
            return self.__inner_predict_np2d(
                mat=mat,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
                preds=None,
            )

    def __create_sparse_native(
        self,
        cs: Union[scipy.sparse.csc_matrix, scipy.sparse.csr_matrix],
        out_shape: np.ndarray,
        out_ptr_indptr: "ctypes._Pointer",
        out_ptr_indices: "ctypes._Pointer",
        out_ptr_data: "ctypes._Pointer",
        indptr_type: int,
        data_type: int,
        is_csr: bool,
    ) -> _LGBM_PredictSparseReturnType:
        # create numpy array from output arrays
        data_indices_len = out_shape[0]
        indptr_len = out_shape[1]
        if indptr_type == _C_API_DTYPE_INT32:
            out_indptr = _cint32_array_to_numpy(cptr=out_ptr_indptr, length=indptr_len)
        elif indptr_type == _C_API_DTYPE_INT64:
            out_indptr = _cint64_array_to_numpy(cptr=out_ptr_indptr, length=indptr_len)
        else:
            raise TypeError("Expected int32 or int64 type for indptr")
        if data_type == _C_API_DTYPE_FLOAT32:
            out_data = _cfloat32_array_to_numpy(cptr=out_ptr_data, length=data_indices_len)
        elif data_type == _C_API_DTYPE_FLOAT64:
            out_data = _cfloat64_array_to_numpy(cptr=out_ptr_data, length=data_indices_len)
        else:
            raise TypeError("Expected float32 or float64 type for data")
        out_indices = _cint32_array_to_numpy(cptr=out_ptr_indices, length=data_indices_len)
        # break up indptr based on number of rows (note more than one matrix in multiclass case)
        per_class_indptr_shape = cs.indptr.shape[0]
        # for CSC there is extra column added
        if not is_csr:
            per_class_indptr_shape += 1
        out_indptr_arrays = np.split(out_indptr, out_indptr.shape[0] / per_class_indptr_shape)
        # reformat output into a csr or csc matrix or list of csr or csc matrices
        cs_output_matrices = []
        offset = 0
        for cs_indptr in out_indptr_arrays:
            matrix_indptr_len = cs_indptr[cs_indptr.shape[0] - 1]
            cs_indices = out_indices[offset + cs_indptr[0] : offset + matrix_indptr_len]
            cs_data = out_data[offset + cs_indptr[0] : offset + matrix_indptr_len]
            offset += matrix_indptr_len
            # same shape as input csr or csc matrix except extra column for expected value
            cs_shape = [cs.shape[0], cs.shape[1] + 1]
            # note: make sure we copy data as it will be deallocated next
            if is_csr:
                cs_output_matrices.append(scipy.sparse.csr_matrix((cs_data, cs_indices, cs_indptr), cs_shape))
            else:
                cs_output_matrices.append(scipy.sparse.csc_matrix((cs_data, cs_indices, cs_indptr), cs_shape))
        # free the temporary native indptr, indices, and data
        _safe_call(
            _LIB.LGBM_BoosterFreePredictSparse(
                out_ptr_indptr,
                out_ptr_indices,
                out_ptr_data,
                ctypes.c_int(indptr_type),
                ctypes.c_int(data_type),
            )
        )
        if len(cs_output_matrices) == 1:
            return cs_output_matrices[0]
        return cs_output_matrices

    def __inner_predict_csr(
        self,
        csr: scipy.sparse.csr_matrix,
        start_iteration: int,
        num_iteration: int,
        predict_type: int,
        preds: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        nrow = len(csr.indptr) - 1
        n_preds = self.__get_num_preds(
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            nrow=nrow,
            predict_type=predict_type,
        )
        if preds is None:
            preds = np.empty(n_preds, dtype=np.float64)
        elif len(preds.shape) != 1 or len(preds) != n_preds:
            raise ValueError("Wrong length of pre-allocated predict array")
        out_num_preds = ctypes.c_int64(0)

        ptr_indptr, type_ptr_indptr, _ = _c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csr.data)

        assert csr.shape[1] <= _MAX_INT32
        csr_indices = csr.indices.astype(np.int32, copy=False)

        _safe_call(
            _LIB.LGBM_BoosterPredictForCSR(
                self._handle,
                ptr_indptr,
                ctypes.c_int(type_ptr_indptr),
                csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int64(len(csr.indptr)),
                ctypes.c_int64(len(csr.data)),
                ctypes.c_int64(csr.shape[1]),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                _c_str(self.pred_parameter),
                ctypes.byref(out_num_preds),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, nrow

    def __inner_predict_csr_sparse(
        self,
        csr: scipy.sparse.csr_matrix,
        start_iteration: int,
        num_iteration: int,
        predict_type: int,
    ) -> Tuple[_LGBM_PredictSparseReturnType, int]:
        ptr_indptr, type_ptr_indptr, __ = _c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csr.data)
        csr_indices = csr.indices.astype(np.int32, copy=False)
        matrix_type = _C_API_MATRIX_TYPE_CSR
        out_ptr_indptr: _ctypes_int_ptr
        if type_ptr_indptr == _C_API_DTYPE_INT32:
            out_ptr_indptr = ctypes.POINTER(ctypes.c_int32)()
        else:
            out_ptr_indptr = ctypes.POINTER(ctypes.c_int64)()
        out_ptr_indices = ctypes.POINTER(ctypes.c_int32)()
        out_ptr_data: _ctypes_float_ptr
        if type_ptr_data == _C_API_DTYPE_FLOAT32:
            out_ptr_data = ctypes.POINTER(ctypes.c_float)()
        else:
            out_ptr_data = ctypes.POINTER(ctypes.c_double)()
        out_shape = np.empty(2, dtype=np.int64)
        _safe_call(
            _LIB.LGBM_BoosterPredictSparseOutput(
                self._handle,
                ptr_indptr,
                ctypes.c_int(type_ptr_indptr),
                csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int64(len(csr.indptr)),
                ctypes.c_int64(len(csr.data)),
                ctypes.c_int64(csr.shape[1]),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                _c_str(self.pred_parameter),
                ctypes.c_int(matrix_type),
                out_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.byref(out_ptr_indptr),
                ctypes.byref(out_ptr_indices),
                ctypes.byref(out_ptr_data),
            )
        )
        matrices = self.__create_sparse_native(
            cs=csr,
            out_shape=out_shape,
            out_ptr_indptr=out_ptr_indptr,
            out_ptr_indices=out_ptr_indices,
            out_ptr_data=out_ptr_data,
            indptr_type=type_ptr_indptr,
            data_type=type_ptr_data,
            is_csr=True,
        )
        nrow = len(csr.indptr) - 1
        return matrices, nrow

    def __pred_for_csr(
        self,
        csr: scipy.sparse.csr_matrix,
        start_iteration: int,
        num_iteration: int,
        predict_type: int,
    ) -> Tuple[_LGBM_PredictSparseReturnType, int]:
        """Predict for a CSR data."""
        if predict_type == _C_API_PREDICT_CONTRIB:
            return self.__inner_predict_csr_sparse(
                csr=csr,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        nrow = len(csr.indptr) - 1
        if nrow > _MAX_INT32:
            sections = [0] + list(np.arange(_MAX_INT32, nrow, _MAX_INT32)) + [nrow]
            # __get_num_preds() cannot work with nrow > MAX_INT32, so calculate overall number of predictions piecemeal
            n_preds = [
                self.__get_num_preds(
                    start_iteration=start_iteration,
                    num_iteration=num_iteration,
                    nrow=int(i),
                    predict_type=predict_type,
                )
                for i in np.diff(sections)
            ]
            n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
            preds = np.empty(sum(n_preds), dtype=np.float64)
            for (start_idx, end_idx), (start_idx_pred, end_idx_pred) in zip(
                zip(sections, sections[1:], strict=True),
                zip(n_preds_sections, n_preds_sections[1:], strict=True),
                strict=True,
            ):
                # avoid memory consumption by arrays concatenation operations
                self.__inner_predict_csr(
                    csr=csr[start_idx:end_idx],
                    start_iteration=start_iteration,
                    num_iteration=num_iteration,
                    predict_type=predict_type,
                    preds=preds[start_idx_pred:end_idx_pred],
                )
            return preds, nrow
        else:
            return self.__inner_predict_csr(
                csr=csr,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
                preds=None,
            )

    def __inner_predict_sparse_csc(
        self,
        csc: scipy.sparse.csc_matrix,
        start_iteration: int,
        num_iteration: int,
        predict_type: int,
    ) -> Tuple[_LGBM_PredictSparseReturnType, int]:
        ptr_indptr, type_ptr_indptr, __ = _c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csc.data)
        csc_indices = csc.indices.astype(np.int32, copy=False)
        matrix_type = _C_API_MATRIX_TYPE_CSC
        out_ptr_indptr: _ctypes_int_ptr
        if type_ptr_indptr == _C_API_DTYPE_INT32:
            out_ptr_indptr = ctypes.POINTER(ctypes.c_int32)()
        else:
            out_ptr_indptr = ctypes.POINTER(ctypes.c_int64)()
        out_ptr_indices = ctypes.POINTER(ctypes.c_int32)()
        out_ptr_data: _ctypes_float_ptr
        if type_ptr_data == _C_API_DTYPE_FLOAT32:
            out_ptr_data = ctypes.POINTER(ctypes.c_float)()
        else:
            out_ptr_data = ctypes.POINTER(ctypes.c_double)()
        out_shape = np.empty(2, dtype=np.int64)
        _safe_call(
            _LIB.LGBM_BoosterPredictSparseOutput(
                self._handle,
                ptr_indptr,
                ctypes.c_int(type_ptr_indptr),
                csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int64(len(csc.indptr)),
                ctypes.c_int64(len(csc.data)),
                ctypes.c_int64(csc.shape[0]),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                _c_str(self.pred_parameter),
                ctypes.c_int(matrix_type),
                out_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.byref(out_ptr_indptr),
                ctypes.byref(out_ptr_indices),
                ctypes.byref(out_ptr_data),
            )
        )
        matrices = self.__create_sparse_native(
            cs=csc,
            out_shape=out_shape,
            out_ptr_indptr=out_ptr_indptr,
            out_ptr_indices=out_ptr_indices,
            out_ptr_data=out_ptr_data,
            indptr_type=type_ptr_indptr,
            data_type=type_ptr_data,
            is_csr=False,
        )
        nrow = csc.shape[0]
        return matrices, nrow

    def __pred_for_csc(
        self,
        csc: scipy.sparse.csc_matrix,
        start_iteration: int,
        num_iteration: int,
        predict_type: int,
    ) -> Tuple[_LGBM_PredictSparseReturnType, int]:
        """Predict for a CSC data."""
        nrow = csc.shape[0]
        if nrow > _MAX_INT32:
            return self.__pred_for_csr(
                csr=csc.tocsr(),
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        if predict_type == _C_API_PREDICT_CONTRIB:
            return self.__inner_predict_sparse_csc(
                csc=csc,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                predict_type=predict_type,
            )
        n_preds = self.__get_num_preds(
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            nrow=nrow,
            predict_type=predict_type,
        )
        preds = np.empty(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)

        ptr_indptr, type_ptr_indptr, __ = _c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csc.data)

        assert csc.shape[0] <= _MAX_INT32
        csc_indices = csc.indices.astype(np.int32, copy=False)

        _safe_call(
            _LIB.LGBM_BoosterPredictForCSC(
                self._handle,
                ptr_indptr,
                ctypes.c_int(type_ptr_indptr),
                csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int64(len(csc.indptr)),
                ctypes.c_int64(len(csc.data)),
                ctypes.c_int64(csc.shape[0]),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                _c_str(self.pred_parameter),
                ctypes.byref(out_num_preds),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, nrow

    def __pred_for_narwhals(
        self,
        data: nw.DataFrame,
        start_iteration: int,
        num_iteration: int,
        predict_type: int,
    ) -> Tuple[np.ndarray, int]:
        """Predict for a narwhals DataFrame."""
        # Prepare prediction output array
        n_preds = self.__get_num_preds(
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            nrow=data.shape[0],
            predict_type=predict_type,
        )
        preds = np.empty(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)

        # Export narwhals DataFrame to Arrow and run prediction
        pycapsule = data.__arrow_c_stream__()
        _safe_call(
            _LIB.LGBM_BoosterPredictForArrowStream(
                self._handle,
                _extract_arrow_stream_capsule_pointer(pycapsule),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                _c_str(self.pred_parameter),
                ctypes.byref(out_num_preds),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, data.shape[0]

    def current_iteration(self) -> int:
        """Get the index of the current iteration.

        Returns
        -------
        cur_iter : int
            The index of the current iteration.
        """
        out_cur_iter = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterGetCurrentIteration(
                self._handle,
                ctypes.byref(out_cur_iter),
            )
        )
        return out_cur_iter.value


class Dataset:
    """
    Dataset in LightGBM.

    LightGBM does not train on raw data.
    It discretizes continuous features into histogram bins, tries to combine categorical features,
    and automatically handles missing and infinite values.

    This class handles that preprocessing, and holds that alternative representation of the input data.
    """

    def __init__(
        self,
        data: _LGBM_TrainDataType,
        label: Optional[_LGBM_LabelType] = None,
        reference: Optional["Dataset"] = None,
        weight: Optional[_LGBM_WeightType] = None,
        group: Optional[_LGBM_GroupType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        feature_name: _LGBM_FeatureNameConfiguration = "auto",
        categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
        params: Optional[Dict[str, Any]] = None,
        free_raw_data: bool = True,
        position: Optional[_LGBM_PositionType] = None,
    ):
        """Initialize Dataset.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, scipy.sparse, Sequence, list of Sequence, list of numpy array, pyarrow Table or polars DataFrame
            Data source of Dataset.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM) or a LightGBM Dataset binary file.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        label : list, numpy 1-D array, pandas Series / one-column DataFrame, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Label of the data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        reference : Dataset or None, optional (default=None)
            If this is Dataset for validation, training data should be used as reference.
        weight : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Weight for each instance. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        group : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow ChunkedArray, pyarrow Table (for multi-class task), polars Series, polars DataFrame (for multi-class task) or None, optional (default=None)
            Init score for Dataset.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        feature_name : list of str, or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame, pyarrow Table, or polars DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        params : dict or None, optional (default=None)
            Other parameters for Dataset.
        free_raw_data : bool, optional (default=True)
            If True, raw data is freed after constructing inner Dataset.
        position : numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Position of items used in unbiased learning-to-rank task.
        """
        self._handle: Optional[_DatasetHandle] = None
        self.data = data
        self.label = label
        self.reference = reference
        self.weight = weight
        self.group = group
        self.position = position
        self.init_score = init_score
        self.feature_name: _LGBM_FeatureNameConfiguration = feature_name
        self.categorical_feature: _LGBM_CategoricalFeatureConfiguration = categorical_feature
        self.params = deepcopy(params)
        self.free_raw_data = free_raw_data
        self.used_indices: Optional[List[int]] = None
        self._need_slice = True
        self._predictor: Optional[_InnerPredictor] = None
        self.pandas_categorical: Optional[List[List]] = None
        self._params_back_up: Optional[Dict[str, Any]] = None
        self.version = 0
        self._start_row = 0  # Used when pushing rows one by one.
        # By default, LightGBM assigns features names like Column_0, Column_1, etc.
        # These may be overridden during construction if that raw training data is in a format
        # that includes feature names (like a pandas DataFrame) or if the 'feature_name' keyword argument
        # is explicitly set.
        #
        # This is here mostly for scikit-learn's benefit, as it tracks whether input data had feature names.
        self._has_non_default_feature_names: bool = False

    def __del__(self) -> None:
        try:
            self._free_handle()
        except AttributeError:
            pass

    def _create_sample_indices(self, *, total_nrow: int) -> np.ndarray:
        """Get an array of randomly chosen indices from this ``Dataset``.

        Indices are sampled without replacement.

        Parameters
        ----------
        total_nrow : int
            Total number of rows to sample from.
            If this value is greater than the value of parameter ``bin_construct_sample_cnt``, only ``bin_construct_sample_cnt`` indices will be used.
            If Dataset has multiple input data, this should be the sum of rows of every file.

        Returns
        -------
        indices : numpy array
            Indices for sampled data.
        """
        param_str = _param_dict_to_str(self.get_params())
        sample_cnt = _get_sample_count(total_nrow, param_str)
        indices = np.empty(sample_cnt, dtype=np.int32)
        ptr_data, _, _ = _c_int_array(indices)
        actual_sample_cnt = ctypes.c_int32(0)

        _safe_call(
            _LIB.LGBM_SampleIndices(
                ctypes.c_int32(total_nrow),
                _c_str(param_str),
                ptr_data,
                ctypes.byref(actual_sample_cnt),
            )
        )
        assert sample_cnt == actual_sample_cnt.value
        return indices

    def _init_from_ref_dataset(
        self,
        total_nrow: int,
        ref_dataset: _DatasetHandle,
    ) -> "Dataset":
        """Create dataset from a reference dataset.

        Parameters
        ----------
        total_nrow : int
            Number of rows expected to add to dataset.
        ref_dataset : object
            Handle of reference dataset to extract metadata from.

        Returns
        -------
        self : Dataset
            Constructed Dataset object.
        """
        self._handle = ctypes.c_void_p()
        _safe_call(
            _LIB.LGBM_DatasetCreateByReference(
                ref_dataset,
                ctypes.c_int64(total_nrow),
                ctypes.byref(self._handle),
            )
        )
        return self

    def _init_from_sample(
        self,
        sample_data: List[np.ndarray],
        sample_indices: List[np.ndarray],
        sample_cnt: int,
        total_nrow: int,
    ) -> "Dataset":
        """Create Dataset from sampled data structures.

        Parameters
        ----------
        sample_data : list of numpy array
            Sample data for each column.
        sample_indices : list of numpy array
            Sample data row index for each column.
        sample_cnt : int
            Number of samples.
        total_nrow : int
            Total number of rows for all input files.

        Returns
        -------
        self : Dataset
            Constructed Dataset object.
        """
        ncol = len(sample_indices)
        assert len(sample_data) == ncol, "#sample data column != #column indices"

        for i in range(ncol):
            if sample_data[i].dtype != np.double:
                raise ValueError(f"sample_data[{i}] type {sample_data[i].dtype} is not double")
            if sample_indices[i].dtype != np.int32:
                raise ValueError(f"sample_indices[{i}] type {sample_indices[i].dtype} is not int32")

        # c type: double**
        # each double* element points to start of each column of sample data.
        sample_col_ptr: _ctypes_float_array = (ctypes.POINTER(ctypes.c_double) * ncol)()
        # c type int**
        # each int* points to start of indices for each column
        indices_col_ptr: _ctypes_int_array = (ctypes.POINTER(ctypes.c_int32) * ncol)()
        for i in range(ncol):
            sample_col_ptr[i] = _c_float_array(sample_data[i])[0]
            indices_col_ptr[i] = _c_int_array(sample_indices[i])[0]

        num_per_col = np.array([len(d) for d in sample_indices], dtype=np.int32)
        num_per_col_ptr, _, _ = _c_int_array(num_per_col)

        self._handle = ctypes.c_void_p()
        params_str = _param_dict_to_str(self.get_params())
        _safe_call(
            _LIB.LGBM_DatasetCreateFromSampledColumn(
                ctypes.cast(sample_col_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ctypes.cast(indices_col_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
                ctypes.c_int32(ncol),
                num_per_col_ptr,
                ctypes.c_int32(sample_cnt),
                ctypes.c_int32(total_nrow),
                ctypes.c_int64(total_nrow),
                _c_str(params_str),
                ctypes.byref(self._handle),
            )
        )
        return self

    def _push_rows(self, data: np.ndarray) -> "Dataset":
        """Add rows to Dataset.

        Parameters
        ----------
        data : numpy 1-D array
            New data to add to the Dataset.

        Returns
        -------
        self : Dataset
            Dataset object.
        """
        nrow, ncol = data.shape
        data = data.reshape(data.size)
        data_ptr, data_type, _ = _c_float_array(data)

        _safe_call(
            _LIB.LGBM_DatasetPushRows(
                self._handle,
                data_ptr,
                data_type,
                ctypes.c_int32(nrow),
                ctypes.c_int32(ncol),
                ctypes.c_int32(self._start_row),
            )
        )
        self._start_row += nrow
        return self

    def get_params(self) -> Dict[str, Any]:
        """Get the used parameters in the Dataset.

        Returns
        -------
        params : dict
            The used parameters in this Dataset object.
        """
        if self.params is not None:
            # no min_data, nthreads and verbose in this function
            dataset_params = _ConfigAliases.get(
                "bin_construct_sample_cnt",
                "categorical_feature",
                "data_random_seed",
                "enable_bundle",
                "feature_pre_filter",
                "forcedbins_filename",
                "group_column",
                "header",
                "ignore_column",
                "is_enable_sparse",
                "label_column",
                "linear_tree",
                "max_bin",
                "max_bin_by_feature",
                "min_data_in_bin",
                "pre_partition",
                "precise_float_parser",
                "two_round",
                "use_missing",
                "weight_column",
                "zero_as_missing",
            )
            return {k: v for k, v in self.params.items() if k in dataset_params}
        else:
            return {}

    def _free_handle(self) -> "Dataset":
        if self._handle is not None:
            _safe_call(_LIB.LGBM_DatasetFree(self._handle))
            self._handle = None
        self._need_slice = True
        if self.used_indices is not None:
            self.data = None
        return self

    def _set_init_score_by_predictor(
        self,
        predictor: Optional[_InnerPredictor],
        data: _LGBM_TrainDataType,
        used_indices: Optional[Union[List[int], np.ndarray]],
    ) -> "Dataset":
        data_has_header = False
        if isinstance(data, (str, Path)) and self.params is not None:
            # check data has header or not
            data_has_header = any(self.params.get(alias, False) for alias in _ConfigAliases.get("header"))
        num_data = self.num_data()
        if predictor is not None:
            init_score: Union[np.ndarray, scipy.sparse.spmatrix] = predictor.predict(
                data=data,
                raw_score=True,
                data_has_header=data_has_header,
            )
            init_score = init_score.ravel()
            if used_indices is not None:
                assert not self._need_slice
                if isinstance(data, (str, Path)):
                    sub_init_score = np.empty(num_data * predictor.num_class, dtype=np.float64)
                    assert num_data == len(used_indices)
                    for i in range(len(used_indices)):
                        for j in range(predictor.num_class):
                            sub_init_score[i * predictor.num_class + j] = init_score[
                                used_indices[i] * predictor.num_class + j
                            ]
                    init_score = sub_init_score
            if predictor.num_class > 1:
                # need to regroup init_score
                new_init_score = np.empty(init_score.size, dtype=np.float64)
                for i in range(num_data):
                    for j in range(predictor.num_class):
                        new_init_score[j * num_data + i] = init_score[i * predictor.num_class + j]
                init_score = new_init_score
        elif self.init_score is not None:
            init_score = np.full_like(self.init_score, fill_value=0.0, dtype=np.float64)
        else:
            return self
        self.set_init_score(init_score)
        return self

    def _lazy_init(
        self,
        data: Optional[_LGBM_TrainDataType],
        label: Optional[_LGBM_LabelType],
        reference: Optional["Dataset"],
        weight: Optional[_LGBM_WeightType],
        group: Optional[_LGBM_GroupType],
        init_score: Optional[_LGBM_InitScoreType],
        predictor: Optional[_InnerPredictor],
        feature_name: _LGBM_FeatureNameConfiguration,
        categorical_feature: _LGBM_CategoricalFeatureConfiguration,
        params: Optional[Dict[str, Any]],
        position: Optional[_LGBM_PositionType],
    ) -> "Dataset":
        if data is None:
            self._handle = None
            return self
        if reference is not None:
            self.pandas_categorical = reference.pandas_categorical
            categorical_feature = reference.categorical_feature
        if isinstance(data, pd_DataFrame):
            data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(
                data=data,
                feature_name=feature_name,
                categorical_feature=categorical_feature,
                pandas_categorical=self.pandas_categorical,
            )
        if nwd.is_into_dataframe(data) and feature_name == "auto":
            feature_name = nw.from_native(data).schema.names()

        # 'feature_name == "auto"' after the block above means no feature names were provided
        # by either the data type (DataFrame/pyarrow) or the user's 'feature_name' argument.
        # LightGBM will assign auto-generated names like Column_0, Column_1, etc.
        self._has_non_default_feature_names = feature_name != "auto"

        # process for args
        params = {} if params is None else params
        args_names = inspect.signature(self.__class__._lazy_init).parameters.keys()
        for key in params.keys():
            if key in args_names:
                _log_warning(
                    f"{key} keyword has been found in `params` and will be ignored.\n"
                    f"Please use {key} argument of the Dataset constructor to pass this parameter."
                )
        # get categorical features
        if isinstance(categorical_feature, list):
            categorical_indices = set()
            feature_dict = {}
            if isinstance(feature_name, list):
                feature_dict = {name: i for i, name in enumerate(feature_name)}
            for name in categorical_feature:
                if isinstance(name, str) and name in feature_dict:
                    categorical_indices.add(feature_dict[name])
                elif isinstance(name, int):
                    categorical_indices.add(name)
                else:
                    raise TypeError(f"Wrong type({type(name).__name__}) or unknown name({name}) in categorical_feature")
            if categorical_indices:
                for cat_alias in _ConfigAliases.get("categorical_feature"):
                    if cat_alias in params:
                        # If the params[cat_alias] is equal to categorical_indices, do not report the warning.
                        if not (isinstance(params[cat_alias], list) and set(params[cat_alias]) == categorical_indices):
                            _log_warning(f"{cat_alias} in param dict is overridden.")
                        params.pop(cat_alias, None)
                params["categorical_column"] = sorted(categorical_indices)

        params_str = _param_dict_to_str(params)
        self.params = params
        # process for reference dataset
        ref_dataset = None
        if isinstance(reference, Dataset):
            ref_dataset = reference.construct()._handle
        elif reference is not None:
            raise TypeError("Reference dataset should be None or dataset instance")
        # start construct data
        if isinstance(data, (str, Path)):
            self._handle = ctypes.c_void_p()
            _safe_call(
                _LIB.LGBM_DatasetCreateFromFile(
                    _c_str(str(data)),
                    _c_str(params_str),
                    ref_dataset,
                    ctypes.byref(self._handle),
                )
            )
        elif isinstance(data, scipy.sparse.csr_matrix):
            self.__init_from_csr(csr=data, params_str=params_str, ref_dataset=ref_dataset)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self.__init_from_csc(csc=data, params_str=params_str, ref_dataset=ref_dataset)
        elif isinstance(data, np.ndarray):
            self.__init_from_np2d(mat=data, params_str=params_str, ref_dataset=ref_dataset)
        elif nwd.is_into_dataframe(data):
            self.__init_from_narwhals(data=nw.from_native(data), params_str=params_str, ref_dataset=ref_dataset)
        elif isinstance(data, list) and len(data) > 0:
            if _is_list_of_numpy_arrays(data):
                self.__init_from_list_np2d(mats=data, params_str=params_str, ref_dataset=ref_dataset)
            elif _is_list_of_sequences(data):
                self.__init_from_seqs(seqs=data, ref_dataset=ref_dataset)
            else:
                raise TypeError("Data list can only be of ndarray or Sequence")
        elif isinstance(data, Sequence):
            self.__init_from_seqs(seqs=[data], ref_dataset=ref_dataset)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self.__init_from_csr(csr=csr, params_str=params_str, ref_dataset=ref_dataset)
            except BaseException as err:
                raise TypeError(f"Cannot initialize Dataset from {type(data).__name__}") from err
        if label is not None:
            self.set_label(label)
        if self.get_label() is None:
            raise ValueError("Label should not be None")
        if weight is not None:
            self.set_weight(weight)
        if group is not None:
            self.set_group(group)
        if position is not None:
            self.set_position(position)
        if isinstance(predictor, _InnerPredictor):
            if self._predictor is None and init_score is not None:
                _log_warning("The init_score will be overridden by the prediction of init_model.")
            self._set_init_score_by_predictor(predictor=predictor, data=data, used_indices=None)
        elif init_score is not None:
            self.set_init_score(init_score)
        elif predictor is not None:
            raise TypeError(f"Wrong predictor type {type(predictor).__name__}")
        # set feature names
        return self.set_feature_name(feature_name)

    @staticmethod
    def _yield_row_from_seqlist(seqs: List[Sequence], indices: Iterable[int]) -> Iterator[np.ndarray]:
        offset = 0
        seq_id = 0
        seq = seqs[seq_id]
        for row_id in indices:
            assert row_id >= offset, "sample indices are expected to be monotonic"
            while row_id >= offset + len(seq):
                offset += len(seq)
                seq_id += 1
                seq = seqs[seq_id]
            id_in_seq = row_id - offset
            row = seq[id_in_seq]
            yield row if row.flags["OWNDATA"] else row.copy()

    def __sample(self, *, seqs: List[Sequence], total_nrow: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Sample data from seqs.

        Mimics behavior in c_api.cpp:LGBM_DatasetCreateFromMats()

        Returns
        -------
            sampled_rows, sampled_row_indices
        """
        indices = self._create_sample_indices(total_nrow=total_nrow)

        # Select sampled rows, transpose to column order.
        sampled = np.array(list(self._yield_row_from_seqlist(seqs, indices)))
        sampled = sampled.T

        filtered = []
        filtered_idx = []
        sampled_row_range = np.arange(len(indices), dtype=np.int32)
        for col in sampled:
            col_predicate = (np.abs(col) > ZERO_THRESHOLD) | np.isnan(col)
            filtered_col = col[col_predicate]
            filtered_row_idx = sampled_row_range[col_predicate]

            filtered.append(filtered_col)
            filtered_idx.append(filtered_row_idx)

        return filtered, filtered_idx

    def __init_from_seqs(
        self,
        *,
        seqs: List[Sequence],
        ref_dataset: Optional[_DatasetHandle],
    ) -> "Dataset":
        """
        Initialize data from list of Sequence objects.

        Sequence: Generic Data Access Object
            Supports random access and access by batch if properly defined by user

        Data scheme uniformity are trusted, not checked
        """
        total_nrow = sum(len(seq) for seq in seqs)

        # create validation dataset from ref_dataset
        if ref_dataset is not None:
            self._init_from_ref_dataset(total_nrow, ref_dataset)
        else:
            param_str = _param_dict_to_str(self.get_params())
            sample_cnt = _get_sample_count(total_nrow, param_str)

            sample_data, col_indices = self.__sample(seqs=seqs, total_nrow=total_nrow)
            self._init_from_sample(sample_data, col_indices, sample_cnt, total_nrow)

        for seq in seqs:
            nrow = len(seq)
            batch_size = getattr(seq, "batch_size", None) or Sequence.batch_size
            for start in range(0, nrow, batch_size):
                end = min(start + batch_size, nrow)
                self._push_rows(seq[start:end])
        return self

    def __init_from_np2d(
        self,
        *,
        mat: np.ndarray,
        params_str: str,
        ref_dataset: Optional[_DatasetHandle],
    ) -> "Dataset":
        """Initialize data from a 2-D numpy matrix."""
        if len(mat.shape) != 2:
            raise ValueError("Input numpy.ndarray must be 2 dimensional")

        self._handle = ctypes.c_void_p()
        data, layout = _np2d_to_np1d(mat)
        ptr_data, type_ptr_data, _ = _c_float_array(data)
        _safe_call(
            _LIB.LGBM_DatasetCreateFromMat(
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int32(mat.shape[0]),
                ctypes.c_int32(mat.shape[1]),
                ctypes.c_int(layout),
                _c_str(params_str),
                ref_dataset,
                ctypes.byref(self._handle),
            )
        )
        return self

    def __init_from_list_np2d(
        self,
        *,
        mats: List[np.ndarray],
        params_str: str,
        ref_dataset: Optional[_DatasetHandle],
    ) -> "Dataset":
        """Initialize data from a list of 2-D numpy matrices."""
        ncol = mats[0].shape[1]
        nrow = np.empty((len(mats),), np.int32)
        ptr_data: _ctypes_float_array
        if mats[0].dtype == np.float64:
            ptr_data = (ctypes.POINTER(ctypes.c_double) * len(mats))()
        else:
            ptr_data = (ctypes.POINTER(ctypes.c_float) * len(mats))()
        layouts = (ctypes.c_int * len(mats))()

        holders = []
        type_ptr_data = -1

        for i, mat in enumerate(mats):
            if len(mat.shape) != 2:
                raise ValueError("Input numpy.ndarray must be 2 dimensional")

            if mat.shape[1] != ncol:
                raise ValueError("Input arrays must have same number of columns")

            nrow[i] = mat.shape[0]

            mat, layout = _np2d_to_np1d(mat)

            chunk_ptr_data, chunk_type_ptr_data, holder = _c_float_array(mat)
            if type_ptr_data != -1 and chunk_type_ptr_data != type_ptr_data:
                raise ValueError("Input chunks must have same type")
            ptr_data[i] = chunk_ptr_data
            layouts[i] = layout
            type_ptr_data = chunk_type_ptr_data
            holders.append(holder)

        self._handle = ctypes.c_void_p()
        _safe_call(
            _LIB.LGBM_DatasetCreateFromMats(
                ctypes.c_int32(len(mats)),
                ctypes.cast(ptr_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ctypes.c_int(type_ptr_data),
                nrow.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int32(ncol),
                layouts,
                _c_str(params_str),
                ref_dataset,
                ctypes.byref(self._handle),
            )
        )
        return self

    def __init_from_csr(
        self,
        *,
        csr: scipy.sparse.csr_matrix,
        params_str: str,
        ref_dataset: Optional[_DatasetHandle],
    ) -> "Dataset":
        """Initialize data from a CSR matrix."""
        if len(csr.indices) != len(csr.data):
            raise ValueError(f"Length mismatch: {len(csr.indices)} vs {len(csr.data)}")
        self._handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr, __ = _c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csr.data)

        assert csr.shape[1] <= _MAX_INT32
        csr_indices = csr.indices.astype(np.int32, copy=False)

        _safe_call(
            _LIB.LGBM_DatasetCreateFromCSR(
                ptr_indptr,
                ctypes.c_int(type_ptr_indptr),
                csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int64(len(csr.indptr)),
                ctypes.c_int64(len(csr.data)),
                ctypes.c_int64(csr.shape[1]),
                _c_str(params_str),
                ref_dataset,
                ctypes.byref(self._handle),
            )
        )
        return self

    def __init_from_csc(
        self,
        *,
        csc: scipy.sparse.csc_matrix,
        params_str: str,
        ref_dataset: Optional[_DatasetHandle],
    ) -> "Dataset":
        """Initialize data from a CSC matrix."""
        if len(csc.indices) != len(csc.data):
            raise ValueError(f"Length mismatch: {len(csc.indices)} vs {len(csc.data)}")
        self._handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr, __ = _c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csc.data)

        assert csc.shape[0] <= _MAX_INT32
        csc_indices = csc.indices.astype(np.int32, copy=False)

        _safe_call(
            _LIB.LGBM_DatasetCreateFromCSC(
                ptr_indptr,
                ctypes.c_int(type_ptr_indptr),
                csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int64(len(csc.indptr)),
                ctypes.c_int64(len(csc.data)),
                ctypes.c_int64(csc.shape[0]),
                _c_str(params_str),
                ref_dataset,
                ctypes.byref(self._handle),
            )
        )
        return self

    def __init_from_narwhals(
        self,
        *,
        data: nw.DataFrame,
        params_str: str,
        ref_dataset: Optional[_DatasetHandle],
    ) -> "Dataset":
        """Initialize data from a narwhals DataFrame."""
        # Export narwhals DataFrame to Arrow
        pycapsule = data.__arrow_c_stream__()

        # Create dataset
        self._handle = ctypes.c_void_p()
        _safe_call(
            _LIB.LGBM_DatasetCreateFromArrowStream(
                _extract_arrow_stream_capsule_pointer(pycapsule),
                _c_str(params_str),
                ref_dataset,
                ctypes.byref(self._handle),
            )
        )
        return self

    @staticmethod
    def _compare_params_for_warning(
        *,
        params: Dict[str, Any],
        other_params: Dict[str, Any],
        ignore_keys: Set[str],
    ) -> bool:
        """Compare two dictionaries with params ignoring some keys.

        It is only for the warning purpose.

        Parameters
        ----------
        params : dict
            One dictionary with parameters to compare.
        other_params : dict
            Another dictionary with parameters to compare.
        ignore_keys : set
            Keys that should be ignored during comparing two dictionaries.

        Returns
        -------
        compare_result : bool
          Returns whether two dictionaries with params are equal.
        """
        for k, v in other_params.items():
            if k not in ignore_keys:
                if k not in params or params[k] != v:
                    return False
        for k, v in params.items():
            if k not in ignore_keys:
                if k not in other_params or v != other_params[k]:
                    return False
        return True

    def construct(self) -> "Dataset":
        """Lazy init.

        Returns
        -------
        self : Dataset
            Constructed Dataset object.
        """
        if self._handle is None:
            if self.reference is not None:
                reference_params = self.reference.get_params()
                params = self.get_params()
                if params != reference_params:
                    if not self._compare_params_for_warning(
                        params=params,
                        other_params=reference_params,
                        ignore_keys=_ConfigAliases.get("categorical_feature"),
                    ):
                        _log_warning("Overriding the parameters from Reference Dataset.")
                    self._update_params(reference_params)
                if self.used_indices is None:
                    # create valid
                    self._lazy_init(
                        data=self.data,
                        label=self.label,
                        reference=self.reference,
                        weight=self.weight,
                        group=self.group,
                        position=self.position,
                        init_score=self.init_score,
                        predictor=self._predictor,
                        feature_name=self.feature_name,
                        categorical_feature="auto",
                        params=self.params,
                    )
                else:
                    # construct subset
                    used_indices = _list_to_1d_numpy(
                        data=self.used_indices,
                        dtype=np.int32,
                        name="used_indices",
                    )
                    assert used_indices.flags.c_contiguous
                    if self.reference.group is not None:
                        group_info = np.array(self.reference.group).astype(np.int32, copy=False)
                        _, self.group = np.unique(
                            np.repeat(range(len(group_info)), repeats=group_info)[self.used_indices], return_counts=True
                        )
                    self._handle = ctypes.c_void_p()
                    params_str = _param_dict_to_str(self.params)
                    _safe_call(
                        _LIB.LGBM_DatasetGetSubset(
                            self.reference.construct()._handle,
                            used_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                            ctypes.c_int32(used_indices.shape[0]),
                            _c_str(params_str),
                            ctypes.byref(self._handle),
                        )
                    )
                    if not self.free_raw_data:
                        self.get_data()
                    if self.group is not None:
                        self.set_group(self.group)
                    if self.position is not None:
                        self.set_position(self.position)
                    if self.get_label() is None:
                        raise ValueError("Label should not be None.")
                    if (
                        isinstance(self._predictor, _InnerPredictor)
                        and self._predictor is not self.reference._predictor
                    ):
                        self.get_data()
                        self._set_init_score_by_predictor(
                            predictor=self._predictor, data=self.data, used_indices=used_indices
                        )
            else:
                # create train
                self._lazy_init(
                    data=self.data,
                    label=self.label,
                    reference=None,
                    weight=self.weight,
                    group=self.group,
                    init_score=self.init_score,
                    predictor=self._predictor,
                    feature_name=self.feature_name,
                    categorical_feature=self.categorical_feature,
                    params=self.params,
                    position=self.position,
                )
            if self.free_raw_data:
                self.data = None
            self.feature_name = self.get_feature_name()
        return self

    def create_valid(
        self,
        data: _LGBM_TrainDataType,
        label: Optional[_LGBM_LabelType] = None,
        weight: Optional[_LGBM_WeightType] = None,
        group: Optional[_LGBM_GroupType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        params: Optional[Dict[str, Any]] = None,
        position: Optional[_LGBM_PositionType] = None,
    ) -> "Dataset":
        """Create validation data align with current Dataset.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, scipy.sparse, Sequence, list of Sequence, list of numpy array, pyarrow Table or polars DataFrame
            Data source of Dataset.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM) or a LightGBM Dataset binary file.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        label : list, numpy 1-D array, pandas Series / one-column DataFrame, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Label of the data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        weight : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Weight for each instance. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        group : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow ChunkedArray, pyarrow Table (for multi-class task), polars Series, polars DataFrame (for multi-class task) or None, optional (default=None)
            Init score for Dataset.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        params : dict or None, optional (default=None)
            Other parameters for validation Dataset.
        position : numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Position of items used in unbiased learning-to-rank task.

        Returns
        -------
        valid : Dataset
            Validation Dataset with reference to self.
        """
        ret = Dataset(
            data,
            label=label,
            reference=self,
            weight=weight,
            group=group,
            position=position,
            init_score=init_score,
            params=params,
            free_raw_data=self.free_raw_data,
        )
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        return ret

    def subset(
        self,
        used_indices: List[int],
        params: Optional[Dict[str, Any]] = None,
    ) -> "Dataset":
        """Get subset of current Dataset.

        Parameters
        ----------
        used_indices : list of int
            Indices used to create the subset.
        params : dict or None, optional (default=None)
            These parameters will be passed to Dataset constructor.

        Returns
        -------
        subset : Dataset
            Subset of the current Dataset.
        """
        if params is None:
            params = self.params
        ret = Dataset(
            None,
            reference=self,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            params=params,
            free_raw_data=self.free_raw_data,
        )
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        ret.used_indices = sorted(used_indices)
        return ret

    def save_binary(self, filename: Union[str, Path]) -> "Dataset":
        """Save Dataset to a binary file.

        .. note::

            Please note that `init_score` is not saved in binary file.
            If you need it, please set it again after loading Dataset.

        Parameters
        ----------
        filename : str or pathlib.Path
            Name of the output file.

        Returns
        -------
        self : Dataset
            Returns self.
        """
        _safe_call(
            _LIB.LGBM_DatasetSaveBinary(
                self.construct()._handle,
                _c_str(str(filename)),
            )
        )
        return self

    def _update_params(self, params: Optional[Dict[str, Any]]) -> "Dataset":
        if not params:
            return self
        params = deepcopy(params)

        def update() -> None:
            if not self.params:
                self.params = params
            else:
                self._params_back_up = deepcopy(self.params)
                self.params.update(params)

        if self._handle is None:
            update()
        elif params is not None:
            ret = _LIB.LGBM_DatasetUpdateParamChecking(
                _c_str(_param_dict_to_str(self.params)),
                _c_str(_param_dict_to_str(params)),
            )
            if ret != 0:
                # could be updated if data is not freed
                if self.data is not None:
                    update()
                    self._free_handle()
                else:
                    raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
        return self

    def _reverse_update_params(self) -> "Dataset":
        if self._handle is None:
            self.params = deepcopy(self._params_back_up)
            self._params_back_up = None
        return self

    def set_field(
        self,
        field_name: str,
        data: Optional[_LGBM_SetFieldType],
    ) -> "Dataset":
        """Set property into the Dataset.

        Parameters
        ----------
        field_name : str
            The field name of the information.
        data : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow ChunkedArray, pyarrow Table (for multi-class task), polars Series, polars DataFrame (for multi-class task) or None
            The data to be set.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        Returns
        -------
        self : Dataset
            Dataset with set property.
        """
        if self._handle is None:
            raise Exception(f"Cannot set {field_name} before construct dataset")
        if data is None:
            # set to None
            _safe_call(
                _LIB.LGBM_DatasetSetField(
                    self._handle,
                    _c_str(field_name),
                    None,
                    ctypes.c_int(0),
                    ctypes.c_int(_FIELD_TYPE_MAPPER[field_name]),
                )
            )
            return self

        # If the data is Arrow, we can just pass it to C
        if (nwd.is_into_dataframe(data) or nwd.is_into_series(data)) and not isinstance(
            data, (pd_DataFrame, pd_Series)
        ):
            pycapsule = nw.from_native(data, allow_series=True).__arrow_c_stream__()
            _safe_call(
                _LIB.LGBM_DatasetSetFieldFromArrowStream(
                    self._handle,
                    _c_str(field_name),
                    _extract_arrow_stream_capsule_pointer(pycapsule),
                )
            )
            self.version += 1
            return self

        dtype: "np.typing.DTypeLike"
        if field_name == "init_score":
            dtype = np.float64
            if _is_1d_collection(data):
                data = _list_to_1d_numpy(data=data, dtype=dtype, name=field_name)
            elif _is_2d_collection(data):
                data = _data_to_2d_numpy(data=data, dtype=dtype, name=field_name)
                data = data.ravel(order="F")
            else:
                raise TypeError(
                    "init_score must be list, numpy 1-D array or pandas Series.\n"
                    "In multiclass classification init_score can also be a list of lists, numpy 2-D array or pandas DataFrame."
                )
        else:
            if field_name in {"group", "position"}:
                dtype = np.int32
            else:
                dtype = np.float32
            data = _list_to_1d_numpy(data=data, dtype=dtype, name=field_name)

        ptr_data: Union[_ctypes_float_ptr, _ctypes_int_ptr]
        if data.dtype == np.float32 or data.dtype == np.float64:
            ptr_data, type_data, _ = _c_float_array(data)
        elif data.dtype == np.int32:
            ptr_data, type_data, _ = _c_int_array(data)
        else:
            raise TypeError(f"Expected np.float32/64 or np.int32, met type({data.dtype})")
        if type_data != _FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("Input type error for set_field")
        _safe_call(
            _LIB.LGBM_DatasetSetField(
                self._handle,
                _c_str(field_name),
                ptr_data,
                ctypes.c_int(len(data)),
                ctypes.c_int(type_data),
            )
        )
        self.version += 1
        return self

    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get property from the Dataset.

        Can only be run on a constructed Dataset.

        Unlike ``get_group()``, ``get_init_score()``, ``get_label()``, ``get_position()``, and ``get_weight()``,
        this method ignores any raw data passed into ``lgb.Dataset()`` on the Python side, and will only read
        data from the constructed C++ ``Dataset`` object.

        Parameters
        ----------
        field_name : str
            The field name of the information.

        Returns
        -------
        info : numpy array or None
            A numpy array with information from the Dataset.
        """
        if self._handle is None:
            raise Exception(f"Cannot get {field_name} before construct Dataset")
        tmp_out_len = ctypes.c_int(0)
        out_type = ctypes.c_int(0)
        ret = ctypes.POINTER(ctypes.c_void_p)()
        _safe_call(
            _LIB.LGBM_DatasetGetField(
                self._handle,
                _c_str(field_name),
                ctypes.byref(tmp_out_len),
                ctypes.byref(ret),
                ctypes.byref(out_type),
            )
        )
        if out_type.value != _FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("Return type error for get_field")
        if tmp_out_len.value == 0:
            return None
        if out_type.value == _C_API_DTYPE_INT32:
            arr = _cint32_array_to_numpy(
                cptr=ctypes.cast(ret, ctypes.POINTER(ctypes.c_int32)),
                length=tmp_out_len.value,
            )
        elif out_type.value == _C_API_DTYPE_FLOAT32:
            arr = _cfloat32_array_to_numpy(
                cptr=ctypes.cast(ret, ctypes.POINTER(ctypes.c_float)),
                length=tmp_out_len.value,
            )
        elif out_type.value == _C_API_DTYPE_FLOAT64:
            arr = _cfloat64_array_to_numpy(
                cptr=ctypes.cast(ret, ctypes.POINTER(ctypes.c_double)),
                length=tmp_out_len.value,
            )
        else:
            raise TypeError("Unknown type")
        if field_name == "init_score":
            num_data = self.num_data()
            num_classes = arr.size // num_data
            if num_classes > 1:
                arr = arr.reshape((num_data, num_classes), order="F")
        return arr

    def set_categorical_feature(
        self,
        categorical_feature: _LGBM_CategoricalFeatureConfiguration,
    ) -> "Dataset":
        """Set categorical features.

        Parameters
        ----------
        categorical_feature : list of str or int, or 'auto'
            Names or indices of categorical features.

        Returns
        -------
        self : Dataset
            Dataset with set categorical features.
        """
        if self.categorical_feature == categorical_feature:
            return self
        if self.data is not None:
            if self.categorical_feature is None:
                self.categorical_feature = categorical_feature
                return self._free_handle()
            elif categorical_feature == "auto":
                return self
            else:
                if self.categorical_feature != "auto":
                    _log_warning(
                        "categorical_feature in Dataset is overridden.\n"
                        f"New categorical_feature is {list(categorical_feature)}"
                    )
                self.categorical_feature = categorical_feature
                return self._free_handle()
        else:
            raise LightGBMError(
                "Cannot set categorical feature after freed raw data, "
                "set free_raw_data=False when construct Dataset to avoid this."
            )

    def _set_predictor(
        self,
        predictor: Optional[_InnerPredictor],
    ) -> "Dataset":
        """Set predictor for continued training.

        It is not recommended for user to call this function.
        Please use init_model argument in engine.train() or engine.cv() instead.
        """
        if predictor is None and self._predictor is None:
            return self
        elif isinstance(predictor, _InnerPredictor) and isinstance(self._predictor, _InnerPredictor):
            if (predictor == self._predictor) and (
                predictor.current_iteration() == self._predictor.current_iteration()
            ):
                return self
        if self._handle is None:
            self._predictor = predictor
        elif self.data is not None:
            self._predictor = predictor
            self._set_init_score_by_predictor(
                predictor=self._predictor,
                data=self.data,
                used_indices=None,
            )
        elif self.used_indices is not None and self.reference is not None and self.reference.data is not None:
            self._predictor = predictor
            self._set_init_score_by_predictor(
                predictor=self._predictor,
                data=self.reference.data,
                used_indices=self.used_indices,
            )
        else:
            raise LightGBMError(
                "Cannot set predictor after freed raw data, "
                "set free_raw_data=False when construct Dataset to avoid this."
            )
        return self

    def set_reference(self, reference: "Dataset") -> "Dataset":
        """Set reference Dataset.

        Parameters
        ----------
        reference : Dataset
            Reference that is used as a template to construct the current Dataset.

        Returns
        -------
        self : Dataset
            Dataset with set reference.
        """
        self.set_categorical_feature(reference.categorical_feature).set_feature_name(
            reference.feature_name
        )._set_predictor(reference._predictor)
        # we're done if self and reference share a common upstream reference
        if self.get_ref_chain().intersection(reference.get_ref_chain()):
            return self
        if self.data is not None:
            self.reference = reference
            return self._free_handle()
        else:
            raise LightGBMError(
                "Cannot set reference after freed raw data, "
                "set free_raw_data=False when construct Dataset to avoid this."
            )

    def set_feature_name(self, feature_name: _LGBM_FeatureNameConfiguration) -> "Dataset":
        """Set feature name.

        Parameters
        ----------
        feature_name : list of str
            Feature names.

        Returns
        -------
        self : Dataset
            Dataset with set feature name.
        """
        if feature_name != "auto":
            self.feature_name = feature_name
            self._has_non_default_feature_names = True
        if self._handle is not None and feature_name is not None and feature_name != "auto":
            if len(feature_name) != self.num_feature():
                raise ValueError(
                    f"Length of feature_name({len(feature_name)}) and num_feature({self.num_feature()}) don't match"
                )
            c_feature_name = [_c_str(name) for name in feature_name]
            _safe_call(
                _LIB.LGBM_DatasetSetFeatureNames(
                    self._handle,
                    _c_array(ctypes.c_char_p, c_feature_name),
                    ctypes.c_int(len(feature_name)),
                )
            )
        return self

    def set_label(self, label: Optional[_LGBM_LabelType]) -> "Dataset":
        """Set label of Dataset.

        Parameters
        ----------
        label : list, numpy 1-D array, pandas Series / one-column DataFrame, pyarrow ChunkedArray, polars Series or None
            The label information to be set into Dataset.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        Returns
        -------
        self : Dataset
            Dataset with set label.
        """
        self.label = label
        if self._handle is not None:
            if isinstance(label, pd_DataFrame):
                if len(label.columns) > 1:
                    raise ValueError("DataFrame for label cannot have multiple columns")
                label_array = np.ravel(_pandas_to_numpy(label, target_dtype=np.float32))
            elif (nwd.is_into_dataframe(label) or nwd.is_into_series(label)) and not isinstance(label, pd_Series):
                label_array = label
            else:
                label_array = _list_to_1d_numpy(data=label, dtype=np.float32, name="label")
            self.set_field("label", label_array)
            self.label = self.get_field("label")  # original values can be modified at cpp side
        return self

    def set_weight(
        self,
        weight: Optional[_LGBM_WeightType],
    ) -> "Dataset":
        """Set weight of each instance.

        Parameters
        ----------
        weight : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None
            Weight to be set for each data point. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        Returns
        -------
        self : Dataset
            Dataset with set weight.
        """
        # Check if the weight contains values other than one
        if weight is not None:
            if nwd.is_into_series(weight):
                if (nw.from_native(weight, series_only=True) == 1).all():
                    weight = None
            elif np.all(weight == 1):
                weight = None
        self.weight = weight

        # Set field
        if self._handle is not None and weight is not None:
            if isinstance(weight, pd_Series) or not nwd.is_into_series(weight):
                weight = _list_to_1d_numpy(data=weight, dtype=np.float32, name="weight")
            self.set_field("weight", weight)
            self.weight = self.get_field("weight")  # original values can be modified at cpp side
        return self

    def set_init_score(
        self,
        init_score: Optional[_LGBM_InitScoreType],
    ) -> "Dataset":
        """Set init score of Booster to start from.

        Parameters
        ----------
        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow ChunkedArray, pyarrow Table (for multi-class task), polars Series, polars DataFrame (for multi-class task) or None
            Init score for Booster.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        Returns
        -------
        self : Dataset
            Dataset with set init score.
        """
        self.init_score = init_score
        if self._handle is not None and init_score is not None:
            self.set_field("init_score", init_score)
            self.init_score = self.get_field("init_score")  # original values can be modified at cpp side
        return self

    def set_group(
        self,
        group: Optional[_LGBM_GroupType],
    ) -> "Dataset":
        """Set group size of Dataset (used for ranking).

        Parameters
        ----------
        group : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        Returns
        -------
        self : Dataset
            Dataset with set group.
        """
        self.group = group
        if self._handle is not None and group is not None:
            if isinstance(group, pd_Series) or not nwd.is_into_series(group):
                group = _list_to_1d_numpy(data=group, dtype=np.int32, name="group")
            self.set_field("group", group)
            # original values can be modified at cpp side
            constructed_group = self.get_field("group")
            if constructed_group is not None:
                self.group = np.diff(constructed_group)
        return self

    def set_position(
        self,
        position: Optional[_LGBM_PositionType],
    ) -> "Dataset":
        """Set position of Dataset (used for ranking).

        Parameters
        ----------
        position : numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Position of items used in unbiased learning-to-rank task.

        Returns
        -------
        self : Dataset
            Dataset with set position.
        """
        self.position = position
        if self._handle is not None and position is not None:
            if isinstance(position, pd_Series) or not nwd.is_into_series(position):
                position = _list_to_1d_numpy(data=position, dtype=np.int32, name="position")
            self.set_field("position", position)
            self.position = self.get_field("position")  # original values can be modified at cpp side
        return self

    def get_feature_name(self) -> List[str]:
        """Get the names of columns (features) in the Dataset.

        Returns
        -------
        feature_names : list of str
            The names of columns (features) in the Dataset.
        """
        if self._handle is None:
            raise LightGBMError("Cannot get feature_name before construct dataset")
        num_feature = self.num_feature()
        tmp_out_len = ctypes.c_int(0)
        reserved_string_buffer_size = 255
        required_string_buffer_size = ctypes.c_size_t(0)
        string_buffers = [ctypes.create_string_buffer(reserved_string_buffer_size) for _ in range(num_feature)]
        ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))  # type: ignore[misc]
        _safe_call(
            _LIB.LGBM_DatasetGetFeatureNames(
                self._handle,
                ctypes.c_int(num_feature),
                ctypes.byref(tmp_out_len),
                ctypes.c_size_t(reserved_string_buffer_size),
                ctypes.byref(required_string_buffer_size),
                ptr_string_buffers,
            )
        )
        if num_feature != tmp_out_len.value:
            raise ValueError("Length of feature names doesn't equal with num_feature")
        actual_string_buffer_size = required_string_buffer_size.value
        # if buffer length is not long enough, reallocate buffers
        if reserved_string_buffer_size < actual_string_buffer_size:
            string_buffers = [ctypes.create_string_buffer(actual_string_buffer_size) for _ in range(num_feature)]
            ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))  # type: ignore[misc]
            _safe_call(
                _LIB.LGBM_DatasetGetFeatureNames(
                    self._handle,
                    ctypes.c_int(num_feature),
                    ctypes.byref(tmp_out_len),
                    ctypes.c_size_t(actual_string_buffer_size),
                    ctypes.byref(required_string_buffer_size),
                    ptr_string_buffers,
                )
            )
        return [string_buffers[i].value.decode("utf-8") for i in range(num_feature)]

    def get_label(self) -> Optional[_LGBM_LabelType]:
        """Get the label of the Dataset.

        Returns
        -------
        label : list, numpy 1-D array, pandas Series / one-column DataFrame, pyarrow ChunkedArray, polars Series or None
            The label information from the Dataset.
            For a constructed ``Dataset``, this will only return a numpy array.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs
        """
        if self.label is None:
            self.label = self.get_field("label")
        return self.label

    def get_weight(self) -> Optional[_LGBM_WeightType]:
        """Get the weight of the Dataset.

        Returns
        -------
        weight : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None
            Weight for each data point from the Dataset. Weights should be non-negative.
            For a constructed ``Dataset``, this will only return ``None`` or a numpy array.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs
        """
        if self.weight is None:
            self.weight = self.get_field("weight")
        return self.weight

    def get_init_score(self) -> Optional[_LGBM_InitScoreType]:
        """Get the initial score of the Dataset.

        Returns
        -------
        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow ChunkedArray, pyarrow Table (for multi-class task), polars Series, polars DataFrame (for multi-class task) or None
            Init score of Booster.
            For a constructed ``Dataset``, this will only return ``None`` or a numpy array.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs
        """
        if self.init_score is None:
            self.init_score = self.get_field("init_score")
        return self.init_score

    def get_data(self) -> Optional[_LGBM_TrainDataType]:
        """Get the raw data of the Dataset.

        Returns
        -------
        data : str, pathlib.Path, numpy array, pandas DataFrame, scipy.sparse, Sequence, list of Sequence, list of numpy array, pyarrow Table, polars DataFrame, or None
            Raw data used in the Dataset construction.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs
        """
        if self._handle is None:
            raise Exception("Cannot get data before construct Dataset")
        if self._need_slice and self.used_indices is not None and self.reference is not None:
            self.data = self.reference.data
            if self.data is not None:
                if isinstance(self.data, (np.ndarray, scipy.sparse.spmatrix)):
                    self.data = self.data[self.used_indices, :]
                elif isinstance(self.data, pd_DataFrame):
                    self.data = self.data.iloc[self.used_indices].copy()
                elif isinstance(self.data, Sequence):
                    self.data = self.data[self.used_indices]
                elif nwd.is_into_dataframe(self.data):
                    indices = np.array(self.used_indices)
                    self.data = nw.from_native(self.data)[indices].to_native()
                elif _is_list_of_sequences(self.data) and len(self.data) > 0:
                    self.data = np.array(list(self._yield_row_from_seqlist(self.data, self.used_indices)))
                else:
                    _log_warning(
                        f"Cannot subset {type(self.data).__name__} type of raw data.\nReturning original raw data"
                    )
            self._need_slice = False
        if self.data is None:
            raise LightGBMError(
                "Cannot call `get_data` after freed raw data, "
                "set free_raw_data=False when construct Dataset to avoid this."
            )
        return self.data

    def get_group(self) -> Optional[_LGBM_GroupType]:
        """Get the group of the Dataset.

        Returns
        -------
        group : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            For a constructed ``Dataset``, this will only return ``None`` or a numpy array.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs
        """
        if self.group is None:
            self.group = self.get_field("group")
            if self.group is not None:
                # group data from LightGBM is boundaries data, need to convert to group size
                self.group = np.diff(self.group)
        return self.group

    def get_position(self) -> Optional[_LGBM_PositionType]:
        """Get the position of the Dataset.

        Returns
        -------
        position : numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None
            Position of items used in unbiased learning-to-rank task.
            For a constructed ``Dataset``, this will only return ``None`` or a numpy array.
        """
        if self.position is None:
            self.position = self.get_field("position")
        return self.position

    def num_data(self) -> int:
        """Get the number of rows in the Dataset.

        Returns
        -------
        number_of_rows : int
            The number of rows in the Dataset.
        """
        if self._handle is not None:
            ret = ctypes.c_int(0)
            _safe_call(
                _LIB.LGBM_DatasetGetNumData(
                    self._handle,
                    ctypes.byref(ret),
                )
            )
            return ret.value
        else:
            raise LightGBMError("Cannot get num_data before construct dataset")

    def num_feature(self) -> int:
        """Get the number of columns (features) in the Dataset.

        Returns
        -------
        number_of_columns : int
            The number of columns (features) in the Dataset.
        """
        if self._handle is not None:
            ret = ctypes.c_int(0)
            _safe_call(
                _LIB.LGBM_DatasetGetNumFeature(
                    self._handle,
                    ctypes.byref(ret),
                )
            )
            return ret.value
        else:
            raise LightGBMError("Cannot get num_feature before construct dataset")

    def feature_num_bin(self, feature: Union[int, str]) -> int:
        """Get the number of bins for a feature.

        .. versionadded:: 4.0.0

        Parameters
        ----------
        feature : int or str
            Index or name of the feature.

        Returns
        -------
        number_of_bins : int
            The number of constructed bins for the feature in the Dataset.
        """
        if self._handle is not None:
            if isinstance(feature, str):
                feature_index = self.feature_name.index(feature)
            else:
                feature_index = feature
            ret = ctypes.c_int(0)
            _safe_call(
                _LIB.LGBM_DatasetGetFeatureNumBin(
                    self._handle,
                    ctypes.c_int(feature_index),
                    ctypes.byref(ret),
                )
            )
            return ret.value
        else:
            raise LightGBMError("Cannot get feature_num_bin before construct dataset")

    def get_ref_chain(self, ref_limit: int = 100) -> Set["Dataset"]:
        """Get a chain of Dataset objects.

        Starts with r, then goes to r.reference (if exists),
        then to r.reference.reference, etc.
        until we hit ``ref_limit`` or a reference loop.

        Parameters
        ----------
        ref_limit : int, optional (default=100)
            The limit number of references.

        Returns
        -------
        ref_chain : set of Dataset
            Chain of references of the Datasets.
        """
        head = self
        ref_chain: Set[Dataset] = set()
        while len(ref_chain) < ref_limit:
            if isinstance(head, Dataset):
                ref_chain.add(head)
                if (head.reference is not None) and (head.reference not in ref_chain):
                    head = head.reference
                else:
                    break
            else:
                break
        return ref_chain

    def add_features_from(self, other: "Dataset") -> "Dataset":
        """Add features from other Dataset to the current Dataset.

        Both Datasets must be constructed before calling this method.

        Parameters
        ----------
        other : Dataset
            The Dataset to take features from.

        Returns
        -------
        self : Dataset
            Dataset with the new features added.
        """
        if self._handle is None or other._handle is None:
            raise ValueError("Both source and target Datasets must be constructed before adding features")
        _safe_call(
            _LIB.LGBM_DatasetAddFeaturesFrom(
                self._handle,
                other._handle,
            )
        )
        was_none = self.data is None
        old_self_data_type = type(self.data).__name__
        if other.data is None:
            self.data = None
        elif self.data is not None:
            if isinstance(self.data, np.ndarray):
                if isinstance(other.data, np.ndarray):
                    self.data = np.hstack((self.data, other.data))
                elif isinstance(other.data, scipy.sparse.spmatrix):
                    self.data = np.hstack((self.data, other.data.toarray()))
                elif isinstance(other.data, pd_DataFrame):
                    self.data = np.hstack((self.data, other.data.values))
                else:
                    self.data = None
            elif isinstance(self.data, scipy.sparse.spmatrix):
                sparse_format = self.data.getformat()
                if isinstance(other.data, (np.ndarray, scipy.sparse.spmatrix)):
                    self.data = scipy.sparse.hstack((self.data, other.data), format=sparse_format)
                elif isinstance(other.data, pd_DataFrame):
                    self.data = scipy.sparse.hstack((self.data, other.data.values), format=sparse_format)
                else:
                    self.data = None
            elif isinstance(self.data, pd_DataFrame):
                if not PANDAS_INSTALLED:
                    raise LightGBMError(
                        "Cannot add features to DataFrame type of raw data "
                        "without pandas installed. "
                        "Install pandas and restart your session."
                    )
                if isinstance(other.data, np.ndarray):
                    self.data = concat((self.data, pd_DataFrame(other.data)), axis=1, ignore_index=True)
                elif isinstance(other.data, scipy.sparse.spmatrix):
                    self.data = concat((self.data, pd_DataFrame(other.data.toarray())), axis=1, ignore_index=True)
                elif isinstance(other.data, pd_DataFrame):
                    self.data = concat((self.data, other.data), axis=1, ignore_index=True)
                else:
                    self.data = None
            else:
                self.data = None
        if self.data is None:
            err_msg = (
                f"Cannot add features from {type(other.data).__name__} type of raw data to "
                f"{old_self_data_type} type of raw data.\n"
            )
            err_msg += (
                "Set free_raw_data=False when construct Dataset to avoid this" if was_none else "Freeing raw data"
            )
            _log_warning(err_msg)
        self.feature_name = self.get_feature_name()
        _log_warning(
            "Resetting categorical features.\n"
            "You can set new categorical features via ``set_categorical_feature`` method"
        )
        self.categorical_feature = "auto"
        self.pandas_categorical = None
        return self

    def _dump_text(self, filename: Union[str, Path]) -> "Dataset":
        """Save Dataset to a text file.

        This format cannot be loaded back in by LightGBM, but is useful for debugging purposes.

        Parameters
        ----------
        filename : str or pathlib.Path
            Name of the output file.

        Returns
        -------
        self : Dataset
            Returns self.
        """
        _safe_call(
            _LIB.LGBM_DatasetDumpText(
                self.construct()._handle,
                _c_str(str(filename)),
            )
        )
        return self


_LGBM_CustomObjectiveFunction = Callable[
    [np.ndarray, Dataset],
    Tuple[np.ndarray, np.ndarray],
]
_LGBM_CustomEvalFunction = Union[
    Callable[
        [np.ndarray, Dataset],
        _LGBM_EvalFunctionResultType,
    ],
    Callable[
        [np.ndarray, Dataset],
        List[_LGBM_EvalFunctionResultType],
    ],
]


class EvalResult(NamedTuple):
    """
    Result from computing an evaluation metric on a dataset.

    In ``lightgbm<4.7.0``, evaluation results were stored in tuples like this:

      * train(): ``(dataset_name, metric_name, metric_value, maximize)``
      * cv(): ``(dataset_name, metric_name, mean(metric_value), maximize, std_dev(metric_value))``

    Parameters
    ----------
    dataset_name : str
        Unique identifier for the dataset this result was computed on.
    metric_name : str
        Unique identifier for the metric (e.g. "rmse").
    metric_value : float
        Value of the evaluation metric.
    maximize : bool
        Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
    metric_std_dev : float or None
        If not ``None``, the standard deviation of metric values computed over a range of results.
        For example, used when aggregating over cross-validation folds in ``cv()``.
    """

    dataset_name: str
    metric_name: str
    metric_value: float
    maximize: bool
    metric_std_dev: Optional[float] = None

    def __len__(self) -> int:
        if not self.is_cv_result():
            return 4
        else:
            return 5

    def __iter__(self) -> Any:
        i = 0
        while i < len(self):
            yield getattr(self, self._fields[i])
            i += 1

    def is_cv_result(self) -> bool:
        """
        Whether the result was created by ``cv()``.

        If ``True``:

          * ``metric_value`` = mean of ``metric_name`` over CV folds
          * ``metric_std_dev`` = standard deviation of ``metric_name`` over CV folds
        """
        return self.metric_std_dev is not None


class Booster:
    """Booster in LightGBM."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        train_set: Optional[Dataset] = None,
        model_file: Optional[Union[str, Path]] = None,
        model_str: Optional[str] = None,
    ):
        """Initialize the Booster.

        Parameters
        ----------
        params : dict or None, optional (default=None)
            Parameters for Booster.
        train_set : Dataset or None, optional (default=None)
            Training dataset.
        model_file : str, pathlib.Path or None, optional (default=None)
            Path to the model file.
        model_str : str or None, optional (default=None)
            Model will be loaded from this string.
        """
        self._handle = ctypes.c_void_p()
        self._network = False
        self.__need_reload_eval_info = True
        self._train_data_name = "training"
        self.__set_objective_to_none = False
        self.best_iteration = -1
        self.best_score: _LGBM_BoosterBestScoreType = {}
        params = {} if params is None else deepcopy(params)
        if train_set is not None:
            # Training task
            if not isinstance(train_set, Dataset):
                raise TypeError(f"Training data should be Dataset instance, met {type(train_set).__name__}")
            params = _choose_param_value(
                main_param_name="machines",
                params=params,
                default_value=None,
            )
            # if "machines" is given, assume user wants to do distributed learning, and set up network
            if params["machines"] is None:
                params.pop("machines", None)
            else:
                machines = params["machines"]
                if isinstance(machines, str):
                    num_machines_from_machine_list = len(machines.split(","))
                elif isinstance(machines, (list, set)):
                    num_machines_from_machine_list = len(machines)
                    machines = ",".join(machines)
                else:
                    raise ValueError("Invalid machines in params.")

                params = _choose_param_value(
                    main_param_name="num_machines",
                    params=params,
                    default_value=num_machines_from_machine_list,
                )
                params = _choose_param_value(
                    main_param_name="local_listen_port",
                    params=params,
                    default_value=12400,
                )
                self.set_network(
                    machines=machines,
                    local_listen_port=params["local_listen_port"],
                    listen_time_out=params.get("time_out", 120),
                    num_machines=params["num_machines"],
                )
            # construct booster object
            train_set.construct()
            # copy the parameters from train_set
            params.update(train_set.get_params())
            params_str = _param_dict_to_str(params)
            _safe_call(
                _LIB.LGBM_BoosterCreate(
                    train_set._handle,
                    _c_str(params_str),
                    ctypes.byref(self._handle),
                )
            )
            # save reference to data
            self.train_set = train_set
            self.valid_sets: List[Dataset] = []
            self.name_valid_sets: List[str] = []
            self.__num_dataset = 1
            self.__init_predictor = train_set._predictor
            if self.__init_predictor is not None:
                _safe_call(
                    _LIB.LGBM_BoosterMerge(
                        self._handle,
                        self.__init_predictor._handle,
                    )
                )
            out_num_class = ctypes.c_int(0)
            _safe_call(
                _LIB.LGBM_BoosterGetNumClasses(
                    self._handle,
                    ctypes.byref(out_num_class),
                )
            )
            self.__num_class = out_num_class.value
            # buffer for inner predict
            self.__inner_predict_buffer: List[Optional[np.ndarray]] = [None]
            self.__is_predicted_cur_iter = [False]
            self.__get_eval_info()
            self.pandas_categorical = train_set.pandas_categorical
            self.train_set_version = train_set.version
        elif model_file is not None:
            # Prediction task
            out_num_iterations = ctypes.c_int(0)
            _safe_call(
                _LIB.LGBM_BoosterCreateFromModelfile(
                    _c_str(str(model_file)),
                    ctypes.byref(out_num_iterations),
                    ctypes.byref(self._handle),
                )
            )
            out_num_class = ctypes.c_int(0)
            _safe_call(
                _LIB.LGBM_BoosterGetNumClasses(
                    self._handle,
                    ctypes.byref(out_num_class),
                )
            )
            self.__num_class = out_num_class.value
            self.pandas_categorical = _load_pandas_categorical(file_name=model_file)
            if params:
                _log_warning("Ignoring params argument, using parameters from model file.")
            params = self._get_loaded_param()
        elif model_str is not None:
            self.model_from_string(model_str)
            if params:
                _log_warning("Ignoring params argument, using parameters from model string.")
            params = self._get_loaded_param()
        else:
            raise TypeError(
                "Need at least one training dataset or model file or model string to create Booster instance"
            )
        self.params = params

    def __del__(self) -> None:
        try:
            if self._network:
                self.free_network()
        except AttributeError:
            pass
        try:
            if self._handle is not None:
                _safe_call(_LIB.LGBM_BoosterFree(self._handle))
        except AttributeError:
            pass

    def __copy__(self) -> "Booster":
        return self.__deepcopy__(None)

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> "Booster":
        model_str = self.model_to_string(num_iteration=-1)
        return Booster(model_str=model_str)

    def __getstate__(self) -> Dict[str, Any]:
        this = self.__dict__.copy()
        handle = this["_handle"]
        this.pop("train_set", None)
        this.pop("valid_sets", None)
        if handle is not None:
            this["_handle"] = self.model_to_string(num_iteration=-1)
        return this

    def __setstate__(self, state: Dict[str, Any]) -> None:
        model_str = state.get("_handle", state.get("handle", None))
        if model_str is not None:
            handle = ctypes.c_void_p()
            out_num_iterations = ctypes.c_int(0)
            _safe_call(
                _LIB.LGBM_BoosterLoadModelFromString(
                    _c_str(model_str),
                    ctypes.byref(out_num_iterations),
                    ctypes.byref(handle),
                )
            )
            state["_handle"] = handle
        self.__dict__.update(state)

    def _get_loaded_param(self) -> Dict[str, Any]:
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
        _safe_call(
            _LIB.LGBM_BoosterGetLoadedParam(
                self._handle,
                ctypes.c_int64(buffer_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer,
            )
        )
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, re-allocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
            _safe_call(
                _LIB.LGBM_BoosterGetLoadedParam(
                    self._handle,
                    ctypes.c_int64(actual_len),
                    ctypes.byref(tmp_out_len),
                    ptr_string_buffer,
                )
            )
        return json.loads(string_buffer.value.decode("utf-8"))

    def free_dataset(self) -> "Booster":
        """Free Booster's Datasets.

        Returns
        -------
        self : Booster
            Booster without Datasets.
        """
        self.__dict__.pop("train_set", None)
        self.__dict__.pop("valid_sets", None)
        self.__num_dataset = 0
        return self

    def _free_buffer(self) -> "Booster":
        self.__inner_predict_buffer = []
        self.__is_predicted_cur_iter = []
        return self

    def set_network(
        self,
        machines: Union[List[str], Set[str], str],
        local_listen_port: int = 12400,
        listen_time_out: int = 120,
        num_machines: int = 1,
    ) -> "Booster":
        """Set the network configuration.

        Parameters
        ----------
        machines : list, set or str
            Names of machines.
        local_listen_port : int, optional (default=12400)
            TCP listen port for local machines.
        listen_time_out : int, optional (default=120)
            Socket time-out in minutes.
        num_machines : int, optional (default=1)
            The number of machines for distributed learning application.

        Returns
        -------
        self : Booster
            Booster with set network.
        """
        if isinstance(machines, (list, set)):
            machines = ",".join(machines)
        _safe_call(
            _LIB.LGBM_NetworkInit(
                _c_str(machines),
                ctypes.c_int(local_listen_port),
                ctypes.c_int(listen_time_out),
                ctypes.c_int(num_machines),
            )
        )
        self._network = True
        return self

    def free_network(self) -> "Booster":
        """Free Booster's network.

        Returns
        -------
        self : Booster
            Booster with freed network.
        """
        _safe_call(_LIB.LGBM_NetworkFree())
        self._network = False
        return self

    def trees_to_dataframe(self) -> pd_DataFrame:
        """Parse the fitted model and return in an easy-to-read pandas DataFrame.

        The returned DataFrame has the following columns.

            - ``tree_index`` : int64, which tree a node belongs to. 0-based, so a value of ``6``, for example, means "this node is in the 7th tree".
            - ``node_depth`` : int64, how far a node is from the root of the tree. The root node has a value of ``1``, its direct children are ``2``, etc.
            - ``node_index`` : str, unique identifier for a node.
            - ``left_child`` : str, ``node_index`` of the child node to the left of a split. ``None`` for leaf nodes.
            - ``right_child`` : str, ``node_index`` of the child node to the right of a split. ``None`` for leaf nodes.
            - ``parent_index`` : str, ``node_index`` of this node's parent. ``None`` for the root node.
            - ``split_feature`` : str, name of the feature used for splitting. ``None`` for leaf nodes.
            - ``split_gain`` : float64, gain from adding this split to the tree. ``NaN`` for leaf nodes.
            - ``threshold`` : float64, value of the feature used to decide which side of the split a record will go down. ``NaN`` for leaf nodes.
            - ``decision_type`` : str, logical operator describing how to compare a value to ``threshold``.
              For example, ``split_feature = "Column_10", threshold = 15, decision_type = "<="`` means that
              records where ``Column_10 <= 15`` follow the left side of the split, otherwise follows the right side of the split. ``None`` for leaf nodes.
            - ``missing_direction`` : str, split direction that missing values should go to. ``None`` for leaf nodes.
            - ``missing_type`` : str, describes what types of values are treated as missing.
            - ``value`` : float64, predicted value for this leaf node, multiplied by the learning rate.
            - ``weight`` : float64 or int64, sum of Hessian (second-order derivative of objective), summed over observations that fall in this node.
            - ``count`` : int64, number of records in the training data that fall into this node.

        Returns
        -------
        result : pandas DataFrame
            Returns a pandas DataFrame of the parsed model.
        """
        if not PANDAS_INSTALLED:
            raise LightGBMError(
                "This method cannot be run without pandas installed. "
                "You must install pandas and restart your session to use this method."
            )

        if self.num_trees() == 0:
            raise LightGBMError("There are no trees in this Booster and thus nothing to parse")

        def _is_split_node(tree: Dict[str, Any]) -> bool:
            return "split_index" in tree.keys()

        def create_node_record(
            tree: Dict[str, Any],
            node_depth: int = 1,
            tree_index: Optional[int] = None,
            feature_names: Optional[List[str]] = None,
            parent_node: Optional[str] = None,
        ) -> Dict[str, Any]:
            def _get_node_index(
                tree: Dict[str, Any],
                tree_index: Optional[int],
            ) -> str:
                tree_num = f"{tree_index}-" if tree_index is not None else ""
                is_split = _is_split_node(tree)
                node_type = "S" if is_split else "L"
                # if a single node tree it won't have `leaf_index` so return 0
                node_num = tree.get("split_index" if is_split else "leaf_index", 0)
                return f"{tree_num}{node_type}{node_num}"

            def _get_split_feature(
                *,
                tree: Dict[str, Any],
                feature_names: Optional[List[str]],
            ) -> Optional[str]:
                if _is_split_node(tree):
                    if feature_names is not None:
                        feature_name = feature_names[tree["split_feature"]]
                    else:
                        feature_name = tree["split_feature"]
                else:
                    feature_name = None
                return feature_name

            def _is_single_node_tree(tree: Dict[str, Any]) -> bool:
                return set(tree.keys()) == {"leaf_value", "leaf_count"}

            # Create the node record, and populate universal data members
            node: Dict[str, Union[int, str, None]] = OrderedDict()
            node["tree_index"] = tree_index
            node["node_depth"] = node_depth
            node["node_index"] = _get_node_index(tree, tree_index)
            node["left_child"] = None
            node["right_child"] = None
            node["parent_index"] = parent_node
            node["split_feature"] = _get_split_feature(tree=tree, feature_names=feature_names)
            node["split_gain"] = None
            node["threshold"] = None
            node["decision_type"] = None
            node["missing_direction"] = None
            node["missing_type"] = None
            node["value"] = None
            node["weight"] = None
            node["count"] = None

            # Update values to reflect node type (leaf or split)
            if _is_split_node(tree):
                node["left_child"] = _get_node_index(tree["left_child"], tree_index)
                node["right_child"] = _get_node_index(tree["right_child"], tree_index)
                node["split_gain"] = tree["split_gain"]
                node["threshold"] = tree["threshold"]
                node["decision_type"] = tree["decision_type"]
                node["missing_direction"] = "left" if tree["default_left"] else "right"
                node["missing_type"] = tree["missing_type"]
                node["value"] = tree["internal_value"]
                node["weight"] = tree["internal_weight"]
                node["count"] = tree["internal_count"]
            else:
                node["value"] = tree["leaf_value"]
                if not _is_single_node_tree(tree):
                    node["weight"] = tree["leaf_weight"]
                    node["count"] = tree["leaf_count"]

            return node

        def tree_dict_to_node_list(
            tree: Dict[str, Any],
            node_depth: int = 1,
            tree_index: Optional[int] = None,
            feature_names: Optional[List[str]] = None,
            parent_node: Optional[str] = None,
        ) -> List[Dict[str, Any]]:
            node = create_node_record(
                tree=tree,
                node_depth=node_depth,
                tree_index=tree_index,
                feature_names=feature_names,
                parent_node=parent_node,
            )

            res = [node]

            if _is_split_node(tree):
                # traverse the next level of the tree
                children = ["left_child", "right_child"]
                for child in children:
                    subtree_list = tree_dict_to_node_list(
                        tree=tree[child],
                        node_depth=node_depth + 1,
                        tree_index=tree_index,
                        feature_names=feature_names,
                        parent_node=node["node_index"],
                    )
                    # In tree format, "subtree_list" is a list of node records (dicts),
                    # and we add node to the list.
                    res.extend(subtree_list)
            return res

        model_dict = self.dump_model()
        feature_names = model_dict["feature_names"]
        model_list = []
        for tree in model_dict["tree_info"]:
            model_list.extend(
                tree_dict_to_node_list(
                    tree=tree["tree_structure"], tree_index=tree["tree_index"], feature_names=feature_names
                )
            )

        return pd_DataFrame(model_list, columns=model_list[0].keys())

    def set_train_data_name(self, name: str) -> "Booster":
        """Set the name to the training Dataset.

        Parameters
        ----------
        name : str
            Name for the training Dataset.

        Returns
        -------
        self : Booster
            Booster with set training Dataset name.
        """
        self._train_data_name = name
        return self

    def add_valid(self, data: Dataset, name: str) -> "Booster":
        """Add validation data.

        Parameters
        ----------
        data : Dataset
            Validation data.
        name : str
            Name of validation data.

        Returns
        -------
        self : Booster
            Booster with set validation data.
        """
        if not isinstance(data, Dataset):
            raise TypeError(f"Validation data should be Dataset instance, met {type(data).__name__}")
        if data._predictor is not self.__init_predictor:
            raise LightGBMError("Add validation data failed, you should use same predictor for these data")
        _safe_call(
            _LIB.LGBM_BoosterAddValidData(
                self._handle,
                data.construct()._handle,
            )
        )
        self.valid_sets.append(data)
        self.name_valid_sets.append(name)
        self.__num_dataset += 1
        self.__inner_predict_buffer.append(None)
        self.__is_predicted_cur_iter.append(False)
        return self

    def reset_parameter(self, params: Dict[str, Any]) -> "Booster":
        """Reset parameters of Booster.

        Parameters
        ----------
        params : dict
            New parameters for Booster.

        Returns
        -------
        self : Booster
            Booster with new parameters.
        """
        params_str = _param_dict_to_str(params)
        if params_str:
            _safe_call(
                _LIB.LGBM_BoosterResetParameter(
                    self._handle,
                    _c_str(params_str),
                )
            )
        self.params.update(params)
        return self

    def update(
        self,
        train_set: Optional[Dataset] = None,
        fobj: Optional[_LGBM_CustomObjectiveFunction] = None,
    ) -> bool:
        """Update Booster for one iteration.

        Parameters
        ----------
        train_set : Dataset or None, optional (default=None)
            Training data.
            If None, last training data is used.
        fobj : callable or None, optional (default=None)
            Customized objective function.
            Should accept two parameters: preds, train_data,
            and return (grad, hess).

                preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                    The predicted values.
                    Predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task.
                train_data : Dataset
                    The training dataset.
                grad : numpy 1-D array or numpy 2-D array (for multi-class task)
                    The value of the first order derivative (gradient) of the loss
                    with respect to the elements of preds for each sample point.
                hess : numpy 1-D array or numpy 2-D array (for multi-class task)
                    The value of the second order derivative (Hessian) of the loss
                    with respect to the elements of preds for each sample point.

            For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes],
            and grad and hess should be returned in the same format.

        Returns
        -------
        produced_empty_tree : bool
            ``True`` if the tree(s) produced by this iteration did not have any splits.
            This usually means that training is "finished" (calling ``update()`` again
            will not change the model's predictions). However, that is not always the
            case. For example, if you have added any randomness (like column sampling by
            setting ``feature_fraction_bynode < 1.0``), it is possible that another call
            to ``update()`` would produce a non-empty tree.
        """
        # need reset training data
        if train_set is None and self.train_set_version != self.train_set.version:
            train_set = self.train_set
            is_the_same_train_set = False
        else:
            is_the_same_train_set = train_set is self.train_set and self.train_set_version == train_set.version
        if train_set is not None and not is_the_same_train_set:
            if not isinstance(train_set, Dataset):
                raise TypeError(f"Training data should be Dataset instance, met {type(train_set).__name__}")
            if train_set._predictor is not self.__init_predictor:
                raise LightGBMError("Replace training data failed, you should use same predictor for these data")
            self.train_set = train_set
            _safe_call(
                _LIB.LGBM_BoosterResetTrainingData(
                    self._handle,
                    self.train_set.construct()._handle,
                )
            )
            self.__inner_predict_buffer[0] = None
            self.train_set_version = self.train_set.version
        produced_empty_tree = ctypes.c_int(0)
        if fobj is None:
            if self.__set_objective_to_none:
                raise LightGBMError("Cannot update due to null objective function.")
            _safe_call(
                _LIB.LGBM_BoosterUpdateOneIter(
                    self._handle,
                    ctypes.byref(produced_empty_tree),
                )
            )
            self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
            return produced_empty_tree.value == 1
        else:
            if not self.__set_objective_to_none:
                self.reset_parameter({"objective": "none"}).__set_objective_to_none = True
            grad, hess = fobj(self.__inner_predict(data_idx=0), self.train_set)
            return self.__boost(grad=grad, hess=hess)

    def __boost(
        self,
        *,
        grad: np.ndarray,
        hess: np.ndarray,
    ) -> bool:
        """Boost Booster for one iteration with customized gradient statistics.

        .. note::

            Score is returned before any transformation,
            e.g. it is raw margin instead of probability of positive class for binary task.
            For multi-class task, score are numpy 2-D array of shape = [n_samples, n_classes],
            and grad and hess should be returned in the same format.

        Parameters
        ----------
        grad : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of score for each sample point.
        hess : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of score for each sample point.

        Returns
        -------
        produced_empty_tree : bool
            ``True`` if the tree(s) produced by this iteration did not have any splits.
            This usually means that training is "finished" (calling ``__boost()`` again
            will not change the model's predictions). However, that is not always the
            case. For example, if you have added any randomness (like column sampling by
            setting ``feature_fraction_bynode < 1.0``), it is possible that another call
            to ``__boost()`` would produce a non-empty tree.
        """
        if self.__num_class > 1:
            grad = grad.ravel(order="F")
            hess = hess.ravel(order="F")
        grad = _list_to_1d_numpy(data=grad, dtype=np.float32, name="gradient")
        hess = _list_to_1d_numpy(data=hess, dtype=np.float32, name="hessian")
        assert grad.flags.c_contiguous
        assert hess.flags.c_contiguous
        if len(grad) != len(hess):
            raise ValueError(f"Lengths of gradient ({len(grad)}) and Hessian ({len(hess)}) don't match")
        num_train_data = self.train_set.num_data()
        if len(grad) != num_train_data * self.__num_class:
            raise ValueError(
                f"Lengths of gradient ({len(grad)}) and Hessian ({len(hess)}) "
                f"don't match training data length ({num_train_data}) * "
                f"number of models per one iteration ({self.__num_class})"
            )
        produced_empty_tree = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterUpdateOneIterCustom(
                self._handle,
                grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(produced_empty_tree),
            )
        )
        self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
        return produced_empty_tree.value == 1

    def rollback_one_iter(self) -> "Booster":
        """Rollback one iteration.

        Returns
        -------
        self : Booster
            Booster with rolled back one iteration.
        """
        _safe_call(_LIB.LGBM_BoosterRollbackOneIter(self._handle))
        self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
        return self

    def current_iteration(self) -> int:
        """Get the index of the current iteration.

        Returns
        -------
        cur_iter : int
            The index of the current iteration.
        """
        out_cur_iter = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterGetCurrentIteration(
                self._handle,
                ctypes.byref(out_cur_iter),
            )
        )
        return out_cur_iter.value

    def num_model_per_iteration(self) -> int:
        """Get number of models per iteration.

        Returns
        -------
        model_per_iter : int
            The number of models per iteration.
        """
        model_per_iter = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterNumModelPerIteration(
                self._handle,
                ctypes.byref(model_per_iter),
            )
        )
        return model_per_iter.value

    def num_trees(self) -> int:
        """Get number of weak sub-models.

        Returns
        -------
        num_trees : int
            The number of weak sub-models.
        """
        num_trees = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterNumberOfTotalModel(
                self._handle,
                ctypes.byref(num_trees),
            )
        )
        return num_trees.value

    def upper_bound(self) -> float:
        """Get upper bound value of a model.

        Returns
        -------
        upper_bound : float
            Upper bound value of the model.
        """
        ret = ctypes.c_double(0)
        _safe_call(
            _LIB.LGBM_BoosterGetUpperBoundValue(
                self._handle,
                ctypes.byref(ret),
            )
        )
        return ret.value

    def lower_bound(self) -> float:
        """Get lower bound value of a model.

        Returns
        -------
        lower_bound : float
            Lower bound value of the model.
        """
        ret = ctypes.c_double(0)
        _safe_call(
            _LIB.LGBM_BoosterGetLowerBoundValue(
                self._handle,
                ctypes.byref(ret),
            )
        )
        return ret.value

    def eval(
        self,
        data: Dataset,
        name: str,
        feval: Optional[Union[_LGBM_CustomEvalFunction, List[_LGBM_CustomEvalFunction]]] = None,
    ) -> List[EvalResult]:
        """Evaluate for data.

        Parameters
        ----------
        data : Dataset
            Data for the evaluating.
        name : str
            Name of the data.
        feval : callable, list of callable, or None, optional (default=None)
            Customized evaluation function.
            Each evaluation function should accept two parameters: preds, eval_data,
            and return (metric_name, metric_value, maximize) or list of such tuples.

                preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                    The predicted values.
                    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                    If custom objective function is used, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                eval_data : Dataset
                    A ``Dataset`` to evaluate.
                metric_name : str
                    Unique identifier for the metric (e.g. "custom_adjusted_mse").
                metric_value : float
                    Value of the evaluation metric.
                maximize : bool
                    Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.

        Returns
        -------
        result : list[EvalResult]
            List of ``lightgbm.EvalResult`` objects, named tuples of the form
            (dataset_name, metric_name, metric_value, maximize).
        """
        if not isinstance(data, Dataset):
            raise TypeError("Can only eval for Dataset instance")
        data_idx = -1
        if data is self.train_set:
            data_idx = 0
        else:
            for i in range(len(self.valid_sets)):
                if data is self.valid_sets[i]:
                    data_idx = i + 1
                    break
        # need to push new valid data
        if data_idx == -1:
            self.add_valid(data, name)
            data_idx = self.__num_dataset - 1

        return self.__inner_eval(data_name=name, data_idx=data_idx, feval=feval)

    def eval_train(
        self,
        feval: Optional[Union[_LGBM_CustomEvalFunction, List[_LGBM_CustomEvalFunction]]] = None,
    ) -> List[EvalResult]:
        """Evaluate for training data.

        Parameters
        ----------
        feval : callable, list of callable, or None, optional (default=None)
            Customized evaluation function.
            Each evaluation function should accept two parameters: preds, eval_data,
            and return (metric_name, metric_value, maximize) or list of such tuples.

                preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                    The predicted values.
                    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                    If custom objective function is used, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                eval_data : Dataset
                    The training dataset.
                metric_name : str
                    Unique identifier for the metric (e.g. "custom_adjusted_mse").
                metric_value : float
                    Value of the evaluation metric.
                maximize : bool
                    Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.

        Returns
        -------
        result : list[EvalResult]
            List of ``lightgbm.EvalResult`` objects, named tuples of the form
            (dataset_name, metric_name, metric_value, maximize).
        """
        return self.__inner_eval(data_name=self._train_data_name, data_idx=0, feval=feval)

    def eval_valid(
        self,
        feval: Optional[Union[_LGBM_CustomEvalFunction, List[_LGBM_CustomEvalFunction]]] = None,
    ) -> List[EvalResult]:
        """Evaluate for validation data.

        Parameters
        ----------
        feval : callable, list of callable, or None, optional (default=None)
            Customized evaluation function.
            Each evaluation function should accept two parameters: preds, eval_data,
            and return (metric_name, metric_value, maximize) or list of such tuples.

                preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                    The predicted values.
                    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                    If custom objective function is used, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                eval_data : Dataset
                    The validation dataset.
                metric_name : str
                    Unique identifier for the metric (e.g. "custom_adjusted_mse").
                metric_value : float
                    Value of the evaluation metric.
                maximize : bool
                    Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.

        Returns
        -------
        result : list
            List of ``lightgbm.EvalResult`` objects, named tuples of the form
            (dataset_name, metric_name, metric_value, maximize).
        """
        return [
            item
            for i in range(1, self.__num_dataset)
            for item in self.__inner_eval(data_name=self.name_valid_sets[i - 1], data_idx=i, feval=feval)
        ]

    def save_model(
        self,
        filename: Union[str, Path],
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = "split",
    ) -> "Booster":
        """Save Booster to file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Filename to save Booster.
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        self : Booster
            Returns self.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration
        importance_type_int = _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        _safe_call(
            _LIB.LGBM_BoosterSaveModel(
                self._handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int(importance_type_int),
                _c_str(str(filename)),
            )
        )
        _dump_pandas_categorical(self.pandas_categorical, filename)
        return self

    def shuffle_models(
        self,
        start_iteration: int = 0,
        end_iteration: int = -1,
    ) -> "Booster":
        """Shuffle models.

        Parameters
        ----------
        start_iteration : int, optional (default=0)
            The first iteration that will be shuffled.
        end_iteration : int, optional (default=-1)
            The last iteration that will be shuffled.
            If <= 0, means the last available iteration.

        Returns
        -------
        self : Booster
            Booster with shuffled models.
        """
        _safe_call(
            _LIB.LGBM_BoosterShuffleModels(
                self._handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(end_iteration),
            )
        )
        return self

    def model_from_string(self, model_str: str) -> "Booster":
        """Load Booster from a string.

        Parameters
        ----------
        model_str : str
            Model will be loaded from this string.

        Returns
        -------
        self : Booster
            Loaded Booster object.
        """
        # ensure that existing Booster is freed before replacing it
        # with a new one created from file
        _safe_call(_LIB.LGBM_BoosterFree(self._handle))
        self._free_buffer()
        self._handle = ctypes.c_void_p()
        out_num_iterations = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterLoadModelFromString(
                _c_str(model_str),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self._handle),
            )
        )
        out_num_class = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterGetNumClasses(
                self._handle,
                ctypes.byref(out_num_class),
            )
        )
        self.__num_class = out_num_class.value
        self.pandas_categorical = _load_pandas_categorical(model_str=model_str)
        return self

    def model_to_string(
        self,
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = "split",
    ) -> str:
        """Save Booster to string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        str_repr : str
            String representation of Booster.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration
        importance_type_int = _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
        _safe_call(
            _LIB.LGBM_BoosterSaveModelToString(
                self._handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int(importance_type_int),
                ctypes.c_int64(buffer_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer,
            )
        )
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, re-allocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
            _safe_call(
                _LIB.LGBM_BoosterSaveModelToString(
                    self._handle,
                    ctypes.c_int(start_iteration),
                    ctypes.c_int(num_iteration),
                    ctypes.c_int(importance_type_int),
                    ctypes.c_int64(actual_len),
                    ctypes.byref(tmp_out_len),
                    ptr_string_buffer,
                )
            )
        ret = string_buffer.value.decode("utf-8")
        ret += _dump_pandas_categorical(self.pandas_categorical)
        return ret

    def dump_model(
        self,
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = "split",
        object_hook: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Dump Booster to JSON format.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be dumped.
            If None, if the best iteration exists, it is dumped; otherwise, all iterations are dumped.
            If <= 0, all iterations are dumped.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be dumped.
        importance_type : str, optional (default="split")
            What type of feature importance should be dumped.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.
        object_hook : callable or None, optional (default=None)
            If not None, ``object_hook`` is a function called while parsing the json
            string returned by the C API. It may be used to alter the json, to store
            specific values while building the json structure. It avoids
            walking through the structure again. It saves a significant amount
            of time if the number of trees is huge.
            Signature is ``def object_hook(node: dict) -> dict``.
            None is equivalent to ``lambda node: node``.
            See documentation of ``json.loads()`` for further details.

        Returns
        -------
        json_repr : dict
            JSON format of Booster.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration
        importance_type_int = _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
        _safe_call(
            _LIB.LGBM_BoosterDumpModel(
                self._handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int(importance_type_int),
                ctypes.c_int64(buffer_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer,
            )
        )
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, reallocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
            _safe_call(
                _LIB.LGBM_BoosterDumpModel(
                    self._handle,
                    ctypes.c_int(start_iteration),
                    ctypes.c_int(num_iteration),
                    ctypes.c_int(importance_type_int),
                    ctypes.c_int64(actual_len),
                    ctypes.byref(tmp_out_len),
                    ptr_string_buffer,
                )
            )
        ret = json.loads(string_buffer.value.decode("utf-8"), object_hook=object_hook)
        ret["pandas_categorical"] = json.loads(json.dumps(self.pandas_categorical))
        return ret

    def predict(
        self,
        data: _LGBM_PredictDataType,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> _LGBM_PredictReturnType:
        """Make a prediction.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, scipy.sparse, pyarrow Table or polars DataFrame
            Data source for prediction.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.

            .. note::

                If you want to get more explanations for your model's predictions using SHAP values,
                like SHAP interaction values,
                you can install the shap package (https://github.com/slundberg/shap).
                Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
                column, where the last column is the expected value.

        data_has_header : bool, optional (default=False)
            Whether the data has header.
            Used only if data is str.
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        predictor = _InnerPredictor.from_booster(
            booster=self,
            pred_parameter=deepcopy(kwargs),
        )
        if num_iteration is None:
            if start_iteration <= 0:
                num_iteration = self.best_iteration
            else:
                num_iteration = -1
        return predictor.predict(
            data=data,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            data_has_header=data_has_header,
            validate_features=validate_features,
        )

    def refit(
        self,
        data: _LGBM_TrainDataType,
        label: _LGBM_LabelType,
        decay_rate: float = 0.9,
        reference: Optional[Dataset] = None,
        weight: Optional[_LGBM_WeightType] = None,
        group: Optional[_LGBM_GroupType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        feature_name: _LGBM_FeatureNameConfiguration = "auto",
        categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
        dataset_params: Optional[Dict[str, Any]] = None,
        free_raw_data: bool = True,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> "Booster":
        """Refit the existing Booster by new data.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, scipy.sparse, Sequence, list of Sequence, list of numpy array, pyarrow Table or polars DataFrame
            Data source for refit.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        label : list, numpy 1-D array, pandas Series / one-column DataFrame, pyarrow ChunkedArray, polars Series or None
            Label for refit.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        decay_rate : float, optional (default=0.9)
            Decay rate of refit,
            will use ``leaf_output = decay_rate * old_leaf_output + (1.0 - decay_rate) * new_leaf_output`` to refit trees.
        reference : Dataset or None, optional (default=None)
            Reference for ``data``.

            .. versionadded:: 4.0.0

        weight : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Weight for each ``data`` instance. Weights should be non-negative.

            .. versionadded:: 4.0.0

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        group : list, numpy 1-D array, pandas Series, pyarrow ChunkedArray, polars Series or None, optional (default=None)
            Group/query size for ``data``.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

            .. versionadded:: 4.0.0

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : list, list of lists (for multi-class task), numpy array, pandas Series, pandas DataFrame (for multi-class task), pyarrow ChunkedArray, pyarrow Table (for multi-class task), polars Series, polars DataFrame (for multi-class task) or None, optional (default=None)
            Init score for ``data``.

            .. versionadded:: 4.0.0

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        feature_name : list of str, or 'auto', optional (default="auto")
            Feature names for ``data``.
            If 'auto' and data is pandas DataFrame, data columns names are used.

            .. versionadded:: 4.0.0

        categorical_feature : list of str or int, or 'auto', optional (default="auto")
            Categorical features for ``data``.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.

            .. versionadded:: 4.0.0

        dataset_params : dict or None, optional (default=None)
            Other parameters for Dataset ``data``.

            .. versionadded:: 4.0.0

        free_raw_data : bool, optional (default=True)
            If True, raw data is freed after constructing inner Dataset for ``data``.

            .. versionadded:: 4.0.0

        validate_features : bool, optional (default=False)
            If True, ensure that the features used to refit the model match the original ones.
            Used only if data is pandas DataFrame.

            .. versionadded:: 4.0.0

        **kwargs
            Other parameters for refit.
            These parameters will be passed to ``predict`` method.

        Returns
        -------
        result : Booster
            Refitted Booster.
        """
        if self.__set_objective_to_none:
            raise LightGBMError("Cannot refit due to null objective function.")
        if dataset_params is None:
            dataset_params = {}
        predictor = _InnerPredictor.from_booster(booster=self, pred_parameter=deepcopy(kwargs))
        leaf_preds: np.ndarray = predictor.predict(  # type: ignore[assignment]
            data=data,
            start_iteration=-1,
            pred_leaf=True,
            validate_features=validate_features,
        )
        nrow, ncol = leaf_preds.shape
        out_is_linear = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterGetLinear(
                self._handle,
                ctypes.byref(out_is_linear),
            )
        )
        new_params = _choose_param_value(
            main_param_name="linear_tree",
            params=self.params,
            default_value=None,
        )
        new_params["linear_tree"] = bool(out_is_linear.value)
        new_params.update(dataset_params)

        # 'categorical_feature' can end up in self.params when a Booster
        # is created from a model string or file... pre-process to ensure it's passed
        # via a keyword argument to the Dataset constructor instead of 'params'.
        new_params = _choose_param_value(
            main_param_name="categorical_feature",
            params=new_params,
            default_value=None,
        )
        cat_features_from_params = new_params.pop("categorical_feature")

        # reconcile params and keyword argument
        if cat_features_from_params:
            if categorical_feature == "auto":
                categorical_feature = cat_features_from_params
            elif cat_features_from_params != categorical_feature:
                error_msg = (
                    "'categorical_feature' value passed to Booster.refit() is different from  "
                    "'categorical_feature' value found in Booster.params. "
                    "Preferring the value passed via keyword argument. "
                    "Using refit() to change which columns are treated as categorical is not supported. "
                    "If you have a valid use case for this, please open an issue at https://github.com/lightgbm-org/LightGBM/issues."
                )
                raise LightGBMError(error_msg)

        train_set = Dataset(
            data=data,
            label=label,
            reference=reference,
            weight=weight,
            group=group,
            init_score=init_score,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            params=new_params,
            free_raw_data=free_raw_data,
        )
        new_params["refit_decay_rate"] = decay_rate
        new_booster = Booster(new_params, train_set)
        # Copy models
        _safe_call(
            _LIB.LGBM_BoosterMerge(
                new_booster._handle,
                predictor._handle,
            )
        )
        leaf_preds = leaf_preds.reshape(-1)
        ptr_data, _, _ = _c_int_array(leaf_preds)
        _safe_call(
            _LIB.LGBM_BoosterRefit(
                new_booster._handle,
                ptr_data,
                ctypes.c_int32(nrow),
                ctypes.c_int32(ncol),
            )
        )
        new_booster._network = self._network
        return new_booster

    def get_leaf_output(self, tree_id: int, leaf_id: int) -> float:
        """Get the output of a leaf.

        Parameters
        ----------
        tree_id : int
            The index of the tree.
        leaf_id : int
            The index of the leaf in the tree.

        Returns
        -------
        result : float
            The output of the leaf.
        """
        ret = ctypes.c_double(0)
        _safe_call(
            _LIB.LGBM_BoosterGetLeafValue(
                self._handle,
                ctypes.c_int(tree_id),
                ctypes.c_int(leaf_id),
                ctypes.byref(ret),
            )
        )
        return ret.value

    def set_leaf_output(
        self,
        tree_id: int,
        leaf_id: int,
        value: float,
    ) -> "Booster":
        """Set the output of a leaf.

        .. versionadded:: 4.0.0

        Parameters
        ----------
        tree_id : int
            The index of the tree.
        leaf_id : int
            The index of the leaf in the tree.
        value : float
            Value to set as the output of the leaf.

        Returns
        -------
        self : Booster
            Booster with the leaf output set.
        """
        _safe_call(
            _LIB.LGBM_BoosterSetLeafValue(
                self._handle,
                ctypes.c_int(tree_id),
                ctypes.c_int(leaf_id),
                ctypes.c_double(value),
            )
        )
        return self

    def num_feature(self) -> int:
        """Get number of features.

        Returns
        -------
        num_feature : int
            The number of features.
        """
        out_num_feature = ctypes.c_int(0)
        _safe_call(
            _LIB.LGBM_BoosterGetNumFeature(
                self._handle,
                ctypes.byref(out_num_feature),
            )
        )
        return out_num_feature.value

    def feature_name(self) -> List[str]:
        """Get names of features.

        Returns
        -------
        result : list of str
            List with names of features.
        """
        num_feature = self.num_feature()
        # Get name of features
        tmp_out_len = ctypes.c_int(0)
        reserved_string_buffer_size = 255
        required_string_buffer_size = ctypes.c_size_t(0)
        string_buffers = [ctypes.create_string_buffer(reserved_string_buffer_size) for _ in range(num_feature)]
        ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))  # type: ignore[misc]
        _safe_call(
            _LIB.LGBM_BoosterGetFeatureNames(
                self._handle,
                ctypes.c_int(num_feature),
                ctypes.byref(tmp_out_len),
                ctypes.c_size_t(reserved_string_buffer_size),
                ctypes.byref(required_string_buffer_size),
                ptr_string_buffers,
            )
        )
        if num_feature != tmp_out_len.value:
            raise ValueError("Length of feature names doesn't equal with num_feature")
        actual_string_buffer_size = required_string_buffer_size.value
        # if buffer length is not long enough, reallocate buffers
        if reserved_string_buffer_size < actual_string_buffer_size:
            string_buffers = [ctypes.create_string_buffer(actual_string_buffer_size) for _ in range(num_feature)]
            ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))  # type: ignore[misc]
            _safe_call(
                _LIB.LGBM_BoosterGetFeatureNames(
                    self._handle,
                    ctypes.c_int(num_feature),
                    ctypes.byref(tmp_out_len),
                    ctypes.c_size_t(actual_string_buffer_size),
                    ctypes.byref(required_string_buffer_size),
                    ptr_string_buffers,
                )
            )
        return [string_buffers[i].value.decode("utf-8") for i in range(num_feature)]

    def feature_importance(
        self,
        importance_type: str = "split",
        iteration: Optional[int] = None,
    ) -> np.ndarray:
        """Get feature importances.

        Parameters
        ----------
        importance_type : str, optional (default="split")
            How the importance is calculated.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.
        iteration : int or None, optional (default=None)
            Limit number of iterations in the feature importance calculation.
            If None, if the best iteration exists, it is used; otherwise, all trees are used.
            If <= 0, all trees are used (no limits).

        Returns
        -------
        result : numpy array
            Array with feature importances.
        """
        if iteration is None:
            iteration = self.best_iteration
        importance_type_int = _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        result = np.empty(self.num_feature(), dtype=np.float64)
        _safe_call(
            _LIB.LGBM_BoosterFeatureImportance(
                self._handle,
                ctypes.c_int(iteration),
                ctypes.c_int(importance_type_int),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if importance_type_int == _C_API_FEATURE_IMPORTANCE_SPLIT:
            return result.astype(np.int32)
        else:
            return result

    def get_split_value_histogram(
        self,
        feature: Union[int, str],
        bins: Optional[Union[int, str]] = None,
        xgboost_style: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, pd_DataFrame]:
        """Get split value histogram for the specified feature.

        Parameters
        ----------
        feature : int or str
            The feature name or index the histogram is calculated for.
            If int, interpreted as index.
            If str, interpreted as name.

            .. warning::

                Categorical features are not supported.

        bins : int, str or None, optional (default=None)
            The maximum number of bins.
            If None, or int and > number of unique split values and ``xgboost_style=True``,
            the number of bins equals number of unique split values.
            If str, it should be one from the list of the supported values by ``numpy.histogram()`` function.
        xgboost_style : bool, optional (default=False)
            Whether the returned result should be in the same form as it is in XGBoost.
            If False, the returned value is tuple of 2 numpy arrays as it is in ``numpy.histogram()`` function.
            If True, the returned value is matrix, in which the first column is the right edges of non-empty bins
            and the second one is the histogram values.

        Returns
        -------
        result_tuple : tuple of 2 numpy arrays
            If ``xgboost_style=False``, the values of the histogram of used splitting values for the specified feature
            and the bin edges.
        result_array_like : numpy array or pandas DataFrame (if pandas is installed)
            If ``xgboost_style=True``, the histogram of used splitting values for the specified feature.
        """

        def add(root: Dict[str, Any]) -> None:
            """Recursively add thresholds."""
            if "split_index" in root:  # non-leaf
                if feature_names is not None and isinstance(feature, str):
                    split_feature = feature_names[root["split_feature"]]
                else:
                    split_feature = root["split_feature"]
                if split_feature == feature:
                    if isinstance(root["threshold"], str):
                        raise LightGBMError("Cannot compute split value histogram for the categorical feature")
                    values.append(root["threshold"])
                add(root["left_child"])
                add(root["right_child"])

        model = self.dump_model()
        feature_names = model.get("feature_names")
        tree_infos = model["tree_info"]
        values: List[float] = []
        for tree_info in tree_infos:
            add(tree_info["tree_structure"])

        if bins is None or isinstance(bins, int) and xgboost_style:
            n_unique = len(np.unique(values))
            bins = max(min(n_unique, bins) if bins is not None else n_unique, 1)
        hist, bin_edges = np.histogram(values, bins=bins)
        if xgboost_style:
            ret = np.column_stack((bin_edges[1:], hist))
            ret = ret[ret[:, 1] > 0]
            if PANDAS_INSTALLED:
                return pd_DataFrame(ret, columns=["SplitValue", "Count"])
            else:
                return ret
        else:
            return hist, bin_edges

    def __inner_eval(
        self,
        *,
        data_name: str,
        data_idx: int,
        feval: Optional[Union[_LGBM_CustomEvalFunction, List[_LGBM_CustomEvalFunction]]],
    ) -> List[EvalResult]:
        """Evaluate training or validation data."""
        if data_idx >= self.__num_dataset:
            raise ValueError("Data_idx should be smaller than number of dataset")
        self.__get_eval_info()
        ret = []
        if self.__num_inner_eval > 0:
            result = np.empty(self.__num_inner_eval, dtype=np.float64)
            tmp_out_len = ctypes.c_int(0)
            _safe_call(
                _LIB.LGBM_BoosterGetEval(
                    self._handle,
                    ctypes.c_int(data_idx),
                    ctypes.byref(tmp_out_len),
                    result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
            )
            if tmp_out_len.value != self.__num_inner_eval:
                raise ValueError("Wrong length of eval results")
            for i in range(self.__num_inner_eval):
                ret.append(
                    EvalResult(
                        dataset_name=data_name,
                        metric_name=self.__name_inner_eval[i],
                        metric_value=result[i],
                        maximize=self.__higher_better_inner_eval[i],
                    )
                )
        if callable(feval):
            feval = [feval]
        if feval is not None:
            if data_idx == 0:
                cur_data = self.train_set
            else:
                cur_data = self.valid_sets[data_idx - 1]
            for eval_function in feval:
                if eval_function is None:
                    continue
                feval_ret = eval_function(self.__inner_predict(data_idx=data_idx), cur_data)
                if isinstance(feval_ret, list):
                    for eval_tuple in feval_ret:
                        ret.append(
                            EvalResult(
                                dataset_name=data_name,
                                metric_name=eval_tuple[0],
                                metric_value=eval_tuple[1],
                                maximize=eval_tuple[2],
                            )
                        )
                else:
                    ret.append(
                        EvalResult(
                            dataset_name=data_name,
                            metric_name=feval_ret[0],
                            metric_value=feval_ret[1],
                            maximize=feval_ret[2],
                        )
                    )
        return ret

    def __inner_predict(self, *, data_idx: int) -> np.ndarray:
        """Predict for training and validation dataset."""
        if data_idx >= self.__num_dataset:
            raise ValueError("Data_idx should be smaller than number of dataset")
        if self.__inner_predict_buffer[data_idx] is None:
            if data_idx == 0:
                n_preds = self.train_set.num_data() * self.__num_class
            else:
                n_preds = self.valid_sets[data_idx - 1].num_data() * self.__num_class
            self.__inner_predict_buffer[data_idx] = np.empty(n_preds, dtype=np.float64)
        # avoid to predict many time in one iteration
        if not self.__is_predicted_cur_iter[data_idx]:
            tmp_out_len = ctypes.c_int64(0)
            data_ptr = self.__inner_predict_buffer[data_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_double))  # type: ignore[union-attr]
            _safe_call(
                _LIB.LGBM_BoosterGetPredict(
                    self._handle,
                    ctypes.c_int(data_idx),
                    ctypes.byref(tmp_out_len),
                    data_ptr,
                )
            )
            if tmp_out_len.value != len(self.__inner_predict_buffer[data_idx]):  # type: ignore[arg-type]
                raise ValueError(f"Wrong length of predict results for data {data_idx}")
            self.__is_predicted_cur_iter[data_idx] = True
        result: np.ndarray = self.__inner_predict_buffer[data_idx]  # type: ignore[assignment]
        if self.__num_class > 1:
            num_data = result.size // self.__num_class
            result = result.reshape(num_data, self.__num_class, order="F")
        return result

    def __get_eval_info(self) -> None:
        """Get inner evaluation count and names."""
        if self.__need_reload_eval_info:
            self.__need_reload_eval_info = False
            out_num_eval = ctypes.c_int(0)
            # Get num of inner evals
            _safe_call(
                _LIB.LGBM_BoosterGetEvalCounts(
                    self._handle,
                    ctypes.byref(out_num_eval),
                )
            )
            self.__num_inner_eval = out_num_eval.value
            if self.__num_inner_eval > 0:
                # Get name of eval metrics
                tmp_out_len = ctypes.c_int(0)
                reserved_string_buffer_size = 255
                required_string_buffer_size = ctypes.c_size_t(0)
                string_buffers = [
                    ctypes.create_string_buffer(reserved_string_buffer_size) for _ in range(self.__num_inner_eval)
                ]
                ptr_string_buffers = (ctypes.c_char_p * self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))  # type: ignore[misc]
                _safe_call(
                    _LIB.LGBM_BoosterGetEvalNames(
                        self._handle,
                        ctypes.c_int(self.__num_inner_eval),
                        ctypes.byref(tmp_out_len),
                        ctypes.c_size_t(reserved_string_buffer_size),
                        ctypes.byref(required_string_buffer_size),
                        ptr_string_buffers,
                    )
                )
                if self.__num_inner_eval != tmp_out_len.value:
                    raise ValueError("Length of eval names doesn't equal with num_evals")
                actual_string_buffer_size = required_string_buffer_size.value
                # if buffer length is not long enough, reallocate buffers
                if reserved_string_buffer_size < actual_string_buffer_size:
                    string_buffers = [
                        ctypes.create_string_buffer(actual_string_buffer_size) for _ in range(self.__num_inner_eval)
                    ]
                    ptr_string_buffers = (ctypes.c_char_p * self.__num_inner_eval)(
                        *map(ctypes.addressof, string_buffers)
                    )  # type: ignore[misc]
                    _safe_call(
                        _LIB.LGBM_BoosterGetEvalNames(
                            self._handle,
                            ctypes.c_int(self.__num_inner_eval),
                            ctypes.byref(tmp_out_len),
                            ctypes.c_size_t(actual_string_buffer_size),
                            ctypes.byref(required_string_buffer_size),
                            ptr_string_buffers,
                        )
                    )
                self.__name_inner_eval = [string_buffers[i].value.decode("utf-8") for i in range(self.__num_inner_eval)]
                self.__higher_better_inner_eval = [
                    name.startswith(("auc", "ndcg@", "map@", "average_precision")) for name in self.__name_inner_eval
                ]
