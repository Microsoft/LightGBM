# coding: utf-8
"""Wrapper for C API of LightGBM."""
import ctypes
import json
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
from logging import Logger
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Set, Union

import numpy as np
import scipy.sparse

from .compat import PANDAS_INSTALLED, concat, dt_DataTable, is_dtype_sparse, pd_DataFrame, pd_Series
from .libpath import find_lib_path


class _DummyLogger:
    def info(self, msg):
        print(msg)

    def warning(self, msg):
        warnings.warn(msg, stacklevel=3)


_LOGGER = _DummyLogger()


def register_logger(logger):
    """Register custom logger.

    Parameters
    ----------
    logger : logging.Logger
        Custom logger.
    """
    if not isinstance(logger, Logger):
        raise TypeError("Logger should inherit logging.Logger class")
    global _LOGGER
    _LOGGER = logger


def _normalize_native_string(func):
    """Join log messages from native library which come by chunks."""
    msg_normalized = []

    @wraps(func)
    def wrapper(msg):
        nonlocal msg_normalized
        if msg.strip() == '':
            msg = ''.join(msg_normalized)
            msg_normalized = []
            return func(msg)
        else:
            msg_normalized.append(msg)

    return wrapper


def _log_info(msg):
    _LOGGER.info(msg)


def _log_warning(msg):
    _LOGGER.warning(msg)


@_normalize_native_string
def _log_native(msg):
    _LOGGER.info(msg)


def _log_callback(msg):
    """Redirect logs from native library into Python."""
    _log_native(str(msg.decode('utf-8')))


def _load_lib():
    """Load LightGBM library."""
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.LGBM_GetLastError.restype = ctypes.c_char_p
    callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
    lib.callback = callback(_log_callback)
    if lib.LGBM_RegisterLogCallback(lib.callback) != 0:
        raise LightGBMError(lib.LGBM_GetLastError().decode('utf-8'))
    return lib


_LIB = _load_lib()


NUMERIC_TYPES = (int, float, bool)


def _safe_call(ret):
    """Check the return value from C API call.

    Parameters
    ----------
    ret : int
        The return value from C API calls.
    """
    if ret != 0:
        raise LightGBMError(_LIB.LGBM_GetLastError().decode('utf-8'))


def is_numeric(obj):
    """Check whether object is a number or not, include numpy number, etc."""
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        # TypeError: obj is not a string or a number
        # ValueError: invalid literal
        return False


def is_numpy_1d_array(data):
    """Check whether data is a numpy 1-D array."""
    return isinstance(data, np.ndarray) and len(data.shape) == 1


def is_numpy_column_array(data):
    """Check whether data is a column numpy array."""
    if not isinstance(data, np.ndarray):
        return False
    shape = data.shape
    return len(shape) == 2 and shape[1] == 1


def cast_numpy_1d_array_to_dtype(array, dtype):
    """Cast numpy 1d array to given dtype."""
    if array.dtype == dtype:
        return array
    return array.astype(dtype=dtype, copy=False)


def is_1d_list(data):
    """Check whether data is a 1-D list."""
    return isinstance(data, list) and (not data or is_numeric(data[0]))


def list_to_1d_numpy(data, dtype=np.float32, name='list'):
    """Convert data to numpy 1-D array."""
    if is_numpy_1d_array(data):
        return cast_numpy_1d_array_to_dtype(data, dtype)
    elif is_numpy_column_array(data):
        _log_warning('Converting column-vector to 1d array')
        array = data.ravel()
        return cast_numpy_1d_array_to_dtype(array, dtype)
    elif is_1d_list(data):
        return np.array(data, dtype=dtype, copy=False)
    elif isinstance(data, pd_Series):
        if _get_bad_pandas_dtypes([data.dtypes]):
            raise ValueError('Series.dtypes must be int, float or bool')
        return np.array(data, dtype=dtype, copy=False)  # SparseArray should be supported as well
    else:
        raise TypeError(f"Wrong type({type(data).__name__}) for {name}.\n"
                        "It should be list, numpy 1-D array or pandas Series")


def cfloat32_array_to_numpy(cptr, length):
    """Convert a ctypes float pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_float)):
        return np.ctypeslib.as_array(cptr, shape=(length,)).copy()
    else:
        raise RuntimeError('Expected float pointer')


def cfloat64_array_to_numpy(cptr, length):
    """Convert a ctypes double pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_double)):
        return np.ctypeslib.as_array(cptr, shape=(length,)).copy()
    else:
        raise RuntimeError('Expected double pointer')


def cint32_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int32)):
        return np.ctypeslib.as_array(cptr, shape=(length,)).copy()
    else:
        raise RuntimeError('Expected int32 pointer')


def cint64_array_to_numpy(cptr, length):
    """Convert a ctypes int pointer array to a numpy array."""
    if isinstance(cptr, ctypes.POINTER(ctypes.c_int64)):
        return np.ctypeslib.as_array(cptr, shape=(length,)).copy()
    else:
        raise RuntimeError('Expected int64 pointer')


def c_str(string):
    """Convert a Python string to C string."""
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Convert a Python array to C array."""
    return (ctype * len(values))(*values)


def json_default_with_numpy(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def param_dict_to_str(data):
    """Convert Python dictionary to string, which is passed to C API."""
    if data is None or not data:
        return ""
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or is_numpy_1d_array(val):
            def to_string(x):
                if isinstance(x, list):
                    return f"[{','.join(map(str, x))}]"
                else:
                    return str(x)
            pairs.append(f"{key}={','.join(map(to_string, val))}")
        elif isinstance(val, (str, NUMERIC_TYPES)) or is_numeric(val):
            pairs.append(f"{key}={val}")
        elif val is not None:
            raise TypeError(f'Unknown type of parameter:{key}, got:{type(val).__name__}')
    return ' '.join(pairs)


class _TempFile:
    def __enter__(self):
        with NamedTemporaryFile(prefix="lightgbm_tmp_", delete=True) as f:
            self.name = f.name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.isfile(self.name):
            os.remove(self.name)

    def readlines(self):
        with open(self.name, "r+") as f:
            ret = f.readlines()
        return ret

    def writelines(self, lines):
        with open(self.name, "w+") as f:
            f.writelines(lines)


class LightGBMError(Exception):
    """Error thrown by LightGBM."""

    pass


# DeprecationWarning is not shown by default, so let's create our own with higher level
class LGBMDeprecationWarning(UserWarning):
    """Custom deprecation warning."""

    pass


class _ConfigAliases:
    aliases = {"bin_construct_sample_cnt": {"bin_construct_sample_cnt",
                                            "subsample_for_bin"},
               "boosting": {"boosting",
                            "boosting_type",
                            "boost"},
               "categorical_feature": {"categorical_feature",
                                       "cat_feature",
                                       "categorical_column",
                                       "cat_column"},
               "data_random_seed": {"data_random_seed",
                                    "data_seed"},
               "early_stopping_round": {"early_stopping_round",
                                        "early_stopping_rounds",
                                        "early_stopping",
                                        "n_iter_no_change"},
               "enable_bundle": {"enable_bundle",
                                 "is_enable_bundle",
                                 "bundle"},
               "eval_at": {"eval_at",
                           "ndcg_eval_at",
                           "ndcg_at",
                           "map_eval_at",
                           "map_at"},
               "group_column": {"group_column",
                                "group",
                                "group_id",
                                "query_column",
                                "query",
                                "query_id"},
               "header": {"header",
                          "has_header"},
               "ignore_column": {"ignore_column",
                                 "ignore_feature",
                                 "blacklist"},
               "is_enable_sparse": {"is_enable_sparse",
                                    "is_sparse",
                                    "enable_sparse",
                                    "sparse"},
               "label_column": {"label_column",
                                "label"},
               "local_listen_port": {"local_listen_port",
                                     "local_port",
                                     "port"},
               "machines": {"machines",
                            "workers",
                            "nodes"},
               "metric": {"metric",
                          "metrics",
                          "metric_types"},
               "num_class": {"num_class",
                             "num_classes"},
               "num_iterations": {"num_iterations",
                                  "num_iteration",
                                  "n_iter",
                                  "num_tree",
                                  "num_trees",
                                  "num_round",
                                  "num_rounds",
                                  "num_boost_round",
                                  "n_estimators"},
               "num_machines": {"num_machines",
                                "num_machine"},
               "num_threads": {"num_threads",
                               "num_thread",
                               "nthread",
                               "nthreads",
                               "n_jobs"},
               "objective": {"objective",
                             "objective_type",
                             "app",
                             "application"},
               "pre_partition": {"pre_partition",
                                 "is_pre_partition"},
               "tree_learner": {"tree_learner",
                                "tree",
                                "tree_type",
                                "tree_learner_type"},
               "two_round": {"two_round",
                             "two_round_loading",
                             "use_two_round_loading"},
               "verbosity": {"verbosity",
                             "verbose"},
               "weight_column": {"weight_column",
                                 "weight"}}

    @classmethod
    def get(cls, *args):
        ret = set()
        for i in args:
            ret |= cls.aliases.get(i, {i})
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

    # find a value, and remove other aliases with .pop()
    # prefer the value of 'main_param_name' if it exists, otherwise search the aliases
    found_value = None
    if main_param_name in params.keys():
        found_value = params[main_param_name]

    for param in _ConfigAliases.get(main_param_name):
        val = params.pop(param, None)
        if found_value is None and val is not None:
            found_value = val

    if found_value is not None:
        params[main_param_name] = found_value
    else:
        params[main_param_name] = default_value

    return params


MAX_INT32 = (1 << 31) - 1

"""Macro definition of data type in C API of LightGBM"""
C_API_DTYPE_FLOAT32 = 0
C_API_DTYPE_FLOAT64 = 1
C_API_DTYPE_INT32 = 2
C_API_DTYPE_INT64 = 3

"""Matrix is row major in Python"""
C_API_IS_ROW_MAJOR = 1

"""Macro definition of prediction type in C API of LightGBM"""
C_API_PREDICT_NORMAL = 0
C_API_PREDICT_RAW_SCORE = 1
C_API_PREDICT_LEAF_INDEX = 2
C_API_PREDICT_CONTRIB = 3

"""Macro definition of sparse matrix type"""
C_API_MATRIX_TYPE_CSR = 0
C_API_MATRIX_TYPE_CSC = 1

"""Macro definition of feature importance type"""
C_API_FEATURE_IMPORTANCE_SPLIT = 0
C_API_FEATURE_IMPORTANCE_GAIN = 1

"""Data type of data field"""
FIELD_TYPE_MAPPER = {"label": C_API_DTYPE_FLOAT32,
                     "weight": C_API_DTYPE_FLOAT32,
                     "init_score": C_API_DTYPE_FLOAT64,
                     "group": C_API_DTYPE_INT32}

"""String name to int feature importance type mapper"""
FEATURE_IMPORTANCE_TYPE_MAPPER = {"split": C_API_FEATURE_IMPORTANCE_SPLIT,
                                  "gain": C_API_FEATURE_IMPORTANCE_GAIN}


def convert_from_sliced_object(data):
    """Fix the memory of multi-dimensional sliced object."""
    if isinstance(data, np.ndarray) and isinstance(data.base, np.ndarray):
        if not data.flags.c_contiguous:
            _log_warning("Usage of np.ndarray subset (sliced data) is not recommended "
                         "due to it will double the peak memory cost in LightGBM.")
            return np.copy(data)
    return data


def c_float_label(label):
    """Get pointer of float numpy array / list for label."""
    if label is None:
        ptr_label = None
        type_label = C_API_DTYPE_FLOAT32
    else:
        label = list_to_1d_numpy(_label_from_pandas(label), name="label")
        label = convert_from_sliced_object(label)
        assert label.flags.c_contiguous
        ptr_label = label.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        type_label = C_API_DTYPE_FLOAT32
    return ptr_label, type_label


def c_float_array(data):
    """Get pointer of float numpy array / list for data."""
    if is_1d_list(data):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        data = convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        if data.dtype == np.float32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            type_data = C_API_DTYPE_FLOAT32
        elif data.dtype == np.float64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            type_data = C_API_DTYPE_FLOAT64
        else:
            raise TypeError(f"Expected np.float32 or np.float64, met type({data.dtype})")
    else:
        raise TypeError(f"Unknown type({type(data).__name__})")
    return (ptr_data, type_data, data)  # return `data` to avoid the temporary copy is freed


def c_int_array(data):
    """Get pointer of int numpy array / list."""
    if is_1d_list(data):
        data = np.array(data, copy=False)
    if is_numpy_1d_array(data):
        data = convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        if data.dtype == np.int32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            type_data = C_API_DTYPE_INT32
        elif data.dtype == np.int64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            type_data = C_API_DTYPE_INT64
        else:
            raise TypeError(f"Expected np.int32 or np.int64, met type({data.dtype})")
    else:
        raise TypeError(f"Unknown type({type(data).__name__})")
    return (ptr_data, type_data, data)  # return `data` to avoid the temporary copy is freed


def _get_bad_pandas_dtypes(dtypes):
    pandas_dtype_mapper = {'int8': 'int', 'int16': 'int', 'int32': 'int',
                           'int64': 'int', 'uint8': 'int', 'uint16': 'int',
                           'uint32': 'int', 'uint64': 'int', 'bool': 'int',
                           'float16': 'float', 'float32': 'float', 'float64': 'float'}
    bad_indices = [i for i, dtype in enumerate(dtypes) if (dtype.name not in pandas_dtype_mapper
                                                           and (not is_dtype_sparse(dtype)
                                                                or dtype.subtype.name not in pandas_dtype_mapper))]
    return bad_indices


def _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical):
    if isinstance(data, pd_DataFrame):
        if len(data.shape) != 2 or data.shape[0] < 1:
            raise ValueError('Input data must be 2 dimensional and non empty.')
        if feature_name == 'auto' or feature_name is None:
            data = data.rename(columns=str)
        cat_cols = list(data.select_dtypes(include=['category']).columns)
        cat_cols_not_ordered = [col for col in cat_cols if not data[col].cat.ordered]
        if pandas_categorical is None:  # train dataset
            pandas_categorical = [list(data[col].cat.categories) for col in cat_cols]
        else:
            if len(cat_cols) != len(pandas_categorical):
                raise ValueError('train and valid dataset categorical_feature do not match.')
            for col, category in zip(cat_cols, pandas_categorical):
                if list(data[col].cat.categories) != list(category):
                    data[col] = data[col].cat.set_categories(category)
        if len(cat_cols):  # cat_cols is list
            data = data.copy()  # not alter origin DataFrame
            data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
        if categorical_feature is not None:
            if feature_name is None:
                feature_name = list(data.columns)
            if categorical_feature == 'auto':  # use cat cols from DataFrame
                categorical_feature = cat_cols_not_ordered
            else:  # use cat cols specified by user
                categorical_feature = list(categorical_feature)
        if feature_name == 'auto':
            feature_name = list(data.columns)
        bad_indices = _get_bad_pandas_dtypes(data.dtypes)
        if bad_indices:
            bad_index_cols_str = ', '.join(data.columns[bad_indices])
            raise ValueError("DataFrame.dtypes for data must be int, float or bool.\n"
                             "Did not expect the data types in the following fields: "
                             f"{bad_index_cols_str}")
        data = data.values
        if data.dtype != np.float32 and data.dtype != np.float64:
            data = data.astype(np.float32)
    else:
        if feature_name == 'auto':
            feature_name = None
        if categorical_feature == 'auto':
            categorical_feature = None
    return data, feature_name, categorical_feature, pandas_categorical


def _label_from_pandas(label):
    if isinstance(label, pd_DataFrame):
        if len(label.columns) > 1:
            raise ValueError('DataFrame for label cannot have multiple columns')
        if _get_bad_pandas_dtypes(label.dtypes):
            raise ValueError('DataFrame.dtypes for label must be int, float or bool')
        label = np.ravel(label.values.astype(np.float32, copy=False))
    return label


def _dump_pandas_categorical(pandas_categorical, file_name=None):
    categorical_json = json.dumps(pandas_categorical, default=json_default_with_numpy)
    pandas_str = f'\npandas_categorical:{categorical_json}\n'
    if file_name is not None:
        with open(file_name, 'a') as f:
            f.write(pandas_str)
    return pandas_str


def _load_pandas_categorical(file_name=None, model_str=None):
    pandas_key = 'pandas_categorical:'
    offset = -len(pandas_key)
    if file_name is not None:
        max_offset = -os.path.getsize(file_name)
        with open(file_name, 'rb') as f:
            while True:
                if offset < max_offset:
                    offset = max_offset
                f.seek(offset, os.SEEK_END)
                lines = f.readlines()
                if len(lines) >= 2:
                    break
                offset *= 2
        last_line = lines[-1].decode('utf-8').strip()
        if not last_line.startswith(pandas_key):
            last_line = lines[-2].decode('utf-8').strip()
    elif model_str is not None:
        idx = model_str.rfind('\n', 0, offset)
        last_line = model_str[idx:].strip()
    if last_line.startswith(pandas_key):
        return json.loads(last_line[len(pandas_key):])
    else:
        return None


class _InnerPredictor:
    """_InnerPredictor of LightGBM.

    Not exposed to user.
    Used only for prediction, usually used for continued training.

    .. note::

        Can be converted from Booster, but cannot be converted to Booster.
    """

    def __init__(self, model_file=None, booster_handle=None, pred_parameter=None):
        """Initialize the _InnerPredictor.

        Parameters
        ----------
        model_file : string or None, optional (default=None)
            Path to the model file.
        booster_handle : object or None, optional (default=None)
            Handle of Booster.
        pred_parameter: dict or None, optional (default=None)
            Other parameters for the prediciton.
        """
        self.handle = ctypes.c_void_p()
        self.__is_manage_handle = True
        if model_file is not None:
            """Prediction task"""
            out_num_iterations = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(
                c_str(model_file),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self.handle)))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.num_class = out_num_class.value
            self.num_total_iteration = out_num_iterations.value
            self.pandas_categorical = _load_pandas_categorical(file_name=model_file)
        elif booster_handle is not None:
            self.__is_manage_handle = False
            self.handle = booster_handle
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.num_class = out_num_class.value
            self.num_total_iteration = self.current_iteration()
            self.pandas_categorical = None
        else:
            raise TypeError('Need model_file or booster_handle to create a predictor')

        pred_parameter = {} if pred_parameter is None else pred_parameter
        self.pred_parameter = param_dict_to_str(pred_parameter)

    def __del__(self):
        try:
            if self.__is_manage_handle:
                _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        except AttributeError:
            pass

    def __getstate__(self):
        this = self.__dict__.copy()
        this.pop('handle', None)
        return this

    def predict(self, data, start_iteration=0, num_iteration=-1,
                raw_score=False, pred_leaf=False, pred_contrib=False, data_has_header=False,
                is_reshape=True):
        """Predict logic.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            When data type is string, it represents the path of txt file.
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
        is_reshape : bool, optional (default=True)
            Whether to reshape to (nrow, ncol).

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        if isinstance(data, Dataset):
            raise TypeError("Cannot use Dataset instance for prediction, please use raw data instead")
        data = _data_from_pandas(data, None, None, self.pandas_categorical)[0]
        predict_type = C_API_PREDICT_NORMAL
        if raw_score:
            predict_type = C_API_PREDICT_RAW_SCORE
        if pred_leaf:
            predict_type = C_API_PREDICT_LEAF_INDEX
        if pred_contrib:
            predict_type = C_API_PREDICT_CONTRIB
        int_data_has_header = 1 if data_has_header else 0

        if isinstance(data, str):
            with _TempFile() as f:
                _safe_call(_LIB.LGBM_BoosterPredictForFile(
                    self.handle,
                    c_str(data),
                    ctypes.c_int(int_data_has_header),
                    ctypes.c_int(predict_type),
                    ctypes.c_int(start_iteration),
                    ctypes.c_int(num_iteration),
                    c_str(self.pred_parameter),
                    c_str(f.name)))
                lines = f.readlines()
                nrow = len(lines)
                preds = [float(token) for line in lines for token in line.split('\t')]
                preds = np.array(preds, dtype=np.float64, copy=False)
        elif isinstance(data, scipy.sparse.csr_matrix):
            preds, nrow = self.__pred_for_csr(data, start_iteration, num_iteration, predict_type)
        elif isinstance(data, scipy.sparse.csc_matrix):
            preds, nrow = self.__pred_for_csc(data, start_iteration, num_iteration, predict_type)
        elif isinstance(data, np.ndarray):
            preds, nrow = self.__pred_for_np2d(data, start_iteration, num_iteration, predict_type)
        elif isinstance(data, list):
            try:
                data = np.array(data)
            except BaseException:
                raise ValueError('Cannot convert data list to numpy array.')
            preds, nrow = self.__pred_for_np2d(data, start_iteration, num_iteration, predict_type)
        elif isinstance(data, dt_DataTable):
            preds, nrow = self.__pred_for_np2d(data.to_numpy(), start_iteration, num_iteration, predict_type)
        else:
            try:
                _log_warning('Converting data to scipy sparse matrix.')
                csr = scipy.sparse.csr_matrix(data)
            except BaseException:
                raise TypeError(f'Cannot predict data for type {type(data).__name__}')
            preds, nrow = self.__pred_for_csr(csr, start_iteration, num_iteration, predict_type)
        if pred_leaf:
            preds = preds.astype(np.int32)
        is_sparse = scipy.sparse.issparse(preds) or isinstance(preds, list)
        if is_reshape and not is_sparse and preds.size != nrow:
            if preds.size % nrow == 0:
                preds = preds.reshape(nrow, -1)
            else:
                raise ValueError(f'Length of predict result ({preds.size}) cannot be divide nrow ({nrow})')
        return preds

    def __get_num_preds(self, start_iteration, num_iteration, nrow, predict_type):
        """Get size of prediction result."""
        if nrow > MAX_INT32:
            raise LightGBMError('LightGBM cannot perform prediction for data'
                                f'with number of rows greater than MAX_INT32 ({MAX_INT32}).\n'
                                'You can split your data into chunks'
                                'and then concatenate predictions for them')
        n_preds = ctypes.c_int64(0)
        _safe_call(_LIB.LGBM_BoosterCalcNumPredict(
            self.handle,
            ctypes.c_int(nrow),
            ctypes.c_int(predict_type),
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.byref(n_preds)))
        return n_preds.value

    def __pred_for_np2d(self, mat, start_iteration, num_iteration, predict_type):
        """Predict for a 2-D numpy matrix."""
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray or list must be 2 dimensional')

        def inner_predict(mat, start_iteration, num_iteration, predict_type, preds=None):
            if mat.dtype == np.float32 or mat.dtype == np.float64:
                data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
            else:  # change non-float data to float data, need to copy
                data = np.array(mat.reshape(mat.size), dtype=np.float32)
            ptr_data, type_ptr_data, _ = c_float_array(data)
            n_preds = self.__get_num_preds(start_iteration, num_iteration, mat.shape[0], predict_type)
            if preds is None:
                preds = np.zeros(n_preds, dtype=np.float64)
            elif len(preds.shape) != 1 or len(preds) != n_preds:
                raise ValueError("Wrong length of pre-allocated predict array")
            out_num_preds = ctypes.c_int64(0)
            _safe_call(_LIB.LGBM_BoosterPredictForMat(
                self.handle,
                ptr_data,
                ctypes.c_int(type_ptr_data),
                ctypes.c_int32(mat.shape[0]),
                ctypes.c_int32(mat.shape[1]),
                ctypes.c_int(C_API_IS_ROW_MAJOR),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                c_str(self.pred_parameter),
                ctypes.byref(out_num_preds),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
            if n_preds != out_num_preds.value:
                raise ValueError("Wrong length for predict results")
            return preds, mat.shape[0]

        nrow = mat.shape[0]
        if nrow > MAX_INT32:
            sections = np.arange(start=MAX_INT32, stop=nrow, step=MAX_INT32)
            # __get_num_preds() cannot work with nrow > MAX_INT32, so calculate overall number of predictions piecemeal
            n_preds = [self.__get_num_preds(start_iteration, num_iteration, i, predict_type) for i in np.diff([0] + list(sections) + [nrow])]
            n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
            preds = np.zeros(sum(n_preds), dtype=np.float64)
            for chunk, (start_idx_pred, end_idx_pred) in zip(np.array_split(mat, sections),
                                                             zip(n_preds_sections, n_preds_sections[1:])):
                # avoid memory consumption by arrays concatenation operations
                inner_predict(chunk, start_iteration, num_iteration, predict_type, preds[start_idx_pred:end_idx_pred])
            return preds, nrow
        else:
            return inner_predict(mat, start_iteration, num_iteration, predict_type)

    def __create_sparse_native(self, cs, out_shape, out_ptr_indptr, out_ptr_indices, out_ptr_data,
                               indptr_type, data_type, is_csr=True):
        # create numpy array from output arrays
        data_indices_len = out_shape[0]
        indptr_len = out_shape[1]
        nrow = out_shape[2]
        ncol = out_shape[3]
        if indptr_type == C_API_DTYPE_INT32:
            out_indptr = cint32_array_to_numpy(out_ptr_indptr, indptr_len)
        elif indptr_type == C_API_DTYPE_INT64:
            out_indptr = cint64_array_to_numpy(out_ptr_indptr, indptr_len)
        else:
            raise TypeError("Expected int32 or int64 type for indptr")
        if data_type == C_API_DTYPE_FLOAT32:
            out_data = cfloat32_array_to_numpy(out_ptr_data, data_indices_len)
        elif data_type == C_API_DTYPE_FLOAT64:
            out_data = cfloat64_array_to_numpy(out_ptr_data, data_indices_len)
        else:
            raise TypeError("Expected float32 or float64 type for data")
        out_indices = cint32_array_to_numpy(out_ptr_indices, data_indices_len)
        # break up indptr based on number of rows (note more than one matrix in multiclass case)
        if is_csr:
            per_class_indptr_shape = cs.indptr.shape[0]
        else:
            per_class_indptr_shape = ncol + 1
        # for CSC there is extra column added
        if not is_csr:
            per_class_indptr_shape += 1
        out_indptr_arrays = np.split(out_indptr, out_indptr.shape[0] / per_class_indptr_shape)
        # reformat output into a csr or csc matrix or list of csr or csc matrices
        cs_output_matrices = []
        offset = 0
        for cs_indptr in out_indptr_arrays:
            matrix_indptr_len = cs_indptr[cs_indptr.shape[0] - 1]
            cs_indices = out_indices[offset + cs_indptr[0]:offset + matrix_indptr_len]
            cs_data = out_data[offset + cs_indptr[0]:offset + matrix_indptr_len]
            offset += matrix_indptr_len
            # same shape as input csr or csc matrix except extra column for expected value
            cs_shape = [nrow, ncol + 1]
            # note: make sure we copy data as it will be deallocated next
            if is_csr:
                cs_output_matrices.append(scipy.sparse.csr_matrix((cs_data, cs_indices, cs_indptr), cs_shape))
            else:
                cs_output_matrices.append(scipy.sparse.csc_matrix((cs_data, cs_indices, cs_indptr), cs_shape))
        # free the temporary native indptr, indices, and data
        _safe_call(_LIB.LGBM_BoosterFreePredictSparse(out_ptr_indptr, out_ptr_indices, out_ptr_data,
                                                      ctypes.c_int(indptr_type), ctypes.c_int(data_type)))
        if len(cs_output_matrices) == 1:
            return cs_output_matrices[0]
        return cs_output_matrices

    def __pred_for_csr(self, csr, start_iteration, num_iteration, predict_type):
        """Predict for a CSR data."""
        def inner_predict(csr, start_iteration, num_iteration, predict_type, preds=None):
            nrow = len(csr.indptr) - 1
            n_preds = self.__get_num_preds(start_iteration, num_iteration, nrow, predict_type)
            if preds is None:
                preds = np.zeros(n_preds, dtype=np.float64)
            elif len(preds.shape) != 1 or len(preds) != n_preds:
                raise ValueError("Wrong length of pre-allocated predict array")
            out_num_preds = ctypes.c_int64(0)

            ptr_indptr, type_ptr_indptr, __ = c_int_array(csr.indptr)
            ptr_data, type_ptr_data, _ = c_float_array(csr.data)

            assert csr.shape[1] <= MAX_INT32
            csr_indices = csr.indices.astype(np.int32, copy=False)

            _safe_call(_LIB.LGBM_BoosterPredictForCSR(
                self.handle,
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
                c_str(self.pred_parameter),
                ctypes.byref(out_num_preds),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
            if n_preds != out_num_preds.value:
                raise ValueError("Wrong length for predict results")
            return preds, nrow

        def inner_predict_sparse(csr, start_iteration, num_iteration, predict_type):
            ptr_indptr, type_ptr_indptr, __ = c_int_array(csr.indptr)
            ptr_data, type_ptr_data, _ = c_float_array(csr.data)
            csr_indices = csr.indices.astype(np.int32, copy=False)
            matrix_type = C_API_MATRIX_TYPE_CSR
            if type_ptr_indptr == C_API_DTYPE_INT32:
                out_ptr_indptr = ctypes.POINTER(ctypes.c_int32)()
            else:
                out_ptr_indptr = ctypes.POINTER(ctypes.c_int64)()
            out_ptr_indices = ctypes.POINTER(ctypes.c_int32)()
            if type_ptr_data == C_API_DTYPE_FLOAT32:
                out_ptr_data = ctypes.POINTER(ctypes.c_float)()
            else:
                out_ptr_data = ctypes.POINTER(ctypes.c_double)()
            out_shape = np.zeros(4, dtype=np.int64)
            _safe_call(_LIB.LGBM_BoosterPredictSparseOutput(
                self.handle,
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
                c_str(self.pred_parameter),
                ctypes.c_int(matrix_type),
                out_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.byref(out_ptr_indptr),
                ctypes.byref(out_ptr_indices),
                ctypes.byref(out_ptr_data)))
            matrices = self.__create_sparse_native(csr, out_shape, out_ptr_indptr, out_ptr_indices, out_ptr_data,
                                                   type_ptr_indptr, type_ptr_data, is_csr=True)
            nrow = len(csr.indptr) - 1
            return matrices, nrow

        if predict_type == C_API_PREDICT_CONTRIB:
            return inner_predict_sparse(csr, start_iteration, num_iteration, predict_type)
        nrow = len(csr.indptr) - 1
        if nrow > MAX_INT32:
            sections = [0] + list(np.arange(start=MAX_INT32, stop=nrow, step=MAX_INT32)) + [nrow]
            # __get_num_preds() cannot work with nrow > MAX_INT32, so calculate overall number of predictions piecemeal
            n_preds = [self.__get_num_preds(start_iteration, num_iteration, i, predict_type) for i in np.diff(sections)]
            n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
            preds = np.zeros(sum(n_preds), dtype=np.float64)
            for (start_idx, end_idx), (start_idx_pred, end_idx_pred) in zip(zip(sections, sections[1:]),
                                                                            zip(n_preds_sections, n_preds_sections[1:])):
                # avoid memory consumption by arrays concatenation operations
                inner_predict(csr[start_idx:end_idx], start_iteration, num_iteration, predict_type, preds[start_idx_pred:end_idx_pred])
            return preds, nrow
        else:
            return inner_predict(csr, start_iteration, num_iteration, predict_type)

    def __pred_for_csc(self, csc, start_iteration, num_iteration, predict_type):
        """Predict for a CSC data."""
        def inner_predict_sparse(csc, start_iteration, num_iteration, predict_type):
            ptr_indptr, type_ptr_indptr, __ = c_int_array(csc.indptr)
            ptr_data, type_ptr_data, _ = c_float_array(csc.data)
            csc_indices = csc.indices.astype(np.int32, copy=False)
            matrix_type = C_API_MATRIX_TYPE_CSC
            if type_ptr_indptr == C_API_DTYPE_INT32:
                out_ptr_indptr = ctypes.POINTER(ctypes.c_int32)()
            else:
                out_ptr_indptr = ctypes.POINTER(ctypes.c_int64)()
            out_ptr_indices = ctypes.POINTER(ctypes.c_int32)()
            if type_ptr_data == C_API_DTYPE_FLOAT32:
                out_ptr_data = ctypes.POINTER(ctypes.c_float)()
            else:
                out_ptr_data = ctypes.POINTER(ctypes.c_double)()
            out_shape = np.zeros(4, dtype=np.int64)
            _safe_call(_LIB.LGBM_BoosterPredictSparseOutput(
                self.handle,
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
                c_str(self.pred_parameter),
                ctypes.c_int(matrix_type),
                out_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                ctypes.byref(out_ptr_indptr),
                ctypes.byref(out_ptr_indices),
                ctypes.byref(out_ptr_data)))
            matrices = self.__create_sparse_native(csc, out_shape, out_ptr_indptr, out_ptr_indices, out_ptr_data,
                                                   type_ptr_indptr, type_ptr_data, is_csr=False)
            nrow = csc.shape[0]
            return matrices, nrow

        nrow = csc.shape[0]
        if nrow > MAX_INT32:
            return self.__pred_for_csr(csc.tocsr(), start_iteration, num_iteration, predict_type)
        if predict_type == C_API_PREDICT_CONTRIB:
            return inner_predict_sparse(csc, start_iteration, num_iteration, predict_type)
        n_preds = self.__get_num_preds(start_iteration, num_iteration, nrow, predict_type)
        preds = np.zeros(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csc.data)

        assert csc.shape[0] <= MAX_INT32
        csc_indices = csc.indices.astype(np.int32, copy=False)

        _safe_call(_LIB.LGBM_BoosterPredictForCSC(
            self.handle,
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
            c_str(self.pred_parameter),
            ctypes.byref(out_num_preds),
            preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError("Wrong length for predict results")
        return preds, nrow

    def current_iteration(self):
        """Get the index of the current iteration.

        Returns
        -------
        cur_iter : int
            The index of the current iteration.
        """
        out_cur_iter = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(
            self.handle,
            ctypes.byref(out_cur_iter)))
        return out_cur_iter.value


class Dataset:
    """Dataset in LightGBM."""

    def __init__(self, data, label=None, reference=None,
                 weight=None, group=None, init_score=None, silent=False,
                 feature_name='auto', categorical_feature='auto', params=None,
                 free_raw_data=True, category_encoders=None):
        """Initialize Dataset.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse or list of numpy arrays
            Data source of Dataset.
            If string, it represents the path to txt file.
        label : list, numpy 1-D array, pandas Series / one-column DataFrame or None, optional (default=None)
            Label of the data.
        reference : Dataset or None, optional (default=None)
            If this is Dataset for validation, training data should be used as reference.
        weight : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Weight for each instance.
        group : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
        init_score : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Init score for Dataset.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        feature_name : list of strings or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of strings or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
        params : dict or None, optional (default=None)
            Other parameters for Dataset.
        free_raw_data : bool, optional (default=True)
            If True, raw data is freed after constructing inner Dataset.
        category_encoders : string, optional (default='raw')
            ways to encode categorical features into numerical values, separated by comma, currently supports:
            1. target[:prior], where `prior` is a real number used to smooth the calculation of encoded values
            target[:prior] is calculated as: (sum_label + prior * prior_weight) / (count + prior_weight)
            if the prior value is missing, use the label mean of training data as default prior
            2. count, the count of the categorical feature value in the dataset
            3. raw, the dynamic encoding method which encodes categorical values with sum_gradients / sum_hessians per leaf per iteration
            for example "target:0.5,target:0.0:count will convert each categorical feature into 3 numerical features, with the 3 different ways separated by ','.
            when category_encoders is empty, we use `raw` by default
            the numbers and names of features will be changed when category_encoders is not `raw`
            suppose the original name of a feature is `NAME`, the naming rules of its target and count encoding features are:
            1. for the encoder `target` (without user specified prior), it will be named as `NAME_label_mean_prior_target_encoding_<label_mean>`
            2. for the encoder `target:<prior>` (with user specified prior), it will be named as `NAME_target_encoding_<prior>`
            3. for the encoder `count`, it will be named as `NAME_count_encoding`
            Use get_feature_name() of python Booster or feature_name() of python Dataset after training to get the actual feature names used when category_encoders is set.
            When category_encoders==None, it is equivalent to 'raw'.
        """
        self.handle = None
        self.data = data
        self.label = label
        self.reference = reference
        self.weight = weight
        self.group = group
        self.init_score = init_score
        self.silent = silent
        self.feature_name = feature_name

        self.categorical_feature = categorical_feature
        self.category_encoders = category_encoders
        self._extract_categorical_info_from_params(params)

        self.params = {} if params is None else deepcopy(params)

        self.free_raw_data = free_raw_data
        self.used_indices = None
        self.need_slice = True
        self._predictor = None
        self.pandas_categorical = None
        self.params_back_up = None
        self.feature_penalty = None
        self.monotone_constraints = None
        self.version = 0

    def __del__(self):
        try:
            self._free_handle()
        except AttributeError:
            pass

    def _extract_categorical_info_from_params(self, params):
        if params is not None:
            categorical_feature_from_params = None
            category_encoders_from_params = None
            if isinstance(params, dict):
                for cat_alias in _ConfigAliases.get("categorical_feature"):
                    if cat_alias in params:
                        categorical_feature_from_params = params.pop(cat_alias)
                if "category_encoders" in params:
                    category_encoders_from_params = params.pop("category_encoders")
            if categorical_feature_from_params is not None:
                if self.categorical_feature == 'auto' or self.categorical_feature is None:
                    if isinstance(categorical_feature_from_params, int):
                        categorical_feature_from_params = [categorical_feature_from_params]
                    self.categorical_feature = categorical_feature_from_params
                elif self.categorical_feature != categorical_feature_from_params:
                    warnings.warn("categorical_feature {0} in params will be ignored, using {1}".format(
                        categorical_feature_from_params, self.categorical_feature))
            if category_encoders_from_params is not None:
                if self.category_encoders is None:
                    self.category_encoders = category_encoders_from_params
                elif self.category_encoders != category_encoders_from_params:
                    warnings.warn("category_encoders {0} in params will be ignored, using {1}".format(
                        category_encoders_from_params, self.category_encoders))

    def get_params(self):
        """Get the used parameters in the Dataset.

        Returns
        -------
        params : dict or None
            The used parameters in this Dataset object.
        """
        if self.params is not None:
            # no min_data, nthreads and verbose in this function
            dataset_params = _ConfigAliases.get("bin_construct_sample_cnt",
                                                "categorical_feature",
                                                "category_encoders",
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
                                                "two_round",
                                                "use_missing",
                                                "weight_column",
                                                "zero_as_missing")
            return {k: v for k, v in self.params.items() if k in dataset_params}

    def _free_handle(self):
        if self.handle is not None:
            _safe_call(_LIB.LGBM_DatasetFree(self.handle))
            self.handle = None
        self.need_slice = True
        if self.used_indices is not None:
            self.data = None
        return self

    def _set_init_score_by_predictor(self, predictor, data, used_indices=None):
        data_has_header = False
        if isinstance(data, str):
            # check data has header or not
            data_has_header = any(self.params.get(alias, False) for alias in _ConfigAliases.get("header"))
        num_data = self.num_data()
        if predictor is not None:
            init_score = predictor.predict(data,
                                           raw_score=True,
                                           data_has_header=data_has_header,
                                           is_reshape=False)
            if used_indices is not None:
                assert not self.need_slice
                if isinstance(data, str):
                    sub_init_score = np.zeros(num_data * predictor.num_class, dtype=np.float32)
                    assert num_data == len(used_indices)
                    for i in range(len(used_indices)):
                        for j in range(predictor.num_class):
                            sub_init_score[i * predictor.num_class + j] = init_score[used_indices[i] * predictor.num_class + j]
                    init_score = sub_init_score
            if predictor.num_class > 1:
                # need to regroup init_score
                new_init_score = np.zeros(init_score.size, dtype=np.float32)
                for i in range(num_data):
                    for j in range(predictor.num_class):
                        new_init_score[j * num_data + i] = init_score[i * predictor.num_class + j]
                init_score = new_init_score
        elif self.init_score is not None:
            init_score = np.zeros(self.init_score.shape, dtype=np.float32)
        else:
            return self
        self.set_init_score(init_score)

    def _lazy_init(self, data, label=None, reference=None,
                   weight=None, group=None, init_score=None, predictor=None,
                   silent=False, feature_name='auto',
                   categorical_feature='auto', params=None):
        if data is None:
            self.handle = None
            return self
        if reference is not None:
            self.pandas_categorical = reference.pandas_categorical
            if self.category_encoders is not None and self.category_encoders != reference.category_encoders:
                warnings.warn("'category_encoders' set in validation data is overridden by that of training data.")
            self.category_encoders = reference.category_encoders
            if self.categorical_feature != 'auto' and self.categorical_feature is not None and\
                    self.categorical_feature != reference.categorical_feature:
                warnings.warn("'categorical_feature' set in validation data is overridden by that of training data.")
            self.categorical_feature = reference.categorical_feature
            categorical_feature = reference.categorical_feature
        data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(data,
                                                                                             feature_name,
                                                                                             categorical_feature,
                                                                                             self.pandas_categorical)
        label = _label_from_pandas(label)

        # process for args
        params = {} if params is None else params
        args_names = (getattr(self.__class__, '_lazy_init')
                      .__code__
                      .co_varnames[:getattr(self.__class__, '_lazy_init').__code__.co_argcount])
        for key, _ in params.items():
            if key in args_names:
                _log_warning(f'{key} keyword has been found in `params` and will be ignored.\n'
                             f'Please use {key} argument of the Dataset constructor to pass this parameter.')
        # user can set verbose with params, it has higher priority
        if not any(verbose_alias in params for verbose_alias in _ConfigAliases.get("verbosity")) and silent:
            params["verbose"] = -1
        # get categorical features
        if categorical_feature is not None:
            categorical_indices = set()
            feature_dict = {}
            if feature_name is not None:
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
                        _log_warning(f'{cat_alias} in param dict is overridden.')
                        params.pop(cat_alias, None)
                params['categorical_column'] = sorted(categorical_indices)
        if self.category_encoders is not None:
            if "category_encoders" in params:
                warnings.warn("category_encoders in param dict is overridden.")
                params.pop("category_encoders", None)
            params["category_encoders"] = self.category_encoders
        params_str = param_dict_to_str(params)
        self.params = params
        # process for reference dataset
        ref_dataset = None
        if isinstance(reference, Dataset):
            ref_dataset = reference.construct().handle
        elif reference is not None:
            raise TypeError('Reference dataset should be None or dataset instance')
        # start construct data
        if isinstance(data, str):
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_DatasetCreateFromFile(
                c_str(data),
                c_str(params_str),
                ref_dataset,
                ctypes.byref(self.handle)))
        elif isinstance(data, scipy.sparse.csr_matrix):
            self.__init_from_csr(data, label, params_str, ref_dataset)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self.__init_from_csc(data, label, params_str, ref_dataset)
        elif isinstance(data, np.ndarray):
            self.__init_from_np2d(data, label, params_str, ref_dataset)
        elif isinstance(data, list) and len(data) > 0 and all(isinstance(x, np.ndarray) for x in data):
            self.__init_from_list_np2d(data, label, params_str, ref_dataset)
        elif isinstance(data, dt_DataTable):
            self.__init_from_np2d(data.to_numpy(), label, params_str, ref_dataset)
        else:
            try:
                csr = scipy.sparse.csr_matrix(data)
                self.__init_from_csr(csr, label, params_str, ref_dataset)
            except BaseException:
                raise TypeError(f'Cannot initialize Dataset from {type(data).__name__}')
        if label is not None:
            self.set_label(label)
        if self.get_label() is None:
            raise ValueError("Label should not be None")
        if weight is not None:
            self.set_weight(weight)
        if group is not None:
            self.set_group(group)
        if isinstance(predictor, _InnerPredictor):
            if self._predictor is None and init_score is not None:
                _log_warning("The init_score will be overridden by the prediction of init_model.")
            self._set_init_score_by_predictor(predictor, data)
        elif init_score is not None:
            self.set_init_score(init_score)
        elif predictor is not None:
            raise TypeError(f'Wrong predictor type {type(predictor).__name__}')
        # set feature names
        return self.set_feature_name(feature_name)

    def __init_from_np2d(self, mat, label, params_str, ref_dataset):
        """Initialize data from a 2-D numpy matrix."""
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')

        self.handle = ctypes.c_void_p()
        if mat.dtype == np.float32 or mat.dtype == np.float64:
            data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
        else:  # change non-float data to float data, need to copy
            data = np.array(mat.reshape(mat.size), dtype=np.float32)

        ptr_data, type_ptr_data, _ = c_float_array(data)
        ptr_label, _ = c_float_label(label)
        _safe_call(_LIB.LGBM_DatasetCreateFromMat(
            ptr_data,
            ptr_label,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int32(mat.shape[0]),
            ctypes.c_int32(mat.shape[1]),
            ctypes.c_int(C_API_IS_ROW_MAJOR),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))
        return self

    def __init_from_list_np2d(self, mats, label, params_str, ref_dataset):
        """Initialize data from a list of 2-D numpy matrices."""
        ncol = mats[0].shape[1]
        nrow = np.zeros((len(mats),), np.int32)
        if mats[0].dtype == np.float64:
            ptr_data = (ctypes.POINTER(ctypes.c_double) * len(mats))()
        else:
            ptr_data = (ctypes.POINTER(ctypes.c_float) * len(mats))()

        holders = []
        type_ptr_data = None

        for i, mat in enumerate(mats):
            if len(mat.shape) != 2:
                raise ValueError('Input numpy.ndarray must be 2 dimensional')

            if mat.shape[1] != ncol:
                raise ValueError('Input arrays must have same number of columns')

            nrow[i] = mat.shape[0]

            if mat.dtype == np.float32 or mat.dtype == np.float64:
                mats[i] = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
            else:  # change non-float data to float data, need to copy
                mats[i] = np.array(mat.reshape(mat.size), dtype=np.float32)

            chunk_ptr_data, chunk_type_ptr_data, holder = c_float_array(mats[i])
            if type_ptr_data is not None and chunk_type_ptr_data != type_ptr_data:
                raise ValueError('Input chunks must have same type')
            ptr_data[i] = chunk_ptr_data
            type_ptr_data = chunk_type_ptr_data
            holders.append(holder)

        ptr_label, _ = c_float_label(label)

        self.handle = ctypes.c_void_p()
        _safe_call(_LIB.LGBM_DatasetCreateFromMats(
            ctypes.c_int32(len(mats)),
            ctypes.cast(ptr_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
            ptr_label,
            ctypes.c_int(type_ptr_data),
            nrow.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(ncol),
            ctypes.c_int(C_API_IS_ROW_MAJOR),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))
        return self

    def __init_from_csr(self, csr, label, params_str, ref_dataset):
        """Initialize data from a CSR matrix."""
        if len(csr.indices) != len(csr.data):
            raise ValueError(f'Length mismatch: {len(csr.indices)} vs {len(csr.data)}')
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csr.data)
        ptr_label, _ = c_float_label(label)

        assert csr.shape[1] <= MAX_INT32
        csr_indices = csr.indices.astype(np.int32, copy=False)

        _safe_call(_LIB.LGBM_DatasetCreateFromCSR(
            ptr_indptr,
            ctypes.c_int(type_ptr_indptr),
            csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ptr_label,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csr.indptr)),
            ctypes.c_int64(len(csr.data)),
            ctypes.c_int64(csr.shape[1]),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))
        return self

    def __init_from_csc(self, csc, label, params_str, ref_dataset):
        """Initialize data from a CSC matrix."""
        if len(csc.indices) != len(csc.data):
            raise ValueError(f'Length mismatch: {len(csc.indices)} vs {len(csc.data)}')
        self.handle = ctypes.c_void_p()

        ptr_indptr, type_ptr_indptr, __ = c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = c_float_array(csc.data)

        ptr_label, _ = c_float_label(label)

        assert csc.shape[0] <= MAX_INT32
        csc_indices = csc.indices.astype(np.int32, copy=False)

        _safe_call(_LIB.LGBM_DatasetCreateFromCSC(
            ptr_indptr,
            ctypes.c_int(type_ptr_indptr),
            csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ptr_label,
            ctypes.c_int(type_ptr_data),
            ctypes.c_int64(len(csc.indptr)),
            ctypes.c_int64(len(csc.data)),
            ctypes.c_int64(csc.shape[0]),
            c_str(params_str),
            ref_dataset,
            ctypes.byref(self.handle)))
        return self

    def construct(self):
        """Lazy init.

        Returns
        -------
        self : Dataset
            Constructed Dataset object.
        """
        if self.handle is None:
            if self.reference is not None:
                reference_params = self.reference.get_params()
                reference_params.pop("category_encoders", None)
                for cat_alias in _ConfigAliases.get("categorical_feature"):
                    reference_params.pop(cat_alias, None)
                self._extract_categorical_info_from_params(self.params)
                if self.get_params() != reference_params:
                    _log_warning('Overriding the parameters from Reference Dataset.')
                    self._update_params(reference_params)
                if self.used_indices is None:
                    # create valid
                    self._lazy_init(self.data, label=self.label, reference=self.reference,
                                    weight=self.weight, group=self.group,
                                    init_score=self.init_score, predictor=self._predictor,
                                    silent=self.silent, feature_name=self.feature_name, params=self.params)
                else:
                    # construct subset
                    used_indices = list_to_1d_numpy(self.used_indices, np.int32, name='used_indices')
                    assert used_indices.flags.c_contiguous
                    if self.reference.group is not None:
                        group_info = np.array(self.reference.group).astype(np.int32, copy=False)
                        _, self.group = np.unique(np.repeat(range(len(group_info)), repeats=group_info)[self.used_indices],
                                                  return_counts=True)
                    self.handle = ctypes.c_void_p()
                    params_str = param_dict_to_str(self.params)
                    _safe_call(_LIB.LGBM_DatasetGetSubset(
                        self.reference.construct().handle,
                        used_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        ctypes.c_int32(used_indices.shape[0]),
                        c_str(params_str),
                        ctypes.byref(self.handle)))
                    if not self.free_raw_data:
                        self.get_data()
                    if self.group is not None:
                        self.set_group(self.group)
                    if self.get_label() is None:
                        raise ValueError("Label should not be None.")
                    if isinstance(self._predictor, _InnerPredictor) and self._predictor is not self.reference._predictor:
                        self.get_data()
                        self._set_init_score_by_predictor(self._predictor, self.data, used_indices)
            else:
                # create train
                self._extract_categorical_info_from_params(self.params)
                self._lazy_init(self.data, label=self.label,
                                weight=self.weight, group=self.group,
                                init_score=self.init_score, predictor=self._predictor,
                                silent=self.silent, feature_name=self.feature_name,
                                categorical_feature=self.categorical_feature,
                                params=self.params)
            if self.free_raw_data:
                self.data = None
        return self

    def create_valid(self, data, label=None, weight=None, group=None,
                     init_score=None, silent=False, params=None):
        """Create validation data align with current Dataset.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse or list of numpy arrays
            Data source of Dataset.
            If string, it represents the path to txt file.
        label : list, numpy 1-D array, pandas Series / one-column DataFrame or None, optional (default=None)
            Label of the data.
        weight : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Weight for each instance.
        group : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
        init_score : list, numpy 1-D array, pandas Series or None, optional (default=None)
            Init score for Dataset.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        params : dict or None, optional (default=None)
            Other parameters for validation Dataset.

        Returns
        -------
        valid : Dataset
            Validation Dataset with reference to self.
        """
        ret = Dataset(data, label=label, reference=self,
                      weight=weight, group=group, init_score=init_score,
                      silent=silent, params=params, free_raw_data=self.free_raw_data)
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        return ret

    def subset(self, used_indices, params=None):
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
        ret = Dataset(None, reference=self, feature_name=self.feature_name,
                      categorical_feature=self.categorical_feature, params=params,
                      free_raw_data=self.free_raw_data)
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        ret.used_indices = sorted(used_indices)
        return ret

    def save_binary(self, filename):
        """Save Dataset to a binary file.

        .. note::

            Please note that `init_score` is not saved in binary file.
            If you need it, please set it again after loading Dataset.

        Parameters
        ----------
        filename : string
            Name of the output file.

        Returns
        -------
        self : Dataset
            Returns self.
        """
        _safe_call(_LIB.LGBM_DatasetSaveBinary(
            self.construct().handle,
            c_str(filename)))
        return self

    def _update_params(self, params):
        if not params:
            return self
        params = deepcopy(params)

        def update():
            if not self.params:
                self.params = params
            else:
                self.params_back_up = deepcopy(self.params)
                self.params.update(params)

        if self.handle is None:
            update()
        elif params is not None:
            ret = _LIB.LGBM_DatasetUpdateParamChecking(
                c_str(param_dict_to_str(self.params)),
                c_str(param_dict_to_str(params)))
            if ret != 0:
                # could be updated if data is not freed
                if self.data is not None:
                    update()
                    self._free_handle()
                else:
                    raise LightGBMError(_LIB.LGBM_GetLastError().decode('utf-8'))
        return self

    def _reverse_update_params(self):
        if self.handle is None:
            self.params = deepcopy(self.params_back_up)
            self.params_back_up = None
        return self

    def set_field(self, field_name, data):
        """Set property into the Dataset.

        Parameters
        ----------
        field_name : string
            The field name of the information.
        data : list, numpy 1-D array, pandas Series or None
            The array of data to be set.

        Returns
        -------
        self : Dataset
            Dataset with set property.
        """
        if self.handle is None:
            raise Exception(f"Cannot set {field_name} before construct dataset")
        if data is None:
            # set to None
            _safe_call(_LIB.LGBM_DatasetSetField(
                self.handle,
                c_str(field_name),
                None,
                ctypes.c_int(0),
                ctypes.c_int(FIELD_TYPE_MAPPER[field_name])))
            return self
        dtype = np.float32
        if field_name == 'group':
            dtype = np.int32
        elif field_name == 'init_score':
            dtype = np.float64
        data = list_to_1d_numpy(data, dtype, name=field_name)
        if data.dtype == np.float32 or data.dtype == np.float64:
            ptr_data, type_data, _ = c_float_array(data)
        elif data.dtype == np.int32:
            ptr_data, type_data, _ = c_int_array(data)
        else:
            raise TypeError(f"Expected np.float32/64 or np.int32, met type({data.dtype})")
        if type_data != FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("Input type error for set_field")
        _safe_call(_LIB.LGBM_DatasetSetField(
            self.handle,
            c_str(field_name),
            ptr_data,
            ctypes.c_int(len(data)),
            ctypes.c_int(type_data)))
        self.version += 1
        return self

    def get_field(self, field_name):
        """Get property from the Dataset.

        Parameters
        ----------
        field_name : string
            The field name of the information.

        Returns
        -------
        info : numpy array
            A numpy array with information from the Dataset.
        """
        if self.handle is None:
            raise Exception(f"Cannot get {field_name} before construct Dataset")
        tmp_out_len = ctypes.c_int(0)
        out_type = ctypes.c_int(0)
        ret = ctypes.POINTER(ctypes.c_void_p)()
        _safe_call(_LIB.LGBM_DatasetGetField(
            self.handle,
            c_str(field_name),
            ctypes.byref(tmp_out_len),
            ctypes.byref(ret),
            ctypes.byref(out_type)))
        if out_type.value != FIELD_TYPE_MAPPER[field_name]:
            raise TypeError("Return type error for get_field")
        if tmp_out_len.value == 0:
            return None
        if out_type.value == C_API_DTYPE_INT32:
            return cint32_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_int32)), tmp_out_len.value)
        elif out_type.value == C_API_DTYPE_FLOAT32:
            return cfloat32_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_float)), tmp_out_len.value)
        elif out_type.value == C_API_DTYPE_FLOAT64:
            return cfloat64_array_to_numpy(ctypes.cast(ret, ctypes.POINTER(ctypes.c_double)), tmp_out_len.value)
        else:
            raise TypeError("Unknown type")

    def set_categorical_feature(self, categorical_feature):
        """Set categorical features.

        Parameters
        ----------
        categorical_feature : list of int or strings
            Names or indices of categorical features.

        Returns
        -------
        self : Dataset
            Dataset with set categorical features.
        """
        if self.categorical_feature == categorical_feature:
            return self
        if categorical_feature is None:
            return self
        if categorical_feature == 'auto' and self.categorical_feature is not None:
            return self
        if self.data is not None:
            if self.categorical_feature is None:
                self.categorical_feature = categorical_feature
                return self._free_handle()
            elif categorical_feature == 'auto':
                _log_warning('Using categorical_feature in Dataset.')
                return self
            else:
                _log_warning('categorical_feature in Dataset is overridden.\n'
                             f'New categorical_feature is {sorted(list(categorical_feature))}')
                self.categorical_feature = categorical_feature
                return self._free_handle()
        else:
            raise LightGBMError("Cannot set categorical feature after freed raw data, "
                                "set free_raw_data=False when construct Dataset to avoid this.")

    def set_category_encoders(self, category_encoders):
        """Set categorical feature converters.

        Parameters
        ----------
        category_encoders : string, optional (default='raw')
            ways to encode categorical features into numerical values, separated by comma, currently supports:
            1. target[:prior], where `prior` is a real number used to smooth the calculation of encoded values
            target[:prior] is calculated as: (sum_label + prior * prior_weight) / (count + prior_weight)
            if the prior value is missing, use the label mean of training data as default prior
            2. count, the count of the categorical feature value in the dataset
            3. raw, the dynamic encoding method which encodes categorical values with sum_gradients / sum_hessians per leaf per iteration
            for example "target:0.5,target:0.0:count will convert each categorical feature into 3 numerical features, with the 3 different ways separated by ','.
            when category_encoders is empty, we use `raw` by default
            the numbers and names of features will be changed when category_encoders is not `raw`
            suppose the original name of a feature is `NAME`, the naming rules of its target and count encoding features are:
            1. for the encoder `target` (without user specified prior), it will be named as `NAME_label_mean_prior_target_encoding_<label_mean>`
            2. for the encoder `target:<prior>` (with user specified prior), it will be named as `NAME_target_encoding_<prior>`
            3. for the encoder `count`, it will be named as `NAME_count_encoding`
            Use get_feature_name() of python Booster or feature_name() of python Dataset after training to get the actual feature names used when category_encoders is set.
            When category_encoders==None, it is equivalent to 'raw'.

        Returns
        -------
        self : Dataset
            Dataset with set categorical converters
        """
        if category_encoders is None:
            return self
        if self.category_encoders == category_encoders:
            return self
        if self.data is not None:
            if self.category_encoders is None:
                self.category_encoders = category_encoders
                return self._free_handle()
            else:
                warnings.warn('category_encoders in Dataset constructor is overridden.\n'
                              'New category_encoders is {}'.format(category_encoders))
                self.category_encoders = category_encoders
                return self._free_handle()
        else:
            raise LightGBMError("Cannot set categorical feature converters after freed raw data, "
                                "set free_raw_data=False when construct Dataset to avoid this.")

    def _set_predictor(self, predictor):
        """Set predictor for continued training.

        It is not recommended for user to call this function.
        Please use init_model argument in engine.train() or engine.cv() instead.
        """
        if predictor is self._predictor and (predictor is None or predictor.current_iteration() == self._predictor.current_iteration()):
            return self
        if self.handle is None:
            self._predictor = predictor
        elif self.data is not None:
            self._predictor = predictor
            self._set_init_score_by_predictor(self._predictor, self.data)
        elif self.used_indices is not None and self.reference is not None and self.reference.data is not None:
            self._predictor = predictor
            self._set_init_score_by_predictor(self._predictor, self.reference.data, self.used_indices)
        else:
            raise LightGBMError("Cannot set predictor after freed raw data, "
                                "set free_raw_data=False when construct Dataset to avoid this.")
        return self

    def set_reference(self, reference):
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
        if self.categorical_feature is not None and self.categorical_feature != 'auto' and \
                self.categorical_feature != reference.categorical_feature:
            warnings.warn("'categorical_feature' set in validation data is overridden by that of training data.")
        self.categorical_feature = reference.categorical_feature
        if self.category_encoders is not None and self.category_encoders != reference.category_encoders:
            warnings.warn("'category_encoders' set in validation data is overridden by that of training data.")
        self.category_encoders = reference.category_encoders
        self.set_feature_name(reference.feature_name) \
            ._set_predictor(reference._predictor)
        # we're done if self and reference share a common upstrem reference
        if self.get_ref_chain().intersection(reference.get_ref_chain()):
            return self
        if self.data is not None:
            self.reference = reference
            return self._free_handle()
        else:
            raise LightGBMError("Cannot set reference after freed raw data, "
                                "set free_raw_data=False when construct Dataset to avoid this.")

    def set_feature_name(self, feature_name):
        """Set feature name.

        Parameters
        ----------
        feature_name : list of strings
            Feature names.

        Returns
        -------
        self : Dataset
            Dataset with set feature name.
        """
        if feature_name != 'auto':
            self.feature_name = feature_name
        if self.handle is not None and feature_name is not None and feature_name != 'auto':
            if len(feature_name) != self.num_original_feature():
                raise ValueError(f"Length of feature_name({len(feature_name)}) and num_feature({self.num_original_feature()}) don't match")
            c_feature_name = [c_str(name) for name in feature_name]
            _safe_call(_LIB.LGBM_DatasetSetFeatureNames(
                self.handle,
                c_array(ctypes.c_char_p, c_feature_name),
                ctypes.c_int(len(feature_name))))
        return self

    def set_label(self, label):
        """Set label of Dataset.

        Parameters
        ----------
        label : list, numpy 1-D array, pandas Series / one-column DataFrame or None
            The label information to be set into Dataset.

        Returns
        -------
        self : Dataset
            Dataset with set label.
        """
        self.label = label
        if self.handle is not None:
            label = list_to_1d_numpy(_label_from_pandas(label), name='label')
            self.set_field('label', label)
            self.label = self.get_field('label')  # original values can be modified at cpp side
        return self

    def set_weight(self, weight):
        """Set weight of each instance.

        Parameters
        ----------
        weight : list, numpy 1-D array, pandas Series or None
            Weight to be set for each data point.

        Returns
        -------
        self : Dataset
            Dataset with set weight.
        """
        if weight is not None and np.all(weight == 1):
            weight = None
        self.weight = weight
        if self.handle is not None and weight is not None:
            weight = list_to_1d_numpy(weight, name='weight')
            self.set_field('weight', weight)
            self.weight = self.get_field('weight')  # original values can be modified at cpp side
        return self

    def set_init_score(self, init_score):
        """Set init score of Booster to start from.

        Parameters
        ----------
        init_score : list, numpy 1-D array, pandas Series or None
            Init score for Booster.

        Returns
        -------
        self : Dataset
            Dataset with set init score.
        """
        self.init_score = init_score
        if self.handle is not None and init_score is not None:
            init_score = list_to_1d_numpy(init_score, np.float64, name='init_score')
            self.set_field('init_score', init_score)
            self.init_score = self.get_field('init_score')  # original values can be modified at cpp side
        return self

    def set_group(self, group):
        """Set group size of Dataset (used for ranking).

        Parameters
        ----------
        group : list, numpy 1-D array, pandas Series or None
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

        Returns
        -------
        self : Dataset
            Dataset with set group.
        """
        self.group = group
        if self.handle is not None and group is not None:
            group = list_to_1d_numpy(group, np.int32, name='group')
            self.set_field('group', group)
        return self

    def get_feature_name(self):
        """Get the names of columns (features) in the Dataset.

        Returns
        -------
        feature_names : list
            The names of columns (features) in the Dataset.
        """
        if self.handle is None:
            raise LightGBMError("Cannot get feature_name before construct dataset")
        num_feature = self.num_feature()
        tmp_out_len = ctypes.c_int(0)
        reserved_string_buffer_size = 255
        required_string_buffer_size = ctypes.c_size_t(0)
        string_buffers = [ctypes.create_string_buffer(reserved_string_buffer_size) for _ in range(num_feature)]
        ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))
        _safe_call(_LIB.LGBM_DatasetGetFeatureNames(
            self.handle,
            ctypes.c_int(num_feature),
            ctypes.byref(tmp_out_len),
            ctypes.c_size_t(reserved_string_buffer_size),
            ctypes.byref(required_string_buffer_size),
            ptr_string_buffers))
        if num_feature != tmp_out_len.value:
            raise ValueError("Length of feature names doesn't equal with num_feature")
        actual_string_buffer_size = required_string_buffer_size.value
        # if buffer length is not long enough, reallocate buffers
        if reserved_string_buffer_size < actual_string_buffer_size:
            string_buffers = [ctypes.create_string_buffer(actual_string_buffer_size) for _ in range(num_feature)]
            ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))
            _safe_call(_LIB.LGBM_DatasetGetFeatureNames(
                self.handle,
                ctypes.c_int(num_feature),
                ctypes.byref(tmp_out_len),
                ctypes.c_size_t(actual_string_buffer_size),
                ctypes.byref(required_string_buffer_size),
                ptr_string_buffers))
        return [string_buffers[i].value.decode('utf-8') for i in range(num_feature)]

    def get_label(self):
        """Get the label of the Dataset.

        Returns
        -------
        label : numpy array or None
            The label information from the Dataset.
        """
        if self.label is None:
            self.label = self.get_field('label')
        return self.label

    def get_weight(self):
        """Get the weight of the Dataset.

        Returns
        -------
        weight : numpy array or None
            Weight for each data point from the Dataset.
        """
        if self.weight is None:
            self.weight = self.get_field('weight')
        return self.weight

    def get_init_score(self):
        """Get the initial score of the Dataset.

        Returns
        -------
        init_score : numpy array or None
            Init score of Booster.
        """
        if self.init_score is None:
            self.init_score = self.get_field('init_score')
        return self.init_score

    def get_data(self):
        """Get the raw data of the Dataset.

        Returns
        -------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame, scipy.sparse, list of numpy arrays or None
            Raw data used in the Dataset construction.
        """
        if self.handle is None:
            raise Exception("Cannot get data before construct Dataset")
        if self.need_slice and self.used_indices is not None and self.reference is not None:
            self.data = self.reference.data
            if self.data is not None:
                if isinstance(self.data, np.ndarray) or scipy.sparse.issparse(self.data):
                    self.data = self.data[self.used_indices, :]
                elif isinstance(self.data, pd_DataFrame):
                    self.data = self.data.iloc[self.used_indices].copy()
                elif isinstance(self.data, dt_DataTable):
                    self.data = self.data[self.used_indices, :]
                else:
                    _log_warning(f"Cannot subset {type(self.data).__name__} type of raw data.\n"
                                 "Returning original raw data")
            self.need_slice = False
        if self.data is None:
            raise LightGBMError("Cannot call `get_data` after freed raw data, "
                                "set free_raw_data=False when construct Dataset to avoid this.")
        return self.data

    def get_group(self):
        """Get the group of the Dataset.

        Returns
        -------
        group : numpy array or None
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
        """
        if self.group is None:
            self.group = self.get_field('group')
            if self.group is not None:
                # group data from LightGBM is boundaries data, need to convert to group size
                self.group = np.diff(self.group)
        return self.group

    def num_data(self):
        """Get the number of rows in the Dataset.

        Returns
        -------
        number_of_rows : int
            The number of rows in the Dataset.
        """
        if self.handle is not None:
            ret = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_DatasetGetNumData(self.handle,
                                                   ctypes.byref(ret)))
            return ret.value
        else:
            raise LightGBMError("Cannot get num_data before construct dataset")

    def num_feature(self):
        """Get the number of columns (features) in the Dataset.

        Returns
        -------
        number_of_columns : int
            The number of columns (features) in the Dataset.
        """
        if self.handle is not None:
            ret = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_DatasetGetNumFeature(self.handle,
                                                      ctypes.byref(ret)))
            return ret.value
        else:
            raise LightGBMError("Cannot get num_feature before construct dataset")

    def num_original_feature(self):
        """Get the number of original columns (features) in the Dataset.

        Returns
        -------
        number_of_columns : int
            The number of columns (features) in the Dataset.
        """
        if self.handle is not None:
            ret = ctypes.c_int()
            _safe_call(_LIB.LGBM_DatasetGetNumOriginalFeature(self.handle,
                                                              ctypes.byref(ret)))
            return ret.value
        else:
            raise LightGBMError("Cannot get num_feature before construct dataset")

    def get_ref_chain(self, ref_limit=100):
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
        ref_chain = set()
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

    def add_features_from(self, other):
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
        if self.handle is None or other.handle is None:
            raise ValueError('Both source and target Datasets must be constructed before adding features')
        _safe_call(_LIB.LGBM_DatasetAddFeaturesFrom(self.handle, other.handle))
        was_none = self.data is None
        old_self_data_type = type(self.data).__name__
        if other.data is None:
            self.data = None
        elif self.data is not None:
            if isinstance(self.data, np.ndarray):
                if isinstance(other.data, np.ndarray):
                    self.data = np.hstack((self.data, other.data))
                elif scipy.sparse.issparse(other.data):
                    self.data = np.hstack((self.data, other.data.toarray()))
                elif isinstance(other.data, pd_DataFrame):
                    self.data = np.hstack((self.data, other.data.values))
                elif isinstance(other.data, dt_DataTable):
                    self.data = np.hstack((self.data, other.data.to_numpy()))
                else:
                    self.data = None
            elif scipy.sparse.issparse(self.data):
                sparse_format = self.data.getformat()
                if isinstance(other.data, np.ndarray) or scipy.sparse.issparse(other.data):
                    self.data = scipy.sparse.hstack((self.data, other.data), format=sparse_format)
                elif isinstance(other.data, pd_DataFrame):
                    self.data = scipy.sparse.hstack((self.data, other.data.values), format=sparse_format)
                elif isinstance(other.data, dt_DataTable):
                    self.data = scipy.sparse.hstack((self.data, other.data.to_numpy()), format=sparse_format)
                else:
                    self.data = None
            elif isinstance(self.data, pd_DataFrame):
                if not PANDAS_INSTALLED:
                    raise LightGBMError("Cannot add features to DataFrame type of raw data "
                                        "without pandas installed. "
                                        "Install pandas and restart your session.")
                if isinstance(other.data, np.ndarray):
                    self.data = concat((self.data, pd_DataFrame(other.data)),
                                       axis=1, ignore_index=True)
                elif scipy.sparse.issparse(other.data):
                    self.data = concat((self.data, pd_DataFrame(other.data.toarray())),
                                       axis=1, ignore_index=True)
                elif isinstance(other.data, pd_DataFrame):
                    self.data = concat((self.data, other.data),
                                       axis=1, ignore_index=True)
                elif isinstance(other.data, dt_DataTable):
                    self.data = concat((self.data, pd_DataFrame(other.data.to_numpy())),
                                       axis=1, ignore_index=True)
                else:
                    self.data = None
            elif isinstance(self.data, dt_DataTable):
                if isinstance(other.data, np.ndarray):
                    self.data = dt_DataTable(np.hstack((self.data.to_numpy(), other.data)))
                elif scipy.sparse.issparse(other.data):
                    self.data = dt_DataTable(np.hstack((self.data.to_numpy(), other.data.toarray())))
                elif isinstance(other.data, pd_DataFrame):
                    self.data = dt_DataTable(np.hstack((self.data.to_numpy(), other.data.values)))
                elif isinstance(other.data, dt_DataTable):
                    self.data = dt_DataTable(np.hstack((self.data.to_numpy(), other.data.to_numpy())))
                else:
                    self.data = None
            else:
                self.data = None
        if self.data is None:
            err_msg = (f"Cannot add features from {type(other.data).__name__} type of raw data to "
                       f"{old_self_data_type} type of raw data.\n")
            err_msg += ("Set free_raw_data=False when construct Dataset to avoid this"
                        if was_none else "Freeing raw data")
            _log_warning(err_msg)
        self.feature_name = self.get_feature_name()
        _log_warning("Reseting categorical features.\n"
                     "You can set new categorical features via ``set_categorical_feature`` method")
        self.categorical_feature = "auto"
        self.pandas_categorical = None
        return self

    def _dump_text(self, filename):
        """Save Dataset to a text file.

        This format cannot be loaded back in by LightGBM, but is useful for debugging purposes.

        Parameters
        ----------
        filename : string
            Name of the output file.

        Returns
        -------
        self : Dataset
            Returns self.
        """
        _safe_call(_LIB.LGBM_DatasetDumpText(
            self.construct().handle,
            c_str(filename)))
        return self


class Booster:
    """Booster in LightGBM."""

    def __init__(self, params=None, train_set=None, model_file=None, model_str=None, silent=False):
        """Initialize the Booster.

        Parameters
        ----------
        params : dict or None, optional (default=None)
            Parameters for Booster.
        train_set : Dataset or None, optional (default=None)
            Training dataset.
        model_file : string or None, optional (default=None)
            Path to the model file.
        model_str : string or None, optional (default=None)
            Model will be loaded from this string.
        silent : bool, optional (default=False)
            Whether to print messages during construction.
        """
        self.handle = None
        self.network = False
        self.__need_reload_eval_info = True
        self._train_data_name = "training"
        self.__attr = {}
        self.__set_objective_to_none = False
        self.best_iteration = -1
        self.best_score = {}
        params = {} if params is None else deepcopy(params)
        # user can set verbose with params, it has higher priority
        if not any(verbose_alias in params for verbose_alias in _ConfigAliases.get("verbosity")) and silent:
            params["verbose"] = -1
        if train_set is not None:
            # Training task
            if not isinstance(train_set, Dataset):
                raise TypeError(f'Training data should be Dataset instance, met {type(train_set).__name__}')
            params = _choose_param_value(
                main_param_name="machines",
                params=params,
                default_value=None
            )
            # if "machines" is given, assume user wants to do distributed learning, and set up network
            if params["machines"] is None:
                params.pop("machines", None)
            else:
                machines = params["machines"]
                if isinstance(machines, str):
                    num_machines_from_machine_list = len(machines.split(','))
                elif isinstance(machines, (list, set)):
                    num_machines_from_machine_list = len(machines)
                    machines = ','.join(machines)
                else:
                    raise ValueError("Invalid machines in params.")

                params = _choose_param_value(
                    main_param_name="num_machines",
                    params=params,
                    default_value=num_machines_from_machine_list
                )
                params = _choose_param_value(
                    main_param_name="local_listen_port",
                    params=params,
                    default_value=12400
                )
                self.set_network(
                    machines=machines,
                    local_listen_port=params["local_listen_port"],
                    listen_time_out=params.get("time_out", 120),
                    num_machines=params["num_machines"]
                )
            # construct booster object
            train_set.construct()
            # copy the parameters from train_set
            params.update(train_set.get_params())
            params_str = param_dict_to_str(params)
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_BoosterCreate(
                train_set.handle,
                c_str(params_str),
                ctypes.byref(self.handle)))
            # save reference to data
            self.train_set = train_set
            self.valid_sets = []
            self.name_valid_sets = []
            self.__num_dataset = 1
            self.__init_predictor = train_set._predictor
            if self.__init_predictor is not None:
                _safe_call(_LIB.LGBM_BoosterMerge(
                    self.handle,
                    self.__init_predictor.handle))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.__num_class = out_num_class.value
            # buffer for inner predict
            self.__inner_predict_buffer = [None]
            self.__is_predicted_cur_iter = [False]
            self.__get_eval_info()
            self.pandas_categorical = train_set.pandas_categorical
            self.train_set_version = train_set.version
        elif model_file is not None:
            # Prediction task
            out_num_iterations = ctypes.c_int(0)
            self.handle = ctypes.c_void_p()
            _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(
                c_str(model_file),
                ctypes.byref(out_num_iterations),
                ctypes.byref(self.handle)))
            out_num_class = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetNumClasses(
                self.handle,
                ctypes.byref(out_num_class)))
            self.__num_class = out_num_class.value
            self.pandas_categorical = _load_pandas_categorical(file_name=model_file)
        elif model_str is not None:
            self.model_from_string(model_str, not silent)
        else:
            raise TypeError('Need at least one training dataset or model file or model string '
                            'to create Booster instance')
        self.params = params

    def __del__(self):
        try:
            if self.network:
                self.free_network()
        except AttributeError:
            pass
        try:
            if self.handle is not None:
                _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        except AttributeError:
            pass

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        model_str = self.model_to_string(num_iteration=-1)
        booster = Booster(model_str=model_str)
        return booster

    def __getstate__(self):
        this = self.__dict__.copy()
        handle = this['handle']
        this.pop('train_set', None)
        this.pop('valid_sets', None)
        if handle is not None:
            this["handle"] = self.model_to_string(num_iteration=-1)
        return this

    def __setstate__(self, state):
        model_str = state.get('handle', None)
        if model_str is not None:
            handle = ctypes.c_void_p()
            out_num_iterations = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterLoadModelFromString(
                c_str(model_str),
                ctypes.byref(out_num_iterations),
                ctypes.byref(handle)))
            state['handle'] = handle
        self.__dict__.update(state)

    def free_dataset(self):
        """Free Booster's Datasets.

        Returns
        -------
        self : Booster
            Booster without Datasets.
        """
        self.__dict__.pop('train_set', None)
        self.__dict__.pop('valid_sets', None)
        self.__num_dataset = 0
        return self

    def _free_buffer(self):
        self.__inner_predict_buffer = []
        self.__is_predicted_cur_iter = []
        return self

    def set_network(
        self,
        machines: Union[List[str], Set[str], str],
        local_listen_port: int = 12400,
        listen_time_out: int = 120,
        num_machines: int = 1
    ) -> "Booster":
        """Set the network configuration.

        Parameters
        ----------
        machines : list, set or string
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
            machines = ','.join(machines)
        _safe_call(_LIB.LGBM_NetworkInit(c_str(machines),
                                         ctypes.c_int(local_listen_port),
                                         ctypes.c_int(listen_time_out),
                                         ctypes.c_int(num_machines)))
        self.network = True
        return self

    def free_network(self):
        """Free Booster's network.

        Returns
        -------
        self : Booster
            Booster with freed network.
        """
        _safe_call(_LIB.LGBM_NetworkFree())
        self.network = False
        return self

    def trees_to_dataframe(self):
        """Parse the fitted model and return in an easy-to-read pandas DataFrame.

        The returned DataFrame has the following columns.

            - ``tree_index`` : int64, which tree a node belongs to. 0-based, so a value of ``6``, for example, means "this node is in the 7th tree".
            - ``node_depth`` : int64, how far a node is from the root of the tree. The root node has a value of ``1``, its direct children are ``2``, etc.
            - ``node_index`` : string, unique identifier for a node.
            - ``left_child`` : string, ``node_index`` of the child node to the left of a split. ``None`` for leaf nodes.
            - ``right_child`` : string, ``node_index`` of the child node to the right of a split. ``None`` for leaf nodes.
            - ``parent_index`` : string, ``node_index`` of this node's parent. ``None`` for the root node.
            - ``split_feature`` : string, name of the feature used for splitting. ``None`` for leaf nodes.
            - ``split_gain`` : float64, gain from adding this split to the tree. ``NaN`` for leaf nodes.
            - ``threshold`` : float64, value of the feature used to decide which side of the split a record will go down. ``NaN`` for leaf nodes.
            - ``decision_type`` : string, logical operator describing how to compare a value to ``threshold``.
              For example, ``split_feature = "Column_10", threshold = 15, decision_type = "<="`` means that
              records where ``Column_10 <= 15`` follow the left side of the split, otherwise follows the right side of the split. ``None`` for leaf nodes.
            - ``missing_direction`` : string, split direction that missing values should go to. ``None`` for leaf nodes.
            - ``missing_type`` : string, describes what types of values are treated as missing.
            - ``value`` : float64, predicted value for this leaf node, multiplied by the learning rate.
            - ``weight`` : float64 or int64, sum of hessian (second-order derivative of objective), summed over observations that fall in this node.
            - ``count`` : int64, number of records in the training data that fall into this node.

        Returns
        -------
        result : pandas DataFrame
            Returns a pandas DataFrame of the parsed model.
        """
        if not PANDAS_INSTALLED:
            raise LightGBMError('This method cannot be run without pandas installed. '
                                'You must install pandas and restart your session to use this method.')

        if self.num_trees() == 0:
            raise LightGBMError('There are no trees in this Booster and thus nothing to parse')

        def _is_split_node(tree):
            return 'split_index' in tree.keys()

        def create_node_record(tree, node_depth=1, tree_index=None,
                               feature_names=None, parent_node=None):

            def _get_node_index(tree, tree_index):
                tree_num = f'{tree_index}-' if tree_index is not None else ''
                is_split = _is_split_node(tree)
                node_type = 'S' if is_split else 'L'
                # if a single node tree it won't have `leaf_index` so return 0
                node_num = tree.get('split_index' if is_split else 'leaf_index', 0)
                return f"{tree_num}{node_type}{node_num}"

            def _get_split_feature(tree, feature_names):
                if _is_split_node(tree):
                    if feature_names is not None:
                        feature_name = feature_names[tree['split_feature']]
                    else:
                        feature_name = tree['split_feature']
                else:
                    feature_name = None
                return feature_name

            def _is_single_node_tree(tree):
                return set(tree.keys()) == {'leaf_value'}

            # Create the node record, and populate universal data members
            node = OrderedDict()
            node['tree_index'] = tree_index
            node['node_depth'] = node_depth
            node['node_index'] = _get_node_index(tree, tree_index)
            node['left_child'] = None
            node['right_child'] = None
            node['parent_index'] = parent_node
            node['split_feature'] = _get_split_feature(tree, feature_names)
            node['split_gain'] = None
            node['threshold'] = None
            node['decision_type'] = None
            node['missing_direction'] = None
            node['missing_type'] = None
            node['value'] = None
            node['weight'] = None
            node['count'] = None

            # Update values to reflect node type (leaf or split)
            if _is_split_node(tree):
                node['left_child'] = _get_node_index(tree['left_child'], tree_index)
                node['right_child'] = _get_node_index(tree['right_child'], tree_index)
                node['split_gain'] = tree['split_gain']
                node['threshold'] = tree['threshold']
                node['decision_type'] = tree['decision_type']
                node['missing_direction'] = 'left' if tree['default_left'] else 'right'
                node['missing_type'] = tree['missing_type']
                node['value'] = tree['internal_value']
                node['weight'] = tree['internal_weight']
                node['count'] = tree['internal_count']
            else:
                node['value'] = tree['leaf_value']
                if not _is_single_node_tree(tree):
                    node['weight'] = tree['leaf_weight']
                    node['count'] = tree['leaf_count']

            return node

        def tree_dict_to_node_list(tree, node_depth=1, tree_index=None,
                                   feature_names=None, parent_node=None):

            node = create_node_record(tree,
                                      node_depth=node_depth,
                                      tree_index=tree_index,
                                      feature_names=feature_names,
                                      parent_node=parent_node)

            res = [node]

            if _is_split_node(tree):
                # traverse the next level of the tree
                children = ['left_child', 'right_child']
                for child in children:
                    subtree_list = tree_dict_to_node_list(
                        tree[child],
                        node_depth=node_depth + 1,
                        tree_index=tree_index,
                        feature_names=feature_names,
                        parent_node=node['node_index'])
                    # In tree format, "subtree_list" is a list of node records (dicts),
                    # and we add node to the list.
                    res.extend(subtree_list)
            return res

        model_dict = self.dump_model()
        feature_names = model_dict['feature_names']
        model_list = []
        for tree in model_dict['tree_info']:
            model_list.extend(tree_dict_to_node_list(tree['tree_structure'],
                                                     tree_index=tree['tree_index'],
                                                     feature_names=feature_names))

        return pd_DataFrame(model_list, columns=model_list[0].keys())

    def set_train_data_name(self, name):
        """Set the name to the training Dataset.

        Parameters
        ----------
        name : string
            Name for the training Dataset.

        Returns
        -------
        self : Booster
            Booster with set training Dataset name.
        """
        self._train_data_name = name
        return self

    def add_valid(self, data, name):
        """Add validation data.

        Parameters
        ----------
        data : Dataset
            Validation data.
        name : string
            Name of validation data.

        Returns
        -------
        self : Booster
            Booster with set validation data.
        """
        if not isinstance(data, Dataset):
            raise TypeError(f'Validation data should be Dataset instance, met {type(data).__name__}')
        if data._predictor is not self.__init_predictor:
            raise LightGBMError("Add validation data failed, "
                                "you should use same predictor for these data")
        _safe_call(_LIB.LGBM_BoosterAddValidData(
            self.handle,
            data.construct().handle))
        self.valid_sets.append(data)
        self.name_valid_sets.append(name)
        self.__num_dataset += 1
        self.__inner_predict_buffer.append(None)
        self.__is_predicted_cur_iter.append(False)
        return self

    def reset_parameter(self, params):
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
        params_str = param_dict_to_str(params)
        if params_str:
            _safe_call(_LIB.LGBM_BoosterResetParameter(
                self.handle,
                c_str(params_str)))
        self.params.update(params)
        return self

    def update(self, train_set=None, fobj=None):
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

                preds : list or numpy 1-D array
                    The predicted values.
                    Predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task.
                train_data : Dataset
                    The training dataset.
                grad : list or numpy 1-D array
                    The value of the first order derivative (gradient) of the loss
                    with respect to the elements of preds for each sample point.
                hess : list or numpy 1-D array
                    The value of the second order derivative (Hessian) of the loss
                    with respect to the elements of preds for each sample point.

            For multi-class task, the preds is group by class_id first, then group by row_id.
            If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
            and you should group grad and hess in this way as well.

        Returns
        -------
        is_finished : bool
            Whether the update was successfully finished.
        """
        # need reset training data
        if train_set is None and self.train_set_version != self.train_set.version:
            train_set = self.train_set
            is_the_same_train_set = False
        else:
            is_the_same_train_set = train_set is self.train_set and self.train_set_version == train_set.version
        if train_set is not None and not is_the_same_train_set:
            if not isinstance(train_set, Dataset):
                raise TypeError(f'Training data should be Dataset instance, met {type(train_set).__name__}')
            if train_set._predictor is not self.__init_predictor:
                raise LightGBMError("Replace training data failed, "
                                    "you should use same predictor for these data")
            self.train_set = train_set
            _safe_call(_LIB.LGBM_BoosterResetTrainingData(
                self.handle,
                self.train_set.construct().handle))
            self.__inner_predict_buffer[0] = None
            self.train_set_version = self.train_set.version
        is_finished = ctypes.c_int(0)
        if fobj is None:
            if self.__set_objective_to_none:
                raise LightGBMError('Cannot update due to null objective function.')
            _safe_call(_LIB.LGBM_BoosterUpdateOneIter(
                self.handle,
                ctypes.byref(is_finished)))
            self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
            return is_finished.value == 1
        else:
            if not self.__set_objective_to_none:
                self.reset_parameter({"objective": "none"}).__set_objective_to_none = True
            grad, hess = fobj(self.__inner_predict(0), self.train_set)
            return self.__boost(grad, hess)

    def __boost(self, grad, hess):
        """Boost Booster for one iteration with customized gradient statistics.

        .. note::

            Score is returned before any transformation,
            e.g. it is raw margin instead of probability of positive class for binary task.
            For multi-class task, the score is group by class_id first, then group by row_id.
            If you want to get i-th row score in j-th class, the access way is score[j * num_data + i]
            and you should group grad and hess in this way as well.

        Parameters
        ----------
        grad : list or numpy 1-D array
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of score for each sample point.
        hess : list or numpy 1-D array
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of score for each sample point.

        Returns
        -------
        is_finished : bool
            Whether the boost was successfully finished.
        """
        grad = list_to_1d_numpy(grad, name='gradient')
        hess = list_to_1d_numpy(hess, name='hessian')
        assert grad.flags.c_contiguous
        assert hess.flags.c_contiguous
        if len(grad) != len(hess):
            raise ValueError(f"Lengths of gradient({len(grad)}) and hessian({len(hess)}) don't match")
        is_finished = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterUpdateOneIterCustom(
            self.handle,
            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(is_finished)))
        self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
        return is_finished.value == 1

    def rollback_one_iter(self):
        """Rollback one iteration.

        Returns
        -------
        self : Booster
            Booster with rolled back one iteration.
        """
        _safe_call(_LIB.LGBM_BoosterRollbackOneIter(
            self.handle))
        self.__is_predicted_cur_iter = [False for _ in range(self.__num_dataset)]
        return self

    def current_iteration(self):
        """Get the index of the current iteration.

        Returns
        -------
        cur_iter : int
            The index of the current iteration.
        """
        out_cur_iter = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(
            self.handle,
            ctypes.byref(out_cur_iter)))
        return out_cur_iter.value

    def num_model_per_iteration(self):
        """Get number of models per iteration.

        Returns
        -------
        model_per_iter : int
            The number of models per iteration.
        """
        model_per_iter = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterNumModelPerIteration(
            self.handle,
            ctypes.byref(model_per_iter)))
        return model_per_iter.value

    def num_trees(self):
        """Get number of weak sub-models.

        Returns
        -------
        num_trees : int
            The number of weak sub-models.
        """
        num_trees = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterNumberOfTotalModel(
            self.handle,
            ctypes.byref(num_trees)))
        return num_trees.value

    def upper_bound(self):
        """Get upper bound value of a model.

        Returns
        -------
        upper_bound : double
            Upper bound value of the model.
        """
        ret = ctypes.c_double(0)
        _safe_call(_LIB.LGBM_BoosterGetUpperBoundValue(
            self.handle,
            ctypes.byref(ret)))
        return ret.value

    def lower_bound(self):
        """Get lower bound value of a model.

        Returns
        -------
        lower_bound : double
            Lower bound value of the model.
        """
        ret = ctypes.c_double(0)
        _safe_call(_LIB.LGBM_BoosterGetLowerBoundValue(
            self.handle,
            ctypes.byref(ret)))
        return ret.value

    def eval(self, data, name, feval=None):
        """Evaluate for data.

        Parameters
        ----------
        data : Dataset
            Data for the evaluating.
        name : string
            Name of the data.
        feval : callable or None, optional (default=None)
            Customized evaluation function.
            Should accept two parameters: preds, eval_data,
            and return (eval_name, eval_result, is_higher_better) or list of such tuples.

                preds : list or numpy 1-D array
                    The predicted values.
                    If ``fobj`` is specified, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                eval_data : Dataset
                    The evaluation dataset.
                eval_name : string
                    The name of evaluation function (without whitespace).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

            For multi-class task, the preds is group by class_id first, then group by row_id.
            If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].

        Returns
        -------
        result : list
            List with evaluation results.
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

        return self.__inner_eval(name, data_idx, feval)

    def eval_train(self, feval=None):
        """Evaluate for training data.

        Parameters
        ----------
        feval : callable or None, optional (default=None)
            Customized evaluation function.
            Should accept two parameters: preds, train_data,
            and return (eval_name, eval_result, is_higher_better) or list of such tuples.

                preds : list or numpy 1-D array
                    The predicted values.
                    If ``fobj`` is specified, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                train_data : Dataset
                    The training dataset.
                eval_name : string
                    The name of evaluation function (without whitespace).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

            For multi-class task, the preds is group by class_id first, then group by row_id.
            If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].

        Returns
        -------
        result : list
            List with evaluation results.
        """
        return self.__inner_eval(self._train_data_name, 0, feval)

    def eval_valid(self, feval=None):
        """Evaluate for validation data.

        Parameters
        ----------
        feval : callable or None, optional (default=None)
            Customized evaluation function.
            Should accept two parameters: preds, valid_data,
            and return (eval_name, eval_result, is_higher_better) or list of such tuples.

                preds : list or numpy 1-D array
                    The predicted values.
                    If ``fobj`` is specified, predicted values are returned before any transformation,
                    e.g. they are raw margin instead of probability of positive class for binary task in this case.
                valid_data : Dataset
                    The validation dataset.
                eval_name : string
                    The name of evaluation function (without whitespace).
                eval_result : float
                    The eval result.
                is_higher_better : bool
                    Is eval result higher better, e.g. AUC is ``is_higher_better``.

            For multi-class task, the preds is group by class_id first, then group by row_id.
            If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].

        Returns
        -------
        result : list
            List with evaluation results.
        """
        return [item for i in range(1, self.__num_dataset)
                for item in self.__inner_eval(self.name_valid_sets[i - 1], i, feval)]

    def save_model(self, filename, num_iteration=None, start_iteration=0, importance_type='split'):
        """Save Booster to file.

        Parameters
        ----------
        filename : string
            Filename to save Booster.
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : string, optional (default="split")
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
        importance_type_int = FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        _safe_call(_LIB.LGBM_BoosterSaveModel(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.c_int(importance_type_int),
            c_str(filename)))
        _dump_pandas_categorical(self.pandas_categorical, filename)
        return self

    def shuffle_models(self, start_iteration=0, end_iteration=-1):
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
        _safe_call(_LIB.LGBM_BoosterShuffleModels(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(end_iteration)))
        return self

    def model_from_string(self, model_str, verbose=True):
        """Load Booster from a string.

        Parameters
        ----------
        model_str : string
            Model will be loaded from this string.
        verbose : bool, optional (default=True)
            Whether to print messages while loading model.

        Returns
        -------
        self : Booster
            Loaded Booster object.
        """
        if self.handle is not None:
            _safe_call(_LIB.LGBM_BoosterFree(self.handle))
        self._free_buffer()
        self.handle = ctypes.c_void_p()
        out_num_iterations = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterLoadModelFromString(
            c_str(model_str),
            ctypes.byref(out_num_iterations),
            ctypes.byref(self.handle)))
        out_num_class = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetNumClasses(
            self.handle,
            ctypes.byref(out_num_class)))
        if verbose:
            _log_info(f'Finished loading model, total used {int(out_num_iterations.value)} iterations')
        self.__num_class = out_num_class.value
        self.pandas_categorical = _load_pandas_categorical(model_str=model_str)
        return self

    def model_to_string(self, num_iteration=None, start_iteration=0, importance_type='split'):
        """Save Booster to string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : string, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        str_repr : string
            String representation of Booster.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration
        importance_type_int = FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterSaveModelToString(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.c_int(importance_type_int),
            ctypes.c_int64(buffer_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, re-allocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
            _safe_call(_LIB.LGBM_BoosterSaveModelToString(
                self.handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int(importance_type_int),
                ctypes.c_int64(actual_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer))
        ret = string_buffer.value.decode('utf-8')
        ret += _dump_pandas_categorical(self.pandas_categorical)
        return ret

    def dump_model(self, num_iteration=None, start_iteration=0, importance_type='split'):
        """Dump Booster to JSON format.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be dumped.
            If None, if the best iteration exists, it is dumped; otherwise, all iterations are dumped.
            If <= 0, all iterations are dumped.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be dumped.
        importance_type : string, optional (default="split")
            What type of feature importance should be dumped.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        json_repr : dict
            JSON format of Booster.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration
        importance_type_int = FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.LGBM_BoosterDumpModel(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.c_int(importance_type_int),
            ctypes.c_int64(buffer_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, reallocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
            _safe_call(_LIB.LGBM_BoosterDumpModel(
                self.handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int(importance_type_int),
                ctypes.c_int64(actual_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer))
        ret = json.loads(string_buffer.value.decode('utf-8'))
        ret['pandas_categorical'] = json.loads(json.dumps(self.pandas_categorical,
                                                          default=json_default_with_numpy))
        return ret

    def predict(self, data, start_iteration=0, num_iteration=None,
                raw_score=False, pred_leaf=False, pred_contrib=False,
                data_has_header=False, is_reshape=True, **kwargs):
        """Make a prediction.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            If string, it represents the path to txt file.
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
            Used only if data is string.
        is_reshape : bool, optional (default=True)
            If True, result is reshaped to [nrow, ncol].
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        predictor = self._to_predictor(deepcopy(kwargs))
        if num_iteration is None:
            if start_iteration <= 0:
                num_iteration = self.best_iteration
            else:
                num_iteration = -1
        return predictor.predict(data, start_iteration, num_iteration,
                                 raw_score, pred_leaf, pred_contrib,
                                 data_has_header, is_reshape)

    def refit(self, data, label, decay_rate=0.9, **kwargs):
        """Refit the existing Booster by new data.

        Parameters
        ----------
        data : string, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for refit.
            If string, it represents the path to txt file.
        label : list, numpy 1-D array or pandas Series / one-column DataFrame
            Label for refit.
        decay_rate : float, optional (default=0.9)
            Decay rate of refit,
            will use ``leaf_output = decay_rate * old_leaf_output + (1.0 - decay_rate) * new_leaf_output`` to refit trees.
        **kwargs
            Other parameters for refit.
            These parameters will be passed to ``predict`` method.

        Returns
        -------
        result : Booster
            Refitted Booster.
        """
        if self.__set_objective_to_none:
            raise LightGBMError('Cannot refit due to null objective function.')
        predictor = self._to_predictor(deepcopy(kwargs))
        leaf_preds = predictor.predict(data, -1, pred_leaf=True)
        nrow, ncol = leaf_preds.shape
        out_is_linear = ctypes.c_bool(False)
        _safe_call(_LIB.LGBM_BoosterGetLinear(
            self.handle,
            ctypes.byref(out_is_linear)))
        new_params = deepcopy(self.params)
        new_params["linear_tree"] = out_is_linear.value
        train_set = Dataset(data, label, silent=True, params=new_params)
        new_params['refit_decay_rate'] = decay_rate
        new_booster = Booster(new_params, train_set)
        # Copy models
        _safe_call(_LIB.LGBM_BoosterMerge(
            new_booster.handle,
            predictor.handle))
        leaf_preds = leaf_preds.reshape(-1)
        ptr_data, _, _ = c_int_array(leaf_preds)
        _safe_call(_LIB.LGBM_BoosterRefit(
            new_booster.handle,
            ptr_data,
            ctypes.c_int32(nrow),
            ctypes.c_int32(ncol)))
        new_booster.network = self.network
        new_booster.__attr = self.__attr.copy()
        return new_booster

    def get_leaf_output(self, tree_id, leaf_id):
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
        _safe_call(_LIB.LGBM_BoosterGetLeafValue(
            self.handle,
            ctypes.c_int(tree_id),
            ctypes.c_int(leaf_id),
            ctypes.byref(ret)))
        return ret.value

    def _to_predictor(self, pred_parameter=None):
        """Convert to predictor."""
        predictor = _InnerPredictor(booster_handle=self.handle, pred_parameter=pred_parameter)
        predictor.pandas_categorical = self.pandas_categorical
        return predictor

    def num_feature(self):
        """Get number of features.

        Returns
        -------
        num_feature : int
            The number of features.
        """
        out_num_feature = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetNumFeature(
            self.handle,
            ctypes.byref(out_num_feature)))
        return out_num_feature.value

    def feature_name(self):
        """Get names of features.

        Returns
        -------
        result : list
            List with names of features.
        """
        num_feature = self.num_feature()
        # Get name of features
        tmp_out_len = ctypes.c_int(0)
        reserved_string_buffer_size = 255
        required_string_buffer_size = ctypes.c_size_t(0)
        string_buffers = [ctypes.create_string_buffer(reserved_string_buffer_size) for _ in range(num_feature)]
        ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))
        _safe_call(_LIB.LGBM_BoosterGetFeatureNames(
            self.handle,
            ctypes.c_int(num_feature),
            ctypes.byref(tmp_out_len),
            ctypes.c_size_t(reserved_string_buffer_size),
            ctypes.byref(required_string_buffer_size),
            ptr_string_buffers))
        if num_feature != tmp_out_len.value:
            raise ValueError("Length of feature names doesn't equal with num_feature")
        actual_string_buffer_size = required_string_buffer_size.value
        # if buffer length is not long enough, reallocate buffers
        if reserved_string_buffer_size < actual_string_buffer_size:
            string_buffers = [ctypes.create_string_buffer(actual_string_buffer_size) for _ in range(num_feature)]
            ptr_string_buffers = (ctypes.c_char_p * num_feature)(*map(ctypes.addressof, string_buffers))
            _safe_call(_LIB.LGBM_BoosterGetFeatureNames(
                self.handle,
                ctypes.c_int(num_feature),
                ctypes.byref(tmp_out_len),
                ctypes.c_size_t(actual_string_buffer_size),
                ctypes.byref(required_string_buffer_size),
                ptr_string_buffers))
        return [string_buffers[i].value.decode('utf-8') for i in range(num_feature)]

    def feature_importance(self, importance_type='split', iteration=None):
        """Get feature importances.

        Parameters
        ----------
        importance_type : string, optional (default="split")
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
        importance_type_int = FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        result = np.zeros(self.num_feature(), dtype=np.float64)
        _safe_call(_LIB.LGBM_BoosterFeatureImportance(
            self.handle,
            ctypes.c_int(iteration),
            ctypes.c_int(importance_type_int),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if importance_type_int == 0:
            return result.astype(np.int32)
        else:
            return result

    def get_split_value_histogram(self, feature, bins=None, xgboost_style=False):
        """Get split value histogram for the specified feature.

        Parameters
        ----------
        feature : int or string
            The feature name or index the histogram is calculated for.
            If int, interpreted as index.
            If string, interpreted as name.

            .. warning::

                Categorical features are not supported.

        bins : int, string or None, optional (default=None)
            The maximum number of bins.
            If None, or int and > number of unique split values and ``xgboost_style=True``,
            the number of bins equals number of unique split values.
            If string, it should be one from the list of the supported values by ``numpy.histogram()`` function.
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
        def add(root):
            """Recursively add thresholds."""
            if 'split_index' in root:  # non-leaf
                if feature_names is not None and isinstance(feature, str):
                    split_feature = feature_names[root['split_feature']]
                else:
                    split_feature = root['split_feature']
                if split_feature == feature:
                    if isinstance(root['threshold'], str):
                        raise LightGBMError('Cannot compute split value histogram for the categorical feature')
                    else:
                        values.append(root['threshold'])
                add(root['left_child'])
                add(root['right_child'])

        model = self.dump_model()
        feature_names = model.get('feature_names')
        tree_infos = model['tree_info']
        values = []
        for tree_info in tree_infos:
            add(tree_info['tree_structure'])

        if bins is None or isinstance(bins, int) and xgboost_style:
            n_unique = len(np.unique(values))
            bins = max(min(n_unique, bins) if bins is not None else n_unique, 1)
        hist, bin_edges = np.histogram(values, bins=bins)
        if xgboost_style:
            ret = np.column_stack((bin_edges[1:], hist))
            ret = ret[ret[:, 1] > 0]
            if PANDAS_INSTALLED:
                return pd_DataFrame(ret, columns=['SplitValue', 'Count'])
            else:
                return ret
        else:
            return hist, bin_edges

    def __inner_eval(self, data_name, data_idx, feval=None):
        """Evaluate training or validation data."""
        if data_idx >= self.__num_dataset:
            raise ValueError("Data_idx should be smaller than number of dataset")
        self.__get_eval_info()
        ret = []
        if self.__num_inner_eval > 0:
            result = np.zeros(self.__num_inner_eval, dtype=np.float64)
            tmp_out_len = ctypes.c_int(0)
            _safe_call(_LIB.LGBM_BoosterGetEval(
                self.handle,
                ctypes.c_int(data_idx),
                ctypes.byref(tmp_out_len),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
            if tmp_out_len.value != self.__num_inner_eval:
                raise ValueError("Wrong length of eval results")
            for i in range(self.__num_inner_eval):
                ret.append((data_name, self.__name_inner_eval[i],
                            result[i], self.__higher_better_inner_eval[i]))
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
                feval_ret = eval_function(self.__inner_predict(data_idx), cur_data)
                if isinstance(feval_ret, list):
                    for eval_name, val, is_higher_better in feval_ret:
                        ret.append((data_name, eval_name, val, is_higher_better))
                else:
                    eval_name, val, is_higher_better = feval_ret
                    ret.append((data_name, eval_name, val, is_higher_better))
        return ret

    def __inner_predict(self, data_idx):
        """Predict for training and validation dataset."""
        if data_idx >= self.__num_dataset:
            raise ValueError("Data_idx should be smaller than number of dataset")
        if self.__inner_predict_buffer[data_idx] is None:
            if data_idx == 0:
                n_preds = self.train_set.num_data() * self.__num_class
            else:
                n_preds = self.valid_sets[data_idx - 1].num_data() * self.__num_class
            self.__inner_predict_buffer[data_idx] = np.zeros(n_preds, dtype=np.float64)
        # avoid to predict many time in one iteration
        if not self.__is_predicted_cur_iter[data_idx]:
            tmp_out_len = ctypes.c_int64(0)
            data_ptr = self.__inner_predict_buffer[data_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            _safe_call(_LIB.LGBM_BoosterGetPredict(
                self.handle,
                ctypes.c_int(data_idx),
                ctypes.byref(tmp_out_len),
                data_ptr))
            if tmp_out_len.value != len(self.__inner_predict_buffer[data_idx]):
                raise ValueError(f"Wrong length of predict results for data {data_idx}")
            self.__is_predicted_cur_iter[data_idx] = True
        return self.__inner_predict_buffer[data_idx]

    def __get_eval_info(self):
        """Get inner evaluation count and names."""
        if self.__need_reload_eval_info:
            self.__need_reload_eval_info = False
            out_num_eval = ctypes.c_int(0)
            # Get num of inner evals
            _safe_call(_LIB.LGBM_BoosterGetEvalCounts(
                self.handle,
                ctypes.byref(out_num_eval)))
            self.__num_inner_eval = out_num_eval.value
            if self.__num_inner_eval > 0:
                # Get name of eval metrics
                tmp_out_len = ctypes.c_int(0)
                reserved_string_buffer_size = 255
                required_string_buffer_size = ctypes.c_size_t(0)
                string_buffers = [
                    ctypes.create_string_buffer(reserved_string_buffer_size) for _ in range(self.__num_inner_eval)
                ]
                ptr_string_buffers = (ctypes.c_char_p * self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))
                _safe_call(_LIB.LGBM_BoosterGetEvalNames(
                    self.handle,
                    ctypes.c_int(self.__num_inner_eval),
                    ctypes.byref(tmp_out_len),
                    ctypes.c_size_t(reserved_string_buffer_size),
                    ctypes.byref(required_string_buffer_size),
                    ptr_string_buffers))
                if self.__num_inner_eval != tmp_out_len.value:
                    raise ValueError("Length of eval names doesn't equal with num_evals")
                actual_string_buffer_size = required_string_buffer_size.value
                # if buffer length is not long enough, reallocate buffers
                if reserved_string_buffer_size < actual_string_buffer_size:
                    string_buffers = [
                        ctypes.create_string_buffer(actual_string_buffer_size) for _ in range(self.__num_inner_eval)
                    ]
                    ptr_string_buffers = (ctypes.c_char_p * self.__num_inner_eval)(*map(ctypes.addressof, string_buffers))
                    _safe_call(_LIB.LGBM_BoosterGetEvalNames(
                        self.handle,
                        ctypes.c_int(self.__num_inner_eval),
                        ctypes.byref(tmp_out_len),
                        ctypes.c_size_t(actual_string_buffer_size),
                        ctypes.byref(required_string_buffer_size),
                        ptr_string_buffers))
                self.__name_inner_eval = [
                    string_buffers[i].value.decode('utf-8') for i in range(self.__num_inner_eval)
                ]
                self.__higher_better_inner_eval = [
                    name.startswith(('auc', 'ndcg@', 'map@', 'average_precision')) for name in self.__name_inner_eval
                ]

    def attr(self, key):
        """Get attribute string from the Booster.

        Parameters
        ----------
        key : string
            The name of the attribute.

        Returns
        -------
        value : string or None
            The attribute value.
            Returns None if attribute does not exist.
        """
        return self.__attr.get(key, None)

    def set_attr(self, **kwargs):
        """Set attributes to the Booster.

        Parameters
        ----------
        **kwargs
            The attributes to set.
            Setting a value to None deletes an attribute.

        Returns
        -------
        self : Booster
            Booster with set attributes.
        """
        for key, value in kwargs.items():
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("Only string values are accepted")
                self.__attr[key] = value
            else:
                self.__attr.pop(key, None)
        return self
