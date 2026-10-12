# coding: utf-8
"""Distributed training with LightGBM and dask.distributed.

This module enables you to perform distributed training with LightGBM on
dask.Array and dask.DataFrame collections.

It is based on dask-lightgbm, which was based on dask-xgboost.
"""

import operator
import socket
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse

import numpy as np
import scipy.sparse as ss

from .basic import LightGBMError, _choose_param_value, _ConfigAliases, _log_info, _log_warning
from .compat import (
    PANDAS_INSTALLED,
    SKLEARN_INSTALLED,
    LGBMNotFittedError,
    concat,
    pd_DataFrame,
    pd_Series,
)
from .sklearn import (
    LGBMClassifier,
    LGBMModel,
    LGBMRanker,
    LGBMRegressor,
    _LGBM_ScikitCustomObjectiveFunction,
    _LGBM_ScikitEvalMetricType,
    _validate_eval_set_Xy,
)

__all__ = [
    "DaskLGBMClassifier",
    "DaskLGBMRanker",
    "DaskLGBMRegressor",
]

if TYPE_CHECKING:
    import dask
    import distributed

_DaskCollection = Union["dask.array.Array", "dask.dataframe.DataFrame", "dask.dataframe.Series"]
_DaskMatrixLike = Union["dask.array.Array", "dask.dataframe.DataFrame"]
_DaskVectorLike = Union["dask.array.Array", "dask.dataframe.Series"]
_DaskPart = Union[np.ndarray, pd_DataFrame, pd_Series, ss.spmatrix]

# catching 'ValueError' here because of this:
# https://github.com/lightgbm-org/LightGBM/issues/6365#issuecomment-2002330003
#
# That's potentially risky as dask does some significant import-time processing,
# like loading configuration from environment variables and files, and catching
# ValueError here might hide issues with that config-loading.
#
# But in exchange, it's less likely that 'import lightgbm' will fail for
# dask-related reasons, which is beneficial for any workloads that are using
# lightgbm but not its Dask functionality.
#
# 'ValueError' can be removed when LightGBM's Dask floor is '>=2024.4.0'.
_DaskImportErrorTypes = (ImportError, ValueError)


class _RemoteSocket:
    def acquire(self) -> int:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("", 0))
        return self.socket.getsockname()[1]

    def release(self) -> None:
        self.socket.close()


def _acquire_port() -> Tuple[_RemoteSocket, int]:
    s = _RemoteSocket()
    port = s.acquire()
    return s, port


class _DatasetNames(Enum):
    """Placeholder names used by lightgbm.dask internals to say 'also evaluate the training data'.

    Avoid duplicating the training data when the validation set refers to elements of training data.
    """

    TRAINSET = auto()
    SAMPLE_WEIGHT = auto()
    INIT_SCORE = auto()
    GROUP = auto()


def _get_dask_client(client: Optional["distributed.Client"]) -> "distributed.Client":
    """Choose a Dask client to use.

    Parameters
    ----------
    client : distributed.Client or None
        Dask client.

    Returns
    -------
    client : distributed.Client
        A Dask client.
    """
    from dask.distributed import default_client  # noqa: PLC0415

    if client is None:
        return default_client()
    else:
        return client


def _assign_open_ports_to_workers(
    *,
    client: "distributed.Client",
    workers: List[str],
) -> Tuple[Dict[str, "distributed.client.Future"], Dict[str, int]]:
    """Assign an open port to each worker.

    Returns
    -------
    worker_to_socket_future: dict
        mapping from worker address to a future pointing to the remote socket.
    worker_to_port: dict
        mapping from worker address to an open port in the worker's host.
    """
    # Acquire port in worker
    worker_to_future = {}
    for worker in workers:
        worker_to_future[worker] = client.submit(
            _acquire_port,
            workers=[worker],
            allow_other_workers=False,
            pure=False,
        )

    # schedule futures to retrieve each element of the tuple
    worker_to_socket_future = {}
    worker_to_port_future = {}
    for worker, socket_future in worker_to_future.items():
        worker_to_socket_future[worker] = client.submit(operator.itemgetter(0), socket_future)
        worker_to_port_future[worker] = client.submit(operator.itemgetter(1), socket_future)

    # retrieve ports
    worker_to_port = client.gather(worker_to_port_future)

    return worker_to_socket_future, worker_to_port


def _concat(seq: List[_DaskPart]) -> _DaskPart:
    if isinstance(seq[0], np.ndarray):
        return np.concatenate(seq, axis=0)
    elif isinstance(seq[0], (pd_DataFrame, pd_Series)):
        return concat(seq, axis=0)
    elif isinstance(seq[0], ss.spmatrix):
        return ss.vstack(seq, format="csr")
    else:
        raise TypeError(
            f"Data must be one of: numpy arrays, pandas dataframes, sparse matrices (from scipy). Got {type(seq[0]).__name__}."
        )


def _remove_list_padding(*args: Any) -> List[List[Any]]:
    return [[z for z in arg if z is not None] for arg in args]


def _pad_eval_names(
    *,
    lgbm_model: LGBMModel,
    required_names: List[str],
) -> LGBMModel:
    """Append missing (key, value) pairs to a LightGBM model's evals_result_ and best_score_ OrderedDict attrs based on a set of required eval_set names.

    Allows users to rely on expected eval_set names being present when fitting DaskLGBM estimators with ``eval_set``.
    """
    for eval_name in required_names:
        if eval_name not in lgbm_model.evals_result_:
            lgbm_model.evals_result_[eval_name] = {}
        if eval_name not in lgbm_model.best_score_:
            lgbm_model.best_score_[eval_name] = {}

    return lgbm_model


def _train_part(
    *,
    params: Dict[str, Any],
    model_factory: Type[LGBMModel],
    list_of_parts: List[Dict[str, _DaskPart]],
    machines: str,
    local_listen_port: int,
    num_machines: int,
    return_model: bool,
    time_out: int,
    remote_socket: _RemoteSocket,
    **kwargs: Any,
) -> Optional[LGBMModel]:
    network_params = {
        "machines": machines,
        "local_listen_port": local_listen_port,
        "time_out": time_out,
        "num_machines": num_machines,
    }
    params.update(network_params)

    is_ranker = issubclass(model_factory, LGBMRanker)

    # Concatenate many parts into one
    data = _concat([x["data"] for x in list_of_parts])
    label = _concat([x["label"] for x in list_of_parts])

    if "weight" in list_of_parts[0]:
        weight = _concat([x["weight"] for x in list_of_parts])
    else:
        weight = None

    if "group" in list_of_parts[0]:
        group = _concat([x["group"] for x in list_of_parts])
    else:
        group = None

    if "init_score" in list_of_parts[0]:
        init_score = _concat([x["init_score"] for x in list_of_parts])
    else:
        init_score = None

    # construct local eval_set data.
    n_evals = max(len(x.get("eval_set", [])) for x in list_of_parts)
    eval_names = kwargs.pop("eval_names", None)
    eval_class_weight = kwargs.get("eval_class_weight")
    local_eval_set = None
    local_eval_names = None
    local_eval_sample_weight = None
    local_eval_init_score = None
    local_eval_group = None

    if n_evals:
        has_eval_sample_weight = any(x.get("eval_sample_weight") is not None for x in list_of_parts)
        has_eval_init_score = any(x.get("eval_init_score") is not None for x in list_of_parts)

        local_eval_set = []
        evals_result_names = []
        if has_eval_sample_weight:
            local_eval_sample_weight = []
        if has_eval_init_score:
            local_eval_init_score = []
        if is_ranker:
            local_eval_group = []

        # store indices of eval_set components that were not contained within local parts.
        missing_eval_component_idx = []

        # consolidate parts of each individual eval component.
        for i in range(n_evals):
            x_e = []
            y_e = []
            w_e = []
            init_score_e = []
            g_e = []
            for part in list_of_parts:
                if not part.get("eval_set"):
                    continue

                # require that eval_name exists in evaluated result data in case dropped due to padding.
                # in distributed training the 'training' eval_set is not detected, will have name 'valid_<index>'.
                if eval_names:
                    evals_result_name = eval_names[i]
                else:
                    evals_result_name = f"valid_{i}"

                eval_set = part["eval_set"][i]
                if eval_set is _DatasetNames.TRAINSET:
                    x_e.append(part["data"])
                    y_e.append(part["label"])
                else:
                    x_e.extend(eval_set[0])
                    y_e.extend(eval_set[1])

                if evals_result_name not in evals_result_names:
                    evals_result_names.append(evals_result_name)

                eval_weight = part.get("eval_sample_weight")
                if eval_weight:
                    if eval_weight[i] is _DatasetNames.SAMPLE_WEIGHT:
                        w_e.append(part["weight"])
                    else:
                        w_e.extend(eval_weight[i])

                eval_init_score = part.get("eval_init_score")
                if eval_init_score:
                    if eval_init_score[i] is _DatasetNames.INIT_SCORE:
                        init_score_e.append(part["init_score"])
                    else:
                        init_score_e.extend(eval_init_score[i])

                eval_group = part.get("eval_group")
                if eval_group:
                    if eval_group[i] is _DatasetNames.GROUP:
                        g_e.append(part["group"])
                    else:
                        g_e.extend(eval_group[i])

            # filter padding from eval parts then _concat each eval_set component.
            x_e, y_e, w_e, init_score_e, g_e = _remove_list_padding(x_e, y_e, w_e, init_score_e, g_e)
            if x_e:
                local_eval_set.append((_concat(x_e), _concat(y_e)))
            else:
                missing_eval_component_idx.append(i)
                continue

            if w_e:
                local_eval_sample_weight.append(_concat(w_e))
            if init_score_e:
                local_eval_init_score.append(_concat(init_score_e))
            if g_e:
                local_eval_group.append(_concat(g_e))

        # reconstruct eval_set fit args/kwargs depending on which components of eval_set are on worker.
        eval_component_idx = [i for i in range(n_evals) if i not in missing_eval_component_idx]
        if eval_names:
            local_eval_names = [eval_names[i] for i in eval_component_idx]
        if eval_class_weight:
            kwargs["eval_class_weight"] = [eval_class_weight[i] for i in eval_component_idx]

    if local_eval_set is None:
        local_eval_X = None
        local_eval_y = None
    else:
        local_eval_X = tuple(X for X, _ in local_eval_set)
        local_eval_y = tuple(y for _, y in local_eval_set)

    model = model_factory(**params)
    if remote_socket is not None:
        remote_socket.release()
    try:
        if is_ranker:
            model.fit(
                data,
                label,
                sample_weight=weight,
                init_score=init_score,
                group=group,
                eval_X=local_eval_X,
                eval_y=local_eval_y,
                eval_sample_weight=local_eval_sample_weight,
                eval_init_score=local_eval_init_score,
                eval_group=local_eval_group,
                eval_names=local_eval_names,
                **kwargs,
            )
        else:
            model.fit(
                data,
                label,
                sample_weight=weight,
                init_score=init_score,
                eval_X=local_eval_X,
                eval_y=local_eval_y,
                eval_sample_weight=local_eval_sample_weight,
                eval_init_score=local_eval_init_score,
                eval_names=local_eval_names,
                **kwargs,
            )

    finally:
        if getattr(model, "fitted_", False):
            model.booster_.free_network()

    if n_evals:
        # ensure that expected keys for evals_result_ and best_score_ exist regardless of padding.
        model = _pad_eval_names(lgbm_model=model, required_names=evals_result_names)

    return model if return_model else None


def _split_to_parts(*, data: _DaskCollection, is_matrix: bool) -> List[_DaskPart]:
    parts = data.to_delayed()
    if isinstance(parts, np.ndarray):
        if is_matrix:
            assert parts.shape[1] == 1
        else:
            assert parts.ndim == 1 or parts.shape[1] == 1
        parts = parts.flatten().tolist()
    return parts


def _machines_to_worker_map(
    *,
    machines: str,
    worker_addresses: Iterable[str],
) -> Dict[str, int]:
    """Create a worker_map from machines list.

    Given ``machines`` and a list of Dask worker addresses, return a mapping where the keys are
    ``worker_addresses`` and the values are ports from ``machines``.

    Parameters
    ----------
    machines : str
        A comma-delimited list of workers, of the form ``ip1:port,ip2:port``.
    worker_addresses : list of str
        An iterable of Dask worker addresses, of the form ``{protocol}{hostname}:{port}``, where ``port`` is the port Dask's scheduler uses to talk to that worker.

    Returns
    -------
    result : Dict[str, int]
        Dictionary where keys are work addresses in the form expected by Dask and values are a port for LightGBM to use.
    """
    machine_addresses = machines.split(",")

    if len(set(machine_addresses)) != len(machine_addresses):
        raise ValueError(
            f"Found duplicates in 'machines' ({machines}). Each entry in 'machines' must be a unique IP-port combination."
        )

    machine_to_port = defaultdict(set)
    for address in machine_addresses:
        host, port = address.split(":")
        machine_to_port[host].add(int(port))

    out = {}
    for address in worker_addresses:
        worker_host = urlparse(address).hostname
        if not worker_host:
            raise ValueError(f"Could not parse host name from worker address '{address}'")
        out[address] = machine_to_port[worker_host].pop()

    return out


def _train(
    *,
    client: "distributed.Client",
    data: _DaskMatrixLike,
    label: _DaskCollection,
    params: Dict[str, Any],
    model_factory: Type[LGBMModel],
    sample_weight: Optional[_DaskVectorLike] = None,
    init_score: Optional[_DaskCollection] = None,
    group: Optional[_DaskVectorLike] = None,
    eval_set: Optional[List[Tuple[_DaskMatrixLike, _DaskCollection]]] = None,
    eval_names: Optional[List[str]] = None,
    eval_X: Optional[Union[_DaskMatrixLike, Tuple[_DaskMatrixLike]]] = None,
    eval_y: Optional[Union[_DaskCollection, Tuple[_DaskCollection]]] = None,
    eval_sample_weight: Optional[List[_DaskVectorLike]] = None,
    eval_class_weight: Optional[List[Union[dict, str]]] = None,
    eval_init_score: Optional[List[_DaskCollection]] = None,
    eval_group: Optional[List[_DaskVectorLike]] = None,
    eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
    eval_at: Optional[Union[List[int], Tuple[int, ...]]] = None,
    **kwargs: Any,
) -> LGBMModel:
    """Inner train routine.

    Parameters
    ----------
    client : distributed.Client
        Dask client.
    data : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
        Input feature matrix.
    label : Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]
        The target values (class labels in classification, real numbers in regression).
    params : dict
        Parameters passed to constructor of the local underlying model.
    model_factory : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Class of the local underlying model.
    sample_weight : Dask Array or Dask Series of shape = [n_samples] or None, optional (default=None)
        Weights of training data. Weights should be non-negative.
    init_score : Dask Array or Dask Series of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task), or Dask Array or Dask DataFrame of shape = [n_samples, n_classes] (for multi-class task), or None, optional (default=None)
        Init score of training data.
    group : Dask Array or Dask Series or None, optional (default=None)
        Group/query data.
        Only used in the learning-to-rank task.
        sum(group) = n_samples.
        For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
        where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
    eval_set : list of (X, y) tuples of Dask data collections, or None, optional (default=None)
        List of (X, y) tuple pairs to use as validation sets.
        Note, that not all workers may receive chunks of every eval set within ``eval_set``. When the returned
        lightgbm estimator is not trained using any chunks of a particular eval set, its corresponding component
        of ``evals_result_`` and ``best_score_`` will be empty dictionaries.
    eval_names : list of str, or None, optional (default=None)
        Unique identifiers for each evaluation dataset.
        Should be the same length as ``eval_set`` / ``eval_X``.
    eval_X : Dask Array or Dask DataFrame, tuple thereof or None, optional (default=None)
        Feature matrix or tuple thereof, e.g. ``(X_val0, X_val1)``, to use as validation sets.
    eval_y : Dask Array or Dask DataFrame or Dask Series, tuple thereof or None, optional (default=None)
        Target values or tuple thereof, e.g. ``(y_val0, y_val1)``, to use as validation sets.
    eval_sample_weight : list of Dask Array or Dask Series, or None, optional (default=None)
        Weights for each validation set in eval_set. Weights should be non-negative.
    eval_class_weight : list of dict or str, or None, optional (default=None)
        Class weights, one dict or str for each validation set in eval_set.
    eval_init_score : list of Dask Array, Dask Series or Dask DataFrame (for multi-class task), or None, optional (default=None)
        Initial model score for each validation set in eval_set.
    eval_group : list of Dask Array or Dask Series, or None, optional (default=None)
        Group/query for each validation set in eval_set.
    eval_metric : str, callable, list or None, optional (default=None)
        If str, it should be a built-in evaluation metric to use.
        If callable, it should be a custom evaluation metric, see note below for more details.
        If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
        In either case, the ``metric`` from the Dask model parameters (or inferred from the objective) will be evaluated and used as well.
        Default: 'l2' for DaskLGBMRegressor, 'binary(multi)_logloss' for DaskLGBMClassifier, 'ndcg' for DaskLGBMRanker.
    eval_at : list or tuple of int, optional (default=None)
        The evaluation positions of the specified ranking metric.
    **kwargs
        Other parameters passed to ``fit`` method of the local underlying model.

    Returns
    -------
    model : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Returns fitted underlying model.

    Note
    ----
    This method handles setting up the following network parameters based on information
    about the Dask cluster referenced by ``client``.

    * ``local_listen_port``: port that each LightGBM worker opens a listening socket on,
            to accept connections from other workers. This can differ from LightGBM worker
            to LightGBM worker, but does not have to.
    * ``machines``: a comma-delimited list of all workers in the cluster, in the
            form ``ip:port,ip:port``. If running multiple Dask workers on the same host, use different
            ports for each worker. For example, for ``LocalCluster(n_workers=3)``, you might
            pass ``"127.0.0.1:12400,127.0.0.1:12401,127.0.0.1:12402"``.
    * ``num_machines``: number of LightGBM workers.
    * ``timeout``: time in minutes to wait before closing unused sockets.

    The default behavior of this function is to generate ``machines`` from the list of
    Dask workers which hold some piece of the training data, and to search for an open
    port on each worker to be used as ``local_listen_port``.

    If ``machines`` is provided explicitly in ``params``, this function uses the hosts
    and ports in that list directly, and does not do any searching. This means that if
    any of the Dask workers are missing from the list or any of those ports are not free
    when training starts, training will fail.

    If ``local_listen_port`` is provided in ``params`` and ``machines`` is not, this function
    constructs ``machines`` from the list of Dask workers which hold some piece of the
    training data, assuming that each one will use the same ``local_listen_port``.
    """
    try:
        from dask import delayed  # noqa: PLC0415
        from dask.distributed import wait  # noqa: PLC0415
    except _DaskImportErrorTypes as err:
        raise LightGBMError("dask is required for lightgbm.dask") from err

    params = deepcopy(params)

    # capture whether local_listen_port or its aliases were provided
    listen_port_in_params = any(alias in params for alias in _ConfigAliases.get("local_listen_port"))

    # capture whether machines or its aliases were provided
    machines_in_params = any(alias in params for alias in _ConfigAliases.get("machines"))

    params = _choose_param_value(
        main_param_name="tree_learner",
        params=params,
        default_value="data",
    )
    allowed_tree_learners = {
        "data",
        "data_parallel",
        "feature",
        "feature_parallel",
        "voting",
        "voting_parallel",
    }
    if params["tree_learner"] not in allowed_tree_learners:
        _log_warning(
            f'Parameter tree_learner set to {params["tree_learner"]}, which is not allowed. Using "data" as default'
        )
        params["tree_learner"] = "data"

    # Some passed-in parameters can be removed:
    #   * 'num_machines': set automatically from Dask worker list
    #   * 'num_threads': overridden to match nthreads on each Dask process
    for param_alias in _ConfigAliases.get("num_machines", "num_threads"):
        if param_alias in params:
            _log_warning(f"Parameter {param_alias} will be ignored.")
            params.pop(param_alias)

    # Split arrays/dataframes into parts. Arrange parts into dicts to enforce co-locality
    data_parts = _split_to_parts(data=data, is_matrix=True)
    label_parts = _split_to_parts(data=label, is_matrix=False)
    parts = [{"data": x, "label": y} for (x, y) in zip(data_parts, label_parts, strict=True)]
    n_parts = len(parts)

    if sample_weight is not None:
        weight_parts = _split_to_parts(data=sample_weight, is_matrix=False)
        for i in range(n_parts):
            parts[i]["weight"] = weight_parts[i]

    if group is not None:
        group_parts = _split_to_parts(data=group, is_matrix=False)
        for i in range(n_parts):
            parts[i]["group"] = group_parts[i]

    if init_score is not None:
        init_score_parts = _split_to_parts(data=init_score, is_matrix=False)
        for i in range(n_parts):
            parts[i]["init_score"] = init_score_parts[i]

    eval_set = _validate_eval_set_Xy(eval_set=eval_set, eval_X=eval_X, eval_y=eval_y)
    # evals_set will to be re-constructed into smaller lists of (X, y) tuples, where
    # X and y are each delayed sub-lists of original eval dask Collections.
    if eval_set:
        # find maximum number of parts in an individual eval set so that we can
        # pad eval sets when they come in different sizes.
        n_largest_eval_parts = max(x[0].npartitions for x in eval_set)

        eval_sets: Dict[
            int, List[Union[_DatasetNames, Tuple[List[Optional[_DaskMatrixLike]], List[Optional[_DaskVectorLike]]]]]
        ] = defaultdict(list)
        if eval_sample_weight:
            eval_sample_weights: Dict[int, List[Union[_DatasetNames, List[Optional[_DaskVectorLike]]]]] = defaultdict(
                list
            )
        if eval_group:
            eval_groups: Dict[int, List[Union[_DatasetNames, List[Optional[_DaskVectorLike]]]]] = defaultdict(list)
        if eval_init_score:
            eval_init_scores: Dict[int, List[Union[_DatasetNames, List[Optional[_DaskMatrixLike]]]]] = defaultdict(list)

        for i, (X_eval, y_eval) in enumerate(eval_set):
            n_this_eval_parts = X_eval.npartitions

            # when individual eval set is equivalent to training data, skip recomputing parts.
            if X_eval is data and y_eval is label:
                for parts_idx in range(n_parts):
                    eval_sets[parts_idx].append(_DatasetNames.TRAINSET)
            else:
                eval_x_parts = _split_to_parts(data=X_eval, is_matrix=True)
                eval_y_parts = _split_to_parts(data=y_eval, is_matrix=False)
                for j in range(n_largest_eval_parts):
                    parts_idx = j % n_parts

                    # add None-padding for individual eval_set member if it is smaller than the largest member.
                    if j < n_this_eval_parts:
                        x_e = eval_x_parts[j]
                        y_e = eval_y_parts[j]
                    else:
                        x_e = None
                        y_e = None

                    if j < n_parts:
                        # first time a chunk of this eval set is added to this part.
                        eval_sets[parts_idx].append(([x_e], [y_e]))
                    else:
                        # append additional chunks of this eval set to this part.
                        eval_sets[parts_idx][-1][0].append(x_e)  # type: ignore[index, union-attr]
                        eval_sets[parts_idx][-1][1].append(y_e)  # type: ignore[index, union-attr]

            if eval_sample_weight:
                if eval_sample_weight[i] is sample_weight:
                    for parts_idx in range(n_parts):
                        eval_sample_weights[parts_idx].append(_DatasetNames.SAMPLE_WEIGHT)
                else:
                    eval_w_parts = _split_to_parts(data=eval_sample_weight[i], is_matrix=False)

                    # ensure that all evaluation parts map uniquely to one part.
                    for j in range(n_largest_eval_parts):
                        if j < n_this_eval_parts:
                            w_e = eval_w_parts[j]
                        else:
                            w_e = None

                        parts_idx = j % n_parts
                        if j < n_parts:
                            eval_sample_weights[parts_idx].append([w_e])
                        else:
                            eval_sample_weights[parts_idx][-1].append(w_e)  # type: ignore[union-attr]

            if eval_init_score:
                if eval_init_score[i] is init_score:
                    for parts_idx in range(n_parts):
                        eval_init_scores[parts_idx].append(_DatasetNames.INIT_SCORE)
                else:
                    eval_init_score_parts = _split_to_parts(data=eval_init_score[i], is_matrix=False)
                    for j in range(n_largest_eval_parts):
                        if j < n_this_eval_parts:
                            init_score_e = eval_init_score_parts[j]
                        else:
                            init_score_e = None

                        parts_idx = j % n_parts
                        if j < n_parts:
                            eval_init_scores[parts_idx].append([init_score_e])
                        else:
                            eval_init_scores[parts_idx][-1].append(init_score_e)  # type: ignore[union-attr]

            if eval_group:
                if eval_group[i] is group:
                    for parts_idx in range(n_parts):
                        eval_groups[parts_idx].append(_DatasetNames.GROUP)
                else:
                    eval_g_parts = _split_to_parts(data=eval_group[i], is_matrix=False)
                    for j in range(n_largest_eval_parts):
                        if j < n_this_eval_parts:
                            g_e = eval_g_parts[j]
                        else:
                            g_e = None

                        parts_idx = j % n_parts
                        if j < n_parts:
                            eval_groups[parts_idx].append([g_e])
                        else:
                            eval_groups[parts_idx][-1].append(g_e)  # type: ignore[union-attr]

        # assign sub-eval_set components to worker parts.
        for parts_idx, e_set in eval_sets.items():
            parts[parts_idx]["eval_set"] = e_set
            if eval_sample_weight:
                parts[parts_idx]["eval_sample_weight"] = eval_sample_weights[parts_idx]
            if eval_init_score:
                parts[parts_idx]["eval_init_score"] = eval_init_scores[parts_idx]
            if eval_group:
                parts[parts_idx]["eval_group"] = eval_groups[parts_idx]

    # Start computation in the background
    parts = list(map(delayed, parts))
    parts = client.compute(parts)
    wait(parts)

    for part in parts:
        if part.status == "error":  # type: ignore
            # trigger error locally
            return part  # type: ignore[return-value]

    # Find locations of all parts and map them to particular Dask workers
    key_to_part_dict = {part.key: part for part in parts}  # type: ignore
    who_has = client.who_has(parts)
    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[next(iter(workers))].append(key_to_part_dict[key])

    # Check that all workers were provided some of eval_set. Otherwise warn user that validation
    # data artifacts may not be populated depending on worker returning final estimator.
    if eval_set:
        for worker in worker_map:
            has_eval_set = False
            for part in worker_map[worker]:
                if "eval_set" in part.result():  # type: ignore[attr-defined]
                    has_eval_set = True
                    break

            if not has_eval_set:
                _log_warning(
                    f"Worker {worker} was not allocated eval_set data. Therefore evals_result_ and best_score_ data may be unreliable. "
                    "Try rebalancing data across workers."
                )

    # assign general validation set settings to fit kwargs.
    if eval_names:
        kwargs["eval_names"] = eval_names
    if eval_class_weight:
        kwargs["eval_class_weight"] = eval_class_weight
    if eval_metric:
        kwargs["eval_metric"] = eval_metric
    if eval_at:
        kwargs["eval_at"] = eval_at

    master_worker = next(iter(worker_map))
    worker_ncores = client.ncores()

    # resolve aliases for network parameters and pop the result off params.
    # these values are added back in calls to `_train_part()`
    params = _choose_param_value(
        main_param_name="local_listen_port",
        params=params,
        default_value=12400,
    )
    local_listen_port = params.pop("local_listen_port")

    params = _choose_param_value(
        main_param_name="machines",
        params=params,
        default_value=None,
    )
    machines = params.pop("machines")

    # figure out network params
    worker_to_socket_future: Dict[str, "distributed.client.Future"] = {}
    worker_addresses = worker_map.keys()
    if machines is not None:
        _log_info("Using passed-in 'machines' parameter")
        worker_address_to_port = _machines_to_worker_map(
            machines=machines,
            worker_addresses=worker_addresses,
        )
    else:
        if listen_port_in_params:
            _log_info("Using passed-in 'local_listen_port' for all workers")
            unique_hosts = {urlparse(a).hostname for a in worker_addresses}
            if len(unique_hosts) < len(worker_addresses):
                msg = (
                    "'local_listen_port' was provided in Dask training parameters, but at least one "
                    "machine in the cluster has multiple Dask worker processes running on it. Please omit "
                    "'local_listen_port' or pass 'machines'."
                )
                raise LightGBMError(msg)

            worker_address_to_port = dict.fromkeys(worker_addresses, local_listen_port)
        else:
            _log_info("Finding random open ports for workers")
            worker_to_socket_future, worker_address_to_port = _assign_open_ports_to_workers(
                client=client,
                workers=list(worker_map.keys()),
            )

        machines = ",".join(
            [f"{urlparse(worker_address).hostname}:{port}" for worker_address, port in worker_address_to_port.items()]
        )

    num_machines = len(worker_address_to_port)

    # Tell each worker to train on the parts that it has locally
    #
    # This code treats ``_train_part()`` calls as not "pure" because:
    #     1. there is randomness in the training process unless parameters ``seed``
    #        and ``deterministic`` are set
    #     2. even with those parameters set, the output of one ``_train_part()`` call
    #        relies on global state (it and all the other LightGBM training processes
    #        coordinate with each other)
    futures_classifiers = [
        client.submit(
            _train_part,
            model_factory=model_factory,
            params={**params, "num_threads": worker_ncores[worker]},
            list_of_parts=list_of_parts,
            machines=machines,
            local_listen_port=worker_address_to_port[worker],
            num_machines=num_machines,
            time_out=params.get("time_out", 120),
            remote_socket=worker_to_socket_future.get(worker, None),
            return_model=(worker == master_worker),
            workers=[worker],
            allow_other_workers=False,
            pure=False,
            **kwargs,
        )
        for worker, list_of_parts in worker_map.items()
    ]

    results = client.gather(futures_classifiers)
    results = [v for v in results if v]
    model = results[0]

    # if network parameters were changed during training, remove them from the
    # returned model so that they're generated dynamically on every run based
    # on the Dask cluster you're connected to and which workers have pieces of
    # the training data
    if not listen_port_in_params:
        for param in _ConfigAliases.get("local_listen_port"):
            model._other_params.pop(param, None)

    if not machines_in_params:
        for param in _ConfigAliases.get("machines"):
            model._other_params.pop(param, None)

    for param in _ConfigAliases.get("num_machines", "timeout"):
        model._other_params.pop(param, None)

    return model


def _predict_part(
    part: _DaskPart,
    *,
    model: LGBMModel,
    raw_score: bool,
    pred_proba: bool,
    pred_leaf: bool,
    pred_contrib: bool,
    **kwargs: Any,
) -> _DaskPart:
    result: _DaskPart
    if part.shape[0] == 0:
        result = np.array([])
    elif pred_proba:
        result = model.predict_proba(
            part,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )
    else:
        result = model.predict(
            part,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )

    # dask.DataFrame.map_partitions() expects each call to return a pandas DataFrame or Series
    if isinstance(part, pd_DataFrame):
        # assert that 'result' is an array, only necessary because predict(..., pred_contrib=True) on
        # sparse matrices returns a list.
        #
        # This can be removed when https://github.com/lightgbm-org/LightGBM/pull/6348 is resolved.
        error_msg = (
            f"predict(X) for lightgbm.dask estimators should always return an array, not '{type(result)}', when X is a pandas Dataframe. "
            "If you're seeing this message, it's a bug in lightgbm. Please report it at https://github.com/lightgbm-org/LightGBM/issues."
        )
        assert hasattr(result, "shape"), error_msg
        if len(result.shape) == 2:
            result = pd_DataFrame(result, index=part.index)
        else:
            result = pd_Series(result, index=part.index, name="predictions")

    return result


def _predict(
    *,
    model: LGBMModel,
    data: _DaskMatrixLike,
    client: "distributed.Client",
    raw_score: bool = False,
    pred_proba: bool = False,
    pred_leaf: bool = False,
    pred_contrib: bool = False,
    **kwargs: Any,
) -> Union["dask.array.Array", List["dask.array.Array"]]:
    """Inner predict routine.

    Parameters
    ----------
    model : lightgbm.LGBMClassifier, lightgbm.LGBMRegressor, or lightgbm.LGBMRanker class
        Fitted underlying model.
    data : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
        Input feature matrix.
    raw_score : bool, optional (default=False)
        Whether to predict raw scores.
    pred_proba : bool, optional (default=False)
        Should method return results of ``predict_proba`` (``pred_proba=True``) or ``predict`` (``pred_proba=False``).
    pred_leaf : bool, optional (default=False)
        Whether to predict leaf index.
    pred_contrib : bool, optional (default=False)
        Whether to predict feature contributions.
    **kwargs
        Other parameters passed to ``predict`` or ``predict_proba`` method.

    Returns
    -------
    predicted_result : Dask Array of shape = [n_samples] or shape = [n_samples, n_classes]
        The predicted values.
    X_leaves : Dask Array of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
        If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
    X_SHAP_values : Dask Array of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or (if multi-class and using sparse inputs) a list of ``n_classes`` Dask Arrays of shape = [n_samples, n_features + 1]
        If ``pred_contrib=True``, the feature contributions for each sample.
    """
    if not all((PANDAS_INSTALLED, SKLEARN_INSTALLED)):
        raise LightGBMError("pandas and scikit-learn are required for lightgbm.dask")

    try:
        import dask.array  # noqa: PLC0415
        import dask.bag  # noqa: PLC0415
        import dask.dataframe  # noqa: PLC0415
        from dask import delayed  # noqa: PLC0415
    except _DaskImportErrorTypes as err:
        raise LightGBMError("dask is required for lightgbm.dask") from err

    if isinstance(data, dask.dataframe.DataFrame):
        # Bind prediction options into the callable to give each mode distinct task keys.
        # ref: https://github.com/dask/dask/pull/12603
        predict_fn = partial(
            _predict_part,
            model=model,
            raw_score=raw_score,
            pred_proba=pred_proba,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )
        return data.map_partitions(predict_fn).values
    elif isinstance(data, dask.array.Array):
        # for multi-class classification with sparse matrices, pred_contrib predictions
        # are returned as a list of sparse matrices (one per class)
        num_classes = model._n_classes

        if num_classes > 2 and pred_contrib and isinstance(data._meta, ss.spmatrix):
            predict_function = partial(
                _predict_part,
                model=model,
                raw_score=False,
                pred_proba=pred_proba,
                pred_leaf=False,
                pred_contrib=True,
                **kwargs,
            )

            delayed_chunks = data.to_delayed()
            bag = dask.bag.from_delayed(delayed_chunks[:, 0])

            @delayed
            def _extract(items: List[Any], i: int) -> Any:
                return items[i]

            preds = bag.map_partitions(predict_function)

            # pred_contrib output will have one column per feature,
            # plus one more for the base value
            num_cols = model.n_features_ + 1

            nrows_per_chunk = data.chunks[0]
            out: List[List[dask.array.Array]] = [[] for _ in range(num_classes)]

            # need to tell Dask the expected type and shape of individual preds
            pred_meta = data._meta

            for j, partition in enumerate(preds.to_delayed()):
                for i in range(num_classes):
                    part = dask.array.from_delayed(
                        value=_extract(partition, i),
                        shape=(nrows_per_chunk[j], num_cols),
                        meta=pred_meta,
                    )
                    out[i].append(part)

            # by default, dask.array.concatenate() concatenates sparse arrays into a COO matrix
            # the code below is used instead to ensure that the sparse type is preserved during concatenation
            if isinstance(pred_meta, ss.csr_matrix):
                concat_fn = partial(ss.vstack, format="csr")
            elif isinstance(pred_meta, ss.csc_matrix):
                concat_fn = partial(ss.vstack, format="csc")
            else:
                concat_fn = ss.vstack

            # At this point, `out` is a list of lists of delayeds (each of which points to a matrix).
            # Concatenate them to return a list of Dask Arrays.
            out_arrays: List[dask.array.Array] = []
            for i in range(num_classes):
                out_arrays.append(
                    dask.array.from_delayed(
                        value=delayed(concat_fn)(out[i]),
                        shape=(data.shape[0], num_cols),
                        meta=pred_meta,
                    )
                )

            return out_arrays

        data_row = client.compute(data[[0]]).result()
        predict_fn = partial(
            _predict_part,
            model=model,
            raw_score=raw_score,
            pred_proba=pred_proba,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )
        pred_row = predict_fn(data_row)  # type: ignore[misc]
        chunks: Tuple[int, ...] = (data.chunks[0],)
        map_blocks_kwargs = {}
        if len(pred_row.shape) > 1:
            chunks += (pred_row.shape[1],)
        else:
            map_blocks_kwargs["drop_axis"] = 1
        return data.map_blocks(
            predict_fn,
            chunks=chunks,
            meta=pred_row,
            **map_blocks_kwargs,
        )
    else:
        raise TypeError(f"Data must be either Dask Array or Dask DataFrame. Got {type(data).__name__}.")


class _DaskLGBMModel:
    @property
    def client_(self) -> "distributed.Client":
        """:obj:`distributed.Client`: Dask client.

        This property can be passed in the constructor or updated
        with ``model.set_params(client=client)``.
        """
        if not getattr(self, "fitted_", False):
            raise LGBMNotFittedError("Cannot access property client_ before calling fit().")

        return _get_dask_client(client=self.client)

    def _lgb_dask_getstate(self) -> Dict[Any, Any]:
        """Remove un-picklable attributes before serialization."""
        client = self.__dict__.pop("client", None)
        self._other_params.pop("client", None)  # type: ignore[attr-defined]
        out = deepcopy(self.__dict__)
        out.update({"client": None})
        self.client = client
        return out

    def _lgb_dask_fit(
        self,
        *,
        model_factory: Type[LGBMModel],
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskVectorLike] = None,
        init_score: Optional[_DaskCollection] = None,
        group: Optional[_DaskVectorLike] = None,
        eval_set: Optional[List[Tuple[_DaskMatrixLike, _DaskCollection]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_X: Optional[Union[_DaskMatrixLike, Tuple[_DaskMatrixLike]]] = None,
        eval_y: Optional[Union[_DaskCollection, Tuple[_DaskCollection]]] = None,
        eval_sample_weight: Optional[List[_DaskVectorLike]] = None,
        eval_class_weight: Optional[List[Union[dict, str]]] = None,
        eval_init_score: Optional[List[_DaskCollection]] = None,
        eval_group: Optional[List[_DaskVectorLike]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        eval_at: Optional[Union[List[int], Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> "_DaskLGBMModel":
        if not all((PANDAS_INSTALLED, SKLEARN_INSTALLED)):
            raise LightGBMError("pandas and scikit-learn are required for lightgbm.dask")

        params = self.get_params(True)  # type: ignore[attr-defined]
        params.pop("client", None)

        model = _train(
            client=_get_dask_client(self.client),
            data=X,
            label=y,
            params=params,
            model_factory=model_factory,
            sample_weight=sample_weight,
            init_score=init_score,
            group=group,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_X=eval_X,
            eval_y=eval_y,
            eval_sample_weight=eval_sample_weight,
            eval_class_weight=eval_class_weight,
            eval_init_score=eval_init_score,
            eval_group=eval_group,
            eval_metric=eval_metric,
            eval_at=eval_at,
            **kwargs,
        )

        self.set_params(**model.get_params())  # type: ignore[attr-defined]
        self._lgb_dask_copy_extra_params(source=model, dest=self)  # type: ignore[attr-defined]

        return self

    def _lgb_dask_to_local(self, model_factory: Type[LGBMModel]) -> LGBMModel:
        params = self.get_params()  # type: ignore[attr-defined]
        params.pop("client", None)
        model = model_factory(**params)
        self._lgb_dask_copy_extra_params(source=self, dest=model)
        model._other_params.pop("client", None)
        return model

    @staticmethod
    def _lgb_dask_copy_extra_params(
        *,
        source: Union["_DaskLGBMModel", LGBMModel],
        dest: Union["_DaskLGBMModel", LGBMModel],
    ) -> None:
        params = source.get_params()  # type: ignore[union-attr]
        attributes = source.__dict__
        extra_param_names = set(attributes.keys()).difference(params.keys())
        for name in extra_param_names:
            setattr(dest, name, attributes[name])


class DaskLGBMClassifier(LGBMClassifier, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMClassifier."""

    def __init__(
        self,
        *,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        class_weight: Optional[Union[dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState, "np.random.Generator"]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        client: Optional["distributed.Client"] = None,
        **kwargs: Any,
    ):
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        client : distributed.Client or None, optional (default=None)
            Dask client.
            If ``None``, ``distributed.default_client()`` will be used at runtime.
            The Dask client used by this class will not be saved if the model object is pickled.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        self.client = client
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs,
        )

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_dask_getstate()

    def fit(  # type: ignore[override]
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskVectorLike] = None,
        init_score: Optional[_DaskCollection] = None,
        eval_set: Optional[List[Tuple[_DaskMatrixLike, _DaskCollection]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_DaskVectorLike]] = None,
        eval_class_weight: Optional[List[Union[dict, str]]] = None,
        eval_init_score: Optional[List[_DaskCollection]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        *,
        eval_X: Optional[Union[_DaskMatrixLike, Tuple[_DaskMatrixLike]]] = None,
        eval_y: Optional[Union[_DaskCollection, Tuple[_DaskCollection]]] = None,
        **kwargs: Any,
    ) -> "DaskLGBMClassifier":
        """
        Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
            Input feature matrix.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        y : Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        sample_weight : Dask Array or Dask Series of shape = [n_samples] or None, optional (default=None)
            Weights of training data. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : Dask Array or Dask Series of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task), or Dask Array or Dask DataFrame of shape = [n_samples, n_classes] (for multi-class task), or None, optional (default=None)
            Init score of training data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        eval_set : list or None, optional (default=None)
            .. deprecated:: 4.7.0
                A list of (X, y) tuple pairs to use as validation sets.
                Use ``eval_X`` and ``eval_y`` instead.
        eval_names : list of str, or None, optional (default=None)
            Unique identifiers for each evaluation dataset.
            Should be the same length as ``eval_set`` / ``eval_X``.
        eval_sample_weight : list of Dask Array or Dask Series, or None, optional (default=None)
            Weights of eval data. Weights should be non-negative.
        eval_class_weight : list or None, optional (default=None)
            Class weights of eval data.
        eval_init_score : list of Dask Array, Dask Series or Dask DataFrame (for multi-class task), or None, optional (default=None)
            Init score of eval data.
        eval_metric : str, callable, list or None, optional (default=None)
            If str, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note below for more details.
            If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
        feature_name : list of str, or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        **kwargs
            Other parameters passed through to ``LGBMClassifier.fit()``.

        Returns
        -------
        self : lightgbm.DaskLGBMClassifier
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (metric_name, metric_value, maximize) or
        list of (metric_name, metric_value, maximize):

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                In case of custom ``objective``, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            metric_name : str
                Unique identifier for the metric (e.g. "custom_adjusted_mse").
            metric_value : float
                Value of the evaluation metric.
            maximize : bool
                Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        self._lgb_dask_fit(
            model_factory=LGBMClassifier,
            X=X,
            y=y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_X=eval_X,
            eval_y=eval_y,
            eval_sample_weight=eval_sample_weight,
            eval_class_weight=eval_class_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            **kwargs,
        )
        return self

    def predict(
        self,
        X: _DaskMatrixLike,  # type: ignore[override]
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> "dask.array.Array":
        """
        Return the predicted value for each sample.

        Parameters
        ----------
        X : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
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

        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_result : Dask Array of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        X_leaves : Dask Array of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : Dask Array of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or (if multi-class and using sparse inputs) a list of ``n_classes`` Dask Arrays of shape = [n_samples, n_features + 1]
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        return _predict(
            model=self.to_local(),
            data=X,
            client=_get_dask_client(self.client),
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )

    def predict_proba(
        self,
        X: _DaskMatrixLike,  # type: ignore[override]
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> "dask.array.Array":
        """
        Return the predicted probability for each class for each sample.

        Parameters
        ----------
        X : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
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

        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_probability : Dask Array of shape = [n_samples] or shape = [n_samples, n_classes]
            The predicted values.
        X_leaves : Dask Array of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : Dask Array of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or (if multi-class and using sparse inputs) a list of ``n_classes`` Dask Arrays of shape = [n_samples, n_features + 1]
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        return _predict(
            model=self.to_local(),
            data=X,
            pred_proba=True,
            client=_get_dask_client(self.client),
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )

    def to_local(self) -> LGBMClassifier:
        """Create regular version of lightgbm.LGBMClassifier from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMClassifier
            Local underlying model.
        """
        return self._lgb_dask_to_local(LGBMClassifier)


class DaskLGBMRegressor(LGBMRegressor, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMRegressor."""

    def __init__(
        self,
        *,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        class_weight: Optional[Union[dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState, "np.random.Generator"]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        client: Optional["distributed.Client"] = None,
        **kwargs: Any,
    ):
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        client : distributed.Client or None, optional (default=None)
            Dask client.
            If ``None``, ``distributed.default_client()`` will be used at runtime.
            The Dask client used by this class will not be saved if the model object is pickled.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        self.client = client
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs,
        )

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_dask_getstate()

    def fit(  # type: ignore[override]
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskVectorLike] = None,
        init_score: Optional[_DaskVectorLike] = None,
        eval_set: Optional[List[Tuple[_DaskMatrixLike, _DaskCollection]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_DaskVectorLike]] = None,
        eval_init_score: Optional[List[_DaskVectorLike]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        *,
        eval_X: Optional[Union[_DaskMatrixLike, Tuple[_DaskMatrixLike]]] = None,
        eval_y: Optional[Union[_DaskCollection, Tuple[_DaskCollection]]] = None,
        **kwargs: Any,
    ) -> "DaskLGBMRegressor":
        """
        Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
            Input feature matrix.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        y : Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        sample_weight : Dask Array or Dask Series of shape = [n_samples] or None, optional (default=None)
            Weights of training data. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : Dask Array or Dask Series of shape = [n_samples] or None, optional (default=None)
            Init score of training data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        eval_set : list or None, optional (default=None)
            .. deprecated:: 4.7.0
                A list of (X, y) tuple pairs to use as validation sets.
                Use ``eval_X`` and ``eval_y`` instead.
        eval_names : list of str, or None, optional (default=None)
            Unique identifiers for each evaluation dataset.
            Should be the same length as ``eval_set`` / ``eval_X``.
        eval_sample_weight : list of Dask Array or Dask Series, or None, optional (default=None)
            Weights of eval data. Weights should be non-negative.
        eval_init_score : list of Dask Array or Dask Series, or None, optional (default=None)
            Init score of eval data.
        eval_metric : str, callable, list or None, optional (default=None)
            If str, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note below for more details.
            If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
        feature_name : list of str, or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        **kwargs
            Other parameters passed through to ``LGBMRegressor.fit()``.

        Returns
        -------
        self : lightgbm.DaskLGBMRegressor
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (metric_name, metric_value, maximize) or
        list of (metric_name, metric_value, maximize):

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                In case of custom ``objective``, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            metric_name : str
                Unique identifier for the metric (e.g. "custom_adjusted_mse").
            metric_value : float
                Value of the evaluation metric.
            maximize : bool
                Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        self._lgb_dask_fit(
            model_factory=LGBMRegressor,
            X=X,
            y=y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_X=eval_X,
            eval_y=eval_y,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            **kwargs,
        )
        return self

    def predict(
        self,
        X: _DaskMatrixLike,  # type: ignore[override]
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> "dask.array.Array":
        """
        Return the predicted value for each sample.

        Parameters
        ----------
        X : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
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

        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_result : Dask Array of shape = [n_samples]
            The predicted values.
        X_leaves : Dask Array of shape = [n_samples, n_trees]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : Dask Array of shape = [n_samples, n_features + 1]
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        return _predict(
            model=self.to_local(),
            data=X,
            client=_get_dask_client(self.client),
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )

    def to_local(self) -> LGBMRegressor:
        """Create regular version of lightgbm.LGBMRegressor from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRegressor
            Local underlying model.
        """
        return self._lgb_dask_to_local(LGBMRegressor)


class DaskLGBMRanker(LGBMRanker, _DaskLGBMModel):
    """Distributed version of lightgbm.LGBMRanker."""

    def __init__(
        self,
        *,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        class_weight: Optional[Union[dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState, "np.random.Generator"]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        client: Optional["distributed.Client"] = None,
        **kwargs: Any,
    ):
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
            If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        client : distributed.Client or None, optional (default=None)
            Dask client.
            If ``None``, ``distributed.default_client()`` will be used at runtime.
            The Dask client used by this class will not be saved if the model object is pickled.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        self.client = client
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            **kwargs,
        )

    def __getstate__(self) -> Dict[Any, Any]:
        return self._lgb_dask_getstate()

    def fit(  # type: ignore[override]
        self,
        X: _DaskMatrixLike,
        y: _DaskCollection,
        sample_weight: Optional[_DaskVectorLike] = None,
        init_score: Optional[_DaskVectorLike] = None,
        group: Optional[_DaskVectorLike] = None,
        eval_set: Optional[List[Tuple[_DaskMatrixLike, _DaskCollection]]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_DaskVectorLike]] = None,
        eval_init_score: Optional[List[_DaskVectorLike]] = None,
        eval_group: Optional[List[_DaskVectorLike]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        eval_at: Union[List[int], Tuple[int, ...]] = (1, 2, 3, 4, 5),
        *,
        eval_X: Optional[Union[_DaskMatrixLike, Tuple[_DaskMatrixLike]]] = None,
        eval_y: Optional[Union[_DaskCollection, Tuple[_DaskCollection]]] = None,
        **kwargs: Any,
    ) -> "DaskLGBMRanker":
        """
        Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
            Input feature matrix.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        y : Dask Array, Dask DataFrame or Dask Series of shape = [n_samples]
            The target values (class labels in classification, real numbers in regression).

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        sample_weight : Dask Array or Dask Series of shape = [n_samples] or None, optional (default=None)
            Weights of training data. Weights should be non-negative.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        init_score : Dask Array or Dask Series of shape = [n_samples] or None, optional (default=None)
            Init score of training data.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        group : Dask Array or Dask Series or None, optional (default=None)
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.

            .. versionadded:: 4.2.0
                Support for ``pyarrow`` inputs

            .. versionadded:: 4.7.0
                Support for ``polars`` inputs

        eval_set : list or None, optional (default=None)
            .. deprecated:: 4.7.0
                A list of (X, y) tuple pairs to use as validation sets.
                Use ``eval_X`` and ``eval_y`` instead.
        eval_names : list of str, or None, optional (default=None)
            Unique identifiers for each evaluation dataset.
            Should be the same length as ``eval_set`` / ``eval_X``.
        eval_sample_weight : list of Dask Array or Dask Series, or None, optional (default=None)
            Weights of eval data. Weights should be non-negative.
        eval_init_score : list of Dask Array or Dask Series, or None, optional (default=None)
            Init score of eval data.
        eval_group : list of Dask Array or Dask Series, or None, optional (default=None)
            Group data of eval data.
        eval_metric : str, callable, list or None, optional (default=None)
            If str, it should be a built-in evaluation metric to use.
            If callable, it should be a custom evaluation metric, see note below for more details.
            If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
            In either case, the ``metric`` from the model parameters will be evaluated and used as well.
            Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
        eval_at : list or tuple of int, optional (default=(1, 2, 3, 4, 5))
            The evaluation positions of the specified metric.
        feature_name : list of str, or 'auto', optional (default='auto')
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default='auto')
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        **kwargs
            Other parameters passed through to ``LGBMRanker.fit()``.

        Returns
        -------
        self : lightgbm.DaskLGBMRanker
            Returns self.

        Note
        ----
        Custom eval function expects a callable with following signatures:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
        ``func(y_true, y_pred, weight, group)``
        and returns (metric_name, metric_value, maximize) or
        list of (metric_name, metric_value, maximize):

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                In case of custom ``objective``, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            metric_name : str
                Unique identifier for the metric (e.g. "custom_adjusted_mse").
            metric_value : float
                Value of the evaluation metric.
            maximize : bool
                Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
        """
        self._lgb_dask_fit(
            model_factory=LGBMRanker,
            X=X,
            y=y,
            sample_weight=sample_weight,
            init_score=init_score,
            group=group,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_X=eval_X,
            eval_y=eval_y,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_group=eval_group,
            eval_metric=eval_metric,
            eval_at=eval_at,
            **kwargs,
        )
        return self

    def predict(
        self,
        X: _DaskMatrixLike,  # type: ignore[override]
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> "dask.array.Array":
        """
        Return the predicted value for each sample.

        Parameters
        ----------
        X : Dask Array or Dask DataFrame of shape = [n_samples, n_features]
            Input features matrix.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
            If <= 0, starts from the first iteration.
        num_iteration : int or None, optional (default=None)
            Total number of iterations used in the prediction.
            If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
            otherwise, all iterations from ``start_iteration`` are used (no limits).
            If <= 0, all iterations from ``start_iteration`` are used (no limits).
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

        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        **kwargs
            Other parameters for the prediction.

        Returns
        -------
        predicted_result : Dask Array of shape = [n_samples]
            The predicted values.
        X_leaves : Dask Array of shape = [n_samples, n_trees]
            If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
        X_SHAP_values : Dask Array of shape = [n_samples, n_features + 1]
            If ``pred_contrib=True``, the feature contributions for each sample.
        """
        return _predict(
            model=self.to_local(),
            data=X,
            client=_get_dask_client(self.client),
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
            **kwargs,
        )

    def to_local(self) -> LGBMRanker:
        """Create regular version of lightgbm.LGBMRanker from the distributed version.

        Returns
        -------
        model : lightgbm.LGBMRanker
            Local underlying model.
        """
        return self._lgb_dask_to_local(LGBMRanker)
