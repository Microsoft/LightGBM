# coding: utf-8
"""Callbacks library."""

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from .basic import (
    Booster,
    EvalResult,
    _ConfigAliases,
    _log_info,
    _log_warning,
)

if TYPE_CHECKING:
    from .engine import CVBooster

__all__ = [
    "EarlyStopException",
    "early_stopping",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
]

_EvalResultDict = Dict[str, Dict[str, List[Any]]]
_EvalResultList = Union[
    List[EvalResult],
    List[Tuple[str, str, float, bool]],
    List[Tuple[str, str, float, bool, float]],
]


class EarlyStopException(Exception):
    """Exception of early stopping.

    Raise this from a callback passed in via keyword argument ``callbacks``
    in ``cv()`` or ``train()`` to trigger early stopping.
    """

    def __init__(self, best_iteration: int, best_score: _EvalResultList) -> None:
        """Create early stopping exception.

        Parameters
        ----------
        best_iteration : int
            The best iteration stopped.
            0-based... pass ``best_iteration=2`` to indicate that the third iteration was the best one.
        best_score : list of (dataset_name, metric_name, metric_value, maximize) tuple, (dataset_name, metric_name, metric_value, maximize, metric_std_dev) tuple, or ``lightgbm.EvalResult``
            Scores for each metric, on each validation set, as of the best iteration.

        Notes
        -----

        .. versionadded:: 4.7.0
            ``best_score`` is stored on the instance as a list of ``lightgbm.EvalResult`` objects.
        """
        super().__init__()
        self.best_iteration = best_iteration
        self.best_score: List[EvalResult] = [EvalResult(*score_tuple) for score_tuple in best_score]


# Callback environment used by callbacks
@dataclass
class CallbackEnv:
    model: Union[Booster, "CVBooster"]
    params: Dict[str, Any]
    iteration: int
    begin_iteration: int
    end_iteration: int
    evaluation_result_list: Optional[List[EvalResult]]


def _is_using_cv(env: CallbackEnv) -> bool:
    """Check if model in callback env is a CVBooster."""
    # this import is here to avoid a circular import
    from .engine import CVBooster  # noqa: PLC0415

    return isinstance(env.model, CVBooster)


def _format_eval_result(value: EvalResult, show_stdv: bool) -> str:
    """Format metric string."""
    out = f"{value.dataset_name}'s {value.metric_name}: {value.metric_value:g}"
    # tuples from cv() sometimes have a 5th item, with standard deviation of
    # the evaluation metric (taken over all cross-validation folds)
    if show_stdv and value.metric_std_dev is not None:
        out += f" + {value.metric_std_dev:g}"
    return out


class _LogEvaluationCallback:
    """Internal log evaluation callable class."""

    def __init__(self, period: int = 1, show_stdv: bool = True) -> None:
        self.order = 10
        self.before_iteration = False

        self.period = period
        self.show_stdv = show_stdv

    def __call__(self, env: CallbackEnv) -> None:
        if self.period > 0 and env.evaluation_result_list and (env.iteration + 1) % self.period == 0:
            result = "\t".join([_format_eval_result(x, self.show_stdv) for x in env.evaluation_result_list])
            _log_info(f"[{env.iteration + 1}]\t{result}")


def log_evaluation(period: int = 1, show_stdv: bool = True) -> _LogEvaluationCallback:
    """Create a callback that logs the evaluation results.

    By default, standard output resource is used.
    Use ``register_logger()`` function to register a custom logger.

    Notes
    -----
    Requires at least one validation data.

    Parameters
    ----------
    period : int, optional (default=1)
        The period to log the evaluation results.
        The last boosting stage or the boosting stage found by using ``early_stopping`` callback is also logged.
    show_stdv : bool, optional (default=True)
        Whether to log stdv (if provided).

    Returns
    -------
    callback : _LogEvaluationCallback
        The callback that logs the evaluation results every ``period`` boosting iteration(s).
    """
    return _LogEvaluationCallback(period=period, show_stdv=show_stdv)


class _RecordEvaluationCallback:
    """Internal record evaluation callable class."""

    def __init__(self, eval_result: _EvalResultDict) -> None:
        self.order = 20
        self.before_iteration = False

        if not isinstance(eval_result, dict):
            raise TypeError("eval_result should be a dictionary")
        self.eval_result = eval_result

    def _init(self, env: CallbackEnv) -> None:
        if env.evaluation_result_list is None:
            raise RuntimeError(
                "record_evaluation() callback enabled but no evaluation results found. This is a probably bug in LightGBM. "
                "Please report it at https://github.com/lightgbm-org/LightGBM/issues"
            )
        self.eval_result.clear()
        for item in env.evaluation_result_list:
            self.eval_result.setdefault(item.dataset_name, OrderedDict())
            if item.is_cv_result():
                # cv()
                self.eval_result[item.dataset_name].setdefault(f"{item.metric_name}-mean", [])
                self.eval_result[item.dataset_name].setdefault(f"{item.metric_name}-stdv", [])
            else:
                # train()
                self.eval_result[item.dataset_name].setdefault(item.metric_name, [])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if env.evaluation_result_list is None:
            raise RuntimeError(
                "record_evaluation() callback enabled but no evaluation results found. This is a probably bug in LightGBM. "
                "Please report it at https://github.com/lightgbm-org/LightGBM/issues"
            )
        for item in env.evaluation_result_list:
            if item.is_cv_result():
                # cv()
                self.eval_result[item.dataset_name][f"{item.metric_name}-mean"].append(item.metric_value)
                self.eval_result[item.dataset_name][f"{item.metric_name}-stdv"].append(item.metric_std_dev)
            else:
                # train()
                self.eval_result[item.dataset_name][item.metric_name].append(item.metric_value)


def record_evaluation(eval_result: Dict[str, Dict[str, List[Any]]]) -> Callable:
    """Create a callback that records the evaluation history into ``eval_result``.

    Parameters
    ----------
    eval_result : dict
        Dictionary used to store all evaluation results of all validation sets.
        This should be initialized outside of your call to ``record_evaluation()`` and should be empty.
        Any initial contents of the dictionary will be deleted.

        .. rubric:: Example

        With two validation sets named 'eval' and 'train', and one evaluation metric named 'logloss'
        this dictionary after finishing a model training process will have the following structure:

        .. code-block::

            {
             'train':
                 {
                  'logloss': [0.48253, 0.35953, ...]
                 },
             'eval':
                 {
                  'logloss': [0.480385, 0.357756, ...]
                 }
            }

    Returns
    -------
    callback : _RecordEvaluationCallback
        The callback that records the evaluation history into the passed dictionary.
    """
    return _RecordEvaluationCallback(eval_result=eval_result)


class _ResetParameterCallback:
    """Internal reset parameter callable class."""

    def __init__(self, **kwargs: Union[list, Callable]) -> None:
        self.order = 10
        self.before_iteration = True

        self.kwargs = kwargs

    def __call__(self, env: CallbackEnv) -> None:
        new_parameters = {}
        for key, value in self.kwargs.items():
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError(f"Length of list {key!r} has to be equal to 'num_boost_round'.")
                new_param = value[env.iteration - env.begin_iteration]
            elif callable(value):
                new_param = value(env.iteration - env.begin_iteration)
            else:
                raise ValueError(
                    "Only list and callable values are supported "
                    "as a mapping from boosting round index to new parameter value."
                )
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            if isinstance(env.model, Booster):
                env.model.reset_parameter(new_parameters)
            else:
                # CVBooster holds a list of Booster objects, each needs to be updated
                for booster in env.model.boosters:
                    booster.reset_parameter(new_parameters)
            env.params.update(new_parameters)


def reset_parameter(**kwargs: Union[list, Callable]) -> Callable:
    """Create a callback that resets the parameter after the first iteration.

    .. note::

        The initial parameter will still take in-effect on first iteration.

    Parameters
    ----------
    **kwargs : value should be list or callable
        List of parameters for each boosting round
        or a callable that calculates the parameter in terms of
        current number of round (e.g. yields learning rate decay).
        If list lst, parameter = lst[current_round].
        If callable func, parameter = func(current_round).

    Returns
    -------
    callback : _ResetParameterCallback
        The callback that resets the parameter after the first iteration.
    """
    return _ResetParameterCallback(**kwargs)


class _EarlyStoppingCallback:
    """Internal early stopping callable class."""

    def __init__(
        self,
        stopping_rounds: int,
        first_metric_only: bool = False,
        verbose: bool = True,
        min_delta: Union[float, List[float]] = 0.0,
    ) -> None:
        self.enabled = _should_enable_early_stopping(stopping_rounds)

        self.order = 30
        self.before_iteration = False

        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta

        self._reset_storages()

    def _reset_storages(self) -> None:
        self.best_score: List[float] = []
        self.best_iter: List[int] = []
        self.best_score_list: List[List[EvalResult]] = []
        self.cmp_op: List[Callable[[float, float, float], bool]] = []
        self.first_metric = ""

    def _gt_delta(self, *, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score > best_score + delta

    def _lt_delta(self, *, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score < best_score - delta

    def _is_train_set(self, *, dataset_name: str, env: CallbackEnv) -> bool:
        """Check, by name, if a given Dataset is the training data."""
        # for lgb.cv() with eval_train_metric=True, evaluation is also done on the training set
        # and those metrics are considered for early stopping
        if _is_using_cv(env) and dataset_name == "train":
            return True

        # for lgb.train(), it's possible to pass the training data via valid_sets with any name
        if isinstance(env.model, Booster) and dataset_name == env.model._train_data_name:
            return True

        return False

    def _init(self, env: CallbackEnv) -> None:
        if env.evaluation_result_list is None or env.evaluation_result_list == []:
            raise ValueError("For early stopping, at least one dataset and eval metric is required for evaluation")

        # re-evaluate here, so a callback re-used across training runs is not left
        # permanently disabled by an earlier run (for example, one using dart boosting)
        self.enabled = _should_enable_early_stopping(self.stopping_rounds)

        is_dart = any(env.params.get(alias, "") == "dart" for alias in _ConfigAliases.get("boosting"))
        if is_dart:
            self.enabled = False
            _log_warning("Early stopping is not available in dart mode")
            return

        # get details of the first dataset
        first_dataset_name = env.evaluation_result_list[0].dataset_name
        first_metric_name = env.evaluation_result_list[0].metric_name

        # validation sets are guaranteed to not be identical to the training data in cv()
        if isinstance(env.model, Booster):
            only_train_set = len(env.evaluation_result_list) == 1 and self._is_train_set(
                dataset_name=first_dataset_name,
                env=env,
            )
            if only_train_set:
                self.enabled = False
                _log_warning("Only training set found, disabling early stopping.")
                return

        if self.verbose:
            _log_info(f"Training until validation scores don't improve for {self.stopping_rounds} rounds")

        self._reset_storages()

        n_metrics = len({m.metric_name for m in env.evaluation_result_list})
        n_datasets = len(env.evaluation_result_list) // n_metrics
        if isinstance(self.min_delta, list):
            if not all(t >= 0 for t in self.min_delta):
                raise ValueError("Values for early stopping min_delta must be non-negative.")
            if len(self.min_delta) == 0:
                if self.verbose:
                    _log_info("Disabling min_delta for early stopping.")
                deltas = [0.0] * n_datasets * n_metrics
            elif len(self.min_delta) == 1:
                if self.verbose:
                    _log_info(f"Using {self.min_delta[0]} as min_delta for all metrics.")
                deltas = self.min_delta * n_datasets * n_metrics
            else:
                if len(self.min_delta) != n_metrics:
                    raise ValueError("Must provide a single value for min_delta or as many as metrics.")
                if self.first_metric_only and self.verbose:
                    _log_info(f"Using only {self.min_delta[0]} as early stopping min_delta.")
                deltas = self.min_delta * n_datasets
        else:
            if self.min_delta < 0:
                raise ValueError("Early stopping min_delta must be non-negative.")
            if self.min_delta > 0 and n_metrics > 1 and not self.first_metric_only and self.verbose:
                _log_info(f"Using {self.min_delta} as min_delta for all metrics.")
            deltas = [self.min_delta] * n_datasets * n_metrics

        self.first_metric = first_metric_name
        for eval_ret, delta in zip(env.evaluation_result_list, deltas, strict=True):
            self.best_iter.append(0)
            if eval_ret.maximize:
                self.best_score.append(float("-inf"))
                self.cmp_op.append(partial(self._gt_delta, delta=delta))
            else:
                self.best_score.append(float("inf"))
                self.cmp_op.append(partial(self._lt_delta, delta=delta))

    def _final_iteration_check(self, *, env: CallbackEnv, metric_name: str, i: int) -> None:
        if env.iteration == env.end_iteration - 1:
            if self.verbose:
                best_score_str = "\t".join([_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]])
                _log_info(
                    f"Did not meet early stopping. Best iteration is:\n[{self.best_iter[i] + 1}]\t{best_score_str}"
                )
                if self.first_metric_only:
                    _log_info(f"Evaluated only: {metric_name}")
            raise EarlyStopException(
                best_iteration=self.best_iter[i],
                best_score=self.best_score_list[i],
            )

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        if env.evaluation_result_list is None:
            raise RuntimeError(
                "early_stopping() callback enabled but no evaluation results found. This is a probably bug in LightGBM. "
                "Please report it at https://github.com/lightgbm-org/LightGBM/issues"
            )
        # self.best_score_list is initialized to an empty list
        first_time_updating_best_score_list = self.best_score_list == []
        for i in range(len(env.evaluation_result_list)):
            eval_result = env.evaluation_result_list[i]
            if first_time_updating_best_score_list or self.cmp_op[i](  # type: ignore[call-arg]
                curr_score=eval_result.metric_value, best_score=self.best_score[i]
            ):
                self.best_score[i] = eval_result.metric_value
                self.best_iter[i] = env.iteration
                if first_time_updating_best_score_list:
                    self.best_score_list.append(env.evaluation_result_list)
                else:
                    self.best_score_list[i] = env.evaluation_result_list
            if self.first_metric_only and self.first_metric != eval_result.metric_name:
                continue  # use only the first metric for early stopping
            if self._is_train_set(
                dataset_name=eval_result.dataset_name,
                env=env,
            ):
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)
            elif env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    eval_result_str = "\t".join(
                        [_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]]
                    )
                    _log_info(f"Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t{eval_result_str}")
                    if self.first_metric_only:
                        _log_info(f"Evaluated only: {eval_result.metric_name}")
                raise EarlyStopException(
                    best_iteration=self.best_iter[i],
                    best_score=self.best_score_list[i],
                )
            self._final_iteration_check(env=env, metric_name=eval_result.metric_name, i=i)


def _should_enable_early_stopping(stopping_rounds: Any) -> bool:
    """Check if early stopping should be activated.

    This function will evaluate to True if the early stopping callback should be
    activated (i.e. stopping_rounds > 0).  It also provides an informative error if the
    type is not int.
    """
    if not isinstance(stopping_rounds, int):
        raise TypeError(f"early_stopping_round should be an integer. Got '{type(stopping_rounds).__name__}'")
    return stopping_rounds > 0


def early_stopping(
    stopping_rounds: int,
    first_metric_only: bool = False,
    verbose: bool = True,
    min_delta: Union[float, List[float]] = 0.0,
) -> _EarlyStoppingCallback:
    """Create a callback that activates early stopping.

    Activates early stopping.
    The model will train until the validation score doesn't improve by at least ``min_delta``.
    Validation score needs to improve at least every ``stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.
    The index of iteration that has the best performance will be saved in the ``best_iteration`` attribute of a model.

    .. note::

        If using ``boosting_type="dart"``, this callback has no effect and early stopping
        will not be performed.

    Parameters
    ----------
    stopping_rounds : int
        The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
        Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to log message with early stopping information.
        By default, standard output resource is used.
        Use ``register_logger()`` function to register a custom logger.
    min_delta : float or list of float, optional (default=0.0)
        Minimum improvement in score to keep training.
        If float, this single value is used for all metrics.
        If list, its length should match the total number of metrics.

        .. versionadded:: 4.0.0

    Returns
    -------
    callback : _EarlyStoppingCallback
        The callback that activates early stopping.
    """
    return _EarlyStoppingCallback(
        stopping_rounds=stopping_rounds,
        first_metric_only=first_metric_only,
        verbose=verbose,
        min_delta=min_delta,
    )
