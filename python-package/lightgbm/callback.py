# coding: utf-8
# pylint: disable = invalid-name, W0105, C0301
"""Callbacks library."""
from __future__ import absolute_import

import collections
import warnings
from operator import gt, lt

from .compat import range_


class EarlyStopException(Exception):
    """Exception of early stopping."""

    def __init__(self, best_iteration, best_score):
        """Create early stopping exception.

        Parameters
        ----------
        best_iteration : int
            The best iteration stopped.
        best_score : float
            The score of the best iteration.
        """
        super(EarlyStopException, self).__init__()
        self.best_iteration = best_iteration
        self.best_score = best_score


# Callback environment used by callbacks
CallbackEnv = collections.namedtuple(
    "LightGBMCallbackEnv",
    ["model",
     "params",
     "iteration",
     "begin_iteration",
     "end_iteration",
     "evaluation_result_list"])


def _format_eval_result(value, show_stdv=True):
    """Format metric string."""
    if len(value) == 4:
        return '%s\'s %s: %g' % (value[0], value[1], value[2])
    elif len(value) == 5:
        if show_stdv:
            return '%s\'s %s: %g + %g' % (value[0], value[1], value[2], value[4])
        else:
            return '%s\'s %s: %g' % (value[0], value[1], value[2])
    else:
        raise ValueError("Wrong metric value")


def print_evaluation(period=1, show_stdv=True):
    """Create a callback that prints the evaluation results.

    Parameters
    ----------
    period : int, optional (default=1)
        The period to print the evaluation results.
    show_stdv : bool, optional (default=True)
        Whether to show stdv (if provided).

    Returns
    -------
    callback : function
        The callback that prints the evaluation results every ``period`` iteration(s).
    """
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            print('[%d]\t%s' % (env.iteration + 1, result))
    _callback.order = 10
    return _callback


def record_evaluation(eval_result):
    """Create a callback that records the evaluation history into ``eval_result``.

    Parameters
    ----------
    eval_result : dict
       A dictionary to store the evaluation results.

    Returns
    -------
    callback : function
        The callback that records the evaluation history into the passed dictionary.
    """
    if not isinstance(eval_result, dict):
        raise TypeError('Eval_result should be a dictionary')
    eval_result.clear()

    def _init(env):
        for data_name, _, _, _ in env.evaluation_result_list:
            eval_result.setdefault(data_name, collections.OrderedDict())

    def _callback(env):
        if not eval_result:
            _init(env)
        processed_metric = collections.defaultdict(list)
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            if eval_name in processed_metric[data_name]:
                continue
            if eval_name not in eval_result[data_name].keys():
                eval_result[data_name][eval_name] = []
            eval_result[data_name][eval_name].append(result)
            processed_metric[data_name].append(eval_name)
    _callback.order = 20
    return _callback


def reset_parameter(**kwargs):
    """Create a callback that resets the parameter after the first iteration.

    Note
    ----
    The initial parameter will still take in-effect on first iteration.

    Parameters
    ----------
    **kwargs : value should be list or function
        List of parameters for each boosting round
        or a customized function that calculates the parameter in terms of
        current number of round (e.g. yields learning rate decay).
        If list lst, parameter = lst[current_round].
        If function func, parameter = func(current_round).

    Returns
    -------
    callback : function
        The callback that resets the parameter after the first iteration.
    """
    def _callback(env):
        new_parameters = {}
        for key, value in kwargs.items():
            if key in ['num_class', 'num_classes',
                       'boosting', 'boost', 'boosting_type',
                       'metric', 'metrics', 'metric_types']:
                raise RuntimeError("cannot reset {} during training".format(repr(key)))
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError("Length of list {} has to equal to 'num_boost_round'."
                                     .format(repr(key)))
                new_param = value[env.iteration - env.begin_iteration]
            else:
                new_param = value(env.iteration - env.begin_iteration)
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            env.model.reset_parameter(new_parameters)
            env.params.update(new_parameters)
    _callback.before_iteration = True
    _callback.order = 10
    return _callback


def early_stopping(stopping_rounds, first_metric_only=False, verbose=True):
    """Create a callback that activates early stopping.

    Note
    ----
    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.

    Parameters
    ----------
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
       Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information.

    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list = []
    cmp_op = []
    enabled = [True]
    eval_metric = [None]

    def _init(env):
        enabled[0] = not any((boost_alias in env.params
                              and env.params[boost_alias] == 'dart') for boost_alias in ('boosting',
                                                                                         'boosting_type',
                                                                                         'boost'))
        if not enabled[0]:
            warnings.warn('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')

        if verbose:
            msg = "Training until validation scores don't improve for {} rounds."
            print(msg.format(stopping_rounds))

        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)
        if first_metric_only:
            # Choosing a target metric for early stopping from first element of metrics list as `eval_metric`
            # if first_metric_only == True.
            for metric_alias in ['metric', 'metrics', 'metric_types']:
                if metric_alias in env.params.keys():
                    if isinstance(env.params[metric_alias], (tuple, list)):
                        metric_list = [m for m in env.params[metric_alias] if m is not None]
                        eval_metric[0] = metric_list[0] if len(metric_list) > 0 else None
                    else:
                        if env.evaluation_result_list[0][0] == "cv_agg":
                            # For lgb.cv
                            for i in range(len(env.evaluation_result_list)):
                                # Removing training data
                                if (env.evaluation_result_list[i][1].find(" ") >= 0
                                        and env.evaluation_result_list[i][1].split(" ")[0] != "train"):
                                    eval_metric[0] = env.evaluation_result_list[i][1].split(" ")[1]
                                    break
                        else:
                            # The other than lgb.cv
                            eval_metric[0] = env.params[metric_alias]
                    break
            if eval_metric[0] is None:
                # if `eval_metric` was not chose from `env.params`, using first element of env.evaluation_result_list
                # for early stopping
                eval_metric[0] = env.evaluation_result_list[0][1]
            eval_metric[0] = _metric_alias_matching(eval_metric[0])

    def _metric_alias_matching(target_metric):
        """Convert metric name for early stopping from alias to representative in order to match the param metric."""
        metrics_list = [("l1", {"mean_absolute_error", "mae", "regression_l1"}),
                        ("l2", {"mean_squared_error", "mse", "regression_l2", "regression"}),
                        ("rmse", {"root_mean_squared_error", "l2_root"}),
                        ("mape", {"mean_absolute_percentage_error"}),
                        ("ndcg", {"lambdarank"}),
                        ("map", {"mean_average_precision"}),
                        ("binary_logloss", {"binary"}),
                        ("multi_logloss",
                        {"multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr"}),
                        ("xentropy", {"cross_entropy"}),
                        ("xentlambda", {"cross_entropy_lambda"}),
                        ("kldiv", {"kullback_leibler"})]
        for metric, aliases in metrics_list:
            if target_metric in aliases:
                return metric
        return target_metric

    def _callback(env):
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        for i in range_(len(env.evaluation_result_list)):
            metric_key = env.evaluation_result_list[i][1]
            if env.evaluation_result_list[i][0] == "cv_agg":  # for lgb.cv
                if metric_key.split(" ")[0] == "train":
                    continue  # train metric is not used on early stopping
                if ((first_metric_only and eval_metric[0] is not None and eval_metric[0] != ""
                     and metric_key != "valid {}".format(eval_metric[0]) and metric_key != eval_metric[0])):
                    continue
            elif getattr(env.model, '__train_data_name', False) and env.model._train_data_name == "training":
                continue  # for training set, early stopping is not applied
            else:  # for lgb.train and sklearn wrapper
                if first_metric_only:
                    if metric_key != eval_metric[0]:
                        continue
            score = env.evaluation_result_list[i][2]
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
            elif env.iteration - best_iter[i] >= stopping_rounds:
                if verbose:
                    print('Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                    if first_metric_only:
                        print("Evaluated only: {}".format(metric_key))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                if verbose:
                    print('Did not meet early stopping. Best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                    if first_metric_only:
                        print("Evaluated only: {}".format(metric_key))
                raise EarlyStopException(best_iter[i], best_score_list[i])
    _callback.order = 30
    _callback.first_metric_only = first_metric_only
    return _callback
