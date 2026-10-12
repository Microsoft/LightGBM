#' @name lgb_shared_params
#' @title Shared parameter docs
#' @description Parameter docs shared by \code{lgb.train}, \code{lgb.cv}, and \code{lightgbm}
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param data a \code{lgb.Dataset} object, used for training. Some functions, such as \code{\link{lgb.cv}},
#'             may allow you to pass other types of data like \code{matrix} and then separately supply
#'             \code{label} as a keyword argument.
#' @param early_stopping_rounds int. Activates early stopping. When this parameter is non-null,
#'                              training will stop if the evaluation of any metric on any validation set
#'                              fails to improve for \code{early_stopping_rounds} consecutive boosting rounds.
#'                              If training stops early, the returned model will have attribute \code{best_iter}
#'                              set to the iteration number of the best iteration.
#' @param eval evaluation function(s). This can be a character vector, function, or list with a mixture of
#'             strings and functions.
#'
#'             \itemize{
#'                 \item{\bold{a. character vector}:
#'                     If you provide a character vector to this argument, it should contain strings with valid
#'                     evaluation metrics.
#'                     See \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric}{
#'                     The "metric" section of the documentation}
#'                     for a list of valid metrics.
#'                 }
#'                 \item{\bold{b. function}:
#'                      You can provide a custom evaluation function. This
#'                      should accept the keyword arguments \code{preds} and \code{dtrain} and should return a named
#'                      list with three elements:
#'                      \itemize{
#'                          \item{\code{name}: A string with the name of the metric, used for printing
#'                              and storing results.
#'                          }
#'                          \item{\code{value}: A single number indicating the value of the metric for the
#'                              given predictions and true values
#'                          }
#'                          \item{
#'                              \code{higher_better}: A boolean indicating whether higher values indicate a better fit.
#'                              For example, this would be \code{FALSE} for metrics like MAE or RMSE.
#'                          }
#'                      }
#'                 }
#'                 \item{\bold{c. list}:
#'                     If a list is given, it should only contain character vectors and functions.
#'                     These should follow the requirements from the descriptions above.
#'                 }
#'             }
#' @param eval_freq evaluation output frequency, only effective when verbose > 0 and \code{valids} has been provided
#' @param init_model path of model file or \code{lgb.Booster} object, will continue training from this model
#' @param nrounds number of training rounds
#' @param obj objective function, can be character or custom objective function. Examples include
#'            \code{regression}, \code{regression_l1}, \code{huber},
#'            \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}
#' @param params a list of parameters. See \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html}{
#'               the "Parameters" section of the documentation} for a list of parameters and valid values.
#' @param verbose verbosity for output, if <= 0 and \code{valids} has been provided, also will disable the
#'                printing of evaluation during training
#' @param serializable whether to make the resulting objects serializable through functions such as
#'                     \code{save} or \code{saveRDS} (see section "Model serialization").
#' @section Early Stopping:
#'
#'          "early stopping" refers to stopping the training process if the model's performance on a given
#'          validation set does not improve for several consecutive iterations.
#'
#'          If multiple arguments are given to \code{eval}, their order will be preserved. If you enable
#'          early stopping by setting \code{early_stopping_rounds} in \code{params}, by default all
#'          metrics will be considered for early stopping.
#'
#'          If you want to only consider the first metric for early stopping, pass
#'          \code{first_metric_only = TRUE} in \code{params}. Note that if you also specify \code{metric}
#'          in \code{params}, that metric will be considered the "first" one. If you omit \code{metric},
#'          a default metric will be used based on your choice for the parameter \code{obj} (keyword argument)
#'          or \code{objective} (passed into \code{params}).
#'
#'          \bold{NOTE:} if using \code{boosting_type="dart"}, any early stopping configuration will be ignored
#'          and early stopping will not be performed.
#' @section Model serialization:
#'
#'          LightGBM model objects can be serialized and de-serialized through functions such as \code{save}
#'          or \code{saveRDS}, but similarly to libraries such as 'xgboost', serialization works a bit differently
#'          from typical R objects. In order to make models serializable in R, a copy of the underlying C++ object
#'          as serialized raw bytes is produced and stored in the R model object, and when this R object is
#'          de-serialized, the underlying C++ model object gets reconstructed from these raw bytes, but will only
#'          do so once some function that uses it is called, such as \code{predict}. In order to forcibly
#'          reconstruct the C++ object after deserialization (e.g. after calling \code{readRDS} or similar), one
#'          can use the function \link{lgb.restore_handle} (for example, if one makes predictions in parallel or in
#'          forked processes, it will be faster to restore the handle beforehand).
#'
#'          Producing and keeping these raw bytes however uses extra memory, and if they are not required,
#'          it is possible to avoid producing them by passing `serializable=FALSE`. In such cases, these raw
#'          bytes can be added to the model on demand through function \link{lgb.make_serializable}.
#'
#'          \emph{New in version 4.0.0}
#'
#' @keywords internal
NULL

#' @name lightgbm
#' @title Train a LightGBM model
#' @description High-level R interface to train a LightGBM model. Unlike \code{\link{lgb.train}}, this function
#'              is focused on compatibility with other statistics and machine learning interfaces in R.
#'              This focus on compatibility means that this interface may experience more frequent breaking API changes
#'              than \code{\link{lgb.train}}.
#'              For efficiency-sensitive applications, or for applications where breaking API changes across releases
#'              is very expensive, use \code{\link{lgb.train}}.
#' @inheritParams lgb_shared_params
#' @param label Vector of labels, used if \code{data} is not an \code{\link{lgb.Dataset}}
#' @param weights Sample / observation weights for rows in the input data. If \code{NULL}, will assume that all
#'                observations / rows have the same importance / weight.
#'
#'                \emph{Changed from 'weight', in version 4.0.0}
#'
#' @param objective Optimization objective (e.g. `"regression"`, `"binary"`, etc.).
#'                  For a list of accepted objectives, see
#'                  \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective}{
#'                  the "objective" item of the "Parameters" section of the documentation}.
#'
#'                  If passing \code{"auto"} and \code{data} is not of type \code{lgb.Dataset}, the objective will
#'                  be determined according to what is passed for \code{label}:\itemize{
#'                  \item If passing a factor with two variables, will use objective \code{"binary"}.
#'                  \item If passing a factor with more than two variables, will use objective \code{"multiclass"}
#'                  (note that parameter \code{num_class} in this case will also be determined automatically from
#'                  \code{label}).
#'                  \item Otherwise (or if passing \code{lgb.Dataset} as input), will use objective \code{"regression"}.
#'                  }
#'
#'                  \emph{New in version 4.0.0}
#'
#' @param init_score initial score is the base prediction lightgbm will boost from
#'
#'                   \emph{New in version 4.0.0}
#'
#' @param num_threads Number of parallel threads to use. For best speed, this should be set to the number of
#'                    physical cores in the CPU - in a typical x86-64 machine, this corresponds to half the
#'                    number of maximum threads.
#'
#'                    Be aware that using too many threads can result in speed degradation in smaller datasets
#'                    (see the parameters documentation for more details).
#'
#'                    If passing zero, will use the default number of threads configured for OpenMP
#'                    (typically controlled through an environment variable \code{OMP_NUM_THREADS}).
#'
#'                    If passing \code{NULL} (the default), will try to use the number of physical cores in the
#'                    system, but be aware that getting the number of cores detected correctly requires package
#'                    \code{RhpcBLASctl} to be installed.
#'
#'                    This parameter gets overridden by \code{num_threads} and its aliases under \code{params}
#'                    if passed there.
#'
#'                    \emph{New in version 4.0.0}
#'
#' @param colnames Character vector of features. Only used if \code{data} is not an \code{\link{lgb.Dataset}}.
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#'                            Only used if \code{data} is not an \code{\link{lgb.Dataset}}.
#'
#' @param ... Additional arguments passed to \code{\link{lgb.train}}. For example
#'     \itemize{
#'        \item{\code{valids}: a list of \code{lgb.Dataset} objects, used for validation}
#'        \item{\code{obj}: objective function, can be character or custom objective function. Examples include
#'                   \code{regression}, \code{regression_l1}, \code{huber},
#'                    \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}}
#'        \item{\code{eval}: evaluation function, can be (a list of) character or custom eval function}
#'        \item{\code{record}: Boolean, TRUE will record iteration message to \code{booster$record_evals}}
#'        \item{\code{reset_data}: Boolean, setting it to TRUE (not the default value) will transform the booster model
#'                          into a predictor model which frees up memory and the original datasets}
#'     }
#' @inheritSection lgb_shared_params Early Stopping
#' @return a trained \code{lgb.Booster}
#' @export
lightgbm <- function(data,
                     label = NULL,
                     weights = NULL,
                     params = list(),
                     nrounds = 100L,
                     verbose = 1L,
                     eval_freq = 1L,
                     early_stopping_rounds = NULL,
                     init_model = NULL,
                     callbacks = list(),
                     serializable = TRUE,
                     objective = "auto",
                     init_score = NULL,
                     num_threads = NULL,
                     colnames = NULL,
                     categorical_feature = NULL,
                     ...) {

  # validate inputs early to avoid unnecessary computation
  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }

  if (is.null(num_threads)) {
    num_threads <- .get_default_num_threads()
  }
  params <- .check_wrapper_param(
    main_param_name = "num_threads"
    , params = params
    , alternative_kwarg_value = num_threads
  )
  params <- .check_wrapper_param(
    main_param_name = "verbosity"
    , params = params
    , alternative_kwarg_value = verbose
  )

  # Process factors as labels and auto-determine objective
  if (!.is_Dataset(data)) {
    data_processor <- DataProcessor$new()
    temp <- data_processor$process_label(
        label = label
        , objective = objective
        , params = params
    )
    label <- temp$label
    objective <- temp$objective
    params <- temp$params
    rm(temp)
  } else {
    data_processor <- NULL
    if (objective == "auto") {
      objective <- "regression"
    }
  }

  # Set data to a temporary variable
  dtrain <- data

  # Check whether data is lgb.Dataset, if not then create lgb.Dataset manually
  if (!.is_Dataset(x = dtrain)) {
    dtrain <- lgb.Dataset(
      data = data
      , label = label
      , weight = weights
      , init_score = init_score
      , categorical_feature = categorical_feature
      , colnames = colnames
    )
  }

  train_args <- list(
    "params" = params
    , "data" = dtrain
    , "nrounds" = nrounds
    , "obj" = objective
    , "verbose" = params[["verbosity"]]
    , "eval_freq" = eval_freq
    , "early_stopping_rounds" = early_stopping_rounds
    , "init_model" = init_model
    , "callbacks" = callbacks
    , "serializable" = serializable
  )
  train_args <- append(train_args, list(...))

  if (! "valids" %in% names(train_args)) {
    train_args[["valids"]] <- list()
  }

  # Train a model using the regular way
  bst <- do.call(
    what = lgb.train
    , args = train_args
  )
  bst$data_processor <- data_processor

  return(bst)
}

#' @name agaricus.train
#' @title Training part from Mushroom Data Set
#' @description This data set is originally from the Mushroom data set,
#'              UCI Machine Learning Repository.
#'              This data set includes the following fields:
#'
#'               \itemize{
#'                   \item{\code{label}: the label for each record}
#'                   \item{\code{data}: a sparse Matrix of \code{dgCMatrix} class, with 126 columns.}
#'                }
#'
#' @references
#' https://archive.ics.uci.edu/dataset/73/mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [https://archive.ics.uci.edu/]. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @usage data(agaricus.train)
#' @format A list containing a label vector, and a dgCMatrix object with 6513
#' rows and 127 variables
NULL

#' @name agaricus.test
#' @title Test part from Mushroom Data Set
#' @description This data set is originally from the Mushroom data set,
#'              UCI Machine Learning Repository.
#'              This data set includes the following fields:
#'
#'              \itemize{
#'                  \item{\code{label}: the label for each record}
#'                  \item{\code{data}: a sparse Matrix of \code{dgCMatrix} class, with 126 columns.}
#'              }
#' @references
#' https://archive.ics.uci.edu/dataset/73/mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @usage data(agaricus.test)
#' @format A list containing a label vector, and a dgCMatrix object with 1611
#' rows and 126 variables
NULL

#' @name bank
#' @title Bank Marketing Data Set
#' @description This data set is originally from the Bank Marketing data set,
#'              UCI Machine Learning Repository.
#'
#'              It contains only the following: bank.csv with 10% of the examples and 17 inputs,
#'              randomly selected from 3 (older version of this dataset with less inputs).
#'
#' @references
#' https://archive.ics.uci.edu/dataset/222/bank+marketing
#'
#' S. Moro, P. Cortez and P. Rita. (2014)
#' A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems
#'
#' @docType data
#' @keywords datasets
#' @usage data(bank)
#' @format A data.table with 4521 rows and 17 variables
NULL

# Various imports
#' @import methods
#' @importFrom Matrix Matrix
#' @importFrom R6 R6Class
#' @useDynLib lightgbm , .registration = TRUE
NULL

# Suppress false positive warnings from R CMD CHECK about
# "unrecognized global variable"
globalVariables(c(
    "."
    , ".N"
    , ".SD"
    , "abs_contribution"
    , "bar_color"
    , "Contribution"
    , "Cover"
    , "Feature"
    , "Frequency"
    , "Gain"
    , "internal_count"
    , "internal_value"
    , "leaf_index"
    , "leaf_parent"
    , "leaf_value"
    , "node_parent"
    , "split_feature"
    , "split_gain"
    , "split_index"
    , "tree_index"
))
