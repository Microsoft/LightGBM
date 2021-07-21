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
#' @param eval_freq evaluation output frequency, only effect when verbose > 0
#' @param init_model path of model file of \code{lgb.Booster} object, will continue training from this model
#' @param nrounds number of training rounds
#' @param obj objective function, can be character or custom objective function. Examples include
#'            \code{regression}, \code{regression_l1}, \code{huber},
#'            \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}
#' @param params a list of parameters. See \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html}{
#'               the "Parameters" section of the documentation} for a list of parameters and valid values.
#' @param verbose verbosity for output, if <= 0, also will disable the print of evaluation during training
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
#' @keywords internal
NULL

#' @name lightgbm
#' @title Train a LightGBM model
#' @description Simple interface for training a LightGBM model.
#' @inheritParams lgb_shared_params
#' @param label Vector of labels, used if \code{data} is not an \code{\link{lgb.Dataset}}
#' @param weight vector of response values. If not NULL, will set to dataset
#' @param save_name File name to use when writing the trained model to disk. Should end in ".model".
#' @param ... Additional arguments passed to \code{\link{lgb.train}}. For example
#'     \itemize{
#'        \item{\code{valids}: a list of \code{lgb.Dataset} objects, used for validation}
#'        \item{\code{obj}: objective function, can be character or custom objective function. Examples include
#'                   \code{regression}, \code{regression_l1}, \code{huber},
#'                    \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}}
#'        \item{\code{eval}: evaluation function, can be (a list of) character or custom eval function}
#'        \item{\code{record}: Boolean, TRUE will record iteration message to \code{booster$record_evals}}
#'        \item{\code{colnames}: feature names, if not null, will use this to overwrite the names in dataset}
#'        \item{\code{categorical_feature}: categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g. \code{c(1L, 10L)} to
#'                            say "the first and tenth columns").}
#'        \item{\code{reset_data}: Boolean, setting it to TRUE (not the default value) will transform the booster model
#'                          into a predictor model which frees up memory and the original datasets}
#'         \item{\code{boosting}: Boosting type. \code{"gbdt"}, \code{"rf"}, \code{"dart"},
#'                                               \code{"goss"} or \code{"mvs"}.}
#'         \item{\code{num_leaves}: Maximum number of leaves in one tree.}
#'         \item{\code{max_depth}: Limit the max depth for tree model. This is used to deal with
#'                          overfit when #data is small. Tree still grow by leaf-wise.}
#'          \item{\code{num_threads}: Number of threads for LightGBM. For the best speed, set this to
#'                             the number of real CPU cores(\code{parallel::detectCores(logical = FALSE)}),
#'                             not the number of threads (most CPU using hyper-threading to generate 2 threads
#'                             per CPU core).}
#'     }
#' @inheritSection lgb_shared_params Early Stopping
#' @return a trained \code{lgb.Booster}
#' @export
lightgbm <- function(data,
                     label = NULL,
                     weight = NULL,
                     params = list(),
                     nrounds = 100L,
                     verbose = 1L,
                     eval_freq = 1L,
                     early_stopping_rounds = NULL,
                     save_name = "lightgbm.model",
                     init_model = NULL,
                     callbacks = list(),
                     ...) {

  # validate inputs early to avoid unnecessary computation
  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }

  # Set data to a temporary variable
  dtrain <- data

  # Check whether data is lgb.Dataset, if not then create lgb.Dataset manually
  if (!lgb.is.Dataset(x = dtrain)) {
    dtrain <- lgb.Dataset(data = data, label = label, weight = weight)
  }

  train_args <- list(
    "params" = params
    , "data" = dtrain
    , "nrounds" = nrounds
    , "verbose" = verbose
    , "eval_freq" = eval_freq
    , "early_stopping_rounds" = early_stopping_rounds
    , "init_model" = init_model
    , "callbacks" = callbacks
  )
  train_args <- append(train_args, list(...))

  if (! "valids" %in% names(train_args)) {
    train_args[["valids"]] <- list()
  }

  # Set validation as oneself
  if (verbose > 0L) {
    train_args[["valids"]][["train"]] <- dtrain
  }

  # Train a model using the regular way
  bst <- do.call(
    what = lgb.train
    , args = train_args
  )

  # Store model under a specific name
  bst$save_model(filename = save_name)

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
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
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
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
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
#' http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
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
#' @useDynLib lib_lightgbm , .registration = TRUE
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
