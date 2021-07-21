#' @importFrom R6 R6Class
CVBooster <- R6::R6Class(
  classname = "lgb.CVBooster",
  cloneable = FALSE,
  public = list(
    best_iter = -1L,
    best_score = NA,
    record_evals = list(),
    boosters = list(),
    initialize = function(x) {
      self$boosters <- x
      return(invisible(NULL))
    },
    reset_parameter = function(new_params) {
      for (x in boosters) { x$reset_parameter(new_params) }
      return(invisible(self))
    }
  )
)

#' @name lgb.cv
#' @title Main CV logic for LightGBM
#' @description Cross validation logic used by LightGBM
#' @inheritParams lgb_shared_params
#' @param nfold the original dataset is randomly partitioned into \code{nfold} equal size subsamples.
#' @param label Vector of labels, used if \code{data} is not an \code{\link{lgb.Dataset}}
#' @param weight vector of response values. If not NULL, will set to dataset
#' @param record Boolean, TRUE will record iteration message to \code{booster$record_evals}
#' @param showsd \code{boolean}, whether to show standard deviation of cross validation.
#'               This parameter defaults to \code{TRUE}. Setting it to \code{FALSE} can lead to a
#'               slight speedup by avoiding unnecessary computation.
#' @param stratified a \code{boolean} indicating whether sampling of folds should be stratified
#'                   by the values of outcome labels.
#' @param folds \code{list} provides a possibility to use a list of pre-defined CV folds
#'              (each element must be a vector of test fold's indices). When folds are supplied,
#'              the \code{nfold} and \code{stratified} parameters are ignored.
#' @param colnames feature names, if not null, will use this to overwrite the names in dataset
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param reset_data Boolean, setting it to TRUE (not the default value) will transform the booster model
#'                   into a predictor model which frees up memory and the original datasets
#' @param train_folds \code{list} specifying which indices to use for training. If \code{NULL}
#'                    (the default) all indices not specified in \code{folds} will be used for training.
#' @param ... other parameters, see Parameters.rst for more information. A few key parameters:
#'            \itemize{
#'                \item{\code{boosting}: Boosting type. \code{"gbdt"}, \code{"rf"}, \code{"dart"} or \code{"goss"}.}
#'                \item{\code{num_leaves}: Maximum number of leaves in one tree.}
#'                \item{\code{max_depth}: Limit the max depth for tree model. This is used to deal with
#'                                 overfit when #data is small. Tree still grow by leaf-wise.}
#'                \item{\code{num_threads}: Number of threads for LightGBM. For the best speed, set this to
#'                             the number of real CPU cores(\code{parallel::detectCores(logical = FALSE)}),
#'                             not the number of threads (most CPU using hyper-threading to generate 2 threads
#'                             per CPU core).}
#'            }
#' @inheritSection lgb_shared_params Early Stopping
#' @return a trained model \code{lgb.CVBooster}.
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' params <- list(objective = "regression", metric = "l2")
#' model <- lgb.cv(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , nfold = 3L
#'   , min_data = 1L
#'   , learning_rate = 1.0
#' )
#' }
#' @importFrom data.table data.table setorderv
#' @export
lgb.cv <- function(params = list()
                   , data
                   , nrounds = 100L
                   , nfold = 3L
                   , label = NULL
                   , weight = NULL
                   , obj = NULL
                   , eval = NULL
                   , verbose = 1L
                   , record = TRUE
                   , eval_freq = 1L
                   , showsd = TRUE
                   , stratified = TRUE
                   , folds = NULL
                   , init_model = NULL
                   , colnames = NULL
                   , categorical_feature = NULL
                   , early_stopping_rounds = NULL
                   , callbacks = list()
                   , reset_data = FALSE
                   , train_folds = NULL
                   , ...
                   ) {

  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }

  # If 'data' is not an lgb.Dataset, try to construct one using 'label'
  if (!lgb.is.Dataset(x = data)) {
    if (is.null(label)) {
      stop("'label' must be provided for lgb.cv if 'data' is not an 'lgb.Dataset'")
    }
    data <- lgb.Dataset(data = data, label = label)
  }

  # Setup temporary variables
  params <- append(params, list(...))
  params$verbose <- verbose
  params <- lgb.check.obj(params = params, obj = obj)
  params <- lgb.check.eval(params = params, eval = eval)
  fobj <- NULL
  eval_functions <- list(NULL)

  # set some parameters, resolving the way they were passed in with other parameters
  # in `params`.
  # this ensures that the model stored with Booster$save() correctly represents
  # what was passed in
  params <- lgb.check.wrapper_param(
    main_param_name = "num_iterations"
    , params = params
    , alternative_kwarg_value = nrounds
  )
  params <- lgb.check.wrapper_param(
    main_param_name = "early_stopping_round"
    , params = params
    , alternative_kwarg_value = early_stopping_rounds
  )
  early_stopping_rounds <- params[["early_stopping_round"]]

  # Check for objective (function or not)
  if (is.function(params$objective)) {
    fobj <- params$objective
    params$objective <- "NONE"
  }

  # If eval is a single function, store it as a 1-element list
  # (for backwards compatibility). If it is a list of functions, store
  # all of them. This makes it possible to pass any mix of strings like "auc"
  # and custom functions to eval
  if (is.function(eval)) {
    eval_functions <- list(eval)
  }
  if (methods::is(eval, "list")) {
    eval_functions <- Filter(
      f = is.function
      , x = eval
    )
  }

  # Init predictor to empty
  predictor <- NULL

  # Check for boosting from a trained model
  if (is.character(init_model)) {
    predictor <- Predictor$new(modelfile = init_model)
  } else if (lgb.is.Booster(x = init_model)) {
    predictor <- init_model$to_predictor()
  }

  # Set the iteration to start from / end to (and check for boosting from a trained model, again)
  begin_iteration <- 1L
  if (!is.null(predictor)) {
    begin_iteration <- predictor$current_iter() + 1L
  }
  end_iteration <- begin_iteration + params[["num_iterations"]] - 1L

  # pop interaction_constraints off of params. It needs some preprocessing on the
  # R side before being passed into the Dataset object
  interaction_constraints <- params[["interaction_constraints"]]
  params["interaction_constraints"] <- NULL

  # Construct datasets, if needed
  data$update_params(params = params)
  data$construct()

  # Check interaction constraints
  cnames <- NULL
  if (!is.null(colnames)) {
    cnames <- colnames
  } else if (!is.null(data$get_colnames())) {
    cnames <- data$get_colnames()
  }
  params[["interaction_constraints"]] <- lgb.check_interaction_constraints(
    interaction_constraints = interaction_constraints
    , column_names = cnames
  )

  if (!is.null(weight)) {
    data$setinfo(name = "weight", info = weight)
  }

  # Update parameters with parsed parameters
  data$update_params(params = params)

  # Create the predictor set
  data$.__enclos_env__$private$set_predictor(predictor = predictor)

  # Write column names
  if (!is.null(colnames)) {
    data$set_colnames(colnames = colnames)
  }

  # Write categorical features
  if (!is.null(categorical_feature)) {
    data$set_categorical_feature(categorical_feature = categorical_feature)
  }

  if (!is.null(folds)) {

    # Check for list of folds or for single value
    if (!identical(class(folds), "list") || length(folds) < 2L) {
      stop(sQuote("folds"), " must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    }

    nfold <- length(folds)

  } else {

    if (nfold <= 1L) {
      stop(sQuote("nfold"), " must be > 1")
    }

    # Create folds
    folds <- generate.cv.folds(
      nfold = nfold
      , nrows = nrow(data)
      , stratified = stratified
      , label = getinfo(dataset = data, name = "label")
      , group = getinfo(dataset = data, name = "group")
      , params = params
    )

  }

  # Add printing log callback
  if (verbose > 0L && eval_freq > 0L) {
    callbacks <- add.cb(cb_list = callbacks, cb = cb.print.evaluation(period = eval_freq))
  }

  # Add evaluation log callback
  if (record) {
    callbacks <- add.cb(cb_list = callbacks, cb = cb.record.evaluation())
  }

  # Did user pass parameters that indicate they want to use early stopping?
  using_early_stopping <- !is.null(early_stopping_rounds) && early_stopping_rounds > 0L

  boosting_param_names <- .PARAMETER_ALIASES()[["boosting"]]
  using_dart <- any(
    sapply(
      X = boosting_param_names
      , FUN = function(param) {
        identical(params[[param]], "dart")
      }
    )
  )

  # Cannot use early stopping with 'dart' boosting
  if (using_dart) {
    warning("Early stopping is not available in 'dart' mode.")
    using_early_stopping <- FALSE

    # Remove the cb.early.stop() function if it was passed in to callbacks
    callbacks <- Filter(
      f = function(cb_func) {
        !identical(attr(cb_func, "name"), "cb.early.stop")
      }
      , x = callbacks
    )
  }

  # If user supplied early_stopping_rounds, add the early stopping callback
  if (using_early_stopping) {
    callbacks <- add.cb(
      cb_list = callbacks
      , cb = cb.early.stop(
        stopping_rounds = early_stopping_rounds
        , first_metric_only = isTRUE(params[["first_metric_only"]])
        , verbose = verbose
      )
    )
  }

  cb <- categorize.callbacks(cb_list = callbacks)

  # Construct booster for each fold. The data.table() code below is used to
  # guarantee that indices are sorted while keeping init_score and weight together
  # with the correct indices. Note that it takes advantage of the fact that
  # someDT$some_column returns NULL is 'some_column' does not exist in the data.table
  bst_folds <- lapply(
    X = seq_along(folds)
    , FUN = function(k) {

      # For learning-to-rank, each fold is a named list with two elements:
      #   * `fold` = an integer vector of row indices
      #   * `group` = an integer vector describing which groups are in the fold
      # For classification or regression tasks, it will just be an integer
      # vector of row indices
      folds_have_group <- "group" %in% names(folds[[k]])
      if (folds_have_group) {
        test_indices <- folds[[k]]$fold
        test_group_indices <- folds[[k]]$group
        test_groups <- getinfo(dataset = data, name = "group")[test_group_indices]
        train_groups <- getinfo(dataset = data, name = "group")[-test_group_indices]
      } else {
        test_indices <- folds[[k]]
      }

      # Generate train_indices from either the train_folds argument
      # or as the opposite of (test)folds argument:
      if (!is.null(train_folds)) {
        train_indices <- train_folds[[k]]
      } else {
        train_indices <- seq_len(nrow(data))[-test_indices]
      }

      # set up test set
      indexDT <- data.table::data.table(
        indices = test_indices
        , weight = getinfo(dataset = data, name = "weight")[test_indices]
        , init_score = getinfo(dataset = data, name = "init_score")[test_indices]
      )
      data.table::setorderv(x = indexDT, cols = "indices", order = 1L)
      dtest <- slice(data, indexDT$indices)
      setinfo(dataset = dtest, name = "weight", info = indexDT$weight)
      setinfo(dataset = dtest, name = "init_score", info = indexDT$init_score)

      # set up training set
      indexDT <- data.table::data.table(
        indices = train_indices
        , weight = getinfo(dataset = data, name = "weight")[train_indices]
        , init_score = getinfo(dataset = data, name = "init_score")[train_indices]
      )
      data.table::setorderv(x = indexDT, cols = "indices", order = 1L)
      dtrain <- slice(data, indexDT$indices)
      setinfo(dataset = dtrain, name = "weight", info = indexDT$weight)
      setinfo(dataset = dtrain, name = "init_score", info = indexDT$init_score)

      if (folds_have_group) {
        setinfo(dataset = dtest, name = "group", info = test_groups)
        setinfo(dataset = dtrain, name = "group", info = train_groups)
      }

      booster <- Booster$new(params = params, train_set = dtrain)
      booster$add_valid(data = dtest, name = "valid")
      return(
        list(booster = booster)
      )
    }
  )

  # Create new booster
  cv_booster <- CVBooster$new(x = bst_folds)

  # Callback env
  env <- CB_ENV$new()
  env$model <- cv_booster
  env$begin_iteration <- begin_iteration
  env$end_iteration <- end_iteration

  # Start training model using number of iterations to start and end with
  for (i in seq.int(from = begin_iteration, to = end_iteration)) {

    # Overwrite iteration in environment
    env$iteration <- i
    env$eval_list <- list()

    for (f in cb$pre_iter) {
      f(env)
    }

    # Update one boosting iteration
    msg <- lapply(cv_booster$boosters, function(fd) {
      fd$booster$update(fobj = fobj)
      out <- list()
      for (eval_function in eval_functions) {
        out <- append(out, fd$booster$eval_valid(feval = eval_function))
      }
      return(out)
    })

    # Prepare collection of evaluation results
    merged_msg <- lgb.merge.cv.result(
      msg = msg
      , showsd = showsd
    )

    # Write evaluation result in environment
    env$eval_list <- merged_msg$eval_list

    # Check for standard deviation requirement
    if (showsd) {
      env$eval_err_list <- merged_msg$eval_err_list
    }

    # Loop through env
    for (f in cb$post_iter) {
      f(env)
    }

    # Check for early stopping and break if needed
    if (env$met_early_stop) break

  }

  # When early stopping is not activated, we compute the best iteration / score ourselves
  # based on the first first metric
  if (record && is.na(env$best_score)) {
    # when using a custom eval function, the metric name is returned from the
    # function, so figure it out from record_evals
    if (!is.null(eval_functions[1L])) {
      first_metric <- names(cv_booster$record_evals[["valid"]])[1L]
    } else {
      first_metric <- cv_booster$.__enclos_env__$private$eval_names[1L]
    }
    .find_best <- which.min
    if (isTRUE(env$eval_list[[1L]]$higher_better[1L])) {
      .find_best <- which.max
    }
    cv_booster$best_iter <- unname(
      .find_best(
        unlist(
          cv_booster$record_evals[["valid"]][[first_metric]][[.EVAL_KEY()]]
        )
      )
    )
    cv_booster$best_score <- cv_booster$record_evals[["valid"]][[first_metric]][[.EVAL_KEY()]][[cv_booster$best_iter]]
  }

  if (reset_data) {
    lapply(cv_booster$boosters, function(fd) {
      # Store temporarily model data elsewhere
      booster_old <- list(
        best_iter = fd$booster$best_iter
        , best_score = fd$booster$best_score
        , record_evals = fd$booster$record_evals
      )
      # Reload model
      fd$booster <- lgb.load(model_str = fd$booster$save_model_to_string())
      fd$booster$best_iter <- booster_old$best_iter
      fd$booster$best_score <- booster_old$best_score
      fd$booster$record_evals <- booster_old$record_evals
    })
  }

  return(cv_booster)

}

# Generates random (stratified if needed) CV folds
generate.cv.folds <- function(nfold, nrows, stratified, label, group, params) {

  # Check for group existence
  if (is.null(group)) {

    # Shuffle
    rnd_idx <- sample.int(nrows)

    # Request stratified folds
    if (isTRUE(stratified) && params$objective %in% c("binary", "multiclass") && length(label) == length(rnd_idx)) {

      y <- label[rnd_idx]
      y <- as.factor(y)
      folds <- lgb.stratified.folds(y = y, k = nfold)

    } else {

      # Make simple non-stratified folds
      folds <- list()

      # Loop through each fold
      for (i in seq_len(nfold)) {
        kstep <- length(rnd_idx) %/% (nfold - i + 1L)
        folds[[i]] <- rnd_idx[seq_len(kstep)]
        rnd_idx <- rnd_idx[-seq_len(kstep)]
      }

    }

  } else {

    # When doing group, stratified is not possible (only random selection)
    if (nfold > length(group)) {
      stop("\nYou requested too many folds for the number of available groups.\n")
    }

    # Degroup the groups
    ungrouped <- inverse.rle(list(lengths = group, values = seq_along(group)))

    # Can't stratify, shuffle
    rnd_idx <- sample.int(length(group))

    # Make simple non-stratified folds
    folds <- list()

    # Loop through each fold
    for (i in seq_len(nfold)) {
      kstep <- length(rnd_idx) %/% (nfold - i + 1L)
      folds[[i]] <- list(
        fold = which(ungrouped %in% rnd_idx[seq_len(kstep)])
        , group = rnd_idx[seq_len(kstep)]
      )
      rnd_idx <- rnd_idx[-seq_len(kstep)]
    }

  }

  return(folds)

}

# Creates CV folds stratified by the values of y.
# It was borrowed from caret::createFolds and simplified
# by always returning an unnamed list of fold indices.
#' @importFrom stats quantile
lgb.stratified.folds <- function(y, k) {

  ## Group the numeric data based on their magnitudes
  ## and sample within those groups.
  ## When the number of samples is low, we may have
  ## issues further slicing the numeric data into
  ## groups. The number of groups will depend on the
  ## ratio of the number of folds to the sample size.
  ## At most, we will use quantiles. If the sample
  ## is too small, we just do regular unstratified CV
  if (is.numeric(y)) {

    cuts <- length(y) %/% k
    if (cuts < 2L) {
      cuts <- 2L
    }
    if (cuts > 5L) {
      cuts <- 5L
    }
    y <- cut(
      y
      , unique(stats::quantile(y, probs = seq.int(0.0, 1.0, length.out = cuts)))
      , include.lowest = TRUE
    )

  }

  if (k < length(y)) {

    ## Reset levels so that the possible levels and
    ## the levels in the vector are the same
    y <- as.factor(as.character(y))
    numInClass <- table(y)
    foldVector <- vector(mode = "integer", length(y))

    ## For each class, balance the fold allocation as far
    ## as possible, then resample the remainder.
    ## The final assignment of folds is also randomized.

    for (i in seq_along(numInClass)) {

      ## Create a vector of integers from 1:k as many times as possible without
      ## going over the number of samples in the class. Note that if the number
      ## of samples in a class is less than k, nothing is producd here.
      seqVector <- rep(seq_len(k), numInClass[i] %/% k)

      ## Add enough random integers to get length(seqVector) == numInClass[i]
      if (numInClass[i] %% k > 0L) {
        seqVector <- c(seqVector, sample.int(k, numInClass[i] %% k))
      }

      ## Shuffle the integers for fold assignment and assign to this classes's data
      foldVector[y == dimnames(numInClass)$y[i]] <- sample(seqVector)

    }

  } else {

    foldVector <- seq(along = y)

  }

  out <- split(seq(along = y), foldVector)
  names(out) <- NULL
  return(out)
}

lgb.merge.cv.result <- function(msg, showsd) {

  if (length(msg) == 0L) {
    stop("lgb.cv: size of cv result error")
  }

  eval_len <- length(msg[[1L]])

  if (eval_len == 0L) {
    stop("lgb.cv: should provide at least one metric for CV")
  }

  # Get evaluation results using a list apply
  eval_result <- lapply(seq_len(eval_len), function(j) {
    as.numeric(lapply(seq_along(msg), function(i) {
      msg[[i]][[j]]$value }))
  })

  # Get evaluation. Just taking the first element here to
  # get structure (name, higher_better, data_name)
  ret_eval <- msg[[1L]]

  for (j in seq_len(eval_len)) {
    ret_eval[[j]]$value <- mean(eval_result[[j]])
  }

  ret_eval_err <- NULL

  # Check for standard deviation
  if (showsd) {

    # Parse standard deviation
    for (j in seq_len(eval_len)) {
      ret_eval_err <- c(
        ret_eval_err
        , sqrt(mean(eval_result[[j]] ^ 2L) - mean(eval_result[[j]]) ^ 2L)
      )
    }

    ret_eval_err <- as.list(ret_eval_err)

  }

  return(
    list(
      eval_list = ret_eval
      , eval_err_list = ret_eval_err
    )
  )

}
