#' @importFrom R6 R6Class
#' @importFrom utils modifyList
Booster <- R6::R6Class(
  classname = "lgb.Booster",
  cloneable = FALSE,
  public = list(

    best_iter = -1L,
    best_score = NA_real_,
    params = list(),
    record_evals = list(),
    data_processor = NULL,

    # Initialize will create a starter booster
    initialize = function(params = list(),
                          train_set = NULL,
                          modelfile = NULL,
                          model_str = NULL) {

      handle <- NULL

      if (!is.null(train_set)) {

        if (!.is_Dataset(train_set)) {
          stop("lgb.Booster: Can only use lgb.Dataset as training data")
        }
        train_set_handle <- train_set$.__enclos_env__$private$get_handle()
        params <- utils::modifyList(params, train_set$get_params())
        params_str <- .params2str(params = params)
        # Store booster handle
        handle <- .Call(
          LGBM_BoosterCreate_R
          , train_set_handle
          , params_str
        )

        # Create private booster information
        private$train_set <- train_set
        private$train_set_version <- train_set$.__enclos_env__$private$version
        private$num_dataset <- 1L
        private$init_predictor <- train_set$.__enclos_env__$private$predictor

        if (!is.null(private$init_predictor)) {

          # Merge booster
          .Call(
            LGBM_BoosterMerge_R
            , handle
            , private$init_predictor$.__enclos_env__$private$handle
          )

        }

        # Check current iteration
        private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)

      } else if (!is.null(modelfile)) {

        # Do we have a model file as character?
        if (!is.character(modelfile)) {
          stop("lgb.Booster: Can only use a string as model file path")
        }

        modelfile <- path.expand(modelfile)

        # Create booster from model
        handle <- .Call(
          LGBM_BoosterCreateFromModelfile_R
          , modelfile
        )
        params <- private$get_loaded_param(handle)

      } else if (!is.null(model_str)) {

        # Do we have a model_str as character/raw?
        if (!is.raw(model_str) && !is.character(model_str)) {
          stop("lgb.Booster: Can only use a character/raw vector as model_str")
        }

        # Create booster from model
        handle <- .Call(
          LGBM_BoosterLoadModelFromString_R
          , model_str
        )

      } else {

        # Booster non existent
        stop(
          "lgb.Booster: Need at least either training dataset, "
          , "model file, or model_str to create booster instance"
        )

      }

      class(handle) <- "lgb.Booster.handle"
      private$handle <- handle
      private$num_class <- 1L
      .Call(
        LGBM_BoosterGetNumClasses_R
        , private$handle
        , private$num_class
      )

      self$params <- params

      return(invisible(NULL))

    },

    # Set training data name
    set_train_data_name = function(name) {

      # Set name
      private$name_train_set <- name
      return(invisible(self))

    },

    # Add validation data
    add_valid = function(data, name) {

      if (!.is_Dataset(data)) {
        stop("lgb.Booster.add_valid: Can only use lgb.Dataset as validation data")
      }

      if (!identical(data$.__enclos_env__$private$predictor, private$init_predictor)) {
        stop(
          "lgb.Booster.add_valid: Failed to add validation data; "
          , "you should use the same predictor for these data"
        )
      }

      if (!is.character(name)) {
        stop("lgb.Booster.add_valid: Can only use characters as data name")
      }

      # Add validation data to booster
      .Call(
        LGBM_BoosterAddValidData_R
        , private$handle
        , data$.__enclos_env__$private$get_handle()
      )

      private$valid_sets <- c(private$valid_sets, data)
      private$name_valid_sets <- c(private$name_valid_sets, name)
      private$num_dataset <- private$num_dataset + 1L
      private$is_predicted_cur_iter <- c(private$is_predicted_cur_iter, FALSE)

      return(invisible(self))

    },

    reset_parameter = function(params) {

      if (methods::is(self$params, "list")) {
        params <- utils::modifyList(self$params, params)
      }

      params_str <- .params2str(params = params)

      self$restore_handle()

      .Call(
        LGBM_BoosterResetParameter_R
        , private$handle
        , params_str
      )
      self$params <- params

      return(invisible(self))

    },

    # Perform boosting update iteration
    update = function(train_set = NULL, fobj = NULL) {

      if (is.null(train_set)) {
        if (private$train_set$.__enclos_env__$private$version != private$train_set_version) {
          train_set <- private$train_set
        }
      }

      if (!is.null(train_set)) {

        if (!.is_Dataset(train_set)) {
          stop("lgb.Booster.update: Only can use lgb.Dataset as training data")
        }

        if (!identical(train_set$predictor, private$init_predictor)) {
          stop("lgb.Booster.update: Change train_set failed, you should use the same predictor for these data")
        }

        .Call(
          LGBM_BoosterResetTrainingData_R
          , private$handle
          , train_set$.__enclos_env__$private$get_handle()
        )

        private$train_set <- train_set
        private$train_set_version <- train_set$.__enclos_env__$private$version

      }

      # Check if objective is empty
      if (is.null(fobj)) {
        if (private$set_objective_to_none) {
          stop("lgb.Booster.update: cannot update due to null objective function")
        }
        # Boost iteration from known objective
        .Call(
          LGBM_BoosterUpdateOneIter_R
          , private$handle
        )

      } else {

        if (!is.function(fobj)) {
          stop("lgb.Booster.update: fobj should be a function")
        }
        if (!private$set_objective_to_none) {
          self$reset_parameter(params = list(objective = "none"))
          private$set_objective_to_none <- TRUE
        }
        # Perform objective calculation
        preds <- private$inner_predict(1L)
        gpair <- fobj(preds, private$train_set)

        # Check for gradient and hessian as list
        if (is.null(gpair$grad) || is.null(gpair$hess)) {
          stop("lgb.Booster.update: custom objective should
            return a list with attributes (hess, grad)")
        }

        # Check grad and hess have the right shape
        n_grad <- length(gpair$grad)
        n_hess <- length(gpair$hess)
        n_preds <- length(preds)
        if (n_grad != n_preds) {
          stop(sprintf("Expected custom objective function to return grad with length %d, got %d.", n_preds, n_grad))
        }
        if (n_hess != n_preds) {
          stop(sprintf("Expected custom objective function to return hess with length %d, got %d.", n_preds, n_hess))
        }

        # Return custom boosting gradient/hessian
        .Call(
          LGBM_BoosterUpdateOneIterCustom_R
          , private$handle
          , gpair$grad
          , gpair$hess
          , n_preds
        )

      }

      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }

      return(invisible(self))

    },

    # Return one iteration behind
    rollback_one_iter = function() {

      self$restore_handle()

      .Call(
        LGBM_BoosterRollbackOneIter_R
        , private$handle
      )

      # Loop through each iteration
      for (i in seq_along(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }

      return(invisible(self))

    },

    # Get current iteration
    current_iter = function() {

      self$restore_handle()

      cur_iter <- 0L
      .Call(
        LGBM_BoosterGetCurrentIteration_R
        , private$handle
        , cur_iter
      )
      return(cur_iter)

    },

    # Number of trees per iteration
    num_trees_per_iter = function() {

      self$restore_handle()

      trees_per_iter <- 1L
      .Call(
        LGBM_BoosterNumModelPerIteration_R
        , private$handle
        , trees_per_iter
      )
      return(trees_per_iter)

    },

    # Total number of trees
    num_trees = function() {

      self$restore_handle()

      ntrees <- 0L
      .Call(
        LGBM_BoosterNumberOfTotalModel_R
        , private$handle
        , ntrees
      )
      return(ntrees)

    },

    # Number of iterations (= rounds)
    num_iter = function() {

      ntrees <- self$num_trees()
      trees_per_iter <- self$num_trees_per_iter()

      return(ntrees / trees_per_iter)

    },

    # Get upper bound
    upper_bound = function() {

      self$restore_handle()

      upper_bound <- 0.0
      .Call(
        LGBM_BoosterGetUpperBoundValue_R
        , private$handle
        , upper_bound
      )
      return(upper_bound)

    },

    # Get lower bound
    lower_bound = function() {

      self$restore_handle()

      lower_bound <- 0.0
      .Call(
        LGBM_BoosterGetLowerBoundValue_R
        , private$handle
        , lower_bound
      )
      return(lower_bound)

    },

    # Evaluate data on metrics
    eval = function(data, name, feval = NULL) {

      if (!.is_Dataset(data)) {
        stop("lgb.Booster.eval: Can only use lgb.Dataset to eval")
      }

      # Check for identical data
      data_idx <- 0L
      if (identical(data, private$train_set)) {
        data_idx <- 1L
      } else {

        # Check for validation data
        if (length(private$valid_sets) > 0L) {

          for (i in seq_along(private$valid_sets)) {

            # Check for identical validation data with training data
            if (identical(data, private$valid_sets[[i]])) {

              # Found identical data, skip
              data_idx <- i + 1L
              break

            }

          }

        }

      }

      # Check if evaluation was not done
      if (data_idx == 0L) {

        # Add validation data by name
        self$add_valid(data, name)
        data_idx <- private$num_dataset

      }

      # Evaluate data
      return(
        private$inner_eval(
          data_name = name
          , data_idx = data_idx
          , feval = feval
        )
      )

    },

    # Evaluation training data
    eval_train = function(feval = NULL) {
      return(private$inner_eval(private$name_train_set, 1L, feval))
    },

    # Evaluation validation data
    eval_valid = function(feval = NULL) {

      ret <- list()

      if (length(private$valid_sets) <= 0L) {
        return(ret)
      }

      for (i in seq_along(private$valid_sets)) {
        ret <- append(
          x = ret
          , values = private$inner_eval(private$name_valid_sets[[i]], i + 1L, feval)
        )
      }

      return(ret)

    },

    # Save model
    save_model = function(
      filename
      , num_iteration = NULL
      , feature_importance_type = 0L
      , start_iteration = 1L
    ) {

      self$restore_handle()

      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      filename <- path.expand(filename)

      .Call(
        LGBM_BoosterSaveModel_R
        , private$handle
        , as.integer(num_iteration)
        , as.integer(feature_importance_type)
        , filename
        , as.integer(start_iteration) - 1L  # Turn to 0-based
      )

      return(invisible(self))
    },

    save_model_to_string = function(
      num_iteration = NULL
      , feature_importance_type = 0L
      , as_char = TRUE
      , start_iteration = 1L
    ) {

      self$restore_handle()

      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      model_str <- .Call(
          LGBM_BoosterSaveModelToString_R
          , private$handle
          , as.integer(num_iteration)
          , as.integer(feature_importance_type)
          , as.integer(start_iteration) - 1L  # Turn to 0-based
      )

      if (as_char) {
        model_str <- rawToChar(model_str)
      }

      return(model_str)

    },

    # Dump model in memory
    dump_model = function(
      num_iteration = NULL, feature_importance_type = 0L, start_iteration = 1L
    ) {

      self$restore_handle()

      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      model_str <- .Call(
        LGBM_BoosterDumpModel_R
        , private$handle
        , as.integer(num_iteration)
        , as.integer(feature_importance_type)
        , as.integer(start_iteration) - 1L  # Turn to 0-based
      )

      return(model_str)

    },

    # Predict on new data
    predict = function(data,
                       start_iteration = NULL,
                       num_iteration = NULL,
                       rawscore = FALSE,
                       predleaf = FALSE,
                       predcontrib = FALSE,
                       header = FALSE,
                       params = list()) {

      self$restore_handle()

      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }

      if (is.null(start_iteration)) {
        start_iteration <- 0L
      }

      # possibly override keyword arguments with parameters
      #
      # NOTE: this length() check minimizes the latency introduced by these checks,
      #       for the common case where params is empty
      #
      # NOTE: doing this here instead of in Predictor$predict() to keep
      #       Predictor$predict() as fast as possible
      if (length(params) > 0L) {
        params <- .check_wrapper_param(
          main_param_name = "predict_raw_score"
          , params = params
          , alternative_kwarg_value = rawscore
        )
        params <- .check_wrapper_param(
          main_param_name = "predict_leaf_index"
          , params = params
          , alternative_kwarg_value = predleaf
        )
        params <- .check_wrapper_param(
          main_param_name = "predict_contrib"
          , params = params
          , alternative_kwarg_value = predcontrib
        )
        rawscore <- params[["predict_raw_score"]]
        predleaf <- params[["predict_leaf_index"]]
        predcontrib <- params[["predict_contrib"]]
      }

      # Predict on new data
      predictor <- Predictor$new(
        modelfile = private$handle
        , params = params
        , fast_predict_config = private$fast_predict_config
      )
      return(
        predictor$predict(
          data = data
          , start_iteration = start_iteration
          , num_iteration = num_iteration
          , rawscore = rawscore
          , predleaf = predleaf
          , predcontrib = predcontrib
          , header = header
        )
      )

    },

    # Transform into predictor
    to_predictor = function() {
      return(Predictor$new(modelfile = private$handle))
    },

    configure_fast_predict = function(csr = FALSE,
                                      start_iteration = NULL,
                                      num_iteration = NULL,
                                      rawscore = FALSE,
                                      predleaf = FALSE,
                                      predcontrib = FALSE,
                                      params = list()) {

      self$restore_handle()
      ncols <- .Call(LGBM_BoosterGetNumFeature_R, private$handle)

      if (is.null(num_iteration)) {
        num_iteration <- -1L
      }
      if (is.null(start_iteration)) {
        start_iteration <- 0L
      }

      if (!csr) {
        fun <- LGBM_BoosterPredictForMatSingleRowFastInit_R
      } else {
        fun <- LGBM_BoosterPredictForCSRSingleRowFastInit_R
      }

      fast_handle <- .Call(
        fun
        , private$handle
        , ncols
        , rawscore
        , predleaf
        , predcontrib
        , start_iteration
        , num_iteration
        , .params2str(params = params)
      )

      private$fast_predict_config <- list(
        handle = fast_handle
        , csr = as.logical(csr)
        , ncols = ncols
        , start_iteration = start_iteration
        , num_iteration = num_iteration
        , rawscore = as.logical(rawscore)
        , predleaf = as.logical(predleaf)
        , predcontrib = as.logical(predcontrib)
        , params = params
      )

      return(invisible(NULL))
    },

    # Used for serialization
    raw = NULL,

    # Store serialized raw bytes in model object
    save_raw = function() {
      if (is.null(self$raw)) {
        self$raw <- self$save_model_to_string(NULL, as_char = FALSE)
      }
      return(invisible(NULL))

    },

    drop_raw = function() {
      self$raw <- NULL
      return(invisible(NULL))
    },

    check_null_handle = function() {
      return(.is_null_handle(private$handle))
    },

    restore_handle = function() {
      if (self$check_null_handle()) {
        if (is.null(self$raw)) {
          .Call(LGBM_NullBoosterHandleError_R)
        }
        private$handle <- .Call(LGBM_BoosterLoadModelFromString_R, self$raw)
      }
      return(invisible(NULL))
    },

    get_handle = function() {
      return(private$handle)
    }

  ),
  private = list(
    handle = NULL,
    train_set = NULL,
    name_train_set = "training",
    valid_sets = list(),
    name_valid_sets = list(),
    predict_buffer = list(),
    is_predicted_cur_iter = list(),
    num_class = 1L,
    num_dataset = 0L,
    init_predictor = NULL,
    eval_names = NULL,
    higher_better_inner_eval = NULL,
    set_objective_to_none = FALSE,
    train_set_version = 0L,
    fast_predict_config = list(),

    # finalize() will free up the handles
    finalize = function() {
      .Call(
        LGBM_BoosterFree_R
        , private$handle
      )
      private$handle <- NULL
      return(invisible(NULL))
    },

    # Predict data
    inner_predict = function(idx) {

      # Store data name
      data_name <- private$name_train_set

      if (idx > 1L) {
        data_name <- private$name_valid_sets[[idx - 1L]]
      }

      # Check for unknown dataset (over the maximum provided range)
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }

      # Check for prediction buffer
      if (is.null(private$predict_buffer[[data_name]])) {

        # Store predictions
        npred <- 0L
        .Call(
          LGBM_BoosterGetNumPredict_R
          , private$handle
          , as.integer(idx - 1L)
          , npred
        )
        private$predict_buffer[[data_name]] <- numeric(npred)

      }

      # Check if current iteration was already predicted
      if (!private$is_predicted_cur_iter[[idx]]) {

        # Use buffer
        .Call(
          LGBM_BoosterGetPredict_R
          , private$handle
          , as.integer(idx - 1L)
          , private$predict_buffer[[data_name]]
        )
        private$is_predicted_cur_iter[[idx]] <- TRUE
      }

      return(private$predict_buffer[[data_name]])
    },

    # Get evaluation information
    get_eval_info = function() {

      if (is.null(private$eval_names)) {
        eval_names <- .Call(
          LGBM_BoosterGetEvalNames_R
          , private$handle
        )

        if (length(eval_names) > 0L) {

          # Parse and store privately names
          private$eval_names <- eval_names

          # some metrics don't map cleanly to metric names, for example "ndcg@1" is just the
          # ndcg metric evaluated at the first "query result" in learning-to-rank
          metric_names <- gsub("@.*", "", eval_names)
          private$higher_better_inner_eval <- .METRICS_HIGHER_BETTER()[metric_names]

        }

      }

      return(private$eval_names)

    },

    get_loaded_param = function(handle) {
      params_str <- .Call(
        LGBM_BoosterGetLoadedParam_R
        , handle
      )
      params <- jsonlite::fromJSON(params_str)
      if ("interaction_constraints" %in% names(params)) {
        params[["interaction_constraints"]] <- lapply(params[["interaction_constraints"]], function(x) x + 1L)
      }

      return(params)

    },

    inner_eval = function(data_name, data_idx, feval = NULL) {

      # Check for unknown dataset (over the maximum provided range)
      if (data_idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }

      self$restore_handle()

      private$get_eval_info()

      ret <- list()

      if (length(private$eval_names) > 0L) {

        # Create evaluation values
        tmp_vals <- numeric(length(private$eval_names))
        .Call(
          LGBM_BoosterGetEval_R
          , private$handle
          , as.integer(data_idx - 1L)
          , tmp_vals
        )

        for (i in seq_along(private$eval_names)) {

          # Store evaluation and append to return
          res <- list()
          res$data_name <- data_name
          res$name <- private$eval_names[i]
          res$value <- tmp_vals[i]
          res$higher_better <- private$higher_better_inner_eval[i]
          ret <- append(ret, list(res))

        }

      }

      # Check if there are evaluation metrics
      if (!is.null(feval)) {

        # Check if evaluation metric is a function
        if (!is.function(feval)) {
          stop("lgb.Booster.eval: feval should be a function")
        }

        data <- private$train_set

        # Check if data to assess is existing differently
        if (data_idx > 1L) {
          data <- private$valid_sets[[data_idx - 1L]]
        }

        # Perform function evaluation
        res <- feval(private$inner_predict(data_idx), data)

        if (is.null(res$name) || is.null(res$value) ||  is.null(res$higher_better)) {
          stop(
            "lgb.Booster.eval: custom eval function should return a list with attribute (name, value, higher_better)"
          )
        }

        # Append names and evaluation
        res$data_name <- data_name
        ret <- append(ret, list(res))
      }

      return(ret)

    }

  )
)

#' @name lgb_predict_shared_params
#' @title Shared prediction parameter docs
#' @param type Type of prediction to output. Allowed types are:\itemize{
#'             \item \code{"response"}: will output the predicted score according to the objective function being
#'                   optimized (depending on the link function that the objective uses), after applying any necessary
#'                   transformations - for example, for \code{objective="binary"}, it will output class probabilities.
#'             \item \code{"class"}: for classification objectives, will output the class with the highest predicted
#'                   probability. For other objectives, will output the same as "response". Note that \code{"class"} is
#'                   not a supported type for \link{lgb.configure_fast_predict} (see the documentation of that function
#'                   for more details).
#'             \item \code{"raw"}: will output the non-transformed numbers (sum of predictions from boosting iterations'
#'                   results) from which the "response" number is produced for a given objective function - for example,
#'                   for \code{objective="binary"}, this corresponds to log-odds. For many objectives such as
#'                   "regression", since no transformation is applied, the output will be the same as for "response".
#'             \item \code{"leaf"}: will output the index of the terminal node / leaf at which each observations falls
#'                   in each tree in the model, outputted as integers, with one column per tree.
#'             \item \code{"contrib"}: will return the per-feature contributions for each prediction, including an
#'                   intercept (each feature will produce one column).
#'             }
#'
#'             Note that, if using custom objectives, types "class" and "response" will not be available and will
#'             default towards using "raw" instead.
#'
#'             If the model was fit through function \link{lightgbm} and it was passed a factor as labels,
#'             passing the prediction type through \code{params} instead of through this argument might
#'             result in factor levels for classification objectives not being applied correctly to the
#'             resulting output.
#'
#'             \emph{New in version 4.0.0}
#'
#' @param start_iteration int or None, optional (default=None)
#'                        Start index of the iteration to predict.
#'                        If None or <= 0, starts from the first iteration.
#' @param num_iteration int or None, optional (default=None)
#'                      Limit number of iterations in the prediction.
#'                      If None, if the best iteration exists and start_iteration is None or <= 0, the
#'                      best iteration is used; otherwise, all iterations from start_iteration are used.
#'                      If <= 0, all iterations from start_iteration are used (no limits).
#' @param params a list of additional named parameters. See
#'               \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#predict-parameters}{
#'               the "Predict Parameters" section of the documentation} for a list of parameters and
#'               valid values. Where these conflict with the values of keyword arguments to this function,
#'               the values in \code{params} take precedence.
#' @details This page contains shared documentation for prediction-related parameters used throughout the package.
#' @keywords internal
NULL

#' @name predict.lgb.Booster
#' @title Predict method for LightGBM model
#' @description Predicted values based on class \code{lgb.Booster}
#'
#'              \emph{New in version 4.0.0}
#'
#' @details If the model object has been configured for fast single-row predictions through
#'          \link{lgb.configure_fast_predict}, this function will use the prediction parameters
#'          that were configured for it - as such, extra prediction parameters should not be passed
#'          here, otherwise the configuration will be ignored and the slow route will be taken.
#' @inheritParams lgb_predict_shared_params
#' @param object Object of class \code{lgb.Booster}
#' @param newdata a \code{matrix} object, a \code{dgCMatrix}, a \code{dgRMatrix} object, a \code{dsparseVector} object,
#'                or a character representing a path to a text file (CSV, TSV, or LibSVM).
#'
#'                For sparse inputs, if predictions are only going to be made for a single row, it will be faster to
#'                use CSR format, in which case the data may be passed as either a single-row CSR matrix (class
#'                \code{dgRMatrix} from package \code{Matrix}) or as a sparse numeric vector (class
#'                \code{dsparseVector} from package \code{Matrix}).
#'
#'                If single-row predictions are going to be performed frequently, it is recommended to
#'                pre-configure the model object for fast single-row sparse predictions through function
#'                \link{lgb.configure_fast_predict}.
#'
#'                \emph{Changed from 'data', in version 4.0.0}
#'
#' @param header only used for prediction for text file. True if text file has header
#' @param ... ignored
#' @return For prediction types that are meant to always return one output per observation (e.g. when predicting
#'         \code{type="response"} or \code{type="raw"} on a binary classification or regression objective), will
#'         return a vector with one element per row in \code{newdata}.
#'
#'         For prediction types that are meant to return more than one output per observation (e.g. when predicting
#'         \code{type="response"} or \code{type="raw"} on a multi-class objective, or when predicting
#'         \code{type="leaf"}, regardless of objective), will return a matrix with one row per observation in
#'         \code{newdata} and one column per output.
#'
#'         For \code{type="leaf"} predictions, will return a matrix with one row per observation in \code{newdata}
#'         and one column per tree. Note that for multiclass objectives, LightGBM trains one tree per class at each
#'         boosting iteration. That means that, for example, for a multiclass model with 3 classes, the leaf
#'         predictions for the first class can be found in columns 1, 4, 7, 10, etc.
#'
#'         For \code{type="contrib"}, will return a matrix of SHAP values with one row per observation in
#'         \code{newdata} and columns corresponding to features. For regression, ranking, cross-entropy, and binary
#'         classification objectives, this matrix contains one column per feature plus a final column containing the
#'         Shapley base value. For multiclass objectives, this matrix will represent \code{num_classes} such matrices,
#'         in the order "feature contributions for first class, feature contributions for second class, feature
#'         contributions for third class, etc.".
#'
#'         If the model was fit through function \link{lightgbm} and it was passed a factor as labels, predictions
#'         returned from this function will retain the factor levels (either as values for \code{type="class"}, or
#'         as column names for \code{type="response"} and \code{type="raw"} for multi-class objectives). Note that
#'         passing the requested prediction type under \code{params} instead of through \code{type} might result in
#'         the factor levels not being present in the output.
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , num_threads = 2L
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#' )
#' preds <- predict(model, test$data)
#'
#' # pass other prediction parameters
#' preds <- predict(
#'     model,
#'     test$data,
#'     params = list(
#'         predict_disable_shape_check = TRUE
#'    )
#' )
#' }
#' @importFrom utils modifyList
#' @export
predict.lgb.Booster <- function(object,
                                newdata,
                                type = "response",
                                start_iteration = NULL,
                                num_iteration = NULL,
                                header = FALSE,
                                params = list(),
                                ...) {

  if (!.is_Booster(x = object)) {
    stop("predict.lgb.Booster: object should be an ", sQuote("lgb.Booster", q = FALSE))
  }

  additional_params <- list(...)
  if (length(additional_params) > 0L) {
    additional_params_names <- names(additional_params)
    if ("reshape" %in% additional_params_names) {
      stop("'reshape' argument is no longer supported.")
    }

    old_args_for_type <- list(
      "rawscore" = "raw"
      , "predleaf" = "leaf"
      , "predcontrib" = "contrib"
    )
    for (arg in names(old_args_for_type)) {
      if (arg %in% additional_params_names) {
        stop(sprintf("Argument '%s' is no longer supported. Use type='%s' instead."
                     , arg
                     , old_args_for_type[[arg]]))
      }
    }

    warning(paste0(
      "predict.lgb.Booster: Found the following passed through '...': "
      , toString(names(additional_params))
      , ". These are ignored. Use argument 'params' instead."
    ))
  }

  if (!is.null(object$params$objective) && object$params$objective == "none" && type %in% c("class", "response")) {
    warning("Prediction types 'class' and 'response' are not supported for custom objectives.")
    type <- "raw"
  }

  rawscore <- FALSE
  predleaf <- FALSE
  predcontrib <- FALSE
  if (type == "raw") {
    rawscore <- TRUE
  } else if (type == "leaf") {
    predleaf <- TRUE
  } else if (type == "contrib") {
    predcontrib <- TRUE
  }

  pred <- object$predict(
    data = newdata
    , start_iteration = start_iteration
    , num_iteration = num_iteration
    , rawscore = rawscore
    , predleaf =  predleaf
    , predcontrib =  predcontrib
    , header = header
    , params = params
  )
  if (type == "class") {
    if (object$params$objective %in% .BINARY_OBJECTIVES()) {
      pred <- as.integer(pred >= 0.5)
    } else if (object$params$objective %in% .MULTICLASS_OBJECTIVES()) {
      pred <- max.col(pred) - 1L
    }
  }
  if (!is.null(object$data_processor)) {
    pred <- object$data_processor$process_predictions(
      pred = pred
      , type = type
    )
  }
  return(pred)
}

#' @title Configure Fast Single-Row Predictions
#' @description Pre-configures a LightGBM model object to produce fast single-row predictions
#'              for a given input data type, prediction type, and parameters.
#' @details Calling this function again replaces the previous fast prediction configuration.
#'          Other prediction calls do not change its prediction type or iteration range.
#'
#'          Note that this function will not accept argument \code{type="class"} - for such cases, one
#'          can pass \code{type="response"} to this function and then \code{type="class"} to the
#'          \code{predict} function - the fast configuration will not be lost or altered if the switch
#'          is between "response" and "class".
#'
#'          The configuration does not survive de-serializations, so it has to be generated
#'          anew in every R process that is going to use it (e.g. if loading a model object
#'          through \code{readRDS}, whatever configuration was there previously will be lost).
#'
#'          Requesting a different prediction type or passing parameters to \link{predict.lgb.Booster}
#'          will cause it to ignore the fast-predict configuration and take the slow route instead.
#'
#'          Note that, if configuring a non-default prediction type (such as leaf indices),
#'          then that type must also be passed in the call to \link{predict.lgb.Booster} in
#'          order for it to use the configuration. This also applies for \code{start_iteration}
#'          and \code{num_iteration}, but \bold{the \code{params} list must be empty} in the call to \code{predict}.
#'
#'          Predictions about feature contributions do not allow a fast route for CSR inputs,
#'          and as such, this function will produce an error if passing \code{csr=TRUE} and
#'          \code{type = "contrib"} together.
#' @inheritParams lgb_predict_shared_params
#' @param model LightGBM model object (class \code{lgb.Booster}).
#'
#'              \bold{The object will be modified in-place}.
#' @param csr Whether the prediction function is going to be called on sparse CSR inputs.
#'            If \code{FALSE}, will be assumed that predictions are going to be called on single-row
#'            regular R matrices.
#' @return The same \code{model} that was passed as input, invisibly, with the desired
#'         configuration stored inside it and available to be used in future calls to
#'         \link{predict.lgb.Booster}.
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' library(lightgbm)
#' data(mtcars)
#' X <- as.matrix(mtcars[, -1L])
#' y <- mtcars[, 1L]
#' dtrain <- lgb.Dataset(X, label = y, params = list(max_bin = 5L))
#' params <- list(
#'   min_data_in_leaf = 2L
#'   , num_threads = 2L
#' )
#' model <- lgb.train(
#'   params = params
#'  , data = dtrain
#'  , obj = "regression"
#'  , nrounds = 5L
#'  , verbose = -1L
#' )
#' lgb.configure_fast_predict(model)
#'
#' x_single <- X[11L, , drop = FALSE]
#' predict(model, x_single)
#'
#' # Will not use it if the prediction to be made
#' # is different from what was configured
#' predict(model, x_single, type = "leaf")
#' }
#' @export
lgb.configure_fast_predict <- function(model,
                                       csr = FALSE,
                                       start_iteration = NULL,
                                       num_iteration = NULL,
                                       type = "response",
                                       params = list()) {
  if (!.is_Booster(x = model)) {
    stop("lgb.configure_fast_predict: model should be an ", sQuote("lgb.Booster", q = FALSE))
  }
  if (type == "class") {
    stop("type='class' is not supported for 'lgb.configure_fast_predict'. Use 'response' instead.")
  }

  rawscore <- FALSE
  predleaf <- FALSE
  predcontrib <- FALSE
  if (type == "raw") {
    rawscore <- TRUE
  } else if (type == "leaf") {
    predleaf <- TRUE
  } else if (type == "contrib") {
    predcontrib <- TRUE
  }

  if (csr && predcontrib) {
    stop("'lgb.configure_fast_predict' does not support feature contributions for CSR data.")
  }
  model$configure_fast_predict(
    csr = csr
    , start_iteration = start_iteration
    , num_iteration = num_iteration
    , rawscore = rawscore
    , predleaf = predleaf
    , predcontrib = predcontrib
    , params = params
  )
  return(invisible(model))
}

#' @name print.lgb.Booster
#' @title Print method for LightGBM model
#' @description Show summary information about a LightGBM model object (same as \code{summary}).
#'
#'              \emph{New in version 4.0.0}
#'
#' @param x Object of class \code{lgb.Booster}
#' @param ... Not used
#' @return The same input \code{x}, returned as invisible.
#' @export
print.lgb.Booster <- function(x, ...) {
  # nolint start
  handle <- x$.__enclos_env__$private$handle
  handle_is_null <- .is_null_handle(handle)

  if (!handle_is_null) {
    ntrees <- x$current_iter()
    if (ntrees == 1L) {
      cat("LightGBM Model (1 tree)\n")
    } else {
      cat(sprintf("LightGBM Model (%d trees)\n", ntrees))
    }
  } else {
    cat("LightGBM Model\n")
  }

  if (!handle_is_null) {
    obj <- x$params$objective
    if (is.null(obj)) {
      obj <- "(default)"
    }
    if (obj == "none") {
      obj <- "custom"
    }
    num_class <- x$.__enclos_env__$private$num_class
    if (num_class == 1L) {
      cat(sprintf("Objective: %s\n", obj))
    } else {
      cat(sprintf("Objective: %s (%d classes)\n"
          , obj
          , num_class))
    }
  } else {
    cat("(Booster handle is invalid)\n")
  }

  if (!handle_is_null) {
    ncols <- .Call(LGBM_BoosterGetNumFeature_R, handle)
    cat(sprintf("Fitted to dataset with %d columns\n", ncols))
  }
  # nolint end

  return(invisible(x))
}

#' @name summary.lgb.Booster
#' @title Summary method for LightGBM model
#' @description Show summary information about a LightGBM model object (same as \code{print}).
#'
#'              \emph{New in version 4.0.0}
#'
#' @param object Object of class \code{lgb.Booster}
#' @param ... Not used
#' @return The same input \code{object}, returned as invisible.
#' @export
summary.lgb.Booster <- function(object, ...) {
  print(object)
}

#' @name lgb.load
#' @title Load LightGBM model
#' @description Load LightGBM takes in either a file path or model string.
#'              If both are provided, Load will default to loading from file
#' @param filename path of model file
#' @param model_str a str containing the model (as a \code{character} or \code{raw} vector)
#'
#' @return lgb.Booster
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , num_threads = 2L
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#'   , early_stopping_rounds = 3L
#' )
#' model_file <- tempfile(fileext = ".txt")
#' lgb.save(model, model_file)
#' load_booster <- lgb.load(filename = model_file)
#' model_string <- model$save_model_to_string(NULL) # saves best iteration
#' load_booster_from_str <- lgb.load(model_str = model_string)
#' }
#' @export
lgb.load <- function(filename = NULL, model_str = NULL) {

  filename_provided <- !is.null(filename)
  model_str_provided <- !is.null(model_str)

  if (filename_provided) {
    if (!is.character(filename)) {
      stop("lgb.load: filename should be character")
    }
    filename <- path.expand(filename)
    if (!file.exists(filename)) {
      stop(sprintf("lgb.load: file '%s' passed to filename does not exist", filename))
    }
    return(invisible(Booster$new(modelfile = filename)))
  }

  if (model_str_provided) {
    if (!is.raw(model_str) && !is.character(model_str)) {
      stop("lgb.load: model_str should be a character/raw vector")
    }
    return(invisible(Booster$new(model_str = model_str)))
  }

  stop("lgb.load: either filename or model_str must be given")
}

#' @name lgb.save
#' @title Save LightGBM model
#' @description Save LightGBM model
#' @param booster Object of class \code{lgb.Booster}
#' @param filename Saved filename
#' @param num_iteration Number of iterations to save, NULL or <= 0 means use best iteration
#' @param start_iteration Index (1-based) of the first boosting round to save.
#'        For example, passing \code{start_iteration=5, num_iteration=3} for a regression model
#'        means "save the fifth, sixth, and seventh tree"
#'
#'        \emph{New in version 4.4.0}
#'
#' @return lgb.Booster
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , num_threads = 2L
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , valids = valids
#'   , early_stopping_rounds = 5L
#' )
#' lgb.save(model, tempfile(fileext = ".txt"))
#' }
#' @export
lgb.save <- function(
    booster, filename, num_iteration = NULL, start_iteration = 1L
  ) {

  if (!.is_Booster(x = booster)) {
    stop("lgb.save: booster should be an ", sQuote("lgb.Booster", q = FALSE))
  }

  if (!(is.character(filename) && length(filename) == 1L)) {
    stop("lgb.save: filename should be a string")
  }
  filename <- path.expand(filename)

  # Store booster
  return(
    invisible(booster$save_model(
      filename = filename
      , num_iteration = num_iteration
      , start_iteration = start_iteration
    ))
  )

}

#' @name lgb.dump
#' @title Dump LightGBM model to json
#' @description Dump LightGBM model to json
#' @param booster Object of class \code{lgb.Booster}
#' @param num_iteration Number of iterations to be dumped. NULL or <= 0 means use best iteration
#' @param start_iteration Index (1-based) of the first boosting round to dump.
#'        For example, passing \code{start_iteration=5, num_iteration=3} for a regression model
#'        means "dump the fifth, sixth, and seventh tree"
#'
#'        \emph{New in version 4.4.0}
#'
#' @return json format of model
#'
#' @examples
#' \donttest{
#' library(lightgbm)
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , num_threads = 2L
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , valids = valids
#'   , early_stopping_rounds = 5L
#' )
#' json_model <- lgb.dump(model)
#' }
#' @export
lgb.dump <- function(booster, num_iteration = NULL, start_iteration = 1L) {

  if (!.is_Booster(x = booster)) {
    stop("lgb.dump: booster should be an ", sQuote("lgb.Booster", q = FALSE))
  }

  # Return booster at requested iteration
  return(
    booster$dump_model(
      num_iteration = num_iteration, start_iteration = start_iteration
    )
  )

}

#' @name lgb.get.eval.result
#' @title Get record evaluation result from booster
#' @description Given a \code{lgb.Booster}, return evaluation results for a
#'              particular metric on a particular dataset.
#' @param booster Object of class \code{lgb.Booster}
#' @param data_name Name of the dataset to return evaluation results for.
#' @param eval_name Name of the evaluation metric to return results for.
#' @param iters An integer vector of iterations you want to get evaluation results for. If NULL
#'              (the default), evaluation results for all iterations will be returned.
#' @param is_err TRUE will return evaluation error instead
#'
#' @return numeric vector of evaluation result
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' # train a regression model
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' params <- list(
#'   objective = "regression"
#'   , metric = "l2"
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , num_threads = 2L
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#'   , valids = valids
#' )
#'
#' # Examine valid data_name values
#' print(setdiff(names(model$record_evals), "start_iter"))
#'
#' # Examine valid eval_name values for dataset "test"
#' print(names(model$record_evals[["test"]]))
#'
#' # Get L2 values for "test" dataset
#' lgb.get.eval.result(model, "test", "l2")
#' }
#' @export
lgb.get.eval.result <- function(booster, data_name, eval_name, iters = NULL, is_err = FALSE) {

  if (!.is_Booster(x = booster)) {
    stop("lgb.get.eval.result: Can only use ", sQuote("lgb.Booster", q = FALSE), " to get eval result")
  }

  if (!is.character(data_name) || !is.character(eval_name)) {
    stop("lgb.get.eval.result: data_name and eval_name should be characters")
  }

  # NOTE: "start_iter" exists in booster$record_evals but is not a valid data_name
  data_names <- setdiff(names(booster$record_evals), "start_iter")
  if (!(data_name %in% data_names)) {
    stop(paste0(
      "lgb.get.eval.result: data_name "
      , shQuote(data_name)
      , " not found. Only the following datasets exist in record evals: ["
      , toString(data_names)
      , "]"
    ))
  }

  # Check if evaluation result is existing
  eval_names <- names(booster$record_evals[[data_name]])
  if (!(eval_name %in% eval_names)) {
    stop(paste0(
      "lgb.get.eval.result: eval_name "
      , shQuote(eval_name)
      , " not found. Only the following eval_names exist for dataset "
      , shQuote(data_name)
      , ": ["
      , toString(eval_names)
      , "]"
    ))
  }

  result <- booster$record_evals[[data_name]][[eval_name]][[.EVAL_KEY()]]

  # Check if error is requested
  if (is_err) {
    result <- booster$record_evals[[data_name]][[eval_name]][[.EVAL_ERR_KEY()]]
  }

  if (is.null(iters)) {
    return(as.numeric(result))
  }

  # Parse iteration and booster delta
  iters <- as.integer(iters)
  delta <- booster$record_evals$start_iter - 1.0
  iters <- iters - delta

  return(as.numeric(result[iters]))
}
