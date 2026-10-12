data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

set.seed(708L)

# [description] Every time this function is called, it adds 0.1
#               to an accumulator then returns the current value.
#               This is used to mock the situation where an evaluation
#               metric increases every iteration
ACCUMULATOR_NAME <- "INCREASING_METRIC_ACCUMULATOR"
assign(x = ACCUMULATOR_NAME, value = 0.0, envir = .GlobalEnv)

.increasing_metric <- function(preds, dtrain) {
  if (!exists(ACCUMULATOR_NAME, envir = .GlobalEnv)) {
    assign(ACCUMULATOR_NAME, 0.0, envir = .GlobalEnv)
  }
  assign(
    x = ACCUMULATOR_NAME
    , value = get(ACCUMULATOR_NAME, envir = .GlobalEnv) + 0.1
    , envir = .GlobalEnv
  )
  return(list(
    name = "increasing_metric"
    , value = get(ACCUMULATOR_NAME, envir = .GlobalEnv)
    , higher_better = TRUE
  ))
}

# [description] Evaluation function that always returns the
#               same value
CONSTANT_METRIC_VALUE <- 0.2
.constant_metric <- function(preds, dtrain) {
  return(list(
    name = "constant_metric"
    , value = CONSTANT_METRIC_VALUE
    , higher_better = FALSE
  ))
}

# sample datasets to test early stopping
DTRAIN_RANDOM_REGRESSION <- lgb.Dataset(
  data = as.matrix(rnorm(100L), ncol = 1L, drop = FALSE)
  , label = rnorm(100L)
  , params = list(num_threads = .LGB_MAX_THREADS)
)
DVALID_RANDOM_REGRESSION <- lgb.Dataset(
  data = as.matrix(rnorm(50L), ncol = 1L, drop = FALSE)
  , label = rnorm(50L)
  , params = list(num_threads = .LGB_MAX_THREADS)
)
DTRAIN_RANDOM_CLASSIFICATION <- lgb.Dataset(
  data = as.matrix(rnorm(120L), ncol = 1L, drop = FALSE)
  , label = sample(c(0L, 1L), size = 120L, replace = TRUE)
  , params = list(num_threads = .LGB_MAX_THREADS)
)
DVALID_RANDOM_CLASSIFICATION <- lgb.Dataset(
  data = as.matrix(rnorm(37L), ncol = 1L, drop = FALSE)
  , label = sample(c(0L, 1L), size = 37L, replace = TRUE)
  , params = list(num_threads = .LGB_MAX_THREADS)
)

test_that("train and predict binary classification", {
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , params = list(
        num_leaves = 5L
        , objective = "binary"
        , metric = "binary_error"
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = nrounds
    , valids = list(
      "train" = lgb.Dataset(
        data = train$data
        , label = train$label
      )
    )
  )
  expect_false(is.null(bst$record_evals))
  record_results <- lgb.get.eval.result(bst, "train", "binary_error")
  expect_lt(min(record_results), 0.02)

  pred <- predict(bst, test$data)
  expect_equal(length(pred), 1611L)

  pred1 <- predict(bst, train$data, num_iteration = 1L)
  expect_equal(length(pred1), 6513L)
  err_pred1 <- sum((pred1 > 0.5) != train$label) / length(train$label)
  err_log <- record_results[1L]
  expect_lt(abs(err_pred1 - err_log), .LGB_NUMERIC_TOLERANCE)
})


test_that("train and predict softmax", {
  set.seed(708L)
  X_mat <- as.matrix(iris[, -5L])
  lb <- as.numeric(iris$Species) - 1L

  bst <- lightgbm(
    data = X_mat
    , label = lb
    , params = list(
        num_leaves = 4L
        , learning_rate = 0.05
        , min_data = 20L
        , min_hessian = 10.0
        , objective = "multiclass"
        , metric = "multi_error"
        , num_class = 3L
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = 20L
    , valids = list(
      "train" = lgb.Dataset(
        data = X_mat
        , label = lb
      )
    )
  )

  expect_false(is.null(bst$record_evals))
  record_results <- lgb.get.eval.result(bst, "train", "multi_error")
  expect_lt(min(record_results), 0.06)

  pred <- predict(bst, as.matrix(iris[, -5L]))
  expect_equal(length(pred), nrow(iris) * 3L)
})


test_that("use of multiple eval metrics works", {
  metrics <- list("binary_error", "auc", "binary_logloss")
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , params = list(
        num_leaves = 4L
        , learning_rate = 1.0
        , objective = "binary"
        , metric = metrics
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = 10L
    , valids = list(
      "train" = lgb.Dataset(
        data = train$data
        , label = train$label
        , params = list(num_threads = .LGB_MAX_THREADS)
      )
    )
  )
  expect_false(is.null(bst$record_evals))
  expect_named(
    bst$record_evals[["train"]]
    , unlist(metrics)
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
})

test_that("lgb.Booster.upper_bound() and lgb.Booster.lower_bound() work as expected for binary classification", {
  set.seed(708L)
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , params = list(
        num_leaves = 5L
        , objective = "binary"
        , metric = "binary_error"
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = nrounds
  )
  expect_true(abs(bst$lower_bound() - -1.590853) < .LGB_NUMERIC_TOLERANCE)
  expect_true(abs(bst$upper_bound() - 1.871015) <  .LGB_NUMERIC_TOLERANCE)
})

test_that("lgb.Booster.upper_bound() and lgb.Booster.lower_bound() work as expected for regression", {
  set.seed(708L)
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , params = list(
        num_leaves = 5L
        , objective = "regression"
        , metric = "l2"
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = nrounds
  )
  expect_true(abs(bst$lower_bound() - 0.1513859) < .LGB_NUMERIC_TOLERANCE)
  expect_true(abs(bst$upper_bound() - 0.9080349) < .LGB_NUMERIC_TOLERANCE)
})

test_that("lightgbm() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1", num_threads = .LGB_MAX_THREADS)
  for (nround_value in c(-10L, 0L)) {
    expect_error({
      bst <- lightgbm(
        data = dtrain
        , params = params
        , nrounds = nround_value
      )
    }, "nrounds should be greater than zero")
  }
})

test_that("lightgbm() accepts nrounds as either a top-level argument or parameter", {
  nrounds <- 15L

  set.seed(708L)
  top_level_bst <- lightgbm(
    data = train$data
    , label = train$label
    , nrounds = nrounds
    , params = list(
      objective = "regression"
      , metric = "l2"
      , num_leaves = 5L
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )

  set.seed(708L)
  param_bst <- lightgbm(
    data = train$data
    , label = train$label
    , params = list(
      objective = "regression"
      , metric = "l2"
      , num_leaves = 5L
      , nrounds = nrounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )

  set.seed(708L)
  both_customized <- lightgbm(
    data = train$data
    , label = train$label
    , nrounds = 20L
    , params = list(
      objective = "regression"
      , metric = "l2"
      , num_leaves = 5L
      , nrounds = nrounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )

  top_level_l2 <- top_level_bst$eval_train()[[1L]][["value"]]
  params_l2 <- param_bst$eval_train()[[1L]][["value"]]
  both_l2 <- both_customized$eval_train()[[1L]][["value"]]

  # check type just to be sure the subsetting didn't return a NULL
  expect_true(is.numeric(top_level_l2))
  expect_true(is.numeric(params_l2))
  expect_true(is.numeric(both_l2))

  # check that model produces identical performance
  expect_identical(top_level_l2, params_l2)
  expect_identical(both_l2, params_l2)

  expect_identical(param_bst$current_iter(), top_level_bst$current_iter())
  expect_identical(param_bst$current_iter(), both_customized$current_iter())
  expect_identical(param_bst$current_iter(), nrounds)

})

test_that("lightgbm() performs evaluation on validation sets if they are provided", {
  set.seed(708L)
  dvalid1 <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid2 <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , params = list(
        num_leaves = 5L
        , objective = "binary"
        , metric = c(
            "binary_error"
            , "auc"
        )
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid1
      , "valid2" = dvalid2
      , "train" = lgb.Dataset(
        data = train$data
        , label = train$label
        , params = list(num_threads = .LGB_MAX_THREADS)
      )
    )
  )

  expect_named(
    bst$record_evals
    , c("train", "valid1", "valid2", "start_iter")
    , ignore.order = TRUE
    , ignore.case = FALSE
  )
  for (valid_name in c("train", "valid1", "valid2")) {
    eval_results <- bst$record_evals[[valid_name]][["binary_error"]]
    expect_length(eval_results[["eval"]], nrounds)
  }
  expect_true(abs(bst$record_evals[["train"]][["binary_error"]][["eval"]][[1L]] - 0.02226317) < .LGB_NUMERIC_TOLERANCE)
  expect_true(abs(bst$record_evals[["valid1"]][["binary_error"]][["eval"]][[1L]] - 0.02226317) < .LGB_NUMERIC_TOLERANCE)
  expect_true(abs(bst$record_evals[["valid2"]][["binary_error"]][["eval"]][[1L]] - 0.02226317) < .LGB_NUMERIC_TOLERANCE)
})

test_that("training continuation works", {
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
    , free_raw_data = FALSE
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  watchlist <- list(train = dtrain)
  param <- list(
    objective = "binary"
    , metric = "binary_logloss"
    , num_leaves = 5L
    , learning_rate = 1.0
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )

  # train for 10 consecutive iterations
  bst <- lgb.train(param, dtrain, nrounds = 10L, watchlist)
  err_bst <- lgb.get.eval.result(bst, "train", "binary_logloss", 10L)

  #  train for 5 iterations, save, load, train for 5 more
  bst1 <- lgb.train(param, dtrain, nrounds = 5L, watchlist)
  model_file <- tempfile(fileext = ".model")
  lgb.save(bst1, model_file)
  bst2 <- lgb.train(param, dtrain, nrounds = 5L, watchlist, init_model = bst1)
  err_bst2 <- lgb.get.eval.result(bst2, "train", "binary_logloss", 10L)

  # evaluation metrics should be nearly identical for the model trained in 10 coonsecutive
  # iterations and the one trained in 5-then-5.
  expect_lt(abs(err_bst - err_bst2), 0.01)
})

test_that("cv works", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(
    objective = "regression"
    , metric = "l2,l1"
    , min_data = 1L
    , learning_rate = 1.0
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  bst <- lgb.cv(
    params
    , dtrain
    , 10L
    , nfold = 5L
    , early_stopping_rounds = 10L
  )
  expect_false(is.null(bst$record_evals))
})

test_that("CVBooster$reset_parameter() works as expected", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  n_folds <- 2L
  cv_bst <- lgb.cv(
    params = list(
      objective = "regression"
      , min_data = 1L
      , num_leaves = 7L
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = 3L
    , nfold = n_folds
  )
  expect_true(methods::is(cv_bst, "lgb.CVBooster"))
  expect_length(cv_bst$boosters, n_folds)
  for (bst in cv_bst$boosters) {
    expect_equal(bst[["booster"]]$params[["num_leaves"]], 7L)
  }
  cv_bst$reset_parameter(list(num_leaves = 11L))
  for (bst in cv_bst$boosters) {
    expect_equal(bst[["booster"]]$params[["num_leaves"]], 11L)
  }
})

test_that("lgb.cv() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label, params = list(num_threads = 2L))
  params <- list(
    objective = "regression"
    , metric = "l2,l1"
    , min_data = 1L
    , num_threads = .LGB_MAX_THREADS
  )
  for (nround_value in c(-10L, 0L)) {
    expect_error({
      bst <- lgb.cv(
        params
        , dtrain
        , nround_value
        , nfold = 5L
      )
    }, "nrounds should be greater than zero")
  }
})

test_that("lgb.cv() throws an informative error if 'data' is not an lgb.Dataset", {
  bad_values <- list(
    4L
    , "hello"
    , list(a = TRUE, b = seq_len(10L))
    , data.frame(x = seq_len(5L), y = seq_len(5L))
    , data.table::data.table(x = seq_len(5L),  y = seq_len(5L))
    , matrix(data = seq_len(10L), 2L, 5L)
  )
  for (val in bad_values) {
    expect_error({
      bst <- lgb.cv(
        params = list(
            objective = "regression"
            , metric = "l2,l1"
            , min_data = 1L
        )
        , data = val
        , 10L
        , nfold = 5L
      )
    }, regexp = "lgb.cv: data must be an lgb.Dataset instance", fixed = TRUE)
  }
})

test_that("lightgbm.cv() gives the correct best_score and best_iter for a metric where higher values are better", {
  set.seed(708L)
  dtrain <- lgb.Dataset(
    data = as.matrix(runif(n = 500L, min = 0.0, max = 15.0), drop = FALSE)
    , label = rep(c(0L, 1L), 250L)
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L
  cv_bst <- lgb.cv(
    data = dtrain
    , nfold = 5L
    , nrounds = nrounds
    , params = list(
      objective = "binary"
      , metric = "auc,binary_error"
      , learning_rate = 1.5
      , num_leaves = 5L
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )
  expect_true(methods::is(cv_bst, "lgb.CVBooster"))
  expect_named(
    cv_bst$record_evals
    , c("start_iter", "valid")
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
  auc_scores <- unlist(cv_bst$record_evals[["valid"]][["auc"]][["eval"]])
  expect_length(auc_scores, nrounds)
  expect_identical(cv_bst$best_iter, which.max(auc_scores))
  expect_identical(cv_bst$best_score, auc_scores[which.max(auc_scores)])
})

test_that("lgb.cv() fit on linearly-relatead data improves when using linear learners", {
  set.seed(708L)
  .new_dataset <- function() {
    X <- matrix(rnorm(1000L), ncol = 1L)
    return(lgb.Dataset(
      data = X
      , label = 2L * X + runif(nrow(X), 0L, 0.1)
      , params = list(num_threads = .LGB_MAX_THREADS)
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    , num_threads = .LGB_MAX_THREADS
  )

  dtrain <- .new_dataset()
  cv_bst <- lgb.cv(
    data = dtrain
    , nrounds = 10L
    , params = params
    , nfold = 5L
  )
  expect_true(methods::is(cv_bst, "lgb.CVBooster"))

  dtrain <- .new_dataset()
  cv_bst_linear <- lgb.cv(
    data = dtrain
    , nrounds = 10L
    , params = utils::modifyList(params, list(linear_tree = TRUE))
    , nfold = 5L
  )
  expect_true(methods::is(cv_bst_linear, "lgb.CVBooster"))

  expect_true(cv_bst_linear$best_score < cv_bst$best_score)
})

test_that("lgb.cv() respects showsd argument", {
  dtrain <- lgb.Dataset(train$data, label = train$label, params = list(num_threads = .LGB_MAX_THREADS))
  params <- list(
    objective = "regression"
    , metric = "l2"
    , min_data = 1L
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  nrounds <- 5L
  set.seed(708L)
  bst_showsd <- lgb.cv(
    params = params
    , data = dtrain
    , nrounds = nrounds
    , nfold = 3L
    , showsd = TRUE
  )
  evals_showsd <- bst_showsd$record_evals[["valid"]][["l2"]]
  set.seed(708L)
  bst_no_showsd <- lgb.cv(
    params = params
    , data = dtrain
    , nrounds = nrounds
    , nfold = 3L
    , showsd = FALSE
  )
  evals_no_showsd <- bst_no_showsd$record_evals[["valid"]][["l2"]]
  expect_equal(
    evals_showsd[["eval"]]
    , evals_no_showsd[["eval"]]
  )
  expect_true(methods::is(evals_showsd[["eval_err"]], "list"))
  expect_equal(length(evals_showsd[["eval_err"]]), nrounds)
  expect_identical(evals_no_showsd[["eval_err"]], list())
})

test_that("lgb.cv() raises an informative error for unrecognized objectives", {
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  expect_error({
    capture.output({
      bst <- lgb.cv(
        data = dtrain
        , params = list(
          objective_type = "not_a_real_objective"
          , verbosity = .LGB_VERBOSITY
          , num_threads = .LGB_MAX_THREADS
        )
      )
    }, type = "message")
  }, regexp = "Unknown objective type name: not_a_real_objective")
})

test_that("lgb.cv() respects parameter aliases for objective", {
  nrounds <- 3L
  nfold <- 4L
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  cv_bst <- lgb.cv(
    data = dtrain
    , params = list(
      num_leaves = 5L
      , application = "binary"
      , num_iterations = nrounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , nfold = nfold
  )
  expect_equal(cv_bst$best_iter, nrounds)
  expect_named(cv_bst$record_evals[["valid"]], "binary_logloss")
  expect_length(cv_bst$record_evals[["valid"]][["binary_logloss"]][["eval"]], nrounds)
  expect_length(cv_bst$boosters, nfold)
})

test_that("lgb.cv() prefers objective in params to keyword argument", {
  data("EuStockMarkets")
  cv_bst <- lgb.cv(
    data = lgb.Dataset(
      data = EuStockMarkets[, c("SMI", "CAC", "FTSE")]
      , label = EuStockMarkets[, "DAX"]
      , params = list(num_threads = .LGB_MAX_THREADS)
    )
    , params = list(
      application = "regression_l1"
      , verbosity = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = 5L
    , obj = "regression_l2"
  )
  for (bst_list in cv_bst$boosters) {
    bst <- bst_list[["booster"]]
    expect_equal(bst$params$objective, "regression_l1")
    # NOTE: using save_model_to_string() since that is the simplest public API in the R-package
    #       allowing access to the "objective" attribute of the Booster object on the C++ side
    model_txt_lines <- strsplit(
      x = bst$save_model_to_string()
      , split = "\n"
      , fixed = TRUE
    )[[1L]]
    expect_true(any(model_txt_lines == "objective=regression_l1"))
    expect_false(any(model_txt_lines == "objective=regression_l2"))
  }
})

test_that("lgb.cv() respects parameter aliases for metric", {
  nrounds <- 3L
  nfold <- 4L
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  cv_bst <- lgb.cv(
    data = dtrain
    , params = list(
      num_leaves = 5L
      , objective = "binary"
      , num_iterations = nrounds
      , metric_types = c("auc", "binary_logloss")
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , nfold = nfold
  )
  expect_equal(cv_bst$best_iter, nrounds)
  expect_named(cv_bst$record_evals[["valid"]], c("auc", "binary_logloss"))
  expect_length(cv_bst$record_evals[["valid"]][["binary_logloss"]][["eval"]], nrounds)
  expect_length(cv_bst$record_evals[["valid"]][["auc"]][["eval"]], nrounds)
  expect_length(cv_bst$boosters, nfold)
})

test_that("lgb.cv() respects eval_train_metric argument", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(
    objective = "regression"
    , metric = "l2"
    , min_data = 1L
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  nrounds <- 5L
  set.seed(708L)
  bst_train <- lgb.cv(
    params = params
    , data = dtrain
    , nrounds = nrounds
    , nfold = 3L
    , showsd = FALSE
    , eval_train_metric = TRUE
  )
  set.seed(708L)
  bst_no_train <- lgb.cv(
    params = params
    , data = dtrain
    , nrounds = nrounds
    , nfold = 3L
    , showsd = FALSE
    , eval_train_metric = FALSE
  )
  expect_equal(
    bst_train$record_evals[["valid"]][["l2"]]
    , bst_no_train$record_evals[["valid"]][["l2"]]
  )
  expect_true("train" %in% names(bst_train$record_evals))
  expect_false("train" %in% names(bst_no_train$record_evals))
  expect_true(methods::is(bst_train$record_evals[["train"]][["l2"]][["eval"]], "list"))
  expect_equal(
    length(bst_train$record_evals[["train"]][["l2"]][["eval"]])
    , nrounds
  )
})

test_that("lgb.train() works as expected with multiple eval metrics", {
  metrics <- c("binary_error", "auc", "binary_logloss")
  bst <- lgb.train(
    data = lgb.Dataset(
      train$data
      , label = train$label
      , params = list(num_threads = .LGB_MAX_THREADS)
    )
    , nrounds = 10L
    , params = list(
      objective = "binary"
      , metric = metrics
      , learning_rate = 1.0
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , valids = list(
      "train" = lgb.Dataset(
        train$data
        , label = train$label
        , params = list(num_threads = .LGB_MAX_THREADS)
      )
    )
  )
  expect_false(is.null(bst$record_evals))
  expect_named(
    bst$record_evals[["train"]]
    , unlist(metrics)
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
})

test_that("lgb.train() raises an informative error for unrecognized objectives", {
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
  )
  expect_error({
    capture.output({
      bst <- lgb.train(
        data = dtrain
        , params = list(
          objective_type = "not_a_real_objective"
          , verbosity = .LGB_VERBOSITY
        )
      )
    }, type = "message")
  }, regexp = "Unknown objective type name: not_a_real_objective")
})

test_that("lgb.train() raises an informative error for unrecognized metrics", {
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  expect_error({
    capture.output({
      bst <- lgb.train(
        data = dtrain
        , valids = list(valid = dtrain)
        , params = list(
          objective = "binary"
          , metric = "nonsense"
          , verbosity = .LGB_VERBOSITY
          , num_threads = .LGB_MAX_THREADS
          , num_iterations = 2L
        )
      )
    }, type = "message")
  }, regexp = "Unknown metric 'nonsense'")
})

test_that("lgb.cv() raises an informative error for unrecognized metrics", {
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  expect_error({
    capture.output({
      bst <- lgb.cv(
        data = dtrain
        , params = list(
          objective = "binary"
          , metric = "nonsense"
          , verbosity = .LGB_VERBOSITY
          , num_threads = .LGB_MAX_THREADS
        )
        , nfold = 2L
        , nrounds = 2L
      )
    }, type = "message")
  }, regexp = "Unknown metric 'nonsense'")
})

test_that("lgb.train() respects parameter aliases for objective", {
  nrounds <- 3L
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  bst <- lgb.train(
    data = dtrain
    , params = list(
      num_leaves = 5L
      , application = "binary"
      , num_iterations = nrounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , valids = list(
      "the_training_data" = dtrain
    )
  )
  expect_named(bst$record_evals[["the_training_data"]], "binary_logloss")
  expect_length(bst$record_evals[["the_training_data"]][["binary_logloss"]][["eval"]], nrounds)
  expect_equal(bst$params[["objective"]], "binary")
})

test_that("lgb.train() prefers objective in params to keyword argument", {
  data("EuStockMarkets")
  bst <- lgb.train(
    data = lgb.Dataset(
      data = EuStockMarkets[, c("SMI", "CAC", "FTSE")]
      , label = EuStockMarkets[, "DAX"]
      , params = list(num_threads = .LGB_MAX_THREADS)
    )
    , params = list(
        loss = "regression_l1"
        , verbosity = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = 5L
    , obj = "regression_l2"
  )
  expect_equal(bst$params$objective, "regression_l1")
  # NOTE: using save_model_to_string() since that is the simplest public API in the R-package
  #       allowing access to the "objective" attribute of the Booster object on the C++ side
  model_txt_lines <- strsplit(
    x = bst$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(model_txt_lines == "objective=regression_l1"))
  expect_false(any(model_txt_lines == "objective=regression_l2"))
})

test_that("lgb.train() respects parameter aliases for metric", {
  nrounds <- 3L
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  bst <- lgb.train(
    data = dtrain
    , params = list(
      num_leaves = 5L
      , objective = "binary"
      , num_iterations = nrounds
      , metric_types = c("auc", "binary_logloss")
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , valids = list(
      "train" = dtrain
    )
  )
  record_results <- bst$record_evals[["train"]]
  expect_equal(sort(names(record_results)), c("auc", "binary_logloss"))
  expect_length(record_results[["auc"]][["eval"]], nrounds)
  expect_length(record_results[["binary_logloss"]][["eval"]], nrounds)
  expect_equal(bst$params[["metric"]], list("auc", "binary_logloss"))
})

test_that("lgb.train() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label, params = list(num_threads = .LGB_MAX_THREADS))
  params <- list(
    objective = "regression"
    , metric = "l2,l1"
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  for (nround_value in c(-10L, 0L)) {
    expect_error({
      bst <- lgb.train(
        params
        , dtrain
        , nround_value
      )
    }, "nrounds should be greater than zero")
  }
})


test_that("lgb.train() accepts nrounds as either a top-level argument or parameter", {
  nrounds <- 15L

  set.seed(708L)
  top_level_bst <- lgb.train(
    data = lgb.Dataset(
      train$data
      , label = train$label
      , params = list(num_threads = .LGB_MAX_THREADS)
    )
    , nrounds = nrounds
    , params = list(
      objective = "regression"
      , metric = "l2"
      , num_leaves = 5L
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )

  set.seed(708L)
  param_bst <- lgb.train(
    data = lgb.Dataset(
      train$data
      , label = train$label
      , params = list(num_threads = .LGB_MAX_THREADS)
    )
    , params = list(
      objective = "regression"
      , metric = "l2"
      , num_leaves = 5L
      , nrounds = nrounds
      , verbose = .LGB_VERBOSITY
    )
  )

  set.seed(708L)
  both_customized <- lgb.train(
    data = lgb.Dataset(
      train$data
      , label = train$label
      , params = list(num_threads = .LGB_MAX_THREADS)
    )
    , nrounds = 20L
    , params = list(
      objective = "regression"
      , metric = "l2"
      , num_leaves = 5L
      , nrounds = nrounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )

  top_level_l2 <- top_level_bst$eval_train()[[1L]][["value"]]
  params_l2 <- param_bst$eval_train()[[1L]][["value"]]
  both_l2 <- both_customized$eval_train()[[1L]][["value"]]

  # check type just to be sure the subsetting didn't return a NULL
  expect_true(is.numeric(top_level_l2))
  expect_true(is.numeric(params_l2))
  expect_true(is.numeric(both_l2))

  # check that model produces identical performance
  expect_identical(top_level_l2, params_l2)
  expect_identical(both_l2, params_l2)

  expect_identical(param_bst$current_iter(), top_level_bst$current_iter())
  expect_identical(param_bst$current_iter(), both_customized$current_iter())
  expect_identical(param_bst$current_iter(), nrounds)

})


test_that("lgb.train() throws an informative error if 'data' is not an lgb.Dataset", {
  bad_values <- list(
    4L
    , "hello"
    , list(a = TRUE, b = seq_len(10L))
    , data.frame(x = seq_len(5L), y = seq_len(5L))
    , data.table::data.table(x = seq_len(5L),  y = seq_len(5L))
    , matrix(data = seq_len(10L), 2L, 5L)
  )
  for (val in bad_values) {
    expect_error({
      bst <- lgb.train(
        params = list(
            objective = "regression"
            , metric = "l2,l1"
            , verbose = .LGB_VERBOSITY
        )
        , data = val
        , 10L
      )
    }, regexp = "data must be an lgb.Dataset instance", fixed = TRUE)
  }
})

test_that("lgb.train() throws an informative error if 'valids' is not a list of lgb.Dataset objects", {
  valids <- list(
    "valid1" = data.frame(x = rnorm(5L), y = rnorm(5L))
    , "valid2" = data.frame(x = rnorm(5L), y = rnorm(5L))
  )
  expect_error({
    bst <- lgb.train(
      params = list(
        objective = "regression"
        , metric = "l2,l1"
        , verbose = .LGB_VERBOSITY
      )
      , data = lgb.Dataset(train$data, label = train$label)
      , 10L
      , valids = valids
    )
  }, regexp = "valids must be a list of lgb.Dataset elements")
})

test_that("lgb.train() errors if 'valids' is a list of lgb.Dataset objects but some do not have names", {
  valids <- list(
    "valid1" = lgb.Dataset(matrix(rnorm(10L), 5L, 2L))
    , lgb.Dataset(matrix(rnorm(10L), 2L, 5L))
  )
  expect_error({
    bst <- lgb.train(
      params = list(
        objective = "regression"
        , metric = "l2,l1"
        , verbose = .LGB_VERBOSITY
      )
      , data = lgb.Dataset(train$data, label = train$label)
      , 10L
      , valids = valids
    )
  }, regexp = "each element of valids must have a name")
})

test_that("lgb.train() throws an informative error if 'valids' contains lgb.Dataset objects but none have names", {
  valids <- list(
    lgb.Dataset(matrix(rnorm(10L), 5L, 2L))
    , lgb.Dataset(matrix(rnorm(10L), 2L, 5L))
  )
  expect_error({
    bst <- lgb.train(
      params = list(
        objective = "regression"
        , metric = "l2,l1"
        , verbose = .LGB_VERBOSITY
    )
      , data = lgb.Dataset(train$data, label = train$label)
      , 10L
      , valids = valids
    )
  }, regexp = "each element of valids must have a name")
})

test_that("lgb.train() works with force_col_wise and force_row_wise", {
  set.seed(1234L)
  nrounds <- 10L
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  params <- list(
    objective = "binary"
    , metric = "binary_error"
    , force_col_wise = TRUE
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  bst_col_wise <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = nrounds
  )

  params <- list(
    objective = "binary"
    , metric = "binary_error"
    , force_row_wise = TRUE
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  bst_row_wise <- lgb.train(
    params = params
    , data = dtrain
    , nrounds = nrounds
  )

  expected_error <- 0.003070782
  expect_equal(bst_col_wise$eval_train()[[1L]][["value"]], expected_error)
  expect_equal(bst_row_wise$eval_train()[[1L]][["value"]], expected_error)

  # check some basic details of the boosters just to be sure force_col_wise
  # and force_row_wise are not causing any weird side effects
  for (bst in list(bst_row_wise, bst_col_wise)) {
    expect_equal(bst$current_iter(), nrounds)
    parsed_model <- jsonlite::fromJSON(bst$dump_model())
    expect_equal(parsed_model$objective, "binary sigmoid:1")
    expect_false(parsed_model$average_output)
  }
})

test_that("lgb.train() works as expected with sparse features", {
  set.seed(708L)
  num_obs <- 70000L
  trainDF <- data.frame(
    y = sample(c(0L, 1L), size = num_obs, replace = TRUE)
    , x = sample(c(1.0:10.0, rep(NA_real_, 50L)), size = num_obs, replace = TRUE)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["x"]], drop = FALSE)
    , label = trainDF[["y"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 1L
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , min_data = 1L
      , min_data_in_bin = 1L
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = nrounds
  )

  expect_true(.is_Booster(bst))
  expect_equal(bst$current_iter(), nrounds)
  parsed_model <- jsonlite::fromJSON(bst$dump_model())
  expect_equal(parsed_model$objective, "binary sigmoid:1")
  expect_false(parsed_model$average_output)
  expected_error <- 0.6931268
  expect_true(abs(bst$eval_train()[[1L]][["value"]] - expected_error) < .LGB_NUMERIC_TOLERANCE)
})

test_that("lgb.train() works with early stopping for classification", {
  trainDF <- data.frame(
    "feat1" = rep(c(5.0, 10.0), 500L)
    , "target" = rep(c(0L, 1L), 500L)
  )
  validDF <- data.frame(
    "feat1" = rep(c(5.0, 10.0), 50L)
    , "target" = rep(c(0L, 1L), 50L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L

  ################################
  # train with no early stopping #
  ################################
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # a perfect model should be trivial to obtain, but all 10 rounds
  # should happen
  expect_equal(bst$best_score, 0.0)
  expect_equal(bst$best_iter, 1L)
  expect_equal(length(bst$record_evals[["valid1"]][["binary_error"]][["eval"]]), nrounds)

  #############################
  # train with early stopping #
  #############################
  early_stopping_rounds <- 5L
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
      , early_stopping_rounds = early_stopping_rounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # a perfect model should be trivial to obtain, and only 6 rounds
  # should have happen (1 with improvement, 5 consecutive with no improvement)
  expect_equal(bst$best_score, 0.0)
  expect_equal(bst$best_iter, 1L)
  expect_equal(
    length(bst$record_evals[["valid1"]][["binary_error"]][["eval"]])
    , early_stopping_rounds + 1L
  )

})

test_that("lgb.train() treats early_stopping_rounds<=0 as disabling early stopping", {
  set.seed(708L)
  trainDF <- data.frame(
    "feat1" = rep(c(5.0, 10.0), 500L)
    , "target" = rep(c(0L, 1L), 500L)
  )
  validDF <- data.frame(
    "feat1" = rep(c(5.0, 10.0), 50L)
    , "target" = rep(c(0L, 1L), 50L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 5L

  for (value in c(-5L, 0L)) {

    #----------------------------#
    # passed as keyword argument #
    #----------------------------#
    bst <- lgb.train(
      params = list(
        objective = "binary"
        , metric = "binary_error"
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
      )
      , data = dtrain
      , nrounds = nrounds
      , valids = list(
        "valid1" = dvalid
      )
      , early_stopping_rounds = value
    )

    # a perfect model should be trivial to obtain, but all 10 rounds
    # should happen
    expect_equal(bst$best_score, 0.0)
    expect_equal(bst$best_iter, 1L)
    expect_equal(length(bst$record_evals[["valid1"]][["binary_error"]][["eval"]]), nrounds)

    #---------------------------#
    # passed as parameter alias #
    #---------------------------#
    bst <- lgb.train(
      params = list(
        objective = "binary"
        , metric = "binary_error"
        , n_iter_no_change = value
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
      )
      , data = dtrain
      , nrounds = nrounds
      , valids = list(
        "valid1" = dvalid
      )
    )

    # a perfect model should be trivial to obtain, but all 10 rounds
    # should happen
    expect_equal(bst$best_score, 0.0)
    expect_equal(bst$best_iter, 1L)
    expect_equal(length(bst$record_evals[["valid1"]][["binary_error"]][["eval"]]), nrounds)
  }
})

test_that("lgb.train() works with early stopping for classification with a metric that should be maximized", {
  set.seed(708L)
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid <- lgb.Dataset(
    data = test$data
    , label = test$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L

  #############################
  # train with early stopping #
  #############################
  early_stopping_rounds <- 5L
  # the harsh max_depth guarantees that AUC improves over at least the first few iterations
  bst_auc <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "auc"
      , max_depth = 3L
      , early_stopping_rounds = early_stopping_rounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )
  bst_binary_error <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
      , max_depth = 3L
      , early_stopping_rounds = early_stopping_rounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # early stopping should have been hit for binary_error (higher_better = FALSE)
  eval_info <- bst_binary_error$.__enclos_env__$private$get_eval_info()
  expect_identical(eval_info, "binary_error")
  expect_identical(
    unname(bst_binary_error$.__enclos_env__$private$higher_better_inner_eval)
    , FALSE
  )
  expect_identical(bst_binary_error$best_iter, 1L)
  expect_identical(bst_binary_error$current_iter(), early_stopping_rounds + 1L)
  expect_true(abs(bst_binary_error$best_score - 0.01613904) < .LGB_NUMERIC_TOLERANCE)

  # early stopping should not have been hit for AUC (higher_better = TRUE)
  eval_info <- bst_auc$.__enclos_env__$private$get_eval_info()
  expect_identical(eval_info, "auc")
  expect_identical(
    unname(bst_auc$.__enclos_env__$private$higher_better_inner_eval)
    , TRUE
  )
  expect_identical(bst_auc$best_iter, 9L)
  expect_identical(bst_auc$current_iter(), nrounds)
  expect_true(abs(bst_auc$best_score - 0.9999969) < .LGB_NUMERIC_TOLERANCE)
})

test_that("lgb.train() works with early stopping for regression", {
  set.seed(708L)
  trainDF <- data.frame(
    "feat1" = rep(c(10.0, 100.0), 500L)
    , "target" = rep(c(-50.0, 50.0), 500L)
  )
  validDF <- data.frame(
    "feat1" = rep(50.0, 4L)
    , "target" = rep(50.0, 4L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L

  ################################
  # train with no early stopping #
  ################################
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "rmse"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # the best possible model should come from the first iteration, but
  # all 10 training iterations should happen
  expect_equal(bst$best_score, 55.0)
  expect_equal(bst$best_iter, 1L)
  expect_equal(length(bst$record_evals[["valid1"]][["rmse"]][["eval"]]), nrounds)

  #############################
  # train with early stopping #
  #############################
  early_stopping_rounds <- 5L
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "rmse"
      , early_stopping_rounds = early_stopping_rounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # the best model should be from the first iteration, and only 6 rounds
  # should have happen (1 with improvement, 5 consecutive with no improvement)
  expect_equal(bst$best_score, 55.0)
  expect_equal(bst$best_iter, 1L)
  expect_equal(
    length(bst$record_evals[["valid1"]][["rmse"]][["eval"]])
    , early_stopping_rounds + 1L
  )
})

test_that("lgb.train() does not stop early if early_stopping_rounds is not given", {
  set.seed(708L)

  increasing_metric_starting_value <- get(
    ACCUMULATOR_NAME
    , envir = .GlobalEnv
  )
  nrounds <- 10L
  metrics <- list(
    .constant_metric
    , .increasing_metric
  )
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "None"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_REGRESSION
    , nrounds = nrounds
    , valids = list("valid1" = DVALID_RANDOM_REGRESSION)
    , eval = metrics
  )

  # Only the two functions provided to "eval" should have been evaluated
  expect_equal(length(bst$record_evals[["valid1"]]), 2L)

  # all 10 iterations should have happen, and the best_iter should be
  # the first one (based on constant_metric)
  best_iter <- 1L
  expect_equal(bst$best_iter, best_iter)

  # best_score should be taken from the first metric
  expect_equal(
    bst$best_score
    , bst$record_evals[["valid1"]][["constant_metric"]][["eval"]][[best_iter]]
  )

  # early stopping should not have happened. Even though constant_metric
  # had 9 consecutive iterations with no improvement, it is ignored because of
  # first_metric_only = TRUE
  expect_equal(
    length(bst$record_evals[["valid1"]][["constant_metric"]][["eval"]])
    , nrounds
  )
  expect_equal(
    length(bst$record_evals[["valid1"]][["increasing_metric"]][["eval"]])
    , nrounds
  )
})

test_that("If first_metric_only is not given or is FALSE, lgb.train() decides to stop early based on all metrics", {
  set.seed(708L)

  early_stopping_rounds <- 3L
  param_variations <- list(
    list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
      , first_metric_only = FALSE
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )

  for (params in param_variations) {

    nrounds <- 10L
    bst <- lgb.train(
      params = params
      , data = DTRAIN_RANDOM_REGRESSION
      , nrounds = nrounds
      , valids = list(
        "valid1" = DVALID_RANDOM_REGRESSION
      )
      , eval = list(
        .increasing_metric
        , .constant_metric
      )
    )

    # Only the two functions provided to "eval" should have been evaluated
    expect_equal(length(bst$record_evals[["valid1"]]), 2L)

    # early stopping should have happened, and should have stopped early_stopping_rounds + 1 rounds in
    # because constant_metric never improves
    #
    # the best iteration should be the last one, because increasing_metric was first
    # and gets better every iteration
    best_iter <- early_stopping_rounds + 1L
    expect_equal(bst$best_iter, best_iter)

    # best_score should be taken from "increasing_metric" because it was first
    expect_equal(
      bst$best_score
      , bst$record_evals[["valid1"]][["increasing_metric"]][["eval"]][[best_iter]]
    )

    # early stopping should not have happened. even though increasing_metric kept
    # getting better, early stopping should have happened because "constant_metric"
    # did not improve
    expect_equal(
      length(bst$record_evals[["valid1"]][["constant_metric"]][["eval"]])
      , early_stopping_rounds + 1L
    )
    expect_equal(
      length(bst$record_evals[["valid1"]][["increasing_metric"]][["eval"]])
      , early_stopping_rounds + 1L
    )
  }

})

test_that("If first_metric_only is TRUE, lgb.train() decides to stop early based on only the first metric", {
  set.seed(708L)
  nrounds <- 10L
  early_stopping_rounds <- 3L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
      , first_metric_only = TRUE
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_REGRESSION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_REGRESSION
    )
    , eval = list(
      .increasing_metric
      , .constant_metric
    )
  )

  # Only the two functions provided to "eval" should have been evaluated
  expect_equal(length(bst$record_evals[["valid1"]]), 2L)

  # all 10 iterations should happen, and the best_iter should be the final one
  expect_equal(bst$best_iter, nrounds)

  # best_score should be taken from "increasing_metric"
  expect_equal(
    bst$best_score
    , increasing_metric_starting_value + 0.1 * nrounds
  )

  # early stopping should not have happened. Even though constant_metric
  # had 9 consecutive iterations with no improvement, it is ignored because of
  # first_metric_only = TRUE
  expect_equal(
    length(bst$record_evals[["valid1"]][["constant_metric"]][["eval"]])
    , nrounds
  )
  expect_equal(
    length(bst$record_evals[["valid1"]][["increasing_metric"]][["eval"]])
    , nrounds
  )
})

test_that("lgb.train() works when a mixture of functions and strings are passed to eval", {
  set.seed(708L)
  nrounds <- 10L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "None"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_REGRESSION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_REGRESSION
    )
    , eval = list(
      .increasing_metric
      , "rmse"
      , .constant_metric
      , "l2"
    )
  )

  # all 4 metrics should have been used
  expect_named(
    bst$record_evals[["valid1"]]
    , expected = c("rmse", "l2", "increasing_metric", "constant_metric")
    , ignore.order = TRUE
    , ignore.case = FALSE
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid1"]]
  expect_true(abs(results[["rmse"]][["eval"]][[1L]] - 1.105012) < .LGB_NUMERIC_TOLERANCE)
  expect_true(abs(results[["l2"]][["eval"]][[1L]] - 1.221051) < .LGB_NUMERIC_TOLERANCE)
  expected_increasing_metric <- increasing_metric_starting_value + 0.1
  expect_true(
    abs(
      results[["increasing_metric"]][["eval"]][[1L]] - expected_increasing_metric
    ) < .LGB_NUMERIC_TOLERANCE
  )
  expect_true(abs(results[["constant_metric"]][["eval"]][[1L]] - CONSTANT_METRIC_VALUE) < .LGB_NUMERIC_TOLERANCE)

})

test_that("lgb.train() works when a list of strings or a character vector is passed to eval", {

  # testing list and character vector, as well as length-1 and length-2
  eval_variations <- list(
    c("binary_error", "binary_logloss")
    , "binary_logloss"
    , list("binary_error", "binary_logloss")
    , list("binary_logloss")
  )

  for (eval_variation in eval_variations) {

    set.seed(708L)
    nrounds <- 10L
    increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
    bst <- lgb.train(
      params = list(
        objective = "binary"
        , metric = "None"
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
      )
      , data = DTRAIN_RANDOM_CLASSIFICATION
      , nrounds = nrounds
      , valids = list(
        "valid1" = DVALID_RANDOM_CLASSIFICATION
      )
      , eval = eval_variation
    )

    # both metrics should have been used
    expect_named(
      bst$record_evals[["valid1"]]
      , expected = unlist(eval_variation)
      , ignore.order = TRUE
      , ignore.case = FALSE
    )

    # the difference metrics shouldn't have been mixed up with each other
    results <- bst$record_evals[["valid1"]]
    if ("binary_error" %in% unlist(eval_variation)) {
      expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.4864865) < .LGB_NUMERIC_TOLERANCE)
    }
    if ("binary_logloss" %in% unlist(eval_variation)) {
      expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.6932548) < .LGB_NUMERIC_TOLERANCE)
    }
  }
})

test_that("lgb.train() works when you specify both 'metric' and 'eval' with strings", {
  set.seed(708L)
  nrounds <- 10L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_CLASSIFICATION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_CLASSIFICATION
    )
    , eval = "binary_logloss"
  )

  # both metrics should have been used
  expect_named(
    bst$record_evals[["valid1"]]
    , expected = c("binary_error", "binary_logloss")
    , ignore.order = TRUE
    , ignore.case = FALSE
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid1"]]
  expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.4864865) < .LGB_NUMERIC_TOLERANCE)
  expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.6932548) < .LGB_NUMERIC_TOLERANCE)
})

test_that("lgb.train() works when you give a function for eval", {
  set.seed(708L)
  nrounds <- 10L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "None"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_CLASSIFICATION
    , nrounds = nrounds
    , valids = list(
      "valid1" = DVALID_RANDOM_CLASSIFICATION
    )
    , eval = .constant_metric
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid1"]]
  expect_true(abs(results[["constant_metric"]][["eval"]][[1L]] - CONSTANT_METRIC_VALUE) < .LGB_NUMERIC_TOLERANCE)
})

test_that("lgb.train() works with early stopping for regression with a metric that should be minimized", {
  set.seed(708L)
  trainDF <- data.frame(
    "feat1" = rep(c(10.0, 100.0), 500L)
    , "target" = rep(c(-50.0, 50.0), 500L)
  )
  validDF <- data.frame(
    "feat1" = rep(50.0, 4L)
    , "target" = rep(50.0, 4L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L

  #############################
  # train with early stopping #
  #############################
  early_stopping_rounds <- 5L
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = c(
          "mape"
          , "rmse"
          , "mae"
      )
      , min_data_in_bin = 5L
      , early_stopping_rounds = early_stopping_rounds
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )

  # the best model should be from the first iteration, and only 6 rounds
  # should have happened (1 with improvement, 5 consecutive with no improvement)
  expect_equal(bst$best_score, 1.1)
  expect_equal(bst$best_iter, 1L)
  expect_equal(
    length(bst$record_evals[["valid1"]][["mape"]][["eval"]])
    , early_stopping_rounds + 1L
  )

  # Booster should understand that all three of these metrics should be minimized
  eval_info <- bst$.__enclos_env__$private$get_eval_info()
  expect_identical(eval_info, c("mape", "rmse", "l1"))
  expect_identical(
    unname(bst$.__enclos_env__$private$higher_better_inner_eval)
    , rep(FALSE, 3L)
  )
})


test_that("lgb.train() supports non-ASCII feature names", {
  # content below is equivalent to
  #
  #  feature_names <- c("F_零", "F_一", "F_二", "F_三")
  #
  # but using rawToChar() to avoid weird issues when {testthat}
  # sources files and converts their encodings prior to evaluating the code
  feature_names <- c(
    rawToChar(as.raw(c(0x46, 0x5f, 0xe9, 0x9b, 0xb6)))
    , rawToChar(as.raw(c(0x46, 0x5f, 0xe4, 0xb8, 0x80)))
    , rawToChar(as.raw(c(0x46, 0x5f, 0xe4, 0xba, 0x8c)))
    , rawToChar(as.raw(c(0x46, 0x5f, 0xe4, 0xb8, 0x89)))
  )
  dtrain <- lgb.Dataset(
    data = matrix(rnorm(400L), ncol =  4L)
    , label = rnorm(100L)
    , params = list(num_threads = .LGB_MAX_THREADS)
    , colnames = feature_names
  )
  bst <- lgb.train(
    data = dtrain
    , nrounds = 5L
    , obj = "regression"
    , params = list(
      metric = "rmse"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )
  expect_true(.is_Booster(bst))
  dumped_model <- jsonlite::fromJSON(bst$dump_model())

  # UTF-8 strings are not well-supported on Windows
  # * https://developer.r-project.org/Blog/public/2020/05/02/utf-8-support-on-windows/
  # * https://developer.r-project.org/Blog/public/2020/07/30/windows/utf-8-build-of-r-and-cran-packages/index.html
  if (.LGB_UTF8_LOCALE && !.LGB_ON_WINDOWS) {
    expect_identical(
      dumped_model[["feature_names"]]
      , feature_names
    )
  } else {
    expect_identical(
      dumped_model[["feature_names"]]
      , iconv(feature_names, to = "UTF-8")
    )
  }
})

test_that("lgb.train() works with integer, double, and numeric data", {
  data(mtcars)
  X <- as.matrix(mtcars[, -1L])
  y <- mtcars[, 1L, drop = TRUE]
  expected_mae <- 4.263667
  for (data_mode in c("numeric", "double", "integer")) {
    mode(X) <- data_mode
    nrounds <- 10L
    bst <- lightgbm(
      data = X
      , label = y
      , params = list(
        objective = "regression"
        , min_data_in_bin = 1L
        , min_data_in_leaf = 1L
        , learning_rate = 0.01
        , seed = 708L
        , verbose = .LGB_VERBOSITY
      )
      , nrounds = nrounds
    )

    # should have trained for 10 iterations and found splits
    modelDT <- lgb.model.dt.tree(bst)
    expect_equal(modelDT[, max(tree_index)], nrounds - 1L)
    expect_gt(nrow(modelDT), nrounds * 3L)

    # should have achieved expected performance
    preds <- predict(bst, X)
    mae <- mean(abs(y - preds))
    expect_true(abs(mae - expected_mae) < .LGB_NUMERIC_TOLERANCE)
  }
})

test_that("lgb.train() updates params based on keyword arguments", {
  dtrain <- lgb.Dataset(
    data = matrix(rnorm(400L), ncol =  4L)
    , label = rnorm(100L)
    , params = list(num_threads = .LGB_MAX_THREADS)
  )

  # defaults from keyword arguments should be used if not specified in params
  invisible(
    capture.output({
      bst <- lgb.train(
        data = dtrain
        , obj = "regression"
        , params = list(num_threads = .LGB_MAX_THREADS)
      )
    })
  )
  expect_equal(bst$params[["verbosity"]], 1L)
  expect_equal(bst$params[["num_iterations"]], 100L)

  # main param names should be preferred to keyword arguments
  invisible(
    capture.output({
      bst <- lgb.train(
        data = dtrain
        , obj = "regression"
        , params = list(
          "verbosity" = 5L
          , "num_iterations" = 2L
          , num_threads = .LGB_MAX_THREADS
        )
      )
    })
  )
  expect_equal(bst$params[["verbosity"]], 5L)
  expect_equal(bst$params[["num_iterations"]], 2L)

  # aliases should be preferred to keyword arguments, and converted to main parameter name
  invisible(
    capture.output({
      bst <- lgb.train(
        data = dtrain
        , obj = "regression"
        , params = list(
          "verbose" = 5L
          , "num_boost_round" = 2L
          , num_threads = .LGB_MAX_THREADS
        )
      )
    })
  )
  expect_equal(bst$params[["verbosity"]], 5L)
  expect_false("verbose" %in% bst$params)
  expect_equal(bst$params[["num_iterations"]], 2L)
  expect_false("num_boost_round" %in% bst$params)
})

test_that("when early stopping is not activated, best_iter and best_score come from valids and not training data", {
  set.seed(708L)
  trainDF <- data.frame(
    "feat1" = rep(c(10.0, 100.0), 500L)
    , "target" = rep(c(-50.0, 50.0), 500L)
  )
  validDF <- data.frame(
    "feat1" = rep(50.0, 4L)
    , "target" = rep(50.0, 4L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid1 <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid2 <- lgb.Dataset(
    data = as.matrix(validDF[1L:10L, "feat1"], drop = FALSE)
    , label = validDF[1L:10L, "target"]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L
  train_params <- list(
    objective = "regression"
    , metric = "rmse"
    , learning_rate = 1.5
    , num_leaves = 5L
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )

  # example 1: two valids, neither are the training data
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid1
      , "valid2" = dvalid2
    )
    , params = train_params
  )
  expect_named(
    bst$record_evals
    , c("start_iter", "valid1", "valid2")
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
  rmse_scores <- unlist(bst$record_evals[["valid1"]][["rmse"]][["eval"]])
  expect_length(rmse_scores, nrounds)
  expect_identical(bst$best_iter, which.min(rmse_scores))
  expect_identical(bst$best_score, rmse_scores[which.min(rmse_scores)])

  # example 2: train first (called "train") and two valids
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , valids = list(
      "train" = dtrain
      , "valid1" = dvalid1
      , "valid2" = dvalid2
    )
    , params = train_params
  )
  expect_named(
    bst$record_evals
    , c("start_iter", "train", "valid1", "valid2")
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
  rmse_scores <- unlist(bst$record_evals[["valid1"]][["rmse"]][["eval"]])
  expect_length(rmse_scores, nrounds)
  expect_identical(bst$best_iter, which.min(rmse_scores))
  expect_identical(bst$best_score, rmse_scores[which.min(rmse_scores)])

  # example 3: train second (called "train") and two valids
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid1
      , "train" = dtrain
      , "valid2" = dvalid2
    )
    , params = train_params
  )
  # note that "train" still ends up as the first one
  expect_named(
    bst$record_evals
    , c("start_iter", "train", "valid1", "valid2")
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
  rmse_scores <- unlist(bst$record_evals[["valid1"]][["rmse"]][["eval"]])
  expect_length(rmse_scores, nrounds)
  expect_identical(bst$best_iter, which.min(rmse_scores))
  expect_identical(bst$best_score, rmse_scores[which.min(rmse_scores)])

  # example 4: train third (called "train") and two valids
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid1
      , "valid2" = dvalid2
      , "train" = dtrain
    )
    , params = train_params
  )
  # note that "train" still ends up as the first one
  expect_named(
    bst$record_evals
    , c("start_iter", "train", "valid1", "valid2")
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
  rmse_scores <- unlist(bst$record_evals[["valid1"]][["rmse"]][["eval"]])
  expect_length(rmse_scores, nrounds)
  expect_identical(bst$best_iter, which.min(rmse_scores))
  expect_identical(bst$best_score, rmse_scores[which.min(rmse_scores)])

  # example 5: train second (called "something-random-we-would-not-hardcode") and two valids
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid1
      , "something-random-we-would-not-hardcode" = dtrain
      , "valid2" = dvalid2
    )
    , params = train_params
  )
  # note that "something-random-we-would-not-hardcode" was recognized as the training
  # data even though it isn't named "train"
  expect_named(
    bst$record_evals
    , c("start_iter", "something-random-we-would-not-hardcode", "valid1", "valid2")
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
  rmse_scores <- unlist(bst$record_evals[["valid1"]][["rmse"]][["eval"]])
  expect_length(rmse_scores, nrounds)
  expect_identical(bst$best_iter, which.min(rmse_scores))
  expect_identical(bst$best_score, rmse_scores[which.min(rmse_scores)])

  # example 6: the only valid supplied is the training data
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , valids = list(
      "train" = dtrain
    )
    , params = train_params
  )
  expect_identical(bst$best_iter, -1L)
  expect_identical(bst$best_score, NA_real_)
})

test_that("lightgbm.train() gives the correct best_score and best_iter for a metric where higher values are better", {
  set.seed(708L)
  trainDF <- data.frame(
    "feat1" = runif(n = 500L, min = 0.0, max = 15.0)
    , "target" = rep(c(0L, 1L), 500L)
  )
  validDF <- data.frame(
    "feat1" = runif(n = 50L, min = 0.0, max = 15.0)
    , "target" = rep(c(0L, 1L), 50L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid1 <- lgb.Dataset(
    data = as.matrix(validDF[1L:25L, "feat1"], drop = FALSE)
    , label = validDF[1L:25L, "target"]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid1
      , "something-random-we-would-not-hardcode" = dtrain
    )
    , params = list(
      objective = "binary"
      , metric = "auc"
      , learning_rate = 1.5
      , num_leaves = 5L
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
  )
  # note that "something-random-we-would-not-hardcode" was recognized as the training
  # data even though it isn't named "train"
  expect_named(
    bst$record_evals
    , c("start_iter", "something-random-we-would-not-hardcode", "valid1")
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
  auc_scores <- unlist(bst$record_evals[["valid1"]][["auc"]][["eval"]])
  expect_length(auc_scores, nrounds)
  expect_identical(bst$best_iter, which.max(auc_scores))
  expect_identical(bst$best_score, auc_scores[which.max(auc_scores)])
})

test_that("using lightgbm() without early stopping, best_iter and best_score come from valids and not training data", {
  set.seed(708L)
  # example: train second (called "something-random-we-would-not-hardcode"), two valids,
  #          and a metric where higher values are better ("auc")
  trainDF <- data.frame(
    "feat1" = runif(n = 500L, min = 0.0, max = 15.0)
    , "target" = rep(c(0L, 1L), 500L)
  )
  validDF <- data.frame(
    "feat1" = runif(n = 50L, min = 0.0, max = 15.0)
    , "target" = rep(c(0L, 1L), 50L)
  )
  dtrain <- lgb.Dataset(
    data = as.matrix(trainDF[["feat1"]], drop = FALSE)
    , label = trainDF[["target"]]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid1 <- lgb.Dataset(
    data = as.matrix(validDF[1L:25L, "feat1"], drop = FALSE)
    , label = validDF[1L:25L, "target"]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dvalid2 <- lgb.Dataset(
    data = as.matrix(validDF[26L:50L, "feat1"], drop = FALSE)
    , label = validDF[26L:50L, "target"]
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  nrounds <- 10L
  bst <- lightgbm(
    data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid1
      , "something-random-we-would-not-hardcode" = dtrain
      , "valid2" = dvalid2
    )
    , params = list(
      objective = "binary"
      , metric = "auc"
      , learning_rate = 1.5
      , num_leaves = 5L
      , num_threads = .LGB_MAX_THREADS
    )
    , verbose = -7L
  )
  # when verbose <= 0 is passed to lightgbm(), 'valids' is passed through to lgb.train()
  # untouched. If you set verbose to > 0, the training data will still be first but called "train"
  expect_named(
    bst$record_evals
    , c("start_iter", "something-random-we-would-not-hardcode", "valid1", "valid2")
    , ignore.order = FALSE
    , ignore.case = FALSE
  )
  auc_scores <- unlist(bst$record_evals[["valid1"]][["auc"]][["eval"]])
  expect_length(auc_scores, nrounds)
  expect_identical(bst$best_iter, which.max(auc_scores))
  expect_identical(bst$best_score, auc_scores[which.max(auc_scores)])
})

test_that("lgb.cv() works when you specify both 'metric' and 'eval' with strings", {
  set.seed(708L)
  nrounds <- 10L
  nfolds <- 4L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.cv(
    params = list(
      objective = "binary"
      , metric = "binary_error"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_CLASSIFICATION
    , nrounds = nrounds
    , nfold = nfolds
    , eval = "binary_logloss"
  )

  # both metrics should have been used
  expect_named(
    bst$record_evals[["valid"]]
    , expected = c("binary_error", "binary_logloss")
    , ignore.order = TRUE
    , ignore.case = FALSE
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid"]]
  expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.5005654) < .LGB_NUMERIC_TOLERANCE)
  expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.7011232) < .LGB_NUMERIC_TOLERANCE)

  # all boosters should have been created
  expect_length(bst$boosters, nfolds)
})

test_that("lgb.cv() works when you give a function for eval", {
  set.seed(708L)
  nrounds <- 10L
  nfolds <- 3L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.cv(
    params = list(
      objective = "binary"
      , metric = "None"
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_CLASSIFICATION
    , nfold = nfolds
    , nrounds = nrounds
    , eval = .constant_metric
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid"]]
  expect_true(abs(results[["constant_metric"]][["eval"]][[1L]] - CONSTANT_METRIC_VALUE) < .LGB_NUMERIC_TOLERANCE)
  expect_named(results, "constant_metric")
})

test_that("If first_metric_only is TRUE, lgb.cv() decides to stop early based on only the first metric", {
  set.seed(708L)
  nrounds <- 10L
  nfolds <- 5L
  early_stopping_rounds <- 3L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.cv(
    params = list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
      , first_metric_only = TRUE
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_REGRESSION
    , nfold = nfolds
    , nrounds = nrounds
    , eval = list(
      .increasing_metric
      , .constant_metric
    )
  )

  # Only the two functions provided to "eval" should have been evaluated
  expect_named(bst$record_evals[["valid"]], c("increasing_metric", "constant_metric"))

  # all 10 iterations should happen, and the best_iter should be the final one
  expect_equal(bst$best_iter, nrounds)

  # best_score should be taken from "increasing_metric"
  #
  # this expected value looks magical and confusing, but it's because
  # evaluation metrics are averaged over all folds.
  #
  # consider 5-fold CV with a metric that adds 0.1 to a global accumulator
  # each time it's called
  #
  # * iter 1: [0.1, 0.2, 0.3, 0.4, 0.5] (mean = 0.3)
  # * iter 2: [0.6, 0.7, 0.8, 0.9, 1.0] (mean = 1.3)
  # * iter 3: [1.1, 1.2, 1.3, 1.4, 1.5] (mean = 1.8)
  #
  cv_value <- increasing_metric_starting_value + mean(seq_len(nfolds) / 10.0) + (nrounds  - 1L) * 0.1 * nfolds
  expect_equal(bst$best_score, cv_value)

  # early stopping should not have happened. Even though constant_metric
  # had 9 consecutive iterations with no improvement, it is ignored because of
  # first_metric_only = TRUE
  expect_equal(
    length(bst$record_evals[["valid"]][["constant_metric"]][["eval"]])
    , nrounds
  )
  expect_equal(
    length(bst$record_evals[["valid"]][["increasing_metric"]][["eval"]])
    , nrounds
  )
})

test_that("early stopping works with lgb.cv()", {
  set.seed(708L)
  nrounds <- 10L
  nfolds <- 5L
  early_stopping_rounds <- 3L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.cv(
    params = list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
      , first_metric_only = TRUE
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    , data = DTRAIN_RANDOM_REGRESSION
    , nfold = nfolds
    , nrounds = nrounds
    , eval = list(
      .constant_metric
      , .increasing_metric
    )
  )

  # only the two functions provided to "eval" should have been evaluated
  expect_named(bst$record_evals[["valid"]], c("constant_metric", "increasing_metric"))

  # best_iter should be based on the first metric. Since constant_metric
  # never changes, its first iteration was the best oone
  expect_equal(bst$best_iter, 1L)

  # best_score should be taken from the first metric
  expect_equal(bst$best_score, 0.2)

  # early stopping should have happened, since constant_metric was the first
  # one passed to eval and it will not improve over consecutive iterations
  #
  # note that this test is identical to the previous one, but with the
  # order of the eval metrics switched
  expect_equal(
    length(bst$record_evals[["valid"]][["constant_metric"]][["eval"]])
    , early_stopping_rounds + 1L
  )
  expect_equal(
    length(bst$record_evals[["valid"]][["increasing_metric"]][["eval"]])
    , early_stopping_rounds + 1L
  )

  # every booster's predict method should use best_iter as num_iteration in predict
  random_data <- as.matrix(rnorm(10L), ncol = 1L, drop = FALSE)
  for (x in bst$boosters) {
    expect_equal(x$booster$best_iter, bst$best_iter)
    expect_gt(x$booster$current_iter(), bst$best_iter)
    preds_iter <- predict(x$booster, random_data, num_iteration = bst$best_iter)
    preds_no_iter <- predict(x$booster, random_data)
    expect_equal(preds_iter, preds_no_iter)
  }
})

test_that("lgb.cv() respects changes to logging verbosity", {
  dtrain <- lgb.Dataset(
    data = train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  # (verbose = 1) should be INFO and WARNING level logs
  lgb_cv_logs <- capture.output({
    cv_bst <- lgb.cv(
      params = list(num_threads = .LGB_MAX_THREADS)
      , nfold = 2L
      , nrounds = 5L
      , data = dtrain
      , obj = "binary"
      , verbose = 1L
    )
  })
  expect_true(any(grepl("[LightGBM] [Info]", lgb_cv_logs, fixed = TRUE)))
  expect_true(any(grepl("[LightGBM] [Warning]", lgb_cv_logs, fixed = TRUE)))

  # (verbose = 0) should be WARNING level logs only
  lgb_cv_logs <- capture.output({
    cv_bst <- lgb.cv(
      params = list(num_threads = .LGB_MAX_THREADS)
      , nfold = 2L
      , nrounds = 5L
      , data = dtrain
      , obj = "binary"
      , verbose = 0L
    )
  })
  expect_false(any(grepl("[LightGBM] [Info]", lgb_cv_logs, fixed = TRUE)))
  expect_true(any(grepl("[LightGBM] [Warning]", lgb_cv_logs, fixed = TRUE)))

  # (verbose = -1) no logs
  lgb_cv_logs <- capture.output({
    cv_bst <- lgb.cv(
      params = list(num_threads = .LGB_MAX_THREADS)
      , nfold = 2L
      , nrounds = 5L
      , data = dtrain
      , obj = "binary"
      , verbose = -1L
    )
  })
  # NOTE: this is not length(lgb_cv_logs) == 0 because lightgbm's
  #       dependencies might print other messages
  expect_false(any(grepl("[LightGBM] [Info]", lgb_cv_logs, fixed = TRUE)))
  expect_false(any(grepl("[LightGBM] [Warning]", lgb_cv_logs, fixed = TRUE)))
})

test_that("lgb.cv() updates params based on keyword arguments", {
  dtrain <- lgb.Dataset(
    data = matrix(rnorm(400L), ncol =  4L)
    , label = rnorm(100L)
    , params = list(num_threads = .LGB_MAX_THREADS)
  )

  # defaults from keyword arguments should be used if not specified in params
  invisible(
    capture.output({
      cv_bst <- lgb.cv(
        data = dtrain
        , obj = "regression"
        , params = list(num_threads = .LGB_MAX_THREADS)
        , nfold = 2L
      )
    })
  )

  for (bst in cv_bst$boosters) {
    bst_params <- bst[["booster"]]$params
    expect_equal(bst_params[["verbosity"]], 1L)
    expect_equal(bst_params[["num_iterations"]], 100L)
  }

  # main param names should be preferred to keyword arguments
  invisible(
    capture.output({
      cv_bst <- lgb.cv(
        data = dtrain
        , obj = "regression"
        , params = list(
          "verbosity" = 5L
          , "num_iterations" = 2L
          , num_threads = .LGB_MAX_THREADS
        )
        , nfold = 2L
      )
    })
  )
  for (bst in cv_bst$boosters) {
    bst_params <- bst[["booster"]]$params
    expect_equal(bst_params[["verbosity"]], 5L)
    expect_equal(bst_params[["num_iterations"]], 2L)
  }

  # aliases should be preferred to keyword arguments, and converted to main parameter name
  invisible(
    capture.output({
      cv_bst <- lgb.cv(
        data = dtrain
        , obj = "regression"
        , params = list(
          "verbose" = 5L
          , "num_boost_round" = 2L
          , num_threads = .LGB_MAX_THREADS
        )
        , nfold = 2L
      )
    })
  )
  for (bst in cv_bst$boosters) {
    bst_params <- bst[["booster"]]$params
    expect_equal(bst_params[["verbosity"]], 5L)
    expect_false("verbose" %in% bst_params)
    expect_equal(bst_params[["num_iterations"]], 2L)
    expect_false("num_boost_round" %in% bst_params)
  }

})

test_that("lgb.train() fit on linearly-relatead data improves when using linear learners", {
  set.seed(708L)
  .new_dataset <- function() {
    X <- matrix(rnorm(100L), ncol = 1L)
    return(lgb.Dataset(
      data = X
      , label = 2L * X + runif(nrow(X), 0L, 0.1)
      , params = list(num_threads = .LGB_MAX_THREADS)
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = .LGB_VERBOSITY
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    , num_threads = .LGB_MAX_THREADS
  )

  dtrain <- .new_dataset()
  bst <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = params
    , valids = list("train" = dtrain)
  )
  expect_true(.is_Booster(bst))

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = utils::modifyList(params, list(linear_tree = TRUE))
    , valids = list("train" = dtrain)
  )
  expect_true(.is_Booster(bst_linear))

  bst_last_mse <- bst$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  bst_lin_last_mse <- bst_linear$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  expect_true(bst_lin_last_mse <  bst_last_mse)
})


test_that("lgb.train() with linear learner fails already-constructed dataset with linear=false", {
  set.seed(708L)
  params <- list(
    objective = "regression"
    , verbose = .LGB_VERBOSITY
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    , num_threads = .LGB_MAX_THREADS
  )

  dtrain <- lgb.Dataset(
    data = matrix(rnorm(100L), ncol = 1L)
    , label = rnorm(100L)
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  dtrain$construct()
  expect_error({
    capture.output({
      bst_linear <- lgb.train(
        data = dtrain
        , nrounds = 10L
        , params = utils::modifyList(params, list(linear_tree = TRUE))
      )
    }, type = "message")
  }, regexp = "Cannot change linear_tree after constructed Dataset handle")
})

test_that("lgb.train() works with linear learners even if Dataset has missing values", {
  set.seed(708L)
  .new_dataset <- function() {
    values <- rnorm(100L)
    values[sample(seq_along(values), size = 10L)] <- NA_real_
    X <- matrix(
      data = sample(values, size = 100L)
      , ncol = 1L
    )
    return(lgb.Dataset(
      data = X
      , label = 2L * X + runif(nrow(X), 0L, 0.1)
      , params = list(num_threads = .LGB_MAX_THREADS)
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = .LGB_VERBOSITY
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    , num_threads = .LGB_MAX_THREADS
  )

  dtrain <- .new_dataset()
  bst <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = params
    , valids = list("train" = dtrain)
  )
  expect_true(.is_Booster(bst))

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = utils::modifyList(params, list(linear_tree = TRUE))
    , valids = list("train" = dtrain)
  )
  expect_true(.is_Booster(bst_linear))

  bst_last_mse <- bst$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  bst_lin_last_mse <- bst_linear$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  expect_true(bst_lin_last_mse <  bst_last_mse)
})

test_that("lgb.train() works with linear learners, bagging, and a Dataset that has missing values", {
  set.seed(708L)
  .new_dataset <- function() {
    values <- rnorm(100L)
    values[sample(seq_along(values), size = 10L)] <- NA_real_
    X <- matrix(
      data = sample(values, size = 100L)
      , ncol = 1L
    )
    return(lgb.Dataset(
      data = X
      , label = 2L * X + runif(nrow(X), 0L, 0.1)
      , params = list(num_threads = .LGB_MAX_THREADS)
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = .LGB_VERBOSITY
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    , bagging_freq = 1L
    , subsample = 0.8
    , num_threads = .LGB_MAX_THREADS
  )

  dtrain <- .new_dataset()
  bst <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = params
    , valids = list("train" = dtrain)
  )
  expect_true(.is_Booster(bst))

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = utils::modifyList(params, list(linear_tree = TRUE))
    , valids = list("train" = dtrain)
  )
  expect_true(.is_Booster(bst_linear))

  bst_last_mse <- bst$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  bst_lin_last_mse <- bst_linear$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  expect_true(bst_lin_last_mse <  bst_last_mse)
})

test_that("lgb.train() works with linear learners and data where a feature has only 1 non-NA value", {
  set.seed(708L)
  .new_dataset <- function() {
    values <- c(rnorm(100L), rep(NA_real_, 100L))
    values[118L] <- rnorm(1L)
    X <- matrix(
      data = values
      , ncol = 2L
    )
    return(lgb.Dataset(
      data = X
      , label = 2L * X[, 1L] + runif(nrow(X), 0L, 0.1)
      , params = list(
        feature_pre_filter = FALSE
        , num_threads = .LGB_MAX_THREADS
      )
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    , num_threads = .LGB_MAX_THREADS
  )

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = utils::modifyList(params, list(linear_tree = TRUE))
  )
  expect_true(.is_Booster(bst_linear))
})

test_that("lgb.train() works with linear learners when Dataset has categorical features", {
  set.seed(708L)
  .new_dataset <- function() {
    X <- matrix(numeric(200L), nrow = 100L, ncol = 2L)
    X[, 1L] <- rnorm(100L)
    X[, 2L] <- sample(seq_len(4L), size = 100L, replace = TRUE)
    return(lgb.Dataset(
      data = X
      , label = 2L * X[, 1L] + runif(nrow(X), 0L, 0.1)
      , params = list(num_threads = .LGB_MAX_THREADS)
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    , categorical_feature = 1L
    , num_threads = .LGB_MAX_THREADS
  )

  dtrain <- .new_dataset()
  bst <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = params
    , valids = list("train" = dtrain)
  )
  expect_true(.is_Booster(bst))

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = utils::modifyList(params, list(linear_tree = TRUE))
    , valids = list("train" = dtrain)
  )
  expect_true(.is_Booster(bst_linear))

  bst_last_mse <- bst$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  bst_lin_last_mse <- bst_linear$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  expect_true(bst_lin_last_mse <  bst_last_mse)
})

test_that("lgb.train() throws an informative error if interaction_constraints is not a list", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", interaction_constraints = "[1,2],[3]")
    expect_error({
      bst <- lightgbm(
        data = dtrain
        , params = params
        , nrounds = 2L
      )
    }, "interaction_constraints must be a list")
})

test_that(paste0("lgb.train() throws an informative error if the members of interaction_constraints ",
                 "are not character or numeric vectors"), {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", interaction_constraints = list(list(1L, 2L), list(3L)))
    expect_error({
      bst <- lightgbm(
        data = dtrain
        , params = params
        , nrounds = 2L
      )
    }, "every element in interaction_constraints must be a character vector or numeric vector")
})

test_that("lgb.train() throws an informative error if interaction_constraints contains a too large index", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression",
                 interaction_constraints = list(c(1L, ncol(train$data) + 1L:2L), 3L))
    expect_error(
      lightgbm(data = dtrain, params = params, nrounds = 2L)
      , "unknown feature(s) in interaction_constraints: '127', '128'"
      , fixed = TRUE
    )
})

test_that(paste0("lgb.train() gives same result when interaction_constraints is specified as a list of ",
                 "character vectors, numeric vectors, or a combination"), {
  set.seed(1L)
  dtrain <- lgb.Dataset(train$data, label = train$label, params = list(num_threads = .LGB_MAX_THREADS))

  params <- list(
    objective = "regression"
    , interaction_constraints = list(c(1L, 2L), 3L)
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
  )
  pred1 <- bst$predict(test$data)

  cnames <- colnames(train$data)
  params <- list(
    objective = "regression"
    , interaction_constraints = list(c(cnames[[1L]], cnames[[2L]]), cnames[[3L]])
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
  )
  pred2 <- bst$predict(test$data)

  params <- list(
    objective = "regression"
    , interaction_constraints = list(c(cnames[[1L]], cnames[[2L]]), 3L)
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
  )
  pred3 <- bst$predict(test$data)

  expect_equal(pred1, pred2)
  expect_equal(pred2, pred3)

})

test_that(paste0("lgb.train() gives same results when using interaction_constraints and specifying colnames"), {
  set.seed(1L)
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )

  params <- list(
    objective = "regression"
    , interaction_constraints = list(c(1L, 2L), 3L)
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
  )
  pred1 <- bst$predict(test$data)

  new_colnames <- paste0(colnames(train$data), "_x")
  dtrain$set_colnames(new_colnames)
  params <- list(
    objective = "regression"
    , interaction_constraints = list(c(new_colnames[1L], new_colnames[2L]), new_colnames[3L])
    , verbose = .LGB_VERBOSITY
    , num_threads = .LGB_MAX_THREADS
  )
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
  )
  pred2 <- bst$predict(test$data)

  expect_equal(pred1, pred2)

})

test_that("Interaction constraints add missing features correctly as new group", {
  dtrain <- lgb.Dataset(
    train$data[, 1L:6L]  # Pick only some columns
    , label = train$label
    , params = list(num_threads = .LGB_MAX_THREADS)
  )

  list_of_constraints <- list(
    list(3L, 1L:2L)
    , list("cap-shape=convex", c("cap-shape=bell", "cap-shape=conical"))
  )

  for (constraints in list_of_constraints) {
    params <- list(
      objective = "regression"
      , interaction_constraints = constraints
      , verbose = .LGB_VERBOSITY
      , num_threads = .LGB_MAX_THREADS
    )
    bst <- lightgbm(data = dtrain, params = params, nrounds = 10L)

    expected_list <- list("[2]", "[0,1]", "[3,4,5]")
    expect_equal(bst$params$interaction_constraints, expected_list)

    expected_string <- "[interaction_constraints: [2],[0,1],[3,4,5]]"
    expect_true(
      grepl(expected_string, bst$save_model_to_string(), fixed = TRUE)
    )
  }
})

.generate_trainset_for_monotone_constraints_tests <- function(x3_to_categorical) {
  n_samples <- 3000L
  x1_positively_correlated_with_y <- runif(n = n_samples, min = 0.0, max = 1.0)
  x2_negatively_correlated_with_y <- runif(n = n_samples, min = 0.0, max = 1.0)
  x3_negatively_correlated_with_y <- runif(n = n_samples, min = 0.0, max = 1.0)
  if (x3_to_categorical) {
    x3_negatively_correlated_with_y <- as.integer(x3_negatively_correlated_with_y / 0.01)
    categorical_features <- "feature_3"
  } else {
    categorical_features <- NULL
  }
  X <- matrix(
    data = c(
        x1_positively_correlated_with_y
        , x2_negatively_correlated_with_y
        , x3_negatively_correlated_with_y
    )
    , ncol = 3L
  )
  zs <- rnorm(n = n_samples, mean = 0.0, sd = 0.01)
  scales <- 10.0 * (runif(n = 6L, min = 0.0, max = 1.0) + 0.5)
  y <- (
    scales[1L] * x1_positively_correlated_with_y
    + sin(scales[2L] * pi * x1_positively_correlated_with_y)
    - scales[3L] * x2_negatively_correlated_with_y
    - cos(scales[4L] * pi * x2_negatively_correlated_with_y)
    - scales[5L] * x3_negatively_correlated_with_y
    - cos(scales[6L] * pi * x3_negatively_correlated_with_y)
    + zs
  )
  return(lgb.Dataset(
    data = X
    , label = y
    , categorical_feature = categorical_features
    , free_raw_data = FALSE
    , colnames = c("feature_1", "feature_2", "feature_3")
    , params = list(num_threads = .LGB_MAX_THREADS)
  ))
}

.is_increasing <- function(y) {
  return(all(diff(y) >= 0.0))
}

.is_decreasing <- function(y) {
  return(all(diff(y) <= 0.0))
}

.is_non_monotone <- function(y) {
  return(any(diff(y) < 0.0) & any(diff(y) > 0.0))
}

# R equivalent of numpy.linspace()
.linspace <- function(start_val, stop_val, num) {
  weights <- (seq_len(num) - 1L) / (num - 1L)
  return(start_val + weights * (stop_val - start_val))
}

.is_correctly_constrained <- function(learner, x3_to_categorical) {
  iterations <- 10L
  n <- 1000L
  variable_x <- .linspace(0L, 1L, n)
  fixed_xs_values <- .linspace(0L, 1L, n)
  for (i in seq_len(iterations)) {
    fixed_x <- fixed_xs_values[i] * rep(1.0, n)
    monotonically_increasing_x <- matrix(
      data = c(variable_x, fixed_x, fixed_x)
      , ncol = 3L
    )
    monotonically_increasing_y <- predict(
      learner
      , monotonically_increasing_x
    )

    monotonically_decreasing_x <- matrix(
      data = c(fixed_x, variable_x, fixed_x)
      , ncol = 3L
    )
    monotonically_decreasing_y <- predict(
      learner
      , monotonically_decreasing_x
    )

    if (x3_to_categorical) {
      non_monotone_data <- c(
        fixed_x
        , fixed_x
        , as.integer(variable_x / 0.01)
      )
    } else {
      non_monotone_data <- c(fixed_x, fixed_x, variable_x)
    }
    non_monotone_x <- matrix(
      data = non_monotone_data
      , ncol = 3L
    )
    non_monotone_y <- predict(
      learner
      , non_monotone_x
    )
    if (!(.is_increasing(monotonically_increasing_y) &&
          .is_decreasing(monotonically_decreasing_y) &&
          .is_non_monotone(non_monotone_y)
    )) {
      return(FALSE)
    }
  }
  return(TRUE)
}

for (x3_to_categorical in c(TRUE, FALSE)) {
  set.seed(708L)
  dtrain <- .generate_trainset_for_monotone_constraints_tests(
    x3_to_categorical = x3_to_categorical
  )
  for (monotone_constraints_method in c("basic", "intermediate", "advanced")) {
    test_msg <- paste0(
      "lgb.train() supports monotone constraints ("
      , "categoricals="
      , x3_to_categorical
      , ", method="
      , monotone_constraints_method
      , ")"
    )
    test_that(test_msg, {
      params <- list(
        min_data = 20L
        , num_leaves = 20L
        , monotone_constraints = c(1L, -1L, 0L)
        , monotone_constraints_method = monotone_constraints_method
        , use_missing = FALSE
        , verbose = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
      )
      constrained_model <- lgb.train(
        params = params
        , data = dtrain
        , obj = "regression_l2"
        , nrounds = 100L
      )
      expect_true({
        .is_correctly_constrained(
          learner = constrained_model
          , x3_to_categorical = x3_to_categorical
        )
      })
    })
  }
}

test_that("lightgbm() accepts objective as function argument and under params", {
  bst1 <- lightgbm(
    data = train$data
    , label = train$label
    , params = list(objective = "regression_l1", num_threads = .LGB_MAX_THREADS)
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
  )
  expect_equal(bst1$params$objective, "regression_l1")
  model_txt_lines <- strsplit(
    x = bst1$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(model_txt_lines == "objective=regression_l1"))
  expect_false(any(model_txt_lines == "objective=regression_l2"))

  bst2 <- lightgbm(
    data = train$data
    , label = train$label
    , objective = "regression_l1"
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
  )
  expect_equal(bst2$params$objective, "regression_l1")
  model_txt_lines <- strsplit(
    x = bst2$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(model_txt_lines == "objective=regression_l1"))
  expect_false(any(model_txt_lines == "objective=regression_l2"))
})

test_that("lightgbm() prioritizes objective under params over objective as function argument", {
  bst1 <- lightgbm(
    data = train$data
    , label = train$label
    , objective = "regression"
    , params = list(objective = "regression_l1", num_threads = .LGB_MAX_THREADS)
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
  )
  expect_equal(bst1$params$objective, "regression_l1")
  model_txt_lines <- strsplit(
    x = bst1$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(model_txt_lines == "objective=regression_l1"))
  expect_false(any(model_txt_lines == "objective=regression_l2"))

  bst2 <- lightgbm(
    data = train$data
    , label = train$label
    , objective = "regression"
    , params = list(loss = "regression_l1", num_threads = .LGB_MAX_THREADS)
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
  )
  expect_equal(bst2$params$objective, "regression_l1")
  model_txt_lines <- strsplit(
    x = bst2$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(model_txt_lines == "objective=regression_l1"))
  expect_false(any(model_txt_lines == "objective=regression_l2"))
})

test_that("lightgbm() accepts init_score as function argument", {
  bst1 <- lightgbm(
    data = train$data
    , label = train$label
    , objective = "binary"
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  pred1 <- predict(bst1, train$data, type = "raw")

  bst2 <- lightgbm(
    data = train$data
    , label = train$label
    , init_score = pred1
    , objective = "binary"
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  pred2 <- predict(bst2, train$data, type = "raw")

  expect_true(any(pred1 != pred2))
})

test_that("lightgbm() defaults to 'regression' objective if objective not otherwise provided", {
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
    , params = list(num_threads = .LGB_MAX_THREADS)
  )
  expect_equal(bst$params$objective, "regression")
  model_txt_lines <- strsplit(
    x = bst$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(model_txt_lines == "objective=regression"))
  expect_false(any(model_txt_lines == "objective=regression_l1"))
})

test_that("lightgbm() accepts 'num_threads' as either top-level argument or under params", {
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
    , num_threads = 1L
  )
  expect_equal(bst$params$num_threads, 1L)
  model_txt_lines <- strsplit(
    x = bst$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(grepl("[num_threads: 1]", model_txt_lines, fixed = TRUE)))

  bst <- lightgbm(
    data = train$data
    , label = train$label
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
    , params = list(num_threads = 1L)
  )
  expect_equal(bst$params$num_threads, 1L)
  model_txt_lines <- strsplit(
    x = bst$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(grepl("[num_threads: 1]", model_txt_lines, fixed = TRUE)))

  bst <- lightgbm(
    data = train$data
    , label = train$label
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
    , num_threads = 10L
    , params = list(num_threads = 1L)
  )
  expect_equal(bst$params$num_threads, 1L)
  model_txt_lines <- strsplit(
    x = bst$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(grepl("[num_threads: 1]", model_txt_lines, fixed = TRUE)))
})

test_that("lightgbm() accepts 'weight' and 'weights'", {
  data(mtcars)
  X <- as.matrix(mtcars[, -1L])
  y <- as.numeric(mtcars[, 1L])
  w <- rep(1.0, nrow(X))
  model <- lightgbm(
    X
    , y
    , weights = w
    , obj = "regression"
    , nrounds = 5L
    , verbose = .LGB_VERBOSITY
    , params = list(
      min_data_in_bin = 1L
      , min_data_in_leaf = 1L
      , num_threads = .LGB_MAX_THREADS
    )
  )
  expect_equal(model$.__enclos_env__$private$train_set$get_field("weight"), w)

  # Avoid a bad CRAN check due to partial argument matches
  lgb_args <- list(
    X
    , y
    , weight = w
    , obj = "regression"
    , nrounds = 5L
    , verbose = -1L
  )
  model <- do.call(lightgbm, lgb_args)
  expect_equal(model$.__enclos_env__$private$train_set$get_field("weight"), w)
})

.assert_has_expected_logs <- function(log_txt, lgb_info, lgb_warn, early_stopping, valid_eval_msg) {
  expect_identical(
    object = any(grepl("[LightGBM] [Info]", log_txt, fixed = TRUE))
    , expected = lgb_info
  )
  expect_identical(
    object = any(grepl("[LightGBM] [Warning]", log_txt, fixed = TRUE))
    , expected = lgb_warn
  )
  expect_identical(
    object = any(grepl("Will train until there is no improvement in 5 rounds", log_txt, fixed = TRUE))
    , expected = early_stopping
  )
  expect_identical(
    object = any(grepl("Did not meet early stopping", log_txt, fixed = TRUE))
    , expected = early_stopping
  )
  expect_identical(
    object = any(grepl("valid's auc\\:[0-9]+", log_txt))
    , expected = valid_eval_msg
  )
}

.assert_has_expected_record_evals <- function(fitted_model) {
  record_evals <- fitted_model$record_evals
  expect_equal(record_evals$start_iter, 1L)
  if (inherits(fitted_model, "lgb.CVBooster")) {
    expected_valid_auc <- c(0.979056, 0.9844697, 0.9900813, 0.9908026, 0.9935588)
  } else {
    expected_valid_auc <-  c(0.9805752, 0.9805752, 0.9934957, 0.9934957, 0.9949372)
  }
  expect_equal(
    object = unlist(record_evals[["valid"]][["auc"]][["eval"]])
    , expected = expected_valid_auc
    , tolerance = .LGB_NUMERIC_TOLERANCE
  )
   expect_named(record_evals, c("start_iter", "valid"), ignore.order = TRUE, ignore.case = FALSE)
  expect_equal(record_evals[["valid"]][["auc"]][["eval_err"]], list())
}

.train_for_verbosity_test <- function(train_function, verbose_kwarg, verbose_param) {
  set.seed(708L)
  nrounds <- 5L
  params <- list(
    num_leaves = 5L
    , objective = "binary"
    , metric =  "auc"
    , early_stopping_round = nrounds
    , num_threads = .LGB_MAX_THREADS
    # include a nonsense parameter just to trigger a WARN-level log
    , nonsense_param = 1.0
  )
  if (!is.null(verbose_param)) {
    params[["verbose"]] <- verbose_param
  }
  train_kwargs <- list(
    params = params
    , nrounds = nrounds
  )
  if (!is.null(verbose_kwarg)) {
    train_kwargs[["verbose"]] <- verbose_kwarg
  }
  function_name <- deparse(substitute(train_function))
  if (function_name == "lgb.train") {
    train_kwargs[["data"]] <- lgb.Dataset(
      data = train$data
      , label = train$label
      , params = list(num_threads = .LGB_MAX_THREADS)
    )
    train_kwargs[["valids"]] <- list(
      "valid" = lgb.Dataset(data = test$data, label = test$label)
    )
  } else if (function_name == "lightgbm") {
    train_kwargs[["data"]] <- train$data
    train_kwargs[["label"]] <- train$label
    train_kwargs[["valids"]] <- list(
      "valid" = lgb.Dataset(data = test$data, label = test$label)
    )
  } else if (function_name == "lgb.cv") {
    train_kwargs[["data"]] <- lgb.Dataset(
      data = train$data
      , label = train$label
    )
    train_kwargs[["nfold"]] <- 3L
    train_kwargs[["showsd"]] <- FALSE
  }
  log_txt <- capture.output({
    bst <- do.call(
      what = train_function
      , args = train_kwargs
    )
  })
  return(list(booster = bst, logs = log_txt))
}

test_that("lgb.train() only prints eval metrics when expected to", {

  # regardless of value passed to keyword argument 'verbose', value in params
  # should take precedence
  for (verbose_keyword_arg in c(-5L, -1L, 0L, 1L, 5L)) {

    # (verbose = -1) should not be any logs, should be record evals
    out <- .train_for_verbosity_test(
      train_function = lgb.train
      , verbose_kwarg = verbose_keyword_arg
      , verbose_param = -1L
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = FALSE
      , lgb_warn = FALSE
      , early_stopping = FALSE
      , valid_eval_msg = FALSE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )

    # (verbose = 0) should be only WARN-level LightGBM logs
    out <- .train_for_verbosity_test(
      train_function = lgb.train
      , verbose_kwarg = verbose_keyword_arg
      , verbose_param = 0L
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = FALSE
      , lgb_warn = TRUE
      , early_stopping = FALSE
      , valid_eval_msg = FALSE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )

    # (verbose > 0) should be INFO- and WARN-level LightGBM logs, and record eval messages
    out <- .train_for_verbosity_test(
      train_function = lgb.train
      , verbose_kwarg = verbose_keyword_arg
      , verbose_param = 1L
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = TRUE
      , lgb_warn = TRUE
      , early_stopping = TRUE
      , valid_eval_msg = TRUE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )
  }

  # if verbosity isn't specified in `params`, changing keyword argument `verbose` should
  # alter what messages are printed

  # (verbose = -1) should not be any logs, should be record evals
  out <- .train_for_verbosity_test(
    train_function = lgb.train
    , verbose_kwarg = -1L
    , verbose_param = NULL
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = FALSE
    , lgb_warn = FALSE
    , early_stopping = FALSE
    , valid_eval_msg = FALSE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )

  # (verbose = 0) should be only WARN-level LightGBM logs
  out <- .train_for_verbosity_test(
    train_function = lgb.train
    , verbose_kwarg = 0L
    , verbose_param = NULL
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = FALSE
    , lgb_warn = TRUE
    , early_stopping = FALSE
    , valid_eval_msg = FALSE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )

  # (verbose > 0) should be INFO- and WARN-level LightGBM logs, and record eval messages
  out <- .train_for_verbosity_test(
    train_function = lgb.train
    , verbose_kwarg = 1L
    , verbose_param = NULL
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = TRUE
    , lgb_warn = TRUE
    , early_stopping = TRUE
    , valid_eval_msg = TRUE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )
})

test_that("lightgbm() only prints eval metrics when expected to", {

  # regardless of value passed to keyword argument 'verbose', value in params
  # should take precedence
  for (verbose_keyword_arg in c(-5L, -1L, 0L, 1L, 5L)) {

    # (verbose = -1) should not be any logs, train should not be in valids
    out <- .train_for_verbosity_test(
      train_function = lightgbm
      , verbose_kwarg = verbose_keyword_arg
      , verbose_param = -1L
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = FALSE
      , lgb_warn = FALSE
      , early_stopping = FALSE
      , valid_eval_msg = FALSE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )

    # (verbose = 0) should be only WARN-level LightGBM logs, train should not be in valids
    out <- .train_for_verbosity_test(
      train_function = lightgbm
      , verbose_kwarg = verbose_keyword_arg
      , verbose_param = 0L
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = FALSE
      , lgb_warn = TRUE
      , early_stopping = FALSE
      , valid_eval_msg = FALSE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )

    # (verbose > 0) should be INFO- and WARN-level LightGBM logs, and record eval messages, and
    #               train should be in valids
    out <- .train_for_verbosity_test(
      train_function = lightgbm
      , verbose_kwarg = verbose_keyword_arg
      , verbose_param = 1L
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = TRUE
      , lgb_warn = TRUE
      , early_stopping = TRUE
      , valid_eval_msg = TRUE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )
  }

  # if verbosity isn't specified in `params`, changing keyword argument `verbose` should
  # alter what messages are printed

  # (verbose = -1) should not be any logs, train should not be in valids
  out <- .train_for_verbosity_test(
    train_function = lightgbm
    , verbose_kwarg = -1L
    , verbose_param = NULL
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = FALSE
    , lgb_warn = FALSE
    , early_stopping = FALSE
    , valid_eval_msg = FALSE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )

  # (verbose = 0) should be only WARN-level LightGBM logs, train should not be in valids
  out <- .train_for_verbosity_test(
    train_function = lightgbm
    , verbose_kwarg = 0L
    , verbose_param = NULL
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = FALSE
    , lgb_warn = TRUE
    , early_stopping = FALSE
    , valid_eval_msg = FALSE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )

  # (verbose > 0) should be INFO- and WARN-level LightGBM logs, and record eval messages, and
  #               train should be in valids
  out <- .train_for_verbosity_test(
    train_function = lightgbm
    , verbose_kwarg = 1L
    , verbose_param = NULL
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = TRUE
    , lgb_warn = TRUE
    , early_stopping = TRUE
    , valid_eval_msg = TRUE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )
})

test_that("lgb.cv() only prints eval metrics when expected to", {

  # regardless of value passed to keyword argument 'verbose', value in params
  # should take precedence
  for (verbose_keyword_arg in c(-5L, -1L, 0L, 1L, 5L)) {

    # (verbose = -1) should not be any logs, should be record evals
    out <- .train_for_verbosity_test(
      verbose_kwarg = verbose_keyword_arg
      , verbose_param = -1L
      , train_function = lgb.cv
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = FALSE
      , lgb_warn = FALSE
      , early_stopping = FALSE
      , valid_eval_msg = FALSE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )

    # (verbose = 0) should be only WARN-level LightGBM logs
    out <- .train_for_verbosity_test(
      verbose_kwarg = verbose_keyword_arg
      , verbose_param = 0L
      , train_function = lgb.cv
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = FALSE
      , lgb_warn = TRUE
      , early_stopping = FALSE
      , valid_eval_msg = FALSE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )

    # (verbose > 0) should be INFO- and WARN-level LightGBM logs, and record eval messages
    out <- .train_for_verbosity_test(
      verbose_kwarg = verbose_keyword_arg
      , verbose_param = 1L
      , train_function = lgb.cv
    )
    .assert_has_expected_logs(
      log_txt = out[["logs"]]
      , lgb_info = TRUE
      , lgb_warn = TRUE
      , early_stopping = TRUE
      , valid_eval_msg = TRUE
    )
    .assert_has_expected_record_evals(
      fitted_model = out[["booster"]]
    )
  }

  # if verbosity isn't specified in `params`, changing keyword argument `verbose` should
  # alter what messages are printed

  # (verbose = -1) should not be any logs, should be record evals
  out <- .train_for_verbosity_test(
    verbose_kwarg = verbose_keyword_arg
    , verbose_param = -1L
    , train_function = lgb.cv
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = FALSE
    , lgb_warn = FALSE
    , early_stopping = FALSE
    , valid_eval_msg = FALSE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )

  # (verbose = 0) should be only WARN-level LightGBM logs
  out <- .train_for_verbosity_test(
    verbose_kwarg = verbose_keyword_arg
    , verbose_param = 0L
    , train_function = lgb.cv
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = FALSE
    , lgb_warn = TRUE
    , early_stopping = FALSE
    , valid_eval_msg = FALSE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )

  # (verbose > 0) should be INFO- and WARN-level LightGBM logs, and record eval messages
  out <- .train_for_verbosity_test(
    verbose_kwarg = verbose_keyword_arg
    , verbose_param = 1L
    , train_function = lgb.cv
  )
  .assert_has_expected_logs(
    log_txt = out[["logs"]]
    , lgb_info = TRUE
    , lgb_warn = TRUE
    , early_stopping = TRUE
    , valid_eval_msg = TRUE
  )
  .assert_has_expected_record_evals(
    fitted_model = out[["booster"]]
  )
})

test_that("lightgbm() changes objective='auto' appropriately", {
  # Regression
  data("mtcars")
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1L])
  model <- lightgbm(x, y, objective = "auto", verbose = .LGB_VERBOSITY, nrounds = 5L, num_threads = .LGB_MAX_THREADS)
  expect_equal(model$params$objective, "regression")
  model_txt_lines <- strsplit(
    x = model$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(grepl("objective=regression", model_txt_lines, fixed = TRUE)))
  expect_false(any(grepl("objective=regression_l1", model_txt_lines, fixed = TRUE)))

  # Binary classification
  x <- train$data
  y <- factor(train$label)
  model <- lightgbm(x, y, objective = "auto", verbose = .LGB_VERBOSITY, nrounds = 5L, num_threads = .LGB_MAX_THREADS)
  expect_equal(model$params$objective, "binary")
  model_txt_lines <- strsplit(
    x = model$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(grepl("objective=binary", model_txt_lines, fixed = TRUE)))

  # Multi-class classification
  data("iris")
  y <- factor(iris$Species)
  x <- as.matrix(iris[, -5L])
  model <- lightgbm(x, y, objective = "auto", verbose = .LGB_VERBOSITY, nrounds = 5L, num_threads = .LGB_MAX_THREADS)
  expect_equal(model$params$objective, "multiclass")
  expect_equal(model$params$num_class, 3L)
  model_txt_lines <- strsplit(
    x = model$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(grepl("objective=multiclass", model_txt_lines, fixed = TRUE)))
})

test_that("lightgbm() determines number of classes for non-default multiclass objectives", {
  data("iris")
  y <- factor(iris$Species)
  x <- as.matrix(iris[, -5L])
  model <- lightgbm(
    x
    , y
    , objective = "multiclassova"
    , verbose = .LGB_VERBOSITY
    , nrounds = 5L
    , num_threads = .LGB_MAX_THREADS
  )
  expect_equal(model$params$objective, "multiclassova")
  expect_equal(model$params$num_class, 3L)
  model_txt_lines <- strsplit(
    x = model$save_model_to_string()
    , split = "\n"
    , fixed = TRUE
  )[[1L]]
  expect_true(any(grepl("objective=multiclassova", model_txt_lines, fixed = TRUE)))
})

test_that("lightgbm() doesn't accept binary classification with non-binary factors", {
  data("iris")
  y <- factor(iris$Species)
  x <- as.matrix(iris[, -5L])
  expect_error({
    lightgbm(x, y, objective = "binary", verbose = .LGB_VERBOSITY, nrounds = 5L, num_threads = .LGB_MAX_THREADS)
  }, regexp = "Factors with >2 levels as labels only allowed for multi-class objectives")
})

test_that("lightgbm() doesn't accept multi-class classification with binary factors", {
  data("iris")
  y <- as.character(iris$Species)
  y[y == "setosa"] <- "versicolor"
  y <- factor(y)
  x <- as.matrix(iris[, -5L])
  expect_error({
    lightgbm(x, y, objective = "multiclass", verbose = .LGB_VERBOSITY, nrounds = 5L, num_threads = .LGB_MAX_THREADS)
  }, regexp = "Two-level factors as labels only allowed for objective='binary'")
})

test_that("lightgbm() model predictions retain factor levels for multiclass classification", {
  data("iris")
  y <- factor(iris$Species)
  x <- as.matrix(iris[, -5L])
  model <- lightgbm(x, y, objective = "auto", verbose = .LGB_VERBOSITY, nrounds = 5L, num_threads = .LGB_MAX_THREADS)

  pred <- predict(model, x, type = "class")
  expect_true(is.factor(pred))
  expect_equal(levels(pred), levels(y))

  pred <- predict(model, x, type = "response")
  expect_equal(colnames(pred), levels(y))

  pred <- predict(model, x, type = "raw")
  expect_equal(colnames(pred), levels(y))
})

test_that("lightgbm() model predictions retain factor levels for binary classification", {
  data("iris")
  y <- as.character(iris$Species)
  y[y == "setosa"] <- "versicolor"
  y <- factor(y)
  x <- as.matrix(iris[, -5L])
  model <- lightgbm(x, y, objective = "auto", verbose = .LGB_VERBOSITY, nrounds = 5L, num_threads = .LGB_MAX_THREADS)

  pred <- predict(model, x, type = "class")
  expect_true(is.factor(pred))
  expect_equal(levels(pred), levels(y))

  pred <- predict(model, x, type = "response")
  expect_true(is.vector(pred))
  expect_true(is.numeric(pred))
  expect_false(any(pred %in% y))

  pred <- predict(model, x, type = "raw")
  expect_true(is.vector(pred))
  expect_true(is.numeric(pred))
  expect_false(any(pred %in% y))
})

test_that("lightgbm() accepts named categorical_features", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1L])
  model <- lightgbm(
    x
    , y
    , categorical_feature = "cyl"
    , verbose = .LGB_VERBOSITY
    , nrounds = 5L
    , num_threads = .LGB_MAX_THREADS
  )
  expect_true(length(model$params$categorical_feature) > 0L)
})

test_that("lightgbm() correctly sets objective when passing lgb.Dataset as input", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1L])
  ds <- lgb.Dataset(x, label = y)
  model <- lightgbm(
    ds
    , objective = "auto"
    , verbose = .LGB_VERBOSITY
    , nrounds = 5L
    , num_threads = .LGB_MAX_THREADS
  )
  expect_equal(model$params$objective, "regression")
})

test_that("Evaluation metrics aren't printed as a single-element vector", {
  log_txt <- capture_output({
    data(mtcars)
    y <- mtcars$mpg
    x <- as.matrix(mtcars[, -1L])
    cv_result <- lgb.cv(
        data = lgb.Dataset(x, label = y)
        , params = list(
            objective = "regression"
            , metric = "l2"
            , min_data_in_leaf = 5L
            , max_depth = 3L
            , num_threads = .LGB_MAX_THREADS
        )
        , nrounds = 2L
        , nfold = 3L
        , verbose = 1L
        , eval_train_metric = TRUE
    )
  })
  expect_false(grepl("[1] \"[1]", log_txt, fixed = TRUE))
})
