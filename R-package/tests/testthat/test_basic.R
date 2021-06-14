context("lightgbm()")

ON_WINDOWS <- .Platform$OS.type == "windows"

UTF8_LOCALE <- all(grepl(
  pattern = "UTF-8$"
  , x = Sys.getlocale(category = "LC_CTYPE")
))

data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

TOLERANCE <- 1e-6
set.seed(708L)

# [description] Every time this function is called, it adds 0.1
#               to an accumulator then returns the current value.
#               This is used to mock the situation where an evaluation
#               metric increases every iteration
ACCUMULATOR_NAME <- "INCREASING_METRIC_ACUMULATOR"
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
)
DVALID_RANDOM_REGRESSION <- lgb.Dataset(
  data = as.matrix(rnorm(50L), ncol = 1L, drop = FALSE)
  , label = rnorm(50L)
)
DTRAIN_RANDOM_CLASSIFICATION <- lgb.Dataset(
  data = as.matrix(rnorm(120L), ncol = 1L, drop = FALSE)
  , label = sample(c(0L, 1L), size = 120L, replace = TRUE)
)
DVALID_RANDOM_CLASSIFICATION <- lgb.Dataset(
  data = as.matrix(rnorm(37L), ncol = 1L, drop = FALSE)
  , label = sample(c(0L, 1L), size = 37L, replace = TRUE)
)

test_that("train and predict binary classification", {
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 5L
    , nrounds = nrounds
    , objective = "binary"
    , metric = "binary_error"
    , save_name = tempfile(fileext = ".model")
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
  expect_lt(abs(err_pred1 - err_log), TOLERANCE)
})


test_that("train and predict softmax", {
  set.seed(708L)
  lb <- as.numeric(iris$Species) - 1L

  bst <- lightgbm(
    data = as.matrix(iris[, -5L])
    , label = lb
    , num_leaves = 4L
    , learning_rate = 0.05
    , nrounds = 20L
    , min_data = 20L
    , min_hessian = 10.0
    , objective = "multiclass"
    , metric = "multi_error"
    , num_class = 3L
    , save_name = tempfile(fileext = ".model")
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
    , num_leaves = 4L
    , learning_rate = 1.0
    , nrounds = 10L
    , objective = "binary"
    , metric = metrics
    , save_name = tempfile(fileext = ".model")
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
    , num_leaves = 5L
    , nrounds = nrounds
    , objective = "binary"
    , metric = "binary_error"
    , save_name = tempfile(fileext = ".model")
  )
  expect_true(abs(bst$lower_bound() - -1.590853) < TOLERANCE)
  expect_true(abs(bst$upper_bound() - 1.871015) <  TOLERANCE)
})

test_that("lgb.Booster.upper_bound() and lgb.Booster.lower_bound() work as expected for regression", {
  set.seed(708L)
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 5L
    , nrounds = nrounds
    , objective = "regression"
    , metric = "l2"
    , save_name = tempfile(fileext = ".model")
  )
  expect_true(abs(bst$lower_bound() - 0.1513859) < TOLERANCE)
  expect_true(abs(bst$upper_bound() - 0.9080349) < TOLERANCE)
})

test_that("lightgbm() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1")
  for (nround_value in c(-10L, 0L)) {
    expect_error({
      bst <- lightgbm(
        data = dtrain
        , params = params
        , nrounds = nround_value
        , save_name = tempfile(fileext = ".model")
      )
    }, "nrounds should be greater than zero")
  }
})

test_that("lightgbm() performs evaluation on validation sets if they are provided", {
  set.seed(708L)
  dvalid1 <- lgb.Dataset(
    data = train$data
    , label = train$label
  )
  dvalid2 <- lgb.Dataset(
    data = train$data
    , label = train$label
  )
  nrounds <- 10L
  bst <- lightgbm(
    data = train$data
    , label = train$label
    , num_leaves = 5L
    , nrounds = nrounds
    , objective = "binary"
    , metric = c(
      "binary_error"
      , "auc"
    )
    , valids = list(
      "valid1" = dvalid1
      , "valid2" = dvalid2
    )
    , save_name = tempfile(fileext = ".model")
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
  expect_true(abs(bst$record_evals[["train"]][["binary_error"]][["eval"]][[1L]] - 0.02226317) < TOLERANCE)
  expect_true(abs(bst$record_evals[["valid1"]][["binary_error"]][["eval"]][[1L]] - 0.02226317) < TOLERANCE)
  expect_true(abs(bst$record_evals[["valid2"]][["binary_error"]][["eval"]][[1L]] - 0.02226317) < TOLERANCE)
})


context("training continuation")

test_that("training continuation works", {
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
    , free_raw_data = FALSE
  )
  watchlist <- list(train = dtrain)
  param <- list(
    objective = "binary"
    , metric = "binary_logloss"
    , num_leaves = 5L
    , learning_rate = 1.0
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

context("lgb.cv()")

test_that("cv works", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1")
  bst <- lgb.cv(
    params
    , dtrain
    , 10L
    , nfold = 5L
    , min_data = 1L
    , learning_rate = 1.0
    , early_stopping_rounds = 10L
  )
  expect_false(is.null(bst$record_evals))
})

test_that("lgb.cv() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1")
  for (nround_value in c(-10L, 0L)) {
    expect_error({
      bst <- lgb.cv(
        params
        , dtrain
        , nround_value
        , nfold = 5L
        , min_data = 1L
      )
    }, "nrounds should be greater than zero")
  }
})

test_that("lgb.cv() throws an informative error is 'data' is not an lgb.Dataset and labels are not given", {
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
        params = list(objective = "regression", metric = "l2,l1")
        , data = val
        , 10L
        , nfold = 5L
        , min_data = 1L
      )
    }, regexp = "'label' must be provided for lgb.cv if 'data' is not an 'lgb.Dataset'", fixed = TRUE)
  }
})

test_that("lightgbm.cv() gives the correct best_score and best_iter for a metric where higher values are better", {
  set.seed(708L)
  dtrain <- lgb.Dataset(
    data = as.matrix(runif(n = 500L, min = 0.0, max = 15.0), drop = FALSE)
    , label = rep(c(0L, 1L), 250L)
  )
  nrounds <- 10L
  cv_bst <- lgb.cv(
    data = dtrain
    , nfold = 5L
    , nrounds = nrounds
    , num_leaves = 5L
    , params = list(
      objective = "binary"
      , metric = "auc,binary_error"
      , learning_rate = 1.5
    )
  )
  expect_is(cv_bst, "lgb.CVBooster")
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
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
  )

  dtrain <- .new_dataset()
  cv_bst <- lgb.cv(
    data = dtrain
    , nrounds = 10L
    , params = params
    , nfold = 5L
  )
  expect_is(cv_bst, "lgb.CVBooster")

  dtrain <- .new_dataset()
  cv_bst_linear <- lgb.cv(
    data = dtrain
    , nrounds = 10L
    , params = modifyList(params, list(linear_tree = TRUE))
    , nfold = 5L
  )
  expect_is(cv_bst_linear, "lgb.CVBooster")

  expect_true(cv_bst_linear$best_score < cv_bst$best_score)
})

test_that("lgb.cv() respects showsd argument", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2")
  nrounds <- 5L
  set.seed(708L)
  bst_showsd <- lgb.cv(
    params = params
    , data = dtrain
    , nrounds = nrounds
    , nfold = 3L
    , min_data = 1L
    , showsd = TRUE
  )
  evals_showsd <- bst_showsd$record_evals[["valid"]][["l2"]]
  set.seed(708L)
  bst_no_showsd <- lgb.cv(
    params = params
    , data = dtrain
    , nrounds = nrounds
    , nfold = 3L
    , min_data = 1L
    , showsd = FALSE
  )
  evals_no_showsd <- bst_no_showsd$record_evals[["valid"]][["l2"]]
  expect_equal(
    evals_showsd[["eval"]]
    , evals_no_showsd[["eval"]]
  )
  expect_is(evals_showsd[["eval_err"]], "list")
  expect_equal(length(evals_showsd[["eval_err"]]), nrounds)
  expect_identical(evals_no_showsd[["eval_err"]], list())
})

context("lgb.train()")

test_that("lgb.train() works as expected with multiple eval metrics", {
  metrics <- c("binary_error", "auc", "binary_logloss")
  bst <- lgb.train(
    data = lgb.Dataset(
      train$data
      , label = train$label
    )
    , learning_rate = 1.0
    , nrounds = 10L
    , params = list(
      objective = "binary"
      , metric = metrics
    )
    , valids = list(
      "train" = lgb.Dataset(
        train$data
        , label = train$label
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

test_that("lgb.train() rejects negative or 0 value passed to nrounds", {
  dtrain <- lgb.Dataset(train$data, label = train$label)
  params <- list(objective = "regression", metric = "l2,l1")
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
        params = list(objective = "regression", metric = "l2,l1")
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
      params = list(objective = "regression", metric = "l2,l1")
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
      params = list(objective = "regression", metric = "l2,l1")
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
      params = list(objective = "regression", metric = "l2,l1")
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
  )
  params <- list(
    objective = "binary"
    , metric = "binary_error"
    , force_col_wise = TRUE
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
  )
  nrounds <- 1L
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , min_data = 1L
      , min_data_in_bin = 1L
    )
    , data = dtrain
    , nrounds = nrounds
  )

  expect_true(lgb.is.Booster(bst))
  expect_equal(bst$current_iter(), nrounds)
  parsed_model <- jsonlite::fromJSON(bst$dump_model())
  expect_equal(parsed_model$objective, "binary sigmoid:1")
  expect_false(parsed_model$average_output)
  expected_error <- 0.6931268
  expect_true(abs(bst$eval_train()[[1L]][["value"]] - expected_error) < TOLERANCE)
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
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
  )
  nrounds <- 10L

  ################################
  # train with no early stopping #
  ################################
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
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
  bst  <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
      , early_stopping_rounds = early_stopping_rounds
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
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
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
  )
  dvalid <- lgb.Dataset(
    data = test$data
    , label = test$label
  )
  nrounds <- 10L

  #############################
  # train with early stopping #
  #############################
  early_stopping_rounds <- 5L
  # the harsh max_depth guarantees that AUC improves over at least the first few iterations
  bst_auc  <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "auc"
      , max_depth = 3L
      , early_stopping_rounds = early_stopping_rounds
    )
    , data = dtrain
    , nrounds = nrounds
    , valids = list(
      "valid1" = dvalid
    )
  )
  bst_binary_error  <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "binary_error"
      , max_depth = 3L
      , early_stopping_rounds = early_stopping_rounds
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
  expect_true(abs(bst_binary_error$best_score - 0.01613904) < TOLERANCE)

  # early stopping should not have been hit for AUC (higher_better = TRUE)
  eval_info <- bst_auc$.__enclos_env__$private$get_eval_info()
  expect_identical(eval_info, "auc")
  expect_identical(
    unname(bst_auc$.__enclos_env__$private$higher_better_inner_eval)
    , TRUE
  )
  expect_identical(bst_auc$best_iter, 9L)
  expect_identical(bst_auc$current_iter(), nrounds)
  expect_true(abs(bst_auc$best_score - 0.9999969) < TOLERANCE)
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
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
  )
  nrounds <- 10L

  ################################
  # train with no early stopping #
  ################################
  bst <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "rmse"
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
  bst  <- lgb.train(
    params = list(
      objective = "regression"
      , metric = "rmse"
      , early_stopping_rounds = early_stopping_rounds
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
    )
    , list(
      objective = "regression"
      , metric = "None"
      , early_stopping_rounds = early_stopping_rounds
      , first_metric_only = FALSE
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
  expect_true(abs(results[["rmse"]][["eval"]][[1L]] - 1.105012) < TOLERANCE)
  expect_true(abs(results[["l2"]][["eval"]][[1L]] - 1.221051) < TOLERANCE)
  expected_increasing_metric <- increasing_metric_starting_value + 0.1
  expect_true(
    abs(
      results[["increasing_metric"]][["eval"]][[1L]] - expected_increasing_metric
    ) < TOLERANCE
  )
  expect_true(abs(results[["constant_metric"]][["eval"]][[1L]] - CONSTANT_METRIC_VALUE) < TOLERANCE)

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
      expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.4864865) < TOLERANCE)
    }
    if ("binary_logloss" %in% unlist(eval_variation)) {
      expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.6932548) < TOLERANCE)
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
  expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.4864865) < TOLERANCE)
  expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.6932548) < TOLERANCE)
})

test_that("lgb.train() works when you give a function for eval", {
  set.seed(708L)
  nrounds <- 10L
  increasing_metric_starting_value <- get(ACCUMULATOR_NAME, envir = .GlobalEnv)
  bst <- lgb.train(
    params = list(
      objective = "binary"
      , metric = "None"
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
  expect_true(abs(results[["constant_metric"]][["eval"]][[1L]] - CONSTANT_METRIC_VALUE) < TOLERANCE)
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
  )
  dvalid <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
  )
  nrounds <- 10L

  #############################
  # train with early stopping #
  #############################
  early_stopping_rounds <- 5L
  bst  <- lgb.train(
    params = list(
      objective = "regression"
      , metric = c(
          "mape"
          , "rmse"
          , "mae"
      )
      , min_data_in_bin = 5L
      , early_stopping_rounds = early_stopping_rounds
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

  # Booster should understand thatt all three of these metrics should be minimized
  eval_info <- bst$.__enclos_env__$private$get_eval_info()
  expect_identical(eval_info, c("mape", "rmse", "l1"))
  expect_identical(
    unname(bst$.__enclos_env__$private$higher_better_inner_eval)
    , rep(FALSE, 3L)
  )
})


test_that("lgb.train() supports non-ASCII feature names", {
  dtrain <- lgb.Dataset(
    data = matrix(rnorm(400L), ncol =  4L)
    , label = rnorm(100L)
  )
  feature_names <- c("F_零", "F_一", "F_二", "F_三")
  bst <- lgb.train(
    data = dtrain
    , nrounds = 5L
    , obj = "regression"
    , params = list(
      metric = "rmse"
    )
    , colnames = feature_names
  )
  expect_true(lgb.is.Booster(bst))
  dumped_model <- jsonlite::fromJSON(bst$dump_model())

  # UTF-8 strings are not well-supported on Windows
  # * https://developer.r-project.org/Blog/public/2020/05/02/utf-8-support-on-windows/
  # * https://developer.r-project.org/Blog/public/2020/07/30/windows/utf-8-build-of-r-and-cran-packages/index.html
  if (UTF8_LOCALE && !ON_WINDOWS) {
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
        , min_data = 1L
        , learning_rate = 0.01
        , seed = 708L
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
    expect_true(abs(mae - expected_mae) < TOLERANCE)
  }
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
  )
  dvalid1 <- lgb.Dataset(
    data = as.matrix(validDF[["feat1"]], drop = FALSE)
    , label = validDF[["target"]]
  )
  dvalid2 <- lgb.Dataset(
    data = as.matrix(validDF[1L:10L, "feat1"], drop = FALSE)
    , label = validDF[1L:10L, "target"]
  )
  nrounds <- 10L
  train_params <- list(
    objective = "regression"
    , metric = "rmse"
    , learning_rate = 1.5
  )

  # example 1: two valids, neither are the training data
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , num_leaves = 5L
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
    , num_leaves = 5L
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
    , num_leaves = 5L
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
    , num_leaves = 5L
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
    , num_leaves = 5L
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
    , num_leaves = 5L
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
  )
  dvalid1 <- lgb.Dataset(
    data = as.matrix(validDF[1L:25L, "feat1"], drop = FALSE)
    , label = validDF[1L:25L, "target"]
  )
  nrounds <- 10L
  bst <- lgb.train(
    data = dtrain
    , nrounds = nrounds
    , num_leaves = 5L
    , valids = list(
      "valid1" = dvalid1
      , "something-random-we-would-not-hardcode" = dtrain
    )
    , params = list(
      objective = "binary"
      , metric = "auc"
      , learning_rate = 1.5
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
  )
  dvalid1 <- lgb.Dataset(
    data = as.matrix(validDF[1L:25L, "feat1"], drop = FALSE)
    , label = validDF[1L:25L, "target"]
  )
  dvalid2 <- lgb.Dataset(
    data = as.matrix(validDF[26L:50L, "feat1"], drop = FALSE)
    , label = validDF[26L:50L, "target"]
  )
  nrounds <- 10L
  bst <- lightgbm(
    data = dtrain
    , nrounds = nrounds
    , num_leaves = 5L
    , valids = list(
      "valid1" = dvalid1
      , "something-random-we-would-not-hardcode" = dtrain
      , "valid2" = dvalid2
    )
    , params = list(
      objective = "binary"
      , metric = "auc"
      , learning_rate = 1.5
    )
    , verbose = -7L
    , save_name = tempfile(fileext = ".model")
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
  expect_true(abs(results[["binary_error"]][["eval"]][[1L]] - 0.5005654) < TOLERANCE)
  expect_true(abs(results[["binary_logloss"]][["eval"]][[1L]] - 0.7011232) < TOLERANCE)

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
    )
    , data = DTRAIN_RANDOM_CLASSIFICATION
    , nfold = nfolds
    , nrounds = nrounds
    , eval = .constant_metric
  )

  # the difference metrics shouldn't have been mixed up with each other
  results <- bst$record_evals[["valid"]]
  expect_true(abs(results[["constant_metric"]][["eval"]][[1L]] - CONSTANT_METRIC_VALUE) < TOLERANCE)
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

  # best_score should be taken from the first metri
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
})

context("linear learner")

test_that("lgb.train() fit on linearly-relatead data improves when using linear learners", {
  set.seed(708L)
  .new_dataset <- function() {
    X <- matrix(rnorm(100L), ncol = 1L)
    return(lgb.Dataset(
      data = X
      , label = 2L * X + runif(nrow(X), 0L, 0.1)
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
  )

  dtrain <- .new_dataset()
  bst <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = params
    , valids = list("train" = dtrain)
  )
  expect_true(lgb.is.Booster(bst))

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = modifyList(params, list(linear_tree = TRUE))
    , valids = list("train" = dtrain)
  )
  expect_true(lgb.is.Booster(bst_linear))

  bst_last_mse <- bst$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  bst_lin_last_mse <- bst_linear$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  expect_true(bst_lin_last_mse <  bst_last_mse)
})


test_that("lgb.train() w/ linear learner fails already-constructed dataset with linear=false", {
  testthat::skip("Skipping this test because it causes issues for valgrind")
  set.seed(708L)
  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
  )

  dtrain <- lgb.Dataset(
    data = matrix(rnorm(100L), ncol = 1L)
    , label = rnorm(100L)
  )
  dtrain$construct()
  expect_error({
    bst_linear <- lgb.train(
      data = dtrain
      , nrounds = 10L
      , params = modifyList(params, list(linear_tree = TRUE))
    )
  }, regexp = "Cannot change linear_tree after constructed Dataset handle")
})

test_that("lgb.train() works with linear learners even if Dataset has missing values", {
  set.seed(708L)
  .new_dataset <- function() {
    values <- rnorm(100L)
    values[sample(seq_len(length(values)), size = 10L)] <- NA_real_
    X <- matrix(
      data = sample(values, size = 100L)
      , ncol = 1L
    )
    return(lgb.Dataset(
      data = X
      , label = 2L * X + runif(nrow(X), 0L, 0.1)
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
  )

  dtrain <- .new_dataset()
  bst <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = params
    , valids = list("train" = dtrain)
  )
  expect_true(lgb.is.Booster(bst))

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = modifyList(params, list(linear_tree = TRUE))
    , valids = list("train" = dtrain)
  )
  expect_true(lgb.is.Booster(bst_linear))

  bst_last_mse <- bst$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  bst_lin_last_mse <- bst_linear$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  expect_true(bst_lin_last_mse <  bst_last_mse)
})

test_that("lgb.train() works with linear learners, bagging, and a Dataset that has missing values", {
  set.seed(708L)
  .new_dataset <- function() {
    values <- rnorm(100L)
    values[sample(seq_len(length(values)), size = 10L)] <- NA_real_
    X <- matrix(
      data = sample(values, size = 100L)
      , ncol = 1L
    )
    return(lgb.Dataset(
      data = X
      , label = 2L * X + runif(nrow(X), 0L, 0.1)
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    , bagging_freq = 1L
    , subsample = 0.8
  )

  dtrain <- .new_dataset()
  bst <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = params
    , valids = list("train" = dtrain)
  )
  expect_true(lgb.is.Booster(bst))

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = modifyList(params, list(linear_tree = TRUE))
    , valids = list("train" = dtrain)
  )
  expect_true(lgb.is.Booster(bst_linear))

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
      , feature_pre_filter = FALSE
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
  )

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = modifyList(params, list(linear_tree = TRUE))
  )
  expect_true(lgb.is.Booster(bst_linear))
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
    ))
  }

  params <- list(
    objective = "regression"
    , verbose = -1L
    , metric = "mse"
    , seed = 0L
    , num_leaves = 2L
    # indices in categorical feature start from 1L, so here should be 2L
    , categorical_feature = 2L
  )

  dtrain <- .new_dataset()
  bst <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = params
    , valids = list("train" = dtrain)
  )
  expect_true(lgb.is.Booster(bst))

  dtrain <- .new_dataset()
  bst_linear <- lgb.train(
    data = dtrain
    , nrounds = 10L
    , params = modifyList(params, list(linear_tree = TRUE))
    , valids = list("train" = dtrain)
  )
  expect_true(lgb.is.Booster(bst_linear))

  bst_last_mse <- bst$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  bst_lin_last_mse <- bst_linear$record_evals[["train"]][["l2"]][["eval"]][[10L]]
  expect_true(bst_lin_last_mse <  bst_last_mse)
})

context("interaction constraints")

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
                 interaction_constraints = list(c(1L, length(colnames(train$data)) + 1L), 3L))
    expect_error({
      bst <- lightgbm(
        data = dtrain
        , params = params
        , nrounds = 2L
      )
    }, "supplied a too large value in interaction_constraints")
})

test_that(paste0("lgb.train() gives same result when interaction_constraints is specified as a list of ",
                 "character vectors, numeric vectors, or a combination"), {
  set.seed(1L)
  dtrain <- lgb.Dataset(train$data, label = train$label)

  params <- list(objective = "regression", interaction_constraints = list(c(1L, 2L), 3L))
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
  )
  pred1 <- bst$predict(test$data)

  cnames <- colnames(train$data)
  params <- list(objective = "regression", interaction_constraints = list(c(cnames[[1L]], cnames[[2L]]), cnames[[3L]]))
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
  )
  pred2 <- bst$predict(test$data)

  params <- list(objective = "regression", interaction_constraints = list(c(cnames[[1L]], cnames[[2L]]), 3L))
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
  dtrain <- lgb.Dataset(train$data, label = train$label)

  params <- list(objective = "regression", interaction_constraints = list(c(1L, 2L), 3L))
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
  )
  pred1 <- bst$predict(test$data)

  new_colnames <- paste0(colnames(train$data), "_x")
  params <- list(objective = "regression"
                 , interaction_constraints = list(c(new_colnames[1L], new_colnames[2L]), new_colnames[3L]))
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 2L
    , colnames = new_colnames
  )
  pred2 <- bst$predict(test$data)

  expect_equal(pred1, pred2)

})

test_that(paste0("Category encoding for R package works"), {
  # test category_encoders
  set.seed(1L)
  dtrain <- lgb.Dataset(train$data, label = train$label)
  dtest <- lgb.Dataset(test$data, label = test$label, reference = dtrain)
  cat_fid <- c(1L, 2L, 3L, 4L)
  # ``` category_encoders = "" ```   is equal to   ``` category_encoders = "raw" ```
  params <- list(objective = "binary", categorical_feature = cat_fid, category_encoders = "")
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 10L
    , verbose = 2L
    , valids = list("valid1" = dtest)
  )
  pred1 <- bst$predict(test$data)

  # treat the first 4 features as categorical features
  dtrain <- lgb.Dataset(train$data, label = train$label,
    categorical_feature = cat_fid, category_encoders = "raw")
  dtest <- lgb.Dataset(test$data, label = test$label,
    categorical_feature = cat_fid, reference = dtrain)
  params <- list(objective = "binary")
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 10L
    , verbose = 2L
    , valids = list("valid1" = dtest)
  )
  pred2 <- bst$predict(test$data)
  expect_equal(pred1, pred2)

  dtrain <- lgb.Dataset(train$data, label = train$label,
    categorical_feature = cat_fid)
  dtest <- lgb.Dataset(test$data, label = test$label,
    categorical_feature = cat_fid, reference = dtrain)
  params <- list(objective = "binary", category_encoders = "target,count,raw")
  bst <- lightgbm(
    data = dtrain
    , params = params
    , nrounds = 10L
    , verbose = 2L
    , valids = list("valid1" = dtest)
  )
  pred3 <- bst$predict(test$data)
  expect_equal(dim(dtrain), c(6513L, 134L))

  # test gbdt model with category_encoders
  model_file <- tempfile(fileext = ".model")
  lgb.save(bst, model_file)
  # finalize the booster and destroy it so you know we aren't cheating
  bst$finalize()
  expect_null(bst$.__enclos_env__$private$handle)
  rm(bst)

  bst2 <- lgb.load(
      filename = model_file
  )
  pred4 <- predict(bst2, test$data)
  expect_equal(pred3, pred4)


  # test Dataset binary store with category_encoders
  tmp_file <- tempfile(pattern = "lgb.Dataset_Category_Encoding_")
  lgb.Dataset.save(
    dataset = dtrain
    , fname = tmp_file
  )
  dtrain_read_in <- lgb.Dataset(data = tmp_file)

  tmp_file <- tempfile(pattern = "lgb.Dataset_Category_Encoding_2_")
  lgb.Dataset.save(
    dataset = dtest
    , fname = tmp_file
  )
  dtest_read_in <- lgb.Dataset(data = tmp_file)

  bst <- lightgbm(
    data = dtrain_read_in
    , params = params
    , nrounds = 10L
    , verbose = 2L
    , valids = list("valid1" = dtest_read_in)
  )
  pred5 <- bst$predict(test$data)
  expect_equal(pred3, pred5)
})
