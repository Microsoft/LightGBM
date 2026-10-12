data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
train <- agaricus.train
test <- agaricus.test

test_that("Feature penalties work properly", {
  # Fit a series of models with varying penalty on most important variable
  var_name <- "odor=none"
  var_index <- which(train$data@Dimnames[[2L]] == var_name)

  bst <- lapply(seq(1.0, 0.0, by = -0.1), function(x) {
    feature_penalties <- rep(1.0, ncol(train$data))
    feature_penalties[var_index] <- x
    lightgbm(
      data = train$data
      , label = train$label
      , params = list(
        num_leaves = 5L
        , learning_rate = 0.05
        , objective = "binary"
        , feature_penalty = paste(feature_penalties, collapse = ",")
        , metric = "binary_error"
        , num_threads = .LGB_MAX_THREADS
      )
      , nrounds = 5L
      , verbose = -1L
    )
  })

  var_gain <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Gain])
  var_cover <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Cover])
  var_freq <- lapply(bst, function(x) lgb.importance(x)[Feature == var_name, Frequency])

  # Ensure that feature gain, cover, and frequency decreases with stronger penalties
  expect_true(all(diff(unlist(var_gain)) <= 0.0))
  expect_true(all(diff(unlist(var_cover)) <= 0.0))
  expect_true(all(diff(unlist(var_freq)) <= 0.0))

  expect_lt(min(diff(unlist(var_gain))), 0.0)
  expect_lt(min(diff(unlist(var_cover))), 0.0)
  expect_lt(min(diff(unlist(var_freq))), 0.0)

  # Ensure that feature is not used when feature_penalty = 0
  expect_length(var_gain[[length(var_gain)]], 0L)
})

test_that(".PARAMETER_ALIASES() returns a named list of character vectors, where names are unique", {
  param_aliases <- .PARAMETER_ALIASES()
  expect_identical(class(param_aliases), "list")
  expect_true(length(param_aliases) > 100L)
  expect_true(is.character(names(param_aliases)))
  expect_true(is.character(param_aliases[["boosting"]]))
  expect_true(is.character(param_aliases[["early_stopping_round"]]))
  expect_true(is.character(param_aliases[["num_iterations"]]))
  expect_true(is.character(param_aliases[["pre_partition"]]))
  expect_true(length(names(param_aliases)) == length(param_aliases))
  expect_true(all(sapply(param_aliases, is.character)))
  expect_true(length(unique(names(param_aliases))) == length(param_aliases))
  expect_equal(sort(param_aliases[["task"]]), c("task", "task_type"))
  expect_equal(param_aliases[["bagging_fraction"]], c("bagging_fraction", "bagging", "sub_row", "subsample"))
})

test_that(".PARAMETER_ALIASES() uses the internal session cache", {

  cache_key <- "PARAMETER_ALIASES"

  # clear cache, so this test isn't reliant on the order unit tests are run in
  if (exists(cache_key, where = .lgb_session_cache_env)) {
    rm(list = cache_key, envir = .lgb_session_cache_env)
  }
  expect_false(exists(cache_key, where = .lgb_session_cache_env))

  # check that result looks correct for at least one parameter
  iter_aliases <- .PARAMETER_ALIASES()[["num_iterations"]]
  expect_true(is.character(iter_aliases))
  expect_true(all(c("num_round", "nrounds") %in% iter_aliases))

  # patch the cache to check that .PARAMETER_ALIASES() checks it
  assign(
    x = cache_key
    , value = list(num_iterations = c("test", "other_test"))
    , envir = .lgb_session_cache_env
  )
  iter_aliases <- .PARAMETER_ALIASES()[["num_iterations"]]
  expect_equal(iter_aliases, c("test", "other_test"))

  # re-set cache so this doesn't interfere with other unit tests
  if (exists(cache_key, where = .lgb_session_cache_env)) {
    rm(list = cache_key, envir = .lgb_session_cache_env)
  }
  expect_false(exists(cache_key, where = .lgb_session_cache_env))
})

test_that("training should warn if you use 'dart' boosting with early stopping", {
  for (boosting_param in .PARAMETER_ALIASES()[["boosting"]]) {
    params <- list(
        num_leaves = 5L
        , learning_rate = 0.05
        , objective = "binary"
        , metric = "binary_error"
        , num_threads = .LGB_MAX_THREADS
    )
    params[[boosting_param]] <- "dart"

    # warning: early stopping requested
    expect_warning({
      result <- lightgbm(
        data = train$data
        , label = train$label
        , params = params
        , nrounds = 2L
        , num_threads = .LGB_MAX_THREADS
        , verbose = .LGB_VERBOSITY
        , early_stopping_rounds = 1L
      )
    }, regexp = "Early stopping is not available in 'dart' mode")

    # no warning: early stopping not requested
    expect_silent({
      result <- lightgbm(
        data = train$data
        , label = train$label
        , params = params
        , nrounds = 2L
        , num_threads = .LGB_MAX_THREADS
        , verbose = .LGB_VERBOSITY
        , early_stopping_rounds = NULL
      )
    })
  }
})

test_that("lgb.cv() should warn if you use 'dart' boosting with early stopping", {
  for (boosting_param in .PARAMETER_ALIASES()[["boosting"]]) {
    params <- list(
      num_leaves = 5L
      , objective = "binary"
      , metric = "binary_error"
      , num_threads = .LGB_MAX_THREADS
    )
    params[[boosting_param]] <- "dart"

    # warning: early stopping requested
    expect_warning({
      result <- lgb.cv(
        data = lgb.Dataset(
          data  = train$data
          , label = train$label
        )
        , params = params
        , nrounds = 2L
        , verbose = .LGB_VERBOSITY
        , early_stopping_rounds = 1L
      )
    }, regexp = "Early stopping is not available in 'dart' mode")

    # no warning: early stopping not requested
    expect_silent({
      result <- lgb.cv(
        data = lgb.Dataset(
          data  = train$data
          , label = train$label
        )
        , params = params
        , nrounds = 2L
        , verbose = .LGB_VERBOSITY
        , early_stopping_rounds = NULL
      )
    })
  }
})
