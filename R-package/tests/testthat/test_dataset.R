context("testing lgb.Dataset functionality")

data(agaricus.test, package = "lightgbm")
test_data <- agaricus.test$data[1L:100L, ]
test_label <- agaricus.test$label[1L:100L]

test_that("lgb.Dataset: basic construction, saving, loading", {
  # from sparse matrix
  dtest1 <- lgb.Dataset(test_data, label = test_label)
  # from dense matrix
  dtest2 <- lgb.Dataset(as.matrix(test_data), label = test_label)
  expect_equal(getinfo(dtest1, "label"), getinfo(dtest2, "label"))

  # save to a local file
  tmp_file <- tempfile("lgb.Dataset_")
  lgb.Dataset.save(dtest1, tmp_file)
  # read from a local file
  dtest3 <- lgb.Dataset(tmp_file)
  lgb.Dataset.construct(dtest3)
  unlink(tmp_file)
  expect_equal(getinfo(dtest1, "label"), getinfo(dtest3, "label"))
})

test_that("lgb.Dataset: getinfo & setinfo", {
  dtest <- lgb.Dataset(test_data)
  dtest$construct()

  setinfo(dtest, "label", test_label)
  labels <- getinfo(dtest, "label")
  expect_equal(test_label, getinfo(dtest, "label"))

  expect_true(length(getinfo(dtest, "weight")) == 0L)
  expect_true(length(getinfo(dtest, "init_score")) == 0L)

  # any other label should error
  expect_error(setinfo(dtest, "asdf", test_label))
})

test_that("lgb.Dataset: slice, dim", {
  dtest <- lgb.Dataset(test_data, label = test_label)
  lgb.Dataset.construct(dtest)
  expect_equal(dim(dtest), dim(test_data))
  dsub1 <- slice(dtest, seq_len(42L))
  lgb.Dataset.construct(dsub1)
  expect_equal(nrow(dsub1), 42L)
  expect_equal(ncol(dsub1), ncol(test_data))
})

test_that("lgb.Dataset: colnames", {
  dtest <- lgb.Dataset(test_data, label = test_label)
  expect_equal(colnames(dtest), colnames(test_data))
  lgb.Dataset.construct(dtest)
  expect_equal(colnames(dtest), colnames(test_data))
  expect_error({
    colnames(dtest) <- "asdf"
  })
  new_names <- make.names(seq_len(ncol(test_data)))
  expect_silent(colnames(dtest) <- new_names)
  expect_equal(colnames(dtest), new_names)
})

test_that("lgb.Dataset: nrow is correct for a very sparse matrix", {
  nr <- 1000L
  x <- Matrix::rsparsematrix(nr, 100L, density = 0.0005)
  # we want it very sparse, so that last rows are empty
  expect_lt(max(x@i), nr)
  dtest <- lgb.Dataset(x)
  expect_equal(dim(dtest), dim(x))
})

test_that("lgb.Dataset: Dataset should be able to construct from matrix and return non-null handle", {
  rawData <- matrix(runif(1000L), ncol = 10L)
  ref_handle <- NULL
  handle <- .Call(
    LGBM_DatasetCreateFromMat_R
    , rawData
    , NULL
    , nrow(rawData)
    , ncol(rawData)
    , lightgbm:::lgb.params2str(params = list())
    , ref_handle
  )
  expect_is(handle, "externalptr")
  expect_false(is.null(handle))
  .Call(LGBM_DatasetFree_R, handle)
  handle <- NULL
})

test_that("cpp errors should be raised as proper R errors", {
  data(agaricus.train, package = "lightgbm")
  train <- agaricus.train
  dtrain <- lgb.Dataset(
    train$data
    , label = train$label
    , init_score = seq_len(10L)
  )
  expect_error({
    dtrain$construct()
  }, regexp = "Initial score size doesn't match data size")
})

test_that("lgb.Dataset$setinfo() should convert 'group' to integer", {
  ds <- lgb.Dataset(
    data = matrix(rnorm(100L), nrow = 50L, ncol = 2L)
    , label = sample(c(0L, 1L), size = 50L, replace = TRUE)
  )
  ds$construct()
  current_group <- ds$getinfo("group")
  expect_null(current_group)
  group_as_numeric <- rep(25.0, 2L)
  ds$setinfo("group", group_as_numeric)
  expect_identical(ds$getinfo("group"), as.integer(group_as_numeric))
})

test_that("lgb.Dataset should throw an error if 'reference' is provided but of the wrong format", {
  data(agaricus.test, package = "lightgbm")
  test_data <- agaricus.test$data[1L:100L, ]
  test_label <- agaricus.test$label[1L:100L]
  # Try to trick lgb.Dataset() into accepting bad input
  expect_error({
    dtest <- lgb.Dataset(
      data = test_data
      , label = test_label
      , reference = data.frame(x = seq_len(10L), y = seq_len(10L))
    )
  }, regexp = "reference must be a")
})

test_that("Dataset$new() should throw an error if 'predictor' is provided but of the wrong format", {
  data(agaricus.test, package = "lightgbm")
  test_data <- agaricus.test$data[1L:100L, ]
  test_label <- agaricus.test$label[1L:100L]
  expect_error({
    dtest <- Dataset$new(
      data = test_data
      , label = test_label
      , predictor = data.frame(x = seq_len(10L), y = seq_len(10L))
    )
  }, regexp = "predictor must be a", fixed = TRUE)
})

test_that("Dataset$get_params() successfully returns parameters if you passed them", {
  # note that this list uses one "main" parameter (feature_pre_filter) and one that
  # is an alias (is_sparse), to check that aliases are handled correctly
  params <- list(
    "feature_pre_filter" = TRUE
    , "is_sparse" = FALSE
  )
  ds <- lgb.Dataset(
    test_data
    , label = test_label
    , params = params
  )
  returned_params <- ds$get_params()
  expect_identical(class(returned_params), "list")
  expect_identical(length(params), length(returned_params))
  expect_identical(sort(names(params)), sort(names(returned_params)))
  for (param_name in names(params)) {
    expect_identical(params[[param_name]], returned_params[[param_name]])
  }
})

test_that("Dataset$get_params() ignores irrelevant parameters", {
  params <- list(
    "feature_pre_filter" = TRUE
    , "is_sparse" = FALSE
    , "nonsense_parameter" = c(1.0, 2.0, 5.0)
  )
  ds <- lgb.Dataset(
    test_data
    , label = test_label
    , params = params
  )
  returned_params <- ds$get_params()
  expect_false("nonsense_parameter" %in% names(returned_params))
})

test_that("Dataset$update_parameters() does nothing for empty inputs", {
  ds <- lgb.Dataset(
    test_data
    , label = test_label
  )
  initial_params <- ds$get_params()
  expect_identical(initial_params, list())

  # update_params() should return "self" so it can be chained
  res <- ds$update_params(
    params = list()
  )
  expect_true(lgb.is.Dataset(res))

  new_params <- ds$get_params()
  expect_identical(new_params, initial_params)
})

test_that("Dataset$update_params() works correctly for recognized Dataset parameters", {
  ds <- lgb.Dataset(
    test_data
    , label = test_label
  )
  initial_params <- ds$get_params()
  expect_identical(initial_params, list())

  new_params <- list(
    "data_random_seed" = 708L
    , "enable_bundle" = FALSE
  )
  res <- ds$update_params(
    params = new_params
  )
  expect_true(lgb.is.Dataset(res))

  updated_params <- ds$get_params()
  for (param_name in names(new_params)) {
    expect_identical(new_params[[param_name]], updated_params[[param_name]])
  }
})

test_that("Dataset$finalize() should not fail on an already-finalized Dataset", {
  dtest <- lgb.Dataset(
    data = test_data
    , label = test_label
  )
  expect_true(lgb.is.null.handle(dtest$.__enclos_env__$private$handle))

  dtest$construct()
  expect_false(lgb.is.null.handle(dtest$.__enclos_env__$private$handle))

  dtest$finalize()
  expect_true(lgb.is.null.handle(dtest$.__enclos_env__$private$handle))

  # calling finalize() a second time shouldn't cause any issues
  dtest$finalize()
  expect_true(lgb.is.null.handle(dtest$.__enclos_env__$private$handle))
})

test_that("lgb.Dataset: should be able to run lgb.train() immediately after using lgb.Dataset() on a file", {
  dtest <- lgb.Dataset(
    data = test_data
    , label = test_label
  )
  tmp_file <- tempfile(pattern = "lgb.Dataset_")
  lgb.Dataset.save(
    dataset = dtest
    , fname = tmp_file
  )

  # read from a local file
  dtest_read_in <- lgb.Dataset(data = tmp_file)

  param <- list(
    objective = "binary"
    , metric = "binary_logloss"
    , num_leaves = 5L
    , learning_rate = 1.0
  )

  # should be able to train right away
  bst <- lgb.train(
    params = param
    , data = dtest_read_in
  )

  expect_true(lgb.is.Booster(x = bst))
})

test_that("lgb.Dataset: should be able to run lgb.cv() immediately after using lgb.Dataset() on a file", {
  dtest <- lgb.Dataset(
    data = test_data
    , label = test_label
  )
  tmp_file <- tempfile(pattern = "lgb.Dataset_")
  lgb.Dataset.save(
    dataset = dtest
    , fname = tmp_file
  )

  # read from a local file
  dtest_read_in <- lgb.Dataset(data = tmp_file)

  param <- list(
    objective = "binary"
    , metric = "binary_logloss"
    , num_leaves = 5L
    , learning_rate = 1.0
  )

  # should be able to train right away
  bst <- lgb.cv(
    params = param
    , data = dtest_read_in
  )

  expect_is(bst, "lgb.CVBooster")
})

test_that("lgb.Dataset: should be able to use and retrieve long feature names", {
  # set one feature to a value longer than the default buffer size used
  # in LGBM_DatasetGetFeatureNames_R
  feature_names <- names(iris)
  long_name <- paste0(rep("a", 1000L), collapse = "")
  feature_names[1L] <- long_name
  names(iris) <- feature_names
  # check that feature name survived the trip from R to C++ and back
  dtrain <- lgb.Dataset(
    data = as.matrix(iris[, -5L])
    , label = as.numeric(iris$Species) - 1L
  )
  dtrain$construct()
  col_names <- dtrain$get_colnames()
  expect_equal(col_names[1L], long_name)
  expect_equal(nchar(col_names[1L]), 1000L)
})
