library(Matrix)

test_that("Predictor's finalizer should not fail", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- lgb.Dataset(X, label = y)
    bst <- lgb.train(
        data = dtrain
        , params = list(
            objective = "regression"
            , num_threads = .LGB_MAX_THREADS
        )
        , verbose = .LGB_VERBOSITY
        , nrounds = 3L
    )
    model_file <- tempfile(fileext = ".model")
    bst$save_model(filename = model_file)
    predictor <- Predictor$new(modelfile = model_file)

    expect_true(.is_Predictor(predictor))

    expect_false(.is_null_handle(predictor$.__enclos_env__$private$handle))

    predictor$.__enclos_env__$private$finalize()
    expect_true(.is_null_handle(predictor$.__enclos_env__$private$handle))

    # calling finalize() a second time shouldn't cause any issues
    predictor$.__enclos_env__$private$finalize()
    expect_true(.is_null_handle(predictor$.__enclos_env__$private$handle))
})

test_that("predictions do not fail for integer input", {
    X <- as.matrix(as.integer(iris[, "Species"]), ncol = 1L)
    y <- iris[["Sepal.Length"]]
    dtrain <- lgb.Dataset(X, label = y)
    fit <- lgb.train(
        data = dtrain
        , params = list(
            objective = "regression"
            , num_threads = .LGB_MAX_THREADS
        )
        , verbose = .LGB_VERBOSITY
        , nrounds = 3L
    )
    X_double <- X[c(1L, 51L, 101L), , drop = FALSE]
    X_integer <- X_double
    storage.mode(X_double) <- "double"
    pred_integer <- predict(fit, X_integer)
    pred_double <- predict(fit, X_double)
    expect_equal(pred_integer, pred_double)
})

test_that("start_iteration works correctly", {
    set.seed(708L)
    data(agaricus.train, package = "lightgbm")
    data(agaricus.test, package = "lightgbm")
    train <- agaricus.train
    test <- agaricus.test
    dtrain <- lgb.Dataset(
        agaricus.train$data
        , label = agaricus.train$label
    )
    dtest <- lgb.Dataset.create.valid(
        dtrain
        , agaricus.test$data
        , label = agaricus.test$label
    )
    bst <- lightgbm(
        data = as.matrix(train$data)
        , label = train$label
        , params = list(
            num_leaves = 4L
            , learning_rate = 0.6
            , objective = "binary"
            , verbosity = .LGB_VERBOSITY
            , num_threads = .LGB_MAX_THREADS
        )
        , nrounds = 50L
        , valids = list("test" = dtest)
        , early_stopping_rounds = 2L
    )
    expect_true(.is_Booster(bst))
    pred1 <- predict(bst, newdata = test$data, type = "raw")
    pred_contrib1 <- predict(bst, test$data, type = "contrib")
    pred2 <- rep(0.0, length(pred1))
    pred_contrib2 <- rep(0.0, length(pred2))
    step <- 11L
    end_iter <- 49L
    if (bst$best_iter != -1L) {
        end_iter <- bst$best_iter - 1L
    }
    start_iters <- seq(0L, end_iter, by = step)
    for (start_iter in start_iters) {
        n_iter <- min(c(end_iter - start_iter + 1L, step))
        inc_pred <- predict(bst, test$data
            , start_iteration = start_iter
            , num_iteration = n_iter
            , type = "raw"
        )
        inc_pred_contrib <- bst$predict(test$data
            , start_iteration = start_iter
            , num_iteration = n_iter
            , predcontrib = TRUE
        )
        pred2 <- pred2 + inc_pred
        pred_contrib2 <- pred_contrib2 + inc_pred_contrib
    }
    expect_equal(pred2, pred1)
    expect_equal(pred_contrib2, pred_contrib1)

    pred_leaf1 <- predict(bst, test$data, type = "leaf")
    pred_leaf2 <- predict(bst, test$data, start_iteration = 0L, num_iteration = end_iter + 1L, type = "leaf")
    expect_equal(pred_leaf1, pred_leaf2)
})

test_that("Feature contributions from sparse inputs produce sparse outputs", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- as.numeric(mtcars[, 1L])
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bins = 5L))
    bst <- lgb.train(
      data = dtrain
      , obj = "regression"
      , nrounds = 5L
      , verbose = .LGB_VERBOSITY
      , params = list(min_data_in_leaf = 5L, num_threads = .LGB_MAX_THREADS)
    )

    pred_dense <- predict(bst, X, type = "contrib")

    Xcsc <- as(X, "CsparseMatrix")
    pred_csc <- predict(bst, Xcsc, type = "contrib")
    expect_s4_class(pred_csc, "dgCMatrix")
    expect_equal(unname(pred_dense), unname(as.matrix(pred_csc)))

    Xcsr <- as(X, "RsparseMatrix")
    pred_csr <- predict(bst, Xcsr, type = "contrib")
    expect_s4_class(pred_csr, "dgRMatrix")
    expect_equal(as(pred_csr, "CsparseMatrix"), pred_csc)

    Xspv <- as(X[1L, , drop = FALSE], "sparseVector")
    pred_spv <- predict(bst, Xspv, type = "contrib")
    expect_s4_class(pred_spv, "dsparseVector")
    expect_equal(Matrix::t(as(pred_spv, "CsparseMatrix")), unname(pred_csc[1L, , drop = FALSE]))
})

test_that("Sparse feature contribution predictions do not take inputs with wrong number of columns", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- as.numeric(mtcars[, 1L])
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bins = 5L))
    bst <- lgb.train(
      data = dtrain
      , obj = "regression"
      , nrounds = 5L
      , verbose = .LGB_VERBOSITY
      , params = list(min_data_in_leaf = 5L, num_threads = .LGB_MAX_THREADS)
    )

    X_wrong <- X[, c(1L:10L, 1L:10L)]
    X_wrong <- as(X_wrong, "CsparseMatrix")
    expect_error(predict(bst, X_wrong, type = "contrib"), regexp = "input data has 20 columns")

    X_wrong <- as(X_wrong, "RsparseMatrix")
    expect_error(predict(bst, X_wrong, type = "contrib"), regexp = "input data has 20 columns")

    X_wrong <- as(X_wrong, "CsparseMatrix")
    X_wrong <- X_wrong[, 1L:3L]
    expect_error(predict(bst, X_wrong, type = "contrib"), regexp = "input data has 3 columns")
})

test_that("Feature contribution predictions do not take non-general CSR or CSC inputs", {
    set.seed(123L)
    y <- runif(25L)
    Dmat <- matrix(runif(625L), nrow = 25L, ncol = 25L)
    Dmat <- crossprod(Dmat)
    Dmat <- as(Dmat, "symmetricMatrix")
    SmatC <- as(Dmat, "sparseMatrix")
    SmatR <- as(SmatC, "RsparseMatrix")

    dtrain <- lgb.Dataset(as.matrix(Dmat), label = y, params = list(max_bins = 5L))
    bst <- lgb.train(
      data = dtrain
      , obj = "regression"
      , nrounds = 5L
      , verbose = .LGB_VERBOSITY
      , params = list(min_data_in_leaf = 5L, num_threads = .LGB_MAX_THREADS)
    )

    expect_error(
      predict(bst, SmatC, type = "contrib")
      , regexp = "Predictions on sparse inputs are only allowed for 'dsparseVector', 'dgRMatrix', 'dgCMatrix' - got: dsCMatrix"  # nolint: line_length.
    )
    expect_error(
      predict(bst, SmatR, type = "contrib")
      , regexp = "Predictions on sparse inputs are only allowed for 'dsparseVector', 'dgRMatrix', 'dgCMatrix' - got: dsRMatrix" # nolint: line_length.
    )
})

test_that("predict() params should override keyword argument for raw-score predictions", {
  data(agaricus.train, package = "lightgbm")
  X <- agaricus.train$data
  y <- agaricus.train$label
  bst <- lgb.train(
    data = lgb.Dataset(
      data = X
      , label = y
      , params = list(
        data_seed = 708L
        , min_data_in_bin = 5L
      )
    )
    , params = list(
      objective = "binary"
      , min_data_in_leaf = 1L
      , seed = 708L
      , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = 10L
    , verbose = .LGB_VERBOSITY
  )

  # check that the predictions from predict.lgb.Booster() really look like raw score predictions
  preds_prob <- predict(bst, X)
  preds_raw_s3_keyword <- predict(bst, X, type = "raw")
  preds_prob_from_raw <- 1.0 / (1.0 + exp(-preds_raw_s3_keyword))
  expect_equal(preds_prob, preds_prob_from_raw, tolerance = .LGB_NUMERIC_TOLERANCE)
  accuracy <- sum(as.integer(preds_prob_from_raw > 0.5) == y) / length(y)
  expect_equal(accuracy, 1.0)

  # should get the same results from Booster$predict() method
  preds_raw_r6_keyword <- bst$predict(X, rawscore = TRUE)
  expect_equal(preds_raw_s3_keyword, preds_raw_r6_keyword)

  # using a parameter alias of predict_raw_score should result in raw scores being returned
  aliases <- .PARAMETER_ALIASES()[["predict_raw_score"]]
  expect_true(length(aliases) > 1L)
  for (rawscore_alias in aliases) {
    params <- as.list(
      stats::setNames(
        object = TRUE
        , nm = rawscore_alias
      )
    )
    preds_raw_s3_param <- predict(bst, X, params = params)
    preds_raw_r6_param <- bst$predict(X, params = params)
    expect_equal(preds_raw_s3_keyword, preds_raw_s3_param)
    expect_equal(preds_raw_s3_keyword, preds_raw_r6_param)
  }
})

test_that("predict() params should override keyword argument for leaf-index predictions", {
  data(mtcars)
  X <- as.matrix(mtcars[, which(names(mtcars) != "mpg")])
  y <- as.numeric(mtcars[, "mpg"])
  bst <- lgb.train(
    data = lgb.Dataset(
      data = X
      , label = y
      , params = list(
        min_data_in_bin = 1L
        , data_seed = 708L
      )
    )
    , params = list(
      objective = "regression"
      , min_data_in_leaf = 1L
      , seed = 708L
      , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = 10L
    , verbose = .LGB_VERBOSITY
  )

  # check that predictions really look like leaf index predictions
  preds_leaf_s3_keyword <- predict(bst, X, type = "leaf")
  expect_true(is.matrix(preds_leaf_s3_keyword))
  expect_equal(dim(preds_leaf_s3_keyword), c(nrow(X), bst$current_iter()))
  expect_true(min(preds_leaf_s3_keyword) >= 0L)
  trees_dt <- lgb.model.dt.tree(bst)
  max_leaf_by_tree_from_dt <- trees_dt[, .(idx = max(leaf_index, na.rm = TRUE)), by = tree_index]$idx
  max_leaf_by_tree_from_preds <- apply(preds_leaf_s3_keyword, 2L, max, na.rm = TRUE)
  expect_equal(max_leaf_by_tree_from_dt, max_leaf_by_tree_from_preds)

  # should get the same results from Booster$predict() method
  preds_leaf_r6_keyword <- bst$predict(X, predleaf = TRUE)
  expect_equal(preds_leaf_s3_keyword, preds_leaf_r6_keyword)

  # using a parameter alias of predict_leaf_index should result in leaf indices being returned
  aliases <- .PARAMETER_ALIASES()[["predict_leaf_index"]]
  expect_true(length(aliases) > 1L)
  for (predleaf_alias in aliases) {
    params <- as.list(
      stats::setNames(
        object = TRUE
        , nm = predleaf_alias
      )
    )
    preds_leaf_s3_param <- predict(bst, X, params = params)
    preds_leaf_r6_param <- bst$predict(X, params = params)
    expect_equal(preds_leaf_s3_keyword, preds_leaf_s3_param)
    expect_equal(preds_leaf_s3_keyword, preds_leaf_r6_param)
  }
})

test_that("predict() params should override keyword argument for feature contributions", {
  data(mtcars)
  X <- as.matrix(mtcars[, which(names(mtcars) != "mpg")])
  y <- as.numeric(mtcars[, "mpg"])
  bst <- lgb.train(
    data = lgb.Dataset(
      data = X
      , label = y
      , params = list(
        min_data_in_bin = 1L
        , data_seed = 708L
      )
    )
    , params = list(
      objective = "regression"
      , min_data_in_leaf = 1L
      , seed = 708L
      , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = 10L
    , verbose = .LGB_VERBOSITY
  )

  # check that predictions really look like feature contributions
  preds_contrib_s3_keyword <- predict(bst, X, type = "contrib")
  num_features <- ncol(X)
  shap_base_value <- unname(preds_contrib_s3_keyword[, ncol(preds_contrib_s3_keyword)])
  expect_true(is.matrix(preds_contrib_s3_keyword))
  expect_equal(dim(preds_contrib_s3_keyword), c(nrow(X), num_features + 1L))
  expect_equal(length(unique(shap_base_value)), 1L)
  expect_equal(mean(y), shap_base_value[1L])
  expect_equal(predict(bst, X), rowSums(preds_contrib_s3_keyword))

  # should get the same results from Booster$predict() method
  preds_contrib_r6_keyword <- bst$predict(X, predcontrib = TRUE)
  expect_equal(preds_contrib_s3_keyword, preds_contrib_r6_keyword)

  # using a parameter alias of predict_contrib should result in feature contributions being returned
  aliases <- .PARAMETER_ALIASES()[["predict_contrib"]]
  expect_true(length(aliases) > 1L)
  for (predcontrib_alias in aliases) {
    params <- as.list(
      stats::setNames(
        object = TRUE
        , nm = predcontrib_alias
      )
    )
    preds_contrib_s3_param <- predict(bst, X, params = params)
    preds_contrib_r6_param <- bst$predict(X, params = params)
    expect_equal(preds_contrib_s3_keyword, preds_contrib_s3_param)
    expect_equal(preds_contrib_s3_keyword, preds_contrib_r6_param)
  }
})

test_that("feature contributions sum to the raw score", {
  data(agaricus.train, package = "lightgbm")
  data(agaricus.test, package = "lightgbm")
  bst <- lgb.train(
    data = lgb.Dataset(agaricus.train$data, label = agaricus.train$label)
    , params = list(
      objective = "binary"
      , num_threads = .LGB_MAX_THREADS
    )
    , nrounds = 20L
    , verbose = .LGB_VERBOSITY
  )
  X <- as.matrix(agaricus.test$data)
  preds_contrib <- predict(bst, X, type = "contrib")
  expect_equal(dim(preds_contrib), c(nrow(X), ncol(X) + 1L))
  expect_equal(rowSums(preds_contrib), predict(bst, X, type = "raw"))
})

.expect_has_row_names <- function(pred, X) {
    if (is.vector(pred)) {
        rnames <- names(pred)
    } else {
        rnames <- row.names(pred)
    }
    expect_false(is.null(rnames))
    expect_true(is.vector(rnames))
    expect_true(length(rnames) > 0L)
    expect_equal(row.names(X), rnames)
}

.expect_doesnt_have_row_names <- function(pred) {
    if (is.vector(pred)) {
        expect_null(names(pred))
    } else {
        expect_null(row.names(pred))
    }
}

.check_all_row_name_expectations <- function(bst, X) {

    # dense matrix with row names
    pred <- predict(bst, X)
    .expect_has_row_names(pred, X)
    pred <- predict(bst, X, type = "raw")
    .expect_has_row_names(pred, X)
    pred <- predict(bst, X, type = "leaf")
    .expect_has_row_names(pred, X)
    pred <- predict(bst, X, type = "contrib")
    .expect_has_row_names(pred, X)

    # dense matrix without row names
    Xcopy <- X
    row.names(Xcopy) <- NULL
    pred <- predict(bst, Xcopy)
    .expect_doesnt_have_row_names(pred)

    # sparse matrix with row names
    Xcsc <- as(X, "CsparseMatrix")
    pred <- predict(bst, Xcsc)
    .expect_has_row_names(pred, Xcsc)
    pred <- predict(bst, Xcsc, type = "raw")
    .expect_has_row_names(pred, Xcsc)
    pred <- predict(bst, Xcsc, type = "leaf")
    .expect_has_row_names(pred, Xcsc)
    pred <- predict(bst, Xcsc, type = "contrib")
    .expect_has_row_names(pred, Xcsc)
    pred <- predict(bst, as(Xcsc, "RsparseMatrix"), type = "contrib")
    .expect_has_row_names(pred, Xcsc)

    # sparse matrix without row names
    Xcopy <- Xcsc
    row.names(Xcopy) <- NULL
    pred <- predict(bst, Xcopy)
    .expect_doesnt_have_row_names(pred)
}

test_that("predict() keeps row names from data (regression)", {
    data("mtcars")
    X <- as.matrix(mtcars[, -1L])
    y <- as.numeric(mtcars[, 1L])
    dtrain <- lgb.Dataset(
      X
      , label = y
      , params = list(
        max_bins = 5L
        , min_data_in_bin = 1L
      )
    )
    bst <- lgb.train(
        data = dtrain
        , obj = "regression"
        , nrounds = 5L
        , verbose = .LGB_VERBOSITY
        , params = list(min_data_in_leaf = 1L, num_threads = .LGB_MAX_THREADS)
    )
    .check_all_row_name_expectations(bst, X)
})

test_that("predict() keeps row names from data (binary classification)", {
    data(agaricus.train, package = "lightgbm")
    X <- as.matrix(agaricus.train$data)
    y <- agaricus.train$label
    row.names(X) <- paste0("rname", seq(1L, nrow(X)))
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bins = 5L))
    bst <- lgb.train(
        data = dtrain
        , obj = "binary"
        , nrounds = 5L
        , verbose = .LGB_VERBOSITY
        , params = list(num_threads = .LGB_MAX_THREADS)
    )
    .check_all_row_name_expectations(bst, X)
})

test_that("predict() keeps row names from data (multi-class classification)", {
    data(iris)
    y <- as.numeric(iris$Species) - 1.0
    X <- as.matrix(iris[, names(iris) != "Species"])
    row.names(X) <- paste0("rname", seq(1L, nrow(X)))
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bins = 5L))
    bst <- lgb.train(
        data = dtrain
        , obj = "multiclass"
        , params = list(num_class = 3L, num_threads = .LGB_MAX_THREADS)
        , nrounds = 5L
        , verbose = .LGB_VERBOSITY
    )
    .check_all_row_name_expectations(bst, X)
})

test_that("predictions for regression and binary classification are returned as vectors", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- as.numeric(mtcars[, 1L])
    dtrain <- lgb.Dataset(
      X
      , label = y
      , params = list(
        max_bins = 5L
        , min_data_in_bin = 1L
      )
    )
    model <- lgb.train(
      data = dtrain
      , obj = "regression"
      , nrounds = 5L
      , verbose = .LGB_VERBOSITY
      , params = list(min_data_in_leaf = 1L, num_threads = .LGB_MAX_THREADS)
    )
    pred <- predict(model, X)
    expect_true(is.vector(pred))
    expect_equal(length(pred), nrow(X))
    pred <- predict(model, X, type = "raw")
    expect_true(is.vector(pred))
    expect_equal(length(pred), nrow(X))

    data(agaricus.train, package = "lightgbm")
    X <- agaricus.train$data
    y <- agaricus.train$label
    dtrain <- lgb.Dataset(X, label = y)
    model <- lgb.train(
      data = dtrain
      , obj = "binary"
      , nrounds = 5L
      , verbose = .LGB_VERBOSITY
      , params = list(num_threads = .LGB_MAX_THREADS)
    )
    pred <- predict(model, X)
    expect_true(is.vector(pred))
    expect_equal(length(pred), nrow(X))
    pred <- predict(model, X, type = "raw")
    expect_true(is.vector(pred))
    expect_equal(length(pred), nrow(X))
})

test_that("predictions for multiclass classification are returned as matrix", {
    data(iris)
    X <- as.matrix(iris[, -5L])
    y <- as.numeric(iris$Species) - 1.0
    dtrain <- lgb.Dataset(X, label = y)
    model <- lgb.train(
      data = dtrain
      , obj = "multiclass"
      , nrounds = 5L
      , verbose = .LGB_VERBOSITY
      , params = list(num_class = 3L, num_threads = .LGB_MAX_THREADS)
    )
    pred <- predict(model, X)
    expect_true(is.matrix(pred))
    expect_equal(nrow(pred), nrow(X))
    expect_equal(ncol(pred), 3L)
    pred <- predict(model, X, type = "raw")
    expect_true(is.matrix(pred))
    expect_equal(nrow(pred), nrow(X))
    expect_equal(ncol(pred), 3L)
})

test_that("Single-row predictions are identical to multi-row ones", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- mtcars[, 1L]
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bin = 5L))
    params <- list(min_data_in_leaf = 2L, num_threads = .LGB_MAX_THREADS)
    model <- lgb.train(
      params = params
     , data = dtrain
     , obj = "regression"
     , nrounds = 5L
     , verbose = -1L
    )

    x1 <- X[1L, , drop = FALSE]
    x11 <- X[11L, , drop = FALSE]
    x1_spv <- as(x1, "sparseVector")
    x11_spv <- as(x11, "sparseVector")
    x1_csr <- as(x1, "RsparseMatrix")
    x11_csr <- as(x11, "RsparseMatrix")

    pred_all <- predict(model, X)
    pred1_wo_config <- predict(model, x1)
    pred11_wo_config <- predict(model, x11)
    pred1_spv_wo_config <- predict(model, x1_spv)
    pred11_spv_wo_config <- predict(model, x11_spv)
    pred1_csr_wo_config <- predict(model, x1_csr)
    pred11_csr_wo_config <- predict(model, x11_csr)

    lgb.configure_fast_predict(model)
    pred1_w_config <- predict(model, x1)
    pred11_w_config <- predict(model, x11)

    model <- lgb.train(
      params = params
     , data = dtrain
     , obj = "regression"
     , nrounds = 5L
     , verbose = -1L
    )
    lgb.configure_fast_predict(model, csr = TRUE)
    pred1_spv_w_config <- predict(model, x1_spv)
    pred11_spv_w_config <- predict(model, x11_spv)
    pred1_csr_w_config <- predict(model, x1_csr)
    pred11_csr_w_config <- predict(model, x11_csr)

    expect_equal(pred1_wo_config, pred_all[1L])
    expect_equal(pred11_wo_config, pred_all[11L])
    expect_equal(pred1_spv_wo_config, unname(pred_all[1L]))
    expect_equal(pred11_spv_wo_config, unname(pred_all[11L]))
    expect_equal(pred1_csr_wo_config, pred_all[1L])
    expect_equal(pred11_csr_wo_config, pred_all[11L])

    expect_equal(pred1_w_config, pred_all[1L])
    expect_equal(pred11_w_config, pred_all[11L])
    expect_equal(pred1_spv_w_config, unname(pred_all[1L]))
    expect_equal(pred11_spv_w_config, unname(pred_all[11L]))
    expect_equal(pred1_csr_w_config, pred_all[1L])
    expect_equal(pred11_csr_w_config, pred_all[11L])
})

test_that("Fast-predict configuration accepts non-default prediction types", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- mtcars[, 1L]
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bin = 5L))
    params <- list(min_data_in_leaf = 2L, num_threads = .LGB_MAX_THREADS)
    model <- lgb.train(
      params = params
     , data = dtrain
     , obj = "regression"
     , nrounds = 5L
     , verbose = -1L
    )

    x1 <- X[1L, , drop = FALSE]
    x11 <- X[11L, , drop = FALSE]

    pred_all <- predict(model, X, type = "leaf")
    pred1_wo_config <- predict(model, x1, type = "leaf")
    pred11_wo_config <- predict(model, x11, type = "leaf")
    expect_equal(pred1_wo_config, pred_all[1L, , drop = FALSE])
    expect_equal(pred11_wo_config, pred_all[11L, , drop = FALSE])

    lgb.configure_fast_predict(model, type = "leaf")
    pred1_w_config <- predict(model, x1, type = "leaf")
    pred11_w_config <- predict(model, x11, type = "leaf")
    expect_equal(pred1_w_config, pred_all[1L, , drop = FALSE])
    expect_equal(pred11_w_config, pred_all[11L, , drop = FALSE])
})

test_that("Fast-predict configuration does not block other prediction types", {
    data(mtcars)
    X <- as.matrix(mtcars[, -1L])
    y <- mtcars[, 1L]
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bin = 5L))
    params <- list(min_data_in_leaf = 2L, num_threads = .LGB_MAX_THREADS)
    model <- lgb.train(
      params = params
     , data = dtrain
     , obj = "regression"
     , nrounds = 5L
     , verbose = -1L
    )

    x1 <- X[1L, , drop = FALSE]
    x11 <- X[11L, , drop = FALSE]

    pred_all <- predict(model, X)
    pred_all_leaf <- predict(model, X, type = "leaf")

    lgb.configure_fast_predict(model)
    pred1_w_config <- predict(model, x1)
    pred11_w_config <- predict(model, x11)
    pred1_leaf_w_config <- predict(model, x1, type = "leaf")
    pred11_leaf_w_config <- predict(model, x11, type = "leaf")

    expect_equal(pred1_w_config, pred_all[1L])
    expect_equal(pred11_w_config, pred_all[11L])
    expect_equal(pred1_leaf_w_config, pred_all_leaf[1L, , drop = FALSE])
    expect_equal(pred11_leaf_w_config, pred_all_leaf[11L, , drop = FALSE])
})

test_that("predict type='class' returns predicted class for classification objectives", {
    data(agaricus.train, package = "lightgbm")
    X <- as.matrix(agaricus.train$data)
    y <- agaricus.train$label
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bins = 5L))
    bst <- lgb.train(
        data = dtrain
        , obj = "binary"
        , nrounds = 5L
        , verbose = .LGB_VERBOSITY
        , params = list(num_threads = .LGB_MAX_THREADS)
    )
    pred <- predict(bst, X, type = "class")
    expect_true(all(pred %in% c(0L, 1L)))

    data(iris)
    X <- as.matrix(iris[, -5L])
    y <- as.numeric(iris$Species) - 1.0
    dtrain <- lgb.Dataset(X, label = y)
    model <- lgb.train(
      data = dtrain
      , obj = "multiclass"
      , nrounds = 5L
      , verbose = .LGB_VERBOSITY
      , params = list(num_class = 3L, num_threads = .LGB_MAX_THREADS)
    )
    pred <- predict(model, X, type = "class")
    expect_true(all(pred %in% c(0L, 1L, 2L)))
})

test_that("predict type='class' returns values in the target's range for regression objectives", {
    data(agaricus.train, package = "lightgbm")
    X <- as.matrix(agaricus.train$data)
    y <- agaricus.train$label
    dtrain <- lgb.Dataset(X, label = y, params = list(max_bins = 5L))
    bst <- lgb.train(
        data = dtrain
        , obj = "regression"
        , nrounds = 5L
        , verbose = .LGB_VERBOSITY
        , params = list(num_threads = .LGB_MAX_THREADS)
    )
    pred <- predict(bst, X, type = "class")
    expect_true(!any(pred %in% c(0.0, 1.0)))
})
