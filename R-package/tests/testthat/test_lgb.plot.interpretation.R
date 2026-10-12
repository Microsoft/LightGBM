.sigmoid <- function(x) {
    1.0 / (1.0 + exp(-x))
}
.logit <- function(x) {
    log(x / (1.0 - x))
}

test_that("lgb.plot.interpretation works as expected for binary classification", {
    data(agaricus.train, package = "lightgbm")
    train <- agaricus.train
    dtrain <- lgb.Dataset(train$data, label = train$label)
    set_field(
        dataset = dtrain
        , field_name = "init_score"
        , data = rep(
            .logit(mean(train$label))
            , length(train$label)
        )
    )
    data(agaricus.test, package = "lightgbm")
    test <- agaricus.test
    params <- list(
        objective = "binary"
        , learning_rate = 0.01
        , num_leaves = 63L
        , max_depth = -1L
        , min_data_in_leaf = 1L
        , min_sum_hessian_in_leaf = 1.0
        , verbosity = .LGB_VERBOSITY
        , num_threads = .LGB_MAX_THREADS
    )
    model <- lgb.train(
        params = params
        , data = dtrain
        , nrounds = 3L
    )
    num_trees <- 5L
    tree_interpretation <- lgb.interpret(
        model = model
        , data = test$data
        , idxset = seq_len(num_trees)
    )
    expect_true({
        lgb.plot.interpretation(
            tree_interpretation_dt = tree_interpretation[[1L]]
            , top_n = 5L
        )
        TRUE
    })

    # should also work when you explicitly pass cex
    plot_res <- lgb.plot.interpretation(
        tree_interpretation_dt = tree_interpretation[[1L]]
        , top_n = 5L
        , cex = 0.95
    )
    expect_null(plot_res)
})

test_that("lgb.plot.interpretation works as expected for multiclass classification", {
    data(iris)

    # We must convert factors to numeric
    # They must be starting from number 0 to use multiclass
    # For instance: 0, 1, 2, 3, 4, 5...
    iris$Species <- as.numeric(as.factor(iris$Species)) - 1L

    # Create imbalanced training data (20, 30, 40 examples for classes 0, 1, 2)
    train <- as.matrix(iris[c(1L:20L, 51L:80L, 101L:140L), ])
    # The 10 last samples of each class are for validation
    test <- as.matrix(iris[c(41L:50L, 91L:100L, 141L:150L), ])
    dtrain <- lgb.Dataset(data = train[, 1L:4L], label = train[, 5L])
    dtest <- lgb.Dataset.create.valid(dtrain, data = test[, 1L:4L], label = test[, 5L])
    params <- list(
        objective = "multiclass"
        , metric = "multi_logloss"
        , num_class = 3L
        , learning_rate = 0.00001
        , min_data = 1L
        , num_threads = .LGB_MAX_THREADS
    )
    model <- lgb.train(
        params = params
        , data = dtrain
        , nrounds = 3L
        , verbose = .LGB_VERBOSITY
    )
    num_trees <- 5L
    tree_interpretation <- lgb.interpret(
        model = model
        , data = test[, 1L:4L]
        , idxset = seq_len(num_trees)
    )
    plot_res <- lgb.plot.interpretation(
        tree_interpretation_dt = tree_interpretation[[1L]]
        , top_n = 5L
    )
    expect_null(plot_res)
})
