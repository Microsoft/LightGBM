#' @name lgb.interpret
#' @title Compute feature contribution of prediction
#' @description Computes feature contribution components of rawscore prediction.
#' @param model object of class \code{lgb.Booster}.
#' @param data a matrix object or a dgCMatrix object.
#' @param idxset an integer vector of indices of rows needed.
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration.
#'
#' @return For regression, binary classification and lambdarank model, a \code{list} of \code{data.table}
#'         with the following columns:
#'         \itemize{
#'             \item{\code{Feature}: Feature names in the model.}
#'             \item{\code{Contribution}: The total contribution of this feature's splits.}
#'         }
#'         For multiclass classification, a \code{list} of \code{data.table} with the Feature column and
#'         Contribution columns to each class.
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' Logit <- function(x) log(x / (1.0 - x))
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' set_field(
#'   dataset = dtrain
#'   , field_name = "init_score"
#'   , data = rep(Logit(mean(train$label)), length(train$label))
#' )
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#'
#' params <- list(
#'     objective = "binary"
#'     , learning_rate = 0.1
#'     , max_depth = -1L
#'     , min_data_in_leaf = 1L
#'     , min_sum_hessian_in_leaf = 1.0
#'     , num_threads = 2L
#' )
#' model <- lgb.train(
#'     params = params
#'     , data = dtrain
#'     , nrounds = 3L
#' )
#'
#' tree_interpretation <- lgb.interpret(model, test$data, 1L:5L)
#' }
#' @importFrom data.table as.data.table
#' @export
lgb.interpret <- function(model,
                           data,
                           idxset,
                           num_iteration = NULL) {

  # Get tree model
  tree_dt <- lgb.model.dt.tree(model = model, num_iteration = num_iteration)

  # Check number of classes
  num_class <- model$.__enclos_env__$private$num_class

  # Get vector list
  tree_interpretation_dt_list <- vector(mode = "list", length = length(idxset))

  # Get parsed predictions of data
  pred_mat <- t(
    model$predict(
      data = data[idxset, , drop = FALSE]
      , num_iteration = num_iteration
      , predleaf = TRUE
    )
  )
  leaf_index_dt <- data.table::as.data.table(x = pred_mat)
  leaf_index_mat_list <- lapply(
    X = leaf_index_dt
    , FUN = matrix
    , ncol = num_class
    , byrow = TRUE
  )

  # Get list of trees
  tree_index_mat_list <- lapply(
    X = leaf_index_mat_list
    , FUN = function(x) {
      matrix(seq_along(x) - 1L, ncol = num_class, byrow = TRUE)
    }
  )

  for (i in seq_along(idxset)) {
    tree_interpretation_dt_list[[i]] <- .single_row_interpret(
      tree_dt = tree_dt
      , num_class = num_class
      , tree_index_mat = tree_index_mat_list[[i]]
      , leaf_index_mat = leaf_index_mat_list[[i]]
    )
  }

  return(tree_interpretation_dt_list)

}

#' @name lgb.interprete
#' @title DEPRECATED - use lgb.interpret() instead
#' @description Alias for \code{lgb.interpret}.
#' @param ... Arguments passed through to \code{lgb.interpret}
#' @export
lgb.interprete <- function(...) {
    warning("lgb.interprete() is deprecated and will be removed in a future release. Use lgb.interpret() instead.")
    return(lgb.interpret(...))
}

#' @importFrom data.table data.table
single.tree.interpret <- function(tree_dt,
                                  tree_id,
                                  leaf_id) {

  # Match tree id
  single_tree_dt <- tree_dt[tree_index == tree_id, ]

  # Get leaves
  leaf_dt <- single_tree_dt[leaf_index == leaf_id, .(leaf_index, leaf_parent, leaf_value)]

  # Get nodes
  node_dt <- single_tree_dt[!is.na(split_index), .(split_index, split_feature, node_parent, internal_value)]

  # Prepare sequences
  tree_data <- new.env()
  tree_data[["feature_seq"]] <- character(0L)
  tree_data[["value_seq"]] <- numeric(0L)

  # Get to root from leaf
  .leaf_to_root <- function(parent_id, current_value) {

    # put value at the beginning
    tree_data[["value_seq"]] <- c(current_value, tree_data[["value_seq"]])

    if (!is.na(parent_id)) {

      # Not null means existing node
      this_node <- node_dt[split_index == parent_id, ]
      tree_data[["feature_seq"]] <- c(this_node[["split_feature"]], tree_data[["feature_seq"]])
      .leaf_to_root(
        parent_id = this_node[["node_parent"]]
        , current_value = this_node[["internal_value"]]
      )

    }
    return(invisible(NULL))
  }

  # Perform leaf to root conversion
  .leaf_to_root(
    parent_id = leaf_dt[["leaf_parent"]]
    , current_value = leaf_dt[["leaf_value"]]
  )

  return(
    data.table::data.table(
      Feature = tree_data[["feature_seq"]]
      , Contribution = diff.default(tree_data[["value_seq"]])
    )
  )

}

#' @importFrom data.table := rbindlist setorder
.multiple_tree_interpret <- function(tree_dt,
                                     tree_index,
                                     leaf_index) {

  interp_dt <- data.table::rbindlist(
    l = mapply(
      FUN = single.tree.interpret
      , tree_id = tree_index
      , leaf_id = leaf_index
      , MoreArgs = list(
        tree_dt = tree_dt
      )
      , SIMPLIFY = FALSE
      , USE.NAMES = TRUE
    )
    , use.names = TRUE
  )

  interp_dt <- interp_dt[, .(Contribution = sum(Contribution)), by = "Feature"]

  # Sort features in descending order by contribution
  interp_dt[, abs_contribution := abs(Contribution)]
  data.table::setorder(
    x = interp_dt
    , -abs_contribution
  )

  # Drop absolute value of contribution (only needed for sorting)
  interp_dt[, abs_contribution := NULL]

  return(interp_dt)

}

#' @importFrom data.table set setnames
.single_row_interpret <- function(tree_dt, num_class, tree_index_mat, leaf_index_mat) {

  # Prepare vector list
  tree_interpretation <- vector(mode = "list", length = num_class)

  # Loop throughout each class
  for (i in seq_len(num_class)) {

    next_interp_dt <- .multiple_tree_interpret(
      tree_dt = tree_dt
      , tree_index = tree_index_mat[, i]
      , leaf_index = leaf_index_mat[, i]
    )

    if (num_class > 1L) {
      data.table::setnames(
        x = next_interp_dt
        , old = "Contribution"
        , new = paste("Class", i - 1L)
      )
    }

    tree_interpretation[[i]] <- next_interp_dt

  }

  if (num_class == 1L) {

    tree_interpretation_dt <- tree_interpretation[[1L]]

  } else {

    # Full interpretation elements
    tree_interpretation_dt <- Reduce(
      f = function(x, y) {
        merge(x, y, by = "Feature", all = TRUE)
      }
      , x = tree_interpretation
    )

    # Loop throughout each tree
    for (j in 2L:ncol(tree_interpretation_dt)) {

      data.table::set(
        x = tree_interpretation_dt
        , i = which(is.na(tree_interpretation_dt[[j]]))
        , j = j
        , value = 0.0
      )

    }

  }

  return(tree_interpretation_dt)
}
