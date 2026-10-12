# constants that control naming in lists
.EVAL_KEY <- function() {
  return("eval")
}
.EVAL_ERR_KEY <- function() {
  return("eval_err")
}

#' @importFrom R6 R6Class
CB_ENV <- R6::R6Class(
  "lgb.cb_env",
  cloneable = FALSE,
  public = list(
    model = NULL,
    iteration = NULL,
    begin_iteration = NULL,
    end_iteration = NULL,
    eval_list = list(),
    eval_err_list = list(),
    best_iter = -1L,
    best_score = NA,
    met_early_stop = FALSE
  )
)

# Format the evaluation metric string
.format_eval_string <- function(eval_res, eval_err) {

  # Check for empty evaluation string
  if (is.null(eval_res) || length(eval_res) == 0L) {
    stop("no evaluation results")
  }

  # Check for empty evaluation error
  if (!is.null(eval_err)) {
    return(sprintf("%s\'s %s:%g+%g", eval_res$data_name, eval_res$name, eval_res$value, eval_err))
  } else {
    return(sprintf("%s\'s %s:%g", eval_res$data_name, eval_res$name, eval_res$value))
  }

}

.merge_eval_string <- function(env) {

  # Check length of evaluation list
  if (length(env$eval_list) <= 0L) {
    return("")
  }

  # Get evaluation
  msg <- list(sprintf("[%d]:", env$iteration))

  # Set if evaluation error
  is_eval_err <- length(env$eval_err_list) > 0L

  # Loop through evaluation list
  for (j in seq_along(env$eval_list)) {

    # Store evaluation error
    eval_err <- NULL
    if (isTRUE(is_eval_err)) {
      eval_err <- env$eval_err_list[[j]]
    }

    # Set error message
    msg <- c(msg, .format_eval_string(eval_res = env$eval_list[[j]], eval_err = eval_err))

  }

  return(paste(msg, collapse = "  "))

}

cb_print_evaluation <- function(period) {

  # Create callback
  callback <- function(env) {

    # Check if period is at least 1 or more
    if (period > 0L) {

      # Store iteration
      i <- env$iteration

      # Check if iteration matches moduo
      if ((i - 1L) %% period == 0L || is.element(i, c(env$begin_iteration, env$end_iteration))) {

        # Merge evaluation string
        msg <- .merge_eval_string(env = env)

        # Check if message is existing
        if (nchar(msg) > 0L) {
          cat(.merge_eval_string(env = env), "\n")
        }

      }

    }

    return(invisible(NULL))

  }

  # Store attributes
  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb_print_evaluation"

  return(callback)

}

cb_record_evaluation <- function() {

  # Create callback
  callback <- function(env) {

    if (length(env$eval_list) <= 0L) {
      return()
    }

    # Set if evaluation error
    is_eval_err <- length(env$eval_err_list) > 0L

    # Check length of recorded evaluation
    if (length(env$model$record_evals) == 0L) {

      # Loop through each evaluation list element
      for (j in seq_along(env$eval_list)) {

        # Store names
        data_name <- env$eval_list[[j]]$data_name
        name <- env$eval_list[[j]]$name
        env$model$record_evals$start_iter <- env$begin_iteration

        # Check if evaluation record exists
        if (is.null(env$model$record_evals[[data_name]])) {
          env$model$record_evals[[data_name]] <- list()
        }

        # Create dummy lists
        env$model$record_evals[[data_name]][[name]] <- list()
        env$model$record_evals[[data_name]][[name]][[.EVAL_KEY()]] <- list()
        env$model$record_evals[[data_name]][[name]][[.EVAL_ERR_KEY()]] <- list()

      }

    }

    # Loop through each evaluation list element
    for (j in seq_along(env$eval_list)) {

      # Get evaluation data
      eval_res <- env$eval_list[[j]]
      eval_err <- NULL
      if (isTRUE(is_eval_err)) {
        eval_err <- env$eval_err_list[[j]]
      }

      # Store names
      data_name <- eval_res$data_name
      name <- eval_res$name

      # Store evaluation data
      env$model$record_evals[[data_name]][[name]][[.EVAL_KEY()]] <- c(
        env$model$record_evals[[data_name]][[name]][[.EVAL_KEY()]]
        , eval_res$value
      )
      env$model$record_evals[[data_name]][[name]][[.EVAL_ERR_KEY()]] <- c(
        env$model$record_evals[[data_name]][[name]][[.EVAL_ERR_KEY()]]
        , eval_err
      )

    }

    return(invisible(NULL))

  }

  # Store attributes
  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb_record_evaluation"

  return(callback)

}

cb_early_stop <- function(stopping_rounds, first_metric_only, verbose) {

  cb_state <- new.env()
  cb_state[["best_iter"]] <- NULL
  cb_state[["best_msg"]] <- NULL
  cb_state[["best_score"]] <- NULL
  cb_state[["eval_len"]] <- NULL
  cb_state[["factor_to_bigger_better"]] <- NULL

  # Initialization function
  init <- function(env) {

    # Early stopping cannot work without metrics
    if (length(env$eval_list) == 0L) {
      stop("For early stopping, valids must have at least one element")
    }

    # Store evaluation length
    cb_state[["eval_len"]] <- length(env$eval_list)

    # Check if verbose or not
    if (isTRUE(verbose)) {
      msg <- paste0(
        "Will train until there is no improvement in "
        , stopping_rounds
        , " rounds.\n"
      )
      cat(msg)
    }

    # Internally treat everything as a maximization task
    cb_state[["factor_to_bigger_better"]] <- rep.int(1.0, cb_state[["eval_len"]])
    cb_state[["best_iter"]] <- rep.int(-1L, cb_state[["eval_len"]])
    cb_state[["best_score"]] <- rep.int(-Inf, cb_state[["eval_len"]])
    cb_state[["best_msg"]] <- list()

    # Loop through evaluation elements
    for (i in seq_len(cb_state[["eval_len"]])) {

      # Prepend message
      cb_state[["best_msg"]] <- c(cb_state[["best_msg"]], "")

      # Internally treat everything as a maximization task
      if (!isTRUE(env$eval_list[[i]]$higher_better)) {
        cb_state[["factor_to_bigger_better"]][i] <- -1.0
      }

    }

    return(invisible(NULL))

  }

  # Create callback
  callback <- function(env) {

    # Check for empty evaluation
    if (is.null(cb_state[["eval_len"]])) {
      init(env = env)
    }

    # Store iteration
    cur_iter <- env$iteration

    # By default, any metric can trigger early stopping. This can be disabled
    # with 'first_metric_only = TRUE'
    if (isTRUE(first_metric_only)) {
      evals_to_check <- 1L
    } else {
      evals_to_check <- seq_len(cb_state[["eval_len"]])
    }

    # Loop through evaluation
    for (i in evals_to_check) {

      # Store score
      score <- env$eval_list[[i]]$value * cb_state[["factor_to_bigger_better"]][i]

        # Check if score is better
        if (score > cb_state[["best_score"]][i]) {

          # Store new scores
          cb_state[["best_score"]][i] <- score
          cb_state[["best_iter"]][i] <- cur_iter

          # Prepare to print if verbose
          if (verbose) {
              cb_state[["best_msg"]][[i]] <- as.character(.merge_eval_string(env = env))
          }

        } else {

          # Check if early stopping is required
          if (cur_iter - cb_state[["best_iter"]][i] >= stopping_rounds) {

            if (!is.null(env$model)) {
              env$model$best_score <- cb_state[["best_score"]][i]
              env$model$best_iter <- cb_state[["best_iter"]][i]
            }

            if (isTRUE(verbose)) {
              cat(paste0("Early stopping, best iteration is: ", cb_state[["best_msg"]][[i]], "\n"))
            }

            # Store best iteration and stop
            env$best_iter <- cb_state[["best_iter"]][i]
            env$met_early_stop <- TRUE
          }

        }

      if (!isTRUE(env$met_early_stop) && cur_iter == env$end_iteration) {

        if (!is.null(env$model)) {
          env$model$best_score <- cb_state[["best_score"]][i]
          env$model$best_iter <- cb_state[["best_iter"]][i]
        }

        if (isTRUE(verbose)) {
          cat(paste0("Did not meet early stopping, best iteration is: ", cb_state[["best_msg"]][[i]], "\n"))
        }

        # Store best iteration and stop
        env$best_iter <- cb_state[["best_iter"]][i]
        env$met_early_stop <- TRUE
      }
    }

    return(invisible(NULL))

  }

  attr(callback, "call") <- match.call()
  attr(callback, "name") <- "cb_early_stop"

  return(callback)

}

# Extract callback names from the list of callbacks
.callback_names <- function(cb_list) {
  return(unlist(lapply(cb_list, attr, "name")))
}

.add_cb <- function(cb_list, cb) {

  # Combine two elements
  cb_list <- c(cb_list, cb)

  # Set names of elements
  names(cb_list) <- .callback_names(cb_list = cb_list)

  if ("cb_early_stop" %in% names(cb_list)) {

    # Concatenate existing elements
    cb_list <- c(cb_list, cb_list["cb_early_stop"])

    # Remove only the first one
    cb_list["cb_early_stop"] <- NULL

  }

  return(cb_list)

}

.categorize_callbacks <- function(cb_list) {

  # Check for pre-iteration or post-iteration
  return(
    list(
      pre_iter = Filter(function(x) {
        pre <- attr(x, "is_pre_iteration")
        !is.null(pre) && pre
      }, cb_list),
      post_iter = Filter(function(x) {
        pre <- attr(x, "is_pre_iteration")
        is.null(pre) || !pre
      }, cb_list)
    )
  )

}
