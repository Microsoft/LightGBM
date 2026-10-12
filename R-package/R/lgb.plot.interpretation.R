#' @name lgb.plot.interpretation
#' @title Plot feature contribution as a bar graph
#' @description Plot previously calculated feature contribution as a bar graph.
#' @param tree_interpretation_dt a \code{data.table} returned by \code{\link{lgb.interpret}}.
#' @param top_n maximal number of top features to include into the plot.
#' @param cols the column numbers of layout, will be used only for multiclass classification feature contribution.
#' @param left_margin (base R barplot) allows to adjust the left margin size to fit feature names.
#' @param cex (base R barplot) passed as \code{cex.names} parameter to \code{barplot}.
#'
#' @details
#' The graph represents each feature as a horizontal bar of length proportional to the defined
#' contribution of a feature. Features are shown ranked in a decreasing contribution order.
#'
#' @return
#' The \code{lgb.plot.interpretation} function creates a \code{barplot}.
#'
#' @examples
#' \donttest{
#' \dontshow{setLGBMthreads(2L)}
#' \dontshow{data.table::setDTthreads(1L)}
#' Logit <- function(x) {
#'   log(x / (1.0 - x))
#' }
#' data(agaricus.train, package = "lightgbm")
#' labels <- agaricus.train$label
#' dtrain <- lgb.Dataset(
#'   agaricus.train$data
#'   , label = labels
#' )
#' set_field(
#'   dataset = dtrain
#'   , field_name = "init_score"
#'   , data = rep(Logit(mean(labels)), length(labels))
#' )
#'
#' data(agaricus.test, package = "lightgbm")
#'
#' params <- list(
#'   objective = "binary"
#'   , learning_rate = 0.1
#'   , max_depth = -1L
#'   , min_data_in_leaf = 1L
#'   , min_sum_hessian_in_leaf = 1.0
#'   , num_threads = 2L
#' )
#' model <- lgb.train(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 5L
#' )
#'
#' tree_interpretation <- lgb.interpret(
#'   model = model
#'   , data = agaricus.test$data
#'   , idxset = 1L:5L
#' )
#' lgb.plot.interpretation(
#'   tree_interpretation_dt = tree_interpretation[[1L]]
#'   , top_n = 3L
#' )
#' }
#' @importFrom data.table setnames
#' @importFrom graphics barplot par
#' @export
lgb.plot.interpretation <- function(tree_interpretation_dt,
                                    top_n = 10L,
                                    cols = 1L,
                                    left_margin = 10L,
                                    cex = NULL) {

  num_class <- ncol(tree_interpretation_dt) - 1L

  # Refresh plot
  op <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(op))

  # Do some magic plotting
  bottom_margin <- 3.0
  top_margin <- 2.0
  right_margin <- op$mar[4L]

  graphics::par(
    mar = c(
      bottom_margin
      , left_margin
      , top_margin
      , right_margin
    )
  )

  if (num_class == 1L) {

    # Only one class, plot straight away
    .multiple_tree_plot_interpretation(
      tree_interpretation = tree_interpretation_dt
      , top_n = top_n
      , title = NULL
      , cex = cex
    )

  } else {

    # More than one class, shape data first
    layout_mat <- matrix(
      seq.int(to = cols * ceiling(num_class / cols))
      , ncol = cols
      , nrow = ceiling(num_class / cols)
    )

    # Shape output
    graphics::par(mfcol = c(nrow(layout_mat), ncol(layout_mat)))

    # Loop throughout all classes
    for (i in seq_len(num_class)) {

      # Prepare interpretation, perform T, get the names, and plot straight away
      plot_dt <- tree_interpretation_dt[, c(1L, i + 1L), with = FALSE]
      data.table::setnames(
        x = plot_dt
        , old = names(plot_dt)
        , new = c("Feature", "Contribution")
      )
      .multiple_tree_plot_interpretation(
        tree_interpretation = plot_dt
        , top_n = top_n
        , title = paste("Class", i - 1L)
        , cex = cex
      )

    }
  }
  return(invisible(NULL))
}

#' @importFrom graphics barplot
.multiple_tree_plot_interpretation <- function(tree_interpretation,
                                              top_n,
                                              title,
                                              cex) {

  # Parse tree
  tree_interpretation <- tree_interpretation[order(abs(Contribution), decreasing = TRUE), ][seq_len(min(top_n, .N)), ]

  # Attempt to setup a correct cex
  if (is.null(cex)) {
    cex <- 2.5 / log2(1.0 + top_n)
  }

  # create plot
  tree_interpretation[abs(Contribution) > 0.0, bar_color := "firebrick"]
  tree_interpretation[Contribution == 0.0, bar_color := "steelblue"]
  tree_interpretation[rev(seq_len(.N)),
                      graphics::barplot(
                          height = Contribution
                          , names.arg = Feature
                          , horiz = TRUE
                          , col = bar_color
                          , border = NA
                          , main = title
                          , cex.names = cex
                          , las = 1L
                      )]

  return(invisible(NULL))

}
