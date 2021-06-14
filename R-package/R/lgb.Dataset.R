#' @importFrom methods is
#' @importFrom R6 R6Class
Dataset <- R6::R6Class(

  classname = "lgb.Dataset",
  cloneable = FALSE,
  public = list(

    # Finalize will free up the handles
    finalize = function() {
      .Call(
        LGBM_DatasetFree_R
        , private$handle
      )
      private$handle <- NULL
      return(invisible(NULL))
    },

    # Initialize will create a starter dataset
    initialize = function(data,
                          params = list(),
                          reference = NULL,
                          colnames = NULL,
                          categorical_feature = NULL,
                          predictor = NULL,
                          free_raw_data = TRUE,
                          used_indices = NULL,
                          info = list(),
                          ...) {

      # validate inputs early to avoid unnecessary computation
      if (!(is.null(reference) || lgb.is.Dataset(reference))) {
          stop("lgb.Dataset: If provided, reference must be a ", sQuote("lgb.Dataset"))
      }
      if (!(is.null(predictor) || lgb.is.Predictor(predictor))) {
          stop("lgb.Dataset: If provided, predictor must be a ", sQuote("lgb.Predictor"))
      }

      # Check for additional parameters
      additional_params <- list(...)

      # Create known attributes list
      INFO_KEYS <- c("label", "weight", "init_score", "group")

      # Check if attribute key is in the known attribute list
      for (key in names(additional_params)) {

        # Key existing
        if (key %in% INFO_KEYS) {

          # Store as info
          info[[key]] <- additional_params[[key]]

        } else {

          # Store as param
          params[[key]] <- additional_params[[key]]

        }

      }

      # Check for matrix format
      if (is.matrix(data)) {
        # Check whether matrix is the correct type first ("double")
        if (storage.mode(data) != "double") {
          storage.mode(data) <- "double"
        }
      }

      # Setup private attributes
      private$raw_data <- data
      private$params <- params
      private$reference <- reference
      private$colnames <- colnames

      private$categorical_feature <- categorical_feature
      private$predictor <- predictor
      private$free_raw_data <- free_raw_data
      private$used_indices <- sort(used_indices, decreasing = FALSE)
      private$info <- info
      private$version <- 0L

      return(invisible(NULL))

    },

    create_valid = function(data,
                            info = list(),
                            ...) {

      # Create new dataset
      ret <- Dataset$new(
        data = data
        , params = private$params
        , reference = self
        , colnames = private$colnames
        , categorical_feature = private$categorical_feature
        , predictor = private$predictor
        , free_raw_data = private$free_raw_data
        , used_indices = NULL
        , info = info
        , ...
      )

      return(invisible(ret))

    },

    # Dataset constructor
    construct = function() {

      # Check for handle null
      if (!lgb.is.null.handle(x = private$handle)) {
        return(invisible(self))
      }

      # Get feature names
      cnames <- NULL
      if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {
        cnames <- colnames(private$raw_data)
      }

      # set feature names if they do not exist
      if (is.null(private$colnames) && !is.null(cnames)) {
        private$colnames <- as.character(cnames)
      }

      # Get categorical feature index
      if (!is.null(private$categorical_feature)) {

        # Check for character name
        if (is.character(private$categorical_feature)) {

            cate_indices <- as.list(match(private$categorical_feature, private$colnames) - 1L)

            # Provided indices, but some indices are missing?
            if (sum(is.na(cate_indices)) > 0L) {
              stop(
                "lgb.self.get.handle: supplied an unknown feature in categorical_feature: "
                , sQuote(private$categorical_feature[is.na(cate_indices)])
              )
            }

          } else {
            # by pass this test when colnames is not set
            if (!is.null(private$colnames) && max(private$categorical_feature) > length(private$colnames)) {
              stop(
                "lgb.self.get.handle: supplied a too large value in categorical_feature: "
                , max(private$categorical_feature)
                , " but only "
                , length(private$colnames)
                , " features"
              )
            }

            # Store indices as [0, n-1] indexed instead of [1, n] indexed
            cate_indices <- as.list(private$categorical_feature - 1L)

          }

        # Store indices for categorical features
        private$params$categorical_feature <- cate_indices

      }

      # Check has header or not
      has_header <- FALSE
      if (!is.null(private$params$has_header) || !is.null(private$params$header)) {
        params_has_header <- tolower(as.character(private$params$has_header)) == "true"
        params_header <- tolower(as.character(private$params$header)) == "true"
        if (params_has_header || params_header) {
          has_header <- TRUE
        }
      }

      # Generate parameter str
      params_str <- lgb.params2str(params = private$params)

      # Get handle of reference dataset
      ref_handle <- NULL
      if (!is.null(private$reference)) {
        ref_handle <- private$reference$.__enclos_env__$private$get_handle()
      }

      # Not subsetting
      if (is.null(private$used_indices)) {

        # Are we using a data file?
        if (is.character(private$raw_data)) {

          handle <- .Call(
            LGBM_DatasetCreateFromFile_R
            , private$raw_data
            , params_str
            , ref_handle
          )

        } else if (is.matrix(private$raw_data)) {
          if (is.null(private$info[["label"]])) {
            label <- NULL
          } else {
            label <- as.numeric(private$info[["label"]])
          }
          # Are we using a matrix?
          handle <- .Call(
            LGBM_DatasetCreateFromMat_R
            , private$raw_data
            , label
            , nrow(private$raw_data)
            , ncol(private$raw_data)
            , params_str
            , ref_handle
          )

        } else if (methods::is(private$raw_data, "dgCMatrix")) {
          if (is.null(private$info[["label"]])) {
            label <- NULL
          } else {
            label <- as.numeric(private$info[["label"]])
          }
          if (length(private$raw_data@p) > 2147483647L) {
            stop("Cannot support large CSC matrix")
          }
          # Are we using a dgCMatrix (sparsed matrix column compressed)
          handle <- .Call(
            LGBM_DatasetCreateFromCSC_R
            , private$raw_data@p
            , private$raw_data@i
            , private$raw_data@x
            , label
            , length(private$raw_data@p)
            , length(private$raw_data@x)
            , nrow(private$raw_data)
            , params_str
            , ref_handle
          )

        } else {

          # Unknown data type
          stop(
            "lgb.Dataset.construct: does not support constructing from "
            , sQuote(class(private$raw_data))
          )

        }

      } else {

        # Reference is empty
        if (is.null(private$reference)) {
          stop("lgb.Dataset.construct: reference cannot be NULL for constructing data subset")
        }

        # Construct subset
        handle <- .Call(
          LGBM_DatasetGetSubset_R
          , ref_handle
          , c(private$used_indices) # Adding c() fixes issue in R v3.5
          , length(private$used_indices)
          , params_str
        )

      }
      if (lgb.is.null.handle(x = handle)) {
        stop("lgb.Dataset.construct: cannot create Dataset handle")
      }
      # Setup class and private type
      class(handle) <- "lgb.Dataset.handle"
      private$handle <- handle

      # Set feature names
      if (!is.null(private$colnames)) {
        self$set_colnames(colnames = private$colnames)
      }

      # Load init score if requested
      if (!is.null(private$predictor) && is.null(private$used_indices)) {

        # Setup initial scores
        init_score <- private$predictor$predict(
          data = private$raw_data
          , rawscore = TRUE
          , reshape = TRUE
        )

        # Not needed to transpose, for is col_marjor
        init_score <- as.vector(init_score)
        private$info$init_score <- init_score

      }

      # Should we free raw data?
      if (isTRUE(private$free_raw_data)) {
        private$raw_data <- NULL
      }

      # Get private information
      if (length(private$info) > 0L) {

        # Set infos
        for (i in seq_along(private$info)) {

          p <- private$info[i]
          self$setinfo(name = names(p), info = p[[1L]])

        }

      }

      # Get label information existence
      if (is.null(self$getinfo(name = "label"))) {
        stop("lgb.Dataset.construct: label should be set")
      }

      return(invisible(self))

    },

    # Dimension function
    dim = function() {

      # Check for handle
      if (!lgb.is.null.handle(x = private$handle)) {

        num_row <- 0L
        num_col <- 0L

        # Get numeric data and numeric features
        .Call(
          LGBM_DatasetGetNumData_R
          , private$handle
          , num_row
        )
        .Call(
          LGBM_DatasetGetNumFeature_R
          , private$handle
          , num_col
        )
        return(
          c(num_row, num_col)
        )

      } else if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {

        # Check if dgCMatrix (sparse matrix column compressed)
        # NOTE: requires Matrix package
        return(dim(private$raw_data))

      } else {

        # Trying to work with unknown dimensions is not possible
        stop(
          "dim: cannot get dimensions before dataset has been constructed, "
          , "please call lgb.Dataset.construct explicitly"
        )

      }

    },

    # Get column names
    get_colnames = function() {

      # Check for handle
      if (!lgb.is.null.handle(x = private$handle)) {
        private$colnames <- .Call(
          LGBM_DatasetGetFeatureNames_R
          , private$handle
        )
        return(private$colnames)

      } else if (is.matrix(private$raw_data) || methods::is(private$raw_data, "dgCMatrix")) {

        # Check if dgCMatrix (sparse matrix column compressed)
        return(colnames(private$raw_data))

      } else {

        # Trying to work with unknown dimensions is not possible
        stop(
          "dim: cannot get dimensions before dataset has been constructed, please call "
          , "lgb.Dataset.construct explicitly"
        )

      }

    },

    # Set column names
    set_colnames = function(colnames) {

      # Check column names non-existence
      if (is.null(colnames)) {
        return(invisible(self))
      }

      # Check empty column names
      colnames <- as.character(colnames)
      if (length(colnames) == 0L) {
        return(invisible(self))
      }

      # Write column names
      private$colnames <- colnames
      if (!lgb.is.null.handle(x = private$handle)) {

        # Merge names with tab separation
        merged_name <- paste0(as.list(private$colnames), collapse = "\t")
        .Call(
          LGBM_DatasetSetFeatureNames_R
          , private$handle
          , merged_name
        )

      }

      return(invisible(self))

    },

    # Get information
    getinfo = function(name) {

      # Create known attributes list
      INFONAMES <- c("label", "weight", "init_score", "group")

      # Check if attribute key is in the known attribute list
      if (!is.character(name) || length(name) != 1L || !name %in% INFONAMES) {
        stop("getinfo: name must one of the following: ", paste0(sQuote(INFONAMES), collapse = ", "))
      }

      # Check for info name and handle
      if (is.null(private$info[[name]])) {

        if (lgb.is.null.handle(x = private$handle)) {
          stop("Cannot perform getinfo before constructing Dataset.")
        }

        # Get field size of info
        info_len <- 0L
        .Call(
          LGBM_DatasetGetFieldSize_R
          , private$handle
          , name
          , info_len
        )

        # Check if info is not empty
        if (info_len > 0L) {

          # Get back fields
          ret <- NULL
          ret <- if (name == "group") {
            integer(info_len) # Integer
          } else {
            numeric(info_len) # Numeric
          }

          .Call(
            LGBM_DatasetGetField_R
            , private$handle
            , name
            , ret
          )

          private$info[[name]] <- ret

        }
      }

      return(private$info[[name]])

    },

    # Set information
    setinfo = function(name, info) {

      # Create known attributes list
      INFONAMES <- c("label", "weight", "init_score", "group")

      # Check if attribute key is in the known attribute list
      if (!is.character(name) || length(name) != 1L || !name %in% INFONAMES) {
        stop("setinfo: name must one of the following: ", paste0(sQuote(INFONAMES), collapse = ", "))
      }

      # Check for type of information
      info <- if (name == "group") {
        as.integer(info) # Integer
      } else {
        as.numeric(info) # Numeric
      }

      # Store information privately
      private$info[[name]] <- info

      if (!lgb.is.null.handle(x = private$handle) && !is.null(info)) {

        if (length(info) > 0L) {

          .Call(
            LGBM_DatasetSetField_R
            , private$handle
            , name
            , info
            , length(info)
          )

          private$version <- private$version + 1L

        }

      }

      return(invisible(self))

    },

    # Slice dataset
    slice = function(idxset, ...) {

      # Perform slicing
      return(
        Dataset$new(
          data = NULL
          , params = private$params
          , reference = self
          , colnames = private$colnames
          , categorical_feature = private$categorical_feature
          , predictor = private$predictor
          , free_raw_data = private$free_raw_data
          , used_indices = sort(idxset, decreasing = FALSE)
          , info = NULL
          , ...
        )
      )

    },

    # [description] Update Dataset parameters. If it has not been constructed yet,
    #               this operation just happens on the R side (updating private$params).
    #               If it has been constructed, parameters will be updated on the C++ side.
    update_params = function(params) {
      if (length(params) == 0L) {
        return(invisible(self))
      }
      if (!is.null(params$categorical_feature)) {
        # when categorical feature is passed in the constructor of lgb.Dataset
        # all indices are subtracted by 1 in construct().
        # however, when it is passed directly to the C API, it used the original indices.
        # this causes unexpected inconsistent results, so we remove
        # categorical features from the parameters and directly set it in R.
        self$set_categorical_feature(params$categorical_feature)
        params$categorical_feature <- NULL
      }
      if (lgb.is.null.handle(x = private$handle)) {
        private$params <- modifyList(private$params, params)
      } else {
        tryCatch({
          .Call(
            LGBM_DatasetUpdateParamChecking_R
            , lgb.params2str(params = private$params)
            , lgb.params2str(params = params)
          )
        }, error = function(e) {
          # If updating failed but raw data is not available, raise an error because
          # achieving what the user asked for is not possible
          if (is.null(private$raw_data)) {
            stop(e)
          }

          # If updating failed but raw data is available, modify the params
          # on the R side and re-set ("deconstruct") the Dataset
          private$params <- modifyList(private$params, params)
          self$finalize()
        })
      }
      return(invisible(self))

    },

    get_params = function() {
      dataset_params <- unname(unlist(.DATASET_PARAMETERS()))
      ret <- list()
      for (param_key in names(private$params)) {
        if (param_key %in% dataset_params) {
          ret[[param_key]] <- private$params[[param_key]]
        }
      }
      return(ret)
    },

    # Set categorical feature parameter
    set_categorical_feature = function(categorical_feature) {

      # Check for identical input
      if (identical(private$categorical_feature, categorical_feature)) {
        return(invisible(self))
      }

      # Check for empty data
      if (is.null(private$raw_data)) {
        stop("set_categorical_feature: cannot set categorical feature after freeing raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset")
      }

      # Overwrite categorical features
      private$categorical_feature <- categorical_feature

      # Finalize and return self
      self$finalize()
      return(invisible(self))

    },

    # Set reference
    set_reference = function(reference) {

      # Set known references
      self$set_categorical_feature(categorical_feature = reference$.__enclos_env__$private$categorical_feature)
      self$set_colnames(colnames = reference$get_colnames())
      private$set_predictor(predictor = reference$.__enclos_env__$private$predictor)

      # Check for identical references
      if (identical(private$reference, reference)) {
        return(invisible(self))
      }

      # Check for empty data
      if (is.null(private$raw_data)) {

        stop("set_reference: cannot set reference after freeing raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset")

      }

      # Check for non-existing reference
      if (!is.null(reference)) {

        # Reference is unknown
        if (!lgb.is.Dataset(reference)) {
          stop("set_reference: Can only use lgb.Dataset as a reference")
        }

      }

      # Store reference
      private$reference <- reference

      # Finalize and return self
      self$finalize()
      return(invisible(self))

    },

    # Save binary model
    save_binary = function(fname) {

      # Store binary data
      self$construct()
      .Call(
        LGBM_DatasetSaveBinary_R
        , private$handle
        , fname
      )
      return(invisible(self))
    }

  ),
  private = list(
    handle = NULL,
    raw_data = NULL,
    params = list(),
    reference = NULL,
    colnames = NULL,
    categorical_feature = NULL,
    predictor = NULL,
    free_raw_data = TRUE,
    used_indices = NULL,
    info = NULL,
    version = 0L,

    # Get handle
    get_handle = function() {

      # Get handle and construct if needed
      if (lgb.is.null.handle(x = private$handle)) {
        self$construct()
      }
      return(private$handle)

    },

    # Set predictor
    set_predictor = function(predictor) {

      if (identical(private$predictor, predictor)) {
        return(invisible(self))
      }

      # Check for empty data
      if (is.null(private$raw_data)) {
        stop("set_predictor: cannot set predictor after free raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset")
      }

      # Check for empty predictor
      if (!is.null(predictor)) {

        # Predictor is unknown
        if (!lgb.is.Predictor(predictor)) {
          stop("set_predictor: Can only use lgb.Predictor as predictor")
        }

      }

      # Store predictor
      private$predictor <- predictor

      # Finalize and return self
      self$finalize()
      return(invisible(self))

    }

  )
)

#' @title Construct \code{lgb.Dataset} object
#' @description Construct \code{lgb.Dataset} object from dense matrix, sparse matrix
#'              or local file (that was created previously by saving an \code{lgb.Dataset}).
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param params a list of parameters. See
#'               \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#dataset-parameters}{
#'               The "Dataset Parameters" section of the documentation} for a list of parameters
#'               and valid values.
#' @param reference reference dataset. When LightGBM creates a Dataset, it does some preprocessing like binning
#'                  continuous features into histograms. If you want to apply the same bin boundaries from an existing
#'                  dataset to new \code{data}, pass that existing Dataset to this argument.
#' @param colnames names of columns
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#' @param free_raw_data LightGBM constructs its data format, called a "Dataset", from tabular data.
#'                      By default, that Dataset object on the R side does not keep a copy of the raw data.
#'                      This reduces LightGBM's memory consumption, but it means that the Dataset object
#'                      cannot be changed after it has been constructed. If you'd prefer to be able to
#'                      change the Dataset object after construction, set \code{free_raw_data = FALSE}.
#' @param info a list of information of the \code{lgb.Dataset} object
#' @param ... other information to pass to \code{info} or parameters pass to \code{params}
#'
#' @return constructed dataset
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data_file <- tempfile(fileext = ".data")
#' lgb.Dataset.save(dtrain, data_file)
#' dtrain <- lgb.Dataset(data_file)
#' lgb.Dataset.construct(dtrain)
#' }
#' @export
lgb.Dataset <- function(data,
                        params = list(),
                        reference = NULL,
                        colnames = NULL,
                        categorical_feature = NULL,
                        free_raw_data = TRUE,
                        info = list(),
                        ...) {

  # Create new dataset
  return(
    invisible(Dataset$new(
      data = data
      , params = params
      , reference = reference
      , colnames = colnames
      , categorical_feature = categorical_feature
      , predictor = NULL
      , free_raw_data = free_raw_data
      , used_indices = NULL
      , info = info
      , ...
    ))
  )

}

#' @name lgb.Dataset.create.valid
#' @title Construct validation data
#' @description Construct validation data according to training data
#' @param dataset \code{lgb.Dataset} object, training data
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param info a list of information of the \code{lgb.Dataset} object
#' @param ... other information to pass to \code{info}.
#'
#' @return constructed dataset
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' }
#' @export
lgb.Dataset.create.valid <- function(dataset, data, info = list(), ...) {

  # Check if dataset is not a dataset
  if (!lgb.is.Dataset(x = dataset)) {
    stop("lgb.Dataset.create.valid: input data should be an lgb.Dataset object")
  }

  # Create validation dataset
  return(invisible(dataset$create_valid(data = data, info = info, ...)))

}

#' @name lgb.Dataset.construct
#' @title Construct Dataset explicitly
#' @description Construct Dataset explicitly
#' @param dataset Object of class \code{lgb.Dataset}
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.construct(dtrain)
#' }
#' @return constructed dataset
#' @export
lgb.Dataset.construct <- function(dataset) {

  # Check if dataset is not a dataset
  if (!lgb.is.Dataset(x = dataset)) {
    stop("lgb.Dataset.construct: input data should be an lgb.Dataset object")
  }

  # Construct the dataset
  return(invisible(dataset$construct()))

}

#' @title Dimensions of an \code{lgb.Dataset}
#' @description Returns a vector of numbers of rows and of columns in an \code{lgb.Dataset}.
#' @param x Object of class \code{lgb.Dataset}
#' @param ... other parameters
#'
#' @return a vector of numbers of rows and of columns
#'
#' @details
#' Note: since \code{nrow} and \code{ncol} internally use \code{dim}, they can also
#' be directly used with an \code{lgb.Dataset} object.
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' stopifnot(ncol(dtrain) == ncol(train$data))
#' stopifnot(all(dim(dtrain) == dim(train$data)))
#' }
#' @rdname dim
#' @export
dim.lgb.Dataset <- function(x, ...) {

  # Check if dataset is not a dataset
  if (!lgb.is.Dataset(x = x)) {
    stop("dim.lgb.Dataset: input data should be an lgb.Dataset object")
  }

  return(x$dim())

}

#' @title Handling of column names of \code{lgb.Dataset}
#' @description Only column names are supported for \code{lgb.Dataset}, thus setting of
#'              row names would have no effect and returned row names would be NULL.
#' @param x object of class \code{lgb.Dataset}
#' @param value a list of two elements: the first one is ignored
#'              and the second one is column names
#'
#' @details
#' Generic \code{dimnames} methods are used by \code{colnames}.
#' Since row names are irrelevant, it is recommended to use \code{colnames} directly.
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.construct(dtrain)
#' dimnames(dtrain)
#' colnames(dtrain)
#' colnames(dtrain) <- make.names(seq_len(ncol(train$data)))
#' print(dtrain, verbose = TRUE)
#' }
#' @rdname dimnames.lgb.Dataset
#' @return A list with the dimension names of the dataset
#' @export
dimnames.lgb.Dataset <- function(x) {

  # Check if dataset is not a dataset
  if (!lgb.is.Dataset(x = x)) {
    stop("dimnames.lgb.Dataset: input data should be an lgb.Dataset object")
  }

  # Return dimension names
  return(list(NULL, x$get_colnames()))

}

#' @rdname dimnames.lgb.Dataset
#' @export
`dimnames<-.lgb.Dataset` <- function(x, value) {

  # Check if invalid element list
  if (!identical(class(value), "list") || length(value) != 2L) {
    stop("invalid ", sQuote("value"), " given: must be a list of two elements")
  }

  # Check for unknown row names
  if (!is.null(value[[1L]])) {
    stop("lgb.Dataset does not have rownames")
  }

  if (is.null(value[[2L]])) {

    x$set_colnames(colnames = NULL)
    return(x)

  }

  # Check for unmatching column size
  if (ncol(x) != length(value[[2L]])) {
    stop(
      "can't assign "
      , sQuote(length(value[[2L]]))
      , " colnames to an lgb.Dataset with "
      , sQuote(ncol(x))
      , " columns"
    )
  }

  # Set column names properly, and return
  x$set_colnames(colnames = value[[2L]])
  return(x)

}

#' @title Slice a dataset
#' @description Get a new \code{lgb.Dataset} containing the specified rows of
#'              original \code{lgb.Dataset} object
#' @param dataset Object of class \code{lgb.Dataset}
#' @param idxset an integer vector of indices of rows needed
#' @param ... other parameters (currently not used)
#' @return constructed sub dataset
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' dsub <- lightgbm::slice(dtrain, seq_len(42L))
#' lgb.Dataset.construct(dsub)
#' labels <- lightgbm::getinfo(dsub, "label")
#' }
#' @export
slice <- function(dataset, ...) {
  UseMethod("slice")
}

#' @rdname slice
#' @export
slice.lgb.Dataset <- function(dataset, idxset, ...) {

  # Check if dataset is not a dataset
  if (!lgb.is.Dataset(x = dataset)) {
    stop("slice.lgb.Dataset: input dataset should be an lgb.Dataset object")
  }

  # Return sliced set
  return(invisible(dataset$slice(idxset = idxset, ...)))

}

#' @name getinfo
#' @title Get information of an \code{lgb.Dataset} object
#' @description Get one attribute of a \code{lgb.Dataset}
#' @param dataset Object of class \code{lgb.Dataset}
#' @param name the name of the information field to get (see details)
#' @param ... other parameters
#' @return info data
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item \code{label}: label lightgbm learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item{\code{group}: used for learning-to-rank tasks. An integer vector describing how to
#'         group rows together as ordered results from the same set of candidate results to be ranked.
#'         For example, if you have a 100-document dataset with \code{group = c(10, 20, 40, 10, 10, 10)},
#'         that means that you have 6 groups, where the first 10 records are in the first group,
#'         records 11-30 are in the second group, etc.}
#'     \item \code{init_score}: initial score is the base prediction lightgbm will boost from.
#' }
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.construct(dtrain)
#'
#' labels <- lightgbm::getinfo(dtrain, "label")
#' lightgbm::setinfo(dtrain, "label", 1 - labels)
#'
#' labels2 <- lightgbm::getinfo(dtrain, "label")
#' stopifnot(all(labels2 == 1 - labels))
#' }
#' @export
getinfo <- function(dataset, ...) {
  UseMethod("getinfo")
}

#' @rdname getinfo
#' @export
getinfo.lgb.Dataset <- function(dataset, name, ...) {

  # Check if dataset is not a dataset
  if (!lgb.is.Dataset(x = dataset)) {
    stop("getinfo.lgb.Dataset: input dataset should be an lgb.Dataset object")
  }

  return(dataset$getinfo(name = name))

}

#' @name setinfo
#' @title Set information of an \code{lgb.Dataset} object
#' @description Set one attribute of a \code{lgb.Dataset}
#' @param dataset Object of class \code{lgb.Dataset}
#' @param name the name of the field to get
#' @param info the specific field of information to set
#' @param ... other parameters
#' @return the dataset you passed in
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item{\code{label}: vector of labels to use as the target variable}
#'     \item{\code{weight}: to do a weight rescale}
#'     \item{\code{init_score}: initial score is the base prediction lightgbm will boost from}
#'     \item{\code{group}: used for learning-to-rank tasks. An integer vector describing how to
#'         group rows together as ordered results from the same set of candidate results to be ranked.
#'         For example, if you have a 100-document dataset with \code{group = c(10, 20, 40, 10, 10, 10)},
#'         that means that you have 6 groups, where the first 10 records are in the first group,
#'         records 11-30 are in the second group, etc.}
#' }
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.construct(dtrain)
#'
#' labels <- lightgbm::getinfo(dtrain, "label")
#' lightgbm::setinfo(dtrain, "label", 1 - labels)
#'
#' labels2 <- lightgbm::getinfo(dtrain, "label")
#' stopifnot(all.equal(labels2, 1 - labels))
#' }
#' @export
setinfo <- function(dataset, ...) {
  UseMethod("setinfo")
}

#' @rdname setinfo
#' @export
setinfo.lgb.Dataset <- function(dataset, name, info, ...) {

  if (!lgb.is.Dataset(x = dataset)) {
    stop("setinfo.lgb.Dataset: input dataset should be an lgb.Dataset object")
  }

  # Set information
  return(invisible(dataset$setinfo(name = name, info = info)))
}

#' @name lgb.Dataset.set.categorical
#' @title Set categorical feature of \code{lgb.Dataset}
#' @description Set the categorical features of an \code{lgb.Dataset} object. Use this function
#'              to tell LightGBM which features should be treated as categorical.
#' @param dataset object of class \code{lgb.Dataset}
#' @param categorical_feature categorical features. This can either be a character vector of feature
#'                            names or an integer vector with the indices of the features (e.g.
#'                            \code{c(1L, 10L)} to say "the first and tenth columns").
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data_file <- tempfile(fileext = ".data")
#' lgb.Dataset.save(dtrain, data_file)
#' dtrain <- lgb.Dataset(data_file)
#' lgb.Dataset.set.categorical(dtrain, 1L:2L)
#' }
#' @rdname lgb.Dataset.set.categorical
#' @export
lgb.Dataset.set.categorical <- function(dataset, categorical_feature) {

  if (!lgb.is.Dataset(x = dataset)) {
    stop("lgb.Dataset.set.categorical: input dataset should be an lgb.Dataset object")
  }

  # Set categoricals
  return(invisible(dataset$set_categorical_feature(categorical_feature = categorical_feature)))

}

#' @name lgb.Dataset.set.reference
#' @title Set reference of \code{lgb.Dataset}
#' @description If you want to use validation data, you should set reference to training data
#' @param dataset object of class \code{lgb.Dataset}
#' @param reference object of class \code{lgb.Dataset}
#'
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package ="lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset(test$data, test = train$label)
#' lgb.Dataset.set.reference(dtest, dtrain)
#' }
#' @rdname lgb.Dataset.set.reference
#' @export
lgb.Dataset.set.reference <- function(dataset, reference) {

  # Check if dataset is not a dataset
  if (!lgb.is.Dataset(x = dataset)) {
    stop("lgb.Dataset.set.reference: input dataset should be an lgb.Dataset object")
  }

  # Set reference
  return(invisible(dataset$set_reference(reference = reference)))
}

#' @name lgb.Dataset.save
#' @title Save \code{lgb.Dataset} to a binary file
#' @description Please note that \code{init_score} is not saved in binary file.
#'              If you need it, please set it again after loading Dataset.
#' @param dataset object of class \code{lgb.Dataset}
#' @param fname object filename of output file
#'
#' @return the dataset you passed in
#'
#' @examples
#' \donttest{
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' lgb.Dataset.save(dtrain, tempfile(fileext = ".bin"))
#' }
#' @export
lgb.Dataset.save <- function(dataset, fname) {

  # Check if dataset is not a dataset
  if (!lgb.is.Dataset(x = dataset)) {
    stop("lgb.Dataset.set: input dataset should be an lgb.Dataset object")
  }

  # File-type is not matching
  if (!is.character(fname)) {
    stop("lgb.Dataset.set: fname should be a character or a file connection")
  }

  # Store binary
  return(invisible(dataset$save_binary(fname = fname)))
}
