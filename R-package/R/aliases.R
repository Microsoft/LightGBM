# Central location for parameter aliases.
# See https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters

# [description] List of respected parameter aliases specific to lgb.Dataset. Wrapped in a function to
#               take advantage of lazy evaluation (so it doesn't matter what order
#               R sources files during installation).
# [return] A named list, where each key is a parameter relevant to lgb.DataSet and each value is a character
#          vector of corresponding aliases.
.DATASET_PARAMETERS <- function() {
    return(
        list(
            "bin_construct_sample_cnt" = c(
                "bin_construct_sample_cnt"
                , "subsample_for_bin"
            )
            , "categorical_feature" = c(
                "categorical_feature"
                , "cat_feature"
                , "categorical_column"
                , "cat_column"
            )
            , "data_random_seed" = c(
                "data_random_seed"
                , "data_seed"
            )
            , "enable_bundle" = c(
                "enable_bundle"
                , "is_enable_bundle"
                , "bundle"
            )
            , "feature_pre_filter" = "feature_pre_filter"
            , "forcedbins_filename" = "forcedbins_filename"
            , "group_column" = c(
                "group_column"
                , "group"
                , "group_id"
                , "query_column"
                , "query"
                , "query_id"
            )
            , "header" = c(
                "header"
                , "has_header"
            )
            , "ignore_column" = c(
                "ignore_column"
                , "ignore_feature"
                , "blacklist"
            )
            , "is_enable_sparse" = c(
                "is_enable_sparse"
                , "is_sparse"
                , "enable_sparse"
                , "sparse"
            )
            , "label_column" = c(
                "label_column"
                , "label"
            )
            , "max_bin" = "max_bin"
            , "max_bin_by_feature" = "max_bin_by_feature"
            , "min_data_in_bin" = "min_data_in_bin"
            , "pre_partition" = c(
                "pre_partition"
                , "is_pre_partition"
            )
            , "two_round" = c(
                "two_round"
                , "two_round_loading"
                , "use_two_round_loading"
            )
            , "use_missing" = "use_missing"
            , "weight_column" = c(
                "weight_column"
                , "weight"
            )
            , "zero_as_missing" = "zero_as_missing"
            , "category_encoders" = "category_encoders"
        )
    )
}

# [description] List of respected parameter aliases. Wrapped in a function to take advantage of
#               lazy evaluation (so it doesn't matter what order R sources files during installation).
# [return] A named list, where each key is a main LightGBM parameter and each value is a character
#          vector of corresponding aliases.
.PARAMETER_ALIASES <- function() {
    learning_params <- list(
        "boosting" = c(
            "boosting"
            , "boost"
            , "boosting_type"
        )
        , "early_stopping_round" = c(
            "early_stopping_round"
            , "early_stopping_rounds"
            , "early_stopping"
            , "n_iter_no_change"
        )
        , "num_iterations" = c(
            "num_iterations"
            , "num_iteration"
            , "n_iter"
            , "num_tree"
            , "num_trees"
            , "num_round"
            , "num_rounds"
            , "num_boost_round"
            , "n_estimators"
        )
    )
    return(c(learning_params, .DATASET_PARAMETERS()))
}

# [description]
#     Per https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst#metric,
#     a few different strings can be used to indicate "no metrics".
# [returns]
#     A character vector
.NO_METRIC_STRINGS <- function() {
    return(
        c(
            "na"
            , "None"
            , "null"
            , "custom"
        )
    )
}
