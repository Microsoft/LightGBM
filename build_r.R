# For macOS users who have decided to use gcc
# (replace 8 with version of gcc installed on your machine)
# NOTE: your gcc / g++ from Homebrew is probably in /usr/local/bin
#export CXX=/usr/local/bin/g++-8 CC=/usr/local/bin/gcc-8
# Sys.setenv("CXX" = "/usr/local/bin/g++-8")
# Sys.setenv("CC" = "/usr/local/bin/gcc-8")

args <- commandArgs(trailingOnly = TRUE)
INSTALL_AFTER_BUILD <- !("--skip-install" %in% args)
TEMP_R_DIR <- file.path(getwd(), "lightgbm_r")
TEMP_SOURCE_DIR <- file.path(TEMP_R_DIR, "src")

# [description]
#     Parse the content of commandArgs() into a structured
#     list. This returns a list with two sections.
#       * "flags" = a character of vector of flags like "--use-gpu"
#       * "keyword_args" = a named character vector, where names
#           refer to options and values are the option values. For
#           example, c("--boost-librarydir" = "/usr/lib/x86_64-linux-gnu")
.parse_args <- function(args) {
  out_list <- list(
    "flags" = character(0L)
    , "keyword_args" = character(0L)
  )
  for (arg in args) {
    if (any(grepl("=", arg))) {
      split_arg <- strsplit(arg, "=")[[1L]]
      arg_name <- split_arg[[1L]]
      arg_value <- split_arg[[2L]]
      out_list[["keyword_args"]][[arg_name]] <- arg_value
    } else {
      out_list[["flags"]] <- c(out_list[["flags"]], arg)
    }
  }
  return(out_list)
}
parsed_args <- .parse_args(args)

USING_GPU <- "--use-gpu" %in% parsed_args[["flags"]]
USING_MINGW <- "--use-mingw" %in% parsed_args[["flags"]]
USING_MSYS2 <- "--use-msys2" %in% parsed_args[["flags"]]

# this maps command-line arguments to defines passed into CMake,
ARGS_TO_DEFINES <- c(
  "--boost-root" = "-DBOOST_ROOT"
  , "--boost-dir" = "-DBoost_DIR"
  , "--boost-include-dir" = "-DBoost_INCLUDE_DIR"
  , "--boost-librarydir" = "-DBOOST_LIBRARYDIR"
  , "--opencl-include-dir" = "-DOpenCL_INCLUDE_DIR"
  , "--opencl-library" = "-DOpenCL_LIBRARY"
)

recognized_args <- c(
  "--skip-install"
  , "--use-gpu"
  , "--use-mingw"
  , "--use-msys2"
  , names(ARGS_TO_DEFINES)
)
given_args <- c(
  parsed_args[["flags"]]
  , names(parsed_args[["keyword_args"]])
)
unrecognized_args <- setdiff(given_args, recognized_args)
if (length(unrecognized_args) > 0L) {
  msg <- paste0(
    "Unrecognized arguments: "
    , paste0(unrecognized_args, collapse = ", ")
  )
  stop(msg)
}

# [description] Replace statements in install.libs.R code based on
#               command-line flags
.replace_flag <- function(variable_name, value, content) {
  out <- gsub(
    pattern = paste0(variable_name, " <-.*")
    , replacement = paste0(variable_name, " <- ", as.character(value))
    , x = content
  )
  return(out)
}

install_libs_content <- readLines(
  file.path("R-package", "src", "install.libs.R")
)
install_libs_content <- .replace_flag("use_gpu", USING_GPU, install_libs_content)
install_libs_content <- .replace_flag("use_mingw", USING_MINGW, install_libs_content)
install_libs_content <- .replace_flag("use_msys2", USING_MSYS2, install_libs_content)

# set up extra flags based on keyword arguments
keyword_args <- parsed_args[["keyword_args"]]
if (length(keyword_args) > 0L) {
  cmake_args_to_add <- NULL
  for (i in seq_len(length(keyword_args))) {
    arg_name <- names(keyword_args)[[i]]
    define_name <- ARGS_TO_DEFINES[[arg_name]]
    arg_value <- shQuote(keyword_args[[arg_name]])
    cmake_args_to_add <- c(cmake_args_to_add, paste0(define_name, "=", arg_value))
  }
  install_libs_content <- gsub(
    pattern = paste0("command_line_args <- NULL")
    , replacement = paste0(
      "command_line_args <- c(\""
      , paste(cmake_args_to_add, collapse = "\", \"")
      , "\")"
    )
    , x = install_libs_content
  )
}

# R returns FALSE (not a non-zero exit code) if a file copy operation
# breaks. Let's fix that
.handle_result <- function(res) {
  if (!all(res)) {
    stop("Copying files failed!")
  }
  return(invisible(NULL))
}

# system() will not raise an R exception if the process called
# fails. Wrapping it here to get that behavior.
#
# system() introduces a lot of overhead, at least on Windows,
# so trying processx if it is available
.run_shell_command <- function(cmd, args, strict = TRUE) {
    on_windows <- .Platform$OS.type == "windows"
    has_processx <- suppressMessages({
      suppressWarnings({
        require("processx")  # nolint
      })
    })
    if (has_processx && on_windows) {
      result <- processx::run(
        command = cmd
        , args = args
        , windows_verbatim_args = TRUE
        , error_on_status = FALSE
        , echo = TRUE
      )
      exit_code <- result$status
    } else {
      if (on_windows) {
        message(paste0(
          "Using system() to run shell commands. Installing "
          , "'processx' with install.packages('processx') might "
          , "make this faster."
        ))
      }
      cmd <- paste0(cmd, " ", paste0(args, collapse = " "))
      exit_code <- system(cmd)
    }

    if (exit_code != 0L && isTRUE(strict)) {
        stop(paste0("Command failed with exit code: ", exit_code))
    }
    return(invisible(exit_code))
}

# Make a new temporary folder to work in
unlink(x = TEMP_R_DIR, recursive = TRUE)
dir.create(TEMP_R_DIR)

# copy in the relevant files
result <- file.copy(
  from = "R-package/./"
  , to = sprintf("%s/", TEMP_R_DIR)
  , recursive = TRUE
  , overwrite = TRUE
)
.handle_result(result)

# overwrite src/install.libs.R with new content based on command-line flags
writeLines(
  text = install_libs_content
  , con = file.path(TEMP_SOURCE_DIR, "install.libs.R")
)

# Add blank Makevars files
result <- file.copy(
  from = file.path(TEMP_R_DIR, "inst", "Makevars")
  , to = file.path(TEMP_SOURCE_DIR, "Makevars")
  , overwrite = TRUE
)
.handle_result(result)
result <- file.copy(
  from = file.path(TEMP_R_DIR, "inst", "Makevars.win")
  , to = file.path(TEMP_SOURCE_DIR, "Makevars.win")
  , overwrite = TRUE
)
.handle_result(result)

result <- file.copy(
  from = "include/"
  , to =  sprintf("%s/", TEMP_SOURCE_DIR)
  , recursive = TRUE
  , overwrite = TRUE
)
.handle_result(result)

result <- file.copy(
  from = "src/"
  , to = sprintf("%s/", TEMP_SOURCE_DIR)
  , recursive = TRUE
  , overwrite = TRUE
)
.handle_result(result)

EIGEN_R_DIR <- file.path(TEMP_SOURCE_DIR, "include", "Eigen")
dir.create(EIGEN_R_DIR)

eigen_modules <- c(
  "Cholesky"
  , "Core"
  , "Dense"
  , "Eigenvalues"
  , "Geometry"
  , "Householder"
  , "Jacobi"
  , "LU"
  , "QR"
  , "SVD"
)
for (eigen_module in eigen_modules) {
  result <- file.copy(
    from = file.path("external_libs", "eigen", "Eigen", eigen_module)
    , to = EIGEN_R_DIR
    , recursive = FALSE
    , overwrite = TRUE
  )
  .handle_result(result)
}

dir.create(file.path(EIGEN_R_DIR, "src"))

for (eigen_module in c(eigen_modules, "misc", "plugins")) {
  if (eigen_module == "Dense") {
    next
  }
  module_dir <- file.path(EIGEN_R_DIR, "src", eigen_module)
  dir.create(module_dir, recursive = TRUE)
  result <- file.copy(
    from = sprintf("%s/", file.path("external_libs", "eigen", "Eigen", "src", eigen_module))
    , to = sprintf("%s/", file.path(EIGEN_R_DIR, "src"))
    , recursive = TRUE
    , overwrite = TRUE
  )
  .handle_result(result)
}

.replace_pragmas <- function(filepath) {
  pragma_patterns <- c(
    "^.*#pragma clang diagnostic.*$"
    , "^.*#pragma diag_suppress.*$"
    , "^.*#pragma GCC diagnostic.*$"
    , "^.*#pragma region.*$"
    , "^.*#pragma endregion.*$"
    , "^.*#pragma warning.*$"
  )
  content <- readLines(filepath)
  for (pragma_pattern in pragma_patterns) {
    content <- content[!grepl(pragma_pattern, content)]
  }
  writeLines(content, filepath)
}

# remove pragmas that suppress warnings, to appease R CMD check
.replace_pragmas(
  file.path(EIGEN_R_DIR, "src", "Core", "arch", "SSE", "Complex.h")
)
.replace_pragmas(
  file.path(EIGEN_R_DIR, "src", "Core", "util", "DisableStupidWarnings.h")
)

result <- file.copy(
  from = "CMakeLists.txt"
  , to = file.path(TEMP_R_DIR, "inst", "bin/")
  , overwrite = TRUE
)
.handle_result(result)

# remove CRAN-specific files
result <- file.remove(
  file.path(TEMP_R_DIR, "cleanup")
  , file.path(TEMP_R_DIR, "configure")
  , file.path(TEMP_R_DIR, "configure.ac")
  , file.path(TEMP_R_DIR, "configure.win")
  , file.path(TEMP_SOURCE_DIR, "Makevars.in")
  , file.path(TEMP_SOURCE_DIR, "Makevars.win.in")
)
.handle_result(result)

#------------#
# submodules #
#------------#
EXTERNAL_LIBS_R_DIR <- file.path(TEMP_SOURCE_DIR, "external_libs")
dir.create(EXTERNAL_LIBS_R_DIR)
for (submodule in list.dirs(
  path = "external_libs"
  , full.names = FALSE
  , recursive = FALSE
)) {
  # compute/ is a submodule with boost, only needed if
  # building the R package with GPU support;
  # eigen/ has a special treatment due to licensing aspects
  if ((submodule == "compute" && !USING_GPU) || submodule == "eigen") {
    next
  }
  result <- file.copy(
    from = sprintf("%s/", file.path("external_libs", submodule))
    , to = sprintf("%s/", EXTERNAL_LIBS_R_DIR)
    , recursive = TRUE
    , overwrite = TRUE
  )
  .handle_result(result)
}

# copy files into the place CMake expects
CMAKE_MODULES_R_DIR <- file.path(TEMP_SOURCE_DIR, "cmake", "modules")
dir.create(CMAKE_MODULES_R_DIR, recursive = TRUE)
result <- file.copy(
  from = file.path("cmake", "modules", "FindLibR.cmake")
  , to = sprintf("%s/", CMAKE_MODULES_R_DIR)
  , overwrite = TRUE
)
.handle_result(result)
for (src_file in c("lightgbm_R.cpp", "lightgbm_R.h")) {
  result <- file.copy(
    from = file.path(TEMP_SOURCE_DIR, src_file)
    , to = file.path(TEMP_SOURCE_DIR, "src", src_file)
    , overwrite = TRUE
  )
  .handle_result(result)
  result <- file.remove(
    file.path(TEMP_SOURCE_DIR, src_file)
  )
  .handle_result(result)
}

result <- file.copy(
  from = file.path("R-package", "inst", "make-r-def.R")
  , to = file.path(TEMP_R_DIR, "inst", "bin/")
  , overwrite = TRUE
)
.handle_result(result)

# R packages cannot have versions like 3.0.0rc1, but
# 3.0.0-1 is acceptable
LGB_VERSION <- readLines("VERSION.txt")[1L]
LGB_VERSION <- gsub(
  pattern = "rc"
  , replacement = "-"
  , x = LGB_VERSION
)

# DESCRIPTION has placeholders for version
# and date so it doesn't have to be updated manually
DESCRIPTION_FILE <- file.path(TEMP_R_DIR, "DESCRIPTION")
description_contents <- readLines(DESCRIPTION_FILE)
description_contents <- gsub(
  pattern = "~~VERSION~~"
  , replacement = LGB_VERSION
  , x = description_contents
)
description_contents <- gsub(
  pattern = "~~DATE~~"
  , replacement = as.character(Sys.Date())
  , x = description_contents
)

# CMake-based builds don't get vignettes built. The
# VignetteBuilder field needs to be removed to avoid
# the R CMD CHECK note
# "Package has a VignetteBuilder field but no prebuilt vignette index"
description_contents <- description_contents[
  !grepl("^VignetteBuilder", description_contents)
]

writeLines(description_contents, DESCRIPTION_FILE)

# CMake-based builds can't currently use R's builtin routine registration,
# so have to update NAMESPACE manually, with a statement like this:
#
# useDynLib(lib_lightgbm, LGBM_GetLastError_R, LGBM_DatasetCreateFromFile_R, ...)
#
# See https://cran.r-project.org/doc/manuals/r-release/R-exts.html#useDynLib for
# documentation of this approach, where the NAMESPACE file uses a statement like
# useDynLib(foo, myRoutine, myOtherRoutine)
NAMESPACE_FILE <- file.path(TEMP_R_DIR, "NAMESPACE")
namespace_contents <- readLines(NAMESPACE_FILE)
dynlib_line <- grep(
  pattern = "^useDynLib"
  , x = namespace_contents
)

c_api_contents <- readLines(file.path(TEMP_SOURCE_DIR, "src", "lightgbm_R.h"))
c_api_contents <- c_api_contents[grepl("^LIGHTGBM_C_EXPORT", c_api_contents)]
c_api_contents <- gsub(
  pattern = "LIGHTGBM_C_EXPORT SEXP "
  , replacement = ""
  , x = c_api_contents
)
c_api_symbols <- gsub(
  pattern = "\\(.*"
  , replacement = ""
  , x = c_api_contents
)
dynlib_statement <- paste0(
  "useDynLib(lib_lightgbm, "
  , paste0(c_api_symbols, collapse = ", ")
  , ")"
)
namespace_contents[dynlib_line] <- dynlib_statement
writeLines(namespace_contents, NAMESPACE_FILE)

# NOTE: --keep-empty-dirs is necessary to keep the deep paths expected
#       by CMake while also meeting the CRAN req to create object files
#       on demand
.run_shell_command("R", c("CMD", "build", TEMP_R_DIR, "--keep-empty-dirs", "--no-build-vignettes"))

# Install the package
version <- gsub(
  "Version: ",
  "",
  grep(
    "Version: "
    , readLines(con = file.path(TEMP_R_DIR, "DESCRIPTION"))
    , value = TRUE
  )
)
tarball <- file.path(getwd(), sprintf("lightgbm_%s.tar.gz", version))

install_cmd <- "R"
install_args <- c("CMD", "INSTALL", "--no-multiarch", "--with-keep.source", tarball)
if (INSTALL_AFTER_BUILD) {
  .run_shell_command(install_cmd, install_args)
} else {
  cmd <- paste0(install_cmd, " ", paste0(install_args, collapse = " "))
  print(sprintf("Skipping installation. Install the package with command '%s'", cmd))
}
