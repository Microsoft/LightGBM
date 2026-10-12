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
    , "make_args" = character(0L)
  )
  for (arg in args) {
    if (any(grepl("^\\-j[0-9]+", arg))) {  # nolint: non_portable_path.
        out_list[["make_args"]] <- arg
    } else if (any(grepl("=", arg, fixed = TRUE))) {
      split_arg <- strsplit(arg, "=", fixed = TRUE)[[1L]]
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

SKIP_VIGNETTES <- "--no-build-vignettes" %in% parsed_args[["flags"]]
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
  "--no-build-vignettes"
  , "--skip-install"
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
    , toString(unrecognized_args)
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
  for (i in seq_along(keyword_args)) {
    arg_name <- names(keyword_args)[[i]]
    define_name <- ARGS_TO_DEFINES[[arg_name]]
    arg_value <- shQuote(normalizePath(keyword_args[[arg_name]], winslash = "/"))
    cmake_args_to_add <- c(cmake_args_to_add, paste0(define_name, "=", arg_value))
  }
  install_libs_content <- gsub(
    pattern = paste0("command_line_args <- NULL")
    , replacement = paste0(
      "command_line_args <- c(\'"
      , paste(cmake_args_to_add, collapse = "', '")
      , "')"
    )
    , x = install_libs_content
    , fixed = TRUE
  )
}

# if provided, set '-j' in 'make' commands in install.libs.R
if (length(parsed_args[["make_args"]]) > 0L) {
  install_libs_content <- gsub(
    pattern = "make_args_from_build_script <- character(0L)"
    , replacement = paste0(
      "make_args_from_build_script <- c(\""
      , paste(parsed_args[["make_args"]], collapse = "\", \"")
      , "\")"
    )
    , x = install_libs_content
    , fixed = TRUE
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
        require("processx")  # nolint: undesirable_function, unused_import.
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
      cmd <- paste0(cmd, " ", paste(args, collapse = " "))
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
  # building the R-package with GPU support;
  # eigen/ has a special treatment due to licensing aspects;
  # nanoarrow/ is only needed by the Arrow-based C API entry points, which
  # are excluded from the R build (the R API never calls into them).
  if ((submodule == "compute" && !USING_GPU)
      || submodule == "eigen"
      || submodule == "nanoarrow") {
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
CMAKE_R_DIR <- file.path(TEMP_SOURCE_DIR, "cmake")
CMAKE_MODULES_R_DIR <- file.path(TEMP_SOURCE_DIR, "cmake", "modules")
dir.create(CMAKE_MODULES_R_DIR, recursive = TRUE)
result <- file.copy(
  from = file.path("cmake", "modules", "FindLibR.cmake")
  , to = sprintf("%s/", CMAKE_MODULES_R_DIR)
  , overwrite = TRUE
)
result <- file.copy(
  from = file.path("cmake", "Utils.cmake")
  , to = sprintf("%s/", CMAKE_R_DIR)
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

# NOTE: --keep-empty-dirs is necessary to keep the deep paths expected
#       by CMake while also meeting the CRAN req to create object files
#       on demand
r_build_args <- c("CMD", "build", TEMP_R_DIR, "--keep-empty-dirs")
if (isTRUE(SKIP_VIGNETTES)) {
  r_build_args <- c(r_build_args, "--no-build-vignettes")
}
.run_shell_command("R", r_build_args)

# Install the package
version <- gsub(
  pattern = "Version: ",
  replacement = "",
  x = grep(
    pattern = "Version: "
    , x = readLines(con = file.path(TEMP_R_DIR, "DESCRIPTION"))
    , value = TRUE
    , fixed = TRUE
  )
  , fixed = TRUE
)
tarball <- file.path(getwd(), sprintf("lightgbm_%s.tar.gz", version))

install_cmd <- "R"
install_args <- c("CMD", "INSTALL", "--no-multiarch", "--with-keep.source", tarball)
if (INSTALL_AFTER_BUILD) {
  .run_shell_command(install_cmd, install_args)
} else {
  cmd <- paste0(install_cmd, " ", paste(install_args, collapse = " "))
  print(sprintf("Skipping installation. Install the package with command '%s'", cmd))
}
