loadNamespace("pkgdown")
loadNamespace("roxygen2")

roxygen2::roxygenize(
    "R-package/"
    , load = "installed"
)

pkgdown::build_site(
    "R-package/"
    , lazy = FALSE
    , install = FALSE
    , devel = FALSE
    , examples = TRUE
    , run_dont_run = TRUE
    , seed = 42L
    , preview = FALSE
    , new_process = TRUE
)
