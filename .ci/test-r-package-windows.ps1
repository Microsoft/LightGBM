# Download a file and retry upon failure. This looks like
# an infinite loop but CI-level timeouts will kill it
function Get-File-With-Tenacity {
    param(
        [Parameter(Mandatory = $true)][string]$url,
        [Parameter(Mandatory = $true)][string]$destfile
    )
    $ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
    do {
        Write-Output "Downloading ${url}"
        sleep 5
        Invoke-WebRequest -Uri $url -OutFile $destfile
    } while (-not $?)
}

# External utilities like R.exe / Rscript.exe writing to stderr (even for harmless
# status information) can cause failures in GitHub Actions PowerShell jobs.
# See https://github.community/t/powershell-steps-fail-nondeterministically/115496
#
# Using standard PowerShell redirection does not work to avoid these errors.
# This function uses R's built-in redirection mechanism, sink(). Any place where
# this function is used is a command that writes harmless messages to stderr
function Invoke-R-Code-Redirect-Stderr {
    param(
        [Parameter(Mandatory = $true)][string]$rcode
    )
    $decorated_code = "out_file <- file(tempfile(), open = 'wt'); sink(out_file, type = 'message'); $rcode; sink()"
    Rscript --vanilla -e $decorated_code
}

# Remove all items matching some pattern from PATH environment variable
function Remove-From-Path {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory = $true)][string]$pattern_to_remove
    )
    if ($PSCmdlet.ShouldProcess($env:PATH, "Removing ${pattern_to_remove}")) {
        $env:PATH = ($env:PATH.Split(';') | Where-Object { $_ -notmatch "$pattern_to_remove" }) -join ';'
    }
}

# remove some details that exist in the GitHub Actions images which might
# cause conflicts with R and other components installed by this script
$env:RTOOLS40_HOME = ""
Remove-From-Path ".*Amazon.*"
Remove-From-Path ".*Anaconda.*"
Remove-From-Path ".*android.*"
Remove-From-Path ".*Android.*"
Remove-From-Path ".*chocolatey.*"
Remove-From-Path ".*Chocolatey.*"
Remove-From-Path ".*cmake.*"
Remove-From-Path ".*CMake.*"
Remove-From-Path ".*\\Git\\.*"
Remove-From-Path "(?!.*pandoc.*).*hostedtoolcache.*"
Remove-From-Path ".*Microsoft SDKs.*"
Remove-From-Path ".*mingw.*"
Remove-From-Path ".*msys64.*"
Remove-From-Path ".*PostgreSQL.*"
Remove-From-Path ".*\\R\\.*"
Remove-From-Path ".*R Client.*"
Remove-From-Path ".*rtools40.*"
Remove-From-Path ".*rtools42.*"
Remove-From-Path ".*rtools43.*"
Remove-From-Path ".*rtools44.*"
Remove-From-Path ".*rtools45.*"
Remove-From-Path ".*shells.*"
Remove-From-Path ".*Strawberry.*"
Remove-From-Path ".*tools.*"

Remove-Item C:\rtools40 -Force -Recurse -ErrorAction Ignore
Remove-Item C:\rtools42 -Force -Recurse -ErrorAction Ignore
Remove-Item C:\rtools43 -Force -Recurse -ErrorAction Ignore
Remove-Item C:\rtools44 -Force -Recurse -ErrorAction Ignore
Remove-Item C:\rtools45 -Force -Recurse -ErrorAction Ignore

# Get details needed for installing R components
#
# NOTES:
#    * some paths and file names are different on R4.0
$env:R_MAJOR_VERSION = $env:R_VERSION.split('.')[0]
if ($env:R_MAJOR_VERSION -eq "4") {
    $RTOOLS_INSTALL_PATH = "C:\rtools43"
    $env:RTOOLS_BIN = "$RTOOLS_INSTALL_PATH\usr\bin"
    $env:RTOOLS_MINGW_BIN = "$RTOOLS_INSTALL_PATH\x86_64-w64-mingw32.static.posix\bin"
    $env:RTOOLS_EXE_FILE = "rtools43-5550-5548.exe"
    $env:R_WINDOWS_VERSION = "4.3.1"
} else {
    Write-Output "[ERROR] Unrecognized R version: $env:R_VERSION"
    Assert-Output $false
}
$env:CMAKE_VERSION = "3.30.0"

$env:R_LIB_PATH = "$env:BUILD_SOURCESDIRECTORY/RLibrary" -replace '[\\]', '/'
$env:R_LIBS = "$env:R_LIB_PATH"
$env:CMAKE_PATH = "$env:BUILD_SOURCESDIRECTORY/CMake_installation"
$env:PATH = @(
    "$env:RTOOLS_BIN",
    "$env:RTOOLS_MINGW_BIN",
    "$env:R_LIB_PATH/R/bin/x64",
    "$env:CMAKE_PATH/cmake-$env:CMAKE_VERSION-windows-x86_64/bin",
    "$env:PATH"
) -join ";"
$env:CRAN_MIRROR = "https://cran.rstudio.com"
$env:MIKTEX_EXCEPTION_PATH = "$env:TEMP\miktex"

# don't fail builds for long-running examples unless they're very long.
# See https://github.com/lightgbm-org/LightGBM/issues/4049#issuecomment-793412254.
if ($env:R_BUILD_TYPE -ne "cran") {
    $env:_R_CHECK_EXAMPLE_TIMING_THRESHOLD_ = 30
}

if (($env:COMPILER -eq "MINGW") -and ($env:R_BUILD_TYPE -eq "cmake")) {
    $env:CXX = "$env:RTOOLS_MINGW_BIN/g++.exe"
    $env:CC = "$env:RTOOLS_MINGW_BIN/gcc.exe"
}

Set-Location "$env:BUILD_SOURCESDIRECTORY"
tzutil /s "GMT Standard Time"
[Void][System.IO.Directory]::CreateDirectory("$env:R_LIB_PATH")
[Void][System.IO.Directory]::CreateDirectory("$env:CMAKE_PATH")

# download R, RTools and CMake
Write-Output "Downloading R, Rtools and CMake"
$params = @{
    url = "$env:CRAN_MIRROR/bin/windows/base/old/$env:R_WINDOWS_VERSION/R-$env:R_WINDOWS_VERSION-win.exe"
    destfile = "R-win.exe"
}
Get-File-With-Tenacity @params

$params = @{
    url = "https://github.com/lightgbm-org/LightGBM/releases/download/v2.0.12/$env:RTOOLS_EXE_FILE"
    destfile = "Rtools.exe"
}
Get-File-With-Tenacity @params

$params = @{
    url = "https://github.com/Kitware/CMake/releases/download/v{0}/cmake-{0}-windows-x86_64.zip" -f $env:CMAKE_VERSION
    destfile = "$env:CMAKE_PATH/cmake.zip"
}
Get-File-With-Tenacity @params

# Install R
Write-Output "Installing R"
$params = @{
    FilePath = "R-win.exe"
    NoNewWindow = $true
    Wait = $true
    ArgumentList = "/VERYSILENT /DIR=$env:R_LIB_PATH/R /COMPONENTS=main,x64,i386"
}
Start-Process @params ; Assert-Output $?
Write-Output "Done installing R"

Write-Output "Installing Rtools"
$params = @{
    FilePath = "Rtools.exe"
    NoNewWindow = $true
    Wait = $true
    ArgumentList = "/VERYSILENT /SUPPRESSMSGBOXES /DIR=$RTOOLS_INSTALL_PATH"
}
Start-Process @params; Assert-Output $?
Write-Output "Done installing Rtools"

Write-Output "Installing CMake"
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("$env:CMAKE_PATH/cmake.zip", "$env:CMAKE_PATH") ; Assert-Output $?
# Remove old CMake shipped with RTools
Remove-Item "$env:RTOOLS_MINGW_BIN/cmake.exe" -Force -ErrorAction Ignore
Write-Output "Done installing CMake"

Write-Output "Installing dependencies"
Rscript.exe --vanilla ".ci/install-r-deps.R" --build --include=processx --test ; Assert-Output $?

Write-Output "Building R-package"

# R CMD check is not used for MSVC builds
if ($env:COMPILER -ne "MSVC") {

    $PKG_FILE_NAME = "lightgbm_$env:LGB_VER.tar.gz"
    $LOG_FILE_NAME = "lightgbm.Rcheck/00check.log"

    if ($env:R_BUILD_TYPE -eq "cmake") {
        if ($env:TOOLCHAIN -eq "MINGW") {
            Write-Output "Telling R to use MinGW"
            $env:BUILD_R_FLAGS = "c('--skip-install', '--use-mingw', '-j4')"
        } elseif ($env:TOOLCHAIN -eq "MSYS") {
            Write-Output "Telling R to use MSYS"
            $env:BUILD_R_FLAGS = "c('--skip-install', '--use-msys2', '-j4')"
        } elseif ($env:TOOLCHAIN -eq "MSVC") {
            $env:BUILD_R_FLAGS = "'--skip-install'"
        } else {
            Write-Output "[ERROR] Unrecognized toolchain: $env:TOOLCHAIN"
            Assert-Output $false
        }
        Invoke-R-Code-Redirect-Stderr "commandArgs <- function(...){$env:BUILD_R_FLAGS}; source('build_r.R')"
        Assert-Output $?
    } elseif ($env:R_BUILD_TYPE -eq "cran") {
        $params = -join @(
            "result <- processx::run(command = 'sh', args = 'build-cran-package.sh', ",
            "echo = TRUE, windows_verbatim_args = FALSE, error_on_status = TRUE)"
        )
        Invoke-R-Code-Redirect-Stderr $params ; Assert-Output $?
        Remove-From-Path ".*msys64.*"
        # Test CRAN source .tar.gz in a directory that is not this repo or below it.
        # When people install.packages('lightgbm'), they won't have the LightGBM
        # git repo around. This is to protect against the use of relative paths
        # like ../../CMakeLists.txt that would only work if you are in the repoo
        $R_CMD_CHECK_DIR = "tmp-r-cmd-check"
        New-Item -Path "C:\" -Name $R_CMD_CHECK_DIR -ItemType "directory" > $null
        Move-Item -Path "$PKG_FILE_NAME" -Destination "C:\$R_CMD_CHECK_DIR\" > $null
        Set-Location "C:\$R_CMD_CHECK_DIR\"
    }

    Write-Output "Running R CMD check"
    if ($env:R_BUILD_TYPE -eq "cran") {
        # CRAN packages must pass without --no-multiarch (build on 64-bit and 32-bit)
        $check_args = "c('CMD', 'check', '--as-cran', '--run-donttest', '$PKG_FILE_NAME')"
    } else {
        $check_args = "c('CMD', 'check', '--no-multiarch', '--as-cran', '--run-donttest', '$PKG_FILE_NAME')"
    }
    $params = -join (
        "result <- processx::run(command = 'R.exe', args = $check_args, ",
        "echo = TRUE, windows_verbatim_args = FALSE, error_on_status = TRUE)"
    )
    Invoke-R-Code-Redirect-Stderr $params ; $check_succeeded = $?

    Write-Output "R CMD check build logs:"
    $INSTALL_LOG_FILE_NAME = "lightgbm.Rcheck\00install.out"
    Get-Content -Path "$INSTALL_LOG_FILE_NAME"

    Assert-Output $check_succeeded

    Write-Output "Looking for issues with R CMD check results"
    if (Get-Content "$LOG_FILE_NAME" | Select-String -Pattern "NOTE|WARNING|ERROR" -CaseSensitive -Quiet) {
        Write-Output "NOTEs, WARNINGs, or ERRORs have been found by R CMD check"
        Assert-Output $False
    }
} else {
    $INSTALL_LOG_FILE_NAME = "$env:BUILD_SOURCESDIRECTORY\00install_out.txt"
    Invoke-R-Code-Redirect-Stderr "source('build_r.R')" 1> $INSTALL_LOG_FILE_NAME ; $install_succeeded = $?
    Write-Output "----- build and install logs -----"
    Get-Content -Path "$INSTALL_LOG_FILE_NAME"
    Write-Output "----- end of build and install logs -----"
    Assert-Output $install_succeeded
    # some errors are not raised above, but can be found in the logs
    if (Get-Content "$INSTALL_LOG_FILE_NAME" | Select-String -Pattern "ERROR" -CaseSensitive -Quiet) {
        Write-Output "ERRORs have been found installing lightgbm"
        Assert-Output $False
    }
}

# Checking that the correct R version was used
if ($env:TOOLCHAIN -ne "MSVC") {
    $checks = Select-String -Path "${LOG_FILE_NAME}" -Pattern "using R version $env:R_WINDOWS_VERSION"
    $checks_cnt = $checks.Matches.length
} else {
    $checksParams = @{
        Path = "${INSTALL_LOG_FILE_NAME}"
        Pattern = "R version passed into FindLibR.* $env:R_WINDOWS_VERSION"
    }
    $checks = Select-String @checksParams
    $checks_cnt = $checks.Matches.length
}
if ($checks_cnt -eq 0) {
    Write-Output "Wrong R version was found (expected '$env:R_WINDOWS_VERSION'). Check the build logs."
    Assert-Output $False
}

# Checking that we actually got the expected compiler. The R-package has some logic
# to fail back to MinGW if MSVC fails, but for CI builds we need to check that the correct
# compiler was used.
if ($env:R_BUILD_TYPE -eq "cmake") {
    $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "Check for working CXX compiler.*$env:COMPILER"
    if ($checks.Matches.length -eq 0) {
        Write-Output "The wrong compiler was used. Check the build logs."
        Assert-Output $False
    }
}

# Checking that we got the right toolchain for MinGW. If using MinGW, both
# MinGW and MSYS toolchains are supported
if (($env:COMPILER -eq "MINGW") -and ($env:R_BUILD_TYPE -eq "cmake")) {
    $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "Trying to build with.*$env:TOOLCHAIN"
    if ($checks.Matches.length -eq 0) {
        Write-Output "The wrong toolchain was used. Check the build logs."
        Assert-Output $False
    }
}

# Checking that MM_PREFETCH preprocessor definition is actually used in CI builds.
if ($env:R_BUILD_TYPE -eq "cran") {
    $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "checking whether MM_PREFETCH work.*yes"
    $checks_cnt = $checks.Matches.length
} elseif ($env:TOOLCHAIN -ne "MSVC") {
    $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern ".*Performing Test MM_PREFETCH - Success"
    $checks_cnt = $checks.Matches.length
} else {
    $checks_cnt = 1
}
if ($checks_cnt -eq 0) {
    Write-Output "MM_PREFETCH preprocessor definition wasn't used. Check the build logs."
    Assert-Output $False
}

# Checking that MM_MALLOC preprocessor definition is actually used in CI builds.
if ($env:R_BUILD_TYPE -eq "cran") {
    $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern "checking whether MM_MALLOC work.*yes"
    $checks_cnt = $checks.Matches.length
} elseif ($env:TOOLCHAIN -ne "MSVC") {
    $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern ".*Performing Test MM_MALLOC - Success"
    $checks_cnt = $checks.Matches.length
} else {
    $checks_cnt = 1
}
if ($checks_cnt -eq 0) {
    Write-Output "MM_MALLOC preprocessor definition wasn't used. Check the build logs."
    Assert-Output $False
}

# Checking that OpenMP is actually used in CMake builds.
if ($env:R_BUILD_TYPE -eq "cmake") {
    $checks = Select-String -Path "${INSTALL_LOG_FILE_NAME}" -Pattern ".*Found OpenMP: TRUE.*"
    if ($checks.Matches.length -eq 0) {
        Write-Output "OpenMP wasn't found. Check the build logs."
        Assert-Output $False
    }
}

if ($env:COMPILER -eq "MSVC") {
    Write-Output "Running tests with testthat.R"
    Set-Location R-package/tests
    # NOTE: using Rscript.exe intentionally here, instead of Invoke-R-Code-Redirect-Stderr,
    #       because something about the interaction between Invoke-R-Code-Redirect-Stderr
    #       and testthat results in failing tests not exiting with a non-0 exit code.
    Rscript.exe --vanilla "testthat.R" ; Assert-Output $?
}

Write-Output "No issues were found checking the R-package"
