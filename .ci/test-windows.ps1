function Assert-Output {
    param( [Parameter(Mandatory = $true)][bool]$success )
    if (-not $success) {
        $host.SetShouldExit(-1)
        exit 1
    }
}

$env:LGB_VER = (Get-Content $env:BUILD_SOURCESDIRECTORY\VERSION.txt).trim()
$UseConda = $env:SETUP_CONDA -ne "false"
# Use custom temp directory to avoid
# > warning MSB8029: The Intermediate directory or Output directory cannot reside under the Temporary directory
# > as it could lead to issues with incremental build.
# And make sure this directory is always clean
$env:TMPDIR = "$env:USERPROFILE\tmp"
Remove-Item $env:TMPDIR -Force -Recurse -ErrorAction Ignore
[Void][System.IO.Directory]::CreateDirectory($env:TMPDIR)

# create the artifact upload directory if it doesn't exist yet
[Void][System.IO.Directory]::CreateDirectory($env:BUILD_ARTIFACTSTAGINGDIRECTORY)

if ($env:TASK -eq "r-package") {
    & .\.ci\test-r-package-windows.ps1 ; Assert-Output $?
    exit 0
}

if ($env:TASK -eq "cpp-tests") {
    cmake -B build -S . -DBUILD_CPP_TEST=ON -DUSE_DEBUG=ON -A x64
    cmake --build build --target testlightgbm --config Debug ; Assert-Output $?
    .\Debug\testlightgbm.exe ; Assert-Output $?
    exit 0
}

if ($env:TASK -eq "swig") {
    $env:JAVA_HOME = $env:JAVA_HOME_8_X64  # there is pre-installed Eclipse Temurin 8 somewhere
    $ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
    $ReleaseInfo = Invoke-RestMethod "https://sourceforge.net/projects/swig/best_release.json"
    $SwigFilename = $ReleaseInfo.platform_releases.windows.filename
    # e.g. "/swigwin/swigwin-4.4.1/swigwin-4.4.1.zip"
    $params = @{
        Uri = "https://sourceforge.net/projects/swig/files$SwigFilename/download"
        OutFile = "$env:BUILD_SOURCESDIRECTORY/swig/swigwin.zip"
        UserAgent = "curl"
    }
    Invoke-WebRequest @params
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory(
        "$env:BUILD_SOURCESDIRECTORY/swig/swigwin.zip",
        "$env:BUILD_SOURCESDIRECTORY/swig"
    ) ; Assert-Output $?
    $SwigFolder = Get-ChildItem -Name -Path "$env:BUILD_SOURCESDIRECTORY/swig" -Attributes Directory
    $env:PATH = @("$env:BUILD_SOURCESDIRECTORY/swig/$SwigFolder", "$env:PATH") -join ";"
    $BuildLogFileName = "$env:BUILD_SOURCESDIRECTORY\cmake_build.log"
    cmake -B build -S . -A x64 -DUSE_SWIG=ON *> "$BuildLogFileName" ; $build_succeeded = $?
    Write-Output "CMake build logs:"
    Get-Content -Path "$BuildLogFileName"
    Assert-Output $build_succeeded
    $checks = Select-String -Path "${BuildLogFileName}" -Pattern "-- Found SWIG.*${SwigFolder}/swig.exe"
    $checks_cnt = $checks.Matches.length
    if ($checks_cnt -eq 0) {
        Write-Output "Wrong SWIG version was found (expected '${SwigFolder}'). Check the build logs."
        Assert-Output $False
    }
    cmake --build build --target ALL_BUILD --config Release ; Assert-Output $?
    if ($env:PRODUCES_ARTIFACTS -eq "true") {
        cp ./build/lightgbmlib.jar $env:BUILD_ARTIFACTSTAGINGDIRECTORY/lightgbmlib_win.jar ; Assert-Output $?
    }
    exit 0
}

# There is no native conda distribution for Windows Arm64.
if (-not $UseConda) {
    python -m pip install `
        --only-binary=:all: `
        --requirement "$env:BUILD_SOURCESDIRECTORY/.ci/pip-envs/requirements-windows-arm64.txt" `
        --upgrade ; Assert-Output $?
} elseif ($env:PYTHON_VERSION -eq "3.10") {
    # 'pixi' is used for end-of-life Python versions
    $activation = ((& pixi shell-hook --locked -e py310 --shell powershell) -join "`n")
    Invoke-Expression $activation ; Assert-Output $?
} else {
    # update conda env
    $env:CONDA_ENV = "test-env"
    conda activate ; Assert-Output $?
    conda config --set always_yes yes --set changeps1 no ; Assert-Output $?
    conda config --remove channels defaults ; Assert-Output $?
    conda config --add channels nodefaults ; Assert-Output $?
    conda config --add channels conda-forge ; Assert-Output $?
    conda config --set channel_priority strict ; Assert-Output $?

    # From Python 3.13 onwards, CPython packages on conda-forge have build strings
    # with formats like "*_cp314".
    #
    # Have to be specific here (no trailing wildcard) to avoid unintentionally pulling
    # in free-threaded builds (e.g. "*_cp314t").
    $PythonMajorVersion, $PythonMinorVersion = $env:PYTHON_VERSION.Split(".")
    if ([int]$PythonMajorVersion -gt 3 -or ([int]$PythonMajorVersion -eq 3 -and [int]$PythonMinorVersion -gt 12)) {
        $env:PYTHON_ABI_TAG = "cp$($env:PYTHON_VERSION -replace '\.', '')"
    } else {
        $env:PYTHON_ABI_TAG = "cpython"
    }
    $env:CONDA_PYTHON_REQUIREMENT = "python=$env:PYTHON_VERSION[build=*_$env:PYTHON_ABI_TAG]"

    conda install -q -y conda "$env:CONDA_PYTHON_REQUIREMENT" ; Assert-Output $?

    # print output of 'conda info', to help in submitting bug reports
    Write-Output "conda info:"
    conda info

    $env:CONDA_REQUIREMENT_FILE = "$env:BUILD_SOURCESDIRECTORY/.ci/conda-envs/ci-core.txt"

    $condaParams = @(
        "-q",
        "-y",
        "-n", "$env:CONDA_ENV",
        "--file", "$env:CONDA_REQUIREMENT_FILE",
        "$env:CONDA_PYTHON_REQUIREMENT"
    )
    conda create @condaParams ; Assert-Output $?

    # print output of 'conda list', to help in submitting bug reports
    Write-Output "conda list:"
    conda list -n $env:CONDA_ENV

    # 'bdist' job invokes 'RefreshEnv' to update PATH from the registry (which may have been modified
    # by building in OpenCL support), so defer activating the conda environment until later for those builds.
    if ($env:TASK -ne "bdist") {
        conda activate $env:CONDA_ENV
    }
}

Set-Location "$env:BUILD_SOURCESDIRECTORY"
if ($env:TASK -eq "regular") {
    cmake -B build -S . -A x64 ; Assert-Output $?
    cmake --build build --target ALL_BUILD --config Release ; Assert-Output $?
    sh ./build-python.sh install --precompile --no-isolation ; Assert-Output $?
    cp ./Release/lib_lightgbm.dll "$env:BUILD_ARTIFACTSTAGINGDIRECTORY"
    cp ./Release/lightgbm.exe "$env:BUILD_ARTIFACTSTAGINGDIRECTORY"
} elseif ($env:TASK -eq "sdist") {
    sh ./build-python.sh --no-isolation sdist ; Assert-Output $?
    sh ./.ci/check-python-dists.sh ./dist ; Assert-Output $?
    Set-Location dist; pip install --no-deps @(Get-ChildItem *.gz) -v ; Assert-Output $?
} elseif ($env:TASK -eq "bdist") {
    if ($UseConda) {
        # Import the Chocolatey profile module so that the RefreshEnv command
        # invoked below properly updates the current PowerShell session environment.
        $module = "$env:ChocolateyInstall\helpers\chocolateyProfile.psm1"
        Import-Module "$module" ; Assert-Output $?
        RefreshEnv

        Write-Output "Current OpenCL drivers:"
        Get-ItemProperty -Path Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors

        # (re-) activate conda environment, in case any activation logic was overridden by that 'RefreshEnv' call above
        conda activate $env:CONDA_ENV
    }

    # TODO: restore --integrated-opencl as part of https://github.com/lightgbm-org/LightGBM/issues/6968
    sh "build-python.sh" --no-isolation bdist_wheel ; Assert-Output $?
    sh ./.ci/check-python-dists.sh ./dist ; Assert-Output $?
    Set-Location dist; pip install --no-deps @(Get-ChildItem "*py3-none-$env:WHEEL_PLATFORM_TAG.whl") ; Assert-Output $?
    cp @(Get-ChildItem "*py3-none-$env:WHEEL_PLATFORM_TAG.whl") "$env:BUILD_ARTIFACTSTAGINGDIRECTORY"
} elseif (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python")) {
    if ($env:COMPILER -eq "MINGW") {
        sh ./build-python.sh install --mingw ; Assert-Output $?
    } else {
        sh ./build-python.sh install; Assert-Output $?
    }
}

if (($env:TASK -eq "sdist") -or (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python"))) {
    # cannot test C API with "sdist" task
    $tests = "$env:BUILD_SOURCESDIRECTORY/tests/python_package_test"
} else {
    $tests = "$env:BUILD_SOURCESDIRECTORY/tests"
}
if ($env:TASK -eq "bdist") {
    # Make sure we can do both CPU and GPU; see tests/python_package_test/test_dual.py
    # TODO: set LIGHTGBM_TEST_DUAL_CPU_GPU back to "1" as part of https://github.com/lightgbm-org/LightGBM/issues/6968
    $env:LIGHTGBM_TEST_DUAL_CPU_GPU = "0"
}

pytest -ra $tests ; Assert-Output $?

if (($env:TASK -eq "regular") -or (($env:APPVEYOR -eq "true") -and ($env:TASK -eq "python"))) {
    Set-Location "$env:BUILD_SOURCESDIRECTORY/examples/python-guide"
    @("import matplotlib", "matplotlib.use('Agg')") + (Get-Content "plot_example.py") | Set-Content "plot_example.py"
    # Prevent interactive window mode
    (Get-Content "plot_example.py").replace(
        'graph.render(view=True)',
        'graph.render(view=False)'
    ) | Set-Content "plot_example.py"

    # install optional plotting libraries
    # (not necessary for pixi-managed environments, where they're just installed by default)
    if ($env:PYTHON_VERSION -ne "3.10") {
        conda install -q -y -n $env:CONDA_ENV "h5py>=3.10" "ipywidgets>=8.1.2" "notebook>=7.1.2"
    }
    # Run all examples
    foreach ($file in @(Get-ChildItem *.py)) {
        @(
            "import sys, warnings",
            -join @(
                "warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: ",
                "sys.stdout.write(warnings.formatwarning(message, category, filename, lineno, line))"
            )
        ) + (Get-Content $file) | Set-Content $file
        python $file ; Assert-Output $?
    }
    # Run all notebooks
    Set-Location "$env:BUILD_SOURCESDIRECTORY/examples/python-guide/notebooks"
    (Get-Content "interactive_plot_example.ipynb").replace(
        'INTERACTIVE = False',
        'assert False, \"Interactive mode disabled\"'
    ) | Set-Content "interactive_plot_example.ipynb"
    jupyter nbconvert --ExecutePreprocessor.timeout=180 --to notebook --execute --inplace *.ipynb ; Assert-Output $?
}
