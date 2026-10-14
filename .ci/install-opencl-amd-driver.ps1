Write-Output "Installing OpenCL CPU platform"

$installer = "AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe"

Write-Output "Downloading OpenCL platform installer"
$ProgressPreference = "SilentlyContinue"  # progress bar bug extremely slows down download speed
$params = @{
    OutFile = "$installer"
    Uri = "https://github.com/lightgbm-org/LightGBM/releases/download/v2.0.12/$installer"
}
Invoke-WebRequest @params

if (Test-Path "$installer") {
    Write-Output "Successfully downloaded OpenCL platform installer"
} else {
    Write-Output "Unable to download OpenCL platform installer"
    Write-Output "Setting EXIT"
    $host.SetShouldExit(-1)
    exit 1
}

# Install OpenCL platform from installer executable
Write-Output "Running OpenCL installer"
Invoke-Command -ScriptBlock {
    Start-Process "$installer" -ArgumentList '/S /V"/quiet /norestart /passive /log opencl.log"' -Wait
}

$property = Get-ItemProperty -Path Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors
if ($null -eq $property) {
    Write-Output "Unable to install OpenCL CPU platform"
    Write-Output "OpenCL installation log:"
    Get-Content "opencl.log"
    Write-Output "Setting EXIT"
    $host.SetShouldExit(-1)
    exit 1
} else {
    Write-Output "Successfully installed OpenCL CPU platform"
    Write-Output "Current OpenCL drivers:"
    Write-Output $property
}
