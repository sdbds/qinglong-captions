#Requires -Version 5.1

<#
.SYNOPSIS
    Runs the preprocess_datasets.py Python script to batch resize and optionally align images.

.DESCRIPTION
    This script configures and executes the Python script 'utils/preprocess_datasets.py' for image processing tasks.
    It allows specifying input directories, alignment options, resizing parameters, and concurrency settings.

.NOTES
    Author: Cascade
    Last Modified: $(Get-Date)
    Ensure Python and necessary dependencies (Pillow, OpenCV-Python, Rich, Torch, Numpy) are installed in the virtual environment.
#>

#region Configuration
# Script settings - MODIFY THESE VALUES AS NEEDED
$Config = @{
    input_dir          = "./datasets"                     # REQUIRED: Input directory path for source images
    align_input_dir    = ""                               # Optional: Path to directory with reference images for alignment
    max_long_edge      = 2048                             # Optional: Maximum value for the longest edge of resized images (e.g., 1024)
    max_short_edge     = $null                            # Optional: Maximum value for the shortest edge of resized images (e.g., 1024)
    max_pixels         = $null                            # Optional: Maximum value for the number of pixels in resized images (e.g., 1024)
    recursive          = $true                            # Optional: Set to $true to recursively process subdirectories
    workers            = 8                                # Optional: Maximum number of worker threads for processing (e.g., 8)
    transform_type     = "auto"                           # Optional: Set to "auto" for automatic alignment, "none" for no alignment
    matcher_backend    = "auto"                           # Optional: Set to "auto", "xfeat", "affine_steerers", or "orb"
    bg_color           = "255 255 255"                    # Optional: Background color for padding (e.g., 255 255 255 for white)
    crop_transparent   = $true                            # Optional: Set to $true to crop transparent borders from RGBA images
    python_script_path = ".\utils\preprocess_datasets.py" # Relative path to the Python script
}
#endregion

#region Environment Setup
# Activate python venv
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"
$VenvPaths = @(
    "./venv/Scripts/activate",
    "./.venv/Scripts/activate",
    "./venv/bin/Activate.ps1",
    "./.venv/bin/activate.ps1"
)

foreach ($Path in $VenvPaths) {
    if (Test-Path $Path) {
        Write-Output "Activating venv: $Path"
        & $Path
        break
    }
}

# Set environment variables
$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:CUDA_HOME = "${env:CUDA_PATH}"
$Env:TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION = "1"
$Env:TF_CUDNN_USE_AUTOTUNE = "1"
$Env:TF_TRT_ALLOW_TF32 = "1"
#$Env:UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
#$Env:CUDA_VISIBLE_DEVICES = "1"  # 设置GPU id，0表示使用第一个GPU，-1表示不使用GPU
#endregion

#region Build Arguments
$ExtArgs = [System.Collections.ArrayList]::new()

function Get-UvEnvName {
    if ($env:VIRTUAL_ENV) {
        return Split-Path -Path $env:VIRTUAL_ENV -Leaf
    }
    if (Test-Path "./.venv") {
        return ".venv"
    }
    if (Test-Path "./venv") {
        return "venv"
    }
    return "uv-managed"
}

function Get-ProjectPython {
    $Candidates = @(
        "./.venv/Scripts/python.exe",
        "./venv/Scripts/python.exe",
        "./.venv/bin/python",
        "./venv/bin/python"
    )

    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return (Resolve-Path $Candidate).Path
        }
    }

    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        return $PythonCommand.Source
    }

    return $null
}

function Install-UvExtraPatch {
    param (
        [string[]]$Extras
    )

    if (-not $Extras -or $Extras.Count -eq 0) {
        return
    }

    $PythonExe = Get-ProjectPython
    $UvEnvName = Get-UvEnvName
    $Profile = ($Extras | Select-Object -Unique | ForEach-Object { "extra:$_" }) -join ", "
    Write-Output "使用共享 .venv 增量安装依赖补丁"
    Write-Output "uv pip install target environment: $UvEnvName"
    Write-Output "uv pip install dependency profile: $Profile"
    Write-Output "直接使用 uv pip install -r pyproject.toml 安装当前依赖 profile"

    $InstallArgs = [System.Collections.ArrayList]::new()
    [void]$InstallArgs.Add("pip")
    [void]$InstallArgs.Add("install")
    [void]$InstallArgs.Add("--no-build-isolation")
    if ($PythonExe) {
        [void]$InstallArgs.Add("--python")
        [void]$InstallArgs.Add($PythonExe)
    }
    [void]$InstallArgs.Add("-r")
    [void]$InstallArgs.Add("pyproject.toml")
    foreach ($Extra in ($Extras | Select-Object -Unique)) {
        [void]$InstallArgs.Add("--extra")
        [void]$InstallArgs.Add($Extra)
    }

    uv @InstallArgs
    if (!($?)) {
        throw "uv pip install failed"
    }
}

function Invoke-PreprocessOpenCvCleanup {
    param (
        [string]$PythonExe,
        [string[]]$Packages
    )

    $CleanupArgs = [System.Collections.ArrayList]::new()
    [void]$CleanupArgs.Add("pip")
    [void]$CleanupArgs.Add("uninstall")
    if ($PythonExe) {
        [void]$CleanupArgs.Add("--python")
        [void]$CleanupArgs.Add($PythonExe)
    }
    foreach ($Package in $Packages) {
        [void]$CleanupArgs.Add($Package)
    }

    Write-Output "preprocess OpenCV cleanup: removing conflicting cv2 wheels"
    Write-Output "uv pip uninstall target packages: $($Packages -join ', ')"
    uv @CleanupArgs
    return $LASTEXITCODE
}

function Get-PreprocessOpenCvInstallPlan {
    param (
        [string]$PythonExe
    )

    $ProbeScript = Join-Path $PSScriptRoot "utils\wdtagger_opencv.py"
    $RawOutput = (& $PythonExe $ProbeScript --plan-install --platform win32 2>&1 | Out-String).Trim()
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "preprocess OpenCV install plan resolution failed: $RawOutput"
    }

    $JsonLine = ($RawOutput -split "\r?\n" | Where-Object { $_.Trim() } | Select-Object -Last 1)

    try {
        return $JsonLine | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "preprocess OpenCV install plan parse failed: $RawOutput"
    }
}

function Test-PreprocessOpenCvImport {
    param (
        [string]$PythonExe
    )

    $ProbeScript = Join-Path $PSScriptRoot "utils\wdtagger_opencv.py"
    $RawOutput = (& $PythonExe $ProbeScript --probe-cv2 2>&1 | Out-String).Trim()
    $ProbeExitCode = $LASTEXITCODE
    $Payload = $null
    $JsonLine = $null
    if ($RawOutput) {
        $JsonLine = ($RawOutput -split "\r?\n" | Where-Object { $_.Trim() } | Select-Object -Last 1)
        try {
            $Payload = $JsonLine | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            $Payload = $null
        }
    }

    return [PSCustomObject]@{
        Success = ($ProbeExitCode -eq 0 -and $Payload -and $Payload.ok)
        ExitCode = $ProbeExitCode
        RawOutput = $RawOutput
        Payload = $Payload
    }
}

# Add configuration arguments for preprocess_datasets.py
if ($Config.input_dir) {
    [void]$ExtArgs.Add("--input=$($Config.input_dir)")
}
else {
    Write-Error "Input directory (--input) is required. Please set it in the configuration."
    exit 1
}

if ($Config.align_input_dir) {
    [void]$ExtArgs.Add("--align-input=$($Config.align_input_dir)")
}

if ($Config.max_short_edge) {
    [void]$ExtArgs.Add("--max-short-edge=$($Config.max_short_edge)")
}

if ($Config.max_long_edge) {
    [void]$ExtArgs.Add("--max-long-edge=$($Config.max_long_edge)")
}

if ($Config.max_pixels) {
    [void]$ExtArgs.Add("--max-pixels=$($Config.max_pixels)")
}

if ($Config.recursive) {
    [void]$ExtArgs.Add("--recursive")
}

if ($Config.workers) {
    [void]$ExtArgs.Add("--workers=$($Config.workers)")
}

if ($Config.transform_type) {
    [void]$ExtArgs.Add("--transform-type=$($Config.transform_type)")
}

if ($Config.matcher_backend) {
    [void]$ExtArgs.Add("--matcher-backend=$($Config.matcher_backend)")
}

if ($Config.bg_color) {
    [void]$ExtArgs.Add("--bg-color")
    $color_components = $Config.bg_color.Split(' ')
    foreach ($component in $color_components) {
        if (-not [string]::IsNullOrWhiteSpace($component)) {
            [void]$ExtArgs.Add($component.Trim())
        }
    }
}

if ($Config.crop_transparent) {
    [void]$ExtArgs.Add("--crop-transparent")
}

#endregion

#region Execute Image Processing Script
Write-Output "Starting Image Processing..."
Install-UvExtraPatch @("image-align")
if ($env:OS -eq "Windows_NT") {
    $PythonExe = Get-ProjectPython
    if (-not $PythonExe) {
        $PythonExe = "python"
    }
    $OpenCvPlan = Get-PreprocessOpenCvInstallPlan -PythonExe $PythonExe

    $OpenCvReady = $false
    $AttemptNumber = 0
    foreach ($Attempt in $OpenCvPlan.attempts) {
        $AttemptNumber += 1
        $CleanupExitCode = Invoke-PreprocessOpenCvCleanup -PythonExe $PythonExe -Packages @($OpenCvPlan.cleanup_packages)
        if ($CleanupExitCode -ne 0) {
            Write-Output "preprocess OpenCV cleanup returned non-zero; continuing with selected reinstall"
        }

        $OpenCvInstallArgs = [System.Collections.ArrayList]::new()
        [void]$OpenCvInstallArgs.Add("pip")
        [void]$OpenCvInstallArgs.Add("install")
        [void]$OpenCvInstallArgs.Add("--no-build-isolation")
        [void]$OpenCvInstallArgs.Add("--index-strategy")
        [void]$OpenCvInstallArgs.Add($(if ($Env:UV_INDEX_STRATEGY) { $Env:UV_INDEX_STRATEGY } else { "unsafe-best-match" }))
        if ($PythonExe) {
            [void]$OpenCvInstallArgs.Add("--python")
            [void]$OpenCvInstallArgs.Add($PythonExe)
        }
        [void]$OpenCvInstallArgs.Add("--reinstall-package")
        [void]$OpenCvInstallArgs.Add($Attempt.package_name)
        [void]$OpenCvInstallArgs.Add($Attempt.package_spec)

        Write-Output "preprocess OpenCV attempt: $AttemptNumber"
        Write-Output "uv pip install target package: $($Attempt.package_name)"
        Write-Output "preprocess OpenCV override source: $($Attempt.source)"
        if ($Attempt.cuda_tag) {
            Write-Output "preprocess OpenCV detected CUDA toolkit: $($Attempt.cuda_tag)"
        }
        Write-Output "preprocess OpenCV selection detail: $($Attempt.detail)"
        Write-Output "preprocess OpenCV package spec: $($Attempt.package_spec)"

        uv @OpenCvInstallArgs
        if ($LASTEXITCODE -ne 0) {
            throw "preprocess OpenCV override install failed"
        }

        $ProbeResult = Test-PreprocessOpenCvImport -PythonExe $PythonExe
        if ($ProbeResult.Success) {
            $CudaCount = 0
            if ($ProbeResult.Payload -and $null -ne $ProbeResult.Payload.cuda_count) {
                $CudaCount = [int]$ProbeResult.Payload.cuda_count
            }
            if ($Attempt.source -eq "cuda-wheel" -and $CudaCount -le 0) {
                Write-Output "preprocess OpenCV GPU probe found no CUDA devices: $($ProbeResult.RawOutput)"
                Write-Output "preprocess OpenCV GPU wheel unavailable; retrying with default CPU package"
                continue
            }
            Write-Output "preprocess OpenCV import probe succeeded: $($ProbeResult.RawOutput)"
            $OpenCvReady = $true
            break
        }

        if ($Attempt.source -eq "cuda-wheel") {
            Write-Output "preprocess OpenCV GPU import probe failed: $($ProbeResult.RawOutput)"
            Write-Output "preprocess OpenCV GPU wheel unavailable; retrying with default CPU package"
            continue
        }

        Write-Output "preprocess OpenCV import probe failed after CPU fallback: $($ProbeResult.RawOutput)"
        throw "preprocess OpenCV import probe failed"
    }

    if (-not $OpenCvReady) {
        throw "preprocess OpenCV setup did not produce a working cv2 import"
    }
}
$UvEnvName = Get-UvEnvName
Write-Output "runtime target environment: $UvEnvName"
Write-Output "runtime dependency profile: extra:image-align"
$RuntimePython = Get-ProjectPython
if (-not $RuntimePython) {
    $RuntimePython = "python"
}
Write-Output "runtime python: $RuntimePython"

# Execute the Python script
& $RuntimePython $Config.python_script_path `
    $ExtArgs

Write-Output "Image Processing finished"
Read-Host | Out-Null

#endregion
