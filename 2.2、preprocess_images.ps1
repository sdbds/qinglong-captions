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
#$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = 1
$Env:UV_NO_CACHE = 0
$Env:UV_LINK_MODE = "symlink"
#$Env:CUDA_VISIBLE_DEVICES = "1"  # 设置GPU id，0表示使用第一个GPU，-1表示不使用GPU
#endregion

#region Build Arguments
$ExtArgs = [System.Collections.ArrayList]::new()

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

# Execute the Python script
uv run $Config.python_script_path `
    $ExtArgs

Write-Output "Image Processing finished"
Read-Host | Out-Null

#endregion
