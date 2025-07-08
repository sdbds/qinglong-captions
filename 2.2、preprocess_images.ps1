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
    input_dir          = "./datasets"  # REQUIRED: Input directory path for source images
    align_input_dir    = ""                             # Optional: Path to directory with reference images for alignment
    max_long_edge      = 2048                             # Optional: Maximum value for the longest edge of resized images (e.g., 1024)
    recursive          = $true                            # Optional: Set to $true to recursively process subdirectories
    workers            = 8                             # Optional: Maximum number of worker threads for processing (e.g., 8)
    transform_type     = "auto"                            # Optional: Set to "auto" for automatic alignment, "none" for no alignment
    bg_color           = "255 255 255"                     # Optional: Background color for padding (e.g., 255 255 255 for white)
    python_script_path = ".\utils\preprocess_datasets.py" # Relative path to the Python script
}
#endregion

#region Environment Setup
# Activate python venv
Push-Location $PSScriptRoot # Change to script's directory, remember to Pop-Location
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"
$VenvPaths = @(
    ".\venv\Scripts\activate.ps1",
    ".\.venv\Scripts\activate.ps1",
    ".\venv\Scripts\Activate", # For Command Prompt style activation if .ps1 not found
    ".\.venv\Scripts\Activate"
)

$venvActivated = $false
foreach ($Path in $VenvPaths) {
    if (Test-Path $Path) {
        Write-Output "Activating venv: $Path"
        try {
            & $Path
            $venvActivated = $true
            Write-Output "Virtual environment activated successfully."
            break
        }
        catch {
            Write-Warning "Failed to activate venv at $Path. Error: $($_.Exception.Message)"
        }
    }
}

if (-not $venvActivated) {
    Write-Warning "Could not find or activate a Python virtual environment. Please ensure a venv exists at ./venv or ./.venv and is correctly set up."
    # Optionally, exit here if venv is critical
    # exit 1
}

# Set common environment variables that might be useful for Python scripts (similar to the example)
$Env:HF_HOME = "huggingface" # Example, adjust if needed
# $Env:HF_ENDPOINT = "https://hf-mirror.com" # Example, adjust if needed
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1" # Example, adjust if needed
#endregion

#region Build Arguments
$PythonExecutable = "python" # Or python.exe, or full path if not in PATH
$ExtArgs = [System.Collections.ArrayList]::new()

# Add configuration arguments for preprocess_datasets.py
if ($Config.input_dir) {
    [void]$ExtArgs.Add("--input=$($Config.input_dir)")
}
else {
    Write-Error "Input directory (--input) is required. Please set it in the configuration."
    Pop-Location
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

#endregion

#region Execute Image Processing Script
Write-Output "Starting Image Processing using $($Config.python_script_path)..."
Write-Output "Arguments: $($ExtArgs -join ' ')"

# Check if Python script exists
if (-not (Test-Path $Config.python_script_path)) {
    Write-Error "Python script not found at $($Config.python_script_path). Please check the path."
    Pop-Location
    exit 1
}

try {
    # Execute the Python script
    & $PythonExecutable $Config.python_script_path $ExtArgs
    Write-Output "Image Processing script finished."
}
catch {
    Write-Error "An error occurred while running the Python script: $($_.Exception.Message)"
    # You might want to inspect $_ for more details on the error
}

#endregion

#region Cleanup
Pop-Location # Return to the original directory
Write-Output "Script execution complete. Press any key to exit..."
Read-Host -Prompt "Press Enter to exit" | Out-Null
#endregion
