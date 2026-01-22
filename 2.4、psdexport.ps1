#region Configuration
# PSD export settings
$Config = @{
    psd_dir             = "H:\东方绘师录 PSD"   # Input PSD folder | PSD输入目录
    results_root        = ""         # Output folder root (empty => <psd_dir>/results)
    max_direct_layers   = 7           # Only used when --no-force-seven-layers
    resize_max_size     = 0
    resize_results_root = ""
}

# Feature flags
$Features = @{
    verbose             = $true
    include_invisible    = $false  # Include invisible layers
    force_seven_layers   = $true   # Always export fixed 7 semantic layers
    merge_lineart        = $true   # Merge all lineart layers into 004_lineart.png
    export_lance_to      = ""      # Optional: extract lance back to a folder
}
#endregion

#region Environment Setup
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"
$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"

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

# uv options
$uv_args = [System.Collections.ArrayList]::new()
#endregion

#region Build Arguments
$ExtArgs = [System.Collections.ArrayList]::new()

if ($Config.results_root) { [void]$ExtArgs.Add("--results-root=$($Config.results_root)") }
if ($Config.max_direct_layers) { [void]$ExtArgs.Add("--max-direct-layers=$($Config.max_direct_layers)") }
if ($Config.resize_max_size -and $Config.resize_max_size -gt 0) { [void]$ExtArgs.Add("--resize-max-size=$($Config.resize_max_size)") }
if ($Config.resize_results_root) { [void]$ExtArgs.Add("--resize-results-root=$($Config.resize_results_root)") }

if ($Features.include_invisible) { [void]$ExtArgs.Add("--include-invisible") }
if ($Features.verbose) { [void]$ExtArgs.Add("--verbose") }
if (-not $Features.force_seven_layers) { [void]$ExtArgs.Add("--no-force-seven-layers") }
if (-not $Features.merge_lineart) { [void]$ExtArgs.Add("--no-merge-lineart") }
if ($Features.export_lance_to) { [void]$ExtArgs.Add("--export-lance-to=$($Features.export_lance_to)") }
#endregion

#region Execute
Write-Output "Starting PSD export pipeline..."

# Install deps (Windows uv supports --with-requirements)
[void]$uv_args.Add("--with-requirements=requirements-psdexport.txt")

uv run $uv_args "./utils/psd_dataset_pipeline.py" `
    $Config.psd_dir `
    $ExtArgs

Write-Output "PSD export pipeline finished"
Read-Host | Out-Null
#endregion
