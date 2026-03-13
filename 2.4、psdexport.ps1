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
$Env:UV_INDEX_STRATEGY = "unsafe-best-match"

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

function Ensure-UvLockFile {
    $LockFile = Join-Path $PSScriptRoot "uv.lock"
    if (Test-Path $LockFile) {
        return
    }

    $IndexStrategy = if ([string]::IsNullOrWhiteSpace($Env:UV_INDEX_STRATEGY)) { "unsafe-best-match" } else { $Env:UV_INDEX_STRATEGY }
    Write-Output "未找到 uv.lock，先生成锁文件 (index-strategy=$IndexStrategy)"
    uv lock --index-strategy $IndexStrategy
    if (!($?)) {
        throw "uv lock failed"
    }
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
    $ReqFile = Join-Path $env:TEMP "qinglong_uv_patch_$PID.txt"
    Ensure-UvLockFile
    uv export --frozen --no-emit-project --format requirements-txt --output-file $ReqFile --extra ($Extras | Select-Object -Unique)
    if (!($?)) {
        throw "uv export failed"
    }

    Write-Output "使用共享 .venv 增量安装依赖补丁"
    Write-Output "uv pip install target environment: $UvEnvName"
    Write-Output "uv pip install dependency profile: $Profile"

    try {
        if ($PythonExe) {
            uv pip install --no-build-isolation --python $PythonExe -r $ReqFile
        }
        else {
            uv pip install --no-build-isolation -r $ReqFile
        }

        if (!($?)) {
            throw "uv pip install failed"
        }
    }
    finally {
        Remove-Item $ReqFile -ErrorAction SilentlyContinue
    }
}
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

# Install deps from pyproject optional extras
[void]$uv_args.Add("--extra=psdexport")
Install-UvExtraPatch @("psdexport")
Write-Output "runtime target environment: $(Get-UvEnvName)"
Write-Output "runtime dependency profile: extra:psdexport"

python "./utils/psd_dataset_pipeline.py" `
    $Config.psd_dir `
    $ExtArgs

Write-Output "PSD export pipeline finished"
Read-Host | Out-Null
#endregion
