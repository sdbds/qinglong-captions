#region Configuration
$Config = @{
    input_path      = "./datasets"                          # Audio file or directory | 音频文件或目录
    repo_id         = "bdsqlsz/BS-ROFO-SW-Fixed-ONNX"      # Model repo ID from Hugging Face
    model_dir       = "audio_separator"                    # Local model folder path | 本地模型文件夹路径
    output_format   = "wav"                                # wav | flac | mp3
    segment_size    = 1151                                 # chunk_size = hop_length * (segment_size - 1)
    overlap         = 8                                    # Higher is smoother but slower
    batch_size      = 1                                    # Number of chunks per ONNX batch
    recursive       = $true                                # Recursively scan directories
    overwrite       = $false                               # Overwrite existing song output directories
    force_download  = $false                               # Force re-download model artifacts
}
#endregion

#region Environment Setup
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot$([System.IO.Path]::PathSeparator)$($env:PYTHONPATH)"
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

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
$Env:UV_INDEX_STRATEGY = "unsafe-best-match"
$Env:PYTHONIOENCODING = "utf-8"
$Env:PYTHONUTF8 = "1"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
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

if (-not $Config.input_path) {
    Write-Error "input_path is required. Please set it in the configuration."
    exit 1
}

if ($Config.repo_id) { [void]$ExtArgs.Add("--repo_id=$($Config.repo_id)") }
if ($Config.model_dir) { [void]$ExtArgs.Add("--model_dir=$($Config.model_dir)") }
if ($Config.output_format) { [void]$ExtArgs.Add("--output_format=$($Config.output_format)") }
if ($Config.segment_size) { [void]$ExtArgs.Add("--segment_size=$($Config.segment_size)") }
if ($Config.overlap) { [void]$ExtArgs.Add("--overlap=$($Config.overlap)") }
if ($Config.batch_size) { [void]$ExtArgs.Add("--batch_size=$($Config.batch_size)") }
if ($Config.recursive) { [void]$ExtArgs.Add("--recursive") }
if ($Config.overwrite) { [void]$ExtArgs.Add("--overwrite") }
if ($Config.force_download) { [void]$ExtArgs.Add("--force_download") }
#endregion

#region Execute Audio Separator
Write-Output "Starting Audio Separator..."
Install-UvExtraPatch @("wdtagger")
Write-Output "runtime target environment: $(Get-UvEnvName)"
Write-Output "runtime dependency profile: extra:wdtagger"
python "./module/audio_separator.py" `
    $Config.input_path `
    $ExtArgs

Write-Output "Audio Separator finished"
Read-Host | Out-Null
#endregion
