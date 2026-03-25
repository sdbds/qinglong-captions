#region Configuration
$Config = @{
    input_path      = "./datasets"                          # Audio file or directory | 音频文件或目录
    repo_id         = "bdsqlsz/BS-ROFO-SW-Fixed-ONNX"      # Model repo ID from Hugging Face
    model_dir       = "audio_separator"                    # Local model folder path | 本地模型文件夹路径
    output_format   = "wav"                                # wav | flac | mp3
    segment_size    = 1101                                 # model default dim_t from inference config
    overlap         = 8                                    # Chunk step in seconds, matching original mdxc_overlap semantics
    batch_size      = 8                                    # Number of chunks per ONNX batch
    harmony_separation = $false                            # Run a second harmony split on non-silent vocals after the 6-stem pass
    harmony_repo_id = "bdsqlsz/mel_band_roformer_karaoke_aufr33-ONNX" # Harmony split model repo ID
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
#$Env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
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
if ($Config.harmony_separation) { [void]$ExtArgs.Add("--harmony_separation") }
if ($Config.harmony_repo_id) { [void]$ExtArgs.Add("--harmony_repo_id=$($Config.harmony_repo_id)") }
if ($Config.overwrite) { [void]$ExtArgs.Add("--overwrite") }
if ($Config.force_download) { [void]$ExtArgs.Add("--force_download") }
#endregion

#region Execute Audio Separator
Write-Output "Starting Audio Separator..."
Install-UvExtraPatch @("vocal-midi")
Write-Output "runtime target environment: $(Get-UvEnvName)"
Write-Output "runtime dependency profile: extra:vocal-midi"
python "./module/audio_separator.py" `
    $Config.input_path `
    $ExtArgs

Write-Output "Audio Separator finished"
Read-Host | Out-Null
#endregion
