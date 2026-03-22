#region Configuration
# 场景检测设置
$Config = @{
    input_video_dir         = "./datasets"                          # 输入视频目录路径
    output_dir              = ""                                    # 输出目录路径，如果不指定则默认为输入目录
    detector                = "AdaptiveDetector"                    # 场景检测器，可选"ContentDetector","AdaptiveDetector","HashDetector","HistogramDetector","ThresholdDetector"
    threshold               = 0.0                                   # 场景检测阈值，数值越低越敏感。ContentDetector: 27.0, AdaptiveDetector: 3.0, HashDetector: 0.395, HistogramDetector: 0.05, ThresholdDetector: 12
    min_scene_len           = 16                                    # 最小场景长度，数值越小越敏感
    luma_only               = $false                                # 是否只使用亮度变化检测
    save_html               = $true                                 # 是否保存HTML报告
    video2images_min_number = 1                                     # 每个场景保存的图像数量，为0则不保存
    recursive               = $false                                # 是否递归搜索子目录
}
#endregion

#region Environment Setup
# 激活Python虚拟环境
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
#endregion

#region Build Arguments
$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
#$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
$Env:UV_INDEX_STRATEGY = "unsafe-best-match"
#$Env:CUDA_VISIBLE_DEVICES = "1"  # 设置GPU id，0表示使用第一个GPU，-1表示不使用GPU

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

# 添加配置参数
if ($Config.output_dir) { [void]$ExtArgs.Add("--output_dir=$($Config.output_dir)") }
if ($Config.detector -ne "AdaptiveDetector") { [void]$ExtArgs.Add("--detector=$($Config.detector)") }
if ($Config.threshold -ne 0.0) { [void]$ExtArgs.Add("--threshold=$($Config.threshold)") }
if ($Config.min_scene_len) { [void]$ExtArgs.Add("--min_scene_len=$($Config.min_scene_len)") }
if ($Config.luma_only) { [void]$ExtArgs.Add("--luma_only") }
if ($Config.save_html) { [void]$ExtArgs.Add("--save_html") }
if ($Config.video2images_min_number -gt 0) { [void]$ExtArgs.Add("--video2images_min_number=$($Config.video2images_min_number)") }
if ($Config.recursive) { [void]$ExtArgs.Add("--recursive") }
#endregion

#region Execute Scene Detection
Write-Output "Starting scene detection..."
Install-UvExtraPatch @("video-split")
Write-Output "runtime target environment: $(Get-UvEnvName)"
Write-Output "runtime dependency profile: extra:video-split"

# 运行场景检测程序
python "./module/videospilter.py" `
    $Config.input_video_dir `
    $ExtArgs

Write-Output "Scene detection finished"
Read-Host | Out-Null
#endregion
