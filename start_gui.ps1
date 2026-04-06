#region Configuration
# GUI settings
$Config = @{
    host = "127.0.0.1"    # Bind address | 绑定地址
    port = 7899           # Port | 端口
    cloud = $false        # Cloud mode (bind 0.0.0.0) | 云模式
    native = $false       # Native window mode | 原生窗口模式 (默认关闭，使用浏览器)
    no_browser = $false   # Don't auto-open browser | 不自动打开浏览器 (默认自动打开浏览器)
}
#endregion

#region Environment Setup
# Run the GUI as a PEP 723 script so uv builds a script-scoped runtime
# instead of attaching the GUI process to the shared project environment.
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"
try {
    $UvCommand = Get-Command uv -ErrorAction Stop
} catch {
    Write-Error "uv not found in PATH. Please install uv first."
    exit 1
}

# Set environment variables
$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:UV_EXTRA_INDEX_URL = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/ https://download.pytorch.org/whl/cu130"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
$Env:UV_INDEX_STRATEGY = "unsafe-best-match"
#endregion

function Get-ProjectEnvName {
    if (Test-Path "./.venv") {
        return ".venv"
    }
    if (Test-Path "./venv") {
        return "venv"
    }
    return "unknown"
}

#region Build Arguments
$exargs = @()

if ($Config.cloud) {
    $exargs += "--cloud"
}

if ($Config.native) {
    $exargs += "--native"
}

if ($Config.no_browser) {
    $exargs += "--no-browser"
}

if ($Config.host -ne "127.0.0.1") {
    $exargs += "--host=$($Config.host)"
}

if ($Config.port -ne 8080) {
    $exargs += "--port=$($Config.port)"
}
#endregion

#region Start GUI
Write-Output "============================================================"
Write-Output "  青龙字幕工具 GUI (Qinglong Captions GUI)"
Write-Output "============================================================"
Write-Output ""
Write-Output "  URL: http://$($Config.host):$($Config.port)"
Write-Output "  UV: $($UvCommand.Source)"
Write-Output "  Runtime: uv run gui/launch.py"
Write-Output "  Project Env: $(Get-ProjectEnvName)"
if ($Config.native) {
    Write-Output "  Mode: Native window (原生窗口模式)"
} else {
    Write-Output "  Mode: Web browser (网页模式)"
}
if ($Config.no_browser) {
    Write-Output ""
    Write-Output "  请手动在浏览器中打开上述 URL"
}
Write-Output ""
Write-Output "  按 Ctrl+C 停止服务"
Write-Output "============================================================"
Write-Output ""

$launchArgs = @("run", "gui/launch.py") + $exargs
& $UvCommand.Source @launchArgs

Write-Output ""
Write-Output "GUI stopped"
#endregion
