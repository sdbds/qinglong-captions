#region Configuration
# image2psd settings
$Config = @{
    input_dir             = "./datasets/images"                              # Input image directory | 输入图片目录
    output_dir            = "workspace/image2psd_output"                     # Output folder root | 输出目录
    repo_id_layerdiff     = "layerdifforg/seethroughv0.0.2_layerdiff3d"      # LayerDiff model repo
    repo_id_depth         = "24yearsold/seethroughv0.0.1_marigold"           # Marigold model repo
    resolution            = 1024                                             # LayerDiff canvas resolution
    resolution_depth      = 720                                              # Marigold depth resolution (-1 => follow canvas size)
    inference_steps_depth = -1                                               # Marigold denoising steps (-1 => pipeline default)
    seed                  = 42                                               # Global random seed
    dtype                 = "bfloat16"                                       # bfloat16 | float16 | float32
    quant_mode            = "none"                                           # none | nf4
    offload_policy        = "delete"                                         # delete | cpu
    limit_images          = 0                                                # 0 => unlimited
    vae_ckpt              = ""                                               # Optional VAE checkpoint path
    unet_ckpt             = ""                                               # Optional UNet checkpoint path
}

# Feature flags
$Features = @{
    group_offload         = $false
    skip_completed        = $true
    continue_on_error     = $true
    save_to_psd           = $true
    tblr_split            = $false
    force_eager_attention = $false
}
#endregion

#region Environment Setup
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot$([System.IO.Path]::PathSeparator)$($env:PYTHONPATH)"
$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
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

#region Helpers
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

function Add-BooleanOptionalArg {
    param(
        [System.Collections.ArrayList]$Args,
        [string]$Name,
        [bool]$Enabled
    )

    if ($Enabled) {
        [void]$Args.Add("--$Name")
    } else {
        [void]$Args.Add("--no-$Name")
    }
}
#endregion

#region Build Arguments
$ExtArgs = [System.Collections.ArrayList]::new()

if (-not $Config.input_dir) {
    Write-Error "input_dir is required. Please set it in the configuration."
    exit 1
}

[void]$ExtArgs.Add("--input_dir=$($Config.input_dir)")
[void]$ExtArgs.Add("--output_dir=$($Config.output_dir)")
[void]$ExtArgs.Add("--repo_id_layerdiff=$($Config.repo_id_layerdiff)")
[void]$ExtArgs.Add("--repo_id_depth=$($Config.repo_id_depth)")
[void]$ExtArgs.Add("--resolution=$($Config.resolution)")
[void]$ExtArgs.Add("--resolution_depth=$($Config.resolution_depth)")
[void]$ExtArgs.Add("--inference_steps_depth=$($Config.inference_steps_depth)")
[void]$ExtArgs.Add("--seed=$($Config.seed)")
[void]$ExtArgs.Add("--dtype=$($Config.dtype)")
[void]$ExtArgs.Add("--quant_mode=$($Config.quant_mode)")
[void]$ExtArgs.Add("--offload_policy=$($Config.offload_policy)")
[void]$ExtArgs.Add("--limit_images=$($Config.limit_images)")

if ($Config.vae_ckpt) { [void]$ExtArgs.Add("--vae_ckpt=$($Config.vae_ckpt)") }
if ($Config.unet_ckpt) { [void]$ExtArgs.Add("--unet_ckpt=$($Config.unet_ckpt)") }

Add-BooleanOptionalArg -Args $ExtArgs -Name "group_offload" -Enabled $Features.group_offload
Add-BooleanOptionalArg -Args $ExtArgs -Name "skip_completed" -Enabled $Features.skip_completed
Add-BooleanOptionalArg -Args $ExtArgs -Name "continue_on_error" -Enabled $Features.continue_on_error
Add-BooleanOptionalArg -Args $ExtArgs -Name "save_to_psd" -Enabled $Features.save_to_psd
Add-BooleanOptionalArg -Args $ExtArgs -Name "tblr_split" -Enabled $Features.tblr_split
Add-BooleanOptionalArg -Args $ExtArgs -Name "force_eager_attention" -Enabled $Features.force_eager_attention
#endregion

#region Execute
Write-Output "Starting image2psd (see-through CLI)..."
Install-UvExtraPatch @("see-through")
Write-Output "runtime target environment: $(Get-UvEnvName)"
Write-Output "runtime dependency profile: extra:see-through"

python -m module.see_through.cli `
    $ExtArgs

Write-Output "image2psd finished"
Read-Host | Out-Null
#endregion
