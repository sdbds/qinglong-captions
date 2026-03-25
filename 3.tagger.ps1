#region Configuration
# set proxy if needed
# $env:HTTP_PROXY="http://127.0.0.1:10809"
# $env:HTTPS_PROXY="http://127.0.0.1:10809"

# Model settings
$Config = @{
    train_data_dir     = "./datasets"                                # Input images path | 图片输入路径
    repo_id            = "cella110n/cl_tagger"      # Model repo ID from Hugging Face
    model_dir          = "wd14_tagger_model"                         # Local model folder path | 本地模型文件夹路径
    batch_size         = 12                                          # Batch size for inference
    thresh             = 0.6                                         # Concept threshold
    general_threshold  = 0.55                                         # General threshold
    character_threshold = 1.0                                       # Character threshold
}

# Feature flags
$Features = @{
    frequency_tags           = $false     # Order by frequency tags
    remove_underscore        = $true      # Convert underscore to space
    use_rating_tags          = $true      # Use rating tags
    use_quality_tags         = $false      # Use quality tags
    use_model_tags           = $false      # Use model tags
    use_rating_tags_as_last_tag = $false  # Put rating tags at the end
    character_tags_first     = $false     # Put character tags first
    character_tag_expand     = $false     # Split character_(series) into character, series
    remove_parents_tag       = $true      # Remove parent tags
    overwrite            = $true          # Overwrite existing tag files
    add_tags_threshold = $false           # Add tags threshold
}

# Tag settings
$TagConfig = @{
    undesired_tags    = ""  # Tags to exclude
    always_first_tags = "1girl,1boy,2girls,3girls,4girls,5girls,6girls,2boys,3boys,4boys,5boys,6boys"
    tag_replacement   = "1girl,1woman;2girls,2women;3girls,3women;4girls,4women;5girls,5women;1boy,1man"
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
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
$Env:UV_INDEX_STRATEGY = "unsafe-best-match"
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

function Invoke-WdtaggerOpenCvCleanup {
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

    Write-Output "wdtagger OpenCV cleanup: removing conflicting cv2 wheels"
    Write-Output "uv pip uninstall target packages: $($Packages -join ', ')"
    uv @CleanupArgs
    return $LASTEXITCODE
}

function Get-WdtaggerOpenCvInstallPlan {
    param (
        [string]$PythonExe
    )

    $ProbeScript = Join-Path $PSScriptRoot "utils\wdtagger_opencv.py"
    $RawOutput = (& $PythonExe $ProbeScript --plan-install --platform win32 2>&1 | Out-String).Trim()
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "wdtagger OpenCV install plan resolution failed: $RawOutput"
    }

    $JsonLine = ($RawOutput -split "\r?\n" | Where-Object { $_.Trim() } | Select-Object -Last 1)

    try {
        return $JsonLine | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "wdtagger OpenCV install plan parse failed: $RawOutput"
    }
}

function Test-WdtaggerOpenCvImport {
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

# Add configuration arguments
if ($Config.repo_id) { [void]$ExtArgs.Add("--repo_id=$($Config.repo_id)") }
if ($Config.model_dir) { [void]$ExtArgs.Add("--model_dir=$($Config.model_dir)") }
if ($Config.batch_size) { [void]$ExtArgs.Add("--batch_size=$($Config.batch_size)") }
if ($Config.general_threshold) { [void]$ExtArgs.Add("--general_threshold=$($Config.general_threshold)") }
if ($Config.character_threshold) { [void]$ExtArgs.Add("--character_threshold=$($Config.character_threshold)") }

# Add feature flags
if ($Features.remove_underscore) { [void]$ExtArgs.Add("--remove_underscore") }
if ($Features.recursive) { [void]$ExtArgs.Add("--recursive") }
if ($Features.frequency_tags) { [void]$ExtArgs.Add("--frequency_tags") }
if ($Features.character_tags_first) { [void]$ExtArgs.Add("--character_tags_first") }
if ($Features.character_tag_expand) { [void]$ExtArgs.Add("--character_tag_expand") }
if ($Features.use_rating_tags_as_last_tag) { [void]$ExtArgs.Add("--use_rating_tags_as_last_tag") }
elseif ($Features.use_rating_tags) { [void]$ExtArgs.Add("--use_rating_tags") }
if ($Features.use_quality_tags) { [void]$ExtArgs.Add("--use_quality_tags") }
if ($Features.use_model_tags) { [void]$ExtArgs.Add("--use_model_tags") }
if ($Features.remove_parents_tag) { [void]$ExtArgs.Add("--remove_parents_tag") }
if ($Features.overwrite) { [void]$ExtArgs.Add("--overwrite") }
if ($Features.add_tags_threshold) { [void]$ExtArgs.Add("--add_tags_threshold") }

# Add tag configuration
if ($TagConfig.undesired_tags) { [void]$ExtArgs.Add("--undesired_tags=$($TagConfig.undesired_tags)") }
if ($TagConfig.always_first_tags) { [void]$ExtArgs.Add("--always_first_tags=$($TagConfig.always_first_tags)") }
if ($TagConfig.tag_replacement) { [void]$ExtArgs.Add("--tag_replacement=$($TagConfig.tag_replacement)") }

#endregion

#region Execute Tagger
Write-Output "Starting tagger..."
Install-UvExtraPatch @("wdtagger")
if ($env:OS -eq "Windows_NT") {
    $PythonExe = Get-ProjectPython
    if (-not $PythonExe) {
        $PythonExe = "python"
    }
    $OpenCvPlan = Get-WdtaggerOpenCvInstallPlan -PythonExe $PythonExe

    $OpenCvReady = $false
    $AttemptNumber = 0
    foreach ($Attempt in $OpenCvPlan.attempts) {
        $AttemptNumber += 1
        $CleanupExitCode = Invoke-WdtaggerOpenCvCleanup -PythonExe $PythonExe -Packages @($OpenCvPlan.cleanup_packages)
        if ($CleanupExitCode -ne 0) {
            Write-Output "wdtagger OpenCV cleanup returned non-zero; continuing with selected reinstall"
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

        Write-Output "wdtagger OpenCV attempt: $AttemptNumber"
        Write-Output "uv pip install target package: $($Attempt.package_name)"
        Write-Output "wdtagger OpenCV override source: $($Attempt.source)"
        if ($Attempt.cuda_tag) {
            Write-Output "wdtagger OpenCV detected CUDA toolkit: $($Attempt.cuda_tag)"
        }
        Write-Output "wdtagger OpenCV selection detail: $($Attempt.detail)"
        Write-Output "wdtagger OpenCV package spec: $($Attempt.package_spec)"

        uv @OpenCvInstallArgs
        if ($LASTEXITCODE -ne 0) {
            throw "wdtagger OpenCV override install failed"
        }

        $ProbeResult = Test-WdtaggerOpenCvImport -PythonExe $PythonExe
        if ($ProbeResult.Success) {
            Write-Output "wdtagger OpenCV import probe succeeded: $($ProbeResult.RawOutput)"
            $OpenCvReady = $true
            break
        }

        if ($Attempt.source -eq "cuda-wheel") {
            Write-Output "wdtagger OpenCV GPU import probe failed: $($ProbeResult.RawOutput)"
            Write-Output "wdtagger OpenCV GPU wheel unavailable; retrying with default CPU package"
            continue
        }

        Write-Output "wdtagger OpenCV import probe failed after CPU fallback: $($ProbeResult.RawOutput)"
        throw "wdtagger OpenCV import probe failed"
    }

    if (-not $OpenCvReady) {
        throw "wdtagger OpenCV setup did not produce a working cv2 import"
    }
}
Write-Output "runtime target environment: $(Get-UvEnvName)"
Write-Output "runtime dependency profile: extra:wdtagger"

# Run tagger
python "./utils/wdtagger.py" `
    $Config.train_data_dir `
    --thresh=$($Config.thresh) `
    --caption_extension .txt `
    $ExtArgs

Write-Output "Tagger finished"
Read-Host | Out-Null

#endregion
