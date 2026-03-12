$dataset_path = "./datasets"
$pair_dir = ""
$gemini_api_key = ""
$gemini_model_path = "gemini-3-pro-preview"
$gemini_task = ""
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$step_api_key = ""
$step_model_path = "step-1.5v-mini"
$kimi_api_key = ""
$kimi_model_path = "kimi-k2.5" # "moonshotai/kimi-k2.5" if you want to use nvidia's endpoint
$kimi_base_url = "https://api.moonshot.cn/v1" # "https://integrate.api.nvidia.com/v1" if you want to use nvidia's endpoint
$kimi_code_api_key = ""
$kimi_code_model_path = "k2p5"
$kimi_code_base_url = "https://api.kimi.com/coding/v1"

# MiniMax API 配置
$minimax_api_key = ""              # MiniMax API 密钥 (从 platform.minimaxi.com 获取)
$minimax_model_path = "MiniMax-M2.5"  # 可选: MiniMax-M2.5, MiniMax-M2.5-highspeed, MiniMax-M2.1, MiniMax-M2.1-highspeed, MiniMax-M2
$minimax_api_base_url = "https://api.minimax.io/v1"

# MiniMax Code API 配置 (针对代码和结构化输出优化)
$minimax_code_api_key = ""         # MiniMax Code API 密钥
$minimax_code_model_path = "MiniMax-M2"  # 默认使用 M2 模型，专为代码和Agent工作流优化
$minimax_code_base_url = "https://api.minimax.io/v1"

$qwenVL_api_key = ""
$qwenVL_model_path = "qwen-vl-max-latest" # qwen2.5-vl-72b-instruct<10mins qwen-vl-max-latest <1min
$glm_api_key = ""
$glm_model_path = "GLM-4V-Plus-0111"
$ark_api_key = ""
$ark_model_path = "doubao-seed-1-6"

# OpenAI Compatible API 配置（本地/自托管服务统一连接参数）
# - 直接走 openai_compatible provider 时：使用 openai_model_name
# - 本地 transformers provider 切到 server backend 时：复用 openai_base_url/openai_api_key/openai_model_name
$openai_api_key = ""           # API 密钥（本地服务可填任意值，如 "sk-no-key"）
$openai_base_url = ""          # 服务器地址，如：
                               # - vLLM:     http://localhost:8000/v1
                               # - SGLang:   http://localhost:30000/v1
                               # - Ollama:   http://localhost:11434/v1
                               # - LM Studio: http://localhost:1234/v1
$openai_model_name = ""        # 模型名称，如：
                               # - Qwen2-VL-7B-Instruct
                               # - llava-v1.5-7b
                               # - gemma-3-4b-it
$openai_temperature = 0.7      # 生成温度 (0.0 - 2.0)
$openai_max_tokens = 2048      # 最大生成 token 数
$openai_json_mode = $true      # 是否尝试使用 JSON 模式（如果服务器支持）
$local_runtime_backend = ""    # "", "direct", "openai"

$dir_name = $false
$mode = "long" # all, short, long
$not_clip_with_caption = $false              # Not clip with caption | 不根据caption裁剪
$wait_time = 1
$max_retries = 100
$segment_time = 600
# OCR model configuration
$ocr_model = ""  # Options: "pixtral_ocr", "deepseek_ocr", "hunyuan_ocr", "olmocr", "paddle_ocr", "moondream", "nanonets_ocr", "firered_ocr", "chandra_ocr", ""
$document_image = $true

# VLM model configuration for image/video tasks
$vlm_image_model = ""  # Options: "moondream", "qwen_vl_local", "step_vl_local", "penguin_vl_local", "reka_edge_local", ""

$scene_detector = "AdaptiveDetector" # from ["ContentDetector","AdaptiveDetector","HashDetector","HistogramDetector","ThresholdDetector"]
$scene_threshold = 0.0 # default value ["ContentDetector": 27.0, "AdaptiveDetector": 3.0, "HashDetector": 0.395, "HistogramDetector": 0.05, "ThresholdDetector": 12]
$scene_min_len = 15
$scene_luma_only = $false
$tags_highlightrate = 0.38

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
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
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG = "1"
#$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
$Env:UV_INDEX_STRATEGY = "unsafe-best-match"
$env:QINGLONG_API_V2="1"
#$Env:CUDA_VISIBLE_DEVICES = "1"  # 设置GPU id，0表示使用第一个GPU，-1表示不使用GPU

#$Env:HTTP_PROXY = "http://127.0.0.1:7890"
#$Env:HTTPS_PROXY = "http://127.0.0.1:7890"

$ext_args = [System.Collections.ArrayList]::new()
$uv_args = [System.Collections.ArrayList]::new()

function Add-UvExtra {
  param (
    [string]$Name
  )

  if ([string]::IsNullOrWhiteSpace($Name)) {
    return
  }

  $Arg = "--extra=$Name"
  if (-not $uv_args.Contains($Arg)) {
    [void]$uv_args.Add($Arg)
  }
}

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

function Get-UvProfile {
  param (
    [System.Collections.ArrayList]$ArgsList
  )

  if ($ArgsList.Count -eq 0) {
    return "default"
  }

  return ($ArgsList | ForEach-Object { $_.ToString().Replace("--extra=", "extra:").Replace("--group=", "group:") }) -join ", "
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

function Install-UvDependencyPatch {
  param (
    [System.Collections.ArrayList]$ArgsList
  )

  if ($ArgsList.Count -eq 0) {
    return
  }

  $Profile = Get-UvProfile $ArgsList
  $UvEnvName = Get-UvEnvName
  $Extras = @()
  $Groups = @()

  foreach ($Arg in $ArgsList) {
    $Text = $Arg.ToString()
    if ($Text.StartsWith("--extra=")) {
      $Extras += $Text.Substring(8)
    }
    elseif ($Text.StartsWith("--group=")) {
      $Groups += $Text.Substring(8)
    }
  }

  if ($Extras -contains "paddleocr") {
    Write-Output "检测到 paddleocr，使用 uv sync --inexact 处理冲突依赖"
    Write-Output "uv sync target environment: $UvEnvName"
    Write-Output "uv sync dependency profile: $Profile"
    uv sync --active --frozen --inexact $ArgsList
    if (!($?)) {
      throw "uv sync failed"
    }
    return
  }

  $PythonExe = Get-ProjectPython
  $ReqFile = Join-Path $env:TEMP "qinglong_uv_patch_$PID.txt"
  $ExportArgs = [System.Collections.ArrayList]::new()
  [void]$ExportArgs.Add("export")
  [void]$ExportArgs.Add("--frozen")
  [void]$ExportArgs.Add("--no-emit-project")
  [void]$ExportArgs.Add("--format")
  [void]$ExportArgs.Add("requirements-txt")
  [void]$ExportArgs.Add("--output-file")
  [void]$ExportArgs.Add($ReqFile)
  foreach ($Extra in ($Extras | Select-Object -Unique)) {
    [void]$ExportArgs.Add("--extra")
    [void]$ExportArgs.Add($Extra)
  }
  foreach ($Group in ($Groups | Select-Object -Unique)) {
    [void]$ExportArgs.Add("--group")
    [void]$ExportArgs.Add($Group)
  }

  Write-Output "导出当前功能所需依赖清单，避免构建本地项目包"
  uv @ExportArgs
  if (!($?)) {
    throw "uv export failed"
  }

  $InstallArgs = [System.Collections.ArrayList]::new()
  [void]$InstallArgs.Add("pip")
  [void]$InstallArgs.Add("install")
  [void]$InstallArgs.Add("--no-build-isolation")
  if ($PythonExe) {
    [void]$InstallArgs.Add("--python")
    [void]$InstallArgs.Add($PythonExe)
  }
  [void]$InstallArgs.Add("-r")
  [void]$InstallArgs.Add($ReqFile)

  Write-Output "使用共享 .venv 增量安装依赖补丁"
  Write-Output "uv pip install target environment: $UvEnvName"
  Write-Output "uv pip install dependency profile: $Profile"
  try {
    uv @InstallArgs
    if (!($?)) {
      throw "uv pip install failed"
    }
  }
  finally {
    Remove-Item $ReqFile -ErrorAction SilentlyContinue
  }
}

if ($pair_dir) {
  [void]$ext_args.Add("--pair_dir=$pair_dir")
}

if ($gemini_api_key) {
  [void]$ext_args.Add("--gemini_api_key=$gemini_api_key")
  if ($gemini_task) {
    [void]$ext_args.Add("--gemini_task=$gemini_task")
  }
}

if ($gemini_model_path) {
  [void]$ext_args.Add("--gemini_model_path=$gemini_model_path")
}

if ($pixtral_api_key) {
  [void]$ext_args.Add("--pixtral_api_key=$pixtral_api_key")
}

if ($pixtral_model_path) {
  [void]$ext_args.Add("--pixtral_model_path=$pixtral_model_path")
}

if ($step_api_key) {
  [void]$ext_args.Add("--step_api_key=$step_api_key")
}

if ($step_model_path) {
  [void]$ext_args.Add("--step_model_path=$step_model_path")
}

if ($kimi_api_key) {
  [void]$ext_args.Add("--kimi_api_key=$kimi_api_key")
}

if ($kimi_model_path) {
  [void]$ext_args.Add("--kimi_model_path=$kimi_model_path")
}

if ($kimi_base_url) {
  [void]$ext_args.Add("--kimi_base_url=$kimi_base_url")
}

if ($kimi_code_api_key) {
  [void]$ext_args.Add("--kimi_code_api_key=$kimi_code_api_key")
}

if ($kimi_code_model_path -and $kimi_code_model_path -ne "k2p5") {
  [void]$ext_args.Add("--kimi_code_model_path=$kimi_code_model_path")
}

if ($kimi_code_base_url) {
  [void]$ext_args.Add("--kimi_code_base_url=$kimi_code_base_url")
}

# MiniMax API 参数
if ($minimax_api_key) {
  [void]$ext_args.Add("--minimax_api_key=$minimax_api_key")
}

if ($minimax_model_path -and $minimax_model_path -ne "MiniMax-M2.5") {
  [void]$ext_args.Add("--minimax_model_path=$minimax_model_path")
}

if ($minimax_api_base_url -and $minimax_api_base_url -ne "https://api.minimax.io/v1") {
  [void]$ext_args.Add("--minimax_api_base_url=$minimax_api_base_url")
}

# MiniMax Code API 参数
if ($minimax_code_api_key) {
  [void]$ext_args.Add("--minimax_code_api_key=$minimax_code_api_key")
}

if ($minimax_code_model_path -and $minimax_code_model_path -ne "MiniMax-M2") {
  [void]$ext_args.Add("--minimax_code_model_path=$minimax_code_model_path")
}

if ($minimax_code_base_url -and $minimax_code_base_url -ne "https://api.minimax.io/v1") {
  [void]$ext_args.Add("--minimax_code_base_url=$minimax_code_base_url")
}

if ($qwenVL_api_key) {
  [void]$ext_args.Add("--qwenVL_api_key=$qwenVL_api_key")
}

if ($qwenVL_model_path) {
  [void]$ext_args.Add("--qwenVL_model_path=$qwenVL_model_path")
}

if ($glm_api_key) {
  [void]$ext_args.Add("--glm_api_key=$glm_api_key")
}

if ($glm_model_path) {
  [void]$ext_args.Add("--glm_model_path=$glm_model_path")
}

if ($ark_api_key) {
  [void]$ext_args.Add("--ark_api_key=$ark_api_key")
}

if ($ark_model_path) {
  [void]$ext_args.Add("--ark_model_path=$ark_model_path")
}

# OpenAI Compatible API 参数
if ($openai_api_key) {
  [void]$ext_args.Add("--openai_api_key=$openai_api_key")
}

if ($openai_base_url) {
  [void]$ext_args.Add("--openai_base_url=$openai_base_url")
}

if ($openai_model_name) {
  [void]$ext_args.Add("--openai_model_name=$openai_model_name")
}

if ($openai_temperature -ine 0.7) {
  [void]$ext_args.Add("--openai_temperature=$openai_temperature")
}

if ($openai_max_tokens -ine 2048) {
  [void]$ext_args.Add("--openai_max_tokens=$openai_max_tokens")
}

if (-not $openai_json_mode) {
  [void]$ext_args.Add("--openai_json_mode=false")
}

if ($dir_name) {
  [void]$ext_args.Add("--dir_name")
}

if ($mode -ine "all") {
  [void]$ext_args.Add("--mode=$mode")
}

if ($not_clip_with_caption) {
  [void]$ext_args.Add("--not_clip_with_caption")
}

if ($wait_time -ine 1) {
  [void]$ext_args.Add("--wait_time=$wait_time")
}

if ($max_retries -ine 20) {
  [void]$ext_args.Add("--max_retries=$max_retries")
}

if ($segment_time -ine 600) {
  [void]$ext_args.Add("--segment_time=$segment_time")
}

# OCR model selection
if ($ocr_model) {
  [void]$ext_args.Add("--ocr_model=$ocr_model")
  if ($document_image) {
    [void]$ext_args.Add("--document_image")
  }
  
  # Model-specific extras
  if ($ocr_model -eq "paddle_ocr") {
    $Env:UV_EXTRA_INDEX_URL = "https://www.paddlepaddle.org.cn/packages/nightly/cu129/"
    Add-UvExtra "paddleocr"
  }
  elseif ($ocr_model -eq "deepseek_ocr") {
    Add-UvExtra "deepseek-ocr"
  }
  elseif ($ocr_model -eq "olmocr") {
    Add-UvExtra "olmocr"
  }
  elseif ($ocr_model -eq "hunyuan_ocr") {
    Add-UvExtra "hunyuan-ocr"
  }
  elseif ($ocr_model -eq "moondream") {
    Add-UvExtra "moondream"
  }
  elseif ($ocr_model -eq "glm_ocr") {
    Add-UvExtra "glm-ocr"
  }
  elseif ($ocr_model -eq "nanonets_ocr") {
    Add-UvExtra "nanonets-ocr"
  }
  elseif ($ocr_model -eq "firered_ocr") {
    Add-UvExtra "firered-ocr"
  }
  elseif ($ocr_model -eq "chandra_ocr") {
    Add-UvExtra "chandra-ocr"
  }
}

# VLM model selection for image tasks
if ($vlm_image_model) {
  [void]$ext_args.Add("--vlm_image_model=$vlm_image_model")
  
  # Model-specific extras
  if ($vlm_image_model -eq "moondream") {
    Add-UvExtra "moondream"
  }
  elseif ($vlm_image_model -eq "qwen_vl_local") {
    Add-UvExtra "qwen-vl-local"
  }
  elseif ($vlm_image_model -eq "step_vl_local") {
    Add-UvExtra "step-vl-local"
  }
}

if ($scene_detector -ne "AdaptiveDetector") {
  [void]$ext_args.Add("--scene_detector=$($scene_detector)")
}

if ($scene_threshold -ne 0.0) {
  [void]$ext_args.Add("--scene_threshold=$scene_threshold")
}

if ($scene_min_len -ne 15) {
  [void]$ext_args.Add("--scene_min_len=$scene_min_len")
}

if ($scene_luma_only) {
  [void]$ext_args.Add("--scene_luma_only")
}

if ($tags_highlightrate -ne 0.4) {
  [void]$ext_args.Add("--tags_highlightrate=$tags_highlightrate")
}

# run train
$UvEnvName = Get-UvEnvName
$UvProfile = Get-UvProfile $uv_args
Install-UvDependencyPatch $uv_args
Write-Output "runtime target environment: $UvEnvName"
Write-Output "runtime dependency profile: $UvProfile"
python "./module/captioner.py" $dataset_path $ext_args

Write-Output "Captioner finished"
Read-Host | Out-Null ;
