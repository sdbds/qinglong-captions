$input_path = "./datasets"                 # 数据集目录或 .lance 路径
$output_name = "dataset"                   # 目录导入时生成的 Lance 名称
$model_id = "tencent/HY-MT1.5-7B"          # 本地翻译模型
$source_lang = "auto"                      # 源语言
$target_lang = "zh_cn"                     # 目标语言，同时用于导出文件后缀
$max_chars = 2200                          # 每个翻译分块的最大字符数
$context_chars = 300                       # 给下一个 chunk 的上下文字符数
$max_new_tokens = 2048                     # 每块最大生成 token
$temperature = 0.0                         # 0 为贪心解码
$glossary_file = ""                        # 可选术语表文件
$source_version = ""                       # 指定读取的 Lance tag/version
$raw_tag = ""                              # 可选 raw tag
$norm_tag = ""                             # 可选 norm tag
$translation_tag = ""                      # 可选 tr tag
$force_reimport = $false                   # 即使已有 .lance 也重新导入
$skip_normalize = $false                   # 直接把 source_version 当作 norm 版本
$normalize_only = $false                   # 只做规范化，不翻译
$no_export = $false                        # 不导出 *_lang.md 文件
$runtime_backend = ""                      # "", "direct", "openai"

# OpenAI Compatible API 配置（仅在 runtime_backend="openai" 时使用）
# - 共享传输参数，不再单独维护 runtime_base_url/runtime_api_key
# - openai_model_name 为空时，默认复用上面的 $model_id
$openai_api_key = ""                       # API 密钥（本地服务可填任意值，如 "sk-no-key"）
$openai_base_url = ""                      # 服务器地址，如 http://127.0.0.1:8000/v1
$openai_model_name = ""                    # 可选：单独覆盖 OpenAI-compatible server 上的模型名

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
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

$ext_args = [System.Collections.ArrayList]::new()
$uv_args = [System.Collections.ArrayList]::new()

[void]$uv_args.Add("--extra=translate")

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

function Has-Value($value) {
  return -not [string]::IsNullOrWhiteSpace([string]$value)
}

if ($output_name) {
  [void]$ext_args.Add("--output_name=$output_name")
}

if ($model_id) {
  [void]$ext_args.Add("--model_id=$model_id")
}

if ($runtime_backend) {
  [void]$ext_args.Add("--runtime_backend=$runtime_backend")
}

if ($openai_api_key) {
  [void]$ext_args.Add("--openai_api_key=$openai_api_key")
}

if ($openai_base_url) {
  [void]$ext_args.Add("--openai_base_url=$openai_base_url")
}

if ($openai_model_name) {
  [void]$ext_args.Add("--openai_model_name=$openai_model_name")
}

if ($source_lang) {
  [void]$ext_args.Add("--source_lang=$source_lang")
}

if ($target_lang) {
  [void]$ext_args.Add("--target_lang=$target_lang")
}

if ((Has-Value $max_chars) -and $max_chars -ne 2200) {
  [void]$ext_args.Add("--max_chars=$max_chars")
}

if ((Has-Value $context_chars) -and $context_chars -ne 300) {
  [void]$ext_args.Add("--context_chars=$context_chars")
}

if ((Has-Value $max_new_tokens) -and $max_new_tokens -ne 2048) {
  [void]$ext_args.Add("--max_new_tokens=$max_new_tokens")
}

if ((Has-Value $temperature) -and $temperature -ne 0.0) {
  [void]$ext_args.Add("--temperature=$temperature")
}

if ($glossary_file) {
  [void]$ext_args.Add("--glossary_file=$glossary_file")
}

if ($source_version) {
  [void]$ext_args.Add("--source_version=$source_version")
}

if ($raw_tag) {
  [void]$ext_args.Add("--raw_tag=$raw_tag")
}

if ($norm_tag) {
  [void]$ext_args.Add("--norm_tag=$norm_tag")
}

if ($translation_tag) {
  [void]$ext_args.Add("--translation_tag=$translation_tag")
}

if ($force_reimport) {
  [void]$ext_args.Add("--force_reimport")
}

if ($skip_normalize) {
  [void]$ext_args.Add("--skip_normalize")
}

if ($normalize_only) {
  [void]$ext_args.Add("--normalize_only")
}

if ($no_export) {
  [void]$ext_args.Add("--no_export")
}

Write-Output "Starting text/document translation pipeline..."
Install-UvExtraPatch @("translate")
Write-Output "runtime target environment: $(Get-UvEnvName)"
Write-Output "runtime dependency profile: extra:translate"
python "./module/texttranslate.py" $input_path $ext_args

Write-Output "Translation pipeline finished"
Read-Host | Out-Null
