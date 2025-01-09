$dataset_path = "./datasets"
$gemini_api_key = ""
$gemini_model_path = "gemini-exp-1206"
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$dir_name = $true
$mode = "long"
$not_clip_with_caption = $false              # Not clip with caption | 不根据caption裁剪

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG = "1"
$ext_args = [System.Collections.ArrayList]::new()
#$Env:HTTP_PROXY = "http://127.0.0.1:7890"
#$Env:HTTPS_PROXY = "http://127.0.0.1:7890"



if ($gemini_api_key) {
  [void]$ext_args.Add("--gemini_api_key=$gemini_api_key")
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

if ($dir_name) {
  [void]$ext_args.Add("--dir_name")
}

if ($mode -ine "all") {
  [void]$ext_args.Add("--mode=$mode")
}

if ($not_clip_with_caption) {
  [void]$ext_args.Add("--not_clip_with_caption")
}

# run train
python captioner.py $dataset_path $ext_args

Write-Output "Captioner finished"
Read-Host | Out-Null ;