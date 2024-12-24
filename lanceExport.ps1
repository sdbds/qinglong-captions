$lance_file = "D:\lora-scripts\input\train\aijou karen\dataset.lance" # input images path | 图片输入路径
$output_dir = "D:\lora-scripts\input\train\aijou karen"
$version = "WDtagger" #WDtagger pixtral

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
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$ext_args = [System.Collections.ArrayList]::new()


if ($api_key) {
  [void]$ext_args.Add("--api_key=$api_key")
}

if ($version) {
  [void]$ext_args.Add("--version=$version")
}

# run train
python "./lanceexport.py" `
  $lance_file `
  --output_dir=$output_dir $ext_args

Write-Output "Export finished"
Read-Host | Out-Null