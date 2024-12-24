$train_data_dir = "D:\lora-scripts\input\train\aijou karen" # input images path | 图片输入路径
$output_name = "dataset"


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

python "./lancedatasets.py" `
  $train_data_dir `
  --output_name=$output_name

Write-Output "Import finished"
Read-Host | Out-Null