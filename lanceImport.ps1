# Input parameters
$train_data_dir = "./datasets" # input images path | 图片输入路径
$caption_dir = $null # Optional caption files directory | 可选的描述文件目录
$output_name = "dataset" # Output dataset name | 输出数据集名称
$no_save_binary = $false # Don't save binary data | 不保存二进制数据
$not_save_disk = $false # Load into memory only | 仅加载到内存
$import_mode = 0 # Video import mode: 0=All, 1=Video only, 2=Audio only, 3=Split | 视频导入模式
$tag = "gemini" # Dataset tag | 数据集标签

# Activate virtual environment
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

# Run the import script
$args = @(
  $train_data_dir
)
if ($caption_dir) { $args += "--caption_dir=$caption_dir" }
if ($output_name) { $args += "--output_name=$output_name" }
if ($no_save_binary) { $args += "--no_save_binary" }
if ($not_save_disk) { $args += "--not_save_disk" }
$args += "--import_mode=$import_mode"
$args += "--tag=$tag"

python -m module.lanceImport @args

Write-Output "Import finished"
Read-Host | Out-Null