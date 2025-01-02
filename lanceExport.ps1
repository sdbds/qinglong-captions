# Input parameters | 输入参数
param(
    [string]$lance_file = "./datasets/dataset.lance",  # Lance dataset path | Lance数据集路径
    [string]$output_dir = "./datasets",                # Output directory | 输出目录
    [string]$version = "gemini"                                                   # Dataset version (gemini/WDtagger/pixtral)
)

# Set working directory | 设置工作目录
Set-Location $PSScriptRoot

# Activate virtual environment | 激活虚拟环境
$venvPaths = @(
    "venv/Scripts/activate",
    ".venv/Scripts/activate",
    "venv/bin/Activate.ps1",
    ".venv/bin/activate.ps1"
)

$activated = $false
foreach ($path in $venvPaths) {
    if (Test-Path $path) {
        Write-Output "Activating virtual environment: $path"
        & $path
        $activated = $true
        break
    }
}

if (-not $activated) {
    Write-Error "No virtual environment found. Please create one first."
    exit 1
}

# Set environment variables | 设置环境变量
$env:HF_HOME = "huggingface"
$env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Prepare arguments | 准备参数
$arguments = @(
    "./lanceexport.py",
    $lance_file,
    "--output_dir=$output_dir"
)

if ($version) {
    $arguments += "--version=$version"
}

# Run export script | 运行导出脚本
Write-Output "Starting export from $lance_file to $output_dir"
& python $arguments

# Wait for user input before closing | 等待用户输入后关闭
Write-Output "`nExport finished. Press Enter to exit..."
$null = Read-Host