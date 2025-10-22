#region Configuration
# Model settings
$Config = @{
    train_data_dir     = "./datasets"                                   # Input images path | 图片输入路径
    # bdsqlsz/Watermark-Detection-SigLIP2-onnx
    # bdsqlsz/joycaption-watermark-detection-onnx
    repo_id            = "bdsqlsz/joycaption-watermark-detection-onnx"  # Model repo ID from Hugging Face
    model_dir          = "watermark_detection"                       # Local model folder path | 本地模型文件夹路径
    batch_size         = 12                                          # Batch size for inference
    thresh             = 1.0                                         # Concept threshold
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
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
#$Env:CUDA_VISIBLE_DEVICES = "1"  # 设置GPU id，0表示使用第一个GPU，-1表示不使用GPU

#endregion

#region Build Arguments
$ExtArgs = [System.Collections.ArrayList]::new()

# Add configuration arguments
if ($Config.repo_id) { [void]$ExtArgs.Add("--repo_id=$($Config.repo_id)") }
if ($Config.model_dir) { [void]$ExtArgs.Add("--model_dir=$($Config.model_dir)") }
if ($Config.batch_size) { [void]$ExtArgs.Add("--batch_size=$($Config.batch_size)") }
if ($Config.thresh -ne 1.0) { [void]$ExtArgs.Add("--thresh=$($Config.thresh)") }

#endregion

#region Execute Watermark Detection
Write-Output "Starting Watermark Detection..."

# Run tagger
uv run "./module/waterdetect.py" `
    $Config.train_data_dir `
    $ExtArgs

Write-Output "Watermark Detection finished"
Read-Host | Out-Null

#endregion
