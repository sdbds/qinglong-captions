#region Configuration
# Model settings
$Config = @{
    train_data_dir     = "./datasets"                                   # Input images path | 图片输入路径
    # bdsqlsz/Watermark-Detection-SigLIP2-onnx
    # bdsqlsz/joycaption-watermark-detection-onnx
    repo_id            = "bdsqlsz/joycaption-watermark-detection-onnx"  # Model repo ID from Hugging Face
    model_dir          = "watermark_detection"                       # Local model folder path | 本地模型文件夹路径
    batch_size         = 8                                          # Batch size for inference
    thresh             = 0.5                                         # Concept threshold
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

#endregion

#region Build Arguments
$ExtArgs = [System.Collections.ArrayList]::new()

# Add configuration arguments
if ($Config.repo_id) { [void]$ExtArgs.Add("--repo_id=$($Config.repo_id)") }
if ($Config.model_dir) { [void]$ExtArgs.Add("--model_dir=$($Config.model_dir)") }
if ($Config.batch_size) { [void]$ExtArgs.Add("--batch_size=$($Config.batch_size)") }
if ($Config.thresh -ne 0.5) { [void]$ExtArgs.Add("--thresh=$($Config.thresh)") }

#endregion

#region Execute Watermark Detection
Write-Output "Starting Watermark Detection..."

# Run tagger
accelerate launch --num_cpu_threads_per_process=8 "./module/waterdetect.py" `
    $Config.train_data_dir `
    $ExtArgs

Write-Output "Watermark Detection finished"
Read-Host | Out-Null

#endregion
