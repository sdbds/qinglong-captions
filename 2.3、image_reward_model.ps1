#region Configuration
# Model settings
$Config = @{
    train_data_dir     = "./datasets"                                   # Input images path | 图片输入路径
    # "RE-N-Y/aesthetic-shadow-v2"
    # "RE-N-Y/clipscore-vit-large-patch14"
    # "RE-N-Y/pickscore"
    # "yuvalkirstain/PickScore_v1"
    # "RE-N-Y/mpsv1"
    # "RE-N-Y/hpsv21"
    # "RE-N-Y/ImageReward"
    # "RE-N-Y/laion-aesthetic"
    # "NagaSaiAbhinay/CycleReward-Combo"
    # "NagaSaiAbhinay/CycleReward-T2I"
    # "NagaSaiAbhinay/CycleReward-I2T"
    # "RE-N-Y/clip-t5-xxl"
    # "RE-N-Y/evalmuse"
    # "RE-N-Y/hpsv3"
    # "RE-N-Y/pickscore-siglip"
    # "RE-N-Y/pickscore-clip"
    # "RE-N-Y/imreward-fidelity_rating-siglip"
    # "RE-N-Y/imreward-fidelity_rating-clip"
    # "RE-N-Y/imreward-fidelity_rating-dinov2"
    # "RE-N-Y/imreward-overall_rating-siglip"
    # "RE-N-Y/imreward-overall_rating-clip"
    # "RE-N-Y/imreward-overall_rating-dinov2"
    # "RE-N-Y/ava-rating-clip-sampled-True"
    # "RE-N-Y/ava-rating-clip-sampled-False"
    # "RE-N-Y/ava-rating-siglip-sampled-True"
    # "RE-N-Y/ava-rating-siglip-sampled-False"
    # "RE-N-Y/ava-rating-dinov2-sampled-True"
    # "RE-N-Y/ava-rating-dinov2-sampled-False"
    repo_id            = "RE-N-Y/evalmuse"                          # Model repo ID from Hugging Face
    batch_size         = 1                                          # Batch size for inference
    device             = "cuda"                                     # Device to use for inference
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
$Env:HF_ENDPOINT = "https://hf-mirror.com"
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
if ($Config.batch_size) { [void]$ExtArgs.Add("--batch_size=$($Config.batch_size)") }
if ($Config.device) { [void]$ExtArgs.Add("--device=$($Config.device)") }

#endregion

#region Execute Reward Model
Write-Output "Starting Reward Model..."

# Run tagger
accelerate launch --num_cpu_threads_per_process=8 "./module/rewardmodel.py" `
    $Config.train_data_dir `
    $ExtArgs

Write-Output "Reward Model finished"
Read-Host | Out-Null

#endregion
