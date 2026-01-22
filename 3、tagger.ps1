#region Configuration
# Model settings
$Config = @{
    train_data_dir     = "./datasets"                                # Input images path | 图片输入路径
    repo_id            = "cella110n/cl_tagger"      # Model repo ID from Hugging Face
    model_dir          = "wd14_tagger_model"                         # Local model folder path | 本地模型文件夹路径
    batch_size         = 12                                          # Batch size for inference
    thresh             = 0.6                                         # Concept threshold
    general_threshold  = 0.55                                         # General threshold
    character_threshold = 1.0                                        # Character threshold
}

# Feature flags
$Features = @{
    frequency_tags           = $false     # Order by frequency tags
    remove_underscore        = $true      # Convert underscore to space
    use_rating_tags          = $true      # Use rating tags
    use_quality_tags         = $false      # Use quality tags
    use_model_tags           = $false      # Use model tags
    use_rating_tags_as_last_tag = $false  # Put rating tags at the end
    character_tags_first     = $false     # Put character tags first
    character_tag_expand     = $false     # Split character_(series) into character, series
    remove_parents_tag       = $true      # Remove parent tags
    overwrite            = $true          # Overwrite existing tag files
    add_tags_threshold = $false           # Add tags threshold
}

# Tag settings
$TagConfig = @{
    undesired_tags    = ""  # Tags to exclude
    always_first_tags = "1girl,1boy,2girls,3girls,4girls,5girls,6girls,2boys,3boys,4boys,5boys,6boys"
    tag_replacement   = "1girl,1woman;2girls,2women;3girls,3women;4girls,4women;5girls,5women;1boy,1man"
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
$uv_args = [System.Collections.ArrayList]::new()

# Add configuration arguments
if ($Config.repo_id) { [void]$ExtArgs.Add("--repo_id=$($Config.repo_id)") }
if ($Config.model_dir) { [void]$ExtArgs.Add("--model_dir=$($Config.model_dir)") }
if ($Config.batch_size) { [void]$ExtArgs.Add("--batch_size=$($Config.batch_size)") }
if ($Config.general_threshold) { [void]$ExtArgs.Add("--general_threshold=$($Config.general_threshold)") }
if ($Config.character_threshold) { [void]$ExtArgs.Add("--character_threshold=$($Config.character_threshold)") }

# Add feature flags
if ($Features.remove_underscore) { [void]$ExtArgs.Add("--remove_underscore") }
if ($Features.recursive) { [void]$ExtArgs.Add("--recursive") }
if ($Features.frequency_tags) { [void]$ExtArgs.Add("--frequency_tags") }
if ($Features.character_tags_first) { [void]$ExtArgs.Add("--character_tags_first") }
if ($Features.character_tag_expand) { [void]$ExtArgs.Add("--character_tag_expand") }
if ($Features.use_rating_tags_as_last_tag) { [void]$ExtArgs.Add("--use_rating_tags_as_last_tag") }
elseif ($Features.use_rating_tags) { [void]$ExtArgs.Add("--use_rating_tags") }
if ($Features.use_quality_tags) { [void]$ExtArgs.Add("--use_quality_tags") }
if ($Features.use_model_tags) { [void]$ExtArgs.Add("--use_model_tags") }
if ($Features.remove_parents_tag) { [void]$ExtArgs.Add("--remove_parents_tag") }
if ($Features.overwrite) { [void]$ExtArgs.Add("--overwrite") }
if ($Features.add_tags_threshold) { [void]$ExtArgs.Add("--add_tags_threshold") }

# Add tag configuration
if ($TagConfig.undesired_tags) { [void]$ExtArgs.Add("--undesired_tags=$($TagConfig.undesired_tags)") }
if ($TagConfig.always_first_tags) { [void]$ExtArgs.Add("--always_first_tags=$($TagConfig.always_first_tags)") }
if ($TagConfig.tag_replacement) { [void]$ExtArgs.Add("--tag_replacement=$($TagConfig.tag_replacement)") }

#endregion

#region Execute Tagger
Write-Output "Starting tagger..."

# Get-ChildItem -Path $env:AGENT_TOOLSDIRECTORY -File -Include msvcp*.dll,concrt*.dll,vccorlib*.dll,vcruntime*.dll -Recurse | Remove-Item -Force -Verbose

if ($os -eq "Windows") {
    [void]$uv_args.Add("--with-requirements=requirements-wdtagger.txt")
}
else {
    uv pip install -r requirements-wdtagger.txt
}

# Run tagger
uv run $uv_args "./utils/wdtagger.py" `
    $Config.train_data_dir `
    --thresh=$($Config.thresh) `
    --caption_extension .txt `
    $ExtArgs

Write-Output "Tagger finished"
Read-Host | Out-Null

#endregion
