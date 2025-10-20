$dataset_path = "./datasets"
$pair_dir = ""
$gemini_api_key = ""
$gemini_model_path = "gemini-2.5-pro"
$gemini_task = ""
$pixtral_api_key = ""
$pixtral_model_path = "pixtral-large-2411"
$step_api_key = ""
$step_model_path = "step-1.5v-mini"
$qwenVL_api_key = ""
$qwenVL_model_path = "qwen-vl-max-latest" # qwen2.5-vl-72b-instruct<10mins qwen-vl-max-latest <1min
$glm_api_key = ""
$glm_model_path = "GLM-4V-Plus-0111"
$dir_name = $false
$mode = "long"
$not_clip_with_caption = $false              # Not clip with caption | 不根据caption裁剪
$wait_time = 1
$max_retries = 100
$segment_time = 600
$ocr = $false
$document_image = $true
$scene_detector = "AdaptiveDetector" # from ["ContentDetector","AdaptiveDetector","HashDetector","HistogramDetector","ThresholdDetector"]
$scene_threshold = 0.0 # default value ["ContentDetector": 27.0, "AdaptiveDetector": 3.0, "HashDetector": 0.395, "HistogramDetector": 0.05, "ThresholdDetector": 12]
$scene_min_len = 15
$scene_luma_only = $false
$tags_highlightrate = 0.38

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

if ($pair_dir) {
  [void]$ext_args.Add("--pair_dir=$pair_dir")
}

if ($gemini_api_key) {
  [void]$ext_args.Add("--gemini_api_key=$gemini_api_key")
  if ($gemini_task) {
    [void]$ext_args.Add("--gemini_task=$gemini_task")
  }
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

if ($step_api_key) {
  [void]$ext_args.Add("--step_api_key=$step_api_key")
}

if ($step_model_path) {
  [void]$ext_args.Add("--step_model_path=$step_model_path")
}

if ($qwenVL_api_key) {
  [void]$ext_args.Add("--qwenVL_api_key=$qwenVL_api_key")
}

if ($qwenVL_model_path) {
  [void]$ext_args.Add("--qwenVL_model_path=$qwenVL_model_path")
}

if ($glm_api_key) {
  [void]$ext_args.Add("--glm_api_key=$glm_api_key")
}

if ($glm_model_path) {
  [void]$ext_args.Add("--glm_model_path=$glm_model_path")
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

if ($wait_time -ine 1) {
  [void]$ext_args.Add("--wait_time=$wait_time")
}

if ($max_retries -ine 20) {
  [void]$ext_args.Add("--max_retries=$max_retries")
}

if ($segment_time -ine 600) {
  [void]$ext_args.Add("--segment_time=$segment_time")
}

if ($ocr) {
  [void]$ext_args.Add("--ocr")
}

if ($document_image) {
  [void]$ext_args.Add("--document_image")
}

if ($scene_detector -ne "AdaptiveDetector") {
  [void]$ext_args.Add("--scene_detector=$($scene_detector)")
}

if ($scene_threshold -ne 0.0) {
  [void]$ext_args.Add("--scene_threshold=$scene_threshold")
}

if ($scene_min_len -ne 15) {
  [void]$ext_args.Add("--scene_min_len=$scene_min_len")
}

if ($scene_luma_only) {
  [void]$ext_args.Add("--scene_luma_only")
}

if ($tags_highlightrate -ne 0.4) {
  [void]$ext_args.Add("--tags_highlightrate=$tags_highlightrate")
}

# run train
uv run -m module.captioner $dataset_path $ext_args

Write-Output "Captioner finished"
Read-Host | Out-Null ;
