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
$pixtral_ocr = $false
$document_image = $true

$paddle_ocr = $false

$deepseek_ocr = $false

# document = "<image>\nConvert the document to markdown."

# other_image = "<image>\nOCR this image."

# without_layouts = "<image>\nFree OCR."

# figures_in_document = "<image>\nParse the figure."

# general = "<image>\nDescribe this image in detail."

#rec = "<image>\nLocate <ref>xxxx</ref> in the image."
$deepseek_ocr_prompt = "" 

$deepseek_base_size = 1280
$deepseek_image_size = 1280
$deepseek_crop_mode = $false
$deepseek_test_compress = $false

$scene_detector = "AdaptiveDetector" # from ["ContentDetector","AdaptiveDetector","HashDetector","HistogramDetector","ThresholdDetector"]
$scene_threshold = 0.0 # default value ["ContentDetector": 27.0, "AdaptiveDetector": 3.0, "HashDetector": 0.395, "HistogramDetector": 0.05, "ThresholdDetector": 12]
$scene_min_len = 15
$scene_luma_only = $false
$tags_highlightrate = 0.38

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot$([System.IO.Path]::PathSeparator)$($env:PYTHONPATH)"
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

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG = "1"
#$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = "1"
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
$Env:UV_INDEX_STRATEGY = "unsafe-best-match"
#$Env:CUDA_VISIBLE_DEVICES = "1"  # 设置GPU id，0表示使用第一个GPU，-1表示不使用GPU

#$Env:HTTP_PROXY = "http://127.0.0.1:7890"
#$Env:HTTPS_PROXY = "http://127.0.0.1:7890"

$ext_args = [System.Collections.ArrayList]::new()
$uv_args = [System.Collections.ArrayList]::new()

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

if ($ark_api_key) {
  [void]$ext_args.Add("--ark_api_key=$ark_api_key")
}

if ($ark_model_path) {
  [void]$ext_args.Add("--ark_model_path=$ark_model_path")
}

if ($ark_fps) {
  [void]$ext_args.Add("--ark_fps=$ark_fps")
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

if ($pixtral_ocr) {
  [void]$ext_args.Add("--pixtral_ocr")
  if ($document_image) {
    [void]$ext_args.Add("--document_image")
  }
}
elseif ($paddle_ocr) {
  [void]$ext_args.Add("--paddle_ocr")
  $Env:UV_EXTRA_INDEX_URL = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
  if ($os -eq "Windows") {
    [void]$uv_args.Add("--with-requirements=requirements-paddleocr.txt")
  }else{
    uv pip install -r requirements-uv-paddleocr-linux.txt
  }
}
elseif ($deepseek_ocr) {
  [void]$ext_args.Add("--deepseek_ocr")
  [void]$uv_args.Add("--with-requirements=requirements-deepseekocr.txt")
  if ($deepseek_base_size -ine 1024) {
    [void]$ext_args.Add("--deepseek_base_size=$deepseek_base_size")
  }

  if ($deepseek_image_size -ine 640) {
    [void]$ext_args.Add("--deepseek_image_size=$deepseek_image_size")
  }

  if ($deepseek_crop_mode) {
    [void]$ext_args.Add("--deepseek_crop_mode")
  }

  if ($deepseek_test_compress) {
    [void]$ext_args.Add("--deepseek_test_compress")
  }
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
uv run $uv_args "./module/captioner.py" $dataset_path $ext_args

Write-Output "Captioner finished"
Read-Host | Out-Null ;
