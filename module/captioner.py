import argparse
import sys

from PIL import Image
from rich.console import Console
from module.caption_pipeline.orchestrator import process_batch as _pipeline_process_batch
from module.providers.catalog import normalize_runtime_args, route_choices
from module.api_handler_v2 import api_process_batch as _api_process_batch_v2


def api_process_batch(uri, mime, config, args, sha256hash, progress=None, task_id=None):
    """将 V2 CaptionResult 转为旧编排层使用的基础类型。"""
    result = _api_process_batch_v2(
        uri=uri,
        mime=mime,
        config=config,
        args=args,
        sha256hash=sha256hash,
        progress=progress,
        task_id=task_id,
    )
    if hasattr(result, "parsed") and result.parsed is not None:
        return result.parsed
    if hasattr(result, "raw"):
        return result.raw
    return result
from module.lanceexport import extract_from_lance
from module.lanceImport import transform2lance

Image.MAX_IMAGE_PIXELS = None  # Disable image size limit check

console = Console(color_system="truecolor", force_terminal=True)


def process_batch(args, config):
    return _pipeline_process_batch(
        args,
        config,
        api_process_batch_fn=api_process_batch,
        transform2lance_fn=transform2lance,
        extract_from_lance_fn=extract_from_lance,
        console_obj=console,
    )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("dataset_dir", type=str, help="directory for dataset")

    parser.add_argument(
        "--pair_dir",
        type=str,
        default="",
        help="directory for pair dataset",
    )

    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default="",
        help="API key for gemini API",
    )

    parser.add_argument(
        "--gemini_model_path",
        type=str,
        default="gemini-exp-1206",
        help="Model path for gemini",
    )

    parser.add_argument(
        "--step_api_key",
        type=str,
        default="",
        help="API key for step API",
    )

    parser.add_argument(
        "--kimi_api_key",
        type=str,
        default="",
        help="API key for Moonshot(Kimi) OpenAI-compatible API",
    )

    parser.add_argument(
        "--kimi_model_path",
        type=str,
        default="kimi-k2.5",
        help="Model path for Kimi",
    )

    parser.add_argument(
        "--kimi_base_url",
        type=str,
        default="https://api.moonshot.cn/v1",
        help="Base URL for Moonshot(Kimi) OpenAI-compatible API",
    )

    # Kimi-Code (独立 API，来自 kimi.com/code)
    parser.add_argument(
        "--kimi_code_api_key",
        type=str,
        default="",
        help="API key for Kimi-Code (from kimi.com/code, NOT platform.moonshot.cn)",
    )
    parser.add_argument(
        "--kimi_code_model_path",
        type=str,
        default="k2p5",
        help="Model name for Kimi-Code (default: k2p5)",
    )
    parser.add_argument(
        "--kimi_code_base_url",
        type=str,
        default="https://api.kimi.com/coding/v1",
        help="Base URL for Kimi-Code API",
    )

    # MiniMax API
    parser.add_argument(
        "--minimax_api_key",
        type=str,
        default="",
        help="API key for MiniMax API (from platform.minimaxi.com)",
    )
    parser.add_argument(
        "--minimax_model_path",
        type=str,
        default="MiniMax-M2.5",
        help="Model name for MiniMax API (default: MiniMax-M2.5, options: MiniMax-M2.5, MiniMax-M2.5-highspeed, MiniMax-M2.1, MiniMax-M2.1-highspeed, MiniMax-M2)",
    )
    parser.add_argument(
        "--minimax_api_base_url",
        type=str,
        default="https://api.minimax.io/v1",
        help="Base URL for MiniMax API",
    )

    # MiniMax Code (针对代码和结构化输出优化)
    parser.add_argument(
        "--minimax_code_api_key",
        type=str,
        default="",
        help="API key for MiniMax Code API (from platform.minimaxi.com)",
    )
    parser.add_argument(
        "--minimax_code_model_path",
        type=str,
        default="MiniMax-M2.5",
        help="Model name for MiniMax Code API (default: MiniMax-M2.5, optimized for coding and structured output)",
    )
    parser.add_argument(
        "--minimax_code_base_url",
        type=str,
        default="https://api.minimax.io/v1",
        help="Base URL for MiniMax Code API",
    )

    parser.add_argument(
        "--step_model_path",
        type=str,
        default="step-1.5v-mini",
        help="video model for step",
    )

    parser.add_argument(
        "--qwenVL_api_key",
        type=str,
        default="",
        help="API key for qwenVL API",
    )

    parser.add_argument(
        "--qwenVL_model_path",
        type=str,
        default="qwen-vl-max-latest",
        help="video model for qwenVL",
    )

    parser.add_argument(
        "--mistral_api_key",
        type=str,
        default="",
        help="API key for Mistral OCR API",
    )

    parser.add_argument(
        "--mistral_model_path",
        type=str,
        default="mistral-large-latest",
        help="Model path for Mistral OCR",
    )

    parser.add_argument(
        "--pixtral_api_key",
        type=str,
        default="",
        help="Deprecated alias for --mistral_api_key",
    )

    parser.add_argument(
        "--pixtral_model_path",
        type=str,
        default="",
        help="Deprecated alias for --mistral_model_path",
    )

    parser.add_argument(
        "--vlm_image_model",
        type=str,
        choices=route_choices("vlm_image_model", include_aliases=True),
        default="",
        help="VLM model for image/video tasks (default: empty)",
    )

    parser.add_argument(
        "--glm_api_key",
        type=str,
        default="",
        help="API key for glm API",
    )

    parser.add_argument(
        "--glm_model_path",
        type=str,
        default="glm-4v-plus-0111",
        help="Model path for glm",
    )

    # Ark (Volcano Engine) options
    parser.add_argument(
        "--ark_api_key",
        type=str,
        default="",
        help="API key for Ark (Volcano Engine) API",
    )
    parser.add_argument(
        "--ark_model_path",
        type=str,
        default="",
        help="Model ID for Ark chat.completions (e.g. your EP model id)",
    )

    # OpenAI Compatible (通用 OpenAI 接口)
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="",
        help="API key for OpenAI-compatible API (vLLM, Ollama, LM Studio, etc.)",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default="",
        help="Base URL for OpenAI-compatible API (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--openai_model_name",
        type=str,
        default="",
        help="Model name for OpenAI-compatible API (e.g., Qwen2-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--local_runtime_backend",
        type=str,
        choices=["", "direct", "openai"],
        default="",
        help="Backend for local transformers providers: empty=use config, direct=in-process, openai=OpenAI-compatible local server",
    )

    parser.add_argument(
        "--dir_name",
        action="store_true",
        help="Use the directory name as the dataset name",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Mode for processing the dataset",
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="config",
        help="Path to config directory containing prompts.toml, model.toml, general.toml",
    )

    parser.add_argument(
        "--not_clip_with_caption",
        action="store_true",
        help="Not clip with caption",
    )

    parser.add_argument(
        "--wait_time",
        type=int,
        default=1,
        help="Wait time",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=20,
        help="Max retries",
    )

    parser.add_argument(
        "--segment_time",
        type=int,
        default=None,
        help="Segment time (provider-specific default when unset)",
    )

    parser.add_argument(
        "--ocr_model",
        type=str,
        choices=route_choices("ocr_model", include_aliases=True),
        default="",
        help="OCR model to use for text extraction (default: empty)",
    )

    parser.add_argument(
        "--alm_model",
        type=str,
        choices=route_choices("alm_model", include_aliases=True),
        default="",
        help="Audio language model to use for local audio captioning (default: empty)",
    )
    parser.add_argument(
        "--alm_language",
        type=str,
        default=None,
        help="Language code hint for ALM transcription tasks (for example: zh, en, ja)",
    )

    parser.add_argument(
        "--document_image",
        action="store_true",
        help="Use OCR to extract image from document",
    )

    parser.add_argument(
        "--scene_detector",
        type=str,
        choices=[
            "ContentDetector",
            "AdaptiveDetector",
            "HashDetector",
            "HistogramDetector",
            "ThresholdDetector",
        ],
        default="AdaptiveDetector",
        help="Detector to use for scene detection",
    )

    parser.add_argument(
        "--scene_threshold",
        type=float,
        default=0.0,
        help="Threshold for scene detection",
    )

    parser.add_argument(
        "--scene_min_len",
        type=int,
        default=15,
        help="Minimum length(frames) for scene detection",
    )

    parser.add_argument(
        "--scene_luma_only",
        action="store_true",
        help="Only use luma (brightness) without color changes for scene detection.",
    )

    parser.add_argument(
        "--scene_detection_timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for async scene detection before falling back to raw subtitle timing.",
    )

    parser.add_argument(
        "--gemini_task",
        type=str,
        default="",
        help="Task for gemini-2.0-flash-exp",
    )

    parser.add_argument(
        "--tags_highlightrate",
        type=float,
        default=0.4,
        help="tags_highlightrate for check captions",
    )

    parser.add_argument(
        "--merge_batch_size",
        type=int,
        default=100,
        help="Batch size for merge_insert to avoid memory overflow on large datasets (default: 100)",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    normalize_runtime_args(args)

    from config.runtime_config import load_runtime_config

    config = load_runtime_config(args.config_dir)

    process_batch(args, config)
