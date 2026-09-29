import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from module.captioner import setup_parser
from module.providers.base import PromptContext, ProviderContext
from module.providers.vision_api.gemini import GeminiProvider
from module.providers.vision_api_base import StructuredOutputConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


def _load_caption_step():
    module_path = ROOT / "gui" / "wizard" / "step4_caption.py"
    spec = importlib.util.spec_from_file_location("api_model_catalog_step4", module_path)
    step4_caption = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    gui_path = str(ROOT / "gui")
    original_sys_path = list(sys.path)
    if gui_path not in sys.path:
        sys.path.append(gui_path)
    try:
        spec.loader.exec_module(step4_caption)
    finally:
        sys.path[:] = original_sys_path

    return step4_caption.CaptionStep


def test_api_catalog_contains_current_multimodal_models_only():
    configs = _load_caption_step().API_CONFIGS

    assert configs["Gemini"]["models"] == [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
    ]
    assert configs["Mistral"]["models"] == [
        "mistral-medium-3-5",
        "mistral-small-2603",
        "mistral-large-2512",
        "ministral-14b-2512",
        "ministral-8b-2512",
        "ministral-3b-2512",
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
    ]
    assert configs["Step"]["models"] == [
        "step-5-preview",
        "step-3.7-flash",
        "step-3",
        "step-r1-v-mini",
        "step-1o-turbo-vision",
        "step-1o-vision-32k",
    ]
    assert configs["Kimi"]["models"] == [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "kimi-k2.6",
    ]
    assert configs["MiMo"]["models"] == [
        "mimo-v2.6-flash",
        "mimo-v2.6-pro",
        "mimo-v2.6-pro-ultraspeed",
        "mimo-v2.5",
    ]
    assert configs["MiniMax"]["models"] == ["MiniMax-M3"]
    assert configs["MiniMax-Code"]["models"] == [
        "MiniMax-M3.1-Flash-Preview",
        "MiniMax-M3",
    ]
    assert configs["GLM"]["models"] == [
        "glm-5.3-flashx",
        "glm-5.3-flash",
        "glm-5v-turbo",
        "glm-4.6v",
        "glm-4.6v-flash",
        "glm-4.6v-flashx",
        "glm-4.5v",
    ]
    assert configs["Ark"]["models"] == [
        "doubao-seed-2-1-pro-260915",
        "doubao-seed-2-1-lite-260915",
        "doubao-seed-2-0-pro-260215",
        "doubao-seed-2-0-lite-260428",
        "doubao-seed-2-0-mini-260428",
    ]


def test_qwen_catalog_tracks_current_video_capable_families():
    models = _load_caption_step().API_CONFIGS["Qwen"]["models"]

    assert models[:9] == [
        "qwen3.8-omni-flash",
        "qwen3.8-max",
        "qwen3.8-max-0902",
        "qwen3.8-flash",
        "qwen3.7-plus",
        "qwen3.7-flash",
        "qwen3.7-max-2026-06-08",
        "qwen3.7-plus-2026-05-26",
        "qwen3.7-flash-2026-07-15",
    ]
    assert {"qwen3.6-plus", "qwen3.6-flash", "qwen3-vl-plus", "qwen3-vl-flash"} <= set(models)
    assert not any("ocr" in model for model in models)


@pytest.mark.parametrize(
    ("api_name", "model_id", "model_arg"),
    [
        ("Gemini", "gemini-3.8-flash", "gemini_model_path"),
        ("Step", "step-5-preview", "step_model_path"),
        ("Qwen", "qwen3.8-omni-flash", "qwenVL_model_path"),
        ("MiMo", "mimo-v2.6-flash", "mimo_model_path"),
        ("MiniMax", "MiniMax-M3", "minimax_model_path"),
        ("MiniMax-Code", "MiniMax-M3.1-Flash-Preview", "minimax_code_model_path"),
        ("GLM", "glm-5.3-flashx", "glm_model_path"),
        ("Ark", "doubao-seed-2-1-pro-260915", "ark_model_path"),
    ],
)
def test_new_api_model_selection_reaches_captioner(api_name, model_id, model_arg):
    step = _load_caption_step()()
    config = step.API_CONFIGS[api_name]
    assert model_id in config["models"]

    key_name = config["key_name"]
    step.api_keys[key_name] = SimpleNamespace(value="test-key")
    setattr(step, f"{key_name}_model", SimpleNamespace(value=model_id))
    step.pair_dir = SimpleNamespace(value="")
    step.mode = SimpleNamespace(value="long")
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")

    args = setup_parser().parse_args(step._build_caption_args("dataset"))

    assert getattr(args, model_arg) == model_id
    assert getattr(args, key_name) == "test-key"


def test_cli_defaults_match_gui_catalog_defaults():
    configs = _load_caption_step().API_CONFIGS
    args = setup_parser().parse_args(["dataset"])

    assert args.gemini_model_path == configs["Gemini"]["default_model"] == "gemini-3.6-flash"
    assert args.step_model_path == configs["Step"]["default_model"]
    assert args.qwenVL_model_path == configs["Qwen"]["default_model"] == "qwen3.7-plus"
    assert args.mistral_model_path == configs["Mistral"]["default_model"]
    assert args.mimo_model_path == configs["MiMo"]["default_model"] == "mimo-v2.6-flash"
    assert args.minimax_model_path == configs["MiniMax"]["default_model"] == "MiniMax-M3"
    assert args.minimax_code_model_path == configs["MiniMax-Code"]["default_model"] == "MiniMax-M3"
    assert args.glm_model_path == configs["GLM"]["default_model"] == "glm-5v-turbo"


@pytest.mark.parametrize(
    ("model_path", "thinking_level"),
    [
        ("gemini-3.8-flash", "medium"),
        ("gemini-3.7-flash", "medium"),
        ("gemini-3.6-flash", "medium"),
        ("gemini-3.5-flash-lite", "minimal"),
        ("gemini-3.5-flash", "medium"),
        ("gemini-3.1-pro-preview", "high"),
    ],
)
def test_gemini_3_models_use_supported_generation_parameters(model_path, thinking_level):
    provider = GeminiProvider(
        ProviderContext(
            console=Console(),
            config={"generation_config": {"candidate_count": 1}},
            args=SimpleNamespace(gemini_task=""),
        )
    )
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "max_output_tokens": 1024,
        "thinking_level": thinking_level,
    }

    with (
        patch("google.genai.types.GenerateContentConfig", side_effect=lambda **kwargs: kwargs),
        patch("google.genai.types.ThinkingConfig", side_effect=lambda **kwargs: kwargs),
    ):
        config = provider._build_genai_config(
            PromptContext(system="system", user="user"),
            generation_config,
            StructuredOutputConfig(),
            model_path,
        )

    assert "temperature" not in config
    assert "top_p" not in config
    assert "top_k" not in config
    assert "candidate_count" not in config
    assert config["max_output_tokens"] == 1024
    assert config["thinking_config"] == {"thinking_level": thinking_level}


@pytest.mark.parametrize("model_path", ["gemini-3.8-flash", "gemini-3.7-flash"])
@pytest.mark.parametrize("config_path", ["config/config.toml", "config/model.toml"])
def test_current_gemini_flash_models_load_supported_text_generation_config(config_path, model_path):
    provider = GeminiProvider(
        ProviderContext(
            console=Console(),
            config=tomllib.loads((ROOT / config_path).read_text(encoding="utf-8")),
            args=SimpleNamespace(gemini_model_path=model_path, gemini_task=""),
        )
    )

    with (
        patch("google.genai.types.GenerateContentConfig", side_effect=lambda **kwargs: kwargs),
        patch("google.genai.types.ThinkingConfig", side_effect=lambda **kwargs: kwargs),
    ):
        config = provider._build_genai_config(
            PromptContext(system="system", user="user"),
            provider._get_generation_config(),
            StructuredOutputConfig(),
            model_path,
        )

    assert config.get("thinking_config") == {"thinking_level": "medium"}
    assert config["max_output_tokens"] == 65536
    assert config["response_modalities"] == ["text"]
    assert config["response_mime_type"] == "text/plain"
    assert {"temperature", "top_p", "top_k", "candidate_count"}.isdisjoint(config)


def test_pre_gemini_3_models_keep_legacy_generation_parameters():
    provider = GeminiProvider(
        ProviderContext(
            console=Console(),
            config={"generation_config": {"candidate_count": 2}},
            args=SimpleNamespace(gemini_task=""),
        )
    )

    with (
        patch("google.genai.types.GenerateContentConfig", side_effect=lambda **kwargs: kwargs),
        patch("google.genai.types.ThinkingConfig", side_effect=lambda **kwargs: kwargs),
    ):
        config = provider._build_genai_config(
            PromptContext(system="system", user="user"),
            {"temperature": 0.2, "top_p": 0.8, "top_k": 20},
            StructuredOutputConfig(),
            "gemini-2.5-flash",
        )

    assert config["temperature"] == 0.2
    assert config["top_p"] == 0.8
    assert config["top_k"] == 20
    assert config["candidate_count"] == 2
    assert config["thinking_config"] == {"thinking_budget": -1}


@pytest.mark.parametrize("config_path", ["config/config.toml", "config/model.toml"])
def test_gemini_generation_config_removes_deprecated_models_and_parameters(config_path):
    config = tomllib.loads((ROOT / config_path).read_text(encoding="utf-8"))["generation_config"]
    model_configs = {key: value for key, value in config.items() if key.startswith("gemini-")}

    assert {"gemini-3_8-flash", "gemini-3_7-flash", "gemini-3_6-flash", "gemini-3_5-flash-lite"} <= set(model_configs)
    assert not any(key.startswith("gemini-2_5") for key in model_configs)
    assert "gemini-3-flash-preview" not in model_configs
    assert "gemini-3_1-flash-lite" not in model_configs

    for model_config in model_configs.values():
        assert "temperature" not in model_config
        assert "top_p" not in model_config
        assert "top_k" not in model_config
        assert "thinking_budget" not in model_config
