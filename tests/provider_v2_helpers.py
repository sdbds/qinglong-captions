import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_provider_args(**kwargs):
    defaults = {
        "step_api_key": "",
        "ark_api_key": "",
        "qwenVL_api_key": "",
        "glm_api_key": "",
        "kimi_code_api_key": "",
        "kimi_api_key": "",
        "mimo_api_key": "",
        "mimo_base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
        "mimo_model_path": "mimo-v2.5",
        "mistral_api_key": "",
        "pixtral_api_key": "",
        "gemini_api_key": "",
        "ocr_model": "",
        "document_image": False,
        "vlm_image_model": "",
        "alm_model": "",
        "audio_task": "",
        "gemma4_model_id": "",
        "pair_dir": "",
        "codex_subscription": False,
        "codex_backend": "sdk_app_server",
        "codex_auth_mode": "chatgpt",
        "codex_api_key": "",
        "codex_command": "codex",
        "codex_model_name": "gpt-5.4",
        "codex_service_tier": "",
        "codex_fast": False,
        "codex_reasoning_effort": "none",
        "codex_home": "",
        "codex_timeout": 60.0,
        "codex_sandbox": "read-only",
        "codex_isolated_cwd": "",
        "codex_output_schema": "",
        "codex_runtime_path": "",
        "codex_max_concurrency": 1,
        "cloud_max_concurrency": 1,
        "codex_auto_install_sdk": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)
