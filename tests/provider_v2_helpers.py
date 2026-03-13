import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def make_provider_args(**kwargs):
    defaults = {
        "step_api_key": "",
        "ark_api_key": "",
        "qwenVL_api_key": "",
        "glm_api_key": "",
        "kimi_code_api_key": "",
        "kimi_api_key": "",
        "mistral_api_key": "",
        "pixtral_api_key": "",
        "gemini_api_key": "",
        "ocr_model": "",
        "document_image": False,
        "vlm_image_model": "",
        "pair_dir": "",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)
