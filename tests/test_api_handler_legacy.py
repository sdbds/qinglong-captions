import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_legacy_api_handler_supports_dots_ocr_provider():
    from providers.base import CaptionResult
    from module.api_handler import api_process_batch

    args = SimpleNamespace(
        ocr_model="dots_ocr",
        document_image=True,
        max_retries=1,
        wait_time=0.01,
        dir_name=False,
        local_runtime_backend="",
        openai_model_name="",
        pair_dir="",
        vlm_image_model="",
        alm_model="",
        step_api_key="",
        ark_api_key="",
        qwenVL_api_key="",
        glm_api_key="",
        kimi_code_api_key="",
        kimi_api_key="",
        mistral_api_key="",
        pixtral_api_key="",
        gemini_api_key="",
    )

    with patch("providers.ocr.dots.DotsOCRProvider.execute", return_value=CaptionResult(raw="legacy dots")):
        result = api_process_batch(
            uri="/fake.png",
            mime="image/png",
            config={"prompts": {}, "dots_ocr": {}},
            args=args,
            sha256hash="abc",
        )

    assert result == "legacy dots"
