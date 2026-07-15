import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def _load_caption_step():
    module_path = ROOT / "gui" / "wizard" / "step4_caption.py"
    spec = importlib.util.spec_from_file_location("test_ovis_step4_caption", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    gui_path = str(ROOT / "gui")
    original_sys_path = list(sys.path)
    if gui_path not in sys.path:
        sys.path.append(gui_path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module.CaptionStep


def test_ovis_ocr2_declaration_exposes_ordered_local_ocr_route():
    from module.providers.catalog import get_provider_declaration, route_choices

    declaration = get_provider_declaration("ovis_ocr2")
    assert declaration is not None
    assert declaration.module_path == "module.providers.ocr.ovis_ocr2"
    assert declaration.priority == 122
    assert declaration.routes[0].route_name == "ocr_model"
    assert declaration.routes[0].order == 15
    assert declaration.routes[0].requires_remote_config is False
    assert "ovis_ocr2" in route_choices("ocr_model")


def test_ovis_ocr2_model_and_legacy_configs_share_runtime_defaults():
    model_config = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))["ovis_ocr2"]
    legacy_config = tomllib.loads((ROOT / "config" / "config.toml").read_text(encoding="utf-8"))["ovis_ocr2"]

    expected = {
        "model_id": "ATH-MaaS/OvisOCR2",
        "runtime_backend": "direct",
        "prompt": "",
        "max_new_tokens": 16384,
        "temperature": 0.0,
        "top_p": 1.0,
        "min_pixels": 448 * 448,
        "max_pixels": 2880 * 2880,
        "visual_region_mode": "crop",
    }
    assert {key: model_config[key] for key in expected} == expected
    assert {key: legacy_config[key] for key in expected} == expected
    assert model_config["model_list"]["OvisOCR2 0.8B"]["model_id"] == "ATH-MaaS/OvisOCR2"


def test_ovis_ocr2_extra_contains_only_required_direct_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["optional-dependencies"]["ovis-ocr2"]

    assert dependencies == [
        "qinglong-captions[torch-base]",
        "transformers[serving]>=5.7.0",
        "qinglong-captions[qwen35-fast-path]",
        "PyMuPDF",
    ]
    assert not any("vllm" in dependency.lower() for dependency in dependencies)
    assert not any("qwen-vl-utils" in dependency.lower() for dependency in dependencies)
    assert "img2pdf" not in dependencies


def test_ovis_ocr2_extra_conflicts_with_incompatible_transformers_profiles():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    conflict_groups = pyproject["tool"]["uv"]["conflicts"]
    ovis_group = next(group for group in conflict_groups if {"extra": "ovis-ocr2"} in group)
    extras = {entry["extra"] for entry in ovis_group if "extra" in entry}

    assert {
        "ovis-ocr2",
        "paddleocr-native",
        "reward-model",
        "deepseek-ocr",
        "unlimited-ocr",
        "dots-ocr",
        "penguin-vl-local",
        "hunyuan-ocr",
        "glm-ocr",
        "qwen-vl-local",
        "music-flamingo-local",
        "eureka-audio-local",
        "cohere-transcribe-local",
    } <= extras


def test_gui_exposes_ovis_route_and_installs_ovis_extra_without_fake_rank():
    CaptionStep = _load_caption_step()
    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="ovis_ocr2")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="")

    assert "ovis_ocr2" in CaptionStep.OCR_MODELS
    assert "ovis_ocr2" not in CaptionStep._OCR_RANK_SCORES
    assert step._build_local_extra_args() == ["--extra", "ovis-ocr2"]


def test_captioner_powershell_declares_ovis_route_and_extra():
    script = (ROOT / "4.captioner.ps1").read_text(encoding="utf-8-sig")

    assert '"ovis_ocr2"' in script.split("# VLM model configuration", 1)[0]
    assert re.search(
        r'elseif \(\$ocr_model -eq "ovis_ocr2"\) \{\s*Add-UvExtra "ovis-ocr2"\s*\}',
        script,
    )
