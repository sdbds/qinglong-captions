import ast
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext


ROOT = Path(__file__).resolve().parent.parent


def test_qwenvl_attempt_builds_video_message_without_shadowing_path(monkeypatch, tmp_path):
    from module.providers.cloud_vlm import qwenvl

    captured = {}

    def fake_attempt_qwenvl(**kwargs):
        captured.update(kwargs)
        return "caption"

    monkeypatch.setattr(qwenvl, "attempt_qwenvl", fake_attempt_qwenvl)
    args = SimpleNamespace(qwenVL_model_path="qwen-vl-max", qwenVL_api_key="test-key", pair_dir="")
    provider = qwenvl.QwenVLProvider(ProviderContext(console=Console(), args=args))
    media_path = tmp_path / "clip.mp4"

    result = provider.attempt(
        MediaContext(
            uri=str(media_path),
            mime="video/mp4",
            sha256hash="hash",
            modality=MediaModality.VIDEO,
        ),
        PromptContext(system="system", user="describe"),
    )

    assert result.raw == "caption"
    assert captured["messages"][1]["content"][0]["video"].startswith("file://")


def test_standalone_ocr_functions_do_not_reference_implicit_self():
    targets = {
        "module/providers/ocr/deepseek.py": {"attempt_deepseek_ocr"},
        "module/providers/ocr/firered.py": {"attempt_firered_ocr"},
        "module/providers/ocr/glm.py": {"attempt_glm_ocr"},
        "module/providers/ocr/hunyuan.py": {"attempt_hunyuan_ocr"},
        "module/providers/ocr/lighton.py": {"attempt_lighton_ocr"},
        "module/providers/ocr/logics.py": {"attempt_logics_ocr"},
        "module/providers/ocr/nanonets.py": {"attempt_nanonets_ocr"},
        "module/providers/ocr/olmocr.py": {"attempt_olmocr", "_generate_for_image"},
    }

    failures = []
    for relative_path, function_names in targets.items():
        path = ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_names
        }
        assert functions.keys() == function_names
        for name, function in functions.items():
            if any(isinstance(node, ast.Name) and node.id == "self" for node in ast.walk(function)):
                failures.append(f"{relative_path}:{name}")

    assert failures == []
