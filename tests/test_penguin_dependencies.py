import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import toml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def _load_caption_step(module_name: str):
    module_path = ROOT / "gui" / "wizard" / "step4_caption.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    step4_caption = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    gui_path = str(ROOT / "gui")
    original_sys_path = list(sys.path)
    if gui_path not in sys.path:
        try:
            insert_at = original_sys_path.index(str(ROOT)) + 1
        except ValueError:
            insert_at = len(sys.path)
        sys.path.insert(insert_at, gui_path)
    try:
        spec.loader.exec_module(step4_caption)
    finally:
        sys.path[:] = original_sys_path

    return step4_caption.CaptionStep


def test_pyproject_declares_penguin_vl_local_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "penguin-vl-local" in optional_deps
    penguin_deps = optional_deps["penguin-vl-local"]
    assert "transformers[serving]==4.51.3" in penguin_deps
    assert any(dep.startswith("decord") for dep in penguin_deps)
    assert "ffmpeg-python==0.2.0" in penguin_deps
    assert any(dep.startswith("einops") for dep in penguin_deps)


def test_pyproject_declares_lfm_vl_local_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "lfm-vl-local" in optional_deps
    lfm_deps = optional_deps["lfm-vl-local"]
    assert "qinglong-captions[onnx-base]" in lfm_deps
    assert any(dep.startswith("transformers") for dep in lfm_deps)
    assert not any("flash-attn" in dep for dep in lfm_deps)


def test_pyproject_declares_lighton_ocr_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "lighton-ocr" in optional_deps
    lighton_deps = optional_deps["lighton-ocr"]
    assert "qinglong-captions[torch-base]" in lighton_deps
    assert any(dep.startswith("transformers[serving]>=5.0.0") for dep in lighton_deps)


def test_pyproject_declares_logics_ocr_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "logics-ocr" in optional_deps
    logics_deps = optional_deps["logics-ocr"]
    assert "qinglong-captions[torch-base]" in logics_deps
    assert any(dep.startswith("transformers[serving]>=4.57.0") for dep in logics_deps)
    assert "PyMuPDF" in logics_deps
    assert "img2pdf" in logics_deps


def test_pyproject_declares_dots_ocr_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]
    uv_sources = pyproject["tool"]["uv"]["sources"]

    assert "dots-ocr" in optional_deps
    dots_deps = optional_deps["dots-ocr"]
    assert "qinglong-captions[torch-base]" in dots_deps
    assert any(dep.startswith("transformers[serving]==4.56.1") for dep in dots_deps)
    assert any("qwen-vl-utils" in dep for dep in dots_deps)
    assert any(dep.startswith("triton-windows") for dep in dots_deps)
    assert any("flash-attn" in dep for dep in dots_deps)
    assert "dots_ocr" in dots_deps
    assert uv_sources["dots_ocr"]["path"] == "third_party/dots.ocr"
    assert uv_sources["dots_ocr"]["editable"] is True


def test_pyproject_declares_qianfan_ocr_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "qianfan-ocr" in optional_deps
    qianfan_deps = optional_deps["qianfan-ocr"]
    assert "qinglong-captions[torch-base]" in qianfan_deps
    assert any(dep.startswith("transformers[serving]>=4.57.0") for dep in qianfan_deps)
    assert any(dep.startswith("timm") for dep in qianfan_deps)
    assert "PyMuPDF" in qianfan_deps
    assert "img2pdf" in qianfan_deps


def test_pyproject_declares_music_flamingo_local_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "music-flamingo-local" in optional_deps
    music_flamingo_deps = optional_deps["music-flamingo-local"]
    assert "qinglong-captions[torch-base]" in music_flamingo_deps
    assert any(dep.startswith("kernels") for dep in music_flamingo_deps)
    assert any(dep.startswith("torchaudio") for dep in music_flamingo_deps)
    assert any(dep.startswith("triton-windows") for dep in music_flamingo_deps)
    assert any(
        dep.startswith(
            "transformers[serving] @ git+https://github.com/lashahub/transformers@modular-mf"
        )
        for dep in music_flamingo_deps
    )


def test_pyproject_declares_eureka_audio_local_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "eureka-audio-local" in optional_deps
    eureka_audio_deps = optional_deps["eureka-audio-local"]
    assert "qinglong-captions[torch-base]" in eureka_audio_deps
    assert any(dep.startswith("eureka-audio @ git+") for dep in eureka_audio_deps)
    assert any(dep.startswith("transformers[serving]>=4.57.0,<5") for dep in eureka_audio_deps)


def test_pyproject_declares_acestep_transcriber_local_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "acestep-transcriber-local" in optional_deps
    acestep_transcriber_deps = optional_deps["acestep-transcriber-local"]
    assert "qinglong-captions[torch-base]" in acestep_transcriber_deps
    assert any(dep.startswith("torchaudio") for dep in acestep_transcriber_deps)
    assert any(dep.startswith("transformers[serving]>=4.57.0") for dep in acestep_transcriber_deps)
    assert any(dep.startswith("flash-attn") for dep in acestep_transcriber_deps)


def test_pyproject_declares_vocal_midi_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "vocal-midi" in optional_deps
    vocal_midi_deps = optional_deps["vocal-midi"]
    assert "qinglong-captions[onnx-base]" in vocal_midi_deps
    assert any(dep.startswith("mido") for dep in vocal_midi_deps)
    assert any(dep.startswith("numpy") for dep in vocal_midi_deps)


def test_caption_step_includes_penguin_extra():
    CaptionStep = _load_caption_step("test_step4_caption")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="penguin_vl_local")

    assert step._build_local_extra_args() == ["--extra", "penguin-vl-local"]


def test_caption_step_includes_lfm_extra():
    CaptionStep = _load_caption_step("test_step4_caption_lfm")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="lfm_vl_local")

    assert step._build_local_extra_args() == ["--extra", "lfm-vl-local"]


def test_caption_step_includes_lighton_extra():
    CaptionStep = _load_caption_step("test_step4_caption_lighton")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="lighton_ocr")
    step.vlm_image_model = SimpleNamespace(value="")

    assert step._build_local_extra_args() == ["--extra", "lighton-ocr"]


def test_caption_step_includes_logics_ocr_extra():
    CaptionStep = _load_caption_step("test_step4_caption_logics")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="logics_ocr")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="")

    assert step._build_local_extra_args() == ["--extra", "logics-ocr"]


def test_caption_step_includes_dots_ocr_extra():
    CaptionStep = _load_caption_step("test_step4_caption_dots")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="dots_ocr")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="")

    assert step._build_local_extra_args() == ["--extra", "dots-ocr"]


def test_caption_step_includes_qianfan_ocr_extra():
    CaptionStep = _load_caption_step("test_step4_caption_qianfan")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="qianfan_ocr")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="")

    assert step._build_local_extra_args() == ["--extra", "qianfan-ocr"]


def test_caption_step_includes_music_flamingo_extra():
    CaptionStep = _load_caption_step("test_step4_caption_music_flamingo")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="music_flamingo_local")

    assert step._build_local_extra_args() == ["--extra", "music-flamingo-local"]


def test_caption_step_includes_eureka_audio_extra():
    CaptionStep = _load_caption_step("test_step4_caption_eureka_audio")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="eureka_audio_local")

    assert step._build_local_extra_args() == ["--extra", "eureka-audio-local"]


def test_caption_step_includes_acestep_transcriber_extra():
    CaptionStep = _load_caption_step("test_step4_caption_acestep_transcriber")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="acestep_transcriber_local")

    assert step._build_local_extra_args() == ["--extra", "acestep-transcriber-local"]


def test_caption_step_treats_music_flamingo_as_local_route():
    CaptionStep = _load_caption_step("test_step4_caption_local_route")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="music_flamingo_local")

    assert step._has_local_route_config() is True


def test_caption_step_treats_eureka_audio_as_local_route():
    CaptionStep = _load_caption_step("test_step4_caption_local_route_eureka")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="eureka_audio_local")

    assert step._has_local_route_config() is True


def test_caption_step_treats_acestep_transcriber_as_local_route():
    CaptionStep = _load_caption_step("test_step4_caption_local_route_acestep")

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="acestep_transcriber_local")

    assert step._has_local_route_config() is True


def test_caption_step_builds_alm_args_without_default_segment_override():
    CaptionStep = _load_caption_step("test_step4_caption_build_args")

    step = CaptionStep()
    step.api_keys = {}
    step.mode = SimpleNamespace(value="all")
    step.pair_dir = SimpleNamespace(value="")
    step.scene_detector = SimpleNamespace(value="AdaptiveDetector")
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="music_flamingo_local")

    args = step._build_caption_args("demo-dataset")

    assert "--alm_model=music_flamingo_local" in args
    assert not any(arg.startswith("--segment_time=") for arg in args)


def test_caption_step_builds_explicit_segment_time_override():
    CaptionStep = _load_caption_step("test_step4_caption_segment_override")

    step = CaptionStep()
    step.api_keys = {}
    step.mode = SimpleNamespace(value="all")
    step.pair_dir = SimpleNamespace(value="")
    step.scene_detector = SimpleNamespace(value="AdaptiveDetector")
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="music_flamingo_local")
    step.config["segment_time"] = 90
    step.config["segment_time_explicit"] = True

    args = step._build_caption_args("demo-dataset")

    assert "--segment_time=90" in args


def test_caption_step_formats_full_args_for_log_with_redaction():
    CaptionStep = _load_caption_step("test_step4_caption_log_format")

    formatted = CaptionStep._format_args_for_log(
        [
            "demo-dataset",
            "--openai_api_key=super-secret",
            "--segment_time=90",
            "--wait_time=1",
        ]
    )

    assert "demo-dataset" in formatted
    assert "--segment_time=90" in formatted
    assert "--wait_time=1" in formatted
    assert "--openai_api_key=***" in formatted
    assert "super-secret" not in formatted


def test_run_ps1_mentions_lfm_vl_local_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"lfm_vl_local"' in content
    assert 'Add-UvExtra "lfm-vl-local"' in content


def test_run_ps1_mentions_lighton_ocr_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"lighton_ocr"' in content
    assert 'Add-UvExtra "lighton-ocr"' in content


def test_run_ps1_mentions_logics_ocr_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"logics_ocr"' in content
    assert 'Add-UvExtra "logics-ocr"' in content


def test_run_ps1_mentions_dots_ocr_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"dots_ocr"' in content
    assert 'Add-UvExtra "dots-ocr"' in content


def test_run_ps1_mentions_qianfan_ocr_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"qianfan_ocr"' in content
    assert 'Add-UvExtra "qianfan-ocr"' in content


def test_run_ps1_uses_generic_extra_fallback_for_music_flamingo():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"music_flamingo_local"' in content
    assert 'Add-UvExtra "music-flamingo-local"' in content
    assert "--alm_model=$alm_model" in content
    assert "if ($null -ne $segment_time)" in content
    assert "Get-PyprojectProfileRequirements" not in content
    assert "直接使用 uv pip install -r pyproject.toml 安装当前依赖 profile" in content
    assert 'if ($Extra -eq "music-flamingo-local")' not in content
    assert "transformers[serving] @ git+https://github.com/lashahub/transformers@modular-mf" not in content


def test_run_ps1_mentions_eureka_audio_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"eureka_audio_local"' in content
    assert 'Add-UvExtra "eureka-audio-local"' in content


def test_run_ps1_mentions_acestep_transcriber_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"acestep_transcriber_local"' in content
    assert 'Add-UvExtra "acestep-transcriber-local"' in content


def test_run_ps1_does_not_lock_or_export_at_runtime():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert "Ensure-UvLockFile" not in content
    assert "uv lock --python 3.11 --index-strategy $IndexStrategy" not in content
    assert "uv export --frozen" not in content


def test_config_declares_dots_ocr_defaults():
    model_toml = (ROOT / "config" / "model.toml").read_text(encoding="utf-8")
    runtime_toml = (ROOT / "config" / "config.toml").read_text(encoding="utf-8")
    prompts_toml = (ROOT / "config" / "prompts.toml").read_text(encoding="utf-8")

    assert "[dots_ocr]" in model_toml
    assert 'prompt_mode = "prompt_ocr"' in model_toml
    assert 'svg_model_id = "davanstrien/dots.ocr-1.5-svg"' in model_toml
    assert "max_new_tokens = 24000" in model_toml
    assert "runtime_temperature = 0.1" in model_toml
    assert "runtime_top_p = 1.0" in model_toml
    assert "runtime_max_tokens = 16384" in model_toml
    assert "fitz_preprocess = false" in model_toml
    assert "dpi = 200" in model_toml
    assert "[dots_ocr]" in runtime_toml
    assert 'prompt_mode = "prompt_ocr"' in runtime_toml
    assert "max_new_tokens = 24000" in runtime_toml
    assert "runtime_temperature = 0.1" in runtime_toml
    assert "runtime_top_p = 1.0" in runtime_toml
    assert "runtime_max_tokens = 16384" in runtime_toml
    assert "fitz_preprocess = false" in runtime_toml
    assert "dpi = 200" in runtime_toml
    assert "dots_ocr_prompt" in prompts_toml
    assert "[prompts.task.dots_ocr]" in prompts_toml
    assert 'prompt_ocr = """Extract the text content from this image."""' in prompts_toml
    assert (
        'prompt_image_to_svg = \'Please generate the SVG code based on the '
        'image.viewBox="0 0 {width} {height}"\''
    ) in prompts_toml


def test_config_declares_qianfan_ocr_defaults():
    model_toml = (ROOT / "config" / "model.toml").read_text(encoding="utf-8")
    runtime_toml = (ROOT / "config" / "config.toml").read_text(encoding="utf-8")

    assert "[qianfan_ocr]" in model_toml
    assert 'model_id = "baidu/Qianfan-OCR"' in model_toml
    assert 'prompt = ""' in model_toml
    assert 'prompt_strategy = "append"' in model_toml
    assert "think_enabled = true" in model_toml
    assert "max_new_tokens = 16384" in model_toml
    assert "input_size = 448" in model_toml
    assert "max_num = 12" in model_toml

    assert "[qianfan_ocr]" in runtime_toml
    assert 'model_id = "baidu/Qianfan-OCR"' in runtime_toml
    assert 'prompt = ""' in runtime_toml
    assert 'prompt_strategy = "append"' in runtime_toml
    assert "think_enabled = true" in runtime_toml
    assert "max_new_tokens = 16384" in runtime_toml
    assert "input_size = 448" in runtime_toml
    assert "max_num = 12" in runtime_toml


def test_config_declares_logics_ocr_defaults():
    model_toml = (ROOT / "config" / "model.toml").read_text(encoding="utf-8")
    runtime_toml = (ROOT / "config" / "config.toml").read_text(encoding="utf-8")
    prompts_toml = (ROOT / "config" / "prompts.toml").read_text(encoding="utf-8")

    assert "[logics_ocr]" in model_toml
    assert 'model_id = "Logics-MLLM/Logics-Parsing-v2"' in model_toml
    assert "max_new_tokens = 16384" in model_toml
    assert "min_pixels = 3136" in model_toml
    assert "max_pixels = 7372800" in model_toml

    assert "[logics_ocr]" in runtime_toml
    assert 'model_id = "Logics-MLLM/Logics-Parsing-v2"' in runtime_toml
    assert "max_new_tokens = 16384" in runtime_toml
    assert "min_pixels = 3136" in runtime_toml
    assert "max_pixels = 7372800" in runtime_toml

    assert 'logics_ocr_prompt = """QwenVL HTML"""' in prompts_toml


def test_runtime_prompt_config_files_parse_with_toml_library():
    for rel in ("config/prompts.toml", "config/config.toml"):
        parsed = toml.load(ROOT / rel)
        assert isinstance(parsed, dict)


def test_has_flash_attn_returns_false_when_module_import_fails(monkeypatch):
    import importlib

    from utils import transformer_loader

    transformer_loader._has_flash_attn.cache_clear()
    monkeypatch.setattr(transformer_loader.importlib.util, "find_spec", lambda name: object() if name == "flash_attn" else None)

    def fake_import_module(name):
        if name == "flash_attn":
            raise ImportError("DLL load failed")
        return importlib.import_module(name)

    monkeypatch.setattr(transformer_loader.importlib, "import_module", fake_import_module)

    assert transformer_loader._has_flash_attn() is False


def test_penguin_provider_uses_resolved_attention_backend(monkeypatch):
    import io
    import types

    from rich.console import Console

    from providers.base import ProviderContext
    from providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

    captured = {}

    class FakeLoader:
        def get_or_load_processor(self, *args, **kwargs):
            captured["processor_cls"] = args[1]
            return "processor"

        def get_or_load_model(self, *args, **kwargs):
            captured["attn_impl"] = kwargs["attn_impl"]
            return "model"

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.AutoModelForCausalLM = object

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "utils.transformer_loader.resolve_device_dtype",
        lambda: ("cuda", "bfloat16", "flash_attention_2"),
    )
    monkeypatch.setattr("utils.transformer_loader.transformerLoader", lambda **kwargs: FakeLoader())

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"penguin_vl_local": {"model_id": "tencent/Penguin-VL-8B"}},
        args=SimpleNamespace(vlm_image_model="penguin_vl_local"),
    )

    cached = PenguinVLLocalProvider(ctx)._load_model()

    assert captured["processor_cls"] is fake_transformers.AutoProcessor
    assert captured["attn_impl"] == "flash_attention_2"
    assert cached["processor"] == "processor"
    assert cached["model"] == "model"


def test_patch_penguin_vision_attention_uses_sdpa_fallback():
    import types

    import torch

    from providers.local_vlm.penguin_vl_local import _patch_penguin_vision_attention

    fake_module = types.ModuleType("test_penguin_encoder_module")
    fake_module.apply_multimodal_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
    sys.modules[fake_module.__name__] = fake_module

    class FakePenguinAttention(torch.nn.Module):
        __module__ = "test_penguin_encoder_module"

        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.num_key_value_groups = 2
            self.layer_idx = 0
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 2, bias=False)
            self.v_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.attention_dropout = 0.0
            self.is_causal = False
            self.config = SimpleNamespace()

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakePenguinAttention()

    class FakeEncoder:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeVisionEncoder:
        def __init__(self):
            self.encoder = FakeEncoder()

    class FakeInnerModel:
        def __init__(self):
            self._vision_encoder = FakeVisionEncoder()

        def get_vision_encoder(self):
            return self._vision_encoder

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="eager")
            self._model = FakeInnerModel()

        def get_model(self):
            return self._model

    model = FakeModel()
    assert _patch_penguin_vision_attention(model) is True

    attention = model.get_model().get_vision_encoder().encoder.layers[0].self_attn
    hidden_states = torch.randn(1, 4, 4)
    output, weights = attention(
        hidden_states=hidden_states,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
    )

    assert output.shape == hidden_states.shape
    assert weights is None


def test_penguin_sdpa_fallback_does_not_import_qwen3_modeling(monkeypatch):
    import builtins
    import types

    import torch

    from providers.local_vlm.penguin_vl_local import _patch_penguin_vision_attention

    fake_module = types.ModuleType("test_penguin_encoder_module_no_qwen3")
    fake_module.apply_multimodal_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
    sys.modules[fake_module.__name__] = fake_module

    class FakePenguinAttention(torch.nn.Module):
        __module__ = "test_penguin_encoder_module_no_qwen3"

        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.num_key_value_groups = 2
            self.layer_idx = 0
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 2, bias=False)
            self.v_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.attention_dropout = 0.0
            self.is_causal = False
            self.config = SimpleNamespace()

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakePenguinAttention()

    class FakeEncoder:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeVisionEncoder:
        def __init__(self):
            self.encoder = FakeEncoder()

    class FakeInnerModel:
        def __init__(self):
            self._vision_encoder = FakeVisionEncoder()

        def get_vision_encoder(self):
            return self._vision_encoder

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="eager")
            self._model = FakeInnerModel()

        def get_model(self):
            return self._model

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.models.qwen3.modeling_qwen3":
            raise AssertionError("unexpected qwen3 modeling import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    model = FakeModel()
    assert _patch_penguin_vision_attention(model) is True

    attention = model.get_model().get_vision_encoder().encoder.layers[0].self_attn
    hidden_states = torch.randn(1, 4, 4)
    output, weights = attention(
        hidden_states=hidden_states,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
    )

    assert output.shape == hidden_states.shape
    assert weights is None


def test_penguin_attempt_preserves_non_tensor_outputs_and_casts_float_inputs(monkeypatch, tmp_path):
    import io

    import torch
    from rich.console import Console

    from providers.base import PromptContext, ProviderContext
    from providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")

    captured = {}

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.bfloat16

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[10, 11, 12]])

    class FakeProcessor:
        def __call__(self, **kwargs):
            captured["processor_kwargs"] = kwargs
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.randn(2, 8, dtype=torch.float32),
                "grid_sizes": torch.tensor([[1, 1, 1]], dtype=torch.int64),
                "modals": ["image"],
            }

        def decode(self, *_args, **_kwargs):
            return "ok"

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"penguin_vl_local": {"model_id": "tencent/Penguin-VL-8B"}},
        args=SimpleNamespace(vlm_image_model="penguin_vl_local", pair_dir=""),
    )
    provider = PenguinVLLocalProvider(ctx)
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {"model": FakeModel(), "processor": FakeProcessor(), "device": "cpu"},
    )

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, PromptContext(system="system", user="user"))

    assert captured["generate_kwargs"]["modals"] == ["image"]
    assert captured["generate_kwargs"]["input_ids"].device.type == "cpu"
    assert captured["generate_kwargs"]["input_ids"].dtype == torch.int64
    assert captured["generate_kwargs"]["pixel_values"].dtype == torch.bfloat16
    assert captured["generate_kwargs"]["grid_sizes"].dtype == torch.int64
    assert result.raw == "ok"


def test_penguin_attempt_keeps_generated_tokens_when_model_returns_new_tokens_only(monkeypatch, tmp_path):
    import io

    import torch
    from rich.console import Console

    from providers.base import PromptContext, ProviderContext
    from providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")

    captured = {}

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.float32

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[101, 102, 103]])

    class FakeProcessor:
        def __call__(self, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                "modals": ["image"],
            }

        def decode(self, token_ids, *_args, **_kwargs):
            captured["decoded_ids"] = token_ids.tolist()
            return "decoded-new-tokens"

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"penguin_vl_local": {"model_id": "tencent/Penguin-VL-8B"}},
        args=SimpleNamespace(vlm_image_model="penguin_vl_local", pair_dir=""),
    )
    provider = PenguinVLLocalProvider(ctx)
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {"model": FakeModel(), "processor": FakeProcessor(), "device": "cpu"},
    )

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, PromptContext(system="system", user="user"))

    assert captured["decoded_ids"] == [101, 102, 103]
    assert result.raw == "decoded-new-tokens"
