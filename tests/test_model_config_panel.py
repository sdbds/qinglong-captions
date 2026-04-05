import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from gui.utils.toml_helpers import ModelListEntry


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "gui") not in sys.path:
    sys.path.insert(1, str(ROOT / "gui"))


def _load_model_config_panel_module(module_name: str):
    module_path = ROOT / "gui" / "components" / "model_config_panel.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeContext:
    def classes(self, *_args, **_kwargs):
        return self

    def style(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeLabel:
    def classes(self, *_args, **_kwargs):
        return self

    def style(self, *_args, **_kwargs):
        return self


class _FakeSelect:
    def __init__(self, **kwargs):
        self.props_calls = []
        self.style_calls = []
        self.on_value_change_handler = None
        self.options = kwargs.get("options")
        self.value = kwargs.get("value")
        self.new_value_mode = kwargs.get("new_value_mode")

    def classes(self, *_args, **_kwargs):
        return self

    def props(self, value):
        self.props_calls.append(value)
        return self

    def style(self, value):
        self.style_calls.append(value)
        return self

    def on_value_change(self, handler):
        self.on_value_change_handler = handler
        return self

    def set_value(self, value):
        self.value = value
        return self

    def set_options(self, options, *, value=None):
        self.options = options
        self.value = value
        return self


class _FakeUI:
    def __init__(self, fake_selects):
        self._fake_selects = list(fake_selects)
        self.created_selects = []

    def row(self):
        return _FakeContext()

    def column(self):
        return _FakeContext()

    def label(self, *_args, **_kwargs):
        return _FakeLabel()

    def select(self, *_args, **kwargs):
        fake_select = self._fake_selects.pop(0)
        fake_select.options = kwargs.get("options")
        fake_select.value = kwargs.get("value")
        fake_select.new_value_mode = kwargs.get("new_value_mode")
        self.created_selects.append(fake_select)
        return fake_select


def test_model_config_panel_renders_model_id_as_editable_searchable_combobox(monkeypatch):
    panel_module = _load_model_config_panel_module("test_model_config_panel_combobox")
    fake_model_list_select = _FakeSelect()
    fake_model_id_select = _FakeSelect()

    monkeypatch.setattr(panel_module, "ui", _FakeUI([fake_model_list_select, fake_model_id_select]))
    monkeypatch.setattr(
        panel_module.ModelConfigPanel,
        "_build_model_id_options",
        staticmethod(lambda route_name, current_model_id, probe=None: {current_model_id: current_model_id}),
    )

    panel = panel_module.ModelConfigPanel(parent=SimpleNamespace(clear=lambda: None))
    panel._current_route = "qwen_vl_local"
    parent_dict = {"model_id": "Qwen/Qwen3.5-9B"}

    panel._render_field("model_id", "Qwen/Qwen3.5-9B", parent_dict)

    combined_props = " ".join(fake_model_id_select.props_calls)
    assert "use-input" in combined_props
    assert "fill-input" in combined_props
    assert "clearable" in combined_props
    assert 'dropdown-icon="search"' in combined_props
    assert "hide-selected" not in combined_props

    assert fake_model_id_select.on_value_change_handler is not None
    fake_model_id_select.on_value_change_handler(SimpleNamespace(value="Qwen/Qwen2.5-VL-31B-Instruct-AWQ"))
    assert parent_dict["model_id"] == "Qwen/Qwen2.5-VL-31B-Instruct-AWQ"


def test_model_config_panel_build_model_id_options_does_not_probe_gpu_by_default(monkeypatch):
    panel_module = _load_model_config_panel_module("test_model_config_panel_no_gpu_probe")
    calls = []

    def _fake_load_model_id_options(route_name, *, current_model_id=""):
        calls.append((route_name, current_model_id))
        return {current_model_id: current_model_id}

    monkeypatch.setattr(panel_module, "load_model_id_options", _fake_load_model_id_options)

    options = panel_module.ModelConfigPanel._build_model_id_options(
        route_name="qwen_vl_local",
        current_model_id="Qwen/Qwen3.5-9B",
    )

    assert options == {"Qwen/Qwen3.5-9B": "Qwen/Qwen3.5-9B"}
    assert calls == [("qwen_vl_local", "Qwen/Qwen3.5-9B")]


def test_model_config_panel_model_list_selection_fills_model_id(monkeypatch):
    panel_module = _load_model_config_panel_module("test_model_config_panel_model_list_selection")
    fake_model_list_select = _FakeSelect()
    fake_model_id_select = _FakeSelect()
    fake_ui = _FakeUI([fake_model_list_select, fake_model_id_select])

    monkeypatch.setattr(panel_module, "ui", fake_ui)
    monkeypatch.setattr(
        panel_module.ModelConfigPanel,
        "_load_model_list_entries",
        staticmethod(
            lambda route_name: (
                ModelListEntry(name="Gemma 4 E2B it", model_id="google/gemma-4-E2B-it"),
                ModelListEntry(name="Gemma 4 E4B it FP8", model_id="protoLabsAI/gemma-4-E4B-it-FP8"),
            )
        ),
    )
    monkeypatch.setattr(
        panel_module.ModelConfigPanel,
        "_build_model_id_options",
        staticmethod(
            lambda route_name, current_model_id, probe=None: {
                "google/gemma-4-E2B-it": "google/gemma-4-E2B-it",
                "protoLabsAI/gemma-4-E4B-it-FP8": "protoLabsAI/gemma-4-E4B-it-FP8",
            }
        ),
    )

    panel = panel_module.ModelConfigPanel(parent=SimpleNamespace(clear=lambda: None))
    panel._current_route = "gemma4_local"
    parent_dict = {"model_id": ""}

    panel._render_field("model_id", "", parent_dict)

    assert fake_model_list_select.value is None
    assert fake_model_id_select.value is None

    fake_model_list_select.on_value_change_handler(SimpleNamespace(value="Gemma 4 E4B it FP8"))

    assert parent_dict["model_id"] == "protoLabsAI/gemma-4-E4B-it-FP8"
    assert fake_model_id_select.value == "protoLabsAI/gemma-4-E4B-it-FP8"


def test_model_config_panel_custom_model_id_clears_model_list_selection(monkeypatch):
    panel_module = _load_model_config_panel_module("test_model_config_panel_model_list_custom_model_id")
    fake_model_list_select = _FakeSelect()
    fake_model_id_select = _FakeSelect()
    fake_ui = _FakeUI([fake_model_list_select, fake_model_id_select])

    monkeypatch.setattr(panel_module, "ui", fake_ui)
    monkeypatch.setattr(
        panel_module.ModelConfigPanel,
        "_load_model_list_entries",
        staticmethod(
            lambda route_name: (
                ModelListEntry(name="Gemma 4 E2B it", model_id="google/gemma-4-E2B-it"),
            )
        ),
    )
    monkeypatch.setattr(
        panel_module.ModelConfigPanel,
        "_build_model_id_options",
        staticmethod(lambda route_name, current_model_id, probe=None: {"google/gemma-4-E2B-it": "google/gemma-4-E2B-it"}),
    )

    panel = panel_module.ModelConfigPanel(parent=SimpleNamespace(clear=lambda: None))
    panel._current_route = "gemma4_local"
    parent_dict = {"model_id": "google/gemma-4-E2B-it"}

    panel._render_field("model_id", "google/gemma-4-E2B-it", parent_dict)

    fake_model_id_select.on_value_change_handler(SimpleNamespace(value="unsloth/gemma-4-E2B-it-bnb-4bit"))

    assert parent_dict["model_id"] == "unsloth/gemma-4-E2B-it-bnb-4bit"
    assert fake_model_list_select.value is None
