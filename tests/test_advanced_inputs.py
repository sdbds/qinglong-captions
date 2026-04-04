import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent


def _load_advanced_inputs(module_name: str):
    module_path = ROOT / "gui" / "components" / "advanced_inputs.py"
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


class _FakeSelect:
    def __init__(self):
        self.props_calls = []
        self.on_calls = []
        self.on_value_change_handler = None

    def classes(self, *_args, **_kwargs):
        return self

    def props(self, value):
        self.props_calls.append(value)
        return self

    def on(self, *args, **kwargs):
        self.on_calls.append((args, kwargs))
        return self

    def on_value_change(self, handler):
        self.on_value_change_handler = handler
        return self


class _FakeUI:
    def __init__(self, select):
        self._select = select

    def column(self):
        return _FakeContext()

    def row(self):
        return _FakeContext()

    def select(self, *_args, **_kwargs):
        return self._select


def test_styled_select_uses_value_change_callback_for_updates():
    advanced_inputs = _load_advanced_inputs("test_advanced_inputs_value_change")
    fake_select = _FakeSelect()
    advanced_inputs.ui = _FakeUI(fake_select)

    calls = []

    advanced_inputs.styled_select(
        options={"cohere_transcribe_local": "Cohere"},
        value="",
        label="",
        on_change=calls.append,
    )

    assert fake_select.on_value_change_handler is not None
    assert fake_select.on_calls == []

    fake_select.on_value_change_handler(SimpleNamespace(value="cohere_transcribe_local"))

    assert calls == ["cohere_transcribe_local"]
    assert any("use-input" in props for props in fake_select.props_calls)
    assert any('dropdown-icon="search"' in props for props in fake_select.props_calls)


def test_styled_select_can_disable_search_input_mode():
    advanced_inputs = _load_advanced_inputs("test_advanced_inputs_plain_select")
    fake_select = _FakeSelect()
    advanced_inputs.ui = _FakeUI(fake_select)

    advanced_inputs.styled_select(
        options={"en": "English (en)", "ja": "Japanese (ja)"},
        value=None,
        label="",
        searchable=False,
    )

    combined_props = " ".join(fake_select.props_calls)

    assert "use-input" not in combined_props
    assert "fill-input" not in combined_props
    assert "hide-selected" not in combined_props
    assert 'dropdown-icon="arrow_drop_down"' in combined_props
