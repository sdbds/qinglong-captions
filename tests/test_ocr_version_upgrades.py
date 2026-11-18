import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from rich.console import Console


def _pages(tmp_path, count=3):
    pages = []
    for index in range(count):
        path = tmp_path / f"page_{index + 1:04d}.png"
        Image.new("RGB", (100, 200), "white").save(path)
        pages.append(path)
    return pages


@pytest.mark.parametrize(
    "family,model_id,version",
    [
        ("unlimited", "baidu/Unlimited-OCR", 1),
        ("unlimited", "baidu/Unlimited-OCR-2", 2),
        ("logics", "Logics-MLLM/Logics-Parsing-v2", 2),
        ("logics", "Logics-MLLM/Logics-Parsing-V3", 3),
        ("lighton", "lightonai/LightOnOCR-2-1B", 2),
        ("lighton", "lightonai/LightOnOCR-3-0.8B", 3),
        ("lighton", "lightonai/LightOnOCR-3-1B", 3),
        ("lighton", "lightonai/LightOnOCR-3-4B", 3),
    ],
)
def test_model_generation_selects_matching_protocol(family, model_id, version):
    from utils.ocr_model_config import ocr_model_version

    assert ocr_model_version(family, model_id) == version


def test_version_override_supports_server_alias_and_rejects_unknown_version():
    from utils.ocr_model_config import ocr_model_version

    assert ocr_model_version("logics", "my-local-server", "3") == 3
    with pytest.raises(ValueError, match="model_version"):
        ocr_model_version("lighton", "alias", "4")


def test_unlimited_v2_merges_cross_page_text_and_exports_boxes(tmp_path):
    from module.providers.ocr.unlimited_v2 import process_output

    raw = (
        "<PAGE><BLOCK>title<LEVEL1>[10,20,900,100]<FORK>Report\n"
        "<BLOCK>cross_text[10,100,900,900]<FORK>first "
        "<PAGE>"
        "<PAGE><BLOCK>append_text[10,100,900,900]<FORK>second\n"
        "<BLOCK>image[100,200,500,600]<FORK>Diagram"
    )
    markdown = process_output(raw, _pages(tmp_path), tmp_path / "out", model_id="baidu/Unlimited-OCR-2")
    assert "# Report" in markdown
    assert "first second" in markdown
    assert "<BLOCK>" not in markdown and "<PAGE>" not in markdown
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert payload["page_count"] == 3
    assert payload["blocks"][-1]["page_index"] == 2
    assert payload["blocks"][-1]["bbox"] == [100, 200, 500, 600]
    assert (tmp_path / "out" / "page3_100_200_500_600.jpg").is_file()


def test_unlimited_v2_merges_table_headers_without_dropping_rows(tmp_path):
    from module.providers.ocr.unlimited_v2 import process_output

    raw = (
        "<PAGE><BLOCK>cross_table[0,0,1000,1000]<FORK>"
        "<table><tr><th>Name</th></tr><tr><td>A</td></tr></table>"
        "<PAGE><BLOCK>append_table[0,0,1000,1000]<FORK>"
        "<table><tr><th>Name</th></tr><tr><td>B</td></tr></table>"
    )
    markdown = process_output(raw, _pages(tmp_path, 2), tmp_path / "out", model_id="v2")
    assert markdown.count("<table>") == 1
    assert markdown.count("<th>Name</th>") == 1
    assert "<td>A</td>" in markdown and "<td>B</td>" in markdown


def test_logics_v3_passes_state_and_absolute_ids_to_last_window(tmp_path):
    from module.providers.ocr.logics_v3 import run_document

    requests = []
    answers = iter(
        [
            "<|title|><|level|>2<|text|>Results<|loc|>0\n<|text|>first <|loc|>1<|truncated|>",
            "<|continued|><|text|>second<|loc|>2\n"
            "<|figure|><|loc|>2,100,200,500,600\n"
            "<|caption|>Figure one<|loc|>2<|ref|>2,100,200,500,600",
        ]
    )

    def infer(pages, prompt):
        requests.append((pages, prompt))
        return next(answers)

    markdown = run_document(_pages(tmp_path), infer, tmp_path / "out", model_id="v3", window_size=2)
    assert [len(pages) for pages, _ in requests] == [2, 1]
    assert "Task mode: longdoc_parse." in requests[-1][1]
    assert "Current page ids: 2" in requests[-1][1]
    assert "<|open_tree|>" in requests[-1][1]
    assert "Results" in requests[-1][1] and "first" in requests[-1][1]
    assert "## Results" in markdown
    assert "first second" in markdown
    assert "<|" not in markdown
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert payload["blocks"][-1]["refs"] == [[2, 100, 200, 500, 600]]
    assert payload["blocks"][1]["page_ids"] == [1, 2]
    assert (tmp_path / "out" / "hierarchy.md").is_file()
    assert list((tmp_path / "out" / "images").glob("*.png"))


def test_logics_state_is_document_local_and_single_page_uses_single_prompt(tmp_path):
    from module.providers.ocr.logics_v3 import run_document

    prompts = []

    def infer(pages, prompt):
        prompts.append(prompt)
        return "<|title|>Clean document\n<|text|>content"

    pages = _pages(tmp_path, 1)
    for name in ("one", "two"):
        run_document(pages, infer, tmp_path / name, model_id="v3")
    assert prompts[0] == prompts[1]
    assert "Task mode: single_page_parse." in prompts[0]
    assert "Previous-window state" not in prompts[0]


def test_lighton_grounding_preserves_multiline_tables_and_continuation():
    from module.providers.ocr.lighton import parse_grounding

    raw = (
        "![title](10,20,900,90) # Report\n"
        "![table](10,100,900,500) <table>\n<tr><td>42</td></tr>\n</table>\n"
        "![text+](10,600,900,800) Continued text"
    )
    markdown, blocks = parse_grounding(raw)
    assert markdown.startswith("# Report")
    assert "<table>\n<tr><td>42</td></tr>\n</table>" in markdown
    assert "![title]" not in markdown
    assert blocks[-1] == {
        "label": "text",
        "bbox": [10, 600, 900, 800],
        "text": "Continued text",
        "continued": True,
    }


@pytest.mark.parametrize("raw", ["not grounded", "![text](20,30,10,40) bad", "![text](0,0,1200,500) bad"])
def test_lighton_grounding_rejects_invalid_geometry(raw):
    from module.providers.ocr.lighton import parse_grounding

    with pytest.raises(ValueError, match="grounding|bbox"):
        parse_grounding(raw)


def test_lighton_legacy_defaults_to_plain_and_cannot_request_grounding():
    from module.providers.ocr.lighton import resolve_lighton_mode

    assert resolve_lighton_mode(2, "auto", "") == "plain"
    assert resolve_lighton_mode(3, "auto", "") == "grounding"
    assert resolve_lighton_mode(3, "plain", "") == "plain"
    with pytest.raises(ValueError, match="LightOnOCR-3"):
        resolve_lighton_mode(2, "grounding", "")


def test_ocr_runtime_rejects_length_limited_responses(monkeypatch):
    import openai

    from module.providers.backends import OpenAIChatRuntime, RuntimeBackendConfig

    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="partial"))])
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))),
    )
    backend = OpenAIChatRuntime(RuntimeBackendConfig(base_url="http://local/v1", model_id="test"))
    with pytest.raises(RuntimeError, match="truncated|incomplete"):
        backend.complete([], require_complete=True)


def test_native_truncation_detection_accepts_eos_at_budget():
    from module.providers.ocr.document import ensure_complete_generation

    ensure_complete_generation([10, 11, 2], max_new_tokens=3, eos_token_ids=[2])
    with pytest.raises(RuntimeError, match="truncated"):
        ensure_complete_generation([10, 11, 12], max_new_tokens=3, eos_token_ids=[2])


def _unlimited_v2_runtime(monkeypatch, tmp_path, response):
    from module.providers.ocr import unlimited

    calls = []

    class Model:
        def infer(self, tokenizer, **kwargs):
            calls.append(kwargs)
            return response

    monkeypatch.setattr(
        unlimited,
        "_TRANS_LOADER",
        SimpleNamespace(
            get_or_load_processor=lambda *args, **kwargs: SimpleNamespace(eos_token_id=1),
            get_or_load_model=lambda *args, **kwargs: Model(),
        ),
    )
    monkeypatch.setattr(unlimited, "resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr(unlimited, "display_markdown", lambda **kwargs: None)
    return unlimited, calls


def test_unlimited_v2_uses_native_page_guard_and_does_not_apply_legacy_chunk_budget(tmp_path, monkeypatch):
    raw = "".join(f"<PAGE><BLOCK>text[0,0,1000,1000]<FORK>Page {i}" for i in range(3))
    module, calls = _unlimited_v2_runtime(
        monkeypatch,
        tmp_path,
        {
            "text": raw,
            "finish_reason": "stop",
            "num_pages_out": 3,
            "expected_pages": 3,
            "token_ids": [10, 11],
            "prompt_token_ids": [5],
        },
    )
    pages = _pages(tmp_path)
    monkeypatch.setattr(module, "prepare_document_pages", lambda *args, **kwargs: pages, raising=False)
    result = module.attempt_unlimited_ocr(
        uri=str(tmp_path / "doc.pdf"),
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        model_id="baidu/Unlimited-OCR-2",
        output_dir=str(tmp_path / "out"),
        page_budget=1,
    )
    assert len(calls) == 1
    assert calls[0]["image_file"] == [str(page) for page in pages]
    assert calls[0]["prompt"] == "Multi page parsing."
    assert calls[0]["page_count_guard"] is True
    assert calls[0]["max_length"] == 131072
    assert calls[0]["return_dict"] is True
    assert calls[0]["save_results"] is False
    assert "Page 2" in result and "<BLOCK>" not in result
    assert (tmp_path / "out" / "result.md").read_text() == result


@pytest.mark.parametrize("finish,page_count", [("length", 1), ("stop", 0)])
def test_unlimited_v2_rejects_partial_output_before_publication(tmp_path, monkeypatch, finish, page_count):
    module, _ = _unlimited_v2_runtime(
        monkeypatch,
        tmp_path,
        {
            "text": "<PAGE><BLOCK>text[0,0,1000,1000]<FORK>partial",
            "finish_reason": finish,
            "num_pages_out": page_count,
            "expected_pages": 1,
            "token_ids": [10],
            "prompt_token_ids": [5],
        },
    )
    path = _pages(tmp_path, 1)[0]
    with pytest.raises(RuntimeError, match="incomplete|truncated"):
        module.attempt_unlimited_ocr(
            uri=str(path),
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            model_id="baidu/Unlimited-OCR-2",
            output_dir=str(tmp_path / "out"),
        )
    assert not (tmp_path / "out" / "result.md").exists()
    assert not (tmp_path / "out" / "result.json").exists()


def _native_runtime(monkeypatch, module, answers, *, tokens=(11, 12, 2)):
    import sys
    import types

    import torch

    templates, generations, loaded = [], [], []
    decoded = iter(answers)

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        def to(self, *args, **kwargs):
            return self

    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=2)
        image_processor = SimpleNamespace(min_pixels=1, max_pixels=1)

        def apply_chat_template(self, messages, **kwargs):
            templates.append((messages, kwargs))
            return Inputs(input_ids=torch.tensor([[101, 102]]), pixel_values=torch.ones((1, 3, 4, 4)))

        def decode(self, ids, **kwargs):
            return next(decoded)

        def batch_decode(self, ids, **kwargs):
            return [next(decoded)]

    class Model:
        device = torch.device("cpu")
        config = SimpleNamespace(eos_token_id=7)
        generation_config = SimpleNamespace(eos_token_id=7)

        def generate(self, **kwargs):
            generations.append(kwargs)
            return torch.tensor([[101, 102, *tokens]])

    def load_model(model_id, model_class, **kwargs):
        loaded.append((model_id, model_class.__name__))
        return Model()

    transformers = types.ModuleType("transformers")
    for name in (
        "AutoProcessor",
        "AutoModelForImageTextToText",
        "Qwen3_5ForConditionalGeneration",
        "LightOnOcrForConditionalGeneration",
    ):
        setattr(transformers, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        module,
        "_TRANS_LOADER",
        SimpleNamespace(
            get_or_load_processor=lambda *args, **kwargs: Processor(),
            get_or_load_model=load_model,
        ),
    )
    monkeypatch.setattr(module, "resolve_device_dtype", lambda: ("cpu", torch.float32, "eager"))
    monkeypatch.setattr(module, "display_markdown", lambda **kwargs: None)
    return templates, generations, loaded


def test_logics_v3_native_uses_qwen35_prompt_and_combined_eos(tmp_path, monkeypatch):
    from module.providers.ocr import logics

    templates, generations, loaded = _native_runtime(monkeypatch, logics, ["<|title|>Report\n<|text|>Body"])
    path = _pages(tmp_path, 1)[0]
    markdown = logics.attempt_logics_ocr(
        uri=str(path),
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        model_id="Logics-MLLM/Logics-Parsing-V3",
        output_dir=str(tmp_path / "out"),
    )
    assert loaded[0][1] == "Qwen3_5ForConditionalGeneration"
    assert templates[0][1]["enable_thinking"] is False
    assert generations[0]["eos_token_id"] == [2, 7]
    assert templates[0][0][0]["role"] == "system"
    assert "# Report" in markdown and "Body" in markdown
    assert (tmp_path / "out" / "result.json").exists()


def test_logics_v2_native_still_accepts_html_and_old_prompt(tmp_path, monkeypatch):
    from module.providers.ocr import logics

    templates, _, loaded = _native_runtime(monkeypatch, logics, ['<div class="text">legacy</div>'])
    path = _pages(tmp_path, 1)[0]
    result = logics.attempt_logics_ocr(
        uri=str(path),
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        model_id="Logics-MLLM/Logics-Parsing-v2",
        output_dir=str(tmp_path / "out"),
    )
    assert result == "legacy"
    assert loaded[0][1] == "AutoModelForImageTextToText"
    assert templates[0][0][-1]["content"][-1]["text"] == "QwenVL HTML"
    assert (tmp_path / "out" / "page_0001_raw.html").exists()


def test_logics_v3_openai_uses_same_stateful_pipeline(tmp_path, monkeypatch):
    from module.providers.backends import OpenAIChatRuntime, RuntimeBackendConfig
    from module.providers.ocr import logics

    requests = []

    def complete(self, messages, **kwargs):
        requests.append((messages, kwargs))
        return "<|title|>Remote\n<|text|>Body"

    monkeypatch.setattr(OpenAIChatRuntime, "complete", complete)
    monkeypatch.setattr(logics, "display_markdown", lambda **kwargs: None)
    path = _pages(tmp_path, 1)[0]
    result = logics.attempt_logics_ocr(
        uri=str(path),
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        model_id="Logics-MLLM/Logics-Parsing-V3",
        output_dir=str(tmp_path / "out"),
        runtime=RuntimeBackendConfig(mode="openai", base_url="http://local/v1", model_id="alias"),
    )
    assert "# Remote" in result
    assert requests[0][0][-1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert requests[0][1]["require_complete"] is True
    assert requests[0][1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize(
    "model_id,model_class,edge",
    [
        ("lightonai/LightOnOCR-3-0.8B", "Qwen3_5ForConditionalGeneration", 2048),
        ("lightonai/LightOnOCR-3-1B", "LightOnOcrForConditionalGeneration", 1540),
        ("lightonai/LightOnOCR-3-4B", "Qwen3_5ForConditionalGeneration", 2048),
    ],
)
def test_lighton_v3_native_loads_architecture_and_writes_separate_layout(tmp_path, monkeypatch, model_id, model_class, edge):
    from module.providers.ocr import lighton

    templates, generations, loaded = _native_runtime(monkeypatch, lighton, ["![text](0,0,1000,1000) Grounded"])
    path = tmp_path / "scan.png"
    Image.new("RGB", (3000, 2000), "white").save(path)
    result = lighton.attempt_lighton_ocr(
        uri=str(path),
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        model_id=model_id,
        output_dir=str(tmp_path / "out"),
    )
    assert result == "Grounded"
    assert loaded[0][1] == model_class
    assert templates[0][0][-1]["content"][-1]["text"] == "grounding"
    assert generations[0]["temperature"] == 0.1
    assert generations[0]["do_sample"] is True
    if "Qwen" in model_class:
        assert templates[0][1]["enable_thinking"] is False
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert payload["mode"] == "grounding"
    assert payload["pages"][0]["width"] == edge
    assert payload["pages"][0]["blocks"][0]["bbox"] == [0, 0, 1000, 1000]
    assert (tmp_path / "out" / "result.md").read_text() == "Grounded"
    page_json = tmp_path / "out" / "page_0001" / "result.json"
    page_payload = json.loads(page_json.read_text())
    assert (page_json.parent / page_payload["image"]).is_file()


def test_lighton_v2_native_stays_plain_without_grounding_prompt(tmp_path, monkeypatch):
    from module.providers.ocr import lighton

    templates, generations, loaded = _native_runtime(monkeypatch, lighton, ["legacy"])
    path = _pages(tmp_path, 1)[0]
    result = lighton.attempt_lighton_ocr(
        uri=str(path),
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        model_id="lightonai/LightOnOCR-2-1B",
        output_dir=str(tmp_path / "out"),
    )
    assert result == "legacy"
    assert len(templates[0][0][-1]["content"]) == 1
    assert generations[0]["do_sample"] is False
    assert loaded[0][1] == "LightOnOcrForConditionalGeneration"
    assert not (tmp_path / "out" / "result.json").exists()


def test_lighton_v3_remote_grounding_also_exports_json(tmp_path, monkeypatch):
    from module.providers.backends import OpenAIChatRuntime, RuntimeBackendConfig
    from module.providers.ocr import lighton

    requests = []

    def complete(self, messages, **kwargs):
        requests.append((messages, kwargs))
        return "![title](1,2,300,100) # Remote"

    monkeypatch.setattr(OpenAIChatRuntime, "complete", complete)
    monkeypatch.setattr(lighton, "display_markdown", lambda **kwargs: None)
    path = _pages(tmp_path, 1)[0]
    result = lighton.attempt_lighton_ocr(
        uri=str(path),
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        model_id="lightonai/LightOnOCR-3-0.8B",
        output_dir=str(tmp_path / "out"),
        runtime=RuntimeBackendConfig(mode="openai", base_url="http://local/v1", model_id="alias", temperature=0.1),
    )
    assert result == "# Remote"
    assert requests[0][0][0]["content"][-1]["text"] == "grounding"
    assert requests[0][1]["require_complete"] is True
    assert (tmp_path / "out" / "result.json").exists()


def test_unlimited_v2_does_not_treat_a_forced_page_token_cap_as_complete(tmp_path, monkeypatch):
    raw = "<PAGE><BLOCK>text[0,0,1000,1000]<FORK>A<PAGE><BLOCK>text[0,0,1000,1000]<FORK>B"
    module, _ = _unlimited_v2_runtime(
        monkeypatch,
        tmp_path,
        {
            "text": raw,
            "finish_reason": "stop",
            "num_pages_out": 2,
            "expected_pages": 2,
            "token_ids": [128827, 10, 11, 128827, 12],
            "prompt_token_ids": [5],
        },
    )
    pages = _pages(tmp_path, 2)
    monkeypatch.setattr(module, "prepare_document_pages", lambda *args, **kwargs: pages)
    with pytest.raises(RuntimeError, match="max_page_tokens"):
        module.attempt_unlimited_ocr(
            uri=str(tmp_path / "doc.pdf"),
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            model_id="baidu/Unlimited-OCR-2",
            output_dir=str(tmp_path / "out"),
            max_page_tokens=2,
        )
    assert not (tmp_path / "out" / "result.json").exists()


@pytest.mark.parametrize(
    "raw",
    [
        "<|title|><|level|>99<|text|>Bad title<|loc|>0",
        "<|text|>Missing page location",
        "<|figure|><|loc|>3,0,0,900,900",
    ],
)
def test_logics_v3_rejects_invalid_window_labels(raw):
    from module.providers.ocr.logics_v3 import parse_blocks

    with pytest.raises(ValueError):
        parse_blocks(raw, page_ids=[0, 1], total_pages=2, single_page=False)


def test_lighton_does_not_hide_a_malformed_second_grounding_marker():
    from module.providers.ocr.lighton import parse_grounding

    with pytest.raises(ValueError, match="grounding"):
        parse_grounding("![text](0,0,1000,500) Good\n![caption](bad,box,coords,here) Lost")


def test_lighton_explicit_plain_mode_rejects_a_grounding_prompt():
    from module.providers.ocr.lighton import resolve_lighton_mode

    with pytest.raises(ValueError, match="plain"):
        resolve_lighton_mode(3, "plain", "grounding")


def test_lighton_native_token_limit_does_not_publish_partial_markdown(tmp_path, monkeypatch):
    from module.providers.ocr import lighton

    _native_runtime(monkeypatch, lighton, ["![text](0,0,1000,1000) Partial"], tokens=(10, 11, 12))
    path = _pages(tmp_path, 1)[0]
    with pytest.raises(RuntimeError, match="truncated"):
        lighton.attempt_lighton_ocr(
            uri=str(path),
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            output_dir=str(tmp_path / "out"),
            max_new_tokens=3,
        )
    assert not list((tmp_path / "out").rglob("*.md"))
    assert not list((tmp_path / "out").rglob("*.json"))


def test_lighton_pdf_keeps_blank_page_positions(tmp_path, monkeypatch):
    from module.providers.ocr import document, lighton

    _native_runtime(monkeypatch, lighton, ["![text](0,0,1000,1000) A", "", "![text](0,0,1000,1000) B"])
    monkeypatch.setattr(
        document,
        "iter_pdf_pages_high_quality",
        lambda *args, **kwargs: [SimpleNamespace(page_number=i, image=Image.new("RGB", (16, 32), "white")) for i in (1, 2, 3)],
    )
    result = lighton.attempt_lighton_ocr(
        uri=str(tmp_path / "input.pdf"),
        console=Console(file=io.StringIO()),
        progress=None,
        task_id=None,
        output_dir=str(tmp_path / "out"),
    )
    assert result.count("<--- Page Split --->") == 2
    payload = json.loads((tmp_path / "out" / "result.json").read_text())
    assert [page["page_index"] for page in payload["pages"]] == [0, 1, 2]
    assert payload["pages"][1]["blocks"] == []


def test_logics_remote_legacy_selection_uses_legacy_prompt_even_with_new_local_default():
    from module.providers.base import ProviderContext
    from module.providers.ocr.logics import LogicsOCRProvider

    provider = LogicsOCRProvider(
        ProviderContext(
            console=Console(file=io.StringIO()),
            config={"prompts": {"logics_ocr_prompt": ""}},
            args=SimpleNamespace(openai_base_url="http://local/v1", openai_model_name="Logics-MLLM/Logics-Parsing-v2"),
        )
    )
    assert provider.get_prompts("image/png") == ("", "QwenVL HTML")


@pytest.mark.parametrize("name,tokens", [("logics_ocr", 16384), ("lighton_ocr", 8192)])
def test_new_ocr_provider_runtime_budget_matches_native_default(name, tokens):
    from module.providers.base import ProviderContext
    from module.providers.ocr.lighton import LightOnOCRProvider
    from module.providers.ocr.logics import LogicsOCRProvider

    cls = {"logics_ocr": LogicsOCRProvider, "lighton_ocr": LightOnOCRProvider}[name]
    provider = cls(ProviderContext(console=Console(file=io.StringIO()), config={}, args=SimpleNamespace()))
    assert provider.get_runtime_backend().max_tokens == tokens


def test_lighton_runtime_preserves_selected_model_sampling_override():
    from module.providers.base import ProviderContext
    from module.providers.ocr.lighton import LightOnOCRProvider

    provider = LightOnOCRProvider(
        ProviderContext(
            console=Console(file=io.StringIO()),
            config={"my_checkpoint": {"model_id": "lightonai/LightOnOCR-3-4B", "temperature": 0.3, "max_new_tokens": 1234}},
            args=SimpleNamespace(openai_base_url="http://local/v1", openai_model_name="lightonai/LightOnOCR-3-4B"),
        )
    )
    runtime = provider.get_runtime_backend()
    assert runtime.temperature == 0.3
    assert runtime.max_tokens == 1234


@pytest.mark.parametrize("route", ["unlimited_ocr", "logics_ocr"])
def test_returned_caption_links_resolve_beside_source_while_artifact_markdown_stays_local(tmp_path, monkeypatch, route):
    import re
    from urllib.parse import unquote

    from module.providers.base import ProviderContext
    from module.providers.ocr import logics, unlimited

    source = _pages(tmp_path, 1)[0]
    output = tmp_path / "output with spaces"
    if route == "unlimited_ocr":
        _unlimited_v2_runtime(
            monkeypatch,
            tmp_path,
            {
                "text": "<PAGE><BLOCK>image[100,200,500,600]<FORK>Diagram",
                "finish_reason": "stop",
                "num_pages_out": 1,
                "expected_pages": 1,
                "token_ids": [128827, 10],
                "prompt_token_ids": [5],
            },
        )
        cls = unlimited.UnlimitedOCRProvider
    else:
        _native_runtime(monkeypatch, logics, ["<|figure|><|loc|>0,100,200,500,600"])
        cls = logics.LogicsOCRProvider
    provider = cls(ProviderContext(console=Console(file=io.StringIO()), config={}, args=SimpleNamespace()))
    result = provider.attempt(
        SimpleNamespace(uri=str(source), mime="image/png", pixels=None, extras={"output_dir": output}),
        SimpleNamespace(user="", system=""),
    )
    caption_link = re.search(r"!\[[^]]*\]\(([^)]+)\)", result.raw)[1]
    assert (source.parent / unquote(caption_link)).is_file()
    artifact = (output / "result.md").read_text()
    artifact_link = re.search(r"!\[[^]]*\]\(([^)]+)\)", artifact)[1]
    assert (output / unquote(artifact_link)).is_file()
