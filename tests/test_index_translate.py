from __future__ import annotations

import sys
import types
from collections import UserDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from utils import transformer_loader

MODEL_IDS = ("IndexTeam/Index-Translate-2B", "IndexTeam/Index-Translate-9B")


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_index_openai_uses_non_thinking_translation_request_and_served_alias(model_id):
    from module.providers.local_llm.index_translate import IndexTranslateProvider

    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="<think>internal</think>\n# Translation __QLP_0__"))]
    )
    provider = IndexTranslateProvider(
        model_id=model_id,
        backend="openai",
        openai_base_url="http://127.0.0.1:9000/v1",
        openai_api_key="local-key",
        openai_model_name="translation-server",
        max_new_tokens=777,
        temperature=0.0,
    )
    with (
        patch("openai.OpenAI", return_value=client),
        patch.object(provider, "_get_or_load_components", side_effect=AssertionError("API must not load weights")),
    ):
        result = provider.translate("# Hello __QLP_0__", "auto", "zh_cn", context="Previous chunk", glossary="Hello: Greeting")

    assert result == "# Translation __QLP_0__"
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "translation-server"
    assert request["max_tokens"] == 777
    assert request["temperature"] == 0.0
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert len(request["messages"]) == 1
    assert request["messages"][0]["role"] == "user"
    prompt = request["messages"][0]["content"]
    assert "简体中文" in prompt
    assert "auto" not in prompt
    assert "〖源文〗\n# Hello __QLP_0__" in prompt
    assert "〖约束要求〗" in prompt
    assert "Markdown" in prompt
    assert "<context>\nPrevious chunk\n</context>" in prompt
    assert "<glossary>\nHello: Greeting\n</glossary>" in prompt


@pytest.mark.parametrize(
    ("source_lang", "target_lang", "source_name", "target_name"),
    [("EN", "zh-CN", "英语", "简体中文"), ("ja", "zh_tw", "日语", "繁体中文"), ("zh-hans", "Esperanto", "简体中文", "Esperanto")],
)
def test_index_prompt_resolves_language_aliases_without_changing_source(source_lang, target_lang, source_name, target_name):
    from module.providers.local_llm.index_translate import build_translation_prompt

    text = "# Header\n\n- __QLP_0__ 42"
    prompt = build_translation_prompt(text=text, source_lang=source_lang, target_lang=target_lang)
    assert source_name in prompt.splitlines()[0]
    assert target_name in prompt.splitlines()[0]
    assert f"〖源文〗\n{text}\n\n〖约束要求〗" in prompt
    assert "<context>" not in prompt
    assert "<glossary>" not in prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  # Result __QLP_0__  ", "# Result __QLP_0__"),
        ("<think>notes\nmore notes</think>\n```markdown\n# Result\n```", "# Result"),
        ("notes</think>\nResult", "Result"),
        ("<think>\n</think>\nResult", "Result"),
    ],
)
def test_index_output_removes_reasoning_and_outer_markdown_fence(raw, expected):
    from module.providers.local_llm.index_translate import clean_translation_output

    assert clean_translation_output(raw) == expected


@pytest.mark.parametrize("raw", ["<think>unfinished reasoning", "<think>reasoning only</think>\n"])
def test_index_output_rejects_thinking_without_a_translation(raw):
    from module.providers.local_llm.index_translate import clean_translation_output

    with pytest.raises(RuntimeError, match="translation"):
        clean_translation_output(raw)


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("temperature", [0.0, 0.4])
def test_index_direct_uses_chat_template_and_preserves_attention_mask(model_id, temperature):
    import torch

    from module.providers.local_llm.index_translate import IndexTranslateProvider

    captured = {}

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["template_kwargs"] = kwargs
            return UserDict(input_ids=torch.tensor([[11, 12]]), attention_mask=torch.tensor([[1, 1]]))

        def decode(self, ids, **kwargs):
            captured["decoded_ids"] = ids.tolist()
            assert kwargs == {"skip_special_tokens": True}
            return "<think></think>\nTranslated __QLP_0__"

    class Model:
        def parameters(self):
            return iter([torch.nn.Parameter(torch.empty(0))])

        def prepare_inputs_for_generation(self, input_ids, attention_mask=None):
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def generate(self, **kwargs):
            captured["generation"] = kwargs
            return torch.tensor([[11, 12, 21, 22]])

    provider = IndexTranslateProvider(model_id=model_id, max_new_tokens=555, temperature=temperature)
    with patch.object(provider, "_get_or_load_components", return_value=(Tokenizer(), Model())):
        result = provider.translate("Hello __QLP_0__", "en", "zh_cn")

    assert result == "Translated __QLP_0__"
    assert captured["decoded_ids"] == [21, 22]
    assert captured["template_kwargs"] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "return_tensors": "pt",
        "return_dict": True,
    }
    generation = captured["generation"]
    assert generation["input_ids"].tolist() == [[11, 12]]
    assert generation["attention_mask"].tolist() == [[1, 1]]
    assert generation["max_new_tokens"] == 555
    assert generation["do_sample"] is (temperature > 0)
    assert "eos_token_id" not in generation, "Keep both stop tokens from the model's generation_config.json"
    assert "repetition_penalty" not in generation
    assert "top_k" not in generation
    assert "top_p" not in generation
    if temperature:
        assert generation["temperature"] == temperature
    else:
        assert "temperature" not in generation


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("attn_impl", ["eager", "sdpa"])
def test_index_loader_uses_qwen35_architecture_and_shared_download_helper(model_id, attn_impl):
    from module.providers.local_llm.index_translate import IndexTranslateProvider

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = type("AutoTokenizer", (), {})
    fake_transformers.Qwen3_5ForConditionalGeneration = type("Qwen3_5ForConditionalGeneration", (), {})
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=2)
    model = MagicMock()
    model.eval.return_value = model
    calls = []

    def load(component_cls, repo_id, **kwargs):
        calls.append((component_cls, repo_id, kwargs))
        return tokenizer if component_cls is fake_transformers.AutoTokenizer else model

    with (
        patch.dict(sys.modules, {"transformers": fake_transformers}),
        patch.object(transformer_loader, "load_pretrained_component", side_effect=load),
        patch.object(IndexTranslateProvider, "_resolve_device_dtype", return_value=("cpu", "bfloat16", attn_impl)),
    ):
        loaded = IndexTranslateProvider(model_id=model_id)._load_components()

    assert loaded == (tokenizer, model)
    assert tokenizer.pad_token_id == 2
    assert calls[0][0] is fake_transformers.AutoTokenizer
    assert calls[1][0] is fake_transformers.Qwen3_5ForConditionalGeneration
    assert all(call[1] == model_id for call in calls)
    kwargs = calls[1][2]
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["device_map"] == "auto"
    assert "torch_dtype" not in kwargs
    if attn_impl == "eager":
        assert "attn_implementation" not in kwargs
    else:
        assert kwargs["attn_implementation"] == "sdpa"
