import pytest
from rich.console import Console

from module.providers.base import PromptContext, ProviderContext
from module.providers.vision_api.gemini import GeminiProvider
from module.providers.vision_api_base import StructuredOutputConfig


def _config(model, **settings):
    provider = GeminiProvider(ProviderContext(console=Console()))
    config = provider._build_genai_config(
        PromptContext(system="Describe the image.", user="Caption it."),
        settings,
        StructuredOutputConfig(enabled=True, schema=provider._build_pair_image_schema()),
        model,
    )
    return config.model_dump(mode="json", by_alias=True, exclude_none=True)


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
@pytest.mark.filterwarnings("ignore:XHIGH is not a valid ThinkingLevel:UserWarning")
def test_hydrogen_sends_only_the_selected_thinking_level(level):
    payload = _config("gemini-4-hydrogen", thinking_level=level, thinking_budget=128)

    assert payload["thinkingConfig"] == {"thinkingLevel": level.upper()}


def test_hydrogen_preserves_structured_output_without_legacy_generation_settings():
    payload = _config("gemini-4-hydrogen", temperature=3, top_p=2, top_k=-1, max_output_tokens=65536)

    assert not {"thinkingConfig", "temperature", "topP", "topK", "candidateCount"} & payload.keys()
    assert payload["maxOutputTokens"] == 65536
    assert payload["responseMimeType"] == "application/json"
    assert payload["responseSchema"]["properties"] == {"prompt": {"type": "string"}}


def test_hydrogen_rejects_unsupported_minimal_thinking_before_the_request():
    with pytest.raises(ValueError, match="thinking_level"):
        _config("gemini-4-hydrogen", thinking_level="minimal")


def test_pre_hydrogen_models_keep_their_generation_settings():
    third = _config("gemini-3-flash-preview", thinking_level="minimal")
    assert third["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}
    assert not {"temperature", "topP", "topK", "candidateCount"} & third.keys()

    older = _config("gemini-2.5-flash", thinking_budget=128, temperature=0.5, top_p=0.8, top_k=25)
    assert older["thinkingConfig"] == {"thinkingBudget": 128}
    assert (older["temperature"], older["topP"], older["topK"], older["candidateCount"]) == (0.5, 0.8, 25, 1)
