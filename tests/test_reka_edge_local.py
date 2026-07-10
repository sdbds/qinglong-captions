import io
import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent


def test_reka_build_messages_wraps_system_prompt_as_text_blocks():
    from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
    from module.providers.local_vlm.reka_edge_local import RekaEdgeLocalProvider

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"reka_edge_local": {"model_id": "RekaAI/reka-edge-2603"}},
        args=SimpleNamespace(vlm_image_model="reka_edge_local"),
    )
    provider = RekaEdgeLocalProvider(ctx)

    media = MediaContext(
        uri="C:/tmp/sample.png",
        mime="image/png",
        sha256hash="",
        modality=MediaModality.IMAGE,
        extras={},
    )

    messages = provider._build_messages(media, PromptContext(system="system", user="describe"))

    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "system"}],
    }
    assert messages[1]["content"] == [
        {"type": "image", "image": str(Path(media.uri).resolve())},
        {"type": "text", "text": "describe"},
    ]
