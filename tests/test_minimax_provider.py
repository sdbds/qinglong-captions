import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rich.console import Console


def test_m31_preview_forces_reasoning_split_when_disabled_in_config():
    from module.providers.cloud_vlm.minimax_code import attempt_minimax_code

    client = MagicMock()
    client.chat.completions.create.return_value = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        reasoning_content=None,
                        content='{"long_description": "ok"}',
                    )
                )
            ]
        )
    ]

    with patch("utils.parse_display.display_caption_and_rate"):
        attempt_minimax_code(
            client=client,
            model_path="MiniMax-M3.1-Flash-Preview",
            messages=[{"role": "user", "content": "describe"}],
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            uri="sample.jpg",
            reasoning_split=False,
        )

    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning_split": True,
    }
