"""Ark (Volcano Engine) Provider"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider
from utils.parse_display import extract_code_block_content


# ---------------------------------------------------------------------------
# Ark streaming / attempt helpers (migrated from ark_provider.py)
# ---------------------------------------------------------------------------

def _collect_stream_ark(responses: Iterable[Any], console: Console) -> str:
    """Collect streamed text from Ark responses.

    Tries OpenAI-like delta.content first, falls back to other possible fields.
    """
    chunks: list[str] = []
    for chunk in responses:
        text_piece = ""
        try:
            # OpenAI-like streaming
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and hasattr(chunk.choices[0], "delta")
                and getattr(chunk.choices[0].delta, "content", None) is not None
            ):
                text_piece = chunk.choices[0].delta.content
            # Sometimes streaming message content can appear fully in message
            elif (
                hasattr(chunk, "choices")
                and chunk.choices
                and hasattr(chunk.choices[0], "message")
                and getattr(chunk.choices[0].message, "content", None)
            ):
                text_piece = chunk.choices[0].message.content  # type: ignore
            else:
                # Generic fallback
                text_piece = getattr(chunk, "text", "") or ""
        except Exception:
            pass

        if text_piece:
            chunks.append(text_piece)
            try:
                console.print(text_piece, end="", overflow="ellipsis")
            except Exception:
                console.print(Text(text_piece), end="", overflow="ellipsis")
            finally:
                console.file.flush()

    console.print("\n")
    return "".join(chunks)


def attempt_ark(
    *,
    client: Any,
    model_path: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
) -> str:
    """Single-attempt Ark request.

    Returns SRT content. Raises on retryable conditions (RETRY_EMPTY_CONTENT).
    """
    start_time = time.time()

    completion = client.chat.completions.create(
        model=model_path.replace(".", "-"),
        messages=messages,
        stream=True,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_ark(completion, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    # Normalize inline color tags to HTML-like for consistency
    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    content = extract_code_block_content(response_text, "srt", console)
    if not content:
        # Trigger retry when content is empty
        raise Exception("RETRY_EMPTY_CONTENT")

    if progress and task_id is not None:
        progress.update(task_id, description="Processing media...")
    return content


# ---------------------------------------------------------------------------
# V2 Provider class
# ---------------------------------------------------------------------------

@register_provider("ark")
class ArkProvider(CloudVLMProvider):
    """Ark (Volcano Engine) API Provider"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, "ark_api_key", "") != "" and mime.startswith("video")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from volcenginesdkarkruntime import Ark

        client = Ark(api_key=self.ctx.args.ark_api_key)

        self.log(f"Ark model: {getattr(self.ctx.args, 'ark_model_path', '')}", "blue")

        # 读取配置
        ark_section = self.ctx.config.get("ark", {})
        cfg_fps = ark_section.get("fps")
        ark_fps = float(cfg_fps) if cfg_fps is not None else getattr(self.ctx.args, "ark_fps", 2)
        self.log(f"Ark fps: {ark_fps}", "blue")

        # 编码视频
        with open(media.uri, "rb") as video_file:
            video_base = base64.b64encode(video_file.read()).decode("utf-8")

        try:
            file_size = Path(media.uri).stat().st_size
        except Exception:
            file_size = -1
        self.log(f"Ark input size: {file_size} bytes; base64 length: {len(video_base)}", "blue")

        messages = [
            {"role": "system", "content": prompts.system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:{media.mime};base64,{video_base}",
                            "fps": ark_fps,
                        },
                    },
                    {"type": "text", "text": prompts.user},
                ],
            },
        ]

        result = attempt_ark(
            client=client,
            model_path=self.ctx.args.ark_model_path,
            messages=messages,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
        )

        return CaptionResult(raw=result, metadata={"provider": self.name})

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "429" in msg:
                return 59.0
            if "502" in msg or "RETRY_EMPTY_CONTENT" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        return cfg
