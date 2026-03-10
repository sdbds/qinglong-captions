"""GLM Provider"""
from __future__ import annotations

import base64
import time
from typing import Any, Iterable, Optional

from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider
from utils.parse_display import extract_code_block_content


def _collect_stream_glm(responses: Iterable[Any], console: Console) -> str:
    """Collect streamed text from GLM responses.

    Preserve original behavior: print raw chunk, print the whole aggregated text each step.
    """
    chunks = ""
    for chunk in responses:
        print(chunk)
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
            chunks += chunk.choices[0].delta.content
        try:
            console.print(chunks, end="", overflow="ellipsis")
        except Exception:
            console.print(Text(chunks), end="", overflow="ellipsis")
        finally:
            console.file.flush()
    console.print("\n")
    return chunks


def attempt_glm(
    *,
    client: Any,
    model_path: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
) -> str:
    """Single-attempt GLM request.

    Returns SRT content, raises on retryable conditions.
    """
    start_time = time.time()

    responses = client.chat.completions.create(
        model=model_path,
        messages=messages,
        stream=True,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_glm(responses, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    content = extract_code_block_content(response_text, "srt", console)
    if not content:
        raise Exception("RETRY_EMPTY_CONTENT")

    if progress and task_id is not None:
        progress.update(task_id, description="Processing media...")
    return content


@register_provider("glm")
class GLMProvider(CloudVLMProvider):
    """GLM API Provider"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, 'glm_api_key', '') != "" and mime.startswith("video")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=self.ctx.args.glm_api_key)

        # GLM 只支持视频
        with open(media.uri, "rb") as video_file:
            video_base = base64.b64encode(video_file.read()).decode("utf-8")

        messages = [
            {"role": "system", "content": prompts.system},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_base}},
                    {"type": "text", "text": prompts.user},
                ],
            },
        ]

        result = attempt_glm(
            client=client,
            model_path=self.ctx.args.glm_model_path,
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
