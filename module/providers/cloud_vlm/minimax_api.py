"""MiniMax API Provider

MiniMax 开放平台 API 提供商
支持文本生成、图像理解、视频理解
API 文档: https://platform.minimaxi.com/docs/api-reference/api-overview

支持模型:
- MiniMax-M2.5: 顶尖性能与极致性价比 (输出速度约60tps)
- MiniMax-M2.5-highspeed: M2.5 极速版 (输出速度约100tps)
- MiniMax-M2.1: 强大多语言编程能力 (输出速度约60tps)
- MiniMax-M2.1-highspeed: M2.1 极速版 (输出速度约100tps)
- MiniMax-M2: 专为高效编码与Agent工作流而生
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, List, Optional

from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider
from providers.utils import build_vision_messages
from utils.console_util import print_exception


def _load_tags_from_json(uri: str, progress: Optional[Progress] = None) -> list[str]:
    """Load pre-generated tags from datasets/tags.json keyed by absolute file path."""
    tags_json_path = Path(__file__).resolve().parents[3] / "datasets" / "tags.json"
    if not tags_json_path.exists():
        return []
    try:
        data = json.loads(tags_json_path.read_text(encoding="utf-8"))
        entry = data.get(str(Path(uri).resolve()))
        if not isinstance(entry, dict):
            return []
        tags = []
        for v in entry.values():
            if isinstance(v, list):
                tags.extend([str(i) for i in v])
        return tags
    except Exception as e:
        if progress:
            print_exception(progress.console, e, prefix=f"Error loading or parsing {tags_json_path}")
        return []


def _inject_tags_into_messages(messages: list[dict[str, Any]], tags: list[str]) -> list[dict[str, Any]]:
    """Append tags hint into the text part of user message so model can see them."""
    if not tags:
        return messages
    tag_str = ", ".join(tags)
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    existing = part.get("text", "")
                    part["text"] = f"{existing}\nExisting tags: {tag_str}".strip()
                    return messages
    return messages


def _collect_stream_minimax(completion: Any, console: Console) -> str:
    """收集 MiniMax (OpenAI兼容) 流式响应"""
    chunks: list[str] = []
    for chunk in completion:
        try:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and hasattr(chunk.choices[0], "delta")
                and getattr(chunk.choices[0].delta, "content", None) is not None
            ):
                chunks.append(chunk.choices[0].delta.content)
                console.print(".", end="", style="blue")
        except Exception:
            pass
    console.print("\n")
    return "".join(chunks)


def attempt_minimax(
    *,
    client: Any,
    model_path: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress] = None,
    task_id: Optional[Any] = None,
    uri: str = "",
    image_pixels: Optional[Any] = None,
    pair_pixels: Optional[Any] = None,
    reasoning_split: bool = False,
    mode: str = "all",
    tags: Optional[List[str]] = None,
) -> str:
    """执行 MiniMax API 请求

    Args:
        client: OpenAI 客户端实例
        model_path: 模型名称
        messages: 消息列表
        console: Rich Console
        progress: 进度条（可选）
        task_id: 任务ID（可选）
        uri: 媒体文件路径
        image_pixels: 图像像素数据（用于显示）
        pair_pixels: 配对图像像素数据
        reasoning_split: 是否分离推理内容
        mode: 输出模式 (all/short/long)
        tags: 预加载的标签列表
    """
    import time

    from utils.parse_display import (
        display_caption_and_rate,
        display_caption_layout,
        display_pair_image_description,
        extract_code_block_content,
        process_llm_response,
    )
    from utils.stream_util import format_description

    start_time = time.time()

    # 构建请求参数
    extra_body = {"reasoning_split": reasoning_split} if reasoning_split else None

    completion = client.chat.completions.create(
        model=model_path,
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        max_tokens=8192,
        stream=True,
        extra_body=extra_body,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")

    response_text = _collect_stream_minimax(completion, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    # 清理可能的格式标记
    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    # 尝试提取代码块内容
    content = extract_code_block_content(response_text, "srt", console)
    if content:
        response_text = content

    # 解析响应以提取标签和描述
    tag_description = ""
    short_description = ""
    long_description = ""

    # 尝试 JSON 解析
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            tags_value = data.get("tags")
            if isinstance(tags_value, list):
                tag_description = ", ".join(str(t) for t in tags_value)
            elif isinstance(tags_value, str):
                tag_description = tags_value
            short_description = data.get("short_description", "") or data.get("short", "")
            long_description = data.get("long_description", "") or data.get("long", data.get("description", ""))
    except Exception:
        pass

    # 如果 JSON 解析失败，使用 ### 分割
    if not short_description and not long_description and "###" in response_text:
        short_description, long_description = process_llm_response(response_text)

    # Fallback
    if not long_description:
        long_description = response_text.strip()
    if not short_description and "\n" in long_description:
        short_description = long_description.split("\n", 1)[0].strip()

    # 如果模型没有返回 tags 但我们有预加载的 tags，使用它们
    if not tag_description and tags:
        tag_description = ", ".join(tags)

    # 使用 tags 高亮描述
    short_highlight_rate = 0
    long_highlight_rate = 0
    if tag_description:
        short_description, short_highlight_rate = format_description(short_description, tag_description)
        long_description, long_highlight_rate = format_description(long_description, tag_description)

    # 显示结果
    if pair_pixels is not None and image_pixels is not None:
        display_pair_image_description(
            title=Path(uri).name,
            description=long_description,
            pixels=image_pixels,
            pair_pixels=pair_pixels,
            panel_height=32,
            console=console,
        )
    elif image_pixels is not None:
        display_caption_layout(
            title=Path(uri).name,
            tag_description=tag_description,
            short_description=short_description,
            long_description=long_description,
            pixels=image_pixels,
            short_highlight_rate=short_highlight_rate,
            long_highlight_rate=long_highlight_rate,
            panel_height=32,
            console=console,
        )
    else:
        display_caption_and_rate(
            title=Path(uri).name,
            tag_description=tag_description,
            long_description=long_description,
            pixels=None,
            rating=[],
            average_score=0,
            panel_height=32,
            console=console,
        )

    return response_text


@register_provider("minimax_api")
class MiniMaxAPIProvider(CloudVLMProvider):
    """MiniMax API Provider

    使用 OpenAI 兼容接口调用 MiniMax API
    Base URL: https://api.minimax.io/v1
    """

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        """检查是否能处理此请求

        minimax_api 优先级检查：
        1. 需要 minimax_api_key
        2. 支持 image 和 video
        """
        return getattr(args, "minimax_api_key", "") != ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        """执行 MiniMax API 调用"""
        from openai import OpenAI

        # 获取配置
        api_key = self.ctx.args.minimax_api_key
        base_url = getattr(self.ctx.args, "minimax_api_base_url", "https://api.minimax.io/v1")
        model_path = getattr(self.ctx.args, "minimax_model_path", "MiniMax-M2.5")

        client = OpenAI(api_key=api_key, base_url=base_url)

        self.log(f"MiniMax API base_url: {base_url}", "blue")
        self.log(f"MiniMax model: {model_path}", "blue")

        # 加载预生成标签
        merged_tags = self._load_tags(media)
        if merged_tags:
            self.log(f"Loaded {len(merged_tags)} tags for image", "blue")

        # 构建消息（包含 tags 注入）
        messages = self._build_messages(media, prompts, merged_tags)
        if not messages:
            return CaptionResult(raw="")

        # 读取配置
        minimax_config = self.ctx.config.get("minimax_api", {})
        reasoning_split = minimax_config.get("reasoning_split", False)

        # 执行请求
        result = attempt_minimax(
            client=client,
            model_path=model_path,
            messages=messages,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            uri=media.uri,
            image_pixels=media.pixels,
            pair_pixels=media.pair_pixels,
            reasoning_split=reasoning_split,
            mode=getattr(self.ctx.args, "mode", "all"),
            tags=merged_tags,
        )

        # 尝试解析 JSON 响应
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if isinstance(parsed, dict):
                mode = getattr(self.ctx.args, "mode", "all")
                # 根据 mode 过滤字段
                if mode == "short":
                    parsed.pop("long", None)
                    parsed.pop("long_description", None)
                elif mode == "long":
                    parsed.pop("short", None)
                    parsed.pop("short_description", None)
                return CaptionResult(
                    raw=json.dumps(parsed, ensure_ascii=False),
                    parsed=parsed,
                    metadata={"provider": self.name}
                )
        except Exception:
            pass

        return CaptionResult(
            raw=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
            metadata={"provider": self.name}
        )

    def _load_tags(self, media: MediaContext) -> list[str]:
        """加载预生成标签（从 sidecar .txt 和 datasets/tags.json）"""
        if not media.mime.startswith("image"):
            return []

        tags: list[str] = []

        # 从 sidecar .txt 文件加载
        captions_path = Path(media.uri).with_suffix(".txt")
        if captions_path.exists():
            try:
                with open(captions_path, "r", encoding="utf-8") as f:
                    tags = [line.strip() for line in f.readlines() if line.strip()]
            except Exception:
                pass

        # 从 datasets/tags.json 加载
        tags_from_json = _load_tags_from_json(media.uri, self.ctx.progress)

        # 合并（优先使用 json 的 tags）
        return tags_from_json if tags_from_json else tags

    def _build_messages(self, media: MediaContext, prompts: PromptContext, tags: Optional[list[str]] = None) -> list:
        """构建 MiniMax 消息格式

        MiniMax 支持标准的 OpenAI 格式，包括:
        - 图像: image_url with base64
        - 视频: video_url with base64
        """
        messages = []

        # 视频处理
        if media.mime.startswith("video"):
            with open(media.uri, "rb") as f:
                video_base = base64.b64encode(f.read()).decode("utf-8")
            video_data_url = f"data:{media.mime};base64,{video_base}"

            messages = [
                {"role": "system", "content": prompts.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_data_url}},
                        {"type": "text", "text": prompts.user},
                    ],
                },
            ]

        # 图像处理
        elif media.mime.startswith("image"):
            if media.blob is None:
                return []

            pair_dir = getattr(self.ctx.args, "pair_dir", "")
            if pair_dir and not media.pair_blob:
                return []

            messages = build_vision_messages(
                prompts.system,
                prompts.user,
                media.blob,
                pair_blob=media.pair_blob if pair_dir else None,
                text_first=False
            )

        # 纯文本 fallback
        else:
            messages = [
                {"role": "system", "content": prompts.system},
                {"role": "user", "content": prompts.user},
            ]

        # 注入 tags 到消息中
        if tags:
            messages = _inject_tags_into_messages(messages, tags)

        return messages

    def get_retry_config(self):
        """获取重试配置"""
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            # 429 限流错误，等待59秒
            if "429" in msg:
                return 59.0
            # 502 错误或空内容，使用基础等待时间
            if "502" in msg or "RETRY_EMPTY_CONTENT" in msg:
                return cfg.base_wait
            # 其他错误不重试
            return None

        cfg.classify_error = classify
        cfg.on_exhausted = lambda e: (
            print_exception(self.ctx.console, e, prefix="MiniMax API retries exhausted", summary_style="yellow") or ""
        )
        return cfg
