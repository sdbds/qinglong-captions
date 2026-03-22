"""MiniMax Code Provider

MiniMax Code 专用 Provider - 针对代码理解和结构化输出优化
基于 MiniMax M2/M2.1 系列的强大编程能力

特性:
- 专为代码分析和理解优化
- 支持结构化 JSON 输出
- 支持 reasoning_split 分离推理过程
- 支持多模态输入（图像、视频）
- 支持 tags 高亮显示（像 kimi_vl 一样）

API 文档: https://platform.minimaxi.com/docs/api-reference/api-overview
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, List, Optional

from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.cloud_vlm_base import CloudVLMProvider
from module.providers.registry import register_provider
from module.providers.utils import build_vision_messages
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


def _collect_stream_minimax_code(completion: Any, console: Console) -> str:
    """收集 MiniMax Code 流式响应

    支持 reasoning_split 模式，会分离推理内容和最终答案
    """
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    is_reasoning = False

    for chunk in completion:
        try:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 检查是否有 reasoning_content (当使用 reasoning_split 时)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_chunks.append(delta.reasoning_content)
                if not is_reasoning:
                    is_reasoning = True
                    console.print("[dim]<thinking>[/dim] ", end="")
                console.print(".", end="", style="dim")
                continue

            # 标准内容
            if hasattr(delta, "content") and delta.content is not None:
                if is_reasoning:
                    is_reasoning = False
                    console.print("[dim]</thinking>[/dim]\n")
                chunks.append(delta.content)
                console.print(".", end="", style="blue")

        except Exception:
            pass

    console.print("\n")
    return "".join(chunks)


def attempt_minimax_code(
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
    reasoning_split: bool = True,
    mode: str = "all",
    tags: Optional[List[str]] = None,
) -> str:
    """执行 MiniMax Code API 请求

    Args:
        client: OpenAI 客户端实例
        model_path: 模型名称 (推荐 MiniMax-M2 或 MiniMax-M2.1)
        messages: 消息列表
        console: Rich Console
        progress: 进度条（可选）
        task_id: 任务ID（可选）
        uri: 媒体文件路径
        image_pixels: 图像像素数据（用于显示）
        pair_pixels: 配对图像像素数据
        reasoning_split: 是否分离推理内容（默认 True）
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

    # 构建请求参数 - MiniMax Code 默认启用 reasoning_split
    extra_body = {"reasoning_split": reasoning_split}

    # 对于 M2/M2.1 模型，可以使用较低的 temperature 获得更确定的输出
    temperature = 0.3 if reasoning_split else 0.7

    completion = client.chat.completions.create(
        model=model_path,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=8192,
        stream=True,
        extra_body=extra_body,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions (code mode)")

    response_text = _collect_stream_minimax_code(completion, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    # 清理可能的格式标记
    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    # 尝试提取代码块内容
    content = extract_code_block_content(response_text, "json", console)
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


@register_provider("minimax_code")
class MiniMaxCodeProvider(CloudVLMProvider):
    """MiniMax Code Provider

    针对代码理解和结构化输出优化的 MiniMax Provider
    特点:
    - 默认启用 reasoning_split 分离推理过程
    - 针对 M2/M2.1 模型优化
    - 更强的 JSON 结构化输出能力
    - 支持 tags 高亮显示

    Base URL: https://api.minimax.io/v1
    """

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        """检查是否能处理此请求

        minimax_code 优先级检查：
        1. 需要 minimax_code_api_key
        2. 支持 image 和 video
        """
        return getattr(args, "minimax_code_api_key", "") != "" and mime.startswith(("image", "video"))

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        """执行 MiniMax Code API 调用"""
        from openai import OpenAI

        # 获取配置
        api_key = self.ctx.args.minimax_code_api_key
        base_url = getattr(self.ctx.args, "minimax_code_base_url", "https://api.minimax.io/v1")
        # 默认使用 M2 模型，专为代码和Agent工作流优化
        model_path = getattr(self.ctx.args, "minimax_code_model_path", "MiniMax-M2")

        client = OpenAI(api_key=api_key, base_url=base_url)

        self.log(f"MiniMax Code base_url: {base_url}", "blue")
        self.log(f"MiniMax Code model: {model_path}", "blue")

        # 加载预生成标签
        merged_tags = self._load_tags(media)
        if merged_tags:
            self.log(f"Loaded {len(merged_tags)} tags for image", "blue")

        # 构建消息（包含 tags 注入）
        messages = self._build_messages(media, prompts, merged_tags)
        if not messages:
            return CaptionResult(raw="")

        # 读取配置
        minimax_config = self.ctx.config.get("minimax_code", {})
        reasoning_split = minimax_config.get("reasoning_split", True)

        # 执行请求
        result = attempt_minimax_code(
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

        # 处理 JSON 响应 - minimax_code 更强调结构化输出
        try:
            raw_result = str(result).strip()
            # 清理可能的 markdown 代码块
            if raw_result.startswith("```"):
                raw_result = json.loads(raw_result.replace("```json", "").replace("```", "").strip())
            else:
                parsed = json.loads(raw_result)

            if isinstance(parsed, dict):
                # 标准化字段名
                short_value = parsed.get("short_description") or parsed.get("short") or ""
                long_value = parsed.get("long_description") or parsed.get("long") or parsed.get("description") or ""

                if "short_description" not in parsed and short_value:
                    parsed["short_description"] = short_value
                if "long_description" not in parsed and long_value:
                    parsed["long_description"] = long_value

                mode = getattr(self.ctx.args, "mode", "all")
                if mode == "short":
                    parsed.pop("long", None)
                    parsed.pop("long_description", None)
                    parsed.pop("short", None)
                    parsed["short_description"] = short_value
                elif mode == "long":
                    parsed.pop("short", None)
                    parsed.pop("short_description", None)
                    parsed.pop("long", None)
                    parsed["long_description"] = long_value

                return CaptionResult(
                    raw=json.dumps(parsed, ensure_ascii=False),
                    parsed=parsed,
                    metadata={"provider": self.name}
                )
        except Exception as e:
            print_exception(self.ctx.console, e, prefix="Failed to parse MiniMax Code JSON response", summary_style="yellow")

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
        """构建 MiniMax Code 消息格式

        针对代码/结构化理解优化，添加结构化输出提示
        """
        # 增强用户提示，要求结构化 JSON 输出
        structured_tail = (
            "\n\n请用JSON格式返回结果，包含以下字段:\n"
            "- tags: 标签数组\n"
            "- short_description: 简短描述\n"
            "- long_description: 详细描述\n"
            "- rating: 评分对象\n"
            "- average_score: 平均分数"
        )
        enhanced_user_prompt = prompts.user + structured_tail

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
                        {"type": "text", "text": enhanced_user_prompt},
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
                enhanced_user_prompt,
                media.blob,
                pair_blob=media.pair_blob if pair_dir else None,
                text_first=False
            )

        # 纯文本 fallback
        else:
            messages = [
                {"role": "system", "content": prompts.system},
                {"role": "user", "content": enhanced_user_prompt},
            ]

        # 注入 tags 到消息中
        if tags:
            messages = _inject_tags_into_messages(messages, tags)

        return messages

    def get_retry_config(self):
        """获取重试配置"""
        from module.providers.utils import classify_remote_api_error

        cfg = super().get_retry_config()
        cfg.classify_error = lambda e: classify_remote_api_error(
            e,
            base_wait=cfg.base_wait,
            retry_markers=("RETRY_EMPTY_CONTENT",),
        )
        cfg.on_exhausted = lambda e: (
            print_exception(self.ctx.console, e, prefix="MiniMax Code retries exhausted", summary_style="yellow") or ""
        )
        return cfg
