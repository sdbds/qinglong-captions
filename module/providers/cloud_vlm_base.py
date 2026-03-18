"""
CloudVLMProvider - 云端 VLM Provider 基类

适用于：StepFun, QwenVL, GLM, Ark, KimiCode, KimiVL

特性：
- 支持 Pair 图像
- 支持多图输入（pair_extras）
- 支持视频上传
- 支持音频
- collect_openai_stream: 通用 OpenAI 兼容流式收集
- attempt_openai_chat: 通用 OpenAI 流式 Chat + SRT 提取
"""

import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

import base64 as _base64

from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from .base import MediaContext, MediaModality, Provider, ProviderContext, PromptContext, ProviderType
from .capabilities import ProviderCapabilities
from .utils import build_vision_messages, encode_image_to_blob
from utils.console_util import print_exception


class CloudVLMProvider(Provider):
    """
    云端 VLM Provider 基类

    提供通用的媒体准备逻辑，支持 Pair 图像和多图输入
    """

    provider_type = ProviderType.CLOUD_VLM
    capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_video=True,
        supports_images=True,
        supports_audio=True,
    )

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        """
        Cloud VLM 通用的媒体准备

        - 图像：编码 base64 + 配对图扫描
        - 音频：小文件读 bytes，大文件留待上传
        - 视频：留待子类处理上传
        """
        file_path = Path(uri)
        file_size = file_path.stat().st_size if file_path.exists() else 0

        # 确定模态
        if mime.startswith("video"):
            modality = MediaModality.VIDEO
        elif mime.startswith("audio"):
            modality = MediaModality.AUDIO
        elif mime.startswith("image"):
            modality = MediaModality.IMAGE
        else:
            modality = MediaModality.UNKNOWN

        # 初始化字段
        blob = None
        pixels = None
        pair_blob = None
        pair_pixels = None
        pair_extras: List[str] = []
        audio_blob = None

        # 图像处理
        if mime.startswith("image"):
            blob, pixels = encode_image_to_blob(uri, to_rgb=True)

            # Pair 图像处理
            pair_dir = getattr(args, "pair_dir", "")
            if pair_dir and blob:
                pair_uri = Path(pair_dir) / file_path.name
                if pair_uri.exists():
                    pair_blob, pair_pixels = encode_image_to_blob(str(pair_uri), to_rgb=True)
                    # 扫描额外配对图
                    pair_extras = self._scan_pair_extras(uri, pair_dir)

        # 音频处理（小文件直接读）
        elif mime.startswith("audio") and file_size < 20 * 1024 * 1024:
            try:
                with open(uri, "rb") as f:
                    audio_blob = f.read()
            except Exception as e:
                print_exception(self.ctx.console, e, prefix="Failed to read audio", summary_style="yellow")

        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=modality,
            file_size=file_size,
            blob=blob,
            pixels=pixels,
            pair_blob=pair_blob,
            pair_pixels=pair_pixels,
            pair_extras=pair_extras,
            audio_blob=audio_blob,
        )

    def build_cloud_vlm_messages(
        self,
        media: MediaContext,
        prompts: PromptContext,
        *,
        text_first: bool = False,
    ) -> list:
        """构建云端 VLM 通用消息格式

        处理 video / image / 纯文本 三种场景，子类如有特殊逻辑可覆盖。
        """
        if media.mime.startswith("video"):
            with open(media.uri, "rb") as f:
                video_base = _base64.b64encode(f.read()).decode("utf-8")
            video_data_url = f"data:{media.mime};base64,{video_base}"
            return [
                {"role": "system", "content": prompts.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_data_url}},
                        {"type": "text", "text": prompts.user},
                    ],
                },
            ]

        if media.mime.startswith("image"):
            if media.blob is None:
                return []
            pair_dir = getattr(self.ctx.args, "pair_dir", "") if self.ctx else ""
            if pair_dir and not media.pair_blob:
                return []
            return build_vision_messages(
                prompts.system,
                prompts.user,
                media.blob,
                pair_blob=media.pair_blob if pair_dir else None,
                text_first=text_first,
            )

        # 纯文本 fallback
        return [
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ]

    # ------------------------------------------------------------------
    #  通用流式收集 & attempt
    # ------------------------------------------------------------------

    @staticmethod
    def collect_openai_stream(
        responses: Iterable[Any],
        console: Console,
        *,
        style: str = "",
    ) -> str:
        """通用 OpenAI 兼容流式文本收集

        适用于 Ark / GLM / StepFun / Kimi 等使用 ``choices[0].delta.content``
        协议的 provider。

        Args:
            responses: 流式响应迭代器
            console: Rich Console
            style: 打印样式，"dot" 表示只打印进度点，空字符串打印实际文本
        """
        chunks: list[str] = []
        for chunk in responses:
            text_piece = ""
            try:
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and hasattr(chunk.choices[0], "delta")
                    and getattr(chunk.choices[0].delta, "content", None) is not None
                ):
                    text_piece = chunk.choices[0].delta.content
                elif (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and hasattr(chunk.choices[0], "message")
                    and getattr(chunk.choices[0].message, "content", None)
                ):
                    text_piece = chunk.choices[0].message.content  # type: ignore[union-attr]
                else:
                    text_piece = getattr(chunk, "text", "") or ""
            except Exception:
                pass

            if text_piece:
                chunks.append(text_piece)
                if style == "dot":
                    console.print(".", end="", style="blue")
                else:
                    try:
                        console.print(text_piece, end="", overflow="ellipsis")
                    except Exception:
                        console.print(Text(text_piece), end="", overflow="ellipsis")
                    finally:
                        console.file.flush()

        console.print("\n")
        return "".join(chunks)

    @staticmethod
    def attempt_openai_chat(
        *,
        client: Any,
        model_path: str,
        messages: list,
        console: Console,
        progress: Optional[Progress] = None,
        task_id: Optional[Any] = None,
        stream_style: str = "",
        extract_format: str = "srt",
        model_path_replace: Optional[tuple[str, str]] = None,
    ) -> str:
        """通用 OpenAI Chat Completion 流式 attempt

        覆盖 ark / glm / stepfun(cloud) 等的公共流程:
          1. client.chat.completions.create(stream=True)
          2. collect_openai_stream
          3. extract_code_block_content

        Args:
            client: OpenAI / Ark / ZhipuAI 客户端
            model_path: 模型路径
            messages: 消息列表
            console: Rich Console
            progress: 进度条（可选）
            task_id: 进度条任务 ID（可选）
            stream_style: 流式打印样式（"dot" 或 ""）
            extract_format: extract_code_block_content 的格式参数
            model_path_replace: 模型路径字符替换 (old, new)，如 Ark 需要 (".", "-")
        """
        from utils.parse_display import extract_code_block_content

        effective_model = model_path
        if model_path_replace:
            effective_model = model_path.replace(*model_path_replace)

        start_time = time.time()

        completion = client.chat.completions.create(
            model=effective_model,
            messages=messages,
            stream=True,
        )

        if progress and task_id is not None:
            progress.update(task_id, description="Generating captions")
        response_text = CloudVLMProvider.collect_openai_stream(
            completion, console, style=stream_style,
        )

        elapsed = time.time() - start_time
        console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")

        try:
            console.print(response_text)
        except Exception:
            console.print(Text(response_text))

        response_text = response_text.replace(
            "[green]", "<font color='green'>"
        ).replace("[/green]", "</font>")

        content = extract_code_block_content(response_text, extract_format, console)
        if not content:
            raise Exception("RETRY_EMPTY_CONTENT")

        if progress and task_id is not None:
            progress.update(task_id, description="Processing media...")
        return content

    def _scan_pair_extras(self, uri: str, pair_dir: str) -> List[str]:
        """
        扫描额外的配对图 (pair_1.jpg, pair_2.jpg 等)

        返回 base64 编码的图像列表
        """
        from pathlib import Path

        base_dir = Path(pair_dir).resolve()
        stem = Path(uri).stem
        primary_ext = Path(uri).suffix.lower()
        extras_paths = []

        try:
            if not base_dir.exists():
                return []

            for pth in base_dir.iterdir():
                if pth.is_file() and pth.name.startswith(f"{stem}_") and pth.suffix.lower() == primary_ext:
                    name_stem = pth.stem
                    if len(name_stem) > len(stem) + 1 and name_stem[len(stem)] == "_":
                        num_part = name_stem[len(stem) + 1 :]
                        if num_part.isdigit():
                            extras_paths.append((int(num_part), pth))
        except Exception as e:
            print_exception(self.ctx.console, e, prefix="Failed to scan pair extras", summary_style="yellow")
            return []

        # 按数字排序
        extras_paths.sort(key=lambda t: t[0])

        # 编码所有额外图片
        result = []
        for _, pth in extras_paths:
            try:
                blob, _ = encode_image_to_blob(str(pth), to_rgb=True)
                if blob:
                    result.append(blob)
                    self.log(f"Paired extra: {pth.name}", "blue")
            except Exception as e:
                print_exception(self.ctx.console, e, prefix=f"Failed to encode pair extra {pth}", summary_style="yellow")

        return result
