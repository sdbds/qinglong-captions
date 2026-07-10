"""
Provider 抽象基类

定义 Provider 架构的核心抽象：
- Provider: 所有 Provider 的基类
- CaptionResult: 统一的返回结果类型
- MediaContext: 媒体上下文
- PromptContext: Prompt 上下文
- MediaModality: 媒体模态枚举
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum, auto
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import Progress

from .policies import SegmentationPolicy
from utils.parse_display import display_caption_and_rate


class MediaModality(Enum):
    """媒体模态类型"""

    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()
    DOCUMENT = auto()
    UNKNOWN = auto()


class CaptionStatus(str, Enum):
    """Persistence-relevant outcome of a caption attempt."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class MediaContext:
    """
    媒体上下文 - 不可变

    支持多模态：图像、视频、音频、文档
    """

    uri: str
    mime: str
    sha256hash: str
    modality: MediaModality

    # 通用元数据
    file_size: int = 0
    duration_ms: int = 0

    # 图像相关
    blob: Optional[str] = None
    pixels: Optional[Any] = None
    pair_blob: Optional[str] = None
    pair_pixels: Optional[Any] = None
    pair_extras: List[str] = field(default_factory=list)

    # 音频/视频相关
    audio_blob: Optional[bytes] = None
    video_file_refs: List[Any] = field(default_factory=list)

    # 扩展字段（预留）
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_large_file(self) -> bool:
        """是否是大文件（>=20MB，需要上传）"""
        return self.file_size >= 20 * 1024 * 1024


@dataclass(frozen=True)
class PromptContext:
    """Prompt 上下文 - 不可变"""

    system: str
    user: str
    character_name: str = ""
    character_prompt: str = ""

    def with_character(self, name: str, prompt: str) -> "PromptContext":
        """创建带有角色 prompt 的新上下文"""
        return PromptContext(
            system=self.system, user=prompt + self.user if prompt else self.user, character_name=name, character_prompt=prompt
        )


def build_chat_text_message(role: str, text: str) -> Dict[str, Any]:
    """Build a text-only chat message compatible with multimodal chat templates."""
    return Provider.build_message(role, [Provider.build_text_part(text)])


@dataclass
class RetryConfig:
    """重试配置"""

    max_retries: int = 10
    base_wait: float = 1.0

    # 错误分类函数：返回等待时间，None 表示不重试
    classify_error: Optional[Callable[[Exception], Optional[float]]] = None

    # 重试耗尽回调
    on_exhausted: Optional[Callable[[Exception], str]] = None

    # 等待时间抖动比例，0 表示固定等待
    jitter_ratio: float = 0.2


@dataclass
class CaptionResult:
    """
    统一的标注结果类型

    修复返回值多态问题（原代码返回 str | dict）
    """

    raw: str
    parsed: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: CaptionStatus = CaptionStatus.SUCCESS
    error: Optional[str] = None

    @classmethod
    def success(
        cls,
        raw: str = "",
        *,
        parsed: Optional[Dict] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CaptionResult":
        return cls(raw=raw, parsed=parsed, metadata=dict(metadata or {}), status=CaptionStatus.SUCCESS)

    @classmethod
    def skipped(
        cls,
        reason: str,
        *,
        raw: str = "",
        parsed: Optional[Dict] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CaptionResult":
        result_metadata = dict(metadata or {})
        result_metadata.setdefault("skip_reason", reason)
        return cls(raw=raw, parsed=parsed, metadata=result_metadata, status=CaptionStatus.SKIPPED)

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        raw: str = "",
        parsed: Optional[Dict] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CaptionResult":
        return cls(
            raw=raw,
            parsed=parsed,
            metadata=dict(metadata or {}),
            status=CaptionStatus.FAILED,
            error=error,
        )

    @property
    def payload(self) -> Dict[str, Any] | str:
        """Canonical payload exposed to legacy boundaries."""
        return self.parsed if self.parsed is not None else self.raw

    @property
    def description(self) -> str:
        """获取描述文本（兼容 captioner.py 的处理）"""
        if self.parsed:
            task_kind = str(self.parsed.get("task_kind") or "").strip().lower()
            subtitle_format = str(self.parsed.get("subtitle_format") or "").strip().lower()
            extension = self.caption_extension
            if task_kind == "ast" or subtitle_format == "srt" or extension == ".srt":
                for key in ("translation_srt", "transcript", "description", "long_description", "short_description"):
                    value = self.parsed.get(key)
                    if str(value or "").strip():
                        return str(value)
            return (
                self.parsed.get("long_description")
                or self.parsed.get("transcript")
                or self.parsed.get("translation_srt")
                or self.parsed.get("description")
                or self.parsed.get("short_description")
                or self.parsed.get("markdown")
                or self.parsed.get("text")
                or self.raw
            )
        return self.raw

    @property
    def text(self) -> str:
        """Canonical text representation for sidecar output."""
        return self.description

    @property
    def caption_extension(self) -> Optional[str]:
        """Optional sidecar extension requested by structured payloads."""
        extension = self.get("caption_extension")
        if extension in (None, ""):
            return None
        value = str(extension).strip()
        if not value:
            return None
        if not value.startswith("."):
            value = f".{value}"
        return value

    @property
    def is_structured(self) -> bool:
        """是否是结构化输出"""
        return self.parsed is not None

    @property
    def has_content(self) -> bool:
        """Whether the result carries semantic caption content."""
        if self.parsed is None:
            return bool(str(self.raw).strip())
        content_fields = (
            "description",
            "long_description",
            "short_description",
            "markdown",
            "text",
            "transcript",
            "translation_srt",
        )
        return any(str(self.parsed.get(name) or "").strip() for name in content_fields)

    @property
    def is_persistable(self) -> bool:
        """Whether this result may replace durable caption data."""
        return self.status is CaptionStatus.SUCCESS and self.has_content

    def to_dataset_caption(self) -> str:
        """Serialize this result for the Lance captions column."""
        if self.parsed is not None:
            return json.dumps(self.parsed, ensure_ascii=False)
        return self.raw

    def get(self, key: str, default: Any = None) -> Any:
        """便捷获取 parsed 中的字段"""
        if self.parsed:
            return self.parsed.get(key, default)
        return default

    def __bool__(self) -> bool:
        return self.is_persistable


@dataclass
class ProviderContext:
    """Provider 运行时上下文"""

    console: Console
    progress: Optional[Progress] = None
    task_id: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    args: Any = None


def _structured_image_display_payload(media: MediaContext, result: CaptionResult) -> Optional[Dict[str, Any]]:
    """Extract a safe shared display payload for structured single-image results."""
    if media.modality != MediaModality.IMAGE:
        return None
    if media.pixels is None or media.pair_pixels is not None:
        return None
    if not bool(result.metadata.get("structured")):
        return None
    if not isinstance(result.parsed, dict):
        return None

    payload = result.parsed
    description = str(payload.get("description") or payload.get("long_description") or "").strip()
    if not description:
        return None

    scores = payload.get("scores")
    if not isinstance(scores, dict):
        scores = {}

    try:
        average_score = float(payload.get("average_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        average_score = 0.0

    return {
        "description": description,
        "scores": scores,
        "average_score": average_score,
    }


def _maybe_display_structured_image_result(console: Console, media: MediaContext, result: CaptionResult) -> None:
    payload = _structured_image_display_payload(media, result)
    if payload is None:
        return

    display_caption_and_rate(
        title=Path(media.uri).name,
        tag_description="",
        long_description=payload["description"],
        pixels=media.pixels,
        rating=payload["scores"],
        average_score=payload["average_score"],
        panel_height=32,
        console=console,
    )


class Provider(ABC):
    """
    Provider 抽象基类

    所有 Provider 必须继承此类并实现抽象方法。
    """

    # 类级名称（由 @register_provider 装饰器自动设置）
    name: str = ""

    def __init__(self, context: ProviderContext):
        self.ctx = context
        self._prompt_resolver: Optional[Any] = None

    @staticmethod
    def build_message(role: str, content: Any) -> Dict[str, Any]:
        """Build a chat message payload for provider-specific runtimes."""
        return {"role": role, "content": content}

    @staticmethod
    def build_text_part(text: str) -> Dict[str, str]:
        """Build a text content block for multimodal chat messages."""
        return {"type": "text", "text": text}

    @staticmethod
    def build_image_part(value: Optional[str] = None, *, field_name: str = "image") -> Dict[str, Any]:
        """Build an image content block, optionally omitting the reference for placeholder-only formats."""
        part: Dict[str, Any] = {"type": "image"}
        if value is not None:
            part[field_name] = value
        return part

    @staticmethod
    def build_video_part(value: Optional[str] = None, *, field_name: str = "video") -> Dict[str, Any]:
        """Build a video content block, optionally omitting the reference for placeholder-only formats."""
        part: Dict[str, Any] = {"type": "video"}
        if value is not None:
            part[field_name] = value
        return part

    @staticmethod
    def build_audio_part(value: Optional[str] = None, *, field_name: str = "path") -> Dict[str, Any]:
        """Build an audio content block with provider-specific reference field selection."""
        part: Dict[str, Any] = {"type": "audio"}
        if value is not None:
            part[field_name] = value
        return part

    @classmethod
    @abstractmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        """
        类方法检查是否能处理

        使用 @classmethod 而非 @staticmethod，便于子类访问类属性
        """
        pass

    @abstractmethod
    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        """准备媒体上下文"""
        pass

    @abstractmethod
    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        """
        执行 API 调用

        返回 CaptionResult 而非 str，解决返回值多态问题
        """
        pass

    def get_response_skip_reason(self, result: CaptionResult, media: MediaContext, args: Any) -> str:
        """Return a skip reason for provider-specific non-caption responses."""
        return ""

    @classmethod
    def segmentation_policy(cls, args: Any, mime: str, config: Any) -> SegmentationPolicy:
        """Return provider-specific media segmentation policy."""
        from .catalog import provider_segmentation_policy

        return provider_segmentation_policy(cls.name, args, mime, config)

    @classmethod
    def prompt_fallback_keys(cls, mime: str, field: str) -> Tuple[str, ...]:
        """Return provider-specific prompt fallback keys for a mime/field pair."""
        from .catalog import provider_prompt_fallback_keys

        return provider_prompt_fallback_keys(cls.name, mime, field)

    def post_validate(self, result: CaptionResult, media: MediaContext, args: Any) -> CaptionResult:
        """
        后验证钩子

        子类可以覆盖此方法实现自定义验证/重试逻辑
        （如 Pixtral 的角色名校验）
        """
        skip_reason = self.get_response_skip_reason(result, media, args)
        if skip_reason:
            metadata = dict(result.metadata)
            metadata.setdefault("provider", self.name)
            metadata["skip_reason"] = skip_reason
            try:
                self.log(f"Skipping {Path(media.uri).name}: {skip_reason}", "yellow")
            except Exception:
                pass
            return CaptionResult.skipped(skip_reason, metadata=metadata)
        return result

    def get_retry_config(self) -> RetryConfig:
        """获取重试配置，子类可覆盖"""
        args = self.ctx.args
        return RetryConfig(
            max_retries=getattr(args, "max_retries", 10),
            base_wait=getattr(args, "wait_time", 1.0),
        )

    def get_image_quality(self) -> int:
        """Resolve shared JPEG quality for API-bound image encoding."""
        from .utils import resolve_image_quality

        return resolve_image_quality(self.ctx.config, self.ctx.args)

    def resolve_prompts(self, uri: str, mime: str, media: Optional[MediaContext] = None) -> PromptContext:
        """解析 provider 执行所需的 prompt。"""
        from .directory_name_context import resolve_directory_name_context
        from .resolver import PromptResolver

        resolver = PromptResolver(self.ctx.config, self.name, provider_class=self.__class__)
        directory_context = resolve_directory_name_context(
            args=self.ctx.args,
            uri=uri,
            mime=mime,
            provider_name=self.name,
            media=media,
        )
        return resolver.resolve(
            mime,
            self.ctx.args,
            character_prompt=directory_context.character_prompt,
            character_name=directory_context.character_name,
            media=media,
        )

    def execute(self, uri: str, mime: str, sha256hash: str) -> CaptionResult:
        """
        完整的执行流程

        1. 准备媒体
        2. 获取 prompts（使用 PromptResolver）
        3. 执行 attempt（带重试）
        4. 后验证
        """
        from .utils import with_retry_impl

        # 准备媒体
        media = self.prepare_media(uri, mime, self.ctx.args)
        if media is None:
            raise RuntimeError(f"{self.name or self.__class__.__name__}.prepare_media() returned None")
        if media.sha256hash != sha256hash:
            media = replace(media, sha256hash=sha256hash)

        # 获取 prompts
        task_contract = getattr(self, "task_contract", None)
        if task_contract is not None and getattr(task_contract, "consumes_prompts", True) is False:
            prompts = PromptContext(system="", user="")
        else:
            prompts = self.resolve_prompts(uri, mime, media=media)

        # 执行（带重试）
        retry_cfg = self.get_retry_config()

        def _attempt_wrapper() -> CaptionResult:
            return self.attempt(media, prompts)

        result = with_retry_impl(_attempt_wrapper, retry_cfg, self.ctx.console)

        # 后验证
        result = self.post_validate(result, media, self.ctx.args)
        _maybe_display_structured_image_result(self.ctx.console, media, result)

        return result

    def _get_character_prompt(self, uri: str, mime: str = "", media: Optional[MediaContext] = None) -> Tuple[str, str]:
        """Compatibility wrapper for legacy callers; new code should use directory_name_context."""
        from .directory_name_context import resolve_directory_name_context

        context = resolve_directory_name_context(
            args=self.ctx.args,
            uri=uri,
            mime=mime,
            provider_name=self.name,
            media=media,
        )
        return context.character_name, context.character_prompt

    def log(self, msg: str, style: str = ""):
        """便捷日志方法"""
        if style:
            self.ctx.console.print(f"[{style}]{msg}[/{style}]")
        else:
            self.ctx.console.print(msg)

    def display_name(self, mime: str) -> str:
        """用户可见的 provider 名称（用于日志和结果元数据）。"""
        return self.name


class ProviderType(Enum):
    """Provider 类型分类"""

    CLOUD_VLM = auto()
    LOCAL_VLM = auto()
    LOCAL_ALM = auto()
    OCR = auto()
    VISION_API = auto()
