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
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import Progress


class MediaModality(Enum):
    """媒体模态类型"""

    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()
    DOCUMENT = auto()
    UNKNOWN = auto()


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


@dataclass
class RetryConfig:
    """重试配置"""

    max_retries: int = 10
    base_wait: float = 1.0

    # 错误分类函数：返回等待时间，None 表示不重试
    classify_error: Optional[Callable[[Exception], Optional[float]]] = None

    # 重试耗尽回调
    on_exhausted: Optional[Callable[[Exception], str]] = None


@dataclass
class CaptionResult:
    """
    统一的标注结果类型

    修复返回值多态问题（原代码返回 str | dict）
    """

    raw: str
    parsed: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def description(self) -> str:
        """获取描述文本（兼容 captioner.py 的处理）"""
        if self.parsed:
            return (
                self.parsed.get("long_description")
                or self.parsed.get("description")
                or self.parsed.get("short_description")
                or self.raw
            )
        return self.raw

    @property
    def is_structured(self) -> bool:
        """是否是结构化输出"""
        return self.parsed is not None

    def get(self, key: str, default: Any = None) -> Any:
        """便捷获取 parsed 中的字段"""
        if self.parsed:
            return self.parsed.get(key, default)
        return default


@dataclass
class ProviderContext:
    """Provider 运行时上下文"""

    console: Console
    progress: Optional[Progress] = None
    task_id: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    args: Any = None


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

    def post_validate(self, result: CaptionResult, media: MediaContext, args: Any) -> CaptionResult:
        """
        后验证钩子

        子类可以覆盖此方法实现自定义验证/重试逻辑
        （如 Pixtral 的角色名校验）
        """
        return result

    def get_retry_config(self) -> RetryConfig:
        """获取重试配置，子类可覆盖"""
        args = self.ctx.args
        return RetryConfig(
            max_retries=getattr(args, "max_retries", 10),
            base_wait=getattr(args, "wait_time", 1.0),
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
        from .resolver import PromptResolver

        # 准备媒体
        media = self.prepare_media(uri, mime, self.ctx.args)

        # 获取 prompts
        resolver = PromptResolver(self.ctx.config, self.name)
        char_name, char_prompt = self._get_character_prompt(uri)
        prompts = resolver.resolve(mime, self.ctx.args, char_prompt)

        # 执行（带重试）
        retry_cfg = self.get_retry_config()

        def _attempt_wrapper() -> CaptionResult:
            return self.attempt(media, prompts)

        result = with_retry_impl(_attempt_wrapper, retry_cfg, self.ctx.console)

        # 后验证
        result = self.post_validate(result, media, self.ctx.args)

        return result

    def _get_character_prompt(self, uri: str) -> Tuple[str, str]:
        """获取角色 prompt"""
        args = self.ctx.args
        if not getattr(args, "dir_name", False):
            return "", ""

        from pathlib import Path
        from utils.stream_util import split_name_series

        dir_name = Path(uri).parent.name or ""
        char_name = split_name_series(dir_name)
        if char_name:
            return char_name, f"If there is a person/character or more in the image you must refer to them as {char_name}.\n"
        return "", ""

    def log(self, msg: str, style: str = ""):
        """便捷日志方法"""
        if style:
            self.ctx.console.print(f"[{style}]{msg}[/{style}]")
        else:
            self.ctx.console.print(msg)


class ProviderType(Enum):
    """Provider 类型分类"""

    CLOUD_VLM = auto()
    LOCAL_VLM = auto()
    OCR = auto()
    VISION_API = auto()
