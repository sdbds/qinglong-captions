"""
Provider 架构包

重构后的模块化 Provider 系统
"""

# 核心抽象
from .base import (
    CaptionResult,
    MediaContext,
    MediaModality,
    PromptContext,
    Provider,
    ProviderContext,
    ProviderType,
    RetryConfig,
)

# 能力声明
from .capabilities import ProviderCapabilities

# 注册表
from .registry import (
    ProviderRegistry,
    get_registry,
    register_provider,
)

# Prompt 解析
from .resolver import PromptResolver, PromptTemplate

# 分类基类
from .cloud_vlm_base import CloudVLMProvider
from .local_vlm_base import LocalVLMProvider
from .ocr_base import OCRProvider
from .vision_api_base import VisionAPIProvider, StructuredOutputConfig

# 工具函数
from .utils import (
    build_vision_messages,
    encode_image_cached,
    encode_image_to_blob,
    with_retry_impl,
)

__all__ = [
    # 核心抽象
    "CaptionResult",
    "MediaContext",
    "MediaModality",
    "PromptContext",
    "Provider",
    "ProviderContext",
    "ProviderType",
    "RetryConfig",
    # 能力声明
    "ProviderCapabilities",
    # 注册表
    "ProviderRegistry",
    "get_registry",
    "register_provider",
    # Prompt 解析
    "PromptResolver",
    "PromptTemplate",
    # 分类基类
    "CloudVLMProvider",
    "LocalVLMProvider",
    "OCRProvider",
    "VisionAPIProvider",
    "StructuredOutputConfig",
    # 工具函数
    "build_vision_messages",
    "encode_image_cached",
    "encode_image_to_blob",
    "with_retry_impl",
]
