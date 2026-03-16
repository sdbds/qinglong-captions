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
from .catalog import (
    canonicalize_provider_name,
    canonicalize_route_value,
    normalize_runtime_args,
    provider_aliases,
    provider_config_sections,
    provider_prompt_prefixes,
    route_choices,
    route_matches_provider,
    route_provider_name,
)
from .registry import (
    ProviderRegistry,
    get_registry,
    register_provider,
)

# Prompt 解析
from .resolver import PromptResolver, PromptTemplate

# 分类基类
from .cloud_vlm_base import CloudVLMProvider
from .local_llm_base import LocalLLMProvider
from .local_alm_base import LocalALMProvider
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
    "canonicalize_provider_name",
    "canonicalize_route_value",
    "normalize_runtime_args",
    "provider_aliases",
    "provider_config_sections",
    "provider_prompt_prefixes",
    "route_choices",
    "route_matches_provider",
    "route_provider_name",
    # 注册表
    "ProviderRegistry",
    "get_registry",
    "register_provider",
    # Prompt 解析
    "PromptResolver",
    "PromptTemplate",
    # 分类基类
    "CloudVLMProvider",
    "LocalLLMProvider",
    "LocalALMProvider",
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
