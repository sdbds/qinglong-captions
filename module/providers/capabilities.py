"""
Provider 能力声明

用于声明 Provider 支持的功能和限制
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProviderCapabilities:
    """Provider 能力声明"""

    supports_streaming: bool = False
    supports_structured_output: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_images: bool = False
    supports_documents: bool = False

    # 最大文件大小限制（MB）
    max_file_size_mb: int = 100

    # 支持的 MIME 类型列表（可选，用于精确匹配）
    supported_mimes: Optional[List[str]] = None
