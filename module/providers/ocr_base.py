"""
OCRProvider - OCR Provider 基类

适用于：DeepSeekOCR, HunyuanOCR, GLMOCR, ChandraOCR,
       OLMOCR, PaddleOCR, NanonetsOCR, FireRedOCR

特性：
- 统一的输出目录处理
- 统一的 prompt 获取
- 简化的重试配置
"""

from pathlib import Path
from typing import Any, ClassVar, Optional

from .base import MediaContext, MediaModality, Provider, ProviderType
from .capabilities import ProviderCapabilities
from .utils import encode_image_to_blob


class OCRProvider(Provider):
    """
    OCR Provider 基类

    消除 8 个 OCR provider 的重复代码
    """

    provider_type = ProviderType.OCR
    capabilities = ProviderCapabilities(
        supports_documents=True,
        supports_images=True,
    )

    # 子类必须定义的类属性
    default_model_id: ClassVar[str] = ""
    default_prompt: ClassVar[str] = ""

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        """
        OCR 通用的 can_handle 逻辑

        - ocr_model 参数匹配类名
        - PDF 总是处理
        - 图像需要 document_image=True
        """
        ocr_model = getattr(args, "ocr_model", "")
        if ocr_model != cls.name:
            return False

        # PDF 总是处理
        if mime.startswith("application"):
            return True

        # 图像需要 document_image=True
        if mime.startswith("image") and getattr(args, "document_image", False):
            return True

        return False

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        """
        OCR 通用的媒体准备

        - 图像：编码 base64
        - 准备输出目录
        """
        blob = None
        pixels = None

        # 编码图像（如果是图像）
        if mime.startswith("image"):
            blob, pixels = encode_image_to_blob(uri, to_rgb=True)

        # 准备输出目录
        output_dir = Path(uri).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=MediaModality.DOCUMENT if mime.startswith("application") else MediaModality.IMAGE,
            blob=blob,
            pixels=pixels,
            extras={"output_dir": output_dir},
        )

    def get_prompts(self, mime: str):
        """
        OCR 使用默认 prompt

        从 config.prompts.{provider_name}_prompt 读取
        或使用类属性 default_prompt
        """
        prompts = self.ctx.config.get("prompts", {})
        prompt_key = f"{self.name}_prompt"
        prompt = prompts.get(prompt_key, self.default_prompt)

        return "", prompt  # OCR 通常不需要 system prompt

    def get_retry_config(self):
        """
        OCR 使用简单的重试配置

        所有错误都重试，使用 base_wait
        """
        cfg = super().get_retry_config()
        cfg.classify_error = lambda e: cfg.base_wait
        return cfg

    def _get_model_config(self, key: str, default: Any = None) -> Any:
        """
        从 config.{provider_name} 读取配置

        例如：deepseek_ocr.model_id
        """
        provider_section = self.ctx.config.get(self.name, {})
        return provider_section.get(key, default)
