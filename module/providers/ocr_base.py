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

from .backends import OpenAIChatRuntime, find_model_config_section, resolve_runtime_backend
from .base import MediaContext, MediaModality, PromptContext, Provider, ProviderType
from .capabilities import ProviderCapabilities
from .utils import build_vision_messages, encode_image_to_blob
from utils.output_writer import (
    has_meaningful_text_content,
    remove_markdown_output_files,
    write_markdown_output,
)


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

    def resolve_prompts(self, uri: str, mime: str) -> PromptContext:
        """OCR provider 直接使用自己的 prompt 配置，而不是通用 PromptResolver。"""
        system, user = self.get_prompts(mime)
        char_name, char_prompt = self._get_character_prompt(uri)
        return PromptContext(
            system=system,
            user=user,
            character_name=char_name,
            character_prompt=char_prompt,
        )

    def get_runtime_backend(self):
        """Resolve OCR runtime backend."""
        provider_section = self.ctx.config.get(self.name, {})
        runtime_model_name = (
            getattr(self.ctx.args, "openai_model_name", "")
            or provider_section.get("runtime_model_id", "")
            or provider_section.get("model_id", "")
            or self.default_model_id
        )
        model_section = find_model_config_section(
            self.ctx.config,
            runtime_model_name,
            preferred_sections=(self.name,),
        )
        default_temperature = float(
            model_section.get("temperature", provider_section.get("runtime_temperature", provider_section.get("temperature", 0.0)))
        )
        default_max_tokens = int(
            model_section.get(
                "runtime_max_tokens",
                model_section.get(
                    "max_new_tokens",
                    provider_section.get("runtime_max_tokens", provider_section.get("max_new_tokens", 4096)),
                ),
            )
        )
        default_model_id = (
            model_section.get("runtime_model_id", "")
            or model_section.get("model_id", "")
            or provider_section.get("runtime_model_id", "")
            or provider_section.get("model_id", "")
            or self.default_model_id
        )
        return resolve_runtime_backend(
            self.ctx.args,
            provider_section,
            arg_prefix="local_runtime",
            shared_prefix="openai",
            default_model_id=default_model_id,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )

    def attempt_via_openai_backend(self, media: MediaContext, prompts):
        """Run OCR via a local OpenAI-compatible server while keeping OCR side effects."""
        from utils.parse_display import display_markdown
        from utils.stream_util import pdf_to_images_high_quality

        runtime = self.get_runtime_backend()
        backend = OpenAIChatRuntime(runtime)
        output_dir = media.extras.get("output_dir")

        def infer_from_blob(blob: str) -> str:
            messages = build_vision_messages(
                prompts.system,
                prompts.user,
                blob,
                text_first=False,
            )
            return backend.complete(messages)

        if media.mime.startswith("application/pdf"):
            images = pdf_to_images_high_quality(media.uri)
            all_contents = []
            for idx, pil_img in enumerate(images):
                page_dir = Path(output_dir) / f"page_{idx + 1:04d}"
                page_dir.mkdir(parents=True, exist_ok=True)
                page_img_path = page_dir / f"page_{idx + 1:04d}.png"
                try:
                    pil_img.save(page_img_path)
                except Exception:
                    try:
                        pil_img.convert("RGB").save(page_img_path)
                    except Exception:
                        continue

                page_blob, _ = encode_image_to_blob(str(page_img_path), to_rgb=True)
                if not page_blob:
                    continue
                page_content = infer_from_blob(page_blob)
                try:
                    write_markdown_output(page_dir, page_content)
                except Exception:
                    pass
                all_contents.append(page_content.strip())

            content = "\n<--- Page Split --->\n".join(all_contents)
        else:
            if not media.blob:
                content = ""
            else:
                content = infer_from_blob(media.blob)

        try:
            if output_dir:
                write_markdown_output(Path(output_dir), content)
        except Exception:
            pass

        try:
            display_markdown(
                title=Path(media.uri).name,
                markdown_content=content,
                pixels=media.pixels,
                panel_height=32,
                console=self.ctx.console,
            )
        except Exception:
            pass

        from .base import CaptionResult

        return CaptionResult(
            raw=content,
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "runtime_backend": runtime.mode,
                "runtime_model_id": runtime.model_id,
            },
        )

    def post_validate(self, result, media: MediaContext, args: Any):
        result = super().post_validate(result, media, args)
        output_dir = media.extras.get("output_dir") or result.metadata.get("output_dir")
        if output_dir and not has_meaningful_text_content(result.description):
            remove_markdown_output_files(Path(output_dir))
        return result

    def get_retry_config(self):
        """
        OCR 使用简单的重试配置

        所有错误都重试，使用 base_wait
        """
        cfg = super().get_retry_config()
        cfg.classify_error = lambda e: cfg.base_wait
        return cfg

    @staticmethod
    def build_ocr_messages(
        image_path: str,
        prompt_text: str,
        *,
        system_prompt: str = "",
        image_uri_prefix: str = "",
    ) -> list:
        """构建 OCR 推理的聊天消息

        Args:
            image_path: 图像文件路径
            prompt_text: 提示文本
            system_prompt: 系统提示（可选，nanonets 等需要）
            image_uri_prefix: 图像路径前缀（如 "file://"）
        """
        image_ref = f"{image_uri_prefix}{image_path}" if image_uri_prefix else image_path
        messages: list = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": prompt_text},
            ],
        })
        return messages

    def _get_model_config(self, key: str, default: Any = None) -> Any:
        """
        从 config.{provider_name} 读取配置

        例如：deepseek_ocr.model_id
        """
        provider_section = self.ctx.config.get(self.name, {})
        return provider_section.get(key, default)
