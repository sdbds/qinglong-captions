"""Pixtral Provider"""

from pathlib import Path

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.registry import register_provider
from providers.utils import build_vision_messages
from providers.vision_api_base import VisionAPIProvider


@register_provider("pixtral")
class PixtralProvider(VisionAPIProvider):
    """Pixtral Provider"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        # pixtral_ocr 模式：用户选择 ocr_model=pixtral_ocr 时走 pixtral
        if getattr(args, "ocr_model", "") == "pixtral_ocr":
            if mime.startswith("application"):
                return True
            if mime.startswith("image") and getattr(args, "document_image", False):
                return True
        return getattr(args, "pixtral_api_key", "") != "" and (mime.startswith("image") or mime.startswith("application"))

    def prepare_media(self, uri: str, mime: str, args) -> MediaContext:
        """Pixtral 支持 PDF，需要特殊处理"""
        media = super().prepare_media(uri, mime, args)

        # PDF 处理
        if mime.startswith("application"):
            from mistralai import Mistral

            client = Mistral(api_key=self.ctx.args.pixtral_api_key)

            # 上传 PDF
            for upload_attempt in range(getattr(args, "max_retries", 10)):
                try:
                    from utils.stream_util import sanitize_filename

                    with open(uri, "rb") as pdf_f:
                        uploaded_pdf = client.files.upload(
                            file={
                                "file_name": f"{sanitize_filename(uri)}.pdf",
                                "content": pdf_f,
                            },
                            purpose="ocr",
                        )
                    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
                    media.extras["signed_url"] = signed_url.url
                    break
                except Exception as e:
                    self.log(f"Error uploading PDF: {e}", "red")
                    if upload_attempt < args.max_retries - 1:
                        self.log(f"Retrying in {args.wait_time}s...", "yellow")
                        import time

                        time.sleep(args.wait_time)
                    else:
                        self.log(f"Failed to upload PDF after {args.max_retries} attempts", "red")

        return media

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from mistralai import Mistral
        from module.providers.pixtral_provider import attempt_pixtral
        from rich.text import Text
        from utils.stream_util import sanitize_filename

        client = Mistral(api_key=self.ctx.args.pixtral_api_key)

        # 检查是否是 OCR 模式
        ocr_mode = getattr(self.ctx.args, "ocr_model", "") == "pixtral"

        # 准备 captions（用于 tag highlight）
        captions = []
        if media.mime.startswith("image"):
            captions_path = Path(media.uri).with_suffix(".txt")
            if captions_path.exists():
                with open(captions_path, "r", encoding="utf-8") as f:
                    captions = [line.strip() for line in f.readlines()]

        # 构建 messages
        if media.mime.startswith("application"):
            # PDF 模式
            signed_url = media.extras.get("signed_url")
            if not signed_url:
                return CaptionResult(raw="", metadata={"error": "PDF upload failed"})

            result = attempt_pixtral(
                client=client,
                model_path=self.ctx.args.pixtral_model_path,
                mime=media.mime,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                uri=media.uri,
                document_image=self.ctx.args.document_image,
                signed_url_url=signed_url,
            )
        elif ocr_mode:
            # OCR 模式
            result = attempt_pixtral(
                client=client,
                model_path=self.ctx.args.pixtral_model_path,
                mime=media.mime,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                uri=media.uri,
                ocr=True,
                base64_image=media.blob,
                pixels=media.pixels,
            )
        else:
            # 标准图像模式
            character_name = ""
            if prompts.character_name:
                character_name = f"{prompts.character_name}, "

            # 读取 captions[0] 或 config prompt
            config_prompt = self.ctx.config.get("prompts", {}).get("pixtral_image_prompt", "")
            prompt_text = Text(
                f"<s>[INST]{prompts.character_prompt}{character_name}{captions[0] if captions else config_prompt}\n[IMG][/INST]"
            ).plain

            messages = build_vision_messages(prompts.system, prompt_text, media.blob, text_first=True)

            result = attempt_pixtral(
                client=client,
                model_path=self.ctx.args.pixtral_model_path,
                mime=media.mime,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                uri=media.uri,
                messages=messages,
                pixels=media.pixels,
                captions=captions,
                prompt_text=prompt_text,
                character_name=character_name,
                tags_highlightrate=getattr(self.ctx.args, "tags_highlightrate", 0.0),
            )

        return CaptionResult(raw=result, metadata={"provider": self.name})

    def post_validate(self, result: CaptionResult, media: MediaContext, args) -> CaptionResult:
        """Pixtral 特殊的后验证：角色名校验"""
        description = result.description

        # 角色名校验（如果设置了）
        if result.parsed and "character_name" in result.parsed:
            expected_char = result.parsed["character_name"]
            if expected_char and expected_char not in description:
                # 需要重试 - 抛出特定异常让 retry 机制处理
                raise Exception(f"RETRY_CHARACTER_NOT_FOUND: Character '{expected_char}' not found")

        return result

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "429" in msg:
                return 59.0
            if "502" in msg or "RETRY_PIXTRAL_" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        return cfg
