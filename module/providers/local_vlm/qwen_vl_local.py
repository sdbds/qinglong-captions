"""Qwen VL Local Provider"""

import tempfile
from pathlib import Path

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.local_vlm_base import LocalVLMProvider
from module.providers.registry import register_provider

_TORCHVISION_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_TRANSCODE_IMAGE_MIMES = {"image/avif", "image/heic", "image/heif"}


def _needs_jpeg_input(path: Path, mime: str = "") -> bool:
    if mime.lower() in _TRANSCODE_IMAGE_MIMES:
        return True
    suffix = path.suffix.lower()
    return bool(suffix) and suffix not in _TORCHVISION_IMAGE_SUFFIXES


def _convert_to_temp_jpeg(path: Path, *, quality: int) -> Path:
    from PIL import Image

    temp_path: Path | None = None
    try:
        with Image.open(path) as image:
            image.load()
            if "xmp" in image.info:
                del image.info["xmp"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            with tempfile.NamedTemporaryFile(prefix=f"{path.stem}_", suffix=".jpg", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            image.save(temp_path, format="JPEG", quality=quality)
            return temp_path
    except Exception as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError(f"Failed to convert image to JPEG before Qwen input: {path}") from exc


def _delete_temp_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@register_provider("qwen_vl_local")
class QwenVLLocalProvider(LocalVLMProvider):
    """Qwen VL Local Provider"""

    default_model_id = "Qwen/Qwen3-VL-8B-Instruct"
    _attn_implementation = "eager"

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, "vlm_image_model", "") == "qwen_vl_local" and mime.startswith("image")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            return self.attempt_via_openai_backend(media, prompts)

        from module.providers.cloud_vlm.qwenvl import attempt_qwenvl

        temp_paths: list[Path] = []

        def image_file_ref(uri: str, mime: str = "") -> str:
            image_path = Path(uri).resolve()
            if _needs_jpeg_input(image_path, mime):
                image_path = _convert_to_temp_jpeg(image_path, quality=self.get_image_quality())
                temp_paths.append(image_path)
                self.log(f"Converted image input to JPEG for Qwen: {Path(uri).name}", "yellow")
            return f"file://{image_path.as_posix()}"

        file = image_file_ref(media.uri, media.mime)

        content_items = []

        if media.extras.get("pair_uri"):
            pair_file = image_file_ref(media.extras["pair_uri"])
            self.log(f"Pair image: {pair_file}", "yellow")
            content_items.extend([{"image": file}, {"image": pair_file}])
        else:
            content_items.append({"image": file})

        content_items.append({"text": prompts.user})

        messages = [
            {"role": "system", "content": [{"text": prompts.system}]},
            {"role": "user", "content": content_items},
        ]

        try:
            result = attempt_qwenvl(
                model_path=self.model_id,
                api_key="",  # 本地模型不需要 API key
                messages=messages,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                local_config=self.model_config,
            )
        finally:
            _delete_temp_files(temp_paths)

        return CaptionResult(raw=result, metadata={"provider": self.name})

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "429" in msg:
                return 59.0
            if "502" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        return cfg
