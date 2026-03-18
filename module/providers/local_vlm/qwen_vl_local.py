"""Qwen VL Local Provider"""

from pathlib import Path

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.local_vlm_base import LocalVLMProvider
from module.providers.registry import register_provider


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

        file = f"file://{Path(media.uri).resolve().as_posix()}"

        content_items = []

        if media.extras.get("pair_uri"):
            pair_file = f"file://{Path(media.extras['pair_uri']).as_posix()}"
            self.log(f"Pair image: {pair_file}", "yellow")
            content_items.extend([{"image": file}, {"image": pair_file}])
        else:
            content_items.append({"image": file})

        content_items.append({"text": prompts.user})

        messages = [
            {"role": "system", "content": [{"text": prompts.system}]},
            {"role": "user", "content": content_items},
        ]

        result = attempt_qwenvl(
            model_path=self.model_id,
            api_key="",  # 本地模型不需要 API key
            messages=messages,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            local_config=self.model_config,
        )

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
