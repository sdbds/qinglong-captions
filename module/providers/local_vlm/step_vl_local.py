"""Step VL Local Provider"""

from pathlib import Path

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.local_vlm_base import LocalVLMProvider
from providers.registry import register_provider


@register_provider("step_vl_local")
class StepVLLocalProvider(LocalVLMProvider):
    """Step VL Local Provider"""

    default_model_id = "stepfun-ai/Step1.5V-mini"
    _attn_implementation = "eager"

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, "vlm_image_model", "") == "step_vl_local" and mime.startswith("image")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from module.providers.cloud_vlm.stepfun import attempt_stepfun

        # 注意：step_vl_local 使用 stepfun_provider 的 attempt_stepfun
        # 但传入的参数不同

        pair_dir = getattr(self.ctx.args, "pair_dir", "")
        has_pair = bool(pair_dir and media.pair_blob)

        result = attempt_stepfun(
            client=None,  # 本地模型
            model_path="",  # 本地模型
            mime=media.mime,
            system_prompt=prompts.system,
            prompt=prompts.user,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            uri=media.uri,
            image_blob=media.blob,
            image_pixels=media.pixels,
            has_pair=has_pair,
            pair_blob=media.pair_blob if has_pair else None,
            pair_pixels=media.pair_pixels if has_pair else None,
            video_file_id=None,
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
