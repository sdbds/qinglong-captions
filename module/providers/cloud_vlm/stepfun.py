"""StepFun Provider"""

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider
from providers.utils import build_vision_messages


@register_provider("stepfun")
class StepfunProvider(CloudVLMProvider):
    """StepFun API Provider"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, "step_api_key", "") != ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from openai import OpenAI
        from module.providers.stepfun_provider import attempt_stepfun

        client = OpenAI(api_key=self.ctx.args.step_api_key, base_url="https://api.stepfun.com/v1")

        # 构建 messages
        messages = self._build_messages(media, prompts)

        result = attempt_stepfun(
            client=client,
            model_path=self.ctx.args.step_model_path,
            messages=messages,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            uri=media.uri,
            image_pixels=media.pixels,
            pair_pixels=media.pair_pixels,
        )

        return CaptionResult(raw=result, metadata={"provider": self.name})

    def _build_messages(self, media: MediaContext, prompts: PromptContext):
        """构建 StepFun 消息格式"""
        if media.mime.startswith("video"):
            # 视频需要上传
            from openai import OpenAI

            client = OpenAI(api_key=self.ctx.args.step_api_key, base_url="https://api.stepfun.com/v1")

            with open(media.uri, "rb") as f:
                file = client.files.create(file=f, purpose="storage")

            self.log(f"Uploaded video: {file.id}", "blue")

            return [
                {"role": "system", "content": prompts.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": f"stepfile://{file.id}"}},
                        {"type": "text", "text": prompts.user},
                    ],
                },
            ]

        elif media.mime.startswith("image"):
            return build_vision_messages(
                prompts.system, prompts.user, media.blob or "", pair_blob=media.pair_blob, text_first=False
            )

        return [
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ]

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "429" in msg:
                return 59.0
            if "502" in msg or "RETRY_EMPTY_CONTENT" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        cfg.on_exhausted = lambda e: (self.ctx.console.print(f"[yellow]StepFun exhausted: {e}[/yellow]") or "")
        return cfg
