"""QwenVL Provider"""

from pathlib import Path

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider


@register_provider("qwenvl")
class QwenVLProvider(CloudVLMProvider):
    """QwenVL API Provider"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, "qwenVL_api_key", "") != "" and mime.startswith("video")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from module.providers.qwenvl_provider import attempt_qwenvl

        file = f"file://{Path(media.uri).resolve().as_posix()}"

        if media.mime.startswith("video"):
            self.log(f"Uploading video: {file}", "blue")
            messages = [
                {
                    "role": "system",
                    "content": [{"text": prompts.system}],
                },
                {
                    "role": "user",
                    "content": [
                        {"video": file},
                        {"text": prompts.user},
                    ],
                },
            ]
        elif media.mime.startswith("image"):
            self.log(f"Preparing image: {file}", "blue")
            content_items = []

            pair_dir = getattr(self.ctx.args, "pair_dir", "")
            if pair_dir:
                from pathlib import Path

                pair_path = (Path(pair_dir) / Path(media.uri).name).resolve()
                if pair_path.exists():
                    pair_file = f"file://{pair_path.as_posix()}"
                    self.log(f"Pair image: {pair_file}", "yellow")
                    content_items.extend([{"image": file}, {"image": pair_file}])
                else:
                    self.log(f"Pair not found: {pair_path}", "red")
                    return CaptionResult(raw="")
            else:
                content_items.append({"image": file})

            content_items.append({"text": prompts.user})

            messages = [
                {"role": "system", "content": [{"text": prompts.system}]},
                {"role": "user", "content": content_items},
            ]
        else:
            messages = [
                {"role": "system", "content": [{"text": prompts.system}]},
                {"role": "user", "content": [{"text": prompts.user}]},
            ]

        result = attempt_qwenvl(
            model_path=self.ctx.args.qwenVL_model_path,
            api_key=self.ctx.args.qwenVL_api_key,
            messages=messages,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
        )

        return CaptionResult(raw=result, metadata={"provider": self.name})

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
        return cfg
