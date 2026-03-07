"""Kimi VL Provider

注意：kimi_vl 和 kimi_code 是两个独立的 provider
- kimi_vl: 使用 integrate.api.nvidia.com，支持 JSON 结构化输出
- kimi_code: 使用 api.moonshot.cn

kimi_code 优先级高于 kimi_vl
"""

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider
from providers.utils import build_vision_messages


@register_provider("kimi_vl")
class KimiVLProvider(CloudVLMProvider):
    """Kimi VL Provider (NVIDIA API)"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        """kimi_vl 在 kimi_code 之后检查"""
        return getattr(args, "kimi_api_key", "") != ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from openai import OpenAI
        from module.providers.kimi_vl_provider import attempt_kimi_vl

        base_url = getattr(self.ctx.args, "kimi_base_url", "https://api.moonshot.cn/v1")
        client = OpenAI(api_key=self.ctx.args.kimi_api_key, base_url=base_url)

        pair_pixels = None
        image_pixels = None

        if media.mime.startswith("video"):
            import base64

            with open(media.uri, "rb") as f:
                video_base = base64.b64encode(f.read()).decode("utf-8")
            video_data_url = f"data:{media.mime};base64,{video_base}"

            messages = [
                {"role": "system", "content": prompts.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_data_url}},
                        {"type": "text", "text": prompts.user},
                    ],
                },
            ]
        elif media.mime.startswith("image"):
            if media.blob is None:
                return CaptionResult(raw="")

            pair_dir = getattr(self.ctx.args, "pair_dir", "")
            if pair_dir and (not media.pair_blob):
                return CaptionResult(raw="")

            messages = build_vision_messages(
                prompts.system, prompts.user, media.blob, pair_blob=media.pair_blob if pair_dir else None, text_first=False
            )
            image_pixels = media.pixels
            pair_pixels = media.pair_pixels
        else:
            return CaptionResult(raw="")

        # 读取 thinking 配置
        kimi_vl_config = self.ctx.config.get("kimi_vl", {})
        thinking = kimi_vl_config.get("thinking", "enabled") if kimi_vl_config else "enabled"

        result = attempt_kimi_vl(
            client=client,
            model_path=getattr(self.ctx.args, "kimi_model_path", "kimi-k2.5"),
            messages=messages,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            uri=media.uri,
            image_pixels=image_pixels,
            pair_pixels=pair_pixels,
            thinking=thinking,
            mode=getattr(self.ctx.args, "mode", "all"),
        )

        # 处理 JSON 解析 - 尝试解析为 dict，根据 mode 过滤字段
        import json
        
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if isinstance(parsed, dict):
                mode = getattr(self.ctx.args, "mode", "all")
                # 根据 mode 过滤字段（与 V1 保持一致）
                if mode == "short":
                    parsed.pop("long", None)
                    parsed.pop("long_description", None)
                elif mode == "long":
                    parsed.pop("short", None)
                    parsed.pop("short_description", None)
                return CaptionResult(raw=json.dumps(parsed, ensure_ascii=False), parsed=parsed, metadata={"provider": self.name})
        except Exception:
            pass

        return CaptionResult(raw=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False), metadata={"provider": self.name})

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
