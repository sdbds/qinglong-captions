"""Kimi Code Provider

注意：kimi_code 和 kimi_vl 是两个独立的 provider：
- kimi_code: 使用 api.moonshot.cn，支持 thinking 模式
- kimi_vl: 使用 integrate.api.nvidia.com，支持 JSON 结构化输出

优先级：kimi_code > kimi_vl
"""

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.cloud_vlm_base import CloudVLMProvider
from module.providers.registry import register_provider
from module.providers.utils import build_vision_messages


@register_provider("kimi_code")
class KimiCodeProvider(CloudVLMProvider):
    """Kimi Code Provider (Moonshot API)"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        """kimi_code 优先级高于 kimi_vl"""
        return getattr(args, "kimi_code_api_key", "") != "" and mime.startswith(("image", "video"))

    # Kimi Code API 通过 User-Agent 识别 coding agent，必须设置此 header
    KIMI_CODE_USER_AGENT = "claude-code/0.1.0"

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from openai import OpenAI
        from module.providers.cloud_vlm.kimi_vl import attempt_kimi_vl, ensure_kimi_dual_caption_prompt

        base_url = getattr(self.ctx.args, "kimi_code_base_url", "https://api.kimi.com/coding/v1")
        client = OpenAI(
            api_key=self.ctx.args.kimi_code_api_key,
            base_url=base_url,
            default_headers={"User-Agent": self.KIMI_CODE_USER_AGENT},
        )

        pair_pixels = None
        image_pixels = None

        if media.mime.startswith("video"):
            # 视频模式 - 编码为 base64
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

            system_prompt = ensure_kimi_dual_caption_prompt(prompts.system)
            messages = build_vision_messages(
                system_prompt, prompts.user, media.blob, pair_blob=media.pair_blob if pair_dir else None, text_first=False
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
            model_path=getattr(self.ctx.args, "kimi_code_model_path", "kimi-for-coding"),
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
        from module.providers.utils import classify_remote_api_error

        cfg = super().get_retry_config()
        cfg.classify_error = lambda e: classify_remote_api_error(
            e,
            base_wait=cfg.base_wait,
            retry_markers=("RETRY_EMPTY_CONTENT",),
        )
        return cfg
