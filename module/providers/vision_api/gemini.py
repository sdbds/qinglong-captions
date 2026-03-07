"""Gemini Provider"""

import json
from pathlib import Path

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.registry import register_provider
from providers.vision_api_base import StructuredOutputConfig, VisionAPIProvider


@register_provider("gemini")
class GeminiProvider(VisionAPIProvider):
    """Gemini Provider - 支持多模态和结构化输出"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, "gemini_api_key", "") != ""

    def get_structured_output_config(self, media: MediaContext, args) -> StructuredOutputConfig:
        """Gemini 结构化输出配置"""
        if not media.mime.startswith("image"):
            return StructuredOutputConfig(enabled=False)

        if getattr(args, "gemini_task", ""):
            # Task 模式不使用结构化输出
            return StructuredOutputConfig(enabled=False)

        if getattr(args, "pair_dir", ""):
            schema = self._build_pair_image_schema()
        else:
            schema = self._build_rating_schema()

        return StructuredOutputConfig(enabled=True, mime_type="application/json", schema=schema, response_modalities=["Text"])

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from google import genai
        from google.genai import types
        from module.providers.gemini_provider import attempt_gemini

        client = genai.Client(api_key=self.ctx.args.gemini_api_key)

        # 获取 generation config
        generation_config = self._get_generation_config()

        # 获取结构化输出配置
        struct_config = self.get_structured_output_config(media, self.ctx.args)

        # 构建 GenAI Config
        genai_config = self._build_genai_config(prompts, generation_config, struct_config)

        # 准备内容
        contents = self._build_contents(media, prompts)

        # 准备 pair extras
        pair_blob_list = media.pair_extras if media.pair_extras else None

        result = attempt_gemini(
            client=client,
            model_path=self.ctx.args.gemini_model_path,
            mime=media.mime,
            prompt=prompts.user,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            uri=media.uri,
            genai_config=genai_config,
            files=media.video_file_refs if media.video_file_refs else None,
            audio_bytes=media.audio_blob,
            image_blob=media.blob,
            pixels=media.pixels,
            pair_blob=media.pair_blob,
            pair_pixels=media.pair_pixels,
            pair_blob_list=pair_blob_list,
            gemini_task=getattr(self.ctx.args, "gemini_task", ""),
        )

        # 解析 JSON 如果启用了结构化输出
        parsed = None
        if struct_config.enabled:
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                pass

        return CaptionResult(raw=result, parsed=parsed, metadata={"provider": self.name, "structured": struct_config.enabled})

    def _get_generation_config(self) -> dict:
        """获取 generation 配置"""
        model_key = self.ctx.args.gemini_model_path.replace(".", "_")

        if self.ctx.config.get("generation_config", {}).get(model_key):
            return self.ctx.config["generation_config"][model_key]
        return self.ctx.config.get("generation_config", {}).get("default", {})

    def _build_genai_config(self, prompts: PromptContext, gen_cfg: dict, struct: StructuredOutputConfig):
        """构建 GenAI Config"""
        from google.genai import types

        config_dict = {
            "system_instruction": prompts.system,
            "temperature": gen_cfg.get("temperature", 0.7),
            "top_p": gen_cfg.get("top_p", 0.95),
            "top_k": gen_cfg.get("top_k", 40),
            "candidate_count": self.ctx.config.get("generation_config", {}).get("candidate_count", 1),
            "max_output_tokens": gen_cfg.get("max_output_tokens", 4096),
            "safety_settings": [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.OFF),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.OFF),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.OFF
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.OFF
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.OFF
                ),
            ],
            "response_mime_type": "application/json" if struct.enabled else gen_cfg.get("response_mime_type", "text/plain"),
            "response_modalities": gen_cfg.get("response_modalities", ["Text"]),
        }

        if struct.enabled and struct.schema:
            config_dict["response_schema"] = struct.schema

        if not getattr(self.ctx.args, "gemini_task", ""):
            config_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=gen_cfg.get("thinking_budget", -1))

        return types.GenerateContentConfig(**config_dict)

    def _build_contents(self, media: MediaContext, prompts: PromptContext):
        """构建内容"""
        from google.genai import types

        if media.mime.startswith("video") or (media.mime.startswith("audio") and media.is_large_file):
            # 视频/大音频 - 使用文件引用
            if media.video_file_refs:
                return [
                    types.Part.from_uri(file_uri=media.video_file_refs[0].uri, mime_type=media.mime),
                    types.Part.from_text(text=prompts.user),
                ]

        elif media.mime.startswith("audio"):
            # 小音频 - 直接 bytes
            if media.audio_blob:
                return [
                    types.Part.from_bytes(data=media.audio_blob, mime_type=media.mime),
                    types.Part.from_text(text=prompts.user),
                ]

        elif media.mime.startswith("image"):
            # 图像
            if media.blob:
                # 处理 pair 图像
                parts = []
                if media.pair_blob:
                    parts.append(types.Part.from_bytes(data=media.pair_blob, mime_type="image/jpeg"))
                if media.pair_extras:
                    for extra_blob in media.pair_extras:
                        parts.append(types.Part.from_bytes(data=extra_blob, mime_type="image/jpeg"))
                parts.append(types.Part.from_bytes(data=media.blob, mime_type="image/jpeg"))
                parts.append(types.Part.from_text(text=prompts.user))
                return parts

        # 默认
        return [types.Part.from_text(text=prompts.user)]

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "429" in msg:
                return 59.0
            if "502" in msg or "RETRY_EMPTY_CONTENT" in msg or "RETRY_UNSUPPORTED_MIME" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        return cfg
