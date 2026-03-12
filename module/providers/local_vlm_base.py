"""
LocalVLMProvider - 本地 VLM Provider 基类

适用于：Moondream, QwenVL_Local, StepVL_Local

特性：
- 统一的模型懒加载
- 设备/精度自动选择
- attention implementation 参数处理（_attn_implementation vs attn_implementation）
- 内存管理（unload/cache）
"""

import base64 as _base64
import gc
import threading
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from .backends import OpenAIChatRuntime, find_model_config_section, resolve_runtime_backend
from .base import MediaContext, MediaModality, Provider, ProviderType
from .capabilities import ProviderCapabilities
from .catalog import provider_config_sections
from .utils import build_vision_messages, encode_image_to_blob

# 全局模型缓存（所有 LocalVLM 共享）
_global_model_cache: Dict[str, Any] = {}
_global_cache_lock = threading.Lock()


class LocalVLMProvider(Provider):
    """
    本地 VLM Provider 基类

    修复：模型加载器缓存模式不一致的问题
    """

    provider_type = ProviderType.LOCAL_VLM
    capabilities = ProviderCapabilities(
        supports_streaming=False,
        supports_images=True,
    )

    # 子类必须定义
    default_model_id: ClassVar[str] = ""

    # 修复 #9：attention 实现参数（带下划线前缀）
    _attn_implementation: ClassVar[str] = "eager"

    def __init__(self, context):
        super().__init__(context)
        self._model_key = f"{self.name}:{self.model_id}"

    @property
    def model_id(self) -> str:
        """从 config 或 args 获取模型 ID"""
        for section_name in provider_config_sections(self.name):
            section = self.ctx.config.get(section_name, {})
            if "model_id" in section:
                return section["model_id"]
        return self.default_model_id

    @property
    def model_config(self) -> Dict[str, Any]:
        """获取模型配置 section"""
        for section_name in provider_config_sections(self.name):
            section = self.ctx.config.get(section_name, {})
            if section:
                return section
        return {}

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        """本地 VLM 通用的媒体准备"""
        file_path = Path(uri)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        blob = None
        pixels = None
        pair_blob = None
        pair_pixels = None
        extras: Dict[str, Any] = {}
        modality = MediaModality.UNKNOWN

        if mime.startswith("image"):
            modality = MediaModality.IMAGE
            blob, pixels = encode_image_to_blob(uri, to_rgb=True)
            pair_dir = getattr(args, "pair_dir", "")
            if pair_dir:
                pair_path = (Path(pair_dir) / Path(uri).name).resolve()
                if pair_path.exists():
                    pair_blob, pair_pixels = encode_image_to_blob(str(pair_path), to_rgb=True)
                    extras["pair_uri"] = str(pair_path)
        elif mime.startswith("video"):
            modality = MediaModality.VIDEO

        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=modality,
            file_size=file_size,
            blob=blob,
            pixels=pixels,
            pair_blob=pair_blob,
            pair_pixels=pair_pixels,
            extras=extras,
        )

    def get_runtime_backend(self):
        """Resolve execution backend for this local provider."""
        provider_section = self.model_config
        runtime_model_name = (
            getattr(self.ctx.args, "openai_model_name", "")
            or provider_section.get("runtime_model_id", "")
            or provider_section.get("model_id", "")
            or self.model_id
        )
        model_section = find_model_config_section(
            self.ctx.config,
            runtime_model_name,
            preferred_sections=tuple(provider_config_sections(self.name)),
        )
        default_temperature = float(model_section.get("temperature", provider_section.get("temperature", 0.0)))
        default_max_tokens = int(
            model_section.get(
                "out_seq_length",
                model_section.get(
                    "max_new_tokens",
                    provider_section.get("out_seq_length", provider_section.get("max_new_tokens", 2048)),
                ),
            )
        )
        default_model_id = (
            model_section.get("runtime_model_id", "")
            or model_section.get("model_id", "")
            or provider_section.get("runtime_model_id", "")
            or self.model_id
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

    def attempt_via_openai_backend(
        self,
        media: MediaContext,
        prompts: Any,
        *,
        text_first: bool = False,
        user_prompt: Optional[str] = None,
        stop: Optional[list[str]] = None,
    ):
        """Run the current provider through an OpenAI-compatible local server."""
        runtime = self.get_runtime_backend()
        backend = OpenAIChatRuntime(runtime)
        messages = self._build_runtime_messages(media, prompts, text_first=text_first, user_prompt=user_prompt)
        if not messages:
            return self._empty_runtime_result(runtime)

        result = backend.complete(messages, stop=stop)
        return self._wrap_runtime_result(result, runtime)

    def _build_runtime_messages(
        self,
        media: MediaContext,
        prompts: Any,
        *,
        text_first: bool = False,
        user_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        prompt_text = user_prompt or prompts.user

        if media.mime.startswith("image"):
            if media.blob is None:
                return []
            return build_vision_messages(
                prompts.system,
                prompt_text,
                media.blob,
                pair_blob=media.pair_blob,
                text_first=text_first,
            )

        if media.mime.startswith("video"):
            try:
                video_base = _base64.b64encode(Path(media.uri).read_bytes()).decode("utf-8")
            except OSError:
                return []

            return [
                {"role": "system", "content": prompts.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": f"data:{media.mime};base64,{video_base}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]

        return []

    def _wrap_runtime_result(self, result: str, runtime):
        from .base import CaptionResult

        return CaptionResult(
            raw=result,
            metadata={
                "provider": self.name,
                "runtime_backend": runtime.mode,
                "runtime_model_id": runtime.model_id,
            },
        )

    def _empty_runtime_result(self, runtime):
        return self._wrap_runtime_result("", runtime)

    def _get_or_load_model(self):
        """获取或加载模型（带全局缓存）"""
        with _global_cache_lock:
            if self._model_key in _global_model_cache:
                return _global_model_cache[self._model_key]

            # 加载模型
            model = self._load_model()
            _global_model_cache[self._model_key] = model
            return model

    def _load_model(self):
        """
        子类必须实现的加载逻辑

        子类应该：
        1. 调用 resolve_device_dtype()
        2. 使用 _get_attn_kwargs() 获取 attention 参数
        3. 加载模型并返回
        """
        raise NotImplementedError(f"{self.name} must implement _load_model()")

    def _resolve_device_dtype(self):
        """解析设备和精度"""
        try:
            from utils.transformer_loader import resolve_device_dtype

            return resolve_device_dtype(
                device_arg=getattr(self.ctx.args, "device", None), dtype_arg=getattr(self.ctx.args, "dtype", None)
            )
        except ImportError:
            # 默认返回
            import torch

            return {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }

    def _get_attn_kwargs(self) -> Dict[str, Any]:
        """
        获取 attention 实现参数

        修复 #9：处理 _attn_implementation vs attn_implementation 差异
        使用带下划线的版本以兼容旧模型
        """
        return {"_attn_implementation": self._attn_implementation}

    @classmethod
    def unload_model(cls, model_id: Optional[str] = None):
        """卸载模型释放内存"""
        with _global_cache_lock:
            if model_id:
                key = f"{cls.name}:{model_id}"
                if key in _global_model_cache:
                    del _global_model_cache[key]
            else:
                # 卸载该 provider 类型的所有模型
                keys_to_remove = [k for k in _global_model_cache if k.startswith(f"{cls.name}:")]
                for k in keys_to_remove:
                    del _global_model_cache[k]

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @classmethod
    def clear_all_cache(cls):
        """清空所有模型缓存"""
        with _global_cache_lock:
            _global_model_cache.clear()

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
