"""
LocalVLMProvider - 本地 VLM Provider 基类

适用于：Moondream, QwenVL_Local, StepVL_Local

特性：
- 统一的模型懒加载
- 设备/精度自动选择
- attention implementation 参数处理（_attn_implementation vs attn_implementation）
- 内存管理（unload/cache）
"""

import gc
import threading
from typing import Any, ClassVar, Dict, Optional

from .catalog import provider_config_sections
from .base import MediaContext, MediaModality, Provider, ProviderType
from .capabilities import ProviderCapabilities
from .utils import encode_image_to_blob

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
        blob = None
        pixels = None

        if mime.startswith("image"):
            blob, pixels = encode_image_to_blob(uri, to_rgb=True)

        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=MediaModality.IMAGE,
            blob=blob,
            pixels=pixels,
        )

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
