"""
ProviderRegistry - 自动发现和路由

修复原方案中每次调用都 discover() 的性能问题
使用模块级单例 + @lru_cache 优化
"""

import importlib
import pkgutil
import sys
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

if TYPE_CHECKING:
    from .base import Provider


if __name__ == "providers.registry":
    sys.modules.setdefault("module.providers.registry", sys.modules[__name__])
elif __name__ == "module.providers.registry":
    sys.modules.setdefault("providers.registry", sys.modules[__name__])


class ProviderRegistry:
    """
    Provider 注册表 - 单例 + 缓存优化

    修复：每次调用都 discover() 的性能问题
    """

    _instance: Optional["ProviderRegistry"] = None
    _initialized: bool = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 避免重复初始化
        if ProviderRegistry._initialized:
            return

        with self._lock:
            if ProviderRegistry._initialized:
                return

            self._providers: Dict[str, Type["Provider"]] = {}
            self._discovered = False

            # 修复 #1: kimi_code 排在 kimi_vl 之前
            # minimax_code 优先级高于 minimax_api（类似 kimi_code > kimi_vl）
            self._priority_order: List[str] = [
                "openai_compatible",  # 通用 OpenAI 兼容接口（最高优先级）
                "stepfun",
                "ark",
                "qwenvl",
                "glm",
                "kimi_code",  # 优先级高于 kimi_vl
                "kimi_vl",
                "minimax_code",  # 优先级高于 minimax_api
                "minimax_api",
                # OCR models
                "deepseek_ocr",
                "dots_ocr",
                "qianfan_ocr",
                "lighton_ocr",
                "hunyuan_ocr",
                "glm_ocr",
                "chandra_ocr",
                "olmocr",
                "paddle_ocr",
                "nanonets_ocr",
                "firered_ocr",
                # Local VLM
                "moondream",
                "qwen_vl_local",
                "step_vl_local",
                "penguin_vl_local",
                "reka_edge_local",
                "lfm_vl_local",
                # Local ALM
                "music_flamingo_local",
                # Vision API
                "mistral_ocr",
                "gemini",
            ]

            ProviderRegistry._initialized = True

    def discover(self) -> None:
        """自动发现 - 只执行一次（幂等）"""
        _flush_pending_registrations()
        if self._discovered:
            return

        with self._lock:
            _flush_pending_registrations()
            if self._discovered:
                return

            try:
                from . import cloud_vlm, local_alm, local_vlm, ocr, vision_api
            except ImportError:
                # 子包可能不存在，跳过
                return

            packages = [
                ("cloud_vlm", cloud_vlm),
                ("local_alm", local_alm),
                ("local_vlm", local_vlm),
                ("ocr", ocr),
                ("vision_api", vision_api),
            ]

            for pkg_name, pkg in packages:
                if not hasattr(pkg, "__path__"):
                    continue

                for _, name, _ in pkgutil.iter_modules(pkg.__path__):
                    try:
                        module = importlib.import_module(f"module.providers.{pkg_name}.{name}")

                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and attr_name
                                not in (
                                    "Provider",
                                    "CloudVLMProvider",
                                    "LocalALMProvider",
                                    "LocalVLMProvider",
                                    "OCRProvider",
                                    "VisionAPIProvider",
                                )
                                and hasattr(attr, "can_handle")
                                and callable(getattr(attr, "can_handle", None))
                            ):
                                # 获取 name 属性
                                provider_name = getattr(attr, "name", None)
                                if provider_name:
                                    self.register(provider_name, attr)

                    except Exception as e:
                        # 静默处理导入错误，避免影响其他 provider
                        pass

            self._discovered = True

    def register(self, name: str, provider_class: Type["Provider"]):
        """注册 provider"""
        self._providers[name] = provider_class
        return provider_class

    def find_provider(self, args: Any, mime: str) -> Optional[Type["Provider"]]:
        """根据参数和 mime 类型找到合适的 provider"""
        self.discover()  # 确保已发现

        for name in self._priority_order:
            provider_class = self._providers.get(name)
            if not provider_class:
                continue

            try:
                # 使用类方法检查
                if provider_class.can_handle(args, mime):
                    return provider_class
            except Exception:
                # 继续尝试下一个
                pass

        for name, provider_class in self._providers.items():
            if name in self._priority_order:
                continue

            try:
                if provider_class.can_handle(args, mime):
                    return provider_class
            except Exception:
                pass

        return None

    def get_provider(self, name: str) -> Optional[Type["Provider"]]:
        """按名称获取 provider"""
        self.discover()
        return self._providers.get(name)

    def list_providers(self) -> List[str]:
        """列出所有已注册 provider"""
        self.discover()
        return list(self._providers.keys())


# 模块级单例
_registry_instance: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """获取全局注册表实例（懒加载）"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ProviderRegistry()
    return _registry_instance


# 延迟注册队列（装饰器使用）
_pending_registrations: List[Tuple[str, Type]] = []


def _flush_pending_registrations():
    """将待注册的 Provider  flush 到注册表"""
    global _pending_registrations
    registry = get_registry()
    for name, cls in _pending_registrations:
        registry.register(name, cls)
    _pending_registrations = []


# 装饰器语法糖
def register_provider(name: str):
    """
    装饰器：@register_provider("stepfun")

    自动设置类属性并添加到待注册队列
    实际注册在 discover() 时完成
    """

    def decorator(cls: Type["Provider"]):
        cls.name = name
        # 添加到待注册队列
        _pending_registrations.append((name, cls))
        return cls

    return decorator
