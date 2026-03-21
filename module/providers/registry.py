"""
ProviderRegistry - 自动发现和路由

修复原方案中每次调用都 discover() 的性能问题
使用模块级单例 + 明确的 provider 模块清单，避免静默吞掉发现错误
"""

import importlib
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

from .catalog import canonicalize_provider_name, route_provider_name

if TYPE_CHECKING:
    from .base import Provider


if __name__ == "providers.registry":
    sys.modules.setdefault("module.providers.registry", sys.modules[__name__])
elif __name__ == "module.providers.registry":
    sys.modules.setdefault("providers.registry", sys.modules[__name__])


@dataclass(frozen=True)
class ProviderImportFailure:
    provider_name: str
    module_path: str
    error: Exception
    traceback: str

    def summary(self) -> str:
        return f"{self.provider_name} ({self.module_path}) failed to import: {type(self.error).__name__}: {self.error}"


class ProviderImportError(RuntimeError):
    """Raised when an explicitly selected provider failed to import."""


class ProviderSelectionError(RuntimeError):
    """Raised when an explicitly selected provider cannot handle the current request."""


class ProviderDiscoveryError(RuntimeError):
    """Raised when strict discovery finds provider import failures."""

    def __init__(self, failures: List[ProviderImportFailure]):
        self.failures = failures
        joined = "\n\n".join(
            f"[{failure.provider_name}] {failure.module_path}\n{failure.traceback.strip()}" for failure in failures
        )
        super().__init__(f"Provider discovery failed for {len(failures)} provider(s):\n\n{joined}")


_PROVIDER_MODULES: Dict[str, str] = {
    "openai_compatible": "module.providers.cloud_vlm.openai_compatible",
    "stepfun": "module.providers.cloud_vlm.stepfun",
    "ark": "module.providers.cloud_vlm.ark",
    "qwenvl": "module.providers.cloud_vlm.qwenvl",
    "glm": "module.providers.cloud_vlm.glm",
    "kimi_code": "module.providers.cloud_vlm.kimi_code",
    "kimi_vl": "module.providers.cloud_vlm.kimi_vl",
    "minimax_code": "module.providers.cloud_vlm.minimax_code",
    "minimax_api": "module.providers.cloud_vlm.minimax_api",
    "deepseek_ocr": "module.providers.ocr.deepseek",
    "logics_ocr": "module.providers.ocr.logics",
    "dots_ocr": "module.providers.ocr.dots",
    "qianfan_ocr": "module.providers.ocr.qianfan",
    "lighton_ocr": "module.providers.ocr.lighton",
    "hunyuan_ocr": "module.providers.ocr.hunyuan",
    "glm_ocr": "module.providers.ocr.glm",
    "chandra_ocr": "module.providers.ocr.chandra",
    "olmocr": "module.providers.ocr.olmocr",
    "paddle_ocr": "module.providers.ocr.paddle",
    "nanonets_ocr": "module.providers.ocr.nanonets",
    "firered_ocr": "module.providers.ocr.firered",
    "moondream": "module.providers.local_vlm.moondream",
    "qwen_vl_local": "module.providers.local_vlm.qwen_vl_local",
    "step_vl_local": "module.providers.local_vlm.step_vl_local",
    "penguin_vl_local": "module.providers.local_vlm.penguin_vl_local",
    "reka_edge_local": "module.providers.local_vlm.reka_edge_local",
    "lfm_vl_local": "module.providers.local_vlm.lfm_vl_local",
    "music_flamingo_local": "module.providers.local_alm.music_flamingo_local",
    "mistral_ocr": "module.providers.vision_api.pixtral",
    "gemini": "module.providers.vision_api.gemini",
}


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
            self._import_failures: Dict[str, ProviderImportFailure] = {}
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
                "logics_ocr",
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

    def discover(self, *, strict: bool = False) -> None:
        """自动发现 - 只执行一次（幂等）"""
        _flush_pending_registrations()
        if self._discovered:
            if strict and self._import_failures:
                raise ProviderDiscoveryError(self.list_import_failures())
            return

        with self._lock:
            _flush_pending_registrations()
            if self._discovered:
                if strict and self._import_failures:
                    raise ProviderDiscoveryError(self.list_import_failures())
                return

            self._import_failures = {}
            for provider_name, module_path in _PROVIDER_MODULES.items():
                self._discover_provider_module(provider_name, module_path)

            self._discovered = True
            if strict and self._import_failures:
                raise ProviderDiscoveryError(self.list_import_failures())

    def _discover_provider_module(self, provider_name: str, module_path: str) -> None:
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            self._record_import_failure(provider_name, module_path, exc)
            return

        self._register_module_providers(module)

        if provider_name not in self._providers:
            exc = RuntimeError(f"Provider module '{module_path}' did not register expected provider '{provider_name}'")
            synthetic_traceback = "".join(traceback.format_exception_only(type(exc), exc))
            self._record_import_failure(provider_name, module_path, exc, formatted_traceback=f"{synthetic_traceback}")

    def _register_module_providers(self, module: Any) -> None:
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
                provider_name = getattr(attr, "name", None)
                if provider_name:
                    self.register(provider_name, attr)

    def _record_import_failure(
        self,
        provider_name: str,
        module_path: str,
        error: Exception,
        *,
        formatted_traceback: Optional[str] = None,
    ) -> None:
        canonical_name = canonicalize_provider_name(provider_name)
        self._import_failures[canonical_name] = ProviderImportFailure(
            provider_name=canonical_name,
            module_path=module_path,
            error=error,
            traceback=(
                f"Provider '{canonical_name}' from '{module_path}' failed to import.\n"
                f"{formatted_traceback or ''.join(traceback.format_exception(type(error), error, error.__traceback__))}"
            ),
        )

    def register(self, name: str, provider_class: Type["Provider"]):
        """注册 provider"""
        canonical_name = canonicalize_provider_name(name)
        self._providers[canonical_name] = provider_class
        self._import_failures.pop(canonical_name, None)
        return provider_class

    def find_provider(self, args: Any, mime: str) -> Optional[Type["Provider"]]:
        """根据参数和 mime 类型找到合适的 provider"""
        self.discover()  # 确保已发现

        explicit_provider = self._find_explicit_provider(args, mime)
        if explicit_provider is not None:
            return explicit_provider

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

    def _find_explicit_provider(self, args: Any, mime: str) -> Optional[Type["Provider"]]:
        for route_name, require_match in self._relevant_route_specs(args, mime):
            route_value = getattr(args, route_name, "")
            if not route_value:
                continue

            provider_name = canonicalize_provider_name(route_provider_name(route_name, route_value))
            failure = self.get_import_failure(provider_name)
            if failure is not None:
                if not require_match:
                    continue
                raise ProviderImportError(
                    f"Explicit provider '{provider_name}' selected via {route_name}='{route_value}' failed to import.\n\n{failure.traceback}"
                ) from failure.error

            provider_class = self._providers.get(provider_name)
            if provider_class is None:
                if not require_match:
                    continue
                raise ProviderSelectionError(
                    f"Explicit provider '{provider_name}' selected via {route_name}='{route_value}' is not registered."
                )

            try:
                if provider_class.can_handle(args, mime):
                    return provider_class
            except Exception as exc:
                if not require_match:
                    continue
                raise ProviderSelectionError(
                    f"Explicit provider '{provider_name}' selected via {route_name}='{route_value}' crashed in can_handle()"
                ) from exc

            if not require_match:
                continue
            raise ProviderSelectionError(
                f"Explicit provider '{provider_name}' selected via {route_name}='{route_value}' cannot handle mime={mime}."
            )

        return None

    @staticmethod
    def _relevant_route_specs(args: Any, mime: str) -> Tuple[Tuple[str, bool], ...]:
        if mime.startswith("audio"):
            return (("alm_model", True),)
        if mime.startswith("application"):
            return (("ocr_model", True),)
        if mime.startswith("image"):
            if getattr(args, "document_image", False):
                return (("ocr_model", True),)
            return (("vlm_image_model", True),)
        if mime.startswith("video"):
            return (("vlm_image_model", False),)
        return tuple()

    def get_provider(self, name: str) -> Optional[Type["Provider"]]:
        """按名称获取 provider"""
        self.discover()
        return self._providers.get(canonicalize_provider_name(name))

    def list_providers(self) -> List[str]:
        """列出所有已注册 provider"""
        self.discover()
        return list(self._providers.keys())

    def get_import_failure(self, name: str) -> Optional[ProviderImportFailure]:
        return self._import_failures.get(canonicalize_provider_name(name))

    def list_import_failures(self) -> List[ProviderImportFailure]:
        return [self._import_failures[name] for name in sorted(self._import_failures)]


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
