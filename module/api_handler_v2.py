"""
Provider V2 统一入口

重构后的 api_process_batch，使用新的 Provider 架构

用法:
    from module.api_handler_v2 import api_process_batch
    result = api_process_batch(uri, mime, config, args, hash)

向后兼容:
    - 函数签名与原 api_process_batch 完全一致
    - 返回值 CaptionResult 可以通过 .raw 获取字符串
    - 不会在失败时自动回退到旧实现
"""

from typing import Any, Optional

from rich.console import Console
from rich.progress import Progress

from config.runtime_config import coerce_runtime_config
from module.providers import CaptionResult, ProviderContext, get_registry
from module.providers.catalog import normalize_runtime_args
from utils.console_util import print_exception


class NoProviderAvailableError(RuntimeError):
    """Raised when no provider can handle the current request."""

    def __init__(self, *, mime: str, available_providers: list[str], import_failures: list[object]):
        self.mime = mime
        self.available_providers = available_providers
        self.import_failures = import_failures

        details = [
            f"No provider available for mime={mime}",
            f"Available providers: {available_providers}",
        ]
        if import_failures:
            details.append("Import failures:")
            details.extend(f"  - {failure.summary()}" for failure in import_failures)
        super().__init__("\n".join(details))


def _build_no_provider_error(registry: Any, mime: str) -> NoProviderAvailableError:
    available_providers = registry.list_providers()
    list_import_failures = getattr(registry, "list_import_failures", None)
    import_failures = list_import_failures() if callable(list_import_failures) else []
    return NoProviderAvailableError(
        mime=mime,
        available_providers=available_providers,
        import_failures=import_failures,
    )


def api_process_batch(
    uri: str,
    mime: str,
    config: dict,
    args: Any,
    sha256hash: str,
    progress: Optional[Progress] = None,
    task_id=None,
) -> CaptionResult:
    """
    新的统一入口

    完整的执行流程：
    1. 获取全局注册表（带缓存）
    2. 自动发现所有 provider
    3. 根据优先级找到合适的 provider
    4. 实例化并执行
    5. 返回 CaptionResult
    """
    console = progress.console if progress else Console(color_system="truecolor", force_terminal=True)
    normalize_runtime_args(args)

    # 创建上下文
    runtime_config = coerce_runtime_config(config)

    ctx = ProviderContext(
        console=console,
        progress=progress,
        task_id=task_id,
        config=runtime_config,
        args=args,
    )

    # 获取全局注册表（带缓存，只 discover 一次）
    registry = get_registry()

    # 查找 provider
    try:
        provider_class = registry.find_provider(args, mime)
    except Exception as e:
        print_exception(console, e, prefix="Provider resolution failed")
        raise

    if not provider_class:
        error = _build_no_provider_error(registry, mime)
        console.print(f"[red]{error}[/red]")
        raise error

    # 实例化 provider
    provider = provider_class(ctx)
    console.print(f"[blue]Using provider: {provider.display_name(mime)}[/blue]")

    # 执行
    try:
        result = provider.execute(uri, mime, sha256hash)
        return result
    except Exception as e:
        print_exception(console, e, prefix=f"Provider {provider_class.name} failed")
        raise


# 兼容旧接口的别名
api_process_batch_v2 = api_process_batch
