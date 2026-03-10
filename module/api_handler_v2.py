"""
Provider V2 统一入口

重构后的 api_process_batch，使用新的 Provider 架构

用法:
    # 默认使用 V2 新架构
    # 通过环境变量显式回退到旧代码
    export QINGLONG_API_V2=0  # 回退到旧代码

    from module.api_handler_v2 import api_process_batch
    result = api_process_batch(uri, mime, config, args, hash)

向后兼容:
    - 函数签名与原 api_process_batch 完全一致
    - 返回值 CaptionResult 可以通过 .raw 获取字符串
    - 失败时自动回退到旧实现（如果配置了）
"""

import os
from typing import Any, Optional

from rich.console import Console
from rich.progress import Progress

from module.providers import CaptionResult, ProviderContext, get_registry


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
    console = progress.console if progress else Console(color_system="truecolor")

    # 创建上下文
    ctx = ProviderContext(
        console=console,
        progress=progress,
        task_id=task_id,
        config=config,
        args=args,
    )

    # 获取全局注册表（带缓存，只 discover 一次）
    registry = get_registry()

    # 查找 provider
    provider_class = registry.find_provider(args, mime)

    if not provider_class:
        # 没有 provider 能处理
        console.print(f"[red]No provider available for mime={mime}[/red]")
        console.print(f"[yellow]Available providers: {registry.list_providers()}[/yellow]")
        return CaptionResult(raw="")

    console.print(f"[blue]Using provider: {provider_class.name}[/blue]")

    # 实例化 provider
    provider = provider_class(ctx)

    # 执行
    try:
        result = provider.execute(uri, mime, sha256hash)
        return result
    except Exception as e:
        console.print(f"[red]Provider {provider_class.name} failed: {e}[/red]")
        raise


# 兼容旧接口的别名
api_process_batch_v2 = api_process_batch


def api_process_batch_legacy(
    uri: str,
    mime: str,
    config: dict,
    args: Any,
    sha256hash: str,
    progress: Optional[Progress] = None,
    task_id=None,
) -> str:
    """
    旧实现调用（用于回退）

    直接调用原 api_handler.api_process_batch
    """
    from module.api_handler import api_process_batch as _legacy

    return _legacy(
        uri=uri,
        mime=mime,
        config=config,
        args=args,
        sha256hash=sha256hash,
        progress=progress,
        task_id=task_id,
    )


# 便捷函数：检查是否使用 V2
def is_v2_enabled() -> bool:
    """检查是否启用了 V2 架构（默认启用）"""
    return os.environ.get("QINGLONG_API_V2", "1") != "0"


# 便捷函数：切换版本
def use_v2(enabled: bool = True):
    """
    设置是否使用 V2 架构

    用法:
        use_v2(True)   # 启用 V2
        use_v2(False)  # 使用旧代码
    """
    os.environ["QINGLONG_API_V2"] = "1" if enabled else "0"
