"""
Provider 工具函数

- encode_image_to_blob: 图像编码
- with_retry_impl: 重试逻辑
- build_vision_messages: 构建视觉消息
"""

import base64
import functools
import io
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from rich.console import Console
from utils.console_util import print_exception

try:
    from rich_pixels import Pixels
except ImportError:
    Pixels = Any  # 降级处理


def encode_image_to_blob(
    image_path: str, to_rgb: bool = False, max_size: int = 1024, quality: int = 95
) -> Tuple[Optional[str], Optional[Pixels]]:
    """
    编码图像为 base64，带尺寸优化

    原 encode_image 函数的改进版（不带 LRU 缓存）
    如需缓存，可在调用处自行添加
    """
    try:
        with Image.open(image_path) as image:
            image.load()

            # 删除 XMP 数据避免错误
            if "xmp" in image.info:
                del image.info["xmp"]

            # 可选 RGB 转换
            if to_rgb and image.mode != "RGB":
                try:
                    image = image.convert("RGB")
                except Exception:
                    pass

            # 计算 16 的倍数尺寸
            width, height = image.size
            aspect = width / height

            if width > height:
                new_w = min(max_size, (width // 16) * 16)
                new_h = ((int(new_w / aspect)) // 16) * 16
            else:
                new_h = min(max_size, (height // 16) * 16)
                new_w = ((int(new_h * aspect)) // 16) * 16

            # 确保不超过 max_size
            if new_w > max_size:
                new_w = max_size
                new_h = ((int(new_w / aspect)) // 16) * 16
            if new_h > max_size:
                new_h = max_size
                new_w = ((int(new_h * aspect)) // 16) * 16

            if (new_w, new_h) != (width, height):
                image = image.resize((new_w, new_h), Image.LANCZOS)

            if image.mode != "RGB":
                image = image.convert("RGB")

            # 创建缩略图用于显示
            pixels = Pixels.from_image(image, resize=(max(1, image.width // 18), max(1, image.height // 18)))

            # 编码为 base64
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=quality)
                blob = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return blob, pixels

    except FileNotFoundError:
        print(f"[red]Error:[/red] File not found - {image_path}")
    except Image.UnidentifiedImageError:
        print(f"[red]Error:[/red] Cannot identify image file - {image_path}")
    except PermissionError:
        print(f"[red]Error:[/red] Permission denied accessing file - {image_path}")
    except OSError as e:
        if "XMP data is too long" in str(e):
            print(f"[yellow]Warning:[/yellow] Skipping image with XMP data error - {image_path}")
        else:
            print(f"[red]Error:[/red] OS error processing file {image_path}: {e}")
    except Exception as e:
        print(f"[red]Error:[/red] Unexpected error processing {image_path}: {e}")

    return None, None


def with_retry_impl(fn: Callable[[], Any], retry_config: Any, console: Optional[Console] = None) -> Any:
    """
    重试逻辑实现

    从原 with_retry 函数提取，适配 RetryConfig
    """
    from .base import RetryConfig

    max_retries = retry_config.max_retries
    base_wait = retry_config.base_wait

    def _default_classifier(e: Exception) -> Optional[float]:
        s = str(e)
        if "429" in s:
            return 59.0
        if "502" in s:
            return base_wait
        return None

    classifier = retry_config.classify_error or _default_classifier

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            return fn()
        except Exception as e:
            if console:
                print_exception(
                    console,
                    e,
                    prefix=f"[retry] {attempt + 1}/{max_retries} failed",
                    summary_style="yellow",
                )

            if attempt >= max_retries - 1:
                if retry_config.on_exhausted:
                    try:
                        return retry_config.on_exhausted(e)
                    except Exception:
                        pass
                raise

            # 计算等待时间
            wait = classifier(e)
            if wait is None:
                raise  # 不重试

            jitter = wait * 0.2
            sleep_for = max(0.0, wait + random.uniform(-jitter, jitter))
            elapsed = time.time() - start_time
            remaining = max(0.0, sleep_for - elapsed)

            if console and remaining > 0:
                console.print(f"[yellow]Retrying in {remaining:.0f}s...[/yellow]")

            if remaining > 0:
                time.sleep(remaining)


def _contains_http_status(message: str, status: int) -> bool:
    return re.search(rf"(?<!\d){status}(?!\d)", message) is not None


def classify_remote_api_error(
    error: Exception,
    *,
    base_wait: float,
    rate_limit_wait: float = 59.0,
    retry_http_statuses: tuple[int, ...] = (502,),
    retry_markers: tuple[str, ...] = (),
    transport_wait: Optional[float] = None,
    fast_fail_http_statuses: tuple[int, ...] = (400, 401, 403),
    fast_fail_markers: tuple[str, ...] = (),
) -> Optional[float]:
    """
    统一判定远程 API 异常是否应重试。

    规则：
    - 429 / rate limit: 重试
    - 传输层异常（连接断开、超时、协议截断等）: 重试
    - provider 显式 RETRY_* 标记: 重试
    - 400/401/403 与典型参数错误: 快速失败
    - 其他错误: 默认重试
    """
    message = str(error)
    lower = message.lower()

    if _contains_http_status(message, 429) or "rate limit" in lower:
        return rate_limit_wait

    default_fast_fail_markers = (
        "unauthorized",
        "forbidden",
        "invalid parameter",
        "invalid parameters",
        "invalid argument",
        "invalid arguments",
        "invalid_request_error",
        "bad request",
        "validation error",
        "unsupported mime",
        "unsupported_mime",
        "malformed request",
        "missing required",
    )
    if any(_contains_http_status(message, status) for status in fast_fail_http_statuses):
        return None
    if any(marker in lower for marker in default_fast_fail_markers):
        return None
    if any(marker.lower() in lower for marker in fast_fail_markers):
        return None

    if any(marker in message for marker in retry_markers):
        return base_wait
    if any(_contains_http_status(message, status) for status in retry_http_statuses):
        return base_wait

    transport_markers = (
        "remoteprotocolerror",
        "incomplete chunked read",
        "peer closed connection",
        "readtimeout",
        "connecttimeout",
        "timed out",
        "timeout",
        "connection reset",
        "connection reset by peer",
        "connection aborted",
        "connection broken",
        "connection refused",
        "connectionerror",
        "connecterror",
        "server disconnected",
        "broken pipe",
        "socket hang up",
        "eof occurred",
        "unexpected eof",
    )
    if any(marker in lower for marker in transport_markers):
        return transport_wait if transport_wait is not None else base_wait

    return base_wait


def build_vision_messages(
    system_prompt: str, user_prompt: str, blob: str, pair_blob: Optional[str] = None, text_first: bool = True
) -> list:
    """构建标准视觉 API 消息格式（OpenAI 兼容格式）"""
    content = []

    if text_first:
        content.append({"type": "text", "text": user_prompt})

    image_url = f"data:image/jpeg;base64,{blob}"
    content.append({"type": "image_url", "image_url": {"url": image_url}})

    if pair_blob:
        pair_url = f"data:image/jpeg;base64,{pair_blob}"
        content.append({"type": "image_url", "image_url": {"url": pair_url}})

    if not text_first:
        content.append({"type": "text", "text": user_prompt})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


# LRU 缓存版本（可选）
@functools.lru_cache(maxsize=128)
def encode_image_cached(image_path: str, to_rgb: bool = False, max_size: int = 1024) -> Tuple[Optional[str], Optional[Pixels]]:
    """
    带缓存的图像编码版本

    适用于批量处理时同一图像被多次编码的场景
    """
    return encode_image_to_blob(image_path, to_rgb, max_size)
