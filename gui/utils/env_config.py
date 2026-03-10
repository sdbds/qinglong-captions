"""环境变量配置管理

对应 4、run.ps1 脚本中 "DO NOT MODIFY" 分隔线以下的环境变量，
在 GUI 中统一管理并注入子进程。
持久化为 config/env_vars.json。
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "env_vars.json"

# 环境变量定义：key, 默认值, 分组
# 分组: runtime / uv / network
ENV_VAR_DEFINITIONS: list[dict] = [
    # ── Runtime ──
    {"key": "QINGLONG_API_V2", "default": "1", "group": "runtime",
     "desc_en": "Use V2 API architecture", "desc_zh": "使用 V2 API 架构"},
    {"key": "HF_HOME", "default": "huggingface", "group": "runtime",
     "desc_en": "HuggingFace cache directory", "desc_zh": "HuggingFace 缓存目录"},
    {"key": "HF_ENDPOINT", "default": "https://hf-mirror.com", "group": "runtime",
     "desc_en": "HuggingFace mirror endpoint", "desc_zh": "HuggingFace 镜像地址"},
    {"key": "XFORMERS_FORCE_DISABLE_TRITON", "default": "1", "group": "runtime",
     "desc_en": "Disable Triton in xformers", "desc_zh": "禁用 xformers 中的 Triton"},
    {"key": "PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG", "default": "1", "group": "runtime",
     "desc_en": "Suppress Pillow XMP warnings", "desc_zh": "忽略 Pillow XMP 数据过长警告"},
    {"key": "CUDA_VISIBLE_DEVICES", "default": "", "group": "runtime",
     "desc_en": "GPU device IDs (e.g. 0, 0,1, -1 for CPU)", "desc_zh": "GPU 设备号 (如 0, 0,1, -1 表示 CPU)"},
    # ── UV Package Manager ──
    {"key": "UV_INDEX_URL", "default": "", "group": "uv",
     "desc_en": "UV primary index URL (PyPI mirror)", "desc_zh": "UV 主索引地址 (PyPI 镜像)"},
    {"key": "UV_EXTRA_INDEX_URL", "default": "https://download.pytorch.org/whl/cu128", "group": "uv",
     "desc_en": "UV extra index URL (PyTorch wheels)", "desc_zh": "UV 额外索引地址 (PyTorch wheels)"},
    {"key": "UV_CACHE_DIR", "default": "", "group": "uv",
     "desc_en": "UV cache directory", "desc_zh": "UV 缓存目录"},
    {"key": "UV_NO_BUILD_ISOLATION", "default": "1", "group": "uv",
     "desc_en": "Disable build isolation", "desc_zh": "禁用构建隔离"},
    {"key": "UV_NO_CACHE", "default": "0", "group": "uv",
     "desc_en": "Disable UV cache", "desc_zh": "禁用 UV 缓存"},
    {"key": "UV_LINK_MODE", "default": "symlink", "group": "uv",
     "desc_en": "UV link mode (symlink/copy)", "desc_zh": "UV 链接模式 (symlink/copy)"},
    {"key": "UV_INDEX_STRATEGY", "default": "unsafe-best-match", "group": "uv",
     "desc_en": "UV index resolution strategy", "desc_zh": "UV 索引解析策略"},
    # ── Network / Proxy ──
    {"key": "HTTP_PROXY", "default": "", "group": "network",
     "desc_en": "HTTP proxy (e.g. http://127.0.0.1:7890)", "desc_zh": "HTTP 代理 (如 http://127.0.0.1:7890)"},
    {"key": "HTTPS_PROXY", "default": "", "group": "network",
     "desc_en": "HTTPS proxy (e.g. http://127.0.0.1:7890)", "desc_zh": "HTTPS 代理 (如 http://127.0.0.1:7890)"},
]


def _defaults() -> Dict[str, str]:
    """返回所有环境变量的默认值字典"""
    return {d["key"]: d["default"] for d in ENV_VAR_DEFINITIONS}


def load_env_config() -> Dict[str, str]:
    """从 config/env_vars.json 加载环境变量配置，缺失项用默认值填充"""
    data = _defaults()
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                data.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return data


def save_env_config(data: Dict[str, str]):
    """保存环境变量配置到 config/env_vars.json"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_env_for_subprocess() -> Dict[str, str]:
    """返回需要注入子进程的环境变量（仅非空值）"""
    cfg = load_env_config()
    return {k: v for k, v in cfg.items() if v}
