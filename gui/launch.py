#!/usr/bin/env python3
"""
青龙字幕工具 GUI 启动脚本
支持 Windows 和 Linux

使用方法:
    cd /path/to/qinglong-captions
    python -m gui.launch

可选参数:
    --host    绑定地址 (默认: 127.0.0.1)
    --port    端口 (默认: 8080)
    --cloud   云模式 (绑定 0.0.0.0)
"""

import sys
import argparse
import os
from pathlib import Path

# 获取项目根目录（当前文件的上级目录）
project_root = Path(__file__).parent.parent.resolve()

# 设置 Python 路径 - 必须在导入其他模块之前
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gui.path_setup import configure_sys_path

configure_sys_path(project_root)

os.chdir(project_root)

# 解析参数
parser = argparse.ArgumentParser(description="启动青龙字幕工具 GUI")
parser.add_argument("--host", type=str, default="127.0.0.1", help="绑定地址")
parser.add_argument("--port", type=int, default=8080, help="端口")
parser.add_argument("--cloud", action="store_true", help="云模式 (绑定 0.0.0.0)")
parser.add_argument("--native", action="store_true", help="原生窗口模式 (需要安装 pywebview)")
parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
args = parser.parse_args()

if args.cloud:
    args.host = "0.0.0.0"

# 导入并运行 GUI
from nicegui import ui, app
from main import (
    APP_TITLE,
    home_page,
    import_page,
    split_page,
    tagger_page,
    caption_page,
    export_page,
    tools_page,
    setup_page,
    console_page,
    not_found_page,  # 404 页面
)
from theme import apply_theme

# 设置页面路由 - 所有页面
ui.page("/")(home_page)
ui.page("/import")(import_page)
ui.page("/split")(split_page)
ui.page("/tagger")(tagger_page)
ui.page("/caption")(caption_page)
ui.page("/export")(export_page)
ui.page("/tools")(tools_page)
ui.page("/setup")(setup_page)
ui.page("/console")(console_page)

# 404 页面 - 捕获所有未定义的路由（如 /cache, /train 等）
ui.page("/{path:path}")(not_found_page)

# 应用主题
apply_theme()

# 设置应用标题
app.config.title = APP_TITLE


def find_available_port(start_port, max_attempts=10):
    """查找可用的端口"""
    import socket

    for i in range(max_attempts):
        port = start_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return None


def print_runtime_snapshot(actual_port: int) -> None:
    env_name = Path(os.environ.get("VIRTUAL_ENV", "")).name or (".venv" if (project_root / ".venv").exists() else "unknown")
    tracked_env_keys = [
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "HF_HOME",
        "HF_ENDPOINT",
        "XFORMERS_FORCE_DISABLE_TRITON",
        "CUDA_HOME",
        "UV_CACHE_DIR",
        "UV_EXTRA_INDEX_URL",
        "UV_NO_BUILD_ISOLATION",
        "UV_NO_CACHE",
        "UV_LINK_MODE",
    ]

    print(f"=" * 60)
    print(f"  {APP_TITLE} GUI")
    print(f"=" * 60)
    print(f"  Language: {get_i18n().lang}")
    print(f"  URL: http://{args.host}:{actual_port}")
    print(f"  Reload: False")
    print(f"  Python: {sys.executable}")
    print(f"  Environment: {env_name}")
    print(f"  Project Root: {project_root}")
    if args.native:
        print(f"  Mode: Native window (原生窗口模式)")
    else:
        print(f"  Mode: Web browser (网页浏览器模式)")
    if not args.no_browser and not args.native:
        print(f"  将自动打开浏览器...")
    elif not args.native:
        print(f"  请手动在浏览器中打开上述 URL")
    print(f"  Environment Variables:")
    for key in tracked_env_keys:
        print(f"    {key}={os.environ.get(key, '')}")
    print(f"=" * 60)


if __name__ == "__main__":
    # 初始化语言
    from gui.utils.i18n import t, get_i18n

    # 查找可用端口
    actual_port = find_available_port(args.port)
    if actual_port is None:
        print(f"❌ 错误: 无法找到可用端口 (尝试了 {args.port} 到 {args.port + 9})")
        exit(1)

    if actual_port != args.port:
        print(f"⚠️  端口 {args.port} 被占用，自动切换到端口 {actual_port}")

    print_runtime_snapshot(actual_port)

    # 显式设置 native=False 确保使用浏览器模式（除非用户指定 --native）
    run_kwargs = {
        "title": APP_TITLE,
        "favicon": "🐉",
        "dark": True,
        "reload": False,
        "host": args.host,
        "port": actual_port,
        "show": not args.no_browser,
        "native": bool(args.native),  # 强制转换为布尔值
    }

    # 只有在原生窗口模式下才添加 window_size
    if args.native:
        run_kwargs["window_size"] = (1400, 900)

    ui.run(**run_kwargs)
