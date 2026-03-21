"""原生控制台包装器

在独立控制台窗口中运行命令，保留 rich 的颜色 / 进度条 / rich_pixels 图片等终端特性。
同时将输出镜像到日志文件，供 GUI 侧异步读取。

使用方式（由 process_runner.py 自动调用）：
    python console_wrapper.py <exit_file> <log_file> <command...>
"""

import subprocess
import sys
import os
from pathlib import Path

# _color_inject/ 目录包含 sitecustomize.py，通过 PYTHONPATH 注入到子进程，
# 在子进程启动时自动替换 sys.stdout 为 _FakeTTY（isatty=True）并
# monkey-patch rich.console.Console 禁用 legacy_windows，让 rich 输出 ANSI 码。
_COLOR_INJECT_DIR = str(Path(__file__).parent / "_color_inject")


def _normalize_rich_color_system(value: str) -> str:
    """Normalize env-provided Rich color system names."""
    normalized = (value or "").strip().lower()
    aliases = {
        "24bit": "truecolor",
        "24-bit": "truecolor",
        "full": "truecolor",
        "256color": "256",
    }
    return aliases.get(normalized, normalized)


def _setup_windows_console() -> int:
    """配置 Windows 控制台: UTF-8 + ANSI 虚拟终端处理 + 调整窗口/缓冲区宽度

    Returns:
        实际设置的控制台列数（0 表示未能调整）
    """
    import ctypes

    kernel32 = ctypes.windll.kernel32
    # 设置控制台代码页为 UTF-8
    kernel32.SetConsoleOutputCP(65001)
    kernel32.SetConsoleCP(65001)
    # 启用 ANSI 转义码（虚拟终端处理）
    STD_OUTPUT_HANDLE = ctypes.c_ulong(-11)
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    mode = ctypes.c_ulong()
    kernel32.GetConsoleMode(handle, ctypes.byref(mode))
    mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
    kernel32.SetConsoleMode(handle, mode)

    # 调整控制台缓冲区列数
    cols = 0
    try:
        user32 = ctypes.windll.user32

        # 获取控制台字体尺寸以计算列数
        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class CONSOLE_FONT_INFO(ctypes.Structure):
            _fields_ = [("nFont", ctypes.c_ulong), ("dwFontSize", COORD)]

        font_info = CONSOLE_FONT_INFO()
        kernel32.GetCurrentConsoleFont(handle, False, ctypes.byref(font_info))
        char_w = font_info.dwFontSize.X or 8   # 默认回退 8px
        char_h = font_info.dwFontSize.Y or 16  # 默认回退 16px

        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            import subprocess as _sp

            # 获取系统 DPI 缩放倍率（96 = 100%，144 = 150%）
            try:
                dpi = user32.GetDpiForSystem()
            except Exception:
                dpi = 96
            scale = dpi / 96.0
            cols = max(120, int(120 * scale))

            _sp.run(f"mode con cols={cols}", shell=True,
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)

    except Exception:
        cols = 0  # 非关键，失败不影响功能

    return cols


def main():
    if len(sys.argv) < 4:
        print("Usage: console_wrapper.py <exit_file> <log_file> <command...>")
        sys.exit(1)

    exit_file = sys.argv[1]
    log_file = sys.argv[2]
    cmd = sys.argv[3:]

    if sys.platform == "win32":
        console_cols = _setup_windows_console()
    else:
        console_cols = 0

    # 构建子进程环境变量
    env = os.environ.copy()
    requested_color_system = _normalize_rich_color_system(env.get("_QINGLONG_RICH_COLOR_SYSTEM", ""))
    env["FORCE_COLOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if requested_color_system == "truecolor":
        env["COLORTERM"] = "truecolor"
        env["TERM"] = "xterm-256color"
    elif requested_color_system == "256":
        env.pop("COLORTERM", None)
        env["TERM"] = "xterm-256color"
    elif requested_color_system == "standard":
        env.pop("COLORTERM", None)
        env["TERM"] = "xterm"
    elif requested_color_system == "windows":
        env.pop("COLORTERM", None)
        env["TERM"] = "windows"
    elif requested_color_system == "auto":
        pass
    else:
        env["COLORTERM"] = "truecolor"
        env["TERM"] = "xterm-256color"

    # 告知 sitecustomize.py 日志文件路径（用于 _FakeTTY 镜像写入）
    env["_QINGLONG_LOG_FILE"] = log_file
    # 注入 _color_inject/ 到 PYTHONPATH 最前面，使 sitecustomize.py 被自动执行
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _COLOR_INJECT_DIR + os.pathsep + existing_pp if existing_pp else _COLOR_INJECT_DIR
    # 告知子进程实际终端列数，使 rich 等库按窗口宽度格式化输出
    if console_cols > 0:
        env["COLUMNS"] = str(console_cols)

    # 通过管道捕获子进程输出（ANSI 码由 sitecustomize 的 _FakeTTY 同步写到日志文件）
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )

    # 日志文件由子进程内 sitecustomize 的 _FakeTTY 直接写入（含 ANSI 码）
    # 这里只负责把 PIPE 输出转发到本控制台窗口（用于实时显示）
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            # 写到控制台（本窗口的 stdout 是真实终端，能渲染 ANSI 颜色）
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except Exception:
                pass
    except KeyboardInterrupt:
        proc.terminate()

    proc.wait()

    # 写入退出码信号文件，通知 GUI 脚本已完成
    try:
        with open(exit_file, "w", encoding="utf-8") as f:
            f.write(str(proc.returncode))
    except Exception:
        pass

    status = "成功" if proc.returncode == 0 else f"失败 (返回码: {proc.returncode})"
    print(f"\n{'=' * 50}")
    print(f"  任务{status}")
    print(f"  按 Enter 关闭此窗口...")
    print(f"{'=' * 50}")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
