"""原生控制台包装器

在独立控制台窗口中运行命令，保留 rich 的颜色 / 进度条 / rich_pixels 图片等终端特性。
同时将输出镜像到日志文件，供 GUI 侧异步读取。

使用方式（由 process_runner.py 自动调用）：
    python console_wrapper.py <exit_file> <log_file> <command...>
"""

import subprocess
import sys
import os


def _setup_windows_console():
    """配置 Windows 控制台: UTF-8 + ANSI 虚拟终端处理"""
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


def main():
    if len(sys.argv) < 4:
        print("Usage: console_wrapper.py <exit_file> <log_file> <command...>")
        sys.exit(1)

    exit_file = sys.argv[1]
    log_file = sys.argv[2]
    cmd = sys.argv[3:]

    if sys.platform == "win32":
        _setup_windows_console()

    # FORCE_COLOR 让 rich 即使在管道模式也输出 ANSI 颜色码
    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    # 通过管道捕获子进程输出，同时写到本控制台和日志文件
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )

    log_fh = None
    try:
        log_fh = open(log_file, "ab")
    except Exception:
        pass

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            # 写到控制台（本窗口的 stdout 是真实终端，能渲染 ANSI）
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            # 同步写到日志文件
            if log_fh:
                log_fh.write(chunk)
                log_fh.flush()
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        if log_fh:
            log_fh.close()

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
