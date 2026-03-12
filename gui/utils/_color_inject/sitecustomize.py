# -*- coding: utf-8 -*-
"""子进程颜色注入 (sitecustomize)

由 console_wrapper.py 通过 PYTHONPATH 自动注入，在所有业务代码之前执行。

功能：
  1. 替换 sys.stdout 为 _FakeTTY — isatty()=True，让 rich 认为在 TTY 环境
  2. Monkey-patch rich.console.Console.__init__，强制 legacy_windows=False，
     禁用 Windows Console API 渲染，让 rich 改为输出标准 ANSI 转义码
  3. _FakeTTY.write() 同时将输出镜像到 _QINGLONG_LOG_FILE 指定的日志文件
"""
import sys
import os

_log_path = os.environ.get("_QINGLONG_LOG_FILE", "")


class _FakeTTY:
    """isatty()=True 的流，内容同步写到原始 stdout buffer 和日志文件"""

    def __init__(self, raw_buf, log_path: str):
        self._raw = raw_buf
        self._log = open(log_path, "ab") if log_path else None
        self.encoding = "utf-8"
        self.errors = "replace"
        self.softspace = 0

    def isatty(self) -> bool:
        return True

    def write(self, s) -> int:
        b = s.encode("utf-8", errors="replace") if isinstance(s, str) else bytes(s)
        self._raw.write(b)
        self._raw.flush()
        if self._log:
            self._log.write(b)
            self._log.flush()
        return len(s)

    def flush(self):
        self._raw.flush()
        if self._log:
            self._log.flush()

    def fileno(self) -> int:
        return self._raw.fileno()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


# 替换 sys.stdout（仅在有日志文件时；若无则保持原样，不影响交互式使用）
if _log_path:
    try:
        _tee = _FakeTTY(sys.stdout.buffer, _log_path)
        sys.stdout = _tee
    except Exception:
        pass


# Monkey-patch rich.console.Console — 禁用 legacy_windows 模式
# Windows 上 rich 默认 legacy_windows=True，使用 SetConsoleTextAttribute API
# 而不输出 ANSI 码。此处强制 legacy_windows=False 让 rich 改用 ANSI 输出。
def _patch_rich():
    try:
        import rich.console as _rc

        _orig = _rc.Console.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.setdefault("legacy_windows", False)
            _orig(self, *args, **kwargs)

        _rc.Console.__init__ = _patched_init
    except Exception:
        pass


_patch_rich()
