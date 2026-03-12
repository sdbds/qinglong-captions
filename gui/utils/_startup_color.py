# -*- coding: utf-8 -*-
"""子进程颜色注入模块 (sitecustomize)

由 console_wrapper.py 通过 PYTHONPATH 注入，在所有业务脚本之前执行。
功能：
  1. 替换 sys.stdout 为 _FakeTTY，使 rich 的 isatty() 检测返回 True
  2. Monkey-patch rich.console.Console.__init__，强制 legacy_windows=False，
     禁用 Windows Console API 渲染模式，让 rich 输出标准 ANSI 颜色码

不需要修改任何业务脚本。
"""
import sys
import os

_log_path = os.environ.get("_QINGLONG_LOG_FILE", "")


# ── 1. 替换 sys.stdout 为伪 TTY ──────────────────────────────────────────────
class _FakeTTY:
    """isatty()=True 的流，把内容同时写到原始 stdout buffer 和日志文件"""

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


if _log_path:
    try:
        _tee = _FakeTTY(sys.stdout.buffer, _log_path)
        sys.stdout = _tee
    except Exception:
        pass


# ── 2. Monkey-patch rich.console.Console，禁用 legacy_windows ───────────────
# Windows 上 rich 默认 legacy_windows=True，改用 SetConsoleTextAttribute API，
# 不输出 ANSI 码。必须在 rich 模块被导入前完成 patch。
# 此文件作为 sitecustomize 执行，早于一切业务模块导入，时机合适。
def _patch_rich():
    try:
        import rich.console as _rc

        _orig = _rc.Console.__init__

        def _new_init(self, *args, **kwargs):
            kwargs.setdefault("legacy_windows", False)
            _orig(self, *args, **kwargs)

        _rc.Console.__init__ = _new_init
    except Exception:
        pass


_patch_rich()
