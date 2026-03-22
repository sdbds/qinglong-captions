"""进程运行工具 - 管理外部 Python 脚本的调用和日志输出

依赖策略与共享 .venv 的 PowerShell 脚本保持一致:
  1. 先按功能往共享环境里打补丁
  2. 再执行目标脚本

关键：使用 asyncio.create_subprocess_exec 进行非阻塞 I/O，
避免阻塞 NiceGUI 事件循环导致 WebSocket 断开。
"""

import asyncio
import io
import json
import re
import shutil
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional
from dataclasses import dataclass
from enum import Enum

from rich.console import Console

from gui.utils.log_buffer import log_buffer as _global_log_buffer, LogBuffer
from gui.utils.ansi_to_html import strip_ansi
from utils.console_util import print_exception


_TRANSIENT_SPINNER_FRAMES = frozenset("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
_PROGRESS_PERCENT_RE = re.compile(r"\d{1,3}%")
_PROGRESS_BAR_RE = re.compile(r"[#=\-━─█▉▊▋▌▍▎▏]{6,}")
_PROGRESS_BYTES_RE = re.compile(
    r"\b\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*(?:B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\b",
    re.IGNORECASE,
)
_PROGRESS_TIME_RE = re.compile(r"(?:\b(?:\d+:)?\d{1,2}:\d{2}\b|-:--:--|-:--)")
_LEADING_NON_SGR_ANSI_RE = re.compile(
    r"^(?:(?:\x1b\[[0-9;?]*[A-LN-Za-ln-z])|(?:\x1b\][^\x07]*\x07))+"
)
_UV_TORCH_EXTRAS = frozenset(
    {
        "translate",
        "wdtagger",
        "moondream",
        "olmocr",
        "deepseek-ocr",
        "logics-ocr",
        "hunyuan-ocr",
        "glm-ocr",
        "nanonets-ocr",
        "firered-ocr",
        "lighton-ocr",
        "dots-ocr",
        "qianfan-ocr",
        "chandra-ocr",
        "qwen-vl-local",
        "step-vl-local",
        "penguin-vl-local",
        "reka-edge-local",
        "lfm-vl-local",
        "music-flamingo-local",
        "eureka-audio-local",
        "acestep-transcriber-local",
    }
)
_UV_TORCHVISION_EXTRAS = frozenset(
    {
        "translate",
        "wdtagger",
        "olmocr",
        "deepseek-ocr",
        "logics-ocr",
        "hunyuan-ocr",
        "glm-ocr",
        "firered-ocr",
        "lighton-ocr",
        "dots-ocr",
        "qianfan-ocr",
        "chandra-ocr",
        "qwen-vl-local",
        "step-vl-local",
        "penguin-vl-local",
        "reka-edge-local",
    }
)
_UV_TORCH_GROUPS = frozenset({"test"})
_UV_TORCHVISION_GROUPS = frozenset()

# 序列化 _patch_shared_environment，防止两个并发 Job 同时修改共享 .venv
_uv_patch_lock: Optional[asyncio.Lock] = None


def _get_uv_patch_lock() -> asyncio.Lock:
    """延迟初始化 uv patch 锁（需在事件循环中调用）"""
    global _uv_patch_lock
    if _uv_patch_lock is None:
        _uv_patch_lock = asyncio.Lock()
    return _uv_patch_lock


def _render_exception_text(exc: BaseException, prefix: Optional[str] = None, *, summary_style: str = "red") -> str:
    """Render exception details via the shared Rich helper for log-buffer output."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None)
    print_exception(console, exc, prefix=prefix, summary_style=summary_style)
    return buffer.getvalue().strip()


class ProcessStatus(Enum):
    IDLE = "空闲"
    RUNNING = "运行中"
    SUCCESS = "成功"
    ERROR = "失败"


@dataclass
class ProcessResult:
    status: ProcessStatus
    return_code: int = 0
    message: str = ""


# 脚本路径映射: module_key -> (script_path, default_extra | None)
# script_path 相对于项目根目录
SCRIPT_REGISTRY = {
    # step 1 - 数据集导入
    "module.lanceImport": ("./module/lanceImport.py", None),
    # step 2 - 视频分割
    "module.videospilter": ("./module/videospilter.py", None),
    # step 3 - 打标
    "utils.wdtagger": ("./utils/wdtagger.py", "wdtagger"),
    # step 4 - 字幕生成
    "module.captioner": ("./module/captioner.py", None),
    # step 5 - 导出
    "module.lanceexport": ("./module/lanceexport.py", None),
    # step 6 - 工具
    "module.waterdetect": ("./module/waterdetect.py", None),
    "module.audio_separator": ("./module/audio_separator.py", None),
    "utils.preprocess_datasets": ("./utils/preprocess_datasets.py", None),
    "module.rewardmodel": ("./module/rewardmodel.py", None),
    "module.texttranslate": ("./module/texttranslate.py", "translate"),
}

# console_wrapper.py 的绝对路径
_WRAPPER_PATH = str(Path(__file__).parent / "console_wrapper.py")


def _ps_escape(s: str) -> str:
    """对字符串做 PowerShell 单引号转义（' → ''）"""
    return str(s).replace("'", "''")


def _powershell_call(parts: List[str]) -> str:
    """将命令参数列表转为 PowerShell 的调用表达式"""
    return "& " + " ".join(f"'{_ps_escape(part)}'" for part in parts)


class ProcessRunner:
    """运行外部进程并捕获输出（非阻塞）

    使用 asyncio.create_subprocess_exec 保证不阻塞 NiceGUI 事件循环。
    """

    TASK_DIVIDER = "================ New Task ================"

    # 项目根目录 (gui/utils/../../ = 项目根)
    PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())

    def __init__(self, log_buffer: Optional[LogBuffer] = None):
        """
        Args:
            log_buffer: 可选，指定此实例的日志缓冲区。
                        若未提供，使用全局 log_buffer（向后兼容）。
        """
        self._log_buffer: LogBuffer = log_buffer if log_buffer is not None else _global_log_buffer
        self.process = None
        self.status_callback: Optional[Callable[[ProcessStatus], None]] = None
        self._running = False
        self._tail_task: Optional[asyncio.Task] = None
        self._tail_line = ""
        self._tail_pending_cr = False
        self._tail_line_overwritten = False
        self._task_divider_emitted = False

    def set_callbacks(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[ProcessStatus], None]] = None,
    ):
        """设置回调函数（log_callback 已废弃，保留参数以兼容旧调用方）"""
        self.status_callback = status_callback

    def _notify_log(self, message: str):
        """推送日志到此实例的 log_buffer（订阅者自动收到）"""
        self._log_buffer.push(message)

    def _notify_status(self, status: ProcessStatus):
        """通知状态回调"""
        if self.status_callback:
            self.status_callback(status)

    def begin_task_log(self):
        """在保留历史的前提下，为新任务插入分隔线。"""
        if self._task_divider_emitted:
            return
        if self._log_buffer.get_all_lines():
            self._notify_log(self.TASK_DIVIDER)
        self._task_divider_emitted = True

    def log(self, message: str):
        """推送一条日志到共享缓冲区和当前回调。"""
        self._notify_log(message)

    def _build_env(self, env_vars: Optional[dict] = None) -> dict:
        """构建子进程环境变量（与 PowerShell 脚本保持一致）

        优先级: env_vars 参数 > env_config (GUI 配置) > 系统环境
        """
        from gui.utils.env_config import get_env_for_subprocess

        env = os.environ.copy()

        # 注入 GUI 环境变量配置（来自 config/env_vars.json）
        env.update(get_env_for_subprocess())

        # 调用方传入的 env_vars 优先级最高
        if env_vars:
            env.update(env_vars)

        # PYTHONPATH
        existing = env.get("PYTHONPATH", "")
        if self.PROJECT_ROOT not in existing:
            env["PYTHONPATH"] = self.PROJECT_ROOT + os.pathsep + existing if existing else self.PROJECT_ROOT

        return env

    @staticmethod
    def _normalize_console_color_system(color_system: Optional[str]) -> Optional[str]:
        """规范化原生控制台颜色系统参数。"""
        if color_system is None:
            return None

        value = str(color_system).strip().lower()
        if not value:
            return None

        aliases = {
            "24bit": "truecolor",
            "24-bit": "truecolor",
            "full": "truecolor",
            "256color": "256",
        }
        value = aliases.get(value, value)

        if value not in {"auto", "standard", "256", "truecolor", "windows"}:
            raise ValueError(f"Unsupported console_color_system: {color_system}")

        return value

    @classmethod
    def _build_native_wrapper_env(cls, env: dict, console_color_system: Optional[str]) -> dict:
        """为原生控制台包装进程附加渲染控制环境变量。"""
        wrapper_env = env.copy()
        normalized = cls._normalize_console_color_system(console_color_system)
        if normalized is None:
            wrapper_env.pop("_QINGLONG_RICH_COLOR_SYSTEM", None)
        else:
            wrapper_env["_QINGLONG_RICH_COLOR_SYSTEM"] = normalized
        return wrapper_env

    @staticmethod
    def _find_uv() -> Optional[str]:
        """查找 uv 可执行文件路径"""
        return shutil.which("uv")

    @staticmethod
    def _detect_project_env_name(work_dir: Path, env: dict) -> str:
        for name in (".venv", "venv"):
            if (work_dir / name).exists():
                return name
        venv_path = env.get("VIRTUAL_ENV")
        if venv_path:
            return Path(venv_path).name or venv_path
        return "uv-managed"

    @staticmethod
    def _collect_uv_profiles(extra_name: Optional[str], uv_extra_args: Optional[List[str]] = None) -> tuple[list[str], list[str]]:
        extras: list[str] = []
        groups: list[str] = []
        seen_extras: set[str] = set()
        seen_groups: set[str] = set()

        def add_value(bucket: list[str], seen: set[str], value: Optional[str]) -> None:
            if not value or value in seen:
                return
            seen.add(value)
            bucket.append(value)

        add_value(extras, seen_extras, extra_name)

        args_iter = iter(uv_extra_args or [])
        for arg in args_iter:
            if arg == "--extra":
                add_value(extras, seen_extras, next(args_iter, None))
                continue
            if arg == "--group":
                add_value(groups, seen_groups, next(args_iter, None))
                continue
            if arg.startswith("--extra="):
                add_value(extras, seen_extras, arg.split("=", 1)[1])
                continue
            if arg.startswith("--group="):
                add_value(groups, seen_groups, arg.split("=", 1)[1])

        return extras, groups

    @staticmethod
    def _profile_parts(extras: list[str], groups: list[str]) -> list[str]:
        parts = ["default"]
        parts.extend(f"extra:{name}" for name in extras)
        parts.extend(f"group:{name}" for name in groups)
        return parts

    @staticmethod
    def _uv_index_strategy(env: dict) -> str:
        return str(env.get("UV_INDEX_STRATEGY", "")).strip() or "unsafe-best-match"

    @staticmethod
    def _resolve_project_python(work_dir: Path, env: dict) -> Optional[str]:
        candidates: list[Path] = []

        for name in (".venv", "venv"):
            base = work_dir / name
            if sys.platform == "win32":
                python_path = base / "Scripts" / "python.exe"
            else:
                python_path = base / "bin" / "python"
            candidates.append(python_path)

        venv_path = env.get("VIRTUAL_ENV")
        if venv_path:
            venv_base = Path(venv_path)
            if sys.platform == "win32":
                candidates.append(venv_base / "Scripts" / "python.exe")
            else:
                candidates.append(venv_base / "bin" / "python")

        candidates.append(Path(sys.executable))

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _needs_sync_patch(extras: list[str]) -> bool:
        return "paddleocr" in extras

    @staticmethod
    def _profile_uses_torch(extras: list[str], groups: list[str]) -> bool:
        return any(extra in _UV_TORCH_EXTRAS for extra in extras) or any(group in _UV_TORCH_GROUPS for group in groups)

    @staticmethod
    def _profile_uses_torchvision(extras: list[str], groups: list[str]) -> bool:
        return any(extra in _UV_TORCHVISION_EXTRAS for extra in extras) or any(
            group in _UV_TORCHVISION_GROUPS for group in groups
        )

    @staticmethod
    def _infer_uv_torch_backend(env: dict) -> Optional[str]:
        backend = str(env.get("UV_TORCH_BACKEND", "")).strip()
        if backend:
            return backend

        for key in ("UV_EXTRA_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
            value = str(env.get(key, "")).strip()
            if not value:
                continue
            match = re.search(r"/(cu\d+)(?:/|$)", value)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _inspect_installed_torch_backend(python_path: Optional[str], env: dict) -> Optional[str]:
        if not python_path:
            return None

        probe = (
            "import json\n"
            "try:\n"
            " import torch\n"
            " print(json.dumps({'version': getattr(torch, '__version__', ''), 'cuda': getattr(torch.version, 'cuda', None)}))\n"
            "except Exception as exc:\n"
            " print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))\n"
        )

        try:
            result = subprocess.run(
                [python_path, "-c", probe],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                check=False,
            )
        except OSError:
            return None

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None

        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            return None

        version = str(payload.get("version", "")).strip()
        suffix_match = re.search(r"\+(cpu|cu\d+)$", version)
        if suffix_match:
            return suffix_match.group(1)

        cuda_version = str(payload.get("cuda", "") or "").strip()
        if cuda_version:
            digits = re.sub(r"\D", "", cuda_version)
            if digits:
                return f"cu{digits}"
        return None

    def _reset_tail_state(self):
        """重置原生控制台日志解析状态。"""
        self._tail_line = ""
        self._tail_pending_cr = False
        self._tail_line_overwritten = False

    @staticmethod
    def _is_transient_console_update(line: str) -> bool:
        """判断是否为仅用于终端刷新的瞬时进度帧。"""
        plain = strip_ansi(line).strip()
        if not plain:
            return True
        if plain[0] in _TRANSIENT_SPINNER_FRAMES:
            return True
        if _PROGRESS_PERCENT_RE.search(plain) and (
            _PROGRESS_BAR_RE.search(plain)
            or _PROGRESS_BYTES_RE.search(plain)
            or _PROGRESS_TIME_RE.search(plain)
        ):
            return True
        return False

    def _emit_tail_line(self, lines: list[str], *, final_flush: bool = False):
        """提交一行稳定日志，跳过 spinner 这类瞬时刷新。"""
        raw = self._tail_line.rstrip()
        overwritten = self._tail_line_overwritten
        self._tail_line = ""
        self._tail_line_overwritten = False

        if not raw:
            return

        if self._is_transient_console_update(raw) and (overwritten or final_flush):
            return

        normalized = _LEADING_NON_SGR_ANSI_RE.sub("", raw)
        if not strip_ansi(normalized).strip():
            return

        lines.append(normalized)

    def _publish_tailed_line(self, raw: str):
        """将稳定日志推送到此实例的 log_buffer（订阅者自动收到）。"""
        if not raw:
            return
        self._log_buffer.push(raw)

    def _consume_native_log_chunk(self, text: str, *, final_flush: bool = False) -> list[str]:
        """按终端语义解析原生控制台输出。

        裸 `\r` 表示覆盖当前行，不应当当成一条新日志。
        """
        committed: list[str] = []
        index = 0

        while index < len(text):
            char = text[index]

            if self._tail_pending_cr:
                self._tail_pending_cr = False
                if char == "\n":
                    self._emit_tail_line(committed)
                    index += 1
                    continue
                self._tail_line = ""
                self._tail_line_overwritten = True
                continue

            if char == "\r":
                self._tail_pending_cr = True
            elif char == "\n":
                self._emit_tail_line(committed)
            else:
                self._tail_line += char

            index += 1

        if final_flush:
            if self._tail_pending_cr:
                self._tail_pending_cr = False
                self._tail_line = ""
                self._tail_line_overwritten = True
            self._emit_tail_line(committed, final_flush=True)

        return committed

    # ------------------------------------------------------------------
    #  非阻塞读取子进程输出
    # ------------------------------------------------------------------
    async def _stream_output(self, proc: asyncio.subprocess.Process):
        """非阻塞逐行读取 stdout 并回调日志"""
        assert proc.stdout is not None
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            try:
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
            except Exception:
                line = str(line_bytes)
            self._notify_log(line)

    async def _stream_output_popen(self, proc: subprocess.Popen):
        """在 selector event loop 下通过线程读取 Popen 输出。"""
        assert proc.stdout is not None
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)
            if not line:
                break
            self._notify_log(str(line).rstrip())

    @staticmethod
    def _requires_threaded_subprocess() -> bool:
        if sys.platform != "win32":
            return False
        return type(asyncio.get_event_loop_policy()).__name__ == "WindowsSelectorEventLoopPolicy"

    async def _run_pipe_with_popen(self, cmd: List[str], work_dir: Path, env: dict) -> int:
        """Windows reload 模式下，避免 asyncio 子进程不可用。"""
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(work_dir),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        await self._stream_output_popen(self.process)
        return await asyncio.to_thread(self.process.wait)

    async def _run_logged_subprocess(self, cmd: List[str], work_dir: Path, env: dict) -> int:
        """在 GUI 日志中同步显示命令输出。"""
        if self._requires_threaded_subprocess():
            self._notify_log("检测到 Windows reload 模式，使用线程子进程回退")
            return await self._run_pipe_with_popen(cmd, work_dir, env)

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(work_dir),
                env=env,
            )
        except NotImplementedError:
            self._notify_log("当前事件循环不支持 asyncio 子进程，回退到线程子进程模式")
            return await self._run_pipe_with_popen(cmd, work_dir, env)

        await self._stream_output(self.process)
        return await self.process.wait()

    async def _patch_shared_environment(
        self,
        uv: str,
        work_dir: Path,
        env: dict,
        env_name: str,
        extras: list[str],
        groups: list[str],
    ) -> Optional[ProcessResult]:
        if not extras and not groups:
            return None

        # 序列化依赖补丁：防止多个并发 Job 同时修改共享 .venv
        async with _get_uv_patch_lock():
            target_python = self._resolve_project_python(work_dir, env)
            profile_parts = self._profile_parts(extras, groups)

            if self._needs_sync_patch(extras):
                uninstall_cmd = [uv, "pip", "uninstall"]
                if target_python:
                    uninstall_cmd.extend(["--python", target_python])
                uninstall_cmd.extend(["-y", "torch", "torchvision", "torchaudio"])
                uninstall_action = "uv pip uninstall"
                self._notify_log("检测到 paddleocr，先卸载共享环境中的 torch / torchvision / torchaudio")
                self._notify_log(f"{uninstall_action} target environment: {env_name}")
                self._notify_log(f"{uninstall_action} dependency profile: {', '.join(profile_parts)}")
                self._notify_log(
                    f"开始同步依赖: {' '.join(uninstall_cmd[:15])}{'...' if len(uninstall_cmd) > 15 else ''}"
                )

                uninstall_code = await self._run_logged_subprocess(uninstall_cmd, work_dir, env)
                self.process = None
                if uninstall_code != 0:
                    self._notify_log("torch 栈卸载命令返回非零，继续安装 paddleocr 当前依赖")

            cmd = [uv, "pip", "install", "--no-build-isolation"]
            action = "uv pip install"
            cmd.extend(["--index-strategy", self._uv_index_strategy(env)])
            if target_python:
                cmd.extend(["--python", target_python])
            has_torch = self._profile_uses_torch(extras, groups)
            has_torchvision = self._profile_uses_torchvision(extras, groups)
            torch_backend = self._infer_uv_torch_backend(env) if has_torch else None
            if torch_backend:
                cmd.extend(["--torch-backend", torch_backend])
                installed_torch_backend = self._inspect_installed_torch_backend(target_python, env)
                if installed_torch_backend and installed_torch_backend != torch_backend:
                    cmd.extend(["--reinstall-package", "torch"])
                    if has_torchvision:
                        cmd.extend(["--reinstall-package", "torchvision"])
                    self._notify_log(
                        f"检测到当前环境 torch backend 为 {installed_torch_backend}，将重装为 {torch_backend}"
                    )
                else:
                    self._notify_log(f"检测到 torch 依赖，使用 uv torch backend={torch_backend}")
            cmd.extend(["-r", "pyproject.toml"])
            for extra in extras:
                cmd.extend(["--extra", extra])
            for group in groups:
                cmd.extend(["--group", group])
            self._notify_log("使用共享 .venv 增量安装依赖补丁")
            self._notify_log("直接使用 uv pip install -r pyproject.toml 安装当前依赖 profile")

            self._notify_log(f"{action} target environment: {env_name}")
            self._notify_log(f"{action} dependency profile: {', '.join(profile_parts)}")
            self._notify_log(f"开始同步依赖: {' '.join(cmd[:15])}{'...' if len(cmd) > 15 else ''}")

            return_code = await self._run_logged_subprocess(cmd, work_dir, env)
            self.process = None

            if return_code != 0:
                message = f"{action} 失败，返回码: {return_code}"
                self._notify_log(message)
                return ProcessResult(ProcessStatus.ERROR, return_code, message)

            return None

    # ------------------------------------------------------------------
    #  日志文件尾随（原生控制台模式）
    # ------------------------------------------------------------------
    async def _tail_log_file(self, log_file: str):
        """异步尾随日志文件，原始 ANSI 推送到 log_buffer，纯文本回传 GUI"""
        offset = 0
        self._reset_tail_state()
        try:
            while self._running:
                try:
                    if os.path.exists(log_file):
                        with open(log_file, "rb") as f:
                            f.seek(offset)
                            new_data = f.read()
                            if new_data:
                                offset += len(new_data)
                                text = new_data.decode("utf-8", errors="replace")
                                for line in self._consume_native_log_chunk(text):
                                    self._publish_tailed_line(line.rstrip())
                except Exception:
                    pass
                await asyncio.sleep(0.5)

            if os.path.exists(log_file):
                with open(log_file, "rb") as f:
                    f.seek(offset)
                    remaining = f.read()
                offset += len(remaining)
                text = remaining.decode("utf-8", errors="replace") if remaining else ""
                for line in self._consume_native_log_chunk(text, final_flush=True):
                    self._publish_tailed_line(line.rstrip())
            else:
                for line in self._consume_native_log_chunk("", final_flush=True):
                    self._publish_tailed_line(line.rstrip())
        except asyncio.CancelledError:
            for line in self._consume_native_log_chunk("", final_flush=True):
                self._publish_tailed_line(line.rstrip())

    # ------------------------------------------------------------------
    #  主要运行方法
    # ------------------------------------------------------------------
    async def run_python_script(
        self,
        script_key: str,
        args: List[str],
        cwd: Optional[str] = None,
        env_vars: Optional[dict] = None,
        uv_extra: Optional[str] = None,
        uv_extra_args: Optional[List[str]] = None,
        native_console: bool = True,
        console_color_system: Optional[str] = "truecolor",
    ) -> ProcessResult:
        """运行 Python 脚本（完全非阻塞）

        Args:
            script_key: 脚本标识，如 'utils.wdtagger'、'module.captioner'。
            args: 传递给脚本的参数列表。
            cwd: 工作目录 (默认项目根目录)。
            env_vars: 额外环境变量。
            uv_extra: 强制指定 pyproject optional dependency extra。
            uv_extra_args: 额外的依赖 profile 参数（如 extra/group）。
            native_console: Windows 下使用原生控制台窗口（支持 rich 颜色/图片）。
            console_color_system: Rich 控制台颜色系统，可选 auto/standard/256/truecolor/windows。
        """
        if self._running:
            return ProcessResult(ProcessStatus.ERROR, -1, "已有任务在运行")

        self._running = True
        self._notify_status(ProcessStatus.RUNNING)
        self.begin_task_log()

        use_native = native_console and sys.platform == "win32"
        exit_file = ""
        log_file = ""

        try:
            work_dir = Path(cwd) if cwd else Path(self.PROJECT_ROOT)
            env = self._build_env(env_vars)
            if not use_native:
                env["PYTHONIOENCODING"] = "utf-8"
                env["FORCE_COLOR"] = "1"
                env["COLORTERM"] = "truecolor"
                env["TERM"] = "xterm-256color"
                env["PYTHONUNBUFFERED"] = "1"

            # 从 registry 查找脚本路径和默认依赖
            registry_entry = SCRIPT_REGISTRY.get(script_key)
            if registry_entry:
                script_path, default_extra = registry_entry
            else:
                script_path = "./" + script_key.replace(".", "/") + ".py"
                default_extra = None

            extra_name = uv_extra or default_extra
            extras, groups = self._collect_uv_profiles(extra_name, uv_extra_args)
            profile_parts = self._profile_parts(extras, groups)
            env_name = self._detect_project_env_name(work_dir, env)
            runtime_python = self._resolve_project_python(work_dir, env)

            # Step 1: 构建运行命令
            uv = self._find_uv()
            if uv:
                patch_result = await self._patch_shared_environment(uv, work_dir, env, env_name, extras, groups)
                if patch_result:
                    self._running = False
                    self._notify_status(ProcessStatus.ERROR)
                    return patch_result

            if runtime_python:
                cmd = [runtime_python, script_path]
            elif uv:
                cmd = [uv, "run", script_path]
            else:
                if extras or groups:
                    self._notify_log("未找到 uv，无法自动安装依赖补丁，将直接尝试运行脚本")
                cmd = [sys.executable, script_path]

            cmd.extend(args)

            self._notify_log(f"runtime target environment: {env_name}")
            self._notify_log(f"runtime dependency profile: {', '.join(profile_parts)}")
            self._notify_log(f"开始执行: {' '.join(cmd[:15])}{'...' if len(cmd) > 15 else ''}")
            self._notify_log(f"工作目录: {work_dir.absolute()}")
            self._notify_log("=" * 60)

            # Step 2: 启动进程
            if use_native:
                return_code = await self._run_native(cmd, work_dir, env, console_color_system)
            else:
                return_code = await self._run_logged_subprocess(cmd, work_dir, env)

            self._running = False

            if return_code == 0:
                self._notify_status(ProcessStatus.SUCCESS)
                return ProcessResult(ProcessStatus.SUCCESS, return_code, "执行成功")
            else:
                self._notify_status(ProcessStatus.ERROR)
                return ProcessResult(ProcessStatus.ERROR, return_code, f"进程返回错误码: {return_code}")

        except Exception as e:
            self._running = False
            self._notify_status(ProcessStatus.ERROR)
            detail = str(e) or repr(e)
            error_msg = f"执行出错: {type(e).__name__}: {detail}"
            self._notify_log(error_msg)
            self._notify_log(_render_exception_text(e, prefix="执行出错"))
            return ProcessResult(ProcessStatus.ERROR, -1, error_msg)
        finally:
            self._task_divider_emitted = False
            self.process = None
            if self._tail_task and not self._tail_task.done():
                self._tail_task.cancel()
                self._tail_task = None

    # ------------------------------------------------------------------
    #  原生控制台模式
    # ------------------------------------------------------------------
    async def _run_native(
        self,
        cmd: List[str],
        work_dir: Path,
        env: dict,
        console_color_system: Optional[str] = "truecolor",
    ) -> int:
        """通过 console_wrapper.py 在原生控制台中运行命令。

        特性:
          - 窗口执行完毕后不会自动关闭，用户可查看输出
          - 输出同时镜像到日志文件，GUI 异步尾随并显示（纯文本）
          - 通过信号文件获取真实退出码
        """
        # 创建临时文件路径
        tmp_dir = tempfile.gettempdir()
        exit_file = os.path.join(tmp_dir, f"qinglong_exit_{os.getpid()}.tmp")
        log_file = os.path.join(tmp_dir, f"qinglong_log_{os.getpid()}.tmp")

        # 清理可能残留的旧文件
        for f in (exit_file, log_file):
            try:
                os.unlink(f)
            except OSError:
                pass

        # 包装命令: 通过 PowerShell 启动 console_wrapper.py
        # 优先使用 pwsh.exe (PowerShell 7)，回退到 powershell.exe (5.1)
        parts = [sys.executable, _WRAPPER_PATH, exit_file, log_file] + cmd
        ps_exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
        ps_cmd = _powershell_call(parts)
        wrapper_cmd = [ps_exe, "-NoProfile", "-NoLogo", "-Command", ps_cmd]
        wrapper_env = self._build_native_wrapper_env(env, console_color_system)

        self.process = subprocess.Popen(
            wrapper_cmd,
            cwd=str(work_dir),
            env=wrapper_env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        self._notify_log("已在 PowerShell 控制台窗口中启动，输出同步显示在下方")

        # 启动日志尾随任务
        self._tail_task = asyncio.create_task(self._tail_log_file(log_file))

        # 等待脚本完成（轮询信号文件）
        while not os.path.exists(exit_file):
            if not self._running:
                break  # 被用户终止
            await asyncio.sleep(0.5)

        # 读取退出码
        if os.path.exists(exit_file):
            try:
                with open(exit_file, "r", encoding="utf-8") as f:
                    return_code = int(f.read().strip())
            except (ValueError, OSError):
                return_code = -1
        else:
            return_code = -1

        # 停止日志尾随，做最后一次读取
        if self._tail_task and not self._tail_task.done():
            self._tail_task.cancel()
            self._tail_task = None
        # 最终读取确保所有日志都被捕获
        await asyncio.sleep(0.2)
        if os.path.exists(log_file):
            try:
                # 一次性读取剩余内容
                pass  # tail 已经在 cancel 前读完了大部分
            except Exception:
                pass

        # 清理临时文件
        for f in (exit_file, log_file):
            try:
                os.unlink(f)
            except OSError:
                pass

        return return_code

    # ------------------------------------------------------------------
    #  accelerate 运行 (训练用，保留原有接口)
    # ------------------------------------------------------------------
    async def run_accelerate(
        self,
        script_module: str,
        args: List[str],
        num_cpu_threads_per_process: int = 1,
        mixed_precision: str = "bf16",
        cwd: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> ProcessResult:
        """使用 accelerate 运行训练脚本（非阻塞）"""
        if self._running:
            return ProcessResult(ProcessStatus.ERROR, -1, "已有任务在运行")

        self._running = True
        self._notify_status(ProcessStatus.RUNNING)

        try:
            cmd = [
                sys.executable,
                "-m",
                "accelerate.commands.launch",
                f"--num_cpu_threads_per_process={num_cpu_threads_per_process}",
            ]
            if mixed_precision:
                cmd.append(f"--mixed_precision={mixed_precision}")
            cmd.extend(script_module.split())
            cmd.extend(args)

            work_dir = Path(cwd) if cwd else Path(self.PROJECT_ROOT)
            env = self._build_env(env_vars)

            self._notify_log(f"开始执行: {' '.join(cmd[:10])}...")
            self._notify_log(f"工作目录: {work_dir.absolute()}")
            self._notify_log("=" * 60)

            if self._requires_threaded_subprocess():
                self._notify_log("检测到 Windows reload 模式，使用线程子进程回退")
                return_code = await self._run_pipe_with_popen(cmd, work_dir, env)
            else:
                self.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(work_dir),
                    env=env,
                )

                await self._stream_output(self.process)
                return_code = await self.process.wait()
            self._running = False

            if return_code == 0:
                self._notify_status(ProcessStatus.SUCCESS)
                return ProcessResult(ProcessStatus.SUCCESS, return_code, "执行成功")
            else:
                self._notify_status(ProcessStatus.ERROR)
                return ProcessResult(ProcessStatus.ERROR, return_code, f"进程返回错误码: {return_code}")

        except Exception as e:
            self._running = False
            self._notify_status(ProcessStatus.ERROR)
            detail = str(e) or repr(e)
            error_msg = f"执行出错: {type(e).__name__}: {detail}"
            self._notify_log(error_msg)
            self._notify_log(_render_exception_text(e, prefix="执行出错"))
            return ProcessResult(ProcessStatus.ERROR, -1, error_msg)
        finally:
            self.process = None

    def run_script_sync(self, script_key: str, args: List[str], **kwargs) -> ProcessResult:
        """同步运行脚本（用于简单调用）"""
        return asyncio.run(self.run_python_script(script_key, args, **kwargs))

    def terminate(self):
        """终止当前进程（包括子进程树）"""
        if self.process and self._running:
            self._notify_log("正在终止进程...")
            try:
                if sys.platform == "win32":
                    # 杀掉整个进程树（wrapper + 子脚本）
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    self.process.terminate()
            except (ProcessLookupError, OSError):
                pass
            self._running = False
            self._notify_status(ProcessStatus.IDLE)

    @property
    def is_running(self) -> bool:
        """检查是否有任务在运行"""
        return self._running


# 全局进程运行器（保留向后兼容：未迁移的调用方仍可使用）
process_runner = ProcessRunner()

# 向后兼容别名：允许 `from gui.utils.process_runner import log_buffer`
log_buffer = _global_log_buffer
