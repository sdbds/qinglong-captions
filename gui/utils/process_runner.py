"""进程运行工具 - 管理外部 Python 脚本的调用和日志输出

执行方式与 PowerShell 启动脚本一致:
  1. (可选) uv pip install -r requirements-xxx.txt  安装依赖
  2. uv run ./script.py <args>                       运行脚本

关键：使用 asyncio.create_subprocess_exec 进行非阻塞 I/O，
避免阻塞 NiceGUI 事件循环导致 WebSocket 断开。
"""

import asyncio
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


# 用于剥离 ANSI 转义码（日志文件 → GUI 纯文本）
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")


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


# 脚本路径映射: module_key -> (script_path, requirements_file | None)
# script_path 相对于项目根目录
SCRIPT_REGISTRY = {
    # step 1 - 数据集导入
    "module.lanceImport": ("./module/lanceImport.py", None),
    # step 2 - 视频分割
    "module.videospilter": ("./module/videospilter.py", None),
    # step 3 - 打标
    "utils.wdtagger": ("./utils/wdtagger.py", "requirements-wdtagger.txt"),
    # step 4 - 字幕生成
    "module.captioner": ("./module/captioner.py", None),
    # step 5 - 导出
    "module.lanceexport": ("./module/lanceexport.py", None),
    # step 6 - 工具
    "module.waterdetect": ("./module/waterdetect.py", None),
    "utils.preprocess_datasets": ("./utils/preprocess_datasets.py", None),
    "module.rewardmodel": ("./module/rewardmodel.py", None),
}

# console_wrapper.py 的绝对路径
_WRAPPER_PATH = str(Path(__file__).parent / "console_wrapper.py")


class ProcessRunner:
    """运行外部进程并捕获输出（非阻塞）

    使用 asyncio.create_subprocess_exec 保证不阻塞 NiceGUI 事件循环。
    """

    # 项目根目录 (gui/utils/../../ = 项目根)
    PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())

    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.log_callback: Optional[Callable[[str], None]] = None
        self.status_callback: Optional[Callable[[ProcessStatus], None]] = None
        self._running = False
        self._tail_task: Optional[asyncio.Task] = None

    def set_callbacks(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[ProcessStatus], None]] = None,
    ):
        """设置回调函数"""
        self.log_callback = log_callback
        self.status_callback = status_callback

    def _notify_log(self, message: str):
        """通知日志回调"""
        if self.log_callback:
            self.log_callback(message)

    def _notify_status(self, status: ProcessStatus):
        """通知状态回调"""
        if self.status_callback:
            self.status_callback(status)

    def _build_env(self, env_vars: Optional[dict] = None) -> dict:
        """构建子进程环境变量（与 PowerShell 脚本保持一致）"""
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # PYTHONPATH
        existing = env.get("PYTHONPATH", "")
        if self.PROJECT_ROOT not in existing:
            env["PYTHONPATH"] = self.PROJECT_ROOT + os.pathsep + existing if existing else self.PROJECT_ROOT

        # HF_HOME - 与 .ps1 脚本一致
        env.setdefault("HF_HOME", "huggingface")
        env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")

        return env

    @staticmethod
    def _find_uv() -> Optional[str]:
        """查找 uv 可执行文件路径"""
        return shutil.which("uv")

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

    # ------------------------------------------------------------------
    #  日志文件尾随（原生控制台模式）
    # ------------------------------------------------------------------
    async def _tail_log_file(self, log_file: str):
        """异步尾随日志文件，将新增内容（去除 ANSI）回传 GUI"""
        offset = 0
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
                                for line in text.splitlines():
                                    clean = _ANSI_RE.sub("", line).rstrip()
                                    if clean:
                                        self._notify_log(clean)
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    #  pip install (可选)
    # ------------------------------------------------------------------
    async def _install_requirements(self, requirements_file: str, work_dir: Path, env: dict) -> bool:
        """安装依赖 (uv pip install -r requirements-xxx.txt)"""
        req_path = work_dir / requirements_file
        if not req_path.exists():
            self._notify_log(f"依赖文件不存在，跳过: {req_path}")
            return True

        uv = self._find_uv()
        if uv:
            cmd = [uv, "pip", "install", "-r", str(req_path)]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]

        self._notify_log(f"安装依赖: {' '.join(cmd)}")

        try:
            install_env = {**env, "PYTHONIOENCODING": "utf-8"}
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(work_dir),
                env=install_env,
            )
            await self._stream_output(proc)
            rc = await proc.wait()
            if rc != 0:
                self._notify_log(f"依赖安装返回错误码 {rc}，继续尝试运行脚本...")
            return True
        except Exception as e:
            self._notify_log(f"依赖安装出错: {e}，继续尝试运行脚本...")
            return True

    # ------------------------------------------------------------------
    #  主要运行方法
    # ------------------------------------------------------------------
    async def run_python_script(
        self,
        script_key: str,
        args: List[str],
        cwd: Optional[str] = None,
        env_vars: Optional[dict] = None,
        requirements: Optional[str] = None,
        uv_extra_args: Optional[List[str]] = None,
        native_console: bool = True,
    ) -> ProcessResult:
        """运行 Python 脚本（完全非阻塞）

        Args:
            script_key: 脚本标识，如 'utils.wdtagger'、'module.captioner'。
            args: 传递给脚本的参数列表。
            cwd: 工作目录 (默认项目根目录)。
            env_vars: 额外环境变量。
            requirements: 强制指定依赖文件。
            uv_extra_args: uv run 额外参数。
            native_console: Windows 下使用原生控制台窗口（支持 rich 颜色/图片）。
        """
        if self._running:
            return ProcessResult(ProcessStatus.ERROR, -1, "已有任务在运行")

        self._running = True
        self._notify_status(ProcessStatus.RUNNING)

        use_native = native_console and sys.platform == "win32"
        exit_file = ""
        log_file = ""

        try:
            work_dir = Path(cwd) if cwd else Path(self.PROJECT_ROOT)
            env = self._build_env(env_vars)
            if not use_native:
                env["PYTHONIOENCODING"] = "utf-8"

            # 从 registry 查找脚本路径和默认依赖
            registry_entry = SCRIPT_REGISTRY.get(script_key)
            if registry_entry:
                script_path, default_req = registry_entry
            else:
                script_path = "./" + script_key.replace(".", "/") + ".py"
                default_req = None

            req_file = requirements or default_req

            # Step 1: 安装依赖（始终使用管道模式，不弹窗）
            if req_file:
                self._notify_log(f"正在安装依赖: {req_file}")
                await self._install_requirements(req_file, work_dir, env)
                self._notify_log("=" * 60)

            # Step 2: 构建运行命令
            uv = self._find_uv()
            if uv:
                cmd = [uv, "run"]
                if uv_extra_args:
                    cmd.extend(uv_extra_args)
                cmd.append(script_path)
            else:
                cmd = [sys.executable, script_path]

            cmd.extend(args)

            self._notify_log(f"开始执行: {' '.join(cmd[:15])}{'...' if len(cmd) > 15 else ''}")
            self._notify_log(f"工作目录: {work_dir.absolute()}")
            self._notify_log("=" * 60)

            # Step 3: 启动进程
            if use_native:
                return_code = await self._run_native(cmd, work_dir, env)
            else:
                # 管道模式：输出回传到 GUI 日志
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
            error_msg = f"执行出错: {str(e)}"
            self._notify_log(error_msg)
            return ProcessResult(ProcessStatus.ERROR, -1, error_msg)
        finally:
            self.process = None
            if self._tail_task and not self._tail_task.done():
                self._tail_task.cancel()
                self._tail_task = None

    # ------------------------------------------------------------------
    #  原生控制台模式
    # ------------------------------------------------------------------
    async def _run_native(self, cmd: List[str], work_dir: Path, env: dict) -> int:
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

        # 包装命令: python console_wrapper.py <exit_file> <log_file> <cmd...>
        wrapper_cmd = [sys.executable, _WRAPPER_PATH, exit_file, log_file] + cmd

        self.process = await asyncio.create_subprocess_exec(
            *wrapper_cmd,
            cwd=str(work_dir),
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        self._notify_log("已在原生控制台窗口中启动，输出同步显示在下方")

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
            error_msg = f"执行出错: {str(e)}"
            self._notify_log(error_msg)
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


# 全局进程运行器
process_runner = ProcessRunner()
