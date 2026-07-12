# 故障排查

先从仓库根目录执行命令，并保留失败任务的完整日志。日志分享前请删除 API Key、路径和个人信息。

## 安装阶段

### `uv` 找不到

1. 关闭并重新打开 PowerShell / 终端，使安装器写入的 PATH 生效。
2. 执行 `uv --version`。
3. 仍然找不到时，重新运行 `1.install-uv-qinglong.ps1`，并确认安装器没有被代理或执行策略拦截。

### Linux 找不到 `pwsh`

Linux 安装脚本的文件名包含空格，且脚本使用 Bash 语法；当前辅助脚本只提供 x86_64 PowerShell 下载：

```bash
sudo bash "./0.install pwsh.sh"
```

ARM64 或其他架构请跳过该脚本，按发行版文档手动安装 `pwsh` 7+。

检查：

```bash
pwsh -NoLogo -NoProfile -Command '$PSVersionTable.PSVersion'
```

### Python 版本不匹配

项目要求 Python `>=3.10,<3.13`。删除错误版本创建的 `.venv` 或 `venv` 前，请确认里面没有需要保留的本地包；然后重新运行安装脚本。不要让系统 Python、项目环境和 GUI 的 PEP 723 runtime 混用而不自知。

## GUI 阶段

### GUI 无法启动

按顺序运行：

```powershell
uv run gui/launch.py --help
uv run gui/launch.py --port 7899 --no-browser
```

如果 `--help` 都失败，问题通常是 `uv`、网络依赖或 PEP 723 runtime；如果 `--help` 成功但页面失败，检查端口、浏览器和 NiceGUI 日志。

`start_gui.ps1` 会自动切换到仓库根目录并运行 `uv run gui/launch.py`。不要先 `cd gui` 后直接猜入口。

### 端口被占用

GUI 会从请求端口开始尝试后续端口。使用终端打印的实际 URL，或显式指定：

```powershell
uv run gui/launch.py --port 7900 --no-browser
```

### 远程访问 / `--cloud`

`--cloud` 会绑定 `0.0.0.0`，但 GUI 没有内置登录鉴权。只在受信任的内网使用，或放在具备认证、TLS 和访问控制的反向代理后面。排查完成后恢复默认的 `127.0.0.1`。

## 模型与依赖

### 本地路由提示缺包

不要把所有 extra 一次装满。回到 `Caption` / `Tools` 页面重新选择目标路由，等待 GUI 补齐当前 profile。若一个虚拟环境已经装过互相冲突的 OCR、CUDA 或 Transformers 组合，优先创建新的项目环境再验证。

### Hugging Face 403 / gated model

1. 登录 Hugging Face 并接受该模型的访问条款。
2. 在当前运行环境设置 `HF_TOKEN`。
3. 确认 `HF_HOME` 指向可写目录，并检查缓存目录的磁盘空间。

不要把 `HF_TOKEN` 写入提交的脚本、截图、任务日志或 `config/env_vars.json` 的共享副本。

### 显存不足或模型加载很慢

- 降低批大小、分辨率、上下文长度或并发数。
- 选择 CPU / offload / quantized 路径（如果该 Provider 支持）。
- 关闭其他占用 GPU 的进程。
- 将模型缓存放到空间充足的磁盘，并避免多个进程同时首次下载同一模型。

## 工作流阶段

### 导入或导出失败

- 输入路径必须存在，并且要从仓库根目录运行脚本。
- `lanceExport.ps1` 的 `lance_file` 必须指向有效 `.lance` 数据集。
- 导出前确认 `version` 与实际 Lance tag 一致。
- 如果数据集正在被其他任务写入，先停止并等待写入完成，再进行导出。

### 翻译失败

先运行 `5.translate.ps1` 补齐 `translate` profile，并在当前终端激活项目环境（`.venv` 或 `venv`），再只运行规范化：

```powershell
python -m module.texttranslate ./datasets --normalize_only
```

然后检查 `chunk_offsets`、模型依赖和可写目录；确认无误后再运行翻译。使用 OpenAI-compatible backend 时，先用 `--runtime_backend openai`、`--openai_base_url` 和 `--openai_model_name` 验证服务连通性。

### 音频分轨失败

检查输入是否为支持的音频文件、目录或 `.lance` 数据集。模型首次下载需要网络和足够磁盘；输出格式只能是 `wav`、`flac` 或 `mp3`。先关闭 `harmony_separation` 验证基础六 stems 流程。

### Image2PSD 失败

确认输入目录包含可读图片、输出目录可写，并预留模型和中间结果所需空间。先使用较小的 `limit_images` 做冒烟测试，再增加分辨率、推理步数或批量规模。

## 日志与安全

任务日志可能含有输入绝对路径、模型 ID、代理地址和错误上下文；部分字幕路径还会把 API Key 作为命令参数传入子进程。保存或分享日志前请脱敏。如果令牌已经出现在日志或 `config/env_vars.json` 中，应立即撤销并重新生成。
