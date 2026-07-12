# GUI 更新说明

本文件记录当前 GUI 的稳定能力；版本级变更请看根目录 [CHANGELOG.md](../CHANGELOG.md)。

## 当前能力

- NiceGUI 浏览器模式和可选原生窗口模式
- 中文、英文、日文、韩文切换
- 深色 / 浅色主题偏好
- Setup、Import、Split、Tagger、Caption、Export、Tools 页面
- 任务标签页、并发任务、独立日志缓冲区和取消操作
- Provider 路由对应的依赖 profile 自动补齐
- ONNX 音频分轨、文本翻译和 Image2PSD 工具箱入口

## 启动行为

日常入口为仓库根目录的 `start_gui.ps1`，它运行 `uv run gui/launch.py`。`gui/launch.py` 使用 PEP 723 依赖声明和 GUI 隔离运行时，默认端口为 `8080`；脚本包装后默认端口为 `7899`。

## 维护约定

- 新页面文案统一放入 `gui/utils/i18n.py`，并同步四种语言。
- 新任务应通过 `gui/utils/job_manager.py` 和 `gui/utils/process_runner.py` 管理，不要在页面中直接阻塞事件循环。
- 依赖 profile 由 `pyproject.toml` 维护，页面只选择路由，不复制完整安装命令。
- 不要把 `config/env_vars.json`、模型缓存、数据集或任务日志加入 Git。

## 相关文档

- [GUI 使用手册](README.md)
- [GUI 参数映射](PARAMETERS.md)
- [根目录使用说明](../README.md)
- [配置指南](../docs/configuration.md)
