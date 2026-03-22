# 青龙字幕工具 GUI (Qinglong Captions GUI)

基于 NiceGUI 的现代化图形化界面，为青龙字幕工具提供直观易用的操作体验。

## 功能特点

- 🎨 **现代化界面** - 使用 NiceGUI 构建，支持深色/浅色主题
- 🌐 **多语言支持** - 支持中文、英文、日文、韩文
- 📊 **实时日志** - 查看命令输出和进度
- 🖱️ **点击操作** - 无需记忆复杂的命令行参数
- 📁 **路径选择** - 可视化的文件/文件夹选择器
- 🎵 **音频分轨** - 在 GUI 工具箱中直接运行 ONNX 音频分离
- 🌍 **文本翻译** - 在 GUI 工具箱中执行文档规范化与翻译

## 使用方法

### 方法 1: 使用 PowerShell 脚本 (推荐)

```powershell
./start_gui.ps1
```

### 方法 2: 使用 Python 直接运行

```bash
# 确保在虚拟环境中
python -m gui.launch

# 或使用参数
python -m gui.launch --host 0.0.0.0 --port 8080
```

### 方法 3: 进入 gui 目录运行

```bash
cd gui
python main.py
```

## 启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 绑定地址 | 127.0.0.1 |
| `--port` | 端口 | 8080 |
| `--cloud` | 云模式 (绑定 0.0.0.0) | False |
| `--native` | 原生窗口模式 | False |
| `--no-browser` | 不自动打开浏览器 | False |

## 页面说明

### 1. 首页 (Home)
- 快速开始向导
- 支持的模型展示
- 功能特点介绍

### 2. 数据导入 (Import)
- 将视频/图像导入 Lance 数据库
- 支持多种导入模式

### 3. 视频分割 (Split)
- 场景检测和分割
- 支持多种检测算法
- 生成场景预览图

### 4. 标签生成 (Tagger)
- WD14 Tagger 打标
- 支持多种模型
- 可配置阈值和选项

### 5. 字幕生成 (Caption)
- 支持多种 API (Gemini, Pixtral, Qwen, Step, Kimi, GLM)
- OCR 支持
- 场景检测参数配置

### 6. 数据导出 (Export)
- 从 Lance 数据库导出字幕
- 支持多种格式

### 7. 工具箱 (Tools)
- 水印检测
- 图像预处理
- 图像评分
- 音频分轨
- 文本翻译

## 项目结构

```
gui/
├── main.py                 # 主入口
├── launch.py              # 启动脚本
├── theme.py               # 主题样式
├── README.md              # 本文件
├── components/            # 可复用组件
│   ├── path_selector.py   # 路径选择器
│   ├── log_viewer.py      # 日志查看器
│   ├── model_selector.py  # 模型选择器
│   └── ...
├── wizard/                # 向导页面
│   ├── step0_setup.py     # 环境检查
│   ├── step1_import.py    # 数据导入
│   ├── step2_video_split.py  # 视频分割
│   ├── step3_tagger.py    # 标签生成
│   ├── step4_caption.py   # 字幕生成
│   ├── step5_export.py    # 数据导出
│   └── step6_tools.py     # 工具箱
└── utils/                 # 工具函数
    ├── i18n.py            # 国际化
    ├── process_runner.py  # 进程运行器
    └── config_manager.py  # 配置管理
```

## 与 PowerShell 脚本的对应关系

| GUI 页面 | PowerShell 脚本 |
|----------|----------------|
| 数据导入 | `lanceImport.ps1` |
| 视频分割 | `2.0.video_spliter.ps1` |
| 标签生成 | `3、tagger.ps1` |
| 字幕生成 | `4、run.ps1` |
| 数据导出 | `lanceExport.ps1` |
| 水印检测 | `2.1.image_watermark_detect.ps1` |
| 图像预处理 | `2.2.preprocess_images.ps1` |
| 图像评分 | `2.3.image_reward_model.ps1` |
| 音频分轨 | `2.5.audio_separator.ps1` |
| 文本翻译 | `5.translate.ps1` |

## 注意事项

1. 使用 GUI 前请确保已运行 `1.install-uv-qinglong.ps1` 安装依赖
2. 某些功能需要配置 API 密钥
3. 建议在虚拟环境中运行
4. 首次运行可能需要下载模型文件

## 故障排除

### GUI 无法启动
- 检查虚拟环境是否激活
- 确认已安装 nicegui: `pip install nicegui`
- 查看端口是否被占用

### 命令执行失败
- 检查工作目录是否正确
- 确认输入路径有效
- 查看日志输出获取详细错误信息

### 主题切换无效
- 刷新页面后重试
- 检查浏览器控制台是否有错误

## 贡献

欢迎提交 Issue 和 Pull Request 来改进 GUI！

## 许可证

与主项目相同的许可证。
