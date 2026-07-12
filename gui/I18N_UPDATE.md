# GUI 国际化维护

GUI 当前支持中文、英文、日文和韩文。运行时翻译入口位于 `gui/utils/i18n.py`，页面通过 `t()` 获取文本，不应在页面中复制另一套翻译字典。

## 当前结构

| 文件 | 责任 |
| --- | --- |
| `gui/utils/i18n.py` | 语言检测、翻译表和 `t()` |
| `gui/main.py` | 页面、导航和全局标题 |
| `gui/launch.py` | 启动参数、路由注册和运行时信息 |
| `gui/wizard/*.py` | Setup、Import、Split、Tagger、Caption、Export、Tools 页面 |
| `gui/theme.py` | 主题变量和共享样式 |

## 新增或修改文案

1. 在 `gui/utils/i18n.py` 的四种语言中增加相同的 key。
2. 页面调用 `t("key", "English fallback")`，不要直接拼接用户可见文案。
3. 对带参数的文案使用现有格式化约定，并为缺失 key 保留可读 fallback。
4. 测试工具属于 `test` dependency group；先从仓库根目录将其安装到 `.venv`（Windows：`uv pip install --python .\.venv\Scripts\python.exe --group test`；Linux：`uv pip install --python .venv/bin/python --group test`），再激活 `.venv` 并运行 GUI i18n 测试：

```shell
# Activate .venv first: PowerShell: . .\.venv\Scripts\Activate.ps1
# Bash: source .venv/bin/activate
python -m pytest tests/test_gui_i18n.py tests/test_gui_main_lazy_import.py -q
```

## 语言切换

页面右上角的语言选择会刷新当前页面。表单值在刷新前应先保存；主题偏好由浏览器本地状态管理。

## 检查清单

- 四种语言都有新 key。
- 页面没有遗留旧的硬编码标题或按钮文本。
- 翻译字符串中的 `{}` 占位符数量和名称一致。
- 长文本在窄窗口中不会遮挡控件。
- 测试不依赖当前系统语言或网络。
