# OvisOCR2 GUI 配置控件设计

## 目标

修正 OvisOCR2 在模型配置面板中的字段类型与默认提示词展示：

- `runtime_backend` 使用固定下拉框，保存值为 `direct` 或 `openai`。
- `visual_region_mode` 使用固定下拉框，保存值为 `crop` 或 `drop`。
- TOML 中 `prompt = ""` 仍表示继承官方默认提示词，但 GUI 直接显示完整官方提示词。
- GUI 中显示的官方提示词若未修改或改回等价值，保存后仍保持 `prompt = ""`，不把默认值复制进 TOML。

## 数据所有权

新增轻量 OvisOCR2 契约模块，只包含官方默认提示词等不依赖推理运行时的常量。OvisOCR2 Provider 与 GUI 配置面板共同导入该常量，避免在 GUI 内复制提示词，也避免 GUI 为读取默认值而导入 Torch、Transformers 或模型加载器。

Provider 继续以该常量作为 `default_prompt`。运行时提示词优先级保持不变：

1. 非空 `[ovis_ocr2].prompt`
2. 非空 legacy `prompts.ovis_ocr2_prompt`
3. 共享契约中的官方默认提示词

## GUI 行为

现有 `ModelConfigPanel` 的 section-scoped 枚举表增加：

```text
(ovis_ocr2, runtime_backend) -> direct, openai
(ovis_ocr2, visual_region_mode) -> crop, drop
```

这些约束只作用于 OvisOCR2，不把所有同名字段假定为具有相同后端或模式集合。

模型配置加载后，若 `[ovis_ocr2].prompt` 为空：

- textarea 的显示值使用共享契约中的完整官方提示词。
- 面板内部待保存数据仍保持空字符串，未发生编辑时不会复制默认值。
- 用户输入自定义提示词时，保存其原始内容。
- 用户清空提示词，或把内容改回与官方默认等价的文本时，待保存值归一化为空字符串。

“与官方默认等价”只忽略首尾空白以及 `CRLF`/`LF` 换行差异，不折叠正文内部空格或换行。这样可以避免把语义不同的自定义提示词误判为默认值。

## 保存契约

字段值在 GUI 事件进入面板数据时归一化，不改变通用 TOML 写盘逻辑：

- 默认或空提示词 -> `prompt = ""`
- 自定义提示词 -> 原样写入

重新打开面板或恢复配置时，空哨兵再次解析为可见的完整官方提示词。

## 兼容性

- 不修改 `config/model.toml` 与 `config/config.toml` 的 `prompt = ""` 默认值。
- 不改变其他 Provider 的字符串字段、枚举选项或保存行为。
- 不让 GUI 导入 OvisOCR2 推理实现，避免增加启动成本和可选依赖要求。
- 不改变 Direct 或 OpenAI-compatible 推理协议。

## 测试

- 断言 OvisOCR2 两个字段渲染为固定下拉框并具有精确选项。
- 断言空 prompt 在 GUI 中显示共享的官方默认提示词。
- 断言未修改的默认提示词、仅首尾空白或换行风格不同的默认提示词保存为空哨兵。
- 断言内部正文发生变化的自定义提示词原样保留。
- 断言 Provider 与 GUI 使用同一个轻量契约常量，且加载配置面板不会导入 OvisOCR2 推理模块。

## 非目标

- 不为整个模型配置系统引入通用 schema 或新的 TOML 元数据格式。
- 不给其他 Provider 猜测 `runtime_backend` 可选值。
- 不把官方默认提示词复制到任何配置文件。
