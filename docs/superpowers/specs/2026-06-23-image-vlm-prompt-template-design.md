# 图片 VLM Prompt 模板选择设计

## 背景

当前图片 caption 的 prompt 选择是**纯 provider 驱动**，没有“用户自由选模板”这一维度。链路如下：

1. GUI `gui/wizard/step4_caption.py` 把选项拼成 CLI flags，调 `module/captioner.py`。
2. `captioner.py` argparse 解析出 `args`，构造 provider 上下文（`config` 来自 `config/prompts.toml`）。
3. `module/providers/base.py:Provider.resolve_prompts()` 实例化 `PromptResolver(self.ctx.config, self.name, ...)` 并调 `resolver.resolve(mime, self.ctx.args, ...)`。
4. `module/providers/resolver.py` 按固定优先级选 prompt：

```text
_base_prompts(mime)        # image -> image_system_prompt / image_prompt
  -> _provider_override     # image -> {provider}_image_system_prompt / {provider}_image_prompt
  -> _pair_override          # 配对图模式
  -> _apply_gemini_task_template  # 仅 gemini + image，改 user
  -> character_prompt 前置
```

也就是说：**选哪个图片 prompt，取决于你选了哪个模型**，而不是用户主动挑的。`config/prompts.toml` 里现有三类典型图片 prompt：

- `image_system_prompt` / `image_prompt`：cosplay/画质**评分**模板（给各维度打分 + 英文详描，敏感词 `**` 隔开）。它是 `_base_prompts` 的 image 默认。
- `mistral_ocr_image_system_prompt` / `mistral_ocr_image_prompt`：客观**详细描述（danbooru tags 风格）**模板，输出 `###Short:` / `###Long:`。当前只通过 `declarations.py` 的 `prompt_fallback` 绑死在 `mistral_ocr` / `pixtral` provider 上。
- `BBOX_prompt`：生成 **Ideogram 4 风格 JSON**（`high_level_description` / `style_description` / `compositional_deconstruction`，含 bbox 与调色板）。它是一个**孤立 key**，没有 `_image_system_prompt` 命名，也没有任何 `prompt_fallback` 声明，**当前任何代码路径都不会使用它**——是死配置。

本次需求：把图片 VLM prompt 升级为**用户可在 GUI 自由选择的模板**，内置三个预设（danbooru tags / 评分 / BBOX JSON），其余由用户自定义。

> 本文档只做设计，不修改任何代码。

## 三个问题（先回答再动手）

1. **这是真问题还是臆想？** 真问题。当前图片 prompt 与 provider 强耦合，用户想“用 Gemini 跑 danbooru tags 风格”或“用任意模型跑 BBOX JSON”都做不到，只能去改 `prompts.toml` 里 `image_system_prompt` 的全局默认，牵一发动全身。`BBOX_prompt` 已经写好却无法触达，本身就是证据。

2. **有没有更简单的做法？** 有，且是本 spec 的核心：模板**不是**新功能，而是给已经存在的 `PromptResolver` 加**一层显式覆盖**。模板内容复用现有 prompt key，不复制文本。默认值 = “跟随模型”，等价于今天的行为，零迁移。

3. **会破坏什么？** 关键风险**不在 prompt 层，而在输出契约层**：Gemini（以及 codex_subscription / grok_build_subscription）对图片**强制结构化输出 schema**。光换 prompt 不动 schema，会让模型“被迫”按旧 schema 吐 JSON，模板（尤其是 BBOX）形同虚设——这是隐性破坏，必须在设计里正面处理。详见“核心洞察”。

## 核心洞察（数据结构 + 特殊情况）

### 洞察 1：模板 = (system, user, **输出契约**)，不是单纯的 prompt 字符串

`module/providers/vision_api/gemini.py:GeminiProvider.get_structured_output_config()` 对 image 且非 `gemini_task`、非 pair 时，**强制**启用 `_build_rating_schema()`（`application/json`，固定字段 `long_description/tags/rating/scores/average_score`…）。`attempt_gemini()` 之后还会 `json.loads` 按该 schema 解析。

后果：

| 模板 | Gemini 强制 schema 下的实际效果 |
| --- | --- |
| 评分（rating） | 基本对得上（schema 本就是评分形态），可用 |
| danbooru tags | 被 schema 形态扭曲，`###Short/###Long` 文本协议失效 |
| BBOX JSON | **彻底失效**——模型被 schema 逼着吐 rating JSON，吐不出 BBOX JSON |

`codex_subscription` / `grok_build_subscription` 的 system prompt 也明写 “Fill long_description, tags, rating, scores, average_score”，同属强制结构化族（其确切机制实现期需核对其 provider 代码，**不在本 spec 里假装确定**）。

结论：一个“模板”必须携带**输出契约**（`text` / `json`），并且当用户显式选了非默认模板时，**结构化输出 provider 必须让位**（关掉强制 schema，改自由文本/自由 JSON 跟随模板），否则“模板覆盖一切”是假的。

### 洞察 2：用数据结构消除特殊情况

不要在 resolver 里为每个模板写 if 分支。模板就是数据：一张 `template_id -> {system_key, user_key, output}` 表，放进 `prompts.toml`（与既有 `[prompts.task]` 子表同构）。resolver 只做一件事：**若 `args.image_prompt_template` 非空且非 `custom`，用表里的内容覆盖 image 的 system/user**。新增模板 = 加一行 TOML，零代码改动。

### 洞察 3：默认值就是“今天的行为”

`image_prompt_template` 默认空（= `custom` / 跟随模型）。空 → resolver 不插入覆盖层 → 完全走现有 provider 驱动逻辑。这是 Never break userspace 的兜底：没勾模板的用户感知不到任何变化。

## 需求理解确认

> 我理解的需求：在图片 caption 流程中引入“图片 VLM Prompt 模板”维度，用户在 GUI 自由选择；内置三个预设——(1) danbooru tags 详描（以 `mistral_ocr_image_system_prompt` 为例）、(2) 画质/cosplay 评分（以 `image_system_prompt` 为例）、(3) BBOX JSON（以 `BBOX_prompt` 为例）；其余模板由用户自定义。已确认采用**方案“模板覆盖一切”**：选中预设后对**所有**图片 provider 强制生效，优先级**高于 provider 专属 key**（如 `mistral_ocr_image_system_prompt`）；默认值 = `custom/跟随模型`，完全保留当前行为，零破坏。

## 目标

1. 新增“图片 VLM Prompt 模板”概念：数据驱动、可扩展、复用现有 prompt key。
2. 内置三个预设模板：`danbooru_tags`、`rating`、`bbox_json`。
3. GUI（`step4_caption.py` 基础设置）提供模板下拉框：3 预设 + `custom/跟随模型`（默认）。
4. 新增 CLI 参数 `--image_prompt_template`，GUI 据此传参。
5. `PromptResolver` 新增**图片模板覆盖层**：优先级高于 `_provider_override`，仅作用于 image mime，仅在显式选择非 `custom` 时生效。
6. 模板携带**输出契约**（`text`/`json`）；强制结构化输出 provider（Gemini 等）在选中非默认模板时**关闭强制 schema**，真正跟随模板。
7. `bbox_json` 的 JSON 输出能完整落盘为 caption（不被 `###` 短/长拆分破坏）。
8. 默认（`custom`）行为与今天逐字节一致。
9. 用测试锁住：默认零变化、模板覆盖优先级、结构化 provider 让位、BBOX JSON 落盘。

## 非目标

- 不改既有 prompt 文本内容（仅引用，不重写 `image_system_prompt` 等）。
- 不动 pair 图模式与 `gemini_task` 图像编辑模式的语义（二者是正交任务模式，见“优先级链”）。
- 不为视频 / 音频 / OCR / 文档引入模板维度（本次只做 image VLM caption）。
- 不在 GUI 内置“富文本编辑器”改 prompt 内容（自定义 = 编辑 `prompts.toml`，见“自定义模板”）。
- 不重写 `PromptResolver` 的整体分层结构，只**加一层**。
- 不删除 / 改名任何现有 prompt key（含已死的 `BBOX_prompt`，本次让它复活而非搬家）。

## 数据结构设计

### 1. 模板注册表（`config/prompts.toml`）

在 `[prompts]` 下新增子表 `[prompts.image_templates]`（与现有 `[prompts.task]` 同构，`load_prompts_from_toml` 天然带出，无需改 loader）：

```toml
[prompts.image_templates.danbooru_tags]
label = "Danbooru tags (detailed caption)"
system_key = "mistral_ocr_image_system_prompt"
user_key = "mistral_ocr_image_prompt"
output = "text"

[prompts.image_templates.rating]
label = "Quality / cosplay rating"
system_key = "image_system_prompt"
user_key = "image_prompt"
output = "text"

[prompts.image_templates.bbox_json]
label = "BBOX JSON (Ideogram 4)"
system_key = "BBOX_prompt"
user_key = "bbox_image_prompt"   # 见下：BBOX 当前没有独立 user prompt
output = "json"
```

字段语义：

- `label`：GUI 下拉框显示名。
- `system_key` / `user_key`：**引用** `[prompts]` 下已存在的字符串 key（单一事实来源，不复制文本）。`user_key` 允许为空字符串，表示该模板不需要独立 user prompt。
- `output`：`"text"` | `"json"`。`text` 走现有短/长/SRT/markdown 文本契约；`json` 表示输出是结构化 JSON，落盘时整体保留、**不做 `###` 拆分**。

> 设计选择：引用 key 而非内联文本。理由——三个预设的文本已存在于 `[prompts]`，内联会制造两份真相。代价是一层间接查表（`prompts[template.system_key]`），可接受。

### 2. `BBOX_prompt` 的 user prompt 缺口

`BBOX_prompt` 是自包含 system prompt，但当前**没有**配套 user prompt。若 `user_key` 留空回退到 `image_prompt`（评分指令“只给分数…”），会和 BBOX 任务冲突。

决策：在 `[prompts]` 新增一个极简 key（命名沿用 `*_image_prompt` 家族）：

```toml
bbox_image_prompt = """Deconstruct the provided image into the strict Ideogram 4 JSON structure as instructed. Output valid JSON only."""
```

不复用 `image_prompt`，避免评分指令污染 BBOX 输出。

### 3. CLI 参数（`module/captioner.py`）

`setup_parser()` 新增：

```python
parser.add_argument(
    "--image_prompt_template",
    type=str,
    default="",
    help="Image VLM prompt template id (e.g. danbooru_tags / rating / bbox_json). Empty = follow model (current behavior).",
)
```

空字符串 / `"custom"` = 不覆盖（现状）。其它值 = 模板 id。

### 4. Resolver 覆盖层（`module/providers/resolver.py`）

在 `resolve()` 中、`_provider_override` 之后、`_pair_override` 之前，插入一层：

```python
# 现有
system, user = self._base_prompts(mime)
system, user = self._provider_override(system, user, mime)
# 新增（仅 image，仅显式非 custom）
if mime.startswith("image"):
    system, user = self._image_template_override(system, user)
# 其后照旧
if mime.startswith("image") and self._is_pair_mode(args, media):
    system, user = self._pair_override(system, user, mime)
...
```

`_image_template_override`：

```python
def _image_template_override(self, system, user):
    template_id = getattr(self.args, "image_prompt_template", "") or ""  # resolve() 需把 args 存到 self
    if not template_id or template_id == "custom":
        return system, user
    templates = self.prompts.get("image_templates", {})
    tpl = templates.get(template_id)
    if not tpl:
        return system, user   # 未知 id：静默回退到 provider 驱动结果，不抛
    sys_key = tpl.get("system_key", "")
    usr_key = tpl.get("user_key", "")
    new_system = self.prompts.get(sys_key, system) if sys_key else system
    new_user = self.prompts.get(usr_key, "") if usr_key else ""
    return new_system, new_user
```

注意：`resolve()` 当前没把 `args` 存成实例属性（只在方法内用形参），实现期需把 `self.args = args` 或将 `template_id` 透传进来。优先**最小改动**：`resolve()` 开头 `self._args = args`，新方法读 `self._args`。

### 5. 结构化输出 provider 让位

强制结构化 schema 的 provider（首要确认对象 `gemini`；待核对 `codex_subscription` / `grok_build_subscription`）需要知道“当前是否有非默认图片模板生效”，从而关闭 schema。

最小侵入做法：新增一个集中判定（放 catalog 或 resolver 旁的 helper），供 provider 查询：

```python
def active_image_template(args):
    tid = getattr(args, "image_prompt_template", "") or ""
    return "" if tid == "custom" else tid
```

`GeminiProvider.get_structured_output_config()` 增加一条早退：

```python
if active_image_template(args):
    return StructuredOutputConfig(enabled=False)   # 让模型自由跟随模板
```

效果：

- `rating` 模板：虽与旧 schema 形态相近，但统一走自由文本，由 postprocess 的 `###` / json 兜底解析，不再依赖 Gemini 专属 schema。
- `bbox_json` 模板：Gemini 不再强制 rating schema，模型按 `BBOX_prompt` 自由吐 JSON。

> 实现期必须核对 `codex_subscription` / `grok_build_subscription` 是否同样需要让位入口；若它们的 schema 是命令行/请求体硬编码，需各自加等价开关，**不得假设三者机制相同**。

## 优先级链（最终）

```text
1. _base_prompts(mime)              # image_system_prompt / image_prompt
2. _provider_override               # {provider}_image_system_prompt ...
3. _image_template_override   ← 新增；非 custom 时覆盖 system/user（高于 #2，满足“覆盖 provider 专属 key”）
4. _pair_override                   # 配对图：正交任务模式，仍可在模板之上生效
5. _apply_gemini_task_template      # gemini 图像编辑：正交任务模式，仅改 user
6. character_prompt 前置
```

正交任务模式说明：

- **pair 模式**与 **gemini_task** 是“换了一个任务”（双图对比 / 图像编辑），有各自的 prompt 与（Gemini）schema 分支。本 spec 把图片模板定位为**单图 caption 风格选择**。
- 设计裁决：图片模板覆盖的是 #1/#2 这条“单图描述”链；当 pair / gemini_task 这类**用户另行显式触发**的任务模式存在时，它们保留各自语义（#4/#5 在 #3 之上）。即“模板覆盖一切”的“一切”范围 = provider 专属 prompt key（与用户已确认选项的措辞一致），不含正交任务模式。这样不破坏 pair / 图像编辑，符合 Never break userspace。

## 输出契约设计

### text 模板（danbooru_tags / rating）

走现有 `module/caption_pipeline/postprocess.py:postprocess_caption_content` 文本路径：`###` → 短/长拆分；否则原样。无新增。

### json 模板（bbox_json）

现有 postprocess 对 image 文本会先 `json.loads`：合法 dict → `CaptionResult(raw=output, parsed=dict)`。BBOX JSON 的字段（`high_level_description` 等）不匹配 `CaptionResult.description` 里找的 `long_description/transcript/...`，故 `description` 回退到 `self.raw`（完整 JSON 串），`to_dataset_caption()` 走 `json.dumps(parsed)`。

结论：**BBOX JSON 能完整落盘**，无需改 postprocess。但需测试锁死：

- `description` / sidecar text == 原始 JSON 串（或其 round-trip 等价），不被 `###` 逻辑触碰（BBOX 文本不含 `###`，天然安全，但仍要测）。
- 不会因 `parsed` 是 dict 而被误判成 rating 结构丢字段。

风险点：若某 provider（如 Gemini 让位后）把 JSON 包在 ```json 代码块里，postprocess 的 `json.loads` 会失败而当成普通文本存。对 `output="json"` 模板，实现期需确认是否要剥 code fence（复用 `extract_code_block_content`）。本 spec 标为实现期校对项，倾向：`output="json"` 时若直接 `json.loads` 失败，尝试剥 ```json 再存；仍失败则原样存字符串。

## GUI 集成（`gui/wizard/step4_caption.py`）

1. 新增常量（从注册表派生，避免 GUI 与 TOML 双份真相）：

```python
def _load_image_templates() -> dict[str, str]:
    # 读 config/prompts.toml -> [prompts.image_templates]，返回 {id: label}
    # 失败回退空 dict
```

2. `config` 默认值新增 `"image_prompt_template": ""`（= custom）。
3. `_render_basic_settings()` 增加下拉框（属图片全局选项，与 mode/pair_dir 同级）：

```python
options = {"": t("template_follow_model")} | _load_image_templates()
self.image_prompt_template = styled_select(
    options=options, value="", label=t("image_prompt_template"),
    icon="dashboard_customize", icon_color=COLORS["primary"], searchable=False, flex=1,
)
```

4. `_build_caption_args()` 末尾：

```python
tpl = getattr(getattr(self, "image_prompt_template", None), "value", "") or ""
if tpl and tpl != "custom":
    args.append(f"--image_prompt_template={tpl}")
```

5. i18n：新增 `image_prompt_template`、`template_follow_model` 文案（中英）。

> GUI 只读 `id` 与 `label`，不感知 `system_key/user_key/output`，模板细节全在 TOML。

## 自定义模板

用户自定义路径（满足“其余需要自己自定义”）：

1. 在 `[prompts]` 写自己的 `my_system_prompt` / `my_user_prompt`（或复用现有 key）。
2. 在 `[prompts.image_templates.my_tpl]` 加一行，指向这些 key，设 `output`。
3. 重启 GUI（或重载 config），下拉框自动出现 `my_tpl`。

无需改任何代码。`custom`（默认空值）则表示“不用模板、跟随模型”，仍可像今天一样直接编辑 `image_system_prompt` 全局默认。

## 涉及文件

新增：

```text
tests/test_image_prompt_template.py
```

修改：

```text
config/prompts.toml                         # [prompts.image_templates] + bbox_image_prompt
module/captioner.py                          # --image_prompt_template 参数 + 透传到 provider 上下文
module/providers/resolver.py                 # _image_template_override 层 + self._args
module/providers/vision_api/gemini.py        # 非默认模板时关闭强制 schema
gui/wizard/step4_caption.py                  # 下拉框 + 默认值 + 传参 + _load_image_templates
gui/utils/i18n.py（或对应 i18n 资源）         # image_prompt_template / template_follow_model 文案
```

待核对后可能修改：

```text
module/providers/cloud_vlm/codex_subscription.py     # 若强制 schema，需让位入口
module/providers/cloud_vlm/grok_build_subscription.py # 同上
module/caption_pipeline/postprocess.py               # 仅当 json 模板需剥 code fence 时
```

## 测试计划（`tests/test_image_prompt_template.py`）

全部不调真实模型，构造 `args` + `config` 直接驱动 `PromptResolver` / postprocess：

1. **默认零变化**：`image_prompt_template=""`，对 `mistral_ocr` / `gemini` / 任意 provider，resolver 输出与“未引入本特性”完全一致（用现有 `_provider_override` 结果断言）。
2. **覆盖优先级**：选 `danbooru_tags`，provider = `mistral_ocr`（本就有专属 key）与 `gemini`（无专属 key），二者 system/user 均等于 `mistral_ocr_image_system_prompt` / `mistral_ocr_image_prompt`，证明模板 > provider 专属 key。
3. **rating 模板**：system/user == `image_system_prompt` / `image_prompt`。
4. **bbox_json 模板**：system == `BBOX_prompt`，user == `bbox_image_prompt`，`output == "json"`。
5. **未知 id 静默回退**：`image_prompt_template="不存在"` → 退回 provider 驱动结果，不抛异常。
6. **结构化 provider 让位**：`GeminiProvider.get_structured_output_config()` 在任一非默认模板下返回 `enabled=False`；`custom` 下保持原 `enabled=True`。
7. **BBOX JSON 落盘**：把一段合法 BBOX JSON 喂给 `postprocess_caption_content`（image 后缀），断言 `description` / `to_dataset_caption()` 完整保留该 JSON，未被 `###` 或字段提取破坏。
8. **GUI 传参**：`_build_caption_args` 在选中模板时产出 `--image_prompt_template=<id>`，`custom`/空时不产出该 flag。
9. **正交模式不被破坏**：pair 模式下即使设了模板，`_pair_override` 仍生效（pair system/user 胜出）。

定向运行：

```powershell
uv run pytest tests/test_image_prompt_template.py
```

## 验收标准

1. 不传 `--image_prompt_template` 时，所有图片 provider 行为与改动前逐字节一致。
2. GUI 下拉框含 `custom/跟随模型`（默认）+ `danbooru_tags` + `rating` + `bbox_json`，且从 TOML 派生。
3. 选中任一预设后，对**任意**图片 provider（含无专属 key 的 Gemini）均生效，优先级高于 provider 专属 key。
4. Gemini 在非默认模板下不再强制 rating schema；BBOX 模板能产出 BBOX JSON 并完整落盘。
5. 新增模板只需改 `prompts.toml`，零代码改动即在 GUI 出现。
6. 未知模板 id 不致命（静默回退）。
7. pair / gemini_task 任务模式语义不变。
8. 不加载真实模型的单测全过。

## 风险与缓解

1. **结构化输出耦合（最高风险）**：模板只换 prompt 不换 schema → Gemini/codex/grok 上模板失效。缓解：把“输出契约 + 让位开关”写进设计；测试 #6 锁 Gemini；codex/grok 的 schema 机制实现期逐个核对，不假设统一。
2. **`resolve()` 未持有 `args`**：现有 resolver 在方法内用形参。缓解：`resolve()` 开头存 `self._args = args`，最小改动。
3. **BBOX 无 user prompt**：缓解：新增 `bbox_image_prompt`，不复用评分 `image_prompt`。
4. **JSON 被 code fence 包裹**：某些 provider 吐 ```json。缓解：`output="json"` 路径实现期决定是否剥 fence；测试覆盖裸 JSON 与 fenced JSON 两种。
5. **GUI / TOML 双份真相漂移**：缓解：GUI 用 `_load_image_templates()` 从 TOML 派生，不在 GUI 硬编码模板列表。
6. **正交模式优先级歧义**：缓解：spec 明确 #3 < #4/#5（pair/gemini_task 胜出），并写测试 #9。

## Linus 风格审查点

- 模板是**数据**，不是代码分支。三个预设 = 三行 TOML，resolver 里**一个**新方法，没有 per-template 的 if。
- 不复制 prompt 文本。模板**引用** key，单一事实来源。
- 默认值 = 今天的行为。没勾模板的用户零感知（Never break userspace）。
- 正面处理 Gemini 结构化输出这个隐藏特殊情况——不要假装“换 prompt 就够了”。
- 不为“覆盖一切”去践踏 pair / 图像编辑这些正交任务模式。

## 实施顺序

1. `config/prompts.toml`：加 `[prompts.image_templates]` 三个预设 + `bbox_image_prompt`。
2. `module/captioner.py`：加 `--image_prompt_template`，透传进 provider 上下文 `args`。
3. `module/providers/resolver.py`：`self._args` + `_image_template_override` 层。
4. `module/providers/vision_api/gemini.py`：非默认模板关闭强制 schema；加 `active_image_template` helper。
5. 核对并按需处理 `codex_subscription` / `grok_build_subscription` 的 schema 让位。
6. `gui/wizard/step4_caption.py`：`_load_image_templates` + 下拉框 + 默认值 + 传参；i18n 文案。
7. 新增 `tests/test_image_prompt_template.py`，覆盖测试计划 1–9。
8. 跑定向单测，确认默认零变化。
9. 实现期校对：BBOX JSON 的 code fence 处理、codex/grok 的真实 schema 机制——以实跑为准修正，不在 spec 里写死。
