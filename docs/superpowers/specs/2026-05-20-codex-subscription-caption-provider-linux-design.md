# Codex 订阅认证打标 Provider Linux-first 设计

## 背景

当前项目已经集成多种云端、本地、OpenAI-compatible provider。现有 OpenAI-compatible 路径的核心约束是：它使用 OpenAI SDK 的 `chat.completions.create`，配置面是 `openai_api_key` / `openai_base_url` / `openai_model_name`。

这条路径适合：

- OpenAI 官方 API key
- vLLM / SGLang / Ollama / LM Studio
- OneAPI / NewAPI 等 OpenAI-compatible 网关

但它不适合直接承载“ChatGPT/Codex 订阅额度打标”。Codex 订阅认证不是普通 `OPENAI_API_KEY`，也不是把 `base_url` 改成 ChatGPT 后端即可复用。OpenClaw 的实现表明，正确抽象是：

- OpenAI API key route：按 API 账单计费
- Codex subscription route：通过 Codex OAuth / app-server runtime 使用订阅额度

因此本设计不把 Codex 订阅认证塞进 `openai_compatible`，而是新增一个显式 provider：`codex_subscription`。

首版目标平台限定为 Linux / WSL。原因：

- Codex CLI / app-server 在 Linux 下更接近自动化与 CI 场景
- 文件权限、`CODEX_HOME`、进程隔离更容易定义
- 可以先绕开 Windows 路径、PowerShell quoting、跨盘符路径转换带来的噪声

## 已确认事实

1. Codex CLI 支持非交互模式：`codex exec`。
2. 当前本机 Windows 侧 `codex-cli 0.130.0` 可用，并且 `codex exec` 支持 `-i, --image <FILE>...` 和 `--output-schema <FILE>`。
3. 当前 WSL 侧能找到 `/mnt/d/nodejs/codex`，但启动失败，错误为缺少 `@openai/codex-linux-x64` 可选依赖。Linux 首版必须把这类环境预检作为显式前置条件，而不是在打标中途才失败。
4. 本项目现有 provider 返回统一的 `CaptionResult(raw, parsed, metadata)`，`codex_subscription` 应遵守这一返回契约。
5. 本项目已有 `mode = all | short | long` 的输出过滤语义，新 provider 不应创造一套平行字段。

## 目标

1. 在 Linux / WSL 下新增可选 `codex_subscription` provider，用 Codex 登录态/订阅额度为图片打标。
2. 首版只支持图片 caption，不支持视频、音频、OCR 文档页。
3. 不读取、不复制、不提交 Codex OAuth token 或 refresh token。
4. 首版通过官方 Codex CLI 非交互接口实现最小可用版本。
5. 输出必须是可解析 JSON，并映射到现有 `CaptionResult` 字段。
6. 所有失败必须可诊断：未安装、未登录、Linux 可选依赖缺失、订阅额度耗尽、JSON 解析失败、超时。
7. 默认串行运行，避免批量任务快速打满订阅额度。
8. 保留后续升级到 Codex app-server / SDK 的接口边界。

## 非目标

- 不把 ChatGPT/Codex 订阅伪装成 `OPENAI_API_KEY`
- 不通过抓包或硬编码 `chatgpt.com/backend-api/codex` 私有 HTTP 细节来调用模型
- 不导入 OpenClaw 大量 gateway / plugin / auth-profile 体系
- 不在首版支持 Windows 原生运行
- 不在首版支持多并发、分布式、队列调度
- 不在首版支持长期持有 Codex app-server 连接
- 不自动绕过 Codex 订阅 usage limit

## 第一性原理分析

打标系统的底层需求不是“调用某个大模型”，而是：

1. 对每个媒体输入产生稳定、结构化、可缓存的描述结果。
2. 批处理时可以恢复、跳过、失败继续。
3. 成本和额度消耗可控。
4. 失败时可以判断是模型质量问题、认证问题、环境问题还是额度问题。

Codex 的底层形态是 agent runtime，不是普通低延迟推理 endpoint。因此它天然有几个不适配点：

- 单次调用开销高
- 输出可能带 agent 过程文本
- 可能使用工作区上下文
- 可能触发工具/命令权限逻辑
- 订阅额度是交互式产品额度，不是无限批处理额度

所以首版必须把 Codex 当作“高质量补标 provider”，而不是默认主力 provider。

## 总体方案

采用两阶段设计。

### 阶段 1：`codex exec` Provider

首版实现 `codex_subscription` provider，内部通过 subprocess 调用：

```bash
codex exec \
  --cd <isolated_work_dir> \
  --sandbox read-only \
  --ephemeral \
  --model <codex_model_name> \
  --image <image_path> \
  --output-schema <caption_schema.json> \
  "<caption_prompt>"
```

关键约束：

- `--image` 只传当前图片。
- `--cd` 指向隔离目录，避免 Codex 默认扫描整个项目。
- `--sandbox read-only`，打标不需要写工作区或执行修改。
- `--ephemeral`，避免为每张图保留长期 thread。
- `--output-schema` 强制最终输出形态。
- Python provider 只读取 stdout 最终结果，不依赖 stderr 进度文本。

优点：

- 实现成本最低
- 不碰 OAuth token 文件
- 可直接复用用户已有 `codex login`
- 环境问题容易暴露

缺点：

- 每张图片启动一次 Codex，开销较大
- 吞吐较低
- 难复用上下文
- 不适合超大批量

### 阶段 2：Codex app-server / SDK Provider

当阶段 1 验证订阅认证、输出质量、额度消耗都可接受后，再实现长期版本：

- 启动或连接 `codex app-server --listen stdio://`
- 使用 JSON-RPC / SDK 创建 thread
- 向 thread 发送 `localImage` + prompt
- 读取 final response
- 按账号 / 模型 / profile 做连接缓存

这一阶段才接近 OpenClaw 的 Codex harness 思路，但仍然只实现项目需要的最小子集。

## 模块布局

首版建议新增：

- `module/providers/cloud_vlm/codex_subscription.py`
- `module/providers/codex_exec.py`
- `module/providers/codex_schema.py`
- `tests/test_codex_subscription_provider.py`
- `tests/test_codex_exec_command.py`
- `docs/codex_subscription_provider.md`

配置入口接入：

- `4.captioner.ps1`
- `module/providers/catalog.py`
- CLI 参数定义所在模块
- GUI step4 caption provider 选择项

## Provider 行为

### Provider 名称

注册名：

```python
@register_provider("codex_subscription")
class CodexSubscriptionProvider(CloudVLMProvider):
    ...
```

### 启用条件

首版必须显式启用，避免误用订阅额度：

- `--codex_subscription=true`
- 或 GUI 中选择 `Codex subscription`

不能只因为 `codex` 命令存在就自动抢占 provider。

### 支持媒体

首版：

- `image/*`: 支持

拒绝：

- `video/*`
- `audio/*`
- document/OCR

### 输出字段

Codex 最终输出必须匹配：

```json
{
  "short_description": "string",
  "long_description": "string",
  "tags": ["string"],
  "rating": "general|sensitive|questionable|explicit",
  "confidence": 0.0
}
```

字段兼容规则：

- `mode = short`：保留 `short_description`
- `mode = long`：保留 `long_description`
- `mode = all`：保留全部字段
- 缺失 `short_description` 时从 `long_description` 截断生成
- 缺失 `long_description` 时使用 `short_description`

## 配置设计

PowerShell 参数建议：

```powershell
$codex_subscription = $false
$codex_command = "codex"
$codex_model_name = "gpt-5.4-mini"
$codex_home = ""
$codex_timeout = 180
$codex_sandbox = "read-only"
$codex_isolated_cwd = ""
$codex_output_schema = ""
```

CLI 参数建议：

```text
--codex_subscription
--codex_command
--codex_model_name
--codex_home
--codex_timeout
--codex_sandbox
--codex_isolated_cwd
--codex_output_schema
```

默认值原则：

- `codex_subscription=false`，必须显式打开
- `codex_model_name=gpt-5.4-mini`，优先低成本模型
- `codex_timeout=180`，单图超时
- `codex_sandbox=read-only`
- `codex_isolated_cwd` 为空时使用系统临时目录下的 `qinglong-codex-caption-work`

## Linux 环境预检

新增 preflight：

```bash
command -v codex
codex --version
codex exec --help
codex exec --ephemeral --sandbox read-only "Reply with exactly: ok"
```

若 WSL 出现类似：

```text
Missing optional dependency @openai/codex-linux-x64
```

提示：

```bash
npm install -g @openai/codex@latest
```

并要求在 Linux/WSL 内部安装，而不是复用 Windows 盘上的 Node 全局包。

推荐 Linux 安装形态：

```bash
npm i -g @openai/codex@latest
codex
```

用户完成 ChatGPT/Codex 登录后再运行打标。

## 命令构造

Python 侧使用 `subprocess.run([...], text=True, capture_output=True, timeout=...)`，不通过 shell 拼字符串。

命令数组示例：

```python
[
    codex_command,
    "exec",
    "--cd",
    isolated_cwd,
    "--sandbox",
    "read-only",
    "--ephemeral",
    "--model",
    model_name,
    "--image",
    image_path,
    "--output-schema",
    schema_path,
    prompt,
]
```

环境变量：

- 如果用户设置 `codex_home`，传入 `CODEX_HOME=<codex_home>`
- 不传入 `OPENAI_API_KEY`
- 不传入 `CODEX_API_KEY`
- 保留代理变量：`HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY`

显式清理 API key 的原因：

- 避免 Codex CLI 在存在 API key 时走 API 计费路径
- 让订阅登录态和 API key 路径保持可诊断

## Prompt 设计

系统约束：

```text
You are a captioning engine. Return only JSON matching the provided schema.
Do not include Markdown, commentary, chain-of-thought, file paths, or analysis steps.
Describe visible content only. Do not infer identity, private attributes, or unverifiable facts.
```

用户 prompt 由现有 `PromptContext` 合成，追加：

```text
Generate caption metadata for the attached image.
Use concise tags and a natural long description.
If uncertain, lower confidence instead of inventing details.
```

中文项目 prompt 可以继续保留，但最终输出字段名必须保持英文，方便现有 pipeline 解析。

## 错误分类

Provider 应把 stderr/stdout 归类为：

### 环境错误

- `codex` 不存在
- Linux optional dependency 缺失
- `codex exec --help` 失败

处理：立即失败，不重试。

### 认证错误

- 未登录
- refresh token 失效
- OAuth 需要重新登录

处理：提示用户在 Linux 中运行 `codex` 完成登录，不重试。

### 额度错误

- usage limit exceeded
- reached Codex subscription usage limit
- retry after / reset time

处理：记录 reset hint，停止当前 provider；不自动切 API key。

### 输出错误

- stdout 不是 JSON
- JSON 不满足 schema
- 字段为空

处理：最多一次修复重试，重试 prompt 明确“previous output was invalid JSON”。

### 超时错误

- 单图超过 `codex_timeout`

处理：返回空结果或抛出 provider 错误，遵循现有 batch 失败策略。

## 缓存策略

Codex 订阅额度不应浪费在重复图片上。

缓存 key：

```text
sha256(image)
prompt_fingerprint
codex_model_name
schema_version
provider_version
```

缓存位置：

```text
.cache/qinglong-captions/codex_subscription/
```

缓存内容：

```json
{
  "raw": "...",
  "parsed": {},
  "metadata": {
    "provider": "codex_subscription",
    "model": "gpt-5.4-mini",
    "codex_cli_version": "codex-cli 0.130.0",
    "created_at": "..."
  }
}
```

首版可以先只做内存级跳过；落盘缓存作为实现阶段的第二个 patch。

## 安全边界

必须遵守：

- 不读取 `~/.codex` 内部 token 文件
- 不打印 token、cookie、OAuth code
- 不把 `CODEX_HOME` 放在项目目录下
- 不把 Codex auth material 写入 `config/*.toml`、`.ps1`、测试 fixture
- 不复用用户真实 repo 工作区作为 `--cd`
- 不允许 caption prompt 请求 Codex 读取任意文件

推荐：

```bash
mkdir -p ~/.local/share/qinglong-captions/codex-home
chmod 700 ~/.local/share/qinglong-captions/codex-home
```

如果用户设置专用 `CODEX_HOME`，登录也应在同一个 `CODEX_HOME` 下完成：

```bash
CODEX_HOME=~/.local/share/qinglong-captions/codex-home codex
```

## GUI 策略

GUI 中该 provider 默认隐藏在“高级 / 实验性”区域：

- 开关：Use Codex subscription provider
- Model：`gpt-5.4-mini` / `gpt-5.4` / 用户自定义
- Timeout
- Codex command
- CODEX_HOME
- Preflight 按钮

GUI 文案必须明确：

- 使用的是 Codex/ChatGPT 订阅额度
- 不使用 `OPENAI_API_KEY`
- 批量运行可能触发 Codex usage limit
- Linux/WSL-only

## 与现有 provider 的关系

优先级建议：

1. 明确指定本地 `vlm_image_model` / `ocr_model` 时，本 provider 不抢占。
2. `codex_subscription=true` 时，图片路由到 `codex_subscription`。
3. `openai_base_url` 仍然走 `openai_compatible`。
4. 其他云端 key provider 保持现状。

这避免出现“用户配置了本地模型，但因为 Codex 登录了就被偷路由”的问题。

## 测试计划

### 单元测试

- 命令数组构造，不允许 shell 字符串拼接
- `CODEX_API_KEY` / `OPENAI_API_KEY` 清理
- `CODEX_HOME` 注入
- JSON schema 文件生成
- stdout JSON 解析
- Markdown fenced JSON 清理
- mode 字段过滤
- 错误分类
- WSL optional dependency 缺失识别

### 集成测试

使用 fake `codex` 可执行文件：

- 成功输出 JSON
- 输出非法 JSON 后二次重试
- stderr 模拟未登录
- stderr 模拟 usage limit
- sleep 模拟超时

### Live smoke

默认不进 CI。手动启用：

```bash
RUN_LIVE_CODEX=1 pytest tests/test_codex_subscription_live.py
```

最小验证：

```bash
codex exec \
  --ephemeral \
  --sandbox read-only \
  --image tests/fixtures/sample.png \
  --output-schema tests/fixtures/codex_caption_schema.json \
  "Return caption JSON for the attached image."
```

## 验收标准

阶段 1 完成标准：

1. Linux/WSL 中 `codex_subscription=true` 可以处理单张图片。
2. 10 张图片串行处理成功率可观察，失败样本有明确错误原因。
3. 输出字段能被现有 caption pipeline 消费。
4. 未登录、额度耗尽、Codex 安装错误都能给出明确提示。
5. 不会读取或打印任何 Codex token。
6. 默认不使用 `OPENAI_API_KEY`。
7. 测试覆盖 command builder、parser、error classifier。

阶段 2 完成标准：

1. 引入长期 app-server / SDK 连接，不再每图启动一次 Codex。
2. 支持复用 thread 或连接池。
3. 支持更清晰的 usage limit reset 处理。
4. 与阶段 1 provider API 保持兼容。

## 实施顺序

1. 写入本 spec。
2. 增加 Linux preflight helper。
3. 增加 fake-codex 驱动的 command builder 和 parser 测试。
4. 新增 `module/providers/cloud_vlm/codex_subscription.py`。
5. 接入 CLI / PowerShell 参数。
6. 接入 GUI 高级选项。
7. 手动 live smoke。
8. 根据真实吞吐和额度消耗决定是否进入 app-server 阶段。

## 决策记录

### 为什么不直接改 `openai_compatible`

因为 `openai_compatible` 的核心契约是 OpenAI Chat Completions / OpenAI-compatible API。Codex 订阅路径是 Codex runtime，不是这个 API 形态。混在一起会导致：

- 计费路径不透明
- 错误分类混乱
- `OPENAI_API_KEY` 与订阅 auth 互相覆盖
- 后续维护困难

### 为什么首版不用 OpenClaw 代码

OpenClaw 解决的是 agent gateway 问题，本项目解决的是 caption batch provider 问题。直接搬运会引入过多不必要概念：

- auth profile store
- plugin runtime
- app-server event projector
- dynamic tools
- multi-agent/session routing

首版只需要最小闭环：图片输入、Codex 订阅认证、JSON 输出、失败可诊断。

### 为什么 Linux-first

不是因为 Windows 不可行，而是因为首个可控变量应该是“认证和 provider 语义”，不是跨平台路径和 shell quoting。Linux-first 能先验证核心命题：

```text
Codex 订阅登录态是否能稳定产出项目所需 caption JSON？
```

若答案为否，就不应继续投入 app-server 深度集成。
