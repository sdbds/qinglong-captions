# Grok Build 订阅打标 Provider 设计

Date: 2026-06-01

## 参考资料

- xAI Build Getting Started: https://docs.x.ai/build/overview
- xAI Build Headless & Scripting: https://docs.x.ai/build/cli/headless-scripting
- xAI Build Modes and Commands: https://docs.x.ai/build/modes-and-commands
- xAI Build Enterprise Deployments: https://docs.x.ai/build/enterprise
- xAI model page for `grok-build-0.1`: https://docs.x.ai/developers/models/grok-build-0.1
- xAI Image Understanding API guide: https://docs.x.ai/docs/guides/image-understanding
- xAI Structured Outputs guide: https://docs.x.ai/docs/guides/structured-outputs

## 背景

项目里已经有 `codex_subscription` provider，用 Codex / ChatGPT 登录态做图片 caption。它的关键经验不是“命令行套壳”，而是把计费和认证语义拆清楚：

- API key route：按平台 API 计费。
- Subscription route：复用本机 agent CLI / app-server 的登录态，消耗产品订阅额度。
- 默认不自动启用，避免批量任务误耗额度。
- 默认串行，避免把交互式订阅当作批处理 quota。
- 不把 `OPENAI_API_KEY` 当成订阅登录失败后的 fallback。

Grok Build 的文档显示它也是 agent CLI，而不是普通 VLM endpoint。它支持交互式 TUI、headless `grok -p`、ACP `grok agent stdio`，认证可走浏览器 OIDC、device code、external auth provider 或 API key。Enterprise 文档还明确 credential resolution 会按模型解析：`model.api_key` > `model.env_key` > active session token > `XAI_API_KEY`。

因此 Grok Build 支持不能直接塞进 `openai_compatible`，也不能盲目复制 Codex 的 SDK app-server 形态。它需要自己的显式 provider：`grok_build_subscription`。

## 已确认事实

1. Grok Build 是 coding agent，可通过 TUI、headless 脚本、ACP 使用。
2. 官方安装路径包括 shell installer 和 npm 包 `@xai-official/grok`。
3. 首次运行会打开浏览器认证；非浏览器环境可使用 API key。
4. Enterprise 文档列出四类 session auth：Browser OIDC、Device code、External auth provider、API key。
5. Headless 模式支持：
   - `grok -p "<prompt>"`
   - `-m, --model <MODEL>`
   - `--cwd <PATH>`
   - `--output-format plain|json|streaming-json`
   - `--no-auto-update`
   - `--always-approve`
6. ACP 模式通过 `grok agent stdio` 使用 JSON-RPC over stdin/stdout。
7. ACP 示例里 `session/prompt` 的 prompt 形态至少支持 `{ "type": "text", "text": "..." }`。
8. Grok Build 模型 `grok-build-0.1` 的官方模型页标注 Modalities 为 Text, Image，输出为 Text，支持 function calling、structured outputs、reasoning。
9. xAI API 的 Image Understanding guide 明确图片输入支持 `input_image` / data URL / public URL，并列出限制：最大 20MiB、支持 jpg/jpeg/png。
10. xAI API 的 Structured Outputs guide 明确语言模型支持 structured outputs，并可用 JSON schema / Pydantic / Zod 约束输出。

## 已验证能力（本机 Phase 0）

本机环境：

- `grok` 路径：`C:\Users\qinglongshengzhe\.grok\bin\grok.exe`
- 版本：`grok 0.2.3 (14d81fd87) [stable]`
- probe 期间已从子进程环境移除 `XAI_API_KEY`、`GROK_CODE_XAI_API_KEY`，避免误走 API key 计费路径。

验证结果：

1. Text-only headless 可用：`grok -p "Reply with exactly: ok" --output-format json --no-auto-update` 返回 JSON envelope，`text` 字段为 `ok`。
2. `grok --help` 未发现 `--image` / `--file` 本地图片参数。
3. `--image <path>` 与 `--file <path>` 会被 CLI 判定为非法参数。
4. `--prompt-json` 支持 ACP content block 数组：text block `{"type":"text","text":"..."}`，image block `{"type":"image","data":"<base64>","mimeType":"image/png"}`。
5. `--prompt-json` 图片 block 已通过红色 PNG smoke：模型能识别图片主色并返回 JSON。
6. `--prompt-json @file` 与直接传文件路径都不被解析为文件引用，只会按字面字符串解析 JSON。
7. `mimeType` 是必填字段；`mime_type` 变体失败。
8. `localImage` path 形态失败，错误提示只接受 `text`、`image`、`audio`、`resource_link`、`resource`。

结论：

- 进入代码实现，但首版只实现已验证的 headless `--prompt-json` content-block backend。
- 不实现 ACP backend；ACP 图片 item 仍未形成可靠本机验证闭环。
- 不实现 API backend；API key 计费路径应另建 `xai_api` / `grok_api` provider。
- 由于 Windows 命令行长度限制，图片必须在 provider 内做自适应缩放/压缩后再以内联 base64 传给 `--prompt-json`。

## 未确认但必须先验证的事实

下面两点不能靠经验推断：

1. Grok Build CLI headless 是否支持类似 Codex `--image <file>` 的本地图片输入参数。
2. Grok Build ACP `session/prompt` 是否支持图片 input item，例如 `localImage`、`input_image`、`image` 或 data URL。

如果两点都不支持，那么 `grok_build_subscription` 不应该进入实现；应改做 `xai_api` / `grok_api` provider，通过 xAI API 的 Image Understanding + Structured Outputs 完成图片打标。那会走 API 计费，不是订阅额度。

## 第一性原理判断

本项目真正需要的是：

1. 给每个图片产生稳定、结构化、可缓存的 caption / rating payload。
2. 批量处理时成本、额度、失败原因可解释。
3. Provider 不抢路由，不污染本地模型、OCR、OpenAI-compatible API 路径。
4. 认证边界清晰，用户知道自己消耗的是订阅额度还是 API 账单。

Grok Build 的本质是 agent runtime。Agent runtime 的优势是登录态、上下文、工具和会话；劣势是吞吐、权限边界、输出稳定性和上下文污染。图片 caption 是一个窄任务，不需要 agent 读仓库、写文件、调用 shell 或保留多轮记忆。

所以正确切面是：

- 使用 Grok Build 的认证和模型能力。
- 把输入限制为当前图片和 prompt。
- 把输出限制为项目现有 caption schema。
- 把工作目录隔离到空目录。
- 默认关闭写工具和自动批准。
- 默认并发为 1。

## 目标

1. 设计一个显式 provider：`grok_build_subscription`。
2. 支持通过本机 Grok Build 登录态 / cached token 做图片 caption。
3. 不把 xAI API key 伪装成订阅额度。
4. 默认后端为 headless `--prompt-json`，因为这是 Phase 0 已验证的订阅路径。
5. 若未来 Grok Build 取消该图片输入形态，则停止订阅 provider，改评估独立 API provider。
6. 输出复用现有 Codex caption schema v2 语义：
   - `short_description`
   - `long_description`
   - `tags`
   - `rating`
   - `confidence`
   - `scores`
   - `total_score`
   - `average_score`
7. 默认串行，后续再接入 `cloud_max_concurrency` / provider 专属 cap。
8. 所有失败可诊断：未安装、未登录、API key 混入、权限模式错误、图片输入不支持、输出 JSON 解析失败、超时、额度限制。

## 非目标

- 不改 `openai_compatible` 来承载 Grok Build 订阅语义。
- 不直接调用 Grok Build 私有 inference proxy。
- 不读取、打印、复制 `~/.grok` 内 token。
- 不默认使用 `XAI_API_KEY`。
- 不自动把订阅失败 fallback 到 API key。
- 不让 Grok Build 写项目文件。
- 不在首版支持视频、音频、PDF、OCR 文档页。
- 不在首版支持多账号、多 profile、跨进程 daemon。
- 不复刻完整 ACP IDE 集成，只实现 caption 所需的最小子集。

## 总体架构

```text
caption pipeline
  -> provider registry
  -> grok_build_subscription provider
      -> GrokBuildHeadlessCaptionClient
          -> grok --prompt-json
              -> cached Grok Build session token / browser or device login
```

后端枚举：

```text
grok_build_backend = "headless"  # 首版唯一实现，grok --prompt-json
grok_build_backend = "acp"       # 未来验证后再实现
grok_build_backend = "api"       # 明确 API key 计费模式，非订阅模式
```

`api` backend 只能作为显式替代方案，不能作为订阅失败的自动 fallback。

## Phase 0：文档与本机能力验证

实现代码前先做 live probe，结论写回 spec 或 implementation plan。

### 安装 / 版本

```powershell
grok --version
grok inspect
grok --help
grok -p "Reply with exactly: ok" --output-format json --no-auto-update
```

### 登录态验证

优先验证订阅 / cached token：

```powershell
grok login
grok -p "Reply with exactly: subscription-ok" --output-format json --no-auto-update
```

非浏览器环境：

```powershell
grok login --device-auth
```

注意：如果环境里有 `XAI_API_KEY`，验证订阅路径时必须临时移除，避免 Grok 按 credential resolution 走 API key。

### Headless 图片输入验证

需要确认是否存在官方参数：

```powershell
grok --help
grok -p "Describe this image as JSON" --output-format json --no-auto-update <candidate image args>
```

候选仅用于验证，不写入实现前不能固化：

- `--image <path>`
- `--file <path>`
- `@<path>` prompt reference
- data URL pasted into prompt

如果只能通过 data URL 进 prompt，也要验证上下文长度、输出质量和隐私边界。

### ACP 图片输入验证

启动：

```powershell
grok agent stdio --no-auto-update
```

按官方 ACP 示例最小化流程：

1. 读取 `authenticate/methods` 或等价能力。
2. 选择 `cached_token`，不选择 `xai.api_key`。
3. `authenticate`。
4. `session/new`，cwd 指向隔离目录。
5. `session/prompt` 先发 text-only smoke。
6. 尝试图片 input item。

候选图片 item：

```json
{ "type": "localImage", "path": "C:/path/to/image.jpg" }
```

```json
{ "type": "input_image", "image_url": "data:image/jpeg;base64,..." }
```

```json
{ "type": "image", "path": "C:/path/to/image.jpg" }
```

任何一种成功前，都不能承诺 provider 支持图片。

## Provider 行为

### Provider 名称

```python
@register_provider("grok_build_subscription")
class GrokBuildSubscriptionProvider(CloudVLMProvider):
    ...
```

### 启用条件

必须显式启用：

```text
--grok_build_subscription
```

并且：

- `mime.startswith("image")`
- 未显式设置 `ocr_model`
- 未显式设置 `vlm_image_model`

不能因为用户安装了 `grok` 或登录了 Grok Build 就自动抢占 provider。

### 支持媒体

首版只支持：

- `image/jpeg`
- `image/png`

原因：xAI API 图片理解文档明确支持 jpg/jpeg/png，Grok Build 订阅路径在验证前也应按这个保守集合处理。AVIF / HEIC / HEIF 可仿照 Codex 先转临时 JPEG。

### 输出字段

复用 `module/providers/codex_schema.py` 的 schema 和 normalize 逻辑，后续可重命名为中立的 `agent_caption_schema.py`。

首版不要复制一份 Grok 专属 schema。复制会造成 Codex / Grok 输出契约漂移。

metadata：

```json
{
  "provider": "grok_build_subscription",
  "backend": "headless",
  "model": "grok-build",
  "auth_mode": "cached_token",
  "structured": true,
  "schema_version": "codex-caption-v2"
}
```

如果 schema 文件以后中立化，版本名再改；首版为了不动代码，先在 spec 里明确复用。

## 配置设计

PowerShell：

```powershell
$grok_build_subscription = $false
$grok_build_backend = "headless"     # headless only in v1
$grok_build_auth_mode = "cached_token" # cached_token, existing
$grok_build_command = "grok"
$grok_build_model_name = "grok-build"
$grok_build_timeout = 180
$grok_build_isolated_cwd = ""
$grok_build_permission_mode = "dontAsk"
$grok_build_sandbox = "read-only"
$grok_build_prompt_json_max_chars = 24000
$grok_build_max_concurrency = 1
```

CLI：

```text
--grok_build_subscription
--grok_build_backend=headless
--grok_build_auth_mode=cached_token|existing
--grok_build_command
--grok_build_model_name
--grok_build_timeout
--grok_build_isolated_cwd
--grok_build_permission_mode
--grok_build_sandbox
--grok_build_prompt_json_max_chars
--grok_build_max_concurrency
```

默认原则：

- `grok_build_subscription=false`
- `grok_build_backend=headless`
- `grok_build_auth_mode=cached_token`
- `grok_build_model_name=grok-build`
- 固定追加 `--no-auto-update`
- `grok_build_permission_mode=dontAsk`
- `grok_build_sandbox=read-only`
- `grok_build_max_concurrency=1`

## 认证设计

### 订阅路径

默认只使用 Grok Build 的 active session token / cached token。未登录时 fail closed，提示用户运行：

```powershell
grok login
```

或：

```powershell
grok login --device-auth
```

### API key 路径

`api_key` 是显式计费模式：

```text
--grok_build_backend=api
--grok_build_auth_mode=api_key
--grok_build_api_key=...
```

或者用户显式选择 `api_key` 时才允许读取 `XAI_API_KEY`。

禁止：

- 默认读取 `XAI_API_KEY`
- 订阅失败后自动使用 `XAI_API_KEY`
- 把 API key backend 的结果标记为 subscription

### 环境变量清理

订阅 backend 启动 `grok` 子进程时应清理：

```text
XAI_API_KEY
GROK_CODE_XAI_API_KEY
```

保留：

```text
HTTPS_PROXY
HTTP_PROXY
NO_PROXY
```

原因：Enterprise 文档说明 credential resolution 会优先走模型 API key / env key，若不清理，用户可能以为消耗订阅额度，实际走 API 账单。

## ACP Backend 设计

### 进程生命周期

每次 caption run lazy 启动一个：

```powershell
grok agent stdio --no-auto-update
```

首版不做跨进程 daemon。当前 Python 进程退出时关闭子进程。

### 初始化流程

概念流程：

```text
spawn grok agent stdio
  -> discover auth methods
  -> authenticate cached_token
  -> session/new with isolated cwd
  -> session/prompt with image + prompt
  -> collect session/update chunks
  -> parse final text as JSON
```

### 工作目录

默认隔离到：

```text
.cache/qinglong-captions/grok-build/work/
```

或系统临时目录：

```text
%TEMP%/qinglong-captions-grok-build-work
```

要求：

- 不使用项目根目录作为 cwd。
- 不注入项目 `AGENTS.md`。
- 不允许 Grok Build 读取不相关文件。

### Prompt

复用 Codex prompt 的约束，但把 agent 语义收紧：

```text
You are a captioning and rating engine.
Return only JSON matching the provided schema.
Do not include Markdown, commentary, file paths, tool output, or analysis steps.
Do not read files, inspect the workspace, or run tools.
Describe only the attached image and project-provided prompt context.
```

### JSON 输出

ACP 本身不一定支持 output schema。若不支持，应让 prompt 强约束 JSON，并复用 parser 做 fenced JSON 清理。若 ACP 支持 schema / structured output，再把 schema 放进 adapter，而不是 provider 业务层。

## Headless Backend 设计

命令形态：

```powershell
grok -p "<prompt>" `
  --model grok-build-0.1 `
  --cwd <isolated_cwd> `
  --output-format json `
  --no-auto-update
```

如果 headless 图片输入能力验证成功，再把图片参数加入 command builder。

要求：

- Python 使用 list argv，不拼 shell 字符串。
- prompt 通过 stdin 或安全参数传递，避免 PowerShell quoting 坑。
- 输出优先读 `--output-format json` 的最终对象。
- 如果 json envelope 里只有 assistant text，继续解析 text。

## API Backend 设计

API backend 不是订阅支持的一部分，但可以作为更短路径的替代方案。

实现方向：

- `base_url=https://api.x.ai/v1`
- `model=grok-build-0.1` 或其他支持图片理解的 Grok 模型。
- 输入使用 xAI Image Understanding 的 `input_image` + `input_text` 格式。
- 输出使用 Structured Outputs JSON schema。

命名建议不要叫 `grok_build_subscription`，而是另建：

```text
xai_api
```

或：

```text
grok_api
```

这样账单语义干净。

## 错误分类

统一异常：

```python
class GrokBuildError(RuntimeError):
    kind: str
    retryable: bool
    detail: str
```

分类：

- `command_not_found`
- `auth_required`
- `api_key_billing_mode`
- `permission_mode`
- `image_input_unsupported`
- `unsupported_media`
- `usage_limit`
- `rate_limited`
- `timeout`
- `transport_closed`
- `protocol_error`
- `output_parse_failed`
- `schema_rejected`

处理：

- `command_not_found`：提示安装 Grok Build。
- `auth_required`：提示 `grok login` / `grok login --device-auth`。
- `api_key_billing_mode`：如果用户没显式选 API key，直接失败。
- `image_input_unsupported`：停止实现，改走 xAI API 方案。
- `usage_limit`：不重试。
- `transport_closed`：重启 ACP 子进程后最多重试一次。
- `output_parse_failed`：最多重试一次，并强化“只返回 JSON”。

## 并发策略

默认：

```text
grok_build_max_concurrency = 1
```

后续接入云端并发时：

```text
effective = min(cloud_max_concurrency, grok_build_max_concurrency)
```

ACP backend 并发前必须满足：

- 每个 worker 独立 session，避免上下文污染。
- ACP client pool 有锁。
- 每个 slot 同时只处理一张图。
- `XAI_API_KEY` 清理逻辑对每个子进程生效。

首版不要让一个 session 处理多张图。上下文污染比少启动一个 session 更贵。

## 安全边界

必须：

- 不读取 `~/.grok` token 文件内容。
- 不打印 token、cookie、OAuth code。
- 不把 auth material 写入 `config/*.toml`、`.ps1`、测试 fixture。
- 不把项目根目录作为默认 cwd。
- 不启用自动写文件权限。
- 不让 prompt 要求 Grok Build 搜索工作区。
- 不在 CI / 脚本中自动 update CLI。

建议：

- `--no-auto-update` 默认打开。
- 单独的 Grok Build cache / cwd。
- live test 必须 opt-in。

## 模块布局建议

若 Phase 0 通过，新增：

- `module/providers/grok_build_acp.py`
- `module/providers/grok_build_headless.py`
- `module/providers/cloud_vlm/grok_build_subscription.py`
- `tests/test_grok_build_subscription_provider.py`
- `docs/grok_build_subscription_provider.md`

不要新增独立 schema 文件，先复用 `codex_schema.py`。后续重命名为中立 schema 是机械重构，不应阻塞 provider 可用性验证。

## 测试计划

### 单元测试

1. Provider 需要 `--grok_build_subscription` 才路由。
2. 显式 `ocr_model` / `vlm_image_model` 时不抢路由。
3. 订阅 backend 清理 `XAI_API_KEY` / `GROK_CODE_XAI_API_KEY`。
4. API key backend 只能显式启用。
5. ACP JSON-RPC message builder 使用 list / dict，不拼 shell。
6. Headless command builder 使用 argv list。
7. 输出 parser 接受 plain JSON、json envelope、fenced JSON。
8. 图片不支持时返回 `image_input_unsupported`。
9. `grok_build_max_concurrency` 默认 1。
10. timeout 会关闭坏掉的 ACP 进程。

### Fake Grok

用 fake executable 模拟：

- 成功 JSON 输出。
- 未登录。
- `XAI_API_KEY` 被误传则失败。
- streaming-json chunk。
- ACP `session/update` chunk。
- 不支持图片 input item。

### Live smoke

默认不进 CI：

```powershell
$env:RUN_LIVE_GROK_BUILD=1
pytest tests/test_grok_build_subscription_live.py
```

live smoke 只跑一张小 jpg/png，并断言输出满足 schema。

## 验收标准

Phase 0：

1. 确认 Grok Build CLI / ACP 至少一种方式能接收图片。
2. 确认在没有 `XAI_API_KEY` 的环境中能用 cached token 完成 text-only smoke。
3. 确认图片 smoke 输出可解析 JSON。
4. 若图片输入不支持，明确停止 subscription provider，实现 `xai_api` 替代方案。

Phase 1：

1. `--grok_build_subscription` 可处理单张 jpg/png。
2. 10 张图片串行处理不会污染上下文。
3. 未登录、未安装、API key 混入、图片不支持、输出非法 JSON 都有明确错误。
4. 不读取或打印任何 Grok token。
5. 不默认使用 `XAI_API_KEY`。
6. 输出能被现有 caption pipeline 消费。
7. 单元测试不依赖真实 Grok 登录态。

Phase 2：

1. ACP client 在一次 run 中复用。
2. 可选 client pool 支持受控并发。
3. 和 `cloud_max_concurrency` 集成。
4. GUI 增加高级实验性配置。

## 实施顺序

1. 完成 Phase 0 live probe，更新本 spec 的“已验证能力”。
2. 抽象共享 agent caption schema 的命名计划，但暂不重构。
3. 写 fake Grok 测试，锁住认证、argv、parser、路由规则。
4. 实现 `grok_build_acp.py` 最小 client。
5. 实现 `grok_build_subscription.py` provider。
6. 接 CLI / PowerShell 参数。
7. 手动 live smoke。
8. 再决定是否接 GUI 和并发。

## Linus 风格审查点

- 不要在图片输入未验证前写 provider 代码；这是把未知假设做成维护债。
- 不要让 `XAI_API_KEY` 静默接管订阅 provider；这是账单事故。
- 不要把 `openai_compatible` 改成 Grok Build 订阅入口；协议和认证语义都错。
- 不要让 agent 看项目根目录；caption 不需要 repo context。
- 不要复制 schema；两个 agent provider 的输出契约应该共享。
- 不要先做 GUI；先证明 CLI/ACP 单图闭环。
- 不要追求并发；先证明认证、输入、输出、失败分类正确。

## 当前建议

先做 Phase 0，不写 provider 代码。Grok Build 文档已经证明它适合 headless / ACP 自动化，也证明 `grok-build-0.1` 模型支持 Text, Image；但文档没有直接证明 Grok Build CLI / ACP 支持本地图片输入。这个缺口决定了后续路线：

- 如果 ACP 支持图片：实现 `grok_build_subscription`。
- 如果只有 API 支持图片：实现 `xai_api`，不要冒充订阅 provider。
- 如果 headless 支持图片但 ACP 不支持：先做 `headless` backend，ACP 延后。
