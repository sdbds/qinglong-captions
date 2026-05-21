# Codex Python SDK 长驻 App-Server 打标 Provider 设计

## 背景

上一版 `codex_subscription` 设计和当前实现证明了一个事实：本项目可以通过本机已登录的 Codex 订阅会话为图片打标。但当前主路径是 `codex exec`，它有三个根本问题：

- 每张图启动一次 Codex，冷启动成本高。
- 依赖用户机器上已有 `codex` 命令和 PATH 解析，Windows / WSL 差异明显。
- 命令行参数、stdin 编码、repo trust check 这类执行边界问题会反复冒出来。

OpenClaw 的方向不是直接要求用户全局安装 NPM 版 Codex CLI，而是把 Codex runtime 当成受管理的 harness/app-server。Codex 官方 Python SDK 当前也走这个方向：Python SDK 控制本地 `codex app-server` JSON-RPC v2，发布版 SDK 通过同版本 runtime 包携带平台对应的 Codex binary。

因此新目标是：把 `codex_subscription` 的主实现从 `codex exec` 升级为 Python SDK / app-server 长驻连接，保留 `codex exec` 只作为 fallback。

## 已确认事实

1. Codex 订阅额度不能通过普通 OpenAI API key 消费。
2. OpenAI API key 登录可以作为 Codex app-server 的认证方式之一，但会走平台 API 计费，不等价于 ChatGPT/Codex 订阅额度。
3. Codex Python SDK 是实验性接口，目标协议是 `codex app-server` JSON-RPC v2。
4. Codex Python SDK 发布包会 pin 对应版本的 runtime 包，runtime 包包含平台 binary，因此用户不需要 NPM 才能获得 Codex runtime。
5. App-server turn 支持：
   - `thread/start`
   - `turn/start`
   - `input` 中的 `{ "type": "text" }`
   - `input` 中的 `{ "type": "localImage", "path": "..." }`
   - `outputSchema`
6. 当前项目的 caption provider 契约是 `CaptionResult(raw, parsed, metadata)`。
7. 当前 `codex exec` fallback 已经在 Windows 单图上验证可用，但性能和部署形态不适合作为长期主路径。

## 目标

1. 新增 `codex_subscription` 的 app-server backend，默认使用 Codex Python SDK。
2. 调用 provider 时自动确保 Python SDK 环境可用，参考项目现有 uv extra / 可选依赖模式。
3. 不要求用户安装 Node.js、NPM 或全局 `codex` CLI。
4. 启动一个长驻 `codex app-server`，在批处理生命周期内复用连接。
5. 通过 Python 发 JSON-RPC：
   - `thread/start`
   - `turn/start`
   - `input = [{ type: "text" }, { type: "localImage", path: "..." }]`
   - `outputSchema = CODEX_CAPTION_SCHEMA`
6. 默认复用 ChatGPT/Codex 登录态，消耗订阅额度。
7. 认证优先使用 ChatGPT/Codex 订阅登录。API key 不是自动 fallback，只能作为用户显式选择的另一种计费模式。
8. 保留现有 `codex exec` backend 作为兼容回退。
9. 错误必须可诊断：SDK 缺失、runtime 缺失、未登录、API key 误用、订阅额度耗尽、turn 超时、JSON schema 输出失败。

## 非目标

- 不直接调用 `chatgpt.com/backend-api/codex/responses` 私有 HTTP 接口。
- 不复刻 OpenClaw 的完整 auth profile、gateway、plugin、dynamic tool 和 approval 体系。
- 不把 `OPENAI_API_KEY` 自动当成订阅认证。
- 不在首版做跨进程共享 app-server daemon。
- 不在首版做多账号、多 profile 并发池。
- 不在首版让 Codex 执行 shell / patch / 文件修改工具。
- 不让 caption provider 读取或打印 refresh token、access token、auth.json 原文。

## 第一性原理判断

本项目需要的是“高质量结构化图片描述”，不是“通用代码 agent”。所以 app-server 集成必须收窄为一个最小受控子集：

1. 输入只有当前图片和 caption prompt。
2. 输出只有 JSON schema 约束后的最终 assistant message。
3. 工作目录隔离，避免 Codex 读取整个项目上下文。
4. 权限默认只读或无工具，打标不需要写文件。
5. 订阅额度是用户个人交互产品额度，不应默认高并发消耗。

如果实现把 Codex 当成完整 agent 运行，复杂度会失控；如果实现把 Codex 当普通 HTTP VLM endpoint，认证和计费边界会失真。正确切面是：使用 Codex app-server 的认证和模型能力，但只暴露 caption provider 需要的单 turn 图像理解能力。

## 总体架构

```text
caption pipeline
  -> provider registry
  -> codex_subscription provider
      -> CodexAppServerCaptionClient
          -> Python SDK / app-server JSON-RPC
              -> managed Codex runtime binary
                  -> ChatGPT/Codex auth session
```

后端选择：

```text
codex_backend = "sdk_app_server"  # 默认
codex_backend = "exec"            # fallback
```

### Backend 1：`sdk_app_server`

主路径。

- Python 进程内创建 app-server client。
- app-server 进程由 SDK 启动并维护。
- 单次批处理复用同一个 app-server client。
- 每张图片创建短生命周期 thread 或复用 provider 内受控 thread。

首版建议每张图一个 ephemeral thread，而不是复用同一个 thread。原因：

- 避免上一张图片上下文污染下一张。
- caption 任务不需要跨图记忆。
- 失败恢复更简单。

优化阶段可引入 thread 池，但必须先证明无上下文污染。

### Backend 2：`exec`

保留现有 `codex exec` 实现：

- 用于 SDK 不可用时手动 fallback。
- 用于快速诊断用户已有 CLI 登录态。
- 不作为默认批量打标路径。

## 依赖与自动安装设计

### uv extra

新增可选 extra：

```toml
[project.optional-dependencies]
codex-subscription = [
  "openai-codex @ git+https://github.com/openai/codex.git@d4f842f3b38595ea348d2a9bc7ed386b74e71a1c#subdirectory=sdk/python",
]
```

说明：

- 当前 PyPI 能解析 runtime 包 `openai-codex-cli-bin==0.131.0a4`，但 SDK wheel 还不能直接解析，所以先用官方仓库 `sdk/python` 子目录的 commit pin。
- OpenAI 发布可解析的 `openai-codex` SDK wheel 后，应把这个 direct reference 切回普通包名或精确版本号。
- 如果 SDK 包名或 runtime 包名变化，只允许在 `codex-subscription` extra 和 adapter 层调整。
- 不把 Codex SDK 放进基础依赖，避免普通用户无故安装大 binary。

### GUI 自动补依赖

参考本项目本地 OCR / VLM / ALM 的依赖补齐方式：

- GUI 中选择 `Codex subscription` 时，将 `codex-subscription` 加入 uv extra profile。
- 如果环境中找不到 SDK，GUI 提示将自动安装该 extra。
- 安装完成后重新执行 caption 任务。

### CLI 自动补依赖

CLI 不应静默联网安装依赖。建议：

- 默认检测 SDK import。
- 缺失时给出明确命令：

```powershell
uv sync --extra codex-subscription
```

或：

```bash
uv sync --extra codex-subscription
```

如果项目已有 CLI 自动安装开关，则只有用户显式打开 `--auto_install_deps` 时才自动执行。

## 认证设计

### 默认：ChatGPT/Codex 订阅登录

默认认证顺序：

1. SDK / app-server 当前 account。
2. `CODEX_HOME` 指向的 Codex auth。
3. 默认 `~/.codex/auth.json`。
4. 如果未登录，提示用户走 SDK 的 `login_chatgpt()` 或 device-code 登录。

实现要求：

- 不读取 auth.json 内容到日志。
- 不打印 access token / refresh token。
- 只展示账号摘要、rate limit 摘要、登录状态。

### API key 登录

支持但不默认，也不作为订阅登录失败后的自动 fallback：

```text
--codex_auth_mode=api_key
--codex_api_key=...
```

或复用 `OPENAI_API_KEY`，但必须同时满足：

```text
--codex_auth_mode=api_key
```

不能仅因为环境里存在 `OPENAI_API_KEY` 就自动切到 API key。

原因：

- API key 登录会走 OpenAI Platform 计费。
- 用户当前目标是使用 Codex 订阅额度。
- 静默 fallback 会造成成本归因错误。
- 订阅登录失败时应直接报 `auth_required` / `auth_failed`，而不是改走 API key。

### 认证模式枚举

```text
codex_auth_mode = "chatgpt" | "api_key" | "existing"
```

- `chatgpt`：默认，要求订阅登录；未登录时触发登录引导；失败时不自动改走 API key。
- `api_key`：显式使用 API key，日志警告“不使用订阅额度”。
- `existing`：只使用当前 app-server / CODEX_HOME 已有账号，不触发登录。

### 认证优先级

默认优先级必须是：

```text
ChatGPT/Codex subscription account
  -> login_chatgpt / device-code 引导
  -> fail closed
```

禁止默认优先级变成：

```text
ChatGPT/Codex subscription account
  -> OPENAI_API_KEY
```

只有当用户明确设置：

```text
--codex_auth_mode=api_key
```

才允许读取 `--codex_api_key` 或 `OPENAI_API_KEY`。

## App-server 生命周期

新增模块：

- `module/providers/codex_app_server.py`
- `module/providers/codex_schema.py`
- `module/providers/cloud_vlm/codex_subscription.py`

### Client 单例范围

首版不做全局 daemon，只在当前 Python 进程内复用：

```text
Caption run starts
  -> lazy create CodexAppServerCaptionClient
  -> process N images
  -> close client on process exit
```

如果当前 pipeline 没有统一 shutdown hook，可以用 context manager 或 `atexit` 兜底关闭。

### 连接缓存 key

```text
(
  codex_backend,
  codex_model_name,
  codex_auth_mode,
  codex_home,
  codex_runtime_path,
  codex_sandbox,
)
```

同一 key 复用连接；key 变化则新建 client。

### 工作目录

默认创建隔离目录：

```text
.cache/qinglong-captions/codex_app_server/work/
```

首版可以使用系统临时目录，但长期建议固定到项目 cache，便于诊断。

要求：

- 不把图片复制进工作目录，`localImage.path` 直接指向原图绝对路径。
- app-server cwd 指向隔离目录。
- 不注入项目 AGENTS.md。

## JSON-RPC Turn 设计

### thread/start

概念 payload：

```json
{
  "model": "gpt-5.4-mini",
  "cwd": "E:/Code/qinglong-captions/.cache/qinglong-captions/codex_app_server/work",
  "ephemeral": true,
  "approvalPolicy": "never",
  "sandbox": "read-only"
}
```

注意：

- 字段名以 SDK 的 Python API 为准，adapter 层负责转换。
- 如果 SDK 不暴露某字段，降级到 config override。

### turn/start

概念 payload：

```json
{
  "threadId": "<thread_id>",
  "model": "gpt-5.4-mini",
  "input": [
    {
      "type": "text",
      "text": "You are a captioning engine. Return only JSON matching the provided schema..."
    },
    {
      "type": "localImage",
      "path": "D:/lora-scripts/input/test/FEB.png"
    }
  ],
  "outputSchema": {
    "type": "object",
    "additionalProperties": false,
    "required": ["short_description", "long_description", "tags", "rating", "confidence"],
    "properties": {
      "short_description": { "type": "string" },
      "long_description": { "type": "string" },
      "tags": { "type": "array", "items": { "type": "string" } },
      "rating": {
        "type": "string",
        "enum": ["general", "sensitive", "questionable", "explicit"]
      },
      "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
    }
  }
}
```

### 输出解析

优先级：

1. SDK `TurnResult.final_response`
2. `turn/completed` 的 final assistant item
3. 最近一个 assistant final message

解析规则复用现有：

- 允许 fenced JSON。
- 允许模型返回对象字符串。
- 字段 normalize 到 `short_description` / `long_description` / `tags` / `rating` / `confidence`。
- `mode=short|long|all` 在 provider 末端过滤。

## Provider 行为

### 启用条件

仍然必须显式启用：

```text
--codex_subscription
```

并且：

- `mime.startswith("image")`
- 未显式选择 `ocr_model`
- 未显式选择 `vlm_image_model`

`document_image=true` 的默认值不能阻止路由；只有 `ocr_model` 明确选择时才让 OCR 路由优先。

### 元数据

返回：

```json
{
  "provider": "codex_subscription",
  "backend": "sdk_app_server",
  "model": "gpt-5.4-mini",
  "auth_mode": "chatgpt",
  "structured": true,
  "schema_version": "codex-caption-v1"
}
```

如果 fallback 到 exec：

```json
{
  "backend": "exec"
}
```

### 并发

默认：

```text
codex_max_concurrency = 1
```

原因：

- 订阅额度不是批量推理 quota。
- app-server turn 状态机需要避免交叉污染。
- 首版先保证正确性。

后续可在确认 SDK 支持多 thread 并发后开放：

```text
codex_max_concurrency = 2..N
```

## 配置设计

PowerShell：

```powershell
$codex_subscription = $false
$codex_backend = "sdk_app_server"  # sdk_app_server, exec
$codex_auth_mode = "chatgpt"       # chatgpt, api_key, existing
$codex_api_key = ""
$codex_model_name = "gpt-5.4-mini"
$codex_home = ""
$codex_runtime_path = ""
$codex_timeout = 180
$codex_sandbox = "read-only"
$codex_isolated_cwd = ""
$codex_output_schema = ""
$codex_max_concurrency = 1
$codex_auto_install_sdk = $false
```

CLI：

```text
--codex_subscription
--codex_backend=sdk_app_server|exec
--codex_auth_mode=chatgpt|api_key|existing
--codex_api_key
--codex_model_name
--codex_home
--codex_runtime_path
--codex_timeout
--codex_sandbox
--codex_isolated_cwd
--codex_output_schema
--codex_max_concurrency
--codex_auto_install_sdk
```

默认原则：

- 不显式打开 `codex_subscription` 时不消耗订阅额度。
- 不显式设置 `codex_auth_mode=api_key` 时不使用 API key。
- 不显式打开 `codex_auto_install_sdk` 时 CLI 不自动安装依赖。
- GUI 可在用户确认后自动安装 SDK extra。

## 安装与预检

### SDK import 预检

```python
try:
    import openai_codex
except ImportError:
    ...
```

预检失败提示：

```text
Codex Python SDK is not installed.
Run: uv sync --extra codex-subscription
```

### Runtime 预检

通过 SDK 创建 `Codex()` 时验证：

- app-server 能启动
- `initialize` 成功
- runtime version 可读
- 当前平台 binary 可执行

失败分类：

- `sdk_missing`
- `runtime_missing`
- `runtime_start_failed`
- `unsupported_platform`
- `protocol_version_mismatch`

### Auth 预检

启动后调用 account/status 能力：

- 已登录：继续。
- 未登录 + `auth_mode=chatgpt`：进入登录引导。
- 未登录 + `auth_mode=existing`：失败并提示登录。
- `auth_mode=api_key`：调用 `login_api_key()`，并标记 metadata / 日志为 API 计费模式。

## 错误分类

统一异常：

```python
class CodexAppServerError(RuntimeError):
    kind: str
    retryable: bool
    detail: str
```

分类：

- `sdk_missing`
- `runtime_missing`
- `auth_required`
- `auth_failed`
- `api_key_billing_mode`
- `usage_limit`
- `rate_limited`
- `timeout`
- `turn_rejected`
- `schema_rejected`
- `output_parse_failed`
- `image_not_found`
- `unsupported_media`
- `transport_closed`
- `protocol_error`

重试策略：

- `rate_limited`：不自动高频重试，尊重 reset hint。
- `transport_closed`：重建 client 后最多重试 1 次。
- `output_parse_failed`：最多重试 1 次，并强化“只返回 JSON”。
- `usage_limit`：不重试。
- `auth_required`：不重试。

## 安全与隐私

必须禁止：

- 日志输出 token。
- 将 auth.json 写入 caption 结果。
- 将用户图片复制到持久 cache，除非用户显式开启 debug。
- 允许 Codex 写项目文件。
- 默认启用 shell / patch / computer-use 工具。

建议：

- app-server cwd 使用隔离目录。
- turn 输入只含当前图片路径和 prompt。
- outputSchema 不允许额外字段。
- metadata 中只放 auth 模式，不放账号邮箱，除非用户打开 debug。

## 模块设计

### `module/providers/codex_schema.py`

职责：

- `CODEX_CAPTION_SCHEMA`
- `CODEX_CAPTION_SCHEMA_VERSION`
- `normalize_codex_caption_payload()`
- `parse_codex_caption_output()`
- `filter_caption_payload_by_mode()`

现有 `codex_exec.py` 里的 schema / parse 逻辑应迁移到这里，避免 exec 与 SDK backend 各自复制。

### `module/providers/codex_app_server.py`

职责：

- SDK import / dependency detection
- app-server client lifecycle
- auth preflight
- `caption_image_with_app_server()`
- SDK result 到 `CodexAppServerCaptionResult` 的转换
- 错误分类

### `module/providers/cloud_vlm/codex_subscription.py`

职责：

- provider route gating
- prompts 构造
- backend 选择
- `CaptionResult` 返回

不应直接处理 SDK wire details。

## 测试设计

### 单元测试

新增：

- `tests/test_codex_schema.py`
- `tests/test_codex_app_server_provider.py`
- `tests/test_codex_subscription_provider.py` 追加 backend 选择测试

覆盖：

1. `codex_subscription=false` 不抢路由。
2. `codex_subscription=true` + image 路由到 provider。
3. OCR / VLM 明确选择时不抢路由。
4. 默认 backend 是 `sdk_app_server`。
5. SDK 缺失错误提示包含 `uv sync --extra codex-subscription`。
6. API key 模式必须显式选择。
7. `OPENAI_API_KEY` 存在但 `auth_mode=chatgpt` 时不调用 `login_api_key()`。
8. `auth_mode=chatgpt` 登录失败时 fail closed，不自动降级到 API key。
9. `thread/start` payload 包含 model / cwd / sandbox / ephemeral。
10. `turn/start` payload 包含 text + localImage + outputSchema。
11. final response JSON 解析和字段 normalize。
12. `transport_closed` 重建 client 最多一次。
13. `usage_limit` 不重试。

### Fake SDK

不要在单元测试里启动真实 Codex。

创建 fake SDK adapter：

```python
class FakeCodex:
    def thread_start(...): ...
```

或在 `codex_app_server.py` 中通过依赖注入替换 SDK client factory。

### Live 测试

显式环境变量启用：

```bash
RUN_LIVE_CODEX_SDK=1 pytest tests/test_codex_app_server_live.py
```

Live 测试只做：

- 检查 SDK / runtime 能启动。
- 检查 account 已登录。
- 用 `D:\lora-scripts\input\test\FEB.png` 或临时小图跑一张。
- 断言输出符合 schema。

不放进默认 CI。

## 验收标准

1. 不安装 Node/NPM 的干净 Python 环境中，`uv sync --extra codex-subscription` 后能获得 Codex runtime。
2. 已登录 ChatGPT/Codex 的机器上，`--codex_subscription --codex_backend=sdk_app_server` 可以处理单张图片。
3. `OPENAI_API_KEY` 存在但未显式 `--codex_auth_mode=api_key` 时不会走 API key。
4. ChatGPT/Codex 订阅登录失败时直接失败并提示登录，不自动切到 API key。
5. app-server 在一次 caption run 中只启动一次。
6. 每张图通过 `thread/start` + `turn/start` 提交，input 包含 `text` 和 `localImage`。
7. `outputSchema` 约束最终输出，provider 返回 `CaptionResult(parsed=...)`。
8. SDK 不可用时能给出清晰安装提示。
9. 用户显式选择 `codex_backend=exec` 时，保留现有 fallback 行为。
10. 默认并发为 1，不会多图并发打爆订阅额度。
11. 单元测试不依赖真实 Codex 网络或登录态。

## 实施顺序

1. 抽出 `module/providers/codex_schema.py`，让 exec backend 和 SDK backend 共用 schema / parse。
2. 在 `pyproject.toml` 增加 `codex-subscription` extra，先 pin 已验证 SDK 版本。
3. 增加 CLI / PowerShell 参数：`codex_backend`、`codex_auth_mode`、`codex_runtime_path`、`codex_max_concurrency`、`codex_auto_install_sdk`。
4. 新增 `module/providers/codex_app_server.py`，先用 fake SDK 单测锁定接口。
5. 改造 `CodexSubscriptionProvider`，默认走 `sdk_app_server`，保留 `exec` fallback。
6. 接入 GUI 依赖自动补齐和 provider 配置项。
7. 添加 live 测试脚本，手动验证 Windows 和 Linux。
8. 跑现有 provider registry / route / catalog 测试，确保不抢其他 provider。

## Linus 风格审查点

- 不要把 API key 做成订阅失败后的自动 fallback；这是账单事故。
- 不要把 SDK 对象散落在 provider 业务代码里；SDK 还实验性，必须隔离在 adapter。
- 不要复用同一个 Codex thread 给多张图片；上下文污染比冷启动省下的时间更贵。
- 不要吞掉 app-server stderr / event error；用户需要知道是没登录、没额度还是 runtime 缺失。
- 不要在基础依赖里塞 Codex runtime；这是可选 provider，不该拖累所有安装。
- 不要直接请求 ChatGPT 私有 backend；短期看省事，长期是维护坑。

## 与旧 spec 的关系

旧文档：

- `2026-05-20-codex-subscription-caption-provider-linux-design.md`

定位调整为：

- `codex exec` fallback 设计记录
- 环境诊断参考
- Windows / WSL 命令行问题复盘

新文档是后续实现主线。
