# Codex SDK 正式版同步升级 Spec

## 背景

`codex_subscription` 当前已经有两条后端：

- `sdk_app_server`：默认路径，通过 `openai_codex` Python SDK / app-server 长驻连接打标。
- `exec`：兼容路径，通过 `codex exec` 子进程单次调用打标。

2026-06-03 重新核对官方资料后，Codex Python SDK 已从“只能 GitHub 子目录 pin 的实验形态”进入正式发布路径。官方 SDK 页面和 OpenAI Codex GitHub 仓库都指向同一个新事实：

- Python SDK 安装方式应是 `pip install openai-codex`。
- 发布版 SDK 会自动使用 pin 住的 Codex CLI runtime；官方文档仍写 `AppServerConfig(codex_bin=...)`，但 PyPI `openai-codex==0.1.0b2` 实际导出的是同构造参数的 `CodexConfig`。
- 公共调用面是 `Codex` / `AsyncCodex`、`Sandbox`、`TextInput`、`LocalImageInput`、`Thread.run(...)`、`TurnResult.final_response`；其中 `Sandbox` 是当前文档公共面，但当前可安装的 `0.131.0a4` 可能缺失。

这意味着本项目旧 spec 中“SDK wheel 不能解析，所以只能长期 pin GitHub commit”的产品方向已经过期；但在当前环境能解析发布包前，GitHub pin 仍是必要过渡。

## 参考来源

- 官方 SDK 文档：<https://developers.openai.com/codex/sdk>
- 官方 SDK Markdown：<https://developers.openai.com/codex/sdk.md>
- 官方 Python SDK README：<https://github.com/openai/codex/tree/main/sdk/python>
- 官方 Python SDK API reference：<https://github.com/openai/codex/blob/main/sdk/python/docs/api-reference.md>
- 官方 Python SDK `pyproject.toml`：<https://github.com/openai/codex/blob/main/sdk/python/pyproject.toml>

## 本地当前事实

### 依赖状态

`pyproject.toml` 已可切换为 PyPI 包：

```toml
codex-subscription = [
    "openai-codex",
]
```

本地历史 alpha `.venv` 曾安装 `openai_codex==0.131.0a4`。该版本：

- 导出 `Codex`、`ApprovalMode`、`TextInput`、`LocalImageInput`。
- 不导出当前官方文档里的 `Sandbox`。
- `Thread.run(...)` 签名仍是旧形态，包含 `sandbox_policy`，不是当前文档中的 `sandbox`。
- 导出 `AppServerConfig`；官方 `sdk.md` 也写的是 `AppServerConfig(codex_bin=...)`。`CodexConfig` 不是官方文档或本地 SDK 中存在的符号。

2026-06-04 复核 PyPI 安装后，本地 `.venv` 中 `openai_codex==0.1.0b2` 的事实是：

- `openai-codex` 的 distribution metadata 声明 `Requires-Dist: openai-codex-cli-bin==0.132.0`，CLI runtime 已拆成单独包但会作为依赖自动安装。
- `openai-codex-cli-bin==0.132.0` 已安装，说明这次报错不是缺 CLI runtime。
- 顶层导出 `CodexConfig`，不导出 `AppServerConfig`。
- `Codex` 签名是 `Codex(config: CodexConfig | None = None)`。
- `Thread.run(...)` 已包含 `sandbox=`，且顶层导出 `Sandbox`。

因此实现门槛不能再硬编码 `AppServerConfig`；应接受 `CodexConfig` 或 `AppServerConfig` 任一配置类。错误提示中的 `uv sync --extra codex-subscription` 只是本项目安装建议，不代表 pip 诊断出的缺失依赖。

### 代码状态

主要实现位于：

- `module/providers/codex_app_server.py`
- `module/providers/cloud_vlm/codex_subscription.py`
- `module/providers/codex_exec.py`

当前 `codex_app_server.py` 的适配层仍然保留大量旧协议兼容逻辑：

- 构造 `AppServerConfig` 时尝试 `codex_bin` / `codexBin` / `env` / 空配置等多种形状。
- 创建 client 时尝试 `CodexOptions`、`app_server`、`appServer`、`config`、空构造等多种形状。
- thread/turn 调用同时尝试 `thread_start`、`start_thread`、`threadStart`、`turn_start`、`start_turn`、`turnStart` 等低层命名。
- `Thread.run` 调用同时尝试 `output_schema` / `outputSchema`、`service_tier` / `serviceTier`、`approval_mode` 等多组参数。
- 为了流式进度，代码直接导入私有模块 `openai_codex._run._collect_turn_result`。

这些兼容分支在 SDK 未稳定时合理，但现在会制造三个问题：

1. 公共 SDK 契约被稀释，测试会验证“猜测的旧协议形状”，而不是验证官方 API。
2. 私有 `_run` 依赖可能在正式 SDK 小版本更新时破裂。
3. 当前文档的 `Sandbox` 预设无法被清晰映射，`read-only` 字符串会继续穿透到旧 app-server 形态。

## 第一性原理判断

项目的真实目标不是“尽可能接近 Codex 内部 JSON-RPC”，而是“用已登录 Codex / ChatGPT 订阅会话为图片产生稳定结构化 caption”。

所以升级后的正确边界应是：

1. 依赖层以发布版 `openai-codex` 包为目标；在目标环境尚不能解析发布包前，保留 GitHub commit pin 作为过渡。
2. 业务层只调用官方公共 API：`Codex(CodexConfig/AppServerConfig(...))`、`thread_start(...)`、`thread.run(...)`。
3. JSON-RPC、私有 collector、旧 camelCase payload 只保留在 `exec` fallback 或临时兼容测试中，不应是默认主路径。
4. SDK 缺少硬性公共符号时，应明确报“SDK 不满足最低契约”，而不是继续猜内部协议；软性漂移参数通过能力探测处理。
5. Caption provider 不需要 Codex 写文件或执行修改，默认 sandbox 语义应是 read-only；具体传入值必须按当前 SDK 能力映射，可以是 `Sandbox.read_only`，也可以是旧版 SDK 接受的字符串 / policy 值。

## 目标

1. 验证目标环境能解析发布版 `openai-codex`；验证通过后，将 `codex-subscription` extra 从 GitHub direct reference 切到发布版包。
2. 将 app-server adapter 收敛到正式 Python SDK 公共 API。
3. 使用 SDK 导出的配置类构造 SDK client：PyPI beta 优先 `CodexConfig`，历史 alpha 兼容 `AppServerConfig`；保留本项目自己的 `CodexAppServerConfig` 数据类作为 provider 配置边界。
4. 将本项目的 `codex_sandbox` 字符串映射为当前 SDK 后端实际能接受的值；`Sandbox` 存在时优先使用预设，缺失时兼容旧版字符串 / policy 参数。
5. 默认通过 `with Codex(...) as codex` 管理 SDK 生命周期；池化 client 也必须明确 close。
6. 使用 `thread.run(input, output_schema=..., service_tier=..., model=..., effort=...)` 获取 `TurnResult.final_response`；sandbox 参数名通过能力探测决定是 `sandbox=`、`sandbox_policy=`，还是只在 `thread_start(...)` 上设置。
7. 保留 `exec` backend 作为手动 fallback。
8. 测试覆盖依赖版本诊断、正式 SDK 调用形状、Sandbox 映射、结构化输出解析、超时/429/登录失败分类。

## 非目标

- 不在这次升级里重写 caption schema。
- 不新增对视频、音频、OCR 文档页的 Codex subscription 支持。
- 不把 `OPENAI_API_KEY` 作为订阅登录失败后的自动 fallback。
- 不调用 ChatGPT / Codex 私有 HTTP 接口。
- 不为了进度日志继续依赖 `openai_codex._run` 私有模块。
- 不默认提高 Codex 并发；订阅额度仍然默认保守消费。

## 设计方案

### 1. 依赖升级

目标状态是把 `pyproject.toml` 中的 `codex-subscription` extra 改为发布版包；2026-06-04 本地已验证该包可安装：

```toml
codex-subscription = [
    "openai-codex",
]
```

迁移前仍应做解析验证：

```powershell
uv sync --extra codex-subscription
```

如果目标机器的包源不能解析 `openai-codex`，才保持 GitHub direct reference 并把“切发布包”作为阻塞项记录。当前机器已解析出 `openai-codex==0.1.0b2` 与 `openai-codex-cli-bin==0.132.0`。

如果项目策略要求源文件 pin，可以使用已验证的下限，例如：

```toml
"openai-codex>=0.1.0b2,<1"
```

是否写下限取决于实际 lock 结果。必须满足：

- SDK 导出 `CodexConfig` 或 `AppServerConfig`。
- SDK 导出 `TextInput` 和 `LocalImageInput`。
- SDK 返回 `TurnResult.final_response`。
- 发布 SDK 自动带 pinned Codex CLI runtime；具体 runtime 包名以发布 metadata 为准，不在本项目中写死。
- `Sandbox` 和 `Thread.run(..., sandbox=...)` 是期望的新公共形态，但不能作为硬门槛；旧版 `0.131.0a4` 缺少 `Sandbox` 且使用 `sandbox_policy=`，仍应能通过兼容路径运行。

### 2. SDK 版本门槛

`load_openai_codex_sdk()` 应检查公共 API，而不是只检查 import 成功。

最低要求：

- `openai_codex.Codex`
- `openai_codex.CodexConfig` 或 `openai_codex.AppServerConfig`
- `openai_codex.TextInput`
- `openai_codex.LocalImageInput`
- `openai_codex.TurnResult`
- `openai_codex.retry_on_overload` 或 `is_retryable_error`

软要求：

- `openai_codex.Sandbox`
- `Thread.run` 签名包含 `sandbox`

软要求缺失时不能直接判定 SDK 太旧；应进入兼容路径，并记录 diagnostics。硬要求缺失才返回明确错误。

缺失时返回明确错误：

```text
Codex Python SDK is missing required public SDK symbols.
Run: uv sync --extra codex-subscription
```

如果当前 pip 源不能解析 `openai-codex`，错误消息应提示检查 Python package index；但在实现迁移前，不应自动删除仍可解析的 GitHub pin。

### 3. Sandbox 映射

本项目 CLI/GUI 仍可以保留字符串配置：

```text
read-only
workspace-write
danger-full-access
```

adapter 内部先映射为语义值：

```text
read-only          -> read_only
workspace-write    -> workspace_write
danger-full-access -> full_access
```

兼容别名：

```text
read_only          -> read_only
workspace_write    -> workspace_write
full-access        -> full_access
full_access        -> full_access
```

最终传入值按 SDK 能力决定：

- 如果 `openai_codex.Sandbox` 存在，优先传 `Sandbox.read_only` / `Sandbox.workspace_write` / `Sandbox.full_access`。
- 如果 `Sandbox` 不存在，但 `openai_codex.types.SandboxMode` 存在，则 thread 级 sandbox 使用 `SandboxMode.read_only` / `workspace_write` / `danger_full_access`。
- 如果 `Thread.run` 只有 `sandbox_policy`，只有在能从 `openai_codex.generated.v2_all` 构造 `SandboxPolicy(ReadOnlySandboxPolicy(type="readOnly"))` 等对象时才传 `sandbox_policy`。
- 如果当前 SDK 不能构造 run 级 sandbox policy，则只在 `thread_start(...)` 上设置 sandbox；不要向 run 级传裸字符串。

不支持的值直接报 `kind="config"`。

### 4. Client 构造

新实现只保留正式构造形态：

```python
import openai_codex

sdk_config_cls = getattr(openai_codex, "CodexConfig", None) or getattr(openai_codex, "AppServerConfig")
sdk_config = sdk_config_cls(
    codex_bin=runtime_path or None,
    cwd=isolated_cwd or None,
    env=sanitized_env,
)
client = openai_codex.Codex(sdk_config)
```

必须继续清理 API key 环境：

- 默认 `chatgpt` / `existing` 模式移除 `OPENAI_API_KEY`、`CODEX_API_KEY`。
- `api_key` 模式只有用户显式选择时才把 `OPENAI_API_KEY` 注入 SDK env，并调用 `codex.login_api_key(...)`。
- `CODEX_HOME` 仍然可通过 env 注入。

### 5. Thread 和 Turn 调用

每张图片仍使用 ephemeral thread，避免跨图上下文污染：

这里的 `provider_config` 指本项目自己的 `CodexAppServerConfig`，`sdk_config` 才是传给 SDK 的 `CodexConfig` / `AppServerConfig`。

```python
thread = codex.thread_start(
    model=provider_config.model,
    sandbox=sandbox,
    ephemeral=True,
    cwd=isolated_cwd,
    config={"tools": {"view_image": True}},
)

result = thread.run(
    [
        TextInput(text=prompt),
        LocalImageInput(path=str(image_path.resolve())),
    ],
    model=provider_config.model,
    effort=reasoning_effort,
    service_tier=service_tier or None,
    output_schema=output_schema,
    # sandbox kwarg is added only when the current SDK signature accepts it.
    **run_sandbox_kwargs,
)
raw = result.final_response or ""
```

`run_sandbox_kwargs` 由能力探测生成：

```text
Thread.run has "sandbox"        -> {"sandbox": sandbox_value}
Thread.run has "sandbox_policy" and SDK policy object is constructible -> {"sandbox_policy": sandbox_policy_value}
otherwise                       -> {}
```

不要把当前文档示例中的 `sandbox=` 直接当成所有可安装版本的稳定契约；也不要把 `read-only` 裸字符串当成 `sandbox_policy` 的合法值。

如果 `final_response` 为空：

- 先从 `result.items` 中按官方类型抽取最后一个 assistant final message。
- 仍为空则报 `kind="output"`，不要把过程事件拼成 caption。

### 6. 进度日志

正式主路径不再导入 `openai_codex._run._collect_turn_result`。

两种可选方案：

- 简化：只记录阶段进度，不记录 token / delta 事件。
- 低层可选：使用公共 `thread.turn(...)` + `TurnHandle.stream()`，由本项目自己收集事件，再调用公共 `TurnHandle.run()` 或等价公共结果收集。不得依赖 `_run` 私有模块。

首版同步建议采用简化方案，先保证 SDK 契约稳定。

### 7. 重试和限流

保留当前 429 分类增强，但改为优先使用 SDK 公共错误 helper：

- `openai_codex.is_retryable_error(exc)`
- `openai_codex.retry_on_overload(...)` 仅在需要 SDK 内部重试包装时使用。

本项目 provider 级重试仍负责：

- 429 / overload 等待 `wait_time`。
- timeout / transport 失败重置 client。
- rate limit 重试耗尽后返回 skip result，而不是中断整个批处理。

### 8. 测试更新

需要新增或修改测试：

1. `test_codex_subscription_dependency_uses_published_sdk`
   - 只在解析验证通过后断言 `pyproject.toml` 不再包含 `git+https://github.com/openai/codex.git`。
   - 断言目标 extra 包含 `openai-codex`，或在包源不可达时用 `pytest.skip` / `pytest.xfail(strict=True, reason=...)` 明确标记 blocked，而不是让安装失败或假绿。

2. `test_codex_sdk_requires_published_contract`
   - fake SDK 缺少 `Codex` / `CodexConfig or AppServerConfig` / `TextInput` / `LocalImageInput` / `TurnResult` 时，抛出 `sdk_too_old` 或 `sdk_missing` 类错误。
   - fake SDK 缺少 `Sandbox` 时，不应抛硬错误，应进入字符串 / policy 兼容路径。

3. `test_codex_sandbox_string_maps_to_sdk_presets`
   - 覆盖 `read-only`、`workspace-write`、`danger-full-access`。

4. `test_codex_app_server_uses_public_thread_run_shape`
   - fake `Codex` 只实现 `thread_start` 和 `Thread.run`。
   - 断言没有调用 `turn_start`、`start_thread`、`outputSchema` 等旧形态。
   - sandbox 参数断言应按 fake `Thread.run` 签名分支分别覆盖 `sandbox=`、`sandbox_policy=` 和 run 级 sandbox 缺失三种情况。

5. `test_codex_app_server_reads_final_response`
   - fake `TurnResult(final_response='...')` 被解析为 caption JSON。

6. `test_codex_app_server_rejects_empty_final_response`
   - final response 为空且无可抽取 assistant item 时，返回 output error。

7. 保留现有：
   - 429 retryable 分类。
   - timeout skip result。
   - API key 环境隔离。
   - HEIC/AVIF 转 JPEG。
   - 默认串行并发。

### 9. 迁移顺序

推荐分四步，不要一次性重写：

1. 更新 spec 和测试预期。
2. 在目标环境验证 `openai-codex` 发布包能被 `uv sync --extra codex-subscription` 真实解析。
3. 解析成功后再改 `pyproject.toml` extra 并刷新 lock；解析失败则保留 GitHub pin，记录 blocked。
4. 改 `codex_app_server.py` 到公共 API + 能力探测路径。
5. 跑 targeted tests，再做一次真实单图 smoke test。

targeted tests：

```powershell
pytest tests/test_codex_subscription_provider.py tests/test_penguin_dependencies.py tests/test_cloud_concurrency_config.py -q
```

真实 smoke：

```powershell
python -m module.captioner <dataset> --codex_subscription --codex_backend=sdk_app_server --codex_sandbox=read-only --max_retries=1
```

## 风险

1. 当前用户环境的 pip 源可能不能解析 `openai-codex`。
   - 处理：这是迁移阻塞项，不只是错误文案问题。先验证解析；失败时保留 GitHub pin。

2. 官方 SDK 仍是 Beta，小版本可能改变公共面。
   - 处理：硬门槛只检查真实公共符号；可漂移参数用 `inspect.signature` 能力探测。

3. 当前实现依赖私有 streaming collector，移除后进度日志会变少。
   - 处理：先换稳定性；如果需要详细进度，再用公共 `turn(...).stream()` 重建。

4. `Sandbox` 符号和 run 级 `sandbox=` 参数在文档与本地可安装版本之间存在漂移。
   - 处理：`Sandbox` 是软能力；run 级 sandbox 参数必须能力探测，不能写死。

5. `Sandbox.full_access` 与项目安全默认冲突。
   - 处理：默认仍是 `read-only`；full access 只作为用户显式配置。

6. API key 模式容易误伤订阅计费目标。
   - 处理：继续要求 `--codex_auth_mode=api_key` 显式选择，不做自动 fallback。

## 验收标准

升级完成后应满足：

- `pyproject.toml` 的 `codex-subscription` extra 不再包含 GitHub direct reference。
- 上一条只在 `openai-codex` 发布包可被目标环境解析后成立；解析失败时，验收状态应标记 blocked，而不是破坏当前可安装环境。
- `load_openai_codex_sdk()` 能识别真实公共契约；缺少 `CodexConfig` / `AppServerConfig` 等硬符号时给出明确错误。
- app-server 主路径不调用 `openai_codex._run`、`turn_start` fallback、camelCase payload fallback。
- `codex_sandbox=read-only` 实际传入当前 SDK 后端可接受的 read-only sandbox 值；不绑定必须是 `Sandbox.read_only`。
- `thread.run(...)` 返回的 `final_response` 是结构化 caption 解析来源。
- `exec` backend 仍可独立工作。
- targeted tests 通过。

## 结论

需要修改本地代码同步最新 SDK，但不能把任何单一文档/alpha/PyPI 符号当绝对真理。正确版本是：发布包可解析时使用 `openai-codex`；承认 SDK/runtime 已拆成两个 distribution 且 runtime 由 metadata 自动依赖；配置类接受 `CodexConfig` 或 `AppServerConfig`；把 `Sandbox` 和 run 级 sandbox 参数作为能力探测；再把 app-server adapter 从旧 JSON-RPC 兼容层收敛到公共 `Codex` / `thread_start` / `thread.run` 路径。这样既能提高代码品味，又不破坏当前可用 provider。
