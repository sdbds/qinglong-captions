# OpenAI-compatible Provider 指南

`openai_compatible` 用统一的 Chat Completions 接口连接本地或远程视觉模型服务。支持 vLLM、SGLang、Ollama、LM Studio 以及其他兼容服务。

## 最小配置

先启动一个兼容服务，再在 GUI `Caption` 页面或 `4.captioner.ps1` 中设置：

```powershell
$openai_api_key = $env:LOCAL_VLM_API_KEY
$openai_base_url = "http://127.0.0.1:8000/v1"
$openai_model_name = "Qwen/Qwen2-VL-7B-Instruct"
```

`4.captioner.ps1` 和 `module.captioner` 当前只暴露上面三个 OpenAI-compatible 参数。Provider 的生成参数目前使用代码默认值：`temperature=0.7`、`max_tokens=2048`，并优先请求 JSON response format；这些值不能通过脚本顶部变量或 CLI 参数覆盖。

本地服务不需要认证时，`openai_api_key` 可以填任意非敏感占位值；远程服务请通过环境变量或受控凭据注入真实 Key，不要写进脚本或日志。

运行：

```powershell
.\4.captioner.ps1
```

## 服务示例

### vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --port 8000
```

Base URL 使用 `http://127.0.0.1:8000/v1`。模型名必须与服务端 `/v1/models` 返回的 ID 一致。

### SGLang

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port 30000
```

Base URL 使用 `http://127.0.0.1:30000/v1`。如服务端返回的模型 ID 是 `default`，就把 `openai_model_name` 设置为 `default`。

### Ollama

```bash
ollama serve
ollama pull llava:7b
```

```powershell
$openai_base_url = "http://127.0.0.1:11434/v1"
$openai_model_name = "llava:7b"
```

### LM Studio

1. 在 LM Studio 中加载模型。
2. 开启 Local Server。
3. 使用服务显示的模型 ID 和 `http://127.0.0.1:1234/v1`。

## 参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `openai_api_key` | 服务认证 Key；本地服务可使用占位值 | 空值 |
| `openai_base_url` | OpenAI API 基础地址；配置后启用该 Provider | 空值 |
| `openai_model_name` | 服务端模型 ID；脚本和 CLI 默认是空值，应显式填写 | 空值 |
| 生成参数 | `temperature=0.7`、`max_tokens=2048`；当前未暴露为脚本或 CLI 参数 | Provider 代码默认值 |
| JSON response format | Provider 会优先请求 JSON；服务端拒绝时自动去掉 `response_format` 重试 | 自动处理 |

图像输入存在显式本地 `vlm_image_model` 或 `ocr_model` 时，路由会优先使用显式本地 Provider，而不是 `openai_compatible`。视频输入仍需要兼容服务支持视觉消息。

## 连接检查

```bash
curl http://127.0.0.1:8000/v1/models
```

- `Connection refused`：服务未启动、端口错误或服务只监听了其他地址。
- `Model not found`：使用 `/v1/models` 返回的确切 ID。
- JSON 模式失败：Provider 会自动去掉 `response_format` 重试；如果服务仍拒绝，请检查服务端对视觉消息和 Chat Completions 的兼容程度。
- CUDA out of memory：降低服务端上下文长度、并行度或使用量化模型。

## 远程服务安全

远程 Base URL 不要使用未经保护的公网 HTTP。服务端应限制监听地址、启用认证和 TLS，并在防火墙中只允许必要来源。不要把 `--cloud` GUI 与一个没有认证的远程模型端点一起暴露。
