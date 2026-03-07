# OpenAI Compatible Provider 使用指南

通用 OpenAI API 兼容 Provider，支持对接各种本地或远程的 OpenAI 兼容服务。

## 支持的 Backend

- **vLLM**: `python -m vllm.entrypoints.openai.api_server`
- **SGLang**: `python -m sglang.launch_server`
- **Ollama**: `ollama serve`
- **LM Studio**: 本地服务器模式
- **任何 OpenAI 兼容 API**: 包括 OneAPI、NewAPI 等聚合服务

## 快速开始

### 1. 启动本地服务器（以 vLLM 为例）

```bash
# 安装 vLLM
pip install vllm

# 启动服务器（Qwen2-VL 示例）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --port 8000
```

### 2. 配置 Qinglong Captions

编辑 `4、run.ps1`：

```powershell
# 注释掉其他 API 配置，只保留 OpenAI 兼容配置
# $kimi_api_key = "..."
# $kimi_base_url = "..."

# OpenAI Compatible 配置
$openai_api_key = "sk-no-key-required"  # 本地服务任意值即可
$openai_base_url = "http://localhost:8000/v1"
$openai_model_name = "Qwen/Qwen2-VL-7B-Instruct"
$openai_temperature = 0.7
$openai_max_tokens = 2048
$openai_json_mode = $true
```

### 3. 运行标注

```powershell
.\4、run.ps1
```

## Backend 配置示例

### SGLang（推荐，高性能）

```powershell
# 启动服务器
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000

# 配置
$openai_base_url = "http://localhost:30000/v1"
$openai_model_name = "default"  # SGLang 通常用 "default"
```

### Ollama

```powershell
# 启动服务器
ollama serve

# 拉取模型
ollama pull llava:7b

# 配置
$openai_base_url = "http://localhost:11434/v1"
$openai_model_name = "llava:7b"
$openai_json_mode = $false  # Ollama 可能不支持 JSON 模式
```

### LM Studio

1. 在 LM Studio 中加载模型
2. 开启 "Local Server" 模式
3. 复制服务器地址（通常是 `http://localhost:1234/v1`）

```powershell
$openai_base_url = "http://localhost:1234/v1"
$openai_model_name = "loaded-model"  # LM Studio 中显示的模型名
```

## 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `openai_api_key` | API 密钥（本地服务可任意） | `""` |
| `openai_base_url` | API 基础地址（**必填**） | `""` |
| `openai_model_name` | 模型名称 | `""` |
| `openai_temperature` | 生成温度 (0.0-2.0) | `0.7` |
| `openai_max_tokens` | 最大生成 token 数 | `2048` |
| `openai_json_mode` | 是否使用 JSON 响应格式 | `$true` |

## 支持的模型

### 推荐模型（VLM）

| 模型 | 显存需求 | 特点 |
|------|---------|------|
| Qwen2-VL-2B-Instruct | ~6GB | 轻量，速度快 |
| Qwen2-VL-7B-Instruct | ~16GB | **推荐**，效果好 |
| Qwen2.5-VL-7B-Instruct | ~16GB | 最新版本 |
| MiniCPM-Llama3-V-2_5 | ~8GB | OCR 能力强 |
| LLaVA-1.5-7B | ~14GB | 社区支持好 |

### 推荐模型（Text-only）

如果只是文本 caption，也可以用纯文本模型：

| 模型 | 显存需求 |
|------|---------|
| Qwen2.5-7B-Instruct | ~16GB |
| gemma-3-4b-it | ~10GB |
| Llama-3.2-3B-Instruct | ~8GB |

## 故障排查

### 1. 连接失败

```
API call failed: Connection refused
```

**解决**: 检查服务器是否启动，端口是否正确
```bash
curl http://localhost:8000/v1/models  # 测试连接
```

### 2. JSON 模式不支持

```
Retrying without JSON mode...
```

**解决**: 某些后端（如 Ollama）不支持 JSON 模式，设置 `$openai_json_mode = $false`

### 3. 模型名称错误

```
Model 'xxx' not found
```

**解决**: 检查服务器上加载的模型名称
```bash
curl http://localhost:8000/v1/models  # 查看可用模型
```

### 4. 显存不足

```
CUDA out of memory
```

**解决**: 
- 减小 `--max-model-len`（vLLM）
- 使用量化版本（AWQ、GPTQ）
- 减小 batch size

## 优先级说明

`openai_compatible` provider 在优先级列表中排在最高，**一旦配置了 `openai_base_url` 就会优先使用**。

如果需要临时禁用：
```powershell
$openai_base_url = ""  # 置空即可使用其他 provider
```

## 高级用法

### 远程服务器

可以连接到远程服务器，不限于本地：

```powershell
$openai_base_url = "http://192.168.1.100:8000/v1"
$openai_api_key = "your-secret-key"  # 如果服务器需要认证
```

### 与云 API 混合使用

通过修改优先级或临时禁用，可以在本地和云端之间切换：

```powershell
# 方案1：注释掉本地配置，启用云端
# $openai_base_url = ""
$kimi_api_key = "your-key"
$kimi_base_url = "https://integrate.api.nvidia.com/v1"

# 方案2：通过环境变量切换
if ($env:USE_LOCAL -eq "1") {
    $openai_base_url = "http://localhost:8000/v1"
} else {
    $openai_base_url = ""
    $kimi_api_key = "your-key"
}
```
