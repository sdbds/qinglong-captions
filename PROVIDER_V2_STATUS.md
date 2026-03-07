# Provider V2 重构完成报告

## 重构概览

成功将 `module/api_handler.py` (2184 行单体文件) 重构为模块化 Provider 架构。

## 目录结构

```
module/providers/
├── __init__.py              # 导出主要类
├── base.py                  # Provider, CaptionResult, MediaContext 等核心抽象
├── capabilities.py          # ProviderCapabilities
├── registry.py              # ProviderRegistry (单例 + 缓存)
├── resolver.py              # PromptResolver (集中化 prompt 选择)
├── utils.py                 # 工具函数
├── cloud_vlm_base.py        # CloudVLMProvider 基类
├── local_vlm_base.py        # LocalVLMProvider 基类
├── ocr_base.py              # OCRProvider 基类
├── vision_api_base.py       # VisionAPIProvider 基类
├── api_handler_v2.py        # 新入口点
├── cloud_vlm/               # 云端 VLM Providers
│   ├── __init__.py
│   ├── stepfun.py
│   ├── qwenvl.py
│   ├── glm.py
│   ├── ark.py
│   ├── kimi_code.py         # kimi_code (优先级高于 kimi_vl)
│   └── kimi_vl.py
├── local_vlm/               # 本地 VLM Providers
│   ├── __init__.py
│   ├── moondream.py
│   ├── qwen_vl_local.py
│   └── step_vl_local.py
├── ocr/                     # OCR Providers (8个)
│   ├── __init__.py
│   ├── deepseek.py
│   ├── hunyuan.py
│   ├── glm.py
│   ├── chandra.py
│   ├── olmocr.py
│   ├── nanonets.py
│   ├── firered.py
│   └── paddle.py
└── vision_api/              # 视觉 API Providers
    ├── __init__.py
    ├── pixtral.py
    └── gemini.py
```

## Provider 统计

| 类别 | 数量 | 列表 |
|------|------|------|
| OCR | 8 | deepseek_ocr, hunyuan_ocr, glm_ocr, chandra_ocr, olmocr, nanonets_ocr, firered_ocr, paddle_ocr |
| Cloud VLM | 6 | stepfun, qwenvl, glm, ark, kimi_code, kimi_vl |
| Vision API | 2 | pixtral, gemini |
| Local VLM | 3 | moondream, qwen_vl_local, step_vl_local |
| **总计** | **19** | |

## 核心修复点

### P0 阻塞问题 (全部修复)
- ✅ **#1 kimi_code 缺失** - 添加 kimi_code provider，优先级高于 kimi_vl
- ✅ **#3 返回值类型多态** - CaptionResult dataclass 统一返回类型
- ✅ **#18 Registry 性能** - @lru_cache + 模块级单例

### P1 重要问题 (全部修复)
- ✅ **#2 音频处理缺失** - MediaContext.audio_blob + video_file_refs
- ✅ **#4 Gemini Task 模板** - PromptResolver._apply_gemini_task_template()
- ✅ **#5 结构化输出** - StructuredOutputConfig + VisionAPIProvider
- ✅ **#13 Prompt 选择集中化** - PromptResolver 类
- ✅ **#19 can_handle_static** - @classmethod can_handle()
- ✅ **#21 LocalVLM 基类** - LocalVLMProvider 基类
- ✅ **#8 模型加载器缓存** - _global_model_cache + 线程锁

### P2 建议问题 (大部分修复)
- ✅ **#6 流式输出** - ProviderCapabilities.supports_streaming
- ✅ **#9 attention 参数** - _attn_implementation 类属性
- ✅ **#10 Pixtral 后验证** - Provider.post_validate() 钩子
- ✅ **#11 Gemini inline_data** - 保留副作用

## 使用方式

### 新架构入口

```python
from module.api_handler_v2 import api_process_batch

result = api_process_batch(
    uri="image.jpg",
    mime="image/jpeg",
    config=config,
    args=args,
    sha256hash="abc123"
)

# result 是 CaptionResult 对象
print(result.raw)           # 原始返回
print(result.description)   # 描述文本
print(result.parsed)        # 结构化数据（如果有）
```

### 切换新旧代码

```bash
# 使用新架构
export QINGLONG_API_V2=1
python module/captioner.py ...

# 使用旧代码（回退）
export QINGLONG_API_V2=0
python module/captioner.py ...
```

## 新增功能

### 1. CaptionResult - 统一返回类型
```python
@dataclass
class CaptionResult:
    raw: str                    # 原始返回文本
    parsed: Optional[Dict]      # 解析后的结构化数据
    metadata: Dict[str, Any]    # 额外元数据
    
    @property
    def description(self) -> str: ...
```

### 2. PromptResolver - 集中化 Prompt 选择
```python
resolver = PromptResolver(config, provider_name)
prompts = resolver.resolve(mime, args, character_prompt)
```

### 3. Provider 分类基类
- **CloudVLMProvider** - 云端 VLM（支持视频上传、Pair 图像）
- **LocalVLMProvider** - 本地 VLM（支持模型缓存、设备选择）
- **OCRProvider** - OCR（统一的输出目录处理）
- **VisionAPIProvider** - 视觉 API（支持结构化输出）

### 4. 注册表自动发现
```python
from providers import get_registry

registry = get_registry()  # 单例，自动缓存
provider_class = registry.find_provider(args, mime)
```

## 向后兼容性

- ✅ 函数签名与原 api_process_batch 完全一致
- ✅ 可通过环境变量切换新旧代码
- ✅ CaptionResult 可以通过 .raw 属性获取字符串
- ✅ 失败时可自动回退到旧实现

## 下一步建议

1. **集成测试** - 在实际数据集上测试新架构
2. **性能基准** - 对比新旧架构的性能
3. **逐步切换** - 先在小范围使用新架构，稳定后全面切换
4. **文档更新** - 更新项目 README 和 API 文档

## 文件统计

| 类型 | 文件数 | 代码行数（约） |
|------|--------|----------------|
| 基础设施 | 9 | ~2000 行 |
| OCR Providers | 8 | ~400 行 |
| Cloud VLM | 6 | ~600 行 |
| Vision API | 2 | ~300 行 |
| Local VLM | 3 | ~300 行 |
| **总计** | **28** | **~3600 行** |

原 `api_handler.py`: 2184 行单体文件
新架构: 28 个文件，职责分离，可维护性大幅提升

---

重构完成日期: 2026-03-07
