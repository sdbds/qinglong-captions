# ONNX Runtime 工具统一设计

## 背景

当前仓库里已经存在一层初步的 ONNX Runtime 基础设施：

- [`module/onnx_runtime/artifacts.py`](E:/Code/qinglong-captions/module/onnx_runtime/artifacts.py)
- [`module/onnx_runtime/config.py`](E:/Code/qinglong-captions/module/onnx_runtime/config.py)
- [`module/onnx_runtime/session.py`](E:/Code/qinglong-captions/module/onnx_runtime/session.py)

这套基础层已经被 [`module/providers/local_vlm/lfm_vl_local.py`](E:/Code/qinglong-captions/module/providers/local_vlm/lfm_vl_local.py) 使用，负责：

- ONNX artifact 下载
- execution provider 选择
- session bundle 创建与缓存

但其他 ONNX 推理工具仍然各自维护一套运行时逻辑，主要包括：

- [`utils/wdtagger.py`](E:/Code/qinglong-captions/utils/wdtagger.py)
- [`module/waterdetect.py`](E:/Code/qinglong-captions/module/waterdetect.py)

这些工具内部仍然直接处理：

- `onnxruntime` provider 分支选择
- `SessionOptions` 拼装
- TensorRT / CUDA / OpenVINO 参数
- `InferenceSession` 初始化
- 模型下载路径与运行时 cache 路径

结果是：

- 相同的 runtime 逻辑重复出现在多个工具中
- provider 参数和 cache 约定无法统一管理
- 后续继续接入新的 ONNX 工具时，还会重复复制这一层代码

本次设计的目标，是在不改动外部 CLI / GUI / PowerShell 调用方式的前提下，把仓库内所有 “和 ONNX Runtime 相关的部分” 抽成统一基础层。

## 已确认决策

- 统一范围只覆盖 ONNX Runtime 相关部分，不改外部 CLI 形态
- 采用“窄而完整的单模型 ONNX 基座”方案，不做通用 pipeline 框架
- `wdtagger` 和水印检测优先复用单模型基座
- `lfm_vl_local` 继续保留多文件 session bundle 形态，但共享底层 runtime/config 能力
- ONNX runtime 配置从 `model.toml` 中拆出，单独放入新的 `config/onnx.toml`
- `onnx.toml` 会成为正式 split config 文件的一部分，而不是工具自己单独读取
- 新旧 runtime 配置会并存一个兼容期：`model.toml` 中的旧字段继续生效，`onnx.toml` 优先级更高，并输出弃用提示
- 第一版只统一 runtime 行为，不统一各任务的预处理 / 后处理 / 输出契约

## 目标

1. 为单模型 ONNX 工具提供统一的装配入口。
2. 把 execution provider、session options、provider options、cache 目录等 runtime 行为集中管理。
3. 把 ONNX runtime 相关配置迁移到独立的 [`config/onnx.toml`](E:/Code/qinglong-captions/config/onnx.toml)。
4. 让 [`utils/wdtagger.py`](E:/Code/qinglong-captions/utils/wdtagger.py) 和 [`module/waterdetect.py`](E:/Code/qinglong-captions/module/waterdetect.py) 删除本地重复的 runtime 逻辑。
5. 保持 [`module/providers/local_vlm/lfm_vl_local.py`](E:/Code/qinglong-captions/module/providers/local_vlm/lfm_vl_local.py) 的现有多文件 artifact 结构和测试不被破坏。

## 非目标

- 本次不重构 `wdtagger` 的标签解析与后处理逻辑
- 本次不重构水印检测的 processor、概率解释和结果落盘逻辑
- 本次不新建通用的 ONNX preprocess / infer / postprocess 框架
- 本次不修改 GUI / `4、run.ps1` / 其他外部入口的参数接口
- 本次不把所有 CUDA / TensorRT 参数都暴露成用户级可配置项

## 现状问题

### 1. 运行时逻辑重复

[`utils/wdtagger.py`](E:/Code/qinglong-captions/utils/wdtagger.py) 和 [`module/waterdetect.py`](E:/Code/qinglong-captions/module/waterdetect.py) 各自维护 provider 选择、session options 构建和 session 初始化逻辑。相同的 TensorRT / CUDA 分支逻辑已经出现多份实现。

### 2. 配置边界混杂

当前 `lfm_vl_local` 把 `execution_provider` 放在 [`config/model.toml`](E:/Code/qinglong-captions/config/model.toml) 的模型配置里，这会把“模型语义配置”和“runtime 行为配置”混在一起。对 `wdtagger` / 水印检测来说，也没有一份统一的 runtime 配置来源。

### 3. split config 未包含 ONNX 配置文件

当前 [`config/loader.py`](E:/Code/qinglong-captions/config/loader.py) 只会加载：

- `prompts.toml`
- `model.toml`
- `general.toml`

如果新增 ONNX 专属配置文件，但不把它纳入正式合并流程，则统一配置仍然不可复用。

## 架构设计

### 1. 保留三层基础职责

现有基础层继续保持清晰边界：

- `artifacts.py`
  - 只负责 artifact 文件名推导、外部数据文件发现和下载
- `config.py`
  - 只负责 runtime 配置解析与默认值
- `session.py`
  - 只负责 provider 选择、session options 构建、session 创建与缓存

在这三层之上新增“单模型 ONNX 基座”，避免把单模型装配逻辑继续散落在工具脚本中。

### 2. 新增单模型 ONNX 基座

新增文件：

- [`module/onnx_runtime/single_model.py`](E:/Code/qinglong-captions/module/onnx_runtime/single_model.py)

职责：

- 描述单模型 ONNX 工具需要的 artifact 与 cache key
- 调用统一 artifact 下载能力
- 调用统一 runtime/session 能力
- 返回一个可直接供工具层使用的 bundle

### 3. 核心对象

#### `OnnxModelSpec`

用于描述“这个工具要加载什么模型”：

- `repo_id`
- `model_dir`
- `filename` 或 `artifacts`
- `bundle_key`
- 可选的额外标识字段

单模型工具的典型场景：

- `wdtagger`: `model.onnx`
- `waterdetect`: `model.onnx`

多模型工具如 `lfm_vl_local` 不强行改成单模型 spec，继续使用现有多 artifact session bundle 方案。

#### `SingleModelOnnxBundle`

用于描述“加载完成以后工具层拿到什么”：

- `model_path`
- `session`
- `providers`
- `input_metas`
- `runtime_config`

这样工具层只需关心：

- 如何准备输入
- 如何解释输出
- 如何写结果

而不再自己创建 `InferenceSession(...)`。

#### `load_single_model_bundle(...)`

统一入口，内部完成：

1. 基于 `OnnxModelSpec` 解析模型路径
2. 下载或复用本地 ONNX 文件
3. 解析共享 runtime 配置
4. 构建 provider 列表和 provider options
5. 创建单模型 `InferenceSession`
6. 根据 bundle key 做 session cache

## 配置设计

### 1. 新增 `config/onnx.toml`

新增独立配置文件：

- [`config/onnx.toml`](E:/Code/qinglong-captions/config/onnx.toml)

只承载 ONNX Runtime 行为配置，不承载模型语义配置。

建议结构：

```toml
[onnx_runtime.defaults]
execution_provider = "auto"
force_download = false
model_cache_dir = ""
session_cache_dir = ""

[onnx_runtime.defaults.session]
graph_optimization_level = "ORT_ENABLE_ALL"
enable_mem_pattern = true
enable_mem_reuse = true

[onnx_runtime.defaults.cuda]
arena_extend_strategy = "kSameAsRequested"
cudnn_conv_algo_search = "EXHAUSTIVE"
do_copy_in_default_stream = true

[onnx_runtime.defaults.tensorrt]
engine_cache_enable = true
timing_cache_enable = true
fp16_enable = true

[onnx_runtime.wdtagger]

[onnx_runtime.waterdetect]

[onnx_runtime.lfm_vl_local]
```

### 2. 配置职责拆分

#### `model.toml`

只保留模型与任务语义相关配置，例如：

- `model_id`
- `max_new_tokens`
- 阈值
- 图像 token 设置
- prompt / 行为开关

#### `onnx.toml`

只保留 runtime 相关配置，例如：

- `execution_provider`
- `force_download`
- cache 路径
- session options
- TensorRT / CUDA / OpenVINO 参数

兼容期内的约束：

- 已存在于 [`config/model.toml`](E:/Code/qinglong-captions/config/model.toml) 的 runtime 字段继续被读取
- 所有新增 runtime 字段只写入 `onnx.toml`
- 不再向 `model.toml` 添加新的 runtime 配置

### 3. 配置合并顺序

建议统一为：

1. `onnx_runtime.defaults`
2. 从 `model.toml` 对应工具 section 中提取 legacy runtime 字段
3. `onnx_runtime.<tool_name>` 覆盖 legacy runtime 字段
4. 工具运行时通过 CLI 传入的 `repo_id` / `model_dir` / `force_download`

说明：

- 外部 CLI 参数仍然有效，但只覆盖和调用上下文直接相关的字段
- provider / session 细节优先由 `onnx.toml` 管理
- `lfm_vl_local` 当前位于 `model.toml` 的 `execution_provider` 将迁移至 `onnx.toml`
- 兼容期内如果命中 `model.toml` 的 legacy runtime 字段，需输出明确的弃用日志
- 旧字段至少保留一个正式发布周期，再决定是否删除

## 配置加载设计

### 1. 扩展 split config 合并入口

[`config/loader.py`](E:/Code/qinglong-captions/config/loader.py) 中的 `_CONFIG_FILES` 需要从：

- `prompts.toml`
- `model.toml`
- `general.toml`

扩展为：

- `prompts.toml`
- `model.toml`
- `general.toml`
- `onnx.toml`

这样：

- `load_config()` 的结果能自然包含 `onnx_runtime`
- [`config/runtime_config.py`](E:/Code/qinglong-captions/config/runtime_config.py) 和 [`config/config.py`](E:/Code/qinglong-captions/config/config.py) 的兼容加载链路自动获得 ONNX 配置
- 工具层不需要自己绕开统一配置系统

### 2. `OnnxRuntimeConfig` 扩展

[`module/onnx_runtime/config.py`](E:/Code/qinglong-captions/module/onnx_runtime/config.py) 需要从当前轻量对象扩展成真正的 runtime 配置中心，至少覆盖：

- 基础字段
  - `execution_provider`
  - `force_download`
  - `model_cache_dir`
  - `session_cache_dir`
- session 选项
  - `graph_optimization_level`
  - `enable_mem_pattern`
  - `enable_mem_reuse`
  - `execution_mode`
  - `inter_op_num_threads`
  - `intra_op_num_threads`
- provider 选项
  - CUDA
  - TensorRT / NvTensorRtRtx
  - OpenVINO

同时提供明确的合并接口，例如：

- `from_mapping(...)`
- `from_runtime_sections(defaults, tool_override, cli_override=None)`

并增加兼容读取逻辑：

- 在工具对应的 `model.toml` section 中识别 legacy runtime 字段
- 先生成 legacy override，再让 `onnx.toml` 中的同名字段覆盖它
- 对每个命中的 legacy runtime 字段输出一次弃用提示，避免静默行为变化

## Session 设计

### 1. 统一 provider 选择

[`module/onnx_runtime/session.py`](E:/Code/qinglong-captions/module/onnx_runtime/session.py) 继续作为 provider 选择的唯一入口。

目标是让：

- `wdtagger`
- `waterdetect`
- `lfm_vl_local`

全部共用同一套 provider 选择和 fallback 规则，而不是各自维护分支。

### 2. 统一 SessionOptions 构建

新增统一的 session options builder，负责：

- graph optimization level
- mem pattern / mem reuse
- CPU 并行选项
- 其他通用 ONNX session 行为

这样工具层不再直接 new `ort.SessionOptions()` 并手工塞参数。

### 3. 统一 provider options 构建

TensorRT / CUDA / OpenVINO 的 options 由统一 builder 产出。

原则：

- 第一版先把当前仓库中重复出现的成熟参数收敛成共享默认值
- 少量字段允许通过 `onnx.toml` 覆盖
- 不追求一次把所有 provider 参数都配置化

### 4. Session cache 与 runtime fingerprint

在 `session.py` 中需要统一 cache key 生成逻辑，而不是只给单模型单独加 key 规则。

要求：

- `lfm_vl_local` 继续使用 bundle cache
- `wdtagger` / `waterdetect` 使用单模型 bundle cache
- 两类 cache 都必须把最终生效的 runtime 配置纳入 key

建议增加稳定的 `runtime_fingerprint` / `runtime_signature`，由最终生效配置生成，例如基于规范化后的配置字典做稳定序列化再哈希。

cache key 应至少包含：

- bundle key
- 规范化后的 model path / session paths
- provider descriptors（provider 名称及其生效 options 摘要）
- runtime fingerprint（生效的 session options、线程数、execution mode、provider options 等摘要）

规则：

- runtime fingerprint 相同，才允许复用已有 session
- runtime fingerprint 发生变化，必须视为不同 session，不能命中旧 cache

## 工具接入设计

### 1. `waterdetect`

[`module/waterdetect.py`](E:/Code/qinglong-captions/module/waterdetect.py) 应保留：

- `AutoImageProcessor` 加载
- 图像预处理
- batch 输入拼装
- 水印概率解释
- 结果目录与 JSON 写盘

应删除：

- execution provider 分支选择
- `SessionOptions` 手工构建
- TensorRT / CUDA 参数拼装
- ONNX session 初始化细节

迁移后它只需：

1. 构造 `OnnxModelSpec`
2. 从统一 runtime 配置中取 `waterdetect` section
3. 调用 `load_single_model_bundle(...)`
4. 使用 `bundle.session` 执行推理

### 2. `wdtagger`

[`utils/wdtagger.py`](E:/Code/qinglong-captions/utils/wdtagger.py) 应保留：

- 标签文件下载与解析
- CSV / JSON 标签映射逻辑
- 图像预处理
- batch 推理与标签后处理
- Lance 数据集写回

应删除：

- execution provider 分支选择
- `SessionOptions` 手工构建
- TensorRT / CUDA / OpenVINO 参数拼装
- 直接 new `InferenceSession(...)`

迁移后它同样只负责：

1. 构造 `OnnxModelSpec`
2. 读取 `onnx_runtime.wdtagger`
3. 调用 `load_single_model_bundle(...)`
4. 使用 `bundle.session` 做批量推理

### 3. `lfm_vl_local`

[`module/providers/local_vlm/lfm_vl_local.py`](E:/Code/qinglong-captions/module/providers/local_vlm/lfm_vl_local.py) 保持多文件 artifact 结构不变，但需要改成从 `onnx.toml` 读取 runtime 配置。

它仍旧使用：

- `download_onnx_artifact_set(...)`
- `load_session_bundle(...)`

但底层共享：

- `OnnxRuntimeConfig`
- provider builder
- session options builder

## 错误处理

### 1. 配置缺失

如果 `onnx.toml` 不存在：

- split config 加载应继续工作
- `OnnxRuntimeConfig` 使用代码内默认值

这样可以保持仓库在过渡期内可运行。

### 2. 非法 provider 配置

如果用户在 `onnx.toml` 中配置了不支持的 `execution_provider`：

- 优先尝试显式 provider
- 不可用时退回现有自动策略
- 同时输出清晰日志，说明配置未命中可用 provider

### 3. 模型下载失败

单模型基座需要把下载失败定位到具体 `repo_id` 与 `filename`，而不是仅抛出抽象异常。

### 4. session 创建失败

异常信息至少应保留：

- 模型路径
- provider 列表
- 关键 runtime 配置摘要

便于诊断 TensorRT / CUDA 初始化问题。

## 测试设计

### 1. 现有回归必须保留

现有测试必须继续通过：

- [`tests/test_onnx_runtime.py`](E:/Code/qinglong-captions/tests/test_onnx_runtime.py)
- [`tests/test_lfm_vl_local.py`](E:/Code/qinglong-captions/tests/test_lfm_vl_local.py)
- [`tests/test_runtime_config.py`](E:/Code/qinglong-captions/tests/test_runtime_config.py)

### 2. 新增配置加载测试

扩展 [`tests/test_runtime_config.py`](E:/Code/qinglong-captions/tests/test_runtime_config.py) 或新增测试，覆盖：

- `onnx.toml` 被 `config.loader` 合并
- `onnx_runtime.defaults` 正常加载
- `onnx_runtime.<tool_name>` 可以覆盖 defaults
- `model.toml` 中的 legacy runtime 字段在兼容期内仍然生效
- 当 `onnx.toml` 与 legacy 字段同时存在时，`onnx.toml` 优先

### 3. 新增单模型基座测试

新增测试文件，例如：

- `tests/test_onnx_single_model.py`

至少覆盖：

- 单模型 artifact 下载
- 单模型 session cache 复用
- bundle 暴露 `model_path`、`session`、`providers`
- runtime config 正确传入 session 构建逻辑
- 相同 runtime fingerprint 命中同一 cache
- 不同 runtime fingerprint 不会错误复用旧 session

### 4. 新增 `waterdetect` 接线测试

新增 focused test，mock 掉：

- 模型下载
- session 创建

验证 [`module/waterdetect.py`](E:/Code/qinglong-captions/module/waterdetect.py) 已改走统一单模型基座，而不是内部直接初始化 `onnxruntime`。

### 5. 新增 `wdtagger` 接线测试

同样新增 focused test，mock 掉：

- 模型下载
- session 创建

验证 [`utils/wdtagger.py`](E:/Code/qinglong-captions/utils/wdtagger.py) 已改走统一单模型基座，并保留现有标签文件逻辑。

## 风险

### 1. 过度抽象风险

如果试图把 preprocess / postprocess 一起统一成通用 pipeline，会明显超出当前需求，导致 `wdtagger` 和水印检测的业务逻辑被迫适配抽象。

本设计通过“单模型 ONNX 基座”刻意把统一范围限制在 runtime 层，避免这个问题。

### 2. 配置迁移风险

把 runtime 配置从 `model.toml` 挪到 `onnx.toml` 后，如果 loader 没同步升级，部分调用链会读不到配置。

因此 `config/loader.py` 的 split file 扩展是迁移中的第一优先级，不应后置。

### 3. provider 参数漂移风险

如果 `wdtagger`、`waterdetect` 和 `lfm_vl_local` 仍保留自己的一套 TensorRT / CUDA 参数，后续很快又会分叉。

因此 provider options builder 必须成为唯一事实来源，工具脚本不再持有独立实现。

## 实施顺序建议

1. 扩展 [`config/loader.py`](E:/Code/qinglong-captions/config/loader.py)，把 `onnx.toml` 纳入正式 split config
2. 新增 [`config/onnx.toml`](E:/Code/qinglong-captions/config/onnx.toml)，同时保留 `model.toml` 的 legacy runtime 字段兼容读取与弃用日志
3. 扩展 [`module/onnx_runtime/config.py`](E:/Code/qinglong-captions/module/onnx_runtime/config.py)
4. 扩展 [`module/onnx_runtime/session.py`](E:/Code/qinglong-captions/module/onnx_runtime/session.py)
5. 新增 [`module/onnx_runtime/single_model.py`](E:/Code/qinglong-captions/module/onnx_runtime/single_model.py)
6. 更新 [`module/onnx_runtime/__init__.py`](E:/Code/qinglong-captions/module/onnx_runtime/__init__.py)
7. 先接入 [`module/waterdetect.py`](E:/Code/qinglong-captions/module/waterdetect.py)
8. 再接入 [`utils/wdtagger.py`](E:/Code/qinglong-captions/utils/wdtagger.py)
9. 跑 focused tests，最后跑现有 ONNX runtime 与 `lfm_vl_local` 回归

## 结论

本设计采用“窄而完整的单模型 ONNX 基座”方案，目标不是统一所有 ONNX 工具的业务流程，而是统一它们共同依赖的 runtime 层。新增的 [`config/onnx.toml`](E:/Code/qinglong-captions/config/onnx.toml) 将承载 execution provider、session options 和 cache 等运行时配置；新增的单模型基座将把单模型 ONNX 工具的下载、provider 选择、session 初始化与缓存统一收口。这样 `wdtagger` 和水印检测可以删除重复的 `onnxruntime` 逻辑，`lfm_vl_local` 继续保留多 artifact 结构但共享同一套 runtime/config 能力，为后续更多 ONNX 推理工具接入提供稳定边界。
