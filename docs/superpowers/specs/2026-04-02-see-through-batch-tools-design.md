# See-through 目录批处理工具设计

## 背景

当前需求不是“把上游仓库跑起来”，而是把 [`https://github.com/shitagaki-lab/see-through`](https://github.com/shitagaki-lab/see-through) 的推理能力正式集成到青龙字幕工具中，并且满足两个明确约束：

1. 在 GUI 的工具页中作为一个正式工具提供目录批处理能力。
2. 首版必须支持 `flash-attn` 加速，并在不同推理 pipeline 之间做显式显存卸载。

上游仓库当前的主推理链路是：

1. `LayerDiff 3D`
2. `Marigold Depth`
3. PSD / 分层后处理

上游实现的特点是：

- 通过 [`C:\Users\QINGLO~1\AppData\Local\Temp\see-through-codex\inference\scripts\inference_psd.py`](C:\Users\QINGLO~1\AppData\Local\Temp\see-through-codex\inference\scripts\inference_psd.py) 串起流程
- 在 [`C:\Users\QINGLO~1\AppData\Local\Temp\see-through-codex\common\utils\inference_utils.py`](C:\Users\QINGLO~1\AppData\Local\Temp\see-through-codex\common\utils\inference_utils.py) 中使用模块级全局变量缓存 pipeline
- pipeline 直接 `.to(device="cuda", dtype=torch.bfloat16)`，没有正式的显存阶段管理
- 批处理语义接近“脚本批量遍历”，而不是工具化的任务运行器

这与本仓库已有的工具结构不同。本仓库已经具备：

- GUI 工具页：[`E:\Code\qinglong-captions\gui\wizard\step6_tools.py`](E:\Code\qinglong-captions\gui\wizard\step6_tools.py)
- 统一任务启动与日志：[`E:\Code\qinglong-captions\gui\utils\process_runner.py`](E:\Code\qinglong-captions\gui\utils\process_runner.py)
- 统一的本地模型加载和缓存思路：[`E:\Code\qinglong-captions\module\providers\local_vlm_base.py`](E:\Code\qinglong-captions\module\providers\local_vlm_base.py) 与 [`E:\Code\qinglong-captions\module\providers\local_alm_base.py`](E:\Code\qinglong-captions\module\providers\local_alm_base.py)
- 已经存在的工具运行时配置抽象：[`E:\Code\qinglong-captions\module\onnx_runtime\config.py`](E:\Code\qinglong-captions\module\onnx_runtime\config.py)

因此首版不应选择“脚本壳包装”，而应选择“内嵌式后端 + 工具页接入”方案。

## 已确认决策

- 采用“方案 2：内嵌式后端 + 工具页接入”
- 工具入口位于 GUI 工具页，而不是字幕主流程
- 首版只做目录批处理，不做单图输入
- 批处理主单位为“按 pipeline 分阶段整目录运行”
- 调度顺序固定为：`LayerDiff(全目录) -> 卸载 -> Marigold(全目录) -> 卸载 -> Postprocess/PSD(全目录)`
- `flash-attn` 是首版必须支持的可选加速能力
- 首版必须支持跨 pipeline 的显存卸载
- 上游代码归属策略明确为“只抽取必要算法，拒绝整仓引入 / submodule”
- 首版必须包含：
  - 抽样预览 / 只跑前 N 张
  - 失败继续
  - 基于 outputs 文件夹的阶段恢复
  - 跳过已完成结果

## 目标

1. 在工具页新增 `See-through` 工具 tab，支持目录级批处理。
2. 在仓库内建立正式的 `see_through` 后端模块，而不是依赖上游脚本直调。
3. 为 `LayerDiff` 和 `Marigold` 提供统一的模型加载、能力探测和阶段级显存生命周期管理。
4. 在支持的 GPU / 环境上启用 `flash-attn`，不支持时自动回退。
5. 在 `LayerDiff -> Marigold -> Postprocess` 阶段切换时显式卸载不再使用的 GPU 模块，降低峰值显存。
6. 为目录批处理提供恢复性：新 run 输出隔离、失败记录、跳过已完成、续跑。
7. 保持 GUI、CLI、任务日志、依赖安装、输出目录布局的一致性。

## 非目标

- 首版不做单图交互式输入
- 首版不做 provider/registry 路由接入到字幕主流程
- 首版不把 see-through 包装成 `module.providers.*` 的 VLM/OCR provider
- 首版不做 Live2D 后续 rigging、后续左/右拆分高级编辑 UI
- 首版不做多机分布式或多 GPU 并行
- 首版不追求与上游完全同构的所有脚本和 notebook 能力
- 首版不支持“多张图同时常驻不同 pipeline 并发推理”

## 现状问题

### 1. 上游的 pipeline 生命周期不适合工具化批处理

上游把 `layerdiff_pipeline` / `marigold_pipeline` 缓存在全局变量中，并且默认一旦创建就长期占据 GPU。这对研究脚本合理，但对 GUI 工具批处理不合理：

- 峰值显存高
- 多次任务之间状态不透明
- 无法明确控制何时卸载

### 2. `flash-attn` 当前不是正式能力，而只是潜在可替换项

上游代码没有统一的 capability probe。当前仓库已有 [`E:\Code\qinglong-captions\utils\transformer_loader.py`](E:\Code\qinglong-captions\utils\transformer_loader.py) 中的 `flash_attn` 检测思路，但它主要服务于 Transformers 本地模型。See-through 这类 diffusers / 自定义 UNet 组合链路还没有接入统一能力探测。

### 3. 工具页现有模式偏向“单脚本发车”

[`E:\Code\qinglong-captions\gui\wizard\step6_tools.py`](E:\Code\qinglong-captions\gui\wizard\step6_tools.py) 中已有工具大多是：

- 收集参数
- 调用一个模块脚本
- 交给共享执行面板

这适合 ONNX 类工具，但 see-through 是多阶段重型视觉推理，如果不把“当前 `output_dir`”定义成恢复边界，批处理失败后恢复会很差。

## 架构设计

### 1. 总体形态

采用“工具页前端 + 独立后端 runner + 分阶段 pipeline 包装”的三层结构：

1. GUI 层
   - 收集目录、模型、输出、抽样、显存策略等参数
   - 启动后端任务
   - 展示日志与结果目录
2. Runner 层
   - 目录扫描
   - run 目录管理
   - outputs 阶段判定
   - 分阶段整目录调度
   - 失败记录与续跑
3. Pipeline 层
   - `LayerDiff`
   - `Marigold`
   - `Postprocess / PSD export`
   - 显存加载与卸载

### 2. 上游代码归属

首版明确不采用：

- vendoring 整个 see-through 仓库
- git submodule 引入上游整仓
- 在运行时依赖上游仓库目录结构和脚本入口

首版明确采用：

- 只抽取 `LayerDiff` 推理、`Marigold` 推理、`further_extr` / PSD 导出所需的最小算法集合
- 把抽取后的代码改写为本仓库内稳定 import 路径
- 在抽取文件头部记录 upstream 来源文件与 commit/版本信息

这样做的目的是把 ownership 边界钉死：

- 本仓库只维护“工具运行所需的最小子集”
- 上游新增 notebook / demo / 脚本不会自动进入本仓库维护面
- 后续升级时按算法子集对比，而不是整仓同步

### 3. 模块布局

建议新增：

- [`E:\Code\qinglong-captions\module\see_through\__init__.py`](E:\Code\qinglong-captions\module\see_through\__init__.py)
- [`E:\Code\qinglong-captions\module\see_through\runner.py`](E:\Code\qinglong-captions\module\see_through\runner.py)
- [`E:\Code\qinglong-captions\module\see_through\runtime.py`](E:\Code\qinglong-captions\module\see_through\runtime.py)
- [`E:\Code\qinglong-captions\module\see_through\model_manager.py`](E:\Code\qinglong-captions\module\see_through\model_manager.py)
- [`E:\Code\qinglong-captions\module\see_through\extracted\layerdiff_core.py`](E:\Code\qinglong-captions\module\see_through\extracted\layerdiff_core.py)
- [`E:\Code\qinglong-captions\module\see_through\extracted\marigold_core.py`](E:\Code\qinglong-captions\module\see_through\extracted\marigold_core.py)
- [`E:\Code\qinglong-captions\module\see_through\extracted\postprocess_core.py`](E:\Code\qinglong-captions\module\see_through\extracted\postprocess_core.py)
- [`E:\Code\qinglong-captions\module\see_through\pipelines\layerdiff.py`](E:\Code\qinglong-captions\module\see_through\pipelines\layerdiff.py)
- [`E:\Code\qinglong-captions\module\see_through\pipelines\marigold.py`](E:\Code\qinglong-captions\module\see_through\pipelines\marigold.py)
- [`E:\Code\qinglong-captions\module\see_through\postprocess.py`](E:\Code\qinglong-captions\module\see_through\postprocess.py)
- [`E:\Code\qinglong-captions\module\see_through\cli.py`](E:\Code\qinglong-captions\module\see_through\cli.py)

辅助接线：

- [`E:\Code\qinglong-captions\gui\wizard\step6_tools.py`](E:\Code\qinglong-captions\gui\wizard\step6_tools.py)
- [`E:\Code\qinglong-captions\gui\utils\process_runner.py`](E:\Code\qinglong-captions\gui\utils\process_runner.py)
- [`E:\Code\qinglong-captions\gui\utils\i18n.py`](E:\Code\qinglong-captions\gui\utils\i18n.py)
- [`E:\Code\qinglong-captions\config\model.toml`](E:\Code\qinglong-captions\config\model.toml)
- [`E:\Code\qinglong-captions\config\model.toml`](E:\Code\qinglong-captions\config\model.toml) 或新增 split 配置文件

### 4. 运行边界

首版不把 see-through 直接注册进现有 `ProviderRegistry`。原因：

- 它不是“给字幕流程产出 caption”的 provider
- 它是目录级工具，不是单媒体 provider
- 它需要更重的显存阶段管理与输出目录控制

因此它应当采用和 [`E:\Code\qinglong-captions\module\waterdetect.py`](E:\Code\qinglong-captions\module\waterdetect.py) / [`E:\Code\qinglong-captions\module\audio_separator.py`](E:\Code\qinglong-captions\module\audio_separator.py) 类似的“工具后端模块”模式。

### 5. 恢复策略

首版不引入 JSON manifest，也不引入行级 Lance 状态表。

恢复模型直接收敛成两件事：

1. `output_dir` 直接代表当前这次任务的最终产物目录
2. 当前 `output_dir` 内的阶段恢复只看 outputs 文件夹

具体约束：

- `runner.py` 负责计算 `config_fingerprint`
- 如果 `output_dir` 下已存在 `run_meta.json`，且其中的 `input_dir` 或 `config_fingerprint` 与本次运行不一致，则直接报错并要求调用方换一个新的 `output_dir`
- 不把旧输出目录的任何状态 merge 到新输出目录
- 不新增 `module/see_through/lance_state.py`
- 不设计行级 `merge_insert` / live state / stage tag

这里要把概念钉死：

- Lance 在首版只承担“输入数据集备份快照”角色，不承担 item 级恢复状态或阶段状态角色
- `dataset.lance` 是原始输入的只读快照，不是任务运行中的 live table
- 首版不存在 `stable tag` / `latest complete version` / `stage version` 这类 item 级状态语义
- 因此首版也不存在行级主键、行级 upsert、按阶段 merge 的写入契约问题

首版允许保留一个极小的 run 级元数据文件，例如：

- `<output_dir>/run_meta.json`

它只用于记录：

- `config_fingerprint`
- `input_dir`
- `created_at`

它不是 item 级状态表；它只参与“这个 `output_dir` 是否还能继续用于当前任务”的目录级身份校验，不参与 item 级阶段恢复。

如果后续版本确实要把 Lance 从“备份快照”升级为“item 级状态表”，那必须额外单开设计，不允许在首版实现上自然生长。升级设计至少要先钉死以下约束：

- 每行必须有稳定 `item_id`，不能拿 `source_path` 直接充当主键
- 所有写入都必须基于“最新完整版本”做 `merge_insert(on="item_id")`
- `source_path` 变化和 `source_hash` 变化要分别定义语义：前者是重命名还是新 item，后者是同 item 新版本还是直接视为不兼容
- stable tag 只能指向完整快照，不能指向“本阶段只覆盖了触达行”的半成品版本

也就是说：要么完全不要 Lance live state；要做，就先把主键和写入契约定义完，再写代码。

## 推理分阶段设计

### 1. 目录级分阶段处理

首版按整目录的固定 phase 顺序执行：

1. 扫描输入目录，建立 item 清单并完成基础校验
2. `LayerDiff` phase：
   - 对所有需要处理的 item 执行 `LayerDiff`
   - 写入分层 PNG 与 `layerdiff` 阶段状态
3. 卸载 `LayerDiff` GPU 模块
4. `Marigold` phase：
   - 只处理 `layerdiff` 阶段有效的 item
   - 写入深度图与 `marigold` 阶段状态
5. 卸载 `Marigold` GPU 模块
6. `Postprocess / PSD` phase：
   - 只处理前置阶段有效的 item
   - 执行 `further_extr` / PSD 导出
   - 写入最终 `completed` 状态

这里的关键不再是“单张图闭环可解释”，而是：

- 同一 phase 内重用同一 pipeline，避免每张图冷启动
- phase 结束后显式释放模型，压低峰值显存
- 通过当前 `output_dir` 的已落盘输出支持续跑

### 2. 失败传播规则

对单个 item：

- `layerdiff` 失败，则该 item 不进入 `marigold` / `postprocess`
- `marigold` 失败，则该 item 不进入 `postprocess`
- 其他 item 不受影响

对整批任务：

- phase 失败不应抹掉已完成 item 的阶段结果
- `continue_on_error=true` 时，任务继续推进可运行的 item
- 最终汇总由当前 `output_dir` 的输出扫描结果计算出 `completed / partial / failed`

### 3. 为什么采用“分阶段整目录批量”

采用该模式的直接原因是：

- 你明确优先要“pipeline 复用 + phase 结束再删模型”
- 这能避免单图模式下反复创建 / 删除重型 pipeline
- 它更贴合“LayerDiff 全跑完再删，接着 Marigold”这一目标

代价也必须承认：

- 当前 `output_dir` 的 outputs 文件夹必须能推导出当前 item 应从哪个阶段继续
- 中间产物成为正式恢复点
- 配置变更时不做细粒度阶段复用

## `flash-attn` 设计

### 1. 能力目标

首版 `flash-attn` 仍是运行时可回退的加速能力，但安装层面不单独做兼容性设计。

规则：

- 环境支持时自动启用
- 启用失败时自动回退
- 日志中必须明确输出最终使用的 attention 后端

### 2. 能力探测位置

新增 [`E:\Code\qinglong-captions\module\see_through\runtime.py`](E:\Code\qinglong-captions\module\see_through\runtime.py)，集中提供：

- `detect_device()`
- `detect_dtype()`
- `detect_flash_attention_support()`
- `resolve_attention_backend()`

输出应至少包含：

- `device`
- `dtype`
- `attention_backend`
- `reason`

例如：

- `flash_attention_2`
- `sdpa`
- `eager`

### 3. 接入策略

#### LayerDiff

`LayerDiff` 基于 SDXL / 自定义 UNet 组合，优先尝试：

1. 通过 diffusers / transformer 风格的 attention 参数开启 `flash_attention_2`
2. 不支持时回退到 `sdpa`
3. 再不支持回退到 `eager`

如果上游自定义 UNet 不完全兼容显式 attention 参数，则首版允许：

- 仅在兼容子模块启用
- 不强行重写全部 attention processor

原则是“拿到稳定加速收益”，而不是首版做侵入式大改。

#### Marigold

Marigold 也是 diffusers 风格 pipeline，应走同样的能力探测与回退策略。

### 4. 用户可见控制

首版不把 `flash_attention_2 | sdpa | eager` 暴露为正式用户 API。

GUI 与 CLI 的正式行为是：

- 默认自动探测
- 日志里输出最终命中的 attention backend

如果确实需要保留调试逃生口，只保留高级设置：

- `force_eager_attention = false`

也就是说，首版的正式承诺不是“用户精确控制后端枚举”，而是“系统自动选一个当前环境下能跑的最快稳定后端”。

## 跨 pipeline 显存卸载设计

### 1. 卸载目标

必须覆盖至少以下模块：

- LayerDiff:
  - `vae`
  - `trans_vae`
  - `unet`
  - `text_encoder`
  - `text_encoder_2`
- Marigold:
  - `unet`
  - `vae`
  - `text_encoder`
  - pipeline 本体引用

### 2. 卸载策略分层

新增配置项：

- `offload_policy = none | cpu | delete`

语义：

- `none`
  - 阶段切换时不做正式卸载
  - 只适合高显存机器
- `cpu`
  - 阶段结束后把模块 `.to("cpu")`
  - 再执行 `torch.cuda.empty_cache()`
- `delete`
  - 删除 pipeline 引用
  - 强制 `gc.collect()`
  - 再执行 `torch.cuda.empty_cache()`
  - 必要时 `torch.cuda.ipc_collect()`

默认建议：

- `delete`

原因：

- 当前调度模型是“整目录跑完整个 phase 后再切换下一 phase”
- 在这个模型下，`delete` 只发生在 phase 边界，而不是每张图边界
- 因此可以同时拿到更低峰值显存和更好的 pipeline 复用

### 3. 生命周期管理器

新增 [`E:\Code\qinglong-captions\module\see_through\model_manager.py`](E:\Code\qinglong-captions\module\see_through\model_manager.py)，负责：

- pipeline 懒加载
- 在当前 phase 内复用已加载 pipeline
- phase 结束后卸载
- 记录当前活跃阶段
- 输出显存日志

建议接口：

- `get_layerdiff_pipeline()`
- `get_marigold_pipeline()`
- `release_layerdiff()`
- `release_marigold()`
- `release_all()`
- `log_vram(stage_name)`

### 4. 显存日志

如果 CUDA 可用，阶段日志应输出：

- `allocated`
- `reserved`
- `max_allocated`

至少在：

- phase 加载前
- phase 首张图后
- phase 结束后
- phase 卸载后

各打一条日志。

## 输出设计

### 1. 输出目录结构

配置中的 `output_dir` 就是本次任务的最终产物目录，不再自动嵌套子 `run_dir`。

对每张输入图，在当前 `output_dir` 下保留独立子目录，目录键直接来自“规范化相对路径”，并保留扩展名，例如：

- 输入：`foo/a.png`
- 输出目录：`<output_dir>/foo/a.png/`

这样可以保证：

- `foo/a.png` 和 `bar/a.png` 不会冲突
- `a.png` 和 `a.jpg` 也不会冲突

内容至少包括：

- `src_img.png`
- 分层 PNG
- 深度图
- `optimized/`
- 最终 `.psd`
- `<output_dir>/run_meta.json`
- 失败时可选的 `error.txt` / `error.json`

### 2. 跳过与续跑

首版必须支持：

- `skip_completed = true`

整体规则：

- 后端把 see-through 视为一个整体任务，不把阶段状态暴露给用户
- 阶段只作为内部调度边界，用于决定“从哪一步继续”
- 已存在的 `output_dir` 只能在“同输入目录 + 同配置”前提下续跑；否则直接报错并要求新的 `output_dir`

具体调度：

- 如果 `postprocess_outputs` 完整存在：该 item 直接跳过
- 如果 `marigold_outputs` 完整存在但 `postprocess_outputs` 缺失：从 `Postprocess` 继续
- 如果 `layerdiff_outputs` 完整存在但 `marigold_outputs` 缺失：从 `Marigold -> Postprocess` 继续
- 如果 `layerdiff_outputs` 缺失：从 `LayerDiff -> Marigold -> Postprocess` 全链路重跑

这里的“完整存在”不是只看路径存在，而是要通过最小完整性校验。

这套恢复规则只对“当前 `output_dir`”成立。

首版明确假设：

- 一个 `output_dir` 内输入集合视为不可变
- 如果用户原地替换源文件或调整有效配置，应该提供新的 `output_dir`，而不是依赖隐藏状态失效

### 3. 为什么不做 `stage_fingerprint`

首版明确不做 `stage_fingerprint`，原因不是不能做，而是没有必要：

- 你已经要求后端视为整体任务
- 配置变化时直接换一个新的 `output_dir` 整批重跑，比设计阶段级算法签名更简单
- 当前真正需要的只是“断点续跑”，不是“参数变化后只重跑下游阶段”

因此首版的恢复模型只依赖：

- 当前 `output_dir`
- `run_meta.json` 中的 `config_fingerprint`
- `run_meta.json` 中的 `input_dir`
- 阶段输出完整性

### 4. 失败继续

首版必须支持：

- `continue_on_error = true`

默认建议：

- `true`

原因：

- 目录批处理里单张图失败不应拖垮整个任务

## GUI 设计

### 1. 工具页新增 tab

在 [`E:\Code\qinglong-captions\gui\wizard\step6_tools.py`](E:\Code\qinglong-captions\gui\wizard\step6_tools.py) 新增：

- `See-through` tab

建议放在：

- `Watermark`
- `Preprocess`
- `Reward`
- `Audio Separator`
- `Translate`
- `See-through`

或直接放在视觉工具附近，与 `Preprocess` / `Reward` 同一组。

### 2. GUI 参数

首版建议暴露：

- 输入目录
- 输出目录
- `repo_id_layerdiff`
- `repo_id_depth`
- `resolution`
- `offload_policy`
- `dtype`
- `limit_images`
- `skip_completed`
- `continue_on_error`
- `save_to_psd`
- `tblr_split`

可选但建议保留为高级设置：

- `vae_ckpt`
- `unet_ckpt`
- `force_eager_attention`

### 3. GUI 交互要求

- 必须是目录选择，不提供单图模式
- 必须能显示当前任务使用的模型 repo 和后端策略
- Start 按钮走共享 `ExecutionPanel`
- 日志和其他工具一致

### 4. 结果入口

首版至少在任务完成后输出：

- 当前 `output_dir` 路径
- `run_meta.json` 路径

不要求首版在 GUI 内嵌 PSD 预览器。

## CLI 设计

新增工具入口模块：

- [`E:\Code\qinglong-captions\module\see_through\cli.py`](E:\Code\qinglong-captions\module\see_through\cli.py)

目标：

- GUI 只是调用者
- 后端必须能独立从命令行跑

CLI 参数与 GUI 尽量一一对应。

## 配置设计

### 1. 配置位置

首版直接放到现有配置文件：

- [`E:\Code\qinglong-captions\config\model.toml`](E:\Code\qinglong-captions\config\model.toml)

配置段名：

- `[see_through]`

原因：

- 当前 `model.toml` 已经承载多个工具级模型配置，不是纯 provider 配置文件
- 不需要为了 see-through 再引入一个新的 split 文件和 loader 变更
- 减少配置发现路径，GUI / CLI / 后端都更容易共用同一份默认值

### 2. 建议配置结构

```toml
[see_through]
repo_id_layerdiff = "layerdifforg/seethroughv0.0.2_layerdiff3d"
repo_id_depth = "24yearsold/seethroughv0.0.1_marigold"
resolution = 1280
dtype = "bfloat16"
offload_policy = "delete"
skip_completed = true
continue_on_error = true
save_to_psd = true
tblr_split = false
limit_images = 0
force_eager_attention = false
output_dir = "workspace/see_through_output"
```

## 依赖设计

### 1. 新增 uv extra

在 [`E:\Code\qinglong-captions\pyproject.toml`](E:\Code\qinglong-captions\pyproject.toml) 中新增例如：

- `see-through`

依赖至少覆盖：

- torch / torchvision
- diffusers
- transformers
- safetensors
- psd-tools
- opencv-python
- huggingface_hub
- imageio / PIL / numpy
- flash-attn

首版约束明确为：

- 不为不同平台拆分安装矩阵
- `see-through` 直接声明完整依赖
- 如果某环境无法安装完整依赖，视为环境前置条件不满足，不在本工具设计内单独兜底

### 2. GUI 运行器接线

需要更新 [`E:\Code\qinglong-captions\gui\utils\process_runner.py`](E:\Code\qinglong-captions\gui\utils\process_runner.py)：

- `SCRIPT_REGISTRY`
- uv extra profile

以保证工具页启动时能自动补依赖。

## 错误处理设计

### 1. 输入目录非法

直接失败，不启动任务。

### 2. 单张图片损坏

在当前 item 输出目录写入错误记录，然后继续下一张。

### 3. `flash-attn` 运行时不可用

不失败，回退并写 warning。

### 4. pipeline 加载失败

记录具体阶段：

- `layerdiff_load`
- `marigold_load`

### 5. 输出已存在但不完整

如果 `skip_completed=true`，仍需校验产物完整性，不完整则重跑。

## 测试设计

### 1. 运行时能力测试

新增测试覆盖：

- `flash_attn` 可用探测
- `attention_backend=auto` 的回退逻辑
- `dtype/device` 选择逻辑

### 2. 生命周期管理测试

覆盖：

- `release_layerdiff()`
- `release_marigold()`
- `offload_policy=cpu`
- `offload_policy=delete`

至少验证：

- 对象引用被释放或迁移到 CPU
- 清 cache 方法被调用

### 3. 文件系统恢复测试

覆盖：

- 首次运行初始化目标 `output_dir`
- 失败记录
- `skip_completed`
- `continue_on_error`
- `limit_images`
- `config_fingerprint` 变化时拒绝复用已有 `output_dir`
- `input_dir` 变化时拒绝复用已有 `output_dir`
- `foo/a.png` 与 `bar/a.png` 的输出目录不会冲突
- 按 outputs 文件夹判断阶段恢复

### 4. GUI 工具接线测试

覆盖：

- `step6_tools.py` 中新 tab 出现
- 参数能正确拼成 CLI
- `process_runner.py` 能正确启动 `module.see_through.cli`

### 5. 后端包装测试

对 `runner.py` 做 focused tests，mock 掉实际模型加载，验证：

- 目录扫描
- `LayerDiff -> release -> Marigold -> release -> Postprocess` phase 顺序正确
- 同一 phase 内 pipeline 被复用
- phase 间 release 被调用

## 风险

### 1. `flash-attn` 与上游自定义模块兼容性风险

缓解：

- 明确允许回退
- 首版不侵入式重写上游所有 attention processor

### 2. 分阶段批处理让恢复逻辑变复杂

缓解：

- 用当前 `output_dir` 的 outputs 记录恢复点
- 把恢复规则限制为“目录身份校验 + 输出存在性”
- 明确中间产物是正式恢复边界

### 3. 显存卸载与 phase 切换导致性能抖动

缓解：

- 用 `offload_policy` 配置化
- 默认按 phase 边界释放，不在单图边界反复释放

### 4. 依赖体积大且安装前置条件强

缓解：

- 使用独立 uv extra
- 在安装说明中明确 CUDA / torch / flash-attn 前置条件

### 5. 只做目录批处理导致调试成本升高

缓解：

- `limit_images`
- 当前 `output_dir`
- 跳过已完成

## 实施顺序建议

1. 新增 `module/see_through` 后端骨架
2. 先抽取最小 upstream 算法子集并固定本地 import 边界
3. 实现 `runtime.py` 与 `model_manager.py`
4. 包装 `LayerDiff`
5. 包装 `Marigold`
6. 包装 `postprocess`
7. 实现 `runner.py` 中的 `output_dir` 身份校验与 outputs 续跑逻辑
8. 实现 CLI 入口
9. 接入 GUI 工具页
10. 接入 `process_runner.py` 与 uv extra
11. 增加 focused tests
12. 最后补 README / GUI 文案 / 安装说明

## 结论

本设计选择把 see-through 作为“工具页中的目录批处理重型视觉工具”正式接入，而不是把上游脚本直接当成黑箱调用。这样做的核心价值不在于代码更漂亮，而在于它能真正解决首版最重要的三个问题：

1. `flash-attn` 成为可检测、可回退、可记录的正式能力
2. `LayerDiff` 与 `Marigold` 在整目录 phase 边界显式卸载，降低峰值显存且避免单图冷启动
3. 目录批处理具备基于当前 `output_dir` 输出的阶段级恢复性，失败不会让整批任务失控或静默复用陈旧结果

这条路径比“先快速壳包装”成本更高，但更接近你真正要的结果：一个能长期维护、能稳定批跑、能嵌入现有工具页体系的 see-through 集成。
