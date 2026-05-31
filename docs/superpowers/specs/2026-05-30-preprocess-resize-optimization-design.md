# 图像预处理普通缩放路径优化设计

## 背景

`utils/preprocess_datasets.py` 的普通图像预处理路径承担两类任务：

1. Resize-only：扫描输入目录，对每张图片按 `max_long_edge` / `max_short_edge` / `max_pixels` 约束缩放。
2. Align：当提供 `align_input_dir` 时，对源图和参考图做匹配、变换和保存。

本 spec 只覆盖 resize-only 路径。当前 resize-only 路径已经用 `ThreadPoolExecutor` 并行处理文件，但每张图片内部仍存在明显无效工作：

- 在判断是否需要缩放前，已经把图片完整解码为像素数组。
- 在仅执行 resize 的情况下，仍做了 RGB -> BGR -> RGB 的通道往返。
- 大量“无需处理”的图片会逐张输出日志，吞掉一部分并行收益。
- 用户无法从日志判断慢在解码、转换、resize 还是保存。

这些问题会让“提高线程数”只能缓解一部分等待，却不能消除单张图片内部的浪费。

## 第一性原理判断

普通缩放预处理的数据流应当是：

```text
路径枚举
  -> 读取图片头信息
  -> 计算目标尺寸
  -> 判断是否需要像素级处理
  -> 解码/转换
  -> resize
  -> 保存
```

最贵的是像素级处理：解码、整图复制、resize、编码。任何能够用文件头和元数据完成的判断，都应该在像素解码之前完成。

因此首要优化不是继续增加线程数，而是减少每个线程里做的无效工作。线程数只决定并发度；如果每个任务都先无条件做整图解码和多次内存复制，更多线程只会更快地耗尽 CPU、内存带宽或磁盘带宽。

## 目标

1. 保持现有 CLI、PowerShell 和 GUI 参数兼容。
2. 只优化 resize-only 路径，不改变 align 路径行为。
3. 对已经满足尺寸和模式要求的图片，在像素解码前早退。
4. 在 CPU resize 路径减少不必要的 RGB/BGR 往返复制。
5. 降低大量小图或无需处理图片时的 per-file 日志开销。
6. 增加轻量 profiling 输出，让用户能判断瓶颈属于扫描、跳过、解码/转换、resize 还是保存。
7. 保留默认输出图像语义：尺寸计算、覆盖保存、质量策略和格式处理不在本次改变。
8. 增加覆盖早退、尺寸计算、模式转换和普通 resize 的测试。

## 非目标

- 不改变保存策略。
- 不调整 JPEG/WebP/AVIF 的质量、subsampling 或编码参数。
- 不优化 GPU resize 分支。
- 不改变 GPU resize 的启用条件。
- 不优化或并行化 `align_input_dir` 对齐模式。
- 不改变 `align_images()` 的特征匹配、warp、padding 逻辑。
- 不改变 `calculate_dimensions()` 的尺寸约束语义。
- 不把预处理重写成新的 pipeline 框架。
- 不引入新的默认用户可见参数来替代现有 `workers`。

## 已确认瓶颈

### 1. 跳过判断太晚

当前 `resize_image()` 在 `Image.open()` 后立即进入像素级处理：

```python
img_for_processing_pil = pil_image
...
image_cv = np.array(img_for_processing_pil.convert("RGB"))
...
new_w, new_h = calculate_dimensions(...)
if (w, h) == (new_w, new_h) and pil_image.mode == img_for_processing_pil.mode:
    return True
```

这意味着即使图片本来已经满足限制，也会先完整解码、RGB 转换、NumPy 分配和可能的 BGR copy。

正确顺序应当是先用 `pil_image.size` 和 `pil_image.mode` 判断是否需要继续。

### 2. CPU resize 不需要 BGR 往返

Resize-only 的 OpenCV `cv2.resize()` 对每个通道独立插值。只要不调用依赖颜色空间语义的 OpenCV 算法，RGB 数组直接 resize 与先转 BGR、resize、再转 RGB 的结果等价。

当前 BGR 往返会多做至少两次整图内存复制：

```python
image_cv = np.array(img_for_processing_pil.convert("RGB"))
image_cv = image_cv[:, :, ::-1].copy()
...
resized_pil = Image.fromarray(cv2.cvtColor(resized_image_cv, cv2.COLOR_BGR2RGB))
```

对于 CPU resize-only，可以保留 RGB 数组到最后，直接 `Image.fromarray(resized_image_cv)`。

### 3. per-file 日志会放大线程争用

当大量图片无需 resize 时，当前每张图片都会输出 skip 日志。Rich console 需要同步输出，多线程下会形成锁竞争和终端 IO 开销。

默认日志应聚合为计数，只有 verbose/debug 模式才输出每张图片的细节。

### 4. 线程数不是根因

普通 resize-only 分支已经在线程池中运行：

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    ...
    executor.submit(self.resize_image, ...)
```

所以提高 `workers` 只能在 IO 等待或 C 扩展释放 GIL 时带来收益。它不能修复单张图片内部的无效解码、复制和日志开销。

## 推荐设计

### 1. 拆出尺寸规划结果

新增内部数据结构，不改变外部 API：

```python
@dataclass(frozen=True)
class ResizePlan:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    original_mode: str
    original_format: str
    requires_resize: bool
    requires_mode_conversion: bool
    requires_transparent_crop: bool

    @property
    def can_skip_pixel_decode(self) -> bool:
        return (
            not self.requires_resize
            and not self.requires_mode_conversion
            and not self.requires_transparent_crop
        )
```

新增 helper：

```python
def _build_resize_plan(
    pil_image: Image.Image,
    image_path: str,
    *,
    max_short_edge: int | None,
    max_long_edge: int | None,
    max_pixels: int | None,
    crop_transparent: bool,
) -> ResizePlan:
    ...
```

`requires_mode_conversion` 必须复用当前规则：

- `mode not in ["RGB", "L"]` 需要转换。
- `mode == "L"` 且目标扩展名不在 `jpeg/jpg/png/avif` 时需要转换。

`requires_transparent_crop` 只在 `crop_transparent=True` 且 `pil_image.mode == "RGBA"` 时为 true。透明裁剪需要读取像素，不能在 header 阶段安全跳过。

### 2. 早退必须发生在像素转换前

新的 `resize_image()` 顺序：

```text
Image.open
  -> build ResizePlan from size/mode/path
  -> if plan.can_skip_pixel_decode: return skipped
  -> optional transparent crop
  -> mode conversion
  -> decode to array
  -> resize if required
  -> save
```

注意：早退只允许在“当前行为本来也会 skip”的情况下发生，不能跳过当前会做格式/模式归一化的图片。

### 3. CPU resize-only 保持 RGB 数组

在非 GPU 分支：

```python
image_rgb = np.asarray(img_for_processing_pil.convert("RGB"))
resized_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
resized_pil = Image.fromarray(resized_rgb)
```

不再做 RGB -> BGR -> RGB 往返。

GPU 分支本次不改。为了降低风险，可以先把 RGB fast path 限定为 `use_gpu is False`。

### 4. 返回处理状态而不是裸 bool

内部可以新增：

```python
@dataclass(frozen=True)
class ProcessImageResult:
    ok: bool
    skipped: bool = False
    resized: bool = False
    converted: bool = False
    cropped: bool = False
    error: str = ""
```

`process_directory()` 继续统计成功数，但额外累计：

- `skipped_count`
- `resized_count`
- `converted_count`
- `cropped_count`
- `failed_count`

对外仍打印 “Successfully processed: successful/total images”，避免破坏调用方预期。

### 5. 默认聚合日志

默认不再逐张输出：

- already meets size
- cropped transparent borders
- converting mode

改为任务结束后输出聚合摘要：

```text
Resize summary:
  skipped without pixel decode: N
  resized: N
  mode converted: N
  transparent cropped: N
  failed: N
```

如果后续需要逐文件日志，单独新增 `--verbose`，但本次不要求引入。

### 6. 轻量 profiling

新增内部计时器，默认只输出总计，不做逐文件日志。

阶段建议：

- `scan`
- `open_and_plan`
- `decode_convert_crop`
- `resize`
- `save`

实现上可以用线程安全的聚合器：

```python
class StageTimer:
    def add(self, stage: str, seconds: float) -> None:
        ...
```

先只在 `--profile` 开启时输出，避免默认日志变多。

CLI 新增可选参数：

```text
--profile
```

GUI 本次不必须暴露该开关；CLI 可先服务开发者和性能排查。

## 实施计划

### Phase 1: 早退和状态统计

1. 新增 `ResizePlan` 和 `_build_resize_plan()`。
2. 调整 `resize_image()`，在像素转换前早退。
3. 新增 `ProcessImageResult`。
4. `process_directory()` 兼容 `bool` 到结果对象的改动，并输出聚合摘要。
5. 测试无需 resize 的 RGB/JPEG 图片不会进入 NumPy/OpenCV resize 路径。

### Phase 2: CPU RGB fast path

1. 非 GPU resize-only 分支保留 RGB 数组。
2. 删除 CPU 路径的 BGR copy 和 `cv2.cvtColor(...BGR2RGB)`。
3. 用小样本测试输出尺寸和通道顺序正确。
4. 对同一张 RGB fixture 对比旧路径与新路径像素一致性。

### Phase 3: Profiling

1. CLI 增加 `--profile`。
2. 增加线程安全阶段计时聚合器。
3. 在 `Image.open/plan`、decode/convert/crop、resize、save 外层打点。
4. 输出总耗时、阶段耗时和每张图平均耗时。

## 测试方案

### 单元测试

新增测试文件：

```text
tests/test_preprocess_resize_optimization.py
```

覆盖：

1. RGB/JPEG，尺寸已满足，`crop_transparent=False`：应早退，不调用 `cv2.resize`。
2. RGB/JPEG，超过 `max_long_edge`：应 resize，输出尺寸符合 `calculate_dimensions()`。
3. P/PNG，尺寸已满足：不能早退，因为当前行为会转 RGB。
4. RGBA/PNG 且 `crop_transparent=True`：不能早退，应执行透明裁剪。
5. RGB/JPEG，`max_short_edge` 和 `max_pixels` 同时限制：最终尺寸满足更严格约束。
6. CPU RGB fast path：输出通道顺序保持 RGB。
7. `process_directory()` 聚合统计：skipped/resized/failed 计数正确。

### 回归测试

使用临时目录生成 3 类图片：

- 已满足尺寸的 RGB JPEG。
- 超大 RGB PNG。
- RGBA 透明边 PNG。

执行：

```powershell
python -m pytest tests/test_preprocess_resize_optimization.py
python -m py_compile utils/preprocess_datasets.py utils/stream_util.py
```

如果项目 `uv run` 受远端依赖影响失败，先用当前可用 Python 跑纯单元测试；依赖恢复后再补 `uv run`。

### 性能验收

准备 100 张已经满足尺寸的 RGB JPEG：

- 优化前：会完整 decode + NumPy/OpenCV 转换再 skip。
- 优化后：只读取头信息并早退。

验收目标：

- 已满足尺寸样本的总耗时明显下降。
- `--profile` 中 `decode_convert_crop` 和 `resize` 对 skipped 图片接近 0。
- 输出文件未被修改。

准备 100 张需要缩小的 RGB JPEG：

- 优化后总耗时不应显著劣化。
- 输出尺寸与旧逻辑一致。
- 通道顺序无反转。

## 风险与约束

1. Pillow 的 `Image.open()` 是惰性读取，但访问部分属性可能触发有限解析；这仍远低于整图 decode。
2. 透明裁剪必须读取像素，不能被早退优化绕过。
3. 模式转换规则必须保持现状，否则会改变用户输出。
4. CPU RGB fast path依赖 `cv2.resize()` 对通道独立处理；本 spec 只用于 resize-only，不用于 OpenCV 颜色语义操作。
5. 默认日志减少可能让用户少看到逐文件细节，但聚合摘要更适合批处理。

## 验收标准

1. resize-only 路径对无需处理的 RGB/JPEG 图片能在像素解码前早退。
2. resize-only 路径的成功/失败统计保持正确。
3. 需要 resize 的图片输出尺寸与 `calculate_dimensions()` 一致。
4. 普通 CPU resize 不再做不必要的 RGB/BGR 往返。
5. 不修改保存质量策略。
6. 不修改 GPU resize 分支。
7. 不修改对齐模式。
8. 新增测试覆盖早退、模式转换、透明裁剪不早退、尺寸限制和统计。
9. `python -m py_compile utils/preprocess_datasets.py utils/stream_util.py` 通过。

## 更低成本替代方案

如果只想马上判断机器瓶颈，而不想先改代码，可以先加 `--profile`，不做早退和 RGB fast path。

但这只能告诉用户慢在哪里，不能解决最明显的无效解码问题。因此推荐按 Phase 1 先做早退，再补 profiling。
