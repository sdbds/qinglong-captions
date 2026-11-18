# S2D glTF Profile 0.1 设计规范

## 状态

设计已获用户确认。本规范冻结首版公开交换与运行时格式：**S2D 不是新的底层容器，
而是一个严格的 glTF 2.0 / GLB Profile，加上 2.5D 角色语义扩展。**

S2D 将在独立新项目中实现。本文不修改、取代或约束 `qinglong-captions` 已有的 auto-rig、
Spine 或 Live2D 规范；当前仓库只暂存设计文档，并提供真实上游样本用于验证输入边界。
新项目独立拥有 Schema、编译器、runtime、validator 和 adapters。

## 目标

1. 上游分层 PSD、逐部件深度和自动绑骨结果可编译为一个标准 `.glb` 文件。
2. 普通 glTF 2.0 软件无需理解 S2D 扩展，也能显示默认角色并播放烘焙动作。
3. S2D-aware runtime 能精确执行图层合成、语义参数、约束、表情和动作。
4. 基础表情与动作可自动生成，并能以稳定语义供 LLM 或专用动作模型调用。
5. 文件保留足够的网格、UV、骨骼和相对深度信息，为以后转换 3D 提供清晰接口。
6. Spine 和 Live2D 导出器只做语义降级与格式编码，不重新推断 Rig。

## 非目标

- 不生成角色背面、隐藏区域纹理、封闭体积、PBR 材质或可 360 度观察的完整 3D 模型。
- 不承诺普通 glTF 软件理解眨眼、PSD 遮罩、IK 或动作语义。
- 不承诺 GLB 经任意 DCC 软件导入再导出后仍保留未知扩展。
- 不在 v0.1 内嵌脚本、任意行为图或 LLM prompt 执行环境。
- 不以 FBX、USD、Spine 或 Cubism 私有工程格式作为唯一事实源。
- 不在 v0.1 使用自定义 GLB Chunk、自定义压缩或新的文件后缀。

## 第一性原则

格式需要解决三件不同的事，不能混在一起：

1. **源工程**保存 PSD、深度、自动绑骨观测和人工修正，服务再次编译。
2. **语义模型**定义 Part、Surface、Drawable、Rig、Control、Action 的含义。
3. **交换编码**把语义模型映射到成熟生态可以读取的 glTF 2.0 / GLB。

S2D 的创新点是第二项，而不是重新发明第三项。

## 项目边界

- 新项目拥有 `S2DSourceDocument`、`S2DDocument` 及全部版本迁移逻辑。
- `qinglong-captions` 只是一个可替换的上游生产者，不是新项目的运行时依赖。
- 两个项目只通过版本化文件输入契约通信；新项目不得直接 import 本仓库内部 Python 模块、
  读取内部数据库或依赖未声明的目录实现细节。
- 首个 importer 读取 source image、分层 PSD、匹配的 depth PSD、逐部件 PNG/depth 和
  `optimized/info.json`；缺失的可选文件必须形成 capability 结果，而不是路径猜测。
- 新项目生成的 GLB、缓存和诊断写入自己的 workspace，不回写上游数据目录。
- Spine 和 Cubism adapters 属于新项目，可独立发布；旧项目现有 exporter 不在本规范范围内。

## 总体架构

```text
layered PSD + part depth + metadata
                 |
                 v
       observe / solve / validate
                 |
                 v
       S2DSourceDocument
                 |
                 v
           S2DDocument
        /          |          \
       v           v           v
 S2D-compatible  Spine 4.2   Cubism runtime
      GLB          adapter       adapter
```

`S2DSourceDocument` 保存源级观测、置信度、override 和阶段证据。`S2DDocument` 是经过验证、
可运行的语义模型。GLB、Spine 和 Cubism 都从 `S2DDocument` 生成；任何 exporter 都不得
再次估计关节、拆分图层或修改蒙皮权重。

正式发行物保持 `.glb` 后缀和 `model/gltf-binary` MIME。调试时可以导出等价的
`.gltf + .bin + .png`，但它不是另外一种语义格式。

runtime profile 的 Texture 和扩展 BufferView 全部嵌入 GLB 的标准 BIN Chunk，不允许外部 URI
或加载时网络请求。这样“单文件可移动”是 profile 契约，而不是依赖打包工具的偶然结果。

## 两级兼容契约

### Core Fallback

移除或忽略 `QINGLONG_character_2d` 后，文件仍必须是有效 glTF 2.0，并至少具有：

- 一个可见的默认 Scene；
- 一个正交 Camera；
- 可见的默认姿态；
- 标准 Mesh、UV、PNG Texture 和 Material；
- 可用 glTF Skin 表达的骨骼与 LBS 权重；
- 可用 Morph Target 表达的面部或局部形变；
- 所有 `required` 基础表情和动作的标准 glTF Animation 回退。

普通 glTF loader 可以忽略自定义扩展。它不保证逐像素等同 S2D 渲染，但不得出现空场景、
缺纹理、完全错误的朝向或无法播放基础动作。

### S2D Exact Runtime

识别 `QINGLONG_character_2d` 的 runtime 在 Core Fallback 之上执行：

- 明确的 back-to-front 合成顺序；
- 裁剪组、遮罩和可见性规则；
- 语义 Control 到骨骼、Morph 和 Drawable 的绑定；
- IK、约束与次级摆动；
- 表情与动作混合；
- Surface 相对深度驱动的 2.5D 投影；
- 格式版本、能力和预算校验。

## glTF 核心映射

| S2D 概念 | glTF 2.0 表达 | 扩展补充 |
|---|---|---|
| Character | Scene + root Node | profile、canvas、能力 |
| Part | Node 或一组 Node | 稳定 ID、语义 role、side frame |
| Surface | Mesh Primitive | 本地深度、原始深度来源 |
| Drawable | Node + Mesh + Material | composite order、mask group |
| Bone | Node hierarchy | 语义 bone role、置信度 |
| Skin | Skin + JOINTS_0/WEIGHTS_0 | 非 LBS deformer 与约束 |
| Expression | Morph Target / Animation | 语义 Control 与 Binding |
| Motion | Animation | action ID、能力、触发语义 |
| Texture | PNG Image + Texture | 源图层与 alpha 信息 |
| Camera | Orthographic Camera | canvas 到米的标定 |
| Node Constraint | VRMC_node_constraint | 2D IK、warp 等额外约束 |
| Secondary Motion | VRMC_springBone | 非骨链的 2D 摆动 |

### 材质基线

v0.1 的兼容基线固定为：

- PNG、straight alpha、sRGB color；
- `alphaMode: "BLEND"`；
- `doubleSided: true`；
- `metallicFactor: 0`、`roughnessFactor: 1`；
- 使用 `KHR_materials_unlit`，但保留合法的 core PBR fallback；
- 不要求 WebP、KTX2、Draco 或 meshopt；以后只能作为带 core fallback 的可选变体。

为避免“规范允许但 loader 实际不兼容”，v0.1 的 Core Fallback 进一步限制为每顶点最多四个
joint influence，只写 `JOINTS_0/WEIGHTS_0`；每个 Primitive 最多八个 Morph Target；
Animation 只使用 `LINEAR` 和 `STEP` 插值。超出部分必须简化、烘焙为其他 core 通道或标记为
S2D-only，不能仍声称 `core.baked-animation` 覆盖该能力。

PSD 裁剪遮罩必须预烘焙进 fallback Texture 的 alpha，确保普通 loader 不需要理解遮罩也能
显示合理结果。扩展仍保存逻辑遮罩关系，供精确 runtime、编辑和重新编译使用。

### 现有扩展复用优先级

S2D 不复制已经有稳定语义的扩展：

- 无光照材质使用 `KHR_materials_unlit`；
- 任意属性动画在有真实兼容收益时使用 `KHR_animation_pointer`，但仍保留 required core 回退；
- roll、aim、rotation constraint 能完全匹配时使用独立的 `VRMC_node_constraint`；
- 连续骨链的头发、衣摆等次级运动能匹配时使用独立的 `VRMC_springBone`；
- 只有 2D IK、warp lattice、合成顺序、遮罩和语义 Control 等现有扩展无法表达的内容，才进入
  `QINGLONG_character_2d`。

不使用完整 `VRMC_vrm`，因为它要求 3D humanoid 和 meta，并假设 VR avatar 工作流。复用独立
VRMC 扩展不代表文件是 VRM，也不把文件后缀改为 `.vrm`。

## 坐标与深度

### 坐标系

源画布使用 `+x` 向右、`+y` 向下的像素坐标。glTF 使用右手坐标、`+Y` 向上、角色正面
朝 `+Z`。默认 Camera 位于角色前方，朝本地 `-Z` 观察。

固定转换为：

```text
X = (canvasX - canvasCenterX) / pixelsPerMeter
Y = (canvasCenterY - canvasY) / pixelsPerMeter
```

`pixelsPerMeter` 必须大于零并写入扩展。没有外部标定时，v0.1 固定使用
`pixelsPerMeter = 1000`、`scaleMetric = nominal`；这只是稳定的工作尺度，不宣称真实身高。
有外部标定时可以写其他值并标为 `calibrated`。转换矩阵及逆矩阵都进入 golden test，
exporter 不得各自重新定义原点或 Y 方向。

### 双重深度模型

跨图层遮挡和单图层表面起伏不能混成一个 Z-buffer 问题：

- `compositeOrder` 是从 `0` 开始连续且唯一的离散整数，较小值在后、较大值在前，唯一决定
  不同 Drawable 的 back-to-front 合成顺序；
- fallback Mesh 的局部 `POSITION.z` 固定为零，Drawable Node 的
  `translation.z = compositeOrder / pixelsPerMeter`，为普通 viewer 提供单调分层；
- `_S2D_DEPTH` 是 `SCALAR/FLOAT` 自定义顶点属性，元素数量与 `POSITION` 相同，值域固定为
  `[0, 1]`，保存 Surface 内部的相对深度；
- 可选 `depthField` 保存更高分辨率的源深度，用于重新网格化和未来 3D 转换；
- S2D runtime 不允许用预测深度的全局 Z-test 取代 `compositeOrder`。

上游相对深度在进入 `S2DDocument` 时统一成 `0 = near`、`1 = far`。局部位移为：

```text
surfaceZ = reliefScale * (0.5 - normalizedDepth)
```

`reliefScale` 使用米。未提供项目设置时，v0.1 固定为 `0.02`，并标记
`reliefMetric = nominal`。因为上游深度是仿射不变的相对估计，扩展必须将
`depthMetric` 标为 `relative`，不得伪装成真实距离。

S2D runtime 在投影前移除仅供 fallback 排序的 Node Z 间距，再应用 `surfaceZ`；跨 Drawable
仍按 `compositeOrder` 合成。v0.1 的可选 `depthField` 固定为 top-left origin、row-major 的
`SCALAR/UNSIGNED_SHORT normalized` Accessor，`count = width * height`，像素值同样为
`0 = near`、`65535 = far`。宽高和 Accessor index 写在扩展中，不依赖图片解码器解释灰度
精度。

每个 Drawable 的排序 Z 放在不可动画的 `orderNode`，真正接受动作的 `poseNode` 或 skinned
Mesh 是其子节点。Animation 不得写 `orderNode.translation`，避免 glTF 的整向量 translation
channel 在动作播放时覆盖合成层级。

## 语义扩展

v0.1 的 wire name 固定为 `QINGLONG_character_2d`。`QINGLONG` 在公开稳定发布前必须申请
Khronos vendor prefix；在完成注册前产物只能标记为 draft。不得抢用保留给 Khronos 或
多厂商扩展的 `KHR_`、`EXT_` 前缀。

扩展挂在 glTF root，结构概念如下：

```text
QINGLONG_character_2d
  specVersion
  profile
  requiredFeatures[]
  capabilities[]
  canvas
  parts[]
  surfaces[]
  drawables[]
  controls[]
  bindings[]
  constraints[]
  expressions[]
  actions[]
  depthFields[]
  provenance
  bridge3d
```

所有元素同时具有稳定字符串 `id` 和对 glTF 对象的整数引用。整数索引用于运行时访问，
稳定 ID 用于重编译、diff、动作生成和外部 API。稳定 ID 全局唯一、最长 127 ASCII bytes，
匹配 `^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$`，并使用 `part.`、`bone.`、`control.`、
`action.` 等类型前缀。Unicode 显示名称单独保存，不能充当引用键。

### Part、Surface、Drawable

- `Part` 表达语义所有权，例如 face、torso、image-xmin arm。
- `Surface` 表达网格、UV、局部深度和可形变区域。
- `Drawable` 表达实际绘制实例、材质、遮罩、可见性和合成顺序。
- 一个 Part 可以拥有多个 Surface；一个 Surface 可以被多个 Drawable 实例引用。
- `sideFrame` 只能是 `anatomical`、`image_x` 或 `none`。未证明解剖朝向时不得把
  `image_xmin` 静默改名为 left。

### Control 与 Binding

Control 是外部驱动、动作和表情共同使用的稳定语义接口。v0.1 支持：

- `scalar`：带 min、max、default；
- `vec2`：例如 gaze；
- `bool`：例如显隐开关；
- `trigger`：只触发 Action，不参与连续插值。

建议的语义 ID 示例：

```text
face.eye.blink.image_xmin
face.mouth.open
face.expression.smile
head.yaw
body.hand.image_xmax.v_sign
```

Binding 将 Control 映射到以下目标之一：

- Node translation、rotation、scale；
- Morph Target weight；
- Drawable opacity 或 visibility；
- S2D constraint input；
- composite order 的受限离散切换。

映射使用显式 piecewise-linear transfer。多个 Binding 写同一目标时必须声明
`override`、`add`、`multiply` 或 `max`，并按 `priority`、稳定 ID 顺序确定性求值。
required Core Fallback 动作不得依赖动态 composite order；需要层级切换的动作必须提供不切换
也可理解的 core 回退，并把精确切换保留给 S2D runtime。

### Expression 与 Action

Expression 是一组 Control 常值或曲线，例如 neutral、smile、sad、surprised。Action 是一个
有时长的动作入口，例如 idle、wave、v-sign。每个 required Expression 和 Action 都引用一个
标准 glTF Animation 作为回退；静态 Expression 使用两个相同采样点构成可保持的 `STEP`
Animation，不依赖 loader 的默认保持行为。

每个 Action 固定包含：

- 稳定 `id` 与语义 `kind`；
- glTF Animation index；
- duration、loop mode 和受影响的 Control/Bone 集合；
- `required` 或 `optional` capability；
- 输入前置条件和降级原因。

v0.1 不内嵌通用行为图或脚本。随机眨眼、呼吸和动作调度由 runtime 策略完成，格式只提供
可调用的 Expression、Action 和简单 procedural driver 描述。

LLM 或专用生成模型不直接写 GLB。它们输出经过 Schema 约束的 `MotionIntent`，内容只允许
引用已知语义骨骼、Control、时长和关键帧；编译器完成 IK、碰撞/约束校验、曲线烘焙，最后
生成标准 glTF Animation 与 Action 元数据。未知 ID、越界值和不可满足约束必须拒绝，不能
由 runtime 猜测。

## Runtime 求值顺序

S2D-aware runtime 固定执行顺序：

```text
Rest Pose
  -> external semantic inputs
  -> active Action / Expression mix
  -> ControlBinding transfer
  -> IK and constraints
  -> secondary motion
  -> skin / warp / morph
  -> 2.5D projection
  -> masking and compositeOrder rendering
```

同一帧不得改变此顺序。普通 glTF viewer 只执行已经烘焙进 Animation、Skin 和 Morph 的结果，
不声称与动态 S2D 求值完全等价。

## glTF 扩展使用规则

典型文件声明：

```json
{
  "extensionsUsed": [
    "KHR_materials_unlit",
    "VRMC_node_constraint",
    "VRMC_springBone",
    "QINGLONG_character_2d"
  ],
  "extensionsRequired": []
}
```

上例中的 VRMC 扩展也只有实际使用时才加入。`KHR_animation_pointer` 同理，并且不能成为
required 基础动作的唯一表达；required 动作必须同时烘焙为 core Node TRS 或 Morph weight
channel。

`QINGLONG_character_2d` 默认不进入 `extensionsRequired`，因为普通 loader 应能加载 fallback。
S2D Profile conformance 与 glTF 的 `extensionsRequired` 是两个不同问题：S2D-aware runtime
必须识别本扩展，普通 glTF runtime 不需要。

扩展二进制数据必须通过标准 Buffer、BufferView、Accessor、Image 引用，不添加自定义 GLB
Chunk。发行 GLB 只含规范允许的 JSON Chunk 和 BIN Chunk。

## 3D 转换边界

`bridge3d` 只保证提供转换证据，不保证自动生成完整 3D：

- canvas-to-meter 矩阵和源 Camera；
- 语义骨骼、bind pose 和稳定 ID；
- Surface mesh、UV、`_S2D_DEPTH` 与可选 depthField；
- 材质、straight-alpha 和双面提示；
- 相对深度类型、比例和有效区域；
- 哪些背面、遮挡纹理和体积信息未知。

转换器可以据此做 relief、挤出或投影到已有 3D 网格，但必须把补全背面、厚度、拓扑和材质
标为生成步骤。仅凭单视图相对深度，不能把结果宣称为真实 3D 重建。

## Spine 与 Live2D 适配

两个适配器读取同一个 `S2DDocument`：

- Spine 优先映射 Bone、Skin、slot order、mesh attachment 和标准 timeline；
- Cubism 优先映射 Control、Warp/Rotation Deformer、parameter、draw order 和 motion；
- 无法原生表达的 Binding 必须烘焙、显式降级或报错；
- exporter 必须产出 capability report，列出 retained、baked、dropped、unsupported；
- exporter 禁止重新识别语义、重新估计关节或静默丢失 required Action。

S2D 是两种输出的共同上层语义，不以其中任一格式的限制反向污染核心模型。

## 版本与能力协商

- glTF `asset.version` 固定为 `2.0`；
- `QINGLONG_character_2d.specVersion` 首版为 `0.1.0`；
- major 不兼容时，S2D runtime 拒绝扩展语义但仍可加载 core fallback；
- minor 新增字段默认可忽略，新增不可忽略语义必须进入 `requiredFeatures`；
- runtime 遇到未知 `requiredFeatures` 时不得部分执行 S2D 语义；
- `capabilities` 只描述实际通过验证的能力，不允许按配置意图预先声明。

v0.1 的最低 capabilities 为：

```text
core.static-fallback
core.baked-animation
s2d.explicit-compositing
s2d.semantic-controls
s2d.relative-depth
```

## 编译流程与失败语义

```text
1. 读取并验证 S2DSourceDocument
2. 建立 S2DDocument 与稳定 ID 表
3. 生成 fallback Mesh、Skin、Morph、Material、Animation
4. 生成精确合成、Control、Binding、Constraint 和 Action 扩展
5. 打包标准 GLB
6. 运行 glTF Validator
7. 运行 S2D schema/semantic validator
8. 运行 generic fallback 与 exact runtime parity fixtures
9. 原子发布产物和 capability report
```

以下情况必须失败：

- glTF 索引、BufferView、Accessor 或 Image 引用非法；
- 稳定 ID 重复或语义引用悬空；
- `pixelsPerMeter <= 0` 或坐标矩阵不可逆；
- required Action 没有 core Animation fallback；
- Control/Binding 越界、目标类型不匹配或约束图有环；
- composite order 不确定，或重叠 Drawable 发生未声明顺序冲突；
- core fallback 无默认 Scene、不可见或纹理缺失；
- 实际使用的 required feature 未声明；
- 文件超过配置预算或解码后资源超过安全上限。

可选表情、动作或非关键肢体求解失败时，可以输出降级资产，但必须省略对应 capability 并在
report 写出结构化原因。不能用假关节、空动画或静默丢字段冒充成功。

## 确定性与安全

同一规范化 `S2DDocument`、编译器版本和配置必须生成逐字节相同的 GLB：

- 所有数组按稳定 ID 或规范规定的拓扑顺序生成；
- JSON 使用固定 key 顺序、数值编码和空格填充；
- BIN 按确定的 BufferView 顺序和 4-byte alignment 打包；
- Texture atlas、PNG encoder、三角形和顶点顺序固定；
- provenance 记录输入摘要、算法版本、模型摘要和编译器版本；
- provenance 默认不记录绝对本地路径、完整 prompt 或其他无运行必要的隐私数据；
- 摘要不包含其自身，避免自引用。

loader 必须在分配 GPU/CPU 资源前检查文件长度、BufferView 边界、元素数量、纹理尺寸、
骨骼/Control/Constraint 上限和扩展版本。扩展不允许脚本、URI 网络请求或任意代码执行。

## 验证矩阵

### Schema 与结构

- Khronos glTF Validator 零 error；
- S2D JSON Schema 与语义 validator 通过；
- 未识别扩展时仍能解析全部 core glTF；
- malformed index、越界 BufferView、重复 ID、约束环均有负向测试。

### Generic Compatibility

- Khronos Sample Viewer 显示默认姿态；
- Three.js GLTFLoader 显示默认姿态并播放 required 动作；
- Blender 导入可见 Mesh、Texture、Skeleton 和 baked Animation；
- generic viewer 截图与 fallback reference 在规定容差内；
- 不要求 Blender 再导出后保留 S2D 扩展。

### S2D Exact Runtime

- neutral、blink、mouth-open、smile、wave、v-sign golden fixtures；
- canvas 与 glTF 坐标双向 landmark parity；
- 半透明异色重叠 fixture 验证 compositeOrder；
- mask/clip fixture 验证边缘和 straight alpha；
- `_S2D_DEPTH` fixture 验证 near/far 方向与 reliefScale；
- 单 Action、Expression 叠加、IK 和次级运动分别验证求值顺序；
- 同输入重复编译验证 byte-for-byte deterministic output。

### Adapter Compatibility

- S2D、Spine、Cubism 对同一 rest pose 和 required Action 做 landmark parity；
- 每个 adapter 的 capability report 与实际产物一致；
- required 语义无法表示时编译失败，而不是静默降级。

## v0.1 交付边界

v0.1 只要求完成一条窄而完整的垂直链路：

1. 当前 `qinglong-captions/datasets/321/outputs/lucy.jpg/` 作为首个只读集成 fixture 来源；
   该路径虽然以 `.jpg` 结尾，实际是包含 `src_img.png`、`final.psd`、`final_depth.psd` 和
   逐部件产物的目录。新项目通过 fixture manifest 和摘要定位它，不硬编码旧项目绝对路径。
2. Lucy 的默认姿态、idle 和 blink 可由普通 glTF viewer 打开并播放。
3. 另用一个通过双臂分离能力门的 fixture 验证 Skin 和 wave；Lucy 当前 merged handwear
   不能被伪装成可靠单臂 wave。
4. v-sign 只有在输入具备独立手指/手势 Morph 或通过审核的生成变体时才声明 capability；
   v0.1 用合成 fixture 验证格式表达，不要求从 Lucy 缺失的信息中凭空恢复手势。
5. S2D runtime 正确执行显式合成顺序、Control 和相对深度。
6. MotionIntent 可生成一个输入能力允许的新 Action，并通过约束后烘焙回 GLB。
7. 3D bridge 能导出标准 mesh/skin/UV/depth 数据，不做体积补全。
8. Spine/Cubism adapter 可后续独立迭代，不阻塞 S2D GLB 首个可运行版本。

## 明确排除的替代方案

- **VRM 直接复用**：其 humanoid、first-person、3D expression 和许可语义假设过强；采用其
  “GLB + 角色扩展”方法，并选择性复用独立的 `VRMC_node_constraint`、
  `VRMC_springBone`，但不继承整个 `VRMC_vrm` schema。
- **OpenUSD 作为 runtime 主格式**：适合大型 DCC 场景组合，不是轻量角色运行时或 Rigging
  System；保留未来 exporter。
- **FBX 作为事实源**：实现和版本差异大，且不适合定义新的开放语义；仅作为可选导出。
- **FlatBuffers 自定义容器**：运行时性能好，但会直接失去 glTF 工具链兼容；不作为公开主格式。
- **将全部数据塞进 `extras`**：适合短期实验，不提供正式扩展的能力发现和约束；v0.1 使用
  vendor extension 与独立 Schema。

## 参考

- [Khronos glTF 2.0.1 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- [Khronos glTF Extension Registry](https://github.com/KhronosGroup/glTF/blob/main/extensions/README.md)
- [KHR_animation_pointer](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_animation_pointer/README.md)
- [VRM as a glTF/GLB-based format](https://vrm.dev/en/vrm/gltf/format/)
- [VRMC_node_constraint 1.0](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_node_constraint-1.0/README.md)
- [VRMC_springBone 1.0](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_springBone-1.0/README.md)
- [OpenUSD Introduction](https://openusd.org/release/intro.html)
